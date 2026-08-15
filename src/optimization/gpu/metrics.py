"""Metric reductions for the Apple MPS screening proxy.

The daily-series reduction approach was adapted from ``src/gpu/torch_metrics.py``
in RustyCZ's Passivbot GPU branch (commit 7c529bc73), then narrowed to the
metrics validated for this foundation and integrated with the current schema.
"""

from __future__ import annotations

import math

import torch

from optimization.gpu.model import GAP_BINS, GAP_MAX_MINUTES


# Keep the public proxy surface deliberately narrow. Exact Rust evaluations
# still emit the normal complete metric set; this list governs only which
# metrics may guide Metal screening or proxy-side limits.
SUPPORTED_METRICS = (
    "adg_strategy_eq",
    "adg_strategy_eq_w",
    "backtest_completion_ratio",
    "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_strategy_eq",
    "fills_gap_longest_days",
    "fills_gap_p95_hours",
    "mdg_strategy_eq",
    "position_held_days_max",
    "sharpe_ratio_strategy_eq",
    "sortino_ratio_strategy_eq",
    "strategy_eq_recovery_days_max",
    "strategy_eq_underwater_pct_mean",
    "volume_pct_per_day_avg",
)

# Reserved for a later PR that validates metrics requiring per-fill/trade
# aggregates. Keeping this empty avoids allocating the larger Metal buffers.
EXTRA_KERNEL_METRICS = ()


def _daily_series_masks(day_min_eq):
    return torch.isfinite(day_min_eq) & (day_min_eq < float("inf"))


def _pct_change(values, active):
    previous = values[:, :-1]
    current = values[:, 1:]
    both = active[:, :-1] & active[:, 1:]
    denominator = previous.abs().clamp(min=1e-12)
    changes = torch.where(
        both, (current - previous) / denominator, torch.zeros_like(current)
    )
    return changes, both


def _masked_median(values, mask):
    filled = torch.where(mask, values, torch.full_like(values, float("nan")))
    return torch.nanmedian(filled, dim=1).values


def _smoothed_adg(day_eq, active):
    batch_size, day_count = day_eq.shape
    counts = active.sum(dim=1)
    indices = (
        torch.arange(day_count, device=day_eq.device)
        .unsqueeze(0)
        .expand(batch_size, day_count)
    )
    first_index = (
        torch.where(active, indices, torch.full_like(indices, day_count))
        .min(dim=1)
        .values.clamp(max=day_count - 1)
    )
    start = day_eq.gather(1, first_index.unsqueeze(1)).squeeze(1)
    order = torch.where(active, indices, torch.full_like(indices, -1))
    sorted_indices, _ = torch.sort(order, dim=1, descending=True)
    tail_count = torch.minimum(counts, torch.full_like(counts, 3)).clamp(min=1)
    tail_values = torch.zeros_like(start)
    for offset in range(3):
        take = (offset < tail_count).to(day_eq.dtype)
        gather_index = sorted_indices[:, offset].clamp(min=0)
        tail_values += take * day_eq.gather(1, gather_index.unsqueeze(1)).squeeze(1)
    end = tail_values / tail_count.to(day_eq.dtype)
    gain = end / start.abs().clamp(min=1e-12) * torch.sign(start)
    duration_days = counts.clamp(min=1).to(day_eq.dtype)
    adg = torch.where(
        (gain > 0) & (counts > 0),
        gain.clamp(min=1e-12) ** (1.0 / duration_days) - 1.0,
        torch.full_like(gain, -1.0),
    )
    return adg


def _sharpe_sortino(changes, mask, adg):
    count = mask.sum(dim=1).clamp(min=1).to(changes.dtype)
    difference = torch.where(
        mask, changes - adg.unsqueeze(1), torch.zeros_like(changes)
    )
    standard_deviation = torch.sqrt((difference * difference).sum(dim=1) / count)
    sharpe_denominator = torch.where(
        standard_deviation != 0.0,
        standard_deviation,
        torch.ones_like(standard_deviation),
    )
    sharpe = torch.where(
        standard_deviation != 0.0,
        adg / sharpe_denominator,
        torch.zeros_like(adg),
    )
    downside = mask & (changes < 0)
    downside_count = downside.sum(dim=1).to(changes.dtype)
    downside_deviation = torch.sqrt(
        torch.where(downside, changes * changes, torch.zeros_like(changes)).sum(dim=1)
        / downside_count.clamp(min=1)
    )
    sortino_denominator = torch.where(
        downside_deviation != 0.0,
        downside_deviation,
        torch.ones_like(downside_deviation),
    )
    sortino = torch.where(
        downside_deviation != 0.0,
        adg / sortino_denominator,
        torch.zeros_like(adg),
    )
    return sharpe, sortino


def _weighted_adg(day_eq, active):
    counts = active.sum(dim=1)
    cumulative = torch.cumsum(active.to(torch.long), dim=1)
    total = torch.zeros(day_eq.shape[0], dtype=day_eq.dtype, device=day_eq.device)
    for index in range(10):
        fraction = 1.0 / (1.0 + index)
        subset_count = torch.round(counts.to(day_eq.dtype) * fraction).to(torch.long)
        start_position = counts - subset_count
        subset = active & (cumulative > start_position.unsqueeze(1))
        total += _smoothed_adg(day_eq, subset)
    return total / 10.0


def _gap_percentile_hours(gap_hist, boundary_lead_min, boundary_trail_min, pct=95.0):
    batch_size = gap_hist.shape[0]
    device = gap_hist.device
    histogram = gap_hist.to(torch.float32).clone()
    log_bin_scale = (GAP_BINS - 1) / math.log1p(GAP_MAX_MINUTES)
    for boundary_gap in (boundary_lead_min, boundary_trail_min):
        bins = (
            (torch.log1p(boundary_gap.clamp(min=0.0)) * log_bin_scale)
            .to(torch.int64)
            .clamp(0, GAP_BINS - 1)
        )
        histogram.scatter_add_(
            1, bins.unsqueeze(1), torch.ones(batch_size, 1, device=device)
        )
    target = histogram.sum(dim=1) * (pct / 100.0)
    cumulative = torch.cumsum(histogram, dim=1)
    reached = cumulative >= target.unsqueeze(1)
    bin_index = (
        torch.where(
            reached,
            torch.arange(GAP_BINS, device=device).unsqueeze(0),
            torch.full_like(histogram, GAP_BINS - 1, dtype=torch.long),
        )
        .min(dim=1)
        .values
    )
    edges = torch.expm1(
        torch.arange(GAP_BINS, device=device, dtype=torch.float32) / log_bin_scale
    )
    return edges[bin_index.clamp(max=GAP_BINS - 1)] / 60.0


def compute_objectives(out: dict, run, data: dict, needed=None) -> dict:
    """Reduce compact Metal output into validated proxy objective metrics."""

    day_end_eq = out["day_end_eq"].to(torch.float64)
    day_min_eq = out["day_min_eq"].to(torch.float64)
    day_max_dd = out["day_max_dd"].to(torch.float64)
    day_volume = out["day_volume"].to(torch.float64)
    day_has_fill = out["day_has_fill"]
    active = _daily_series_masks(out["day_min_eq"])

    adg = _smoothed_adg(day_end_eq, active)
    daily_changes, change_mask = _pct_change(day_end_eq, active)
    mdg = _masked_median(daily_changes, change_mask)
    daily_min_changes, min_change_mask = _pct_change(day_min_eq, active)
    sharpe, sortino = _sharpe_sortino(daily_min_changes, min_change_mask, adg)
    adg_w = _weighted_adg(day_end_eq, active)

    underwater = torch.where(active, day_max_dd, torch.zeros_like(day_max_dd)).sum(
        dim=1
    ) / active.sum(dim=1).clamp(min=1)
    sorted_drawdowns, _ = torch.sort(
        torch.where(active, day_max_dd, torch.zeros_like(day_max_dd)),
        dim=1,
        descending=True,
    )
    worst_count = (
        (0.01 * active.sum(dim=1).to(torch.float64)).floor().clamp(min=1).to(torch.long)
    )
    cumulative_drawdowns = torch.cumsum(sorted_drawdowns, dim=1)
    worst_one_pct = cumulative_drawdowns.gather(
        1, (worst_count - 1).unsqueeze(1)
    ).squeeze(1) / worst_count.to(torch.float64)

    volume_days = day_has_fill.sum(dim=1).clamp(min=1).to(torch.float64)
    volume_pct = day_volume.sum(dim=1) / volume_days

    last_eq_ts = out["last_eq_ts"]
    first_eq_ts = out["first_eq_ts"]
    final_recovery = torch.where(
        torch.isfinite(out["last_high_ts"]),
        last_eq_ts - out["last_high_ts"],
        torch.zeros_like(last_eq_ts),
    )
    recovery_max_days = (
        torch.maximum(out["recovery_max_ms"], final_recovery) / 86_400_000.0
    )
    held_days = out["held_max_ms"] / 86_400_000.0

    boundary_lead = torch.where(
        torch.isfinite(out["first_fill_ts"]),
        (out["first_fill_ts"] - first_eq_ts) / 60_000.0,
        (last_eq_ts - first_eq_ts) / 60_000.0,
    ).clamp(min=0.0)
    boundary_trail = torch.where(
        torch.isfinite(out["last_fill_ts"]),
        (last_eq_ts - out["last_fill_ts"]) / 60_000.0,
        torch.zeros_like(last_eq_ts),
    ).clamp(min=0.0)
    gap_p95_hours = _gap_percentile_hours(
        out["gap_hist"], boundary_lead, boundary_trail
    )
    gap_longest_days = torch.maximum(
        out["gap_max_ms"] / 86_400_000.0,
        torch.maximum(boundary_lead, boundary_trail) / 1440.0,
    )

    requested_start = float(run.guard_ts_ms)
    first_timestamp = data["ts0"]
    candle_count = int(data["n"])
    requested_end = float(first_timestamp + (candle_count - 1) * run.interval_ms)
    covered_end = torch.minimum(
        torch.full_like(last_eq_ts, requested_end),
        last_eq_ts + float(max(1, run.interval_ms // 60_000) * 60_000),
    )
    requested_span = max(requested_end - requested_start, 1.0)
    completion = ((covered_end - requested_start) / requested_span).clamp(0.0, 1.0)

    objectives = {
        "adg_strategy_eq": adg,
        "adg_strategy_eq_w": adg_w,
        "backtest_completion_ratio": completion,
        "drawdown_worst_mean_1pct_strategy_eq": worst_one_pct,
        "drawdown_worst_strategy_eq": out["max_dd"],
        "fills_gap_longest_days": gap_longest_days,
        "fills_gap_p95_hours": gap_p95_hours,
        "mdg_strategy_eq": mdg,
        "position_held_days_max": held_days,
        "sharpe_ratio_strategy_eq": sharpe,
        "sortino_ratio_strategy_eq": sortino,
        "strategy_eq_recovery_days_max": recovery_max_days,
        "strategy_eq_underwater_pct_mean": underwater,
        "volume_pct_per_day_avg": volume_pct,
    }
    if needed is None:
        return objectives
    requested = set(needed)
    return {key: value for key, value in objectives.items() if key in requested}
