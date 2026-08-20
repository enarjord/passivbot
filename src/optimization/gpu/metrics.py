"""Metric reductions for the Apple MPS screening proxy.

The daily-series reduction approach was adapted from ``src/gpu/torch_metrics.py``
in RustyCZ's Passivbot GPU branch (commit 7c529bc73), then narrowed to the
metrics validated for this foundation and integrated with the current schema.
"""

from __future__ import annotations

import torch

# Keep the public proxy surface deliberately narrow. Exact Rust evaluations
# still emit the normal complete metric set; this list governs only which
# metrics may guide Metal screening or proxy-side limits.
SUPPORTED_METRICS = (
    "adg_strategy_eq",
    "adg_strategy_eq_w",
    "backtest_completion_ratio",
    "calmar_ratio_strategy_eq",
    "calmar_ratio_strategy_eq_w",
    "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_strategy_eq",
    "expected_shortfall_1pct_strategy_eq",
    "fills_gap_longest_days",
    "gain_strategy_eq",
    "mdg_strategy_eq",
    "mdg_strategy_eq_w",
    "omega_ratio_strategy_eq",
    "omega_ratio_strategy_eq_w",
    "position_held_days_max",
    "sharpe_ratio_strategy_eq",
    "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_strategy_eq",
    "sortino_ratio_strategy_eq_w",
    "sterling_ratio_strategy_eq",
    "sterling_ratio_strategy_eq_w",
    "strategy_eq_recovery_days_max",
    "strategy_eq_underwater_pct_mean",
    "strategy_eq_underwater_pct_median",
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
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    counts = mask.sum(dim=1)
    filled = torch.where(mask, values, torch.full_like(values, float("inf")))
    ordered = torch.sort(filled, dim=1).values
    lower_index = ((counts - 1).clamp(min=0) // 2).unsqueeze(1)
    upper_index = (counts // 2).unsqueeze(1)
    lower = ordered.gather(1, lower_index).squeeze(1)
    upper = ordered.gather(1, upper_index).squeeze(1)
    medians = (lower + upper) / 2.0
    return torch.where(counts > 0, medians, torch.zeros_like(medians))


def _smoothed_gain_adg(day_eq, active):
    batch_size, day_count = day_eq.shape
    if day_count == 0:
        zeros = torch.zeros(batch_size, dtype=day_eq.dtype, device=day_eq.device)
        return zeros, zeros
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
    for offset in range(min(3, day_count)):
        take = (offset < tail_count).to(day_eq.dtype)
        gather_index = sorted_indices[:, offset].clamp(min=0)
        tail_values += take * day_eq.gather(1, gather_index.unsqueeze(1)).squeeze(1)
    end = tail_values / tail_count.to(day_eq.dtype)
    gain = torch.where(
        start > 0.0,
        torch.where(end > 0.0, end / start, torch.full_like(end, -1.0)),
        torch.full_like(end, float("inf")),
    )
    duration_days = counts.clamp(min=1).to(day_eq.dtype)
    adg = torch.where(
        (gain > 0) & (counts >= 2),
        gain.clamp(min=1e-12) ** (1.0 / duration_days) - 1.0,
        torch.where(counts >= 2, torch.full_like(gain, -1.0), torch.zeros_like(gain)),
    )
    gain = torch.where(counts >= 2, gain, torch.zeros_like(gain))
    return gain, adg


def _smoothed_adg(day_eq, active):
    return _smoothed_gain_adg(day_eq, active)[1]


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


def _omega_ratio(changes, mask):
    gains = torch.where(
        mask & (changes >= 0.0), changes, torch.zeros_like(changes)
    ).sum(dim=1)
    losses = torch.where(
        mask & (changes < 0.0), -changes, torch.zeros_like(changes)
    ).sum(dim=1)
    cap = torch.full_like(gains, 1_000.0)
    ratio = gains / losses.clamp(min=1e-12)
    return torch.where(
        losses <= 1e-12,
        torch.where(gains > 1e-12, cap, torch.zeros_like(gains)),
        ratio.clamp(max=1_000.0),
    )


def _mean_worst_one_pct_abs(values, mask):
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    counts = mask.sum(dim=1)
    ordered = torch.sort(
        torch.where(mask, values, torch.full_like(values, float("inf"))), dim=1
    ).values
    worst_count = (counts.to(values.dtype) * 0.01).floor().clamp(min=1).to(torch.long)
    cumulative = torch.cumsum(ordered.abs(), dim=1)
    result = cumulative.gather(1, (worst_count - 1).unsqueeze(1)).squeeze(1)
    result = result / worst_count.to(values.dtype)
    return torch.where(counts > 0, result, torch.zeros_like(result))


def _mean_worst_one_pct_largest(values, mask):
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    counts = mask.sum(dim=1)
    ordered = torch.sort(
        torch.where(mask, values.abs(), torch.zeros_like(values)),
        dim=1,
        descending=True,
    ).values
    worst_count = (counts.to(values.dtype) * 0.01).floor().clamp(min=1).to(torch.long)
    cumulative = torch.cumsum(ordered, dim=1)
    result = cumulative.gather(1, (worst_count - 1).unsqueeze(1)).squeeze(1)
    result = result / worst_count.to(values.dtype)
    return torch.where(counts > 0, result, torch.zeros_like(result))


def _weighted_subsets(
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
):
    finite_timestamps = torch.isfinite(first_eq_ts) & torch.isfinite(last_eq_ts)
    sample_span = torch.where(
        finite_timestamps,
        (last_eq_ts - first_eq_ts) / float(interval_ms),
        torch.zeros_like(first_eq_ts),
    )
    sample_count = (
        torch.floor(sample_span + 0.5)
        .to(torch.long)
        .add(1)
        .clamp(min=1)
    )
    eligible = finite_timestamps & (sample_count >= 2)
    subsets = [active]
    first_day = int(first_timestamp) // 86_400_000
    day_ids = torch.arange(active.shape[1], device=active.device) + first_day
    for index in range(1, 10):
        fraction = 1.0 / (1.0 + index)
        start_position = torch.floor(
            sample_count.to(first_eq_ts.dtype) * (1.0 - fraction) + 0.5
        ).to(torch.long)
        subset_start_ts = first_eq_ts + start_position.to(first_eq_ts.dtype) * float(
            interval_ms
        )
        subset_start_day = torch.floor(subset_start_ts / 86_400_000.0).to(
            torch.long
        )
        subsets.append(
            active & (day_ids.unsqueeze(0) >= subset_start_day.unsqueeze(1))
        )
    return eligible, subsets


def _weighted_adg(
    day_eq,
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
):
    """Match Rust's minute-sliced weighted ADG using compact daily outputs."""

    eligible, subsets = _weighted_subsets(
        active,
        first_eq_ts,
        last_eq_ts,
        first_timestamp,
        interval_ms,
    )
    total = torch.zeros(
        day_eq.shape[0], dtype=day_eq.dtype, device=day_eq.device
    )
    for subset in subsets:
        total += torch.where(
            eligible, _smoothed_adg(day_eq, subset), torch.zeros_like(total)
        )
    return total / 10.0


WEIGHTED_STRATEGY_EQ_METRICS = {
    "adg_strategy_eq_w",
    "mdg_strategy_eq_w",
    "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_strategy_eq_w",
    "omega_ratio_strategy_eq_w",
    "calmar_ratio_strategy_eq_w",
    "sterling_ratio_strategy_eq_w",
}


def _weighted_strategy_eq_metrics(
    day_end_eq,
    day_min_eq,
    day_max_dd,
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
    requested,
):
    requested = set(requested) & WEIGHTED_STRATEGY_EQ_METRICS
    if not requested:
        return {}
    eligible, subsets = _weighted_subsets(
        active,
        first_eq_ts,
        last_eq_ts,
        first_timestamp,
        interval_ms,
    )
    totals = {
        name: torch.zeros(
            day_end_eq.shape[0],
            dtype=day_end_eq.dtype,
            device=day_end_eq.device,
        )
        for name in requested
    }
    need_return_changes = bool(
        requested & {"mdg_strategy_eq_w", "omega_ratio_strategy_eq_w"}
    )
    need_min_changes = bool(
        requested
        & {"sharpe_ratio_strategy_eq_w", "sortino_ratio_strategy_eq_w"}
    )
    need_drawdowns = bool(
        requested & {"calmar_ratio_strategy_eq_w", "sterling_ratio_strategy_eq_w"}
    )
    for subset in subsets:
        adg = _smoothed_adg(day_end_eq, subset)
        values = {}
        if "adg_strategy_eq_w" in requested:
            values["adg_strategy_eq_w"] = adg
        if need_return_changes:
            returns, return_mask = _pct_change(day_end_eq, subset)
            if "mdg_strategy_eq_w" in requested:
                values["mdg_strategy_eq_w"] = _masked_median(returns, return_mask)
            if "omega_ratio_strategy_eq_w" in requested:
                values["omega_ratio_strategy_eq_w"] = _omega_ratio(
                    returns, return_mask
                )
        if need_min_changes:
            min_returns, min_return_mask = _pct_change(day_min_eq, subset)
            sharpe, sortino = _sharpe_sortino(min_returns, min_return_mask, adg)
            if "sharpe_ratio_strategy_eq_w" in requested:
                values["sharpe_ratio_strategy_eq_w"] = sharpe
            if "sortino_ratio_strategy_eq_w" in requested:
                values["sortino_ratio_strategy_eq_w"] = sortino
        if need_drawdowns:
            drawdown_worst = torch.where(
                subset, day_max_dd, torch.zeros_like(day_max_dd)
            ).max(dim=1).values
            if "calmar_ratio_strategy_eq_w" in requested:
                values["calmar_ratio_strategy_eq_w"] = adg / drawdown_worst.clamp(
                    min=1e-12
                )
            if "sterling_ratio_strategy_eq_w" in requested:
                worst_one_pct = _mean_worst_one_pct_largest(
                    day_max_dd, subset
                )
                values["sterling_ratio_strategy_eq_w"] = adg / worst_one_pct.clamp(
                    min=1e-12
                )
        for name, value in values.items():
            totals[name] += torch.where(
                eligible, value, torch.zeros_like(value)
            )
    return {name: value / 10.0 for name, value in totals.items()}


def compute_objectives(out: dict, run, data: dict, needed=None) -> dict:
    """Reduce compact Metal output into validated proxy objective metrics."""

    day_end_eq = out["day_end_eq"].to(torch.float64)
    day_min_eq = out["day_min_eq"].to(torch.float64)
    day_max_dd = out["day_max_dd"].to(torch.float64)
    day_volume = out["day_volume"].to(torch.float64)
    day_has_fill = out["day_has_fill"]
    active = _daily_series_masks(out["day_min_eq"])
    requested = set(SUPPORTED_METRICS if needed is None else needed)

    gain, adg = _smoothed_gain_adg(day_end_eq, active)
    daily_changes, change_mask = _pct_change(day_end_eq, active)
    mdg = _masked_median(daily_changes, change_mask)
    omega = _omega_ratio(daily_changes, change_mask)
    daily_min_changes, min_change_mask = _pct_change(day_min_eq, active)
    sharpe, sortino = _sharpe_sortino(daily_min_changes, min_change_mask, adg)
    expected_shortfall = _mean_worst_one_pct_abs(daily_min_changes, min_change_mask)
    weighted_metrics = _weighted_strategy_eq_metrics(
        day_end_eq,
        day_min_eq,
        day_max_dd,
        active,
        out["first_eq_ts"],
        out["last_eq_ts"],
        data["ts0"],
        run.interval_ms,
        requested,
    )

    underwater = torch.where(active, day_max_dd, torch.zeros_like(day_max_dd)).sum(
        dim=1
    ) / active.sum(dim=1).clamp(min=1)
    underwater_median = _masked_median(day_max_dd, active)
    worst_one_pct = _mean_worst_one_pct_largest(day_max_dd, active)
    calmar = adg / out["max_dd"].to(torch.float64).clamp(min=1e-12)
    sterling = adg / worst_one_pct.clamp(min=1e-12)

    volume_days = day_has_fill.sum(dim=1).clamp(min=1).to(torch.float64)
    volume_pct = day_volume.sum(dim=1) / volume_days

    last_eq_ts = out["last_eq_ts"]
    first_eq_ts = out["first_eq_ts"]
    has_equity = (
        torch.isfinite(first_eq_ts)
        & torch.isfinite(last_eq_ts)
        & (last_eq_ts >= first_eq_ts)
    )
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
    gap_longest_days = torch.maximum(
        out["gap_max_ms"] / 86_400_000.0,
        torch.maximum(boundary_lead, boundary_trail) / 1440.0,
    )
    gap_longest_days = torch.where(
        has_equity, gap_longest_days, torch.zeros_like(gap_longest_days)
    )

    requested_start = float(run.requested_start_ts_ms)
    first_timestamp = data["ts0"]
    candle_count = int(data["n"])
    requested_end = float(first_timestamp + candle_count * run.interval_ms)
    covered_end = torch.minimum(
        torch.full_like(last_eq_ts, requested_end),
        last_eq_ts + float(max(1, run.interval_ms // 60_000) * 60_000),
    )
    requested_span = max(requested_end - requested_start, 1.0)
    completion = torch.where(
        has_equity,
        ((covered_end - requested_start) / requested_span).clamp(0.0, 1.0),
        torch.zeros_like(last_eq_ts),
    )

    objectives = {
        "adg_strategy_eq": adg,
        "backtest_completion_ratio": completion,
        "calmar_ratio_strategy_eq": calmar,
        "drawdown_worst_mean_1pct_strategy_eq": worst_one_pct,
        "drawdown_worst_strategy_eq": out["max_dd"],
        "expected_shortfall_1pct_strategy_eq": expected_shortfall,
        "fills_gap_longest_days": gap_longest_days,
        "gain_strategy_eq": gain,
        "mdg_strategy_eq": mdg,
        "omega_ratio_strategy_eq": omega,
        "position_held_days_max": held_days,
        "sharpe_ratio_strategy_eq": sharpe,
        "sortino_ratio_strategy_eq": sortino,
        "sterling_ratio_strategy_eq": sterling,
        "strategy_eq_recovery_days_max": recovery_max_days,
        "strategy_eq_underwater_pct_mean": underwater,
        "strategy_eq_underwater_pct_median": underwater_median,
        "volume_pct_per_day_avg": volume_pct,
    }
    objectives.update(weighted_metrics)
    if needed is None:
        return objectives
    return {key: value for key, value in objectives.items() if key in requested}
