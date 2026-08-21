"""Metric reductions for the Apple MPS screening proxy.

The daily-series reduction approach was adapted from ``src/gpu/torch_metrics.py``
in RustyCZ's Passivbot GPU branch (commit 7c529bc73), then narrowed to the
metrics validated for this foundation and integrated with the current schema.
"""

from __future__ import annotations

import math

import torch

_USD_STRATEGY_EQ_ALIASES = {
    "adg_usd": "adg_strategy_eq",
    "adg_w_usd": "adg_strategy_eq_w",
    "calmar_ratio_usd": "calmar_ratio_strategy_eq",
    "calmar_ratio_w_usd": "calmar_ratio_strategy_eq_w",
    "drawdown_worst_mean_1pct_usd": "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_usd": "drawdown_worst_strategy_eq",
    "expected_shortfall_1pct_usd": "expected_shortfall_1pct_strategy_eq",
    "gain_usd": "gain_strategy_eq",
    "mdg_usd": "mdg_strategy_eq",
    "mdg_w_usd": "mdg_strategy_eq_w",
    "omega_ratio_usd": "omega_ratio_strategy_eq",
    "omega_ratio_w_usd": "omega_ratio_strategy_eq_w",
    "sharpe_ratio_usd": "sharpe_ratio_strategy_eq",
    "sharpe_ratio_w_usd": "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_usd": "sortino_ratio_strategy_eq",
    "sortino_ratio_w_usd": "sortino_ratio_strategy_eq_w",
    "sterling_ratio_usd": "sterling_ratio_strategy_eq",
    "sterling_ratio_w_usd": "sterling_ratio_strategy_eq_w",
}

_USD_PER_EXPOSURE_METRICS = {
    "adg_per_exposure_long_usd": ("adg_strategy_eq", "long"),
    "adg_per_exposure_short_usd": ("adg_strategy_eq", "short"),
    "adg_w_per_exposure_long_usd": ("adg_strategy_eq_w", "long"),
    "adg_w_per_exposure_short_usd": ("adg_strategy_eq_w", "short"),
    "gain_per_exposure_long_usd": ("gain_strategy_eq", "long"),
    "gain_per_exposure_short_usd": ("gain_strategy_eq", "short"),
    "mdg_per_exposure_long_usd": ("mdg_strategy_eq", "long"),
    "mdg_per_exposure_short_usd": ("mdg_strategy_eq", "short"),
    "mdg_w_per_exposure_long_usd": ("mdg_strategy_eq_w", "long"),
    "mdg_w_per_exposure_short_usd": ("mdg_strategy_eq_w", "short"),
}

# Keep the public proxy surface deliberately narrow. Exact Rust evaluations
# still emit the normal complete metric set; this list governs only which
# metrics may guide Metal screening or proxy-side limits.
SUPPORTED_METRICS = (
    "adg_pnl",
    "adg_pnl_w",
    "adg_strategy_eq",
    "adg_strategy_eq_w",
    "backtest_completion_ratio",
    "calmar_ratio_strategy_eq",
    "calmar_ratio_strategy_eq_w",
    "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_strategy_eq",
    "expected_shortfall_1pct_strategy_eq",
    "entry_initial_balance_pct_long",
    "entry_initial_balance_pct_short",
    "exposure_mean_ratio_usd",
    "exposure_ratio_usd",
    "fills_gap_longest_days",
    "fills_gap_mean_hours",
    "fills_gap_median_hours",
    "fills_gap_p95_hours",
    "fills_gap_p99_hours",
    "gain_strategy_eq",
    "hard_stop_duration_minutes_max",
    "hard_stop_duration_minutes_mean",
    "hard_stop_flatten_time_minutes_mean",
    "hard_stop_halt_to_restart_equity_loss_pct",
    "hard_stop_panic_close_loss_drawdown_pct_max",
    "hard_stop_panic_close_loss_drawdown_pct_mean",
    "hard_stop_panic_close_loss_drawdown_pct_min",
    "hard_stop_panic_close_loss_max",
    "hard_stop_panic_close_loss_sum",
    "hard_stop_post_restart_retrigger_pct",
    "hard_stop_restarts",
    "hard_stop_restarts_long",
    "hard_stop_restarts_per_year",
    "hard_stop_restarts_per_year_long",
    "hard_stop_restarts_per_year_short",
    "hard_stop_restarts_short",
    "hard_stop_time_in_orange_pct",
    "hard_stop_time_in_red_pct",
    "hard_stop_time_in_yellow_pct",
    "hard_stop_trigger_drawdown_mean",
    "hard_stop_triggers",
    "hard_stop_triggers_long",
    "hard_stop_triggers_per_year",
    "hard_stop_triggers_short",
    "loss_profit_ratio",
    "mdg_strategy_eq",
    "mdg_strategy_eq_w",
    "mdg_pnl",
    "mdg_pnl_w",
    "omega_ratio_strategy_eq",
    "omega_ratio_strategy_eq_w",
    "position_held_days_max",
    "position_held_hours_max",
    "position_unchanged_days_max",
    "position_unchanged_hours_max",
    "peak_recovery_hours_strategy_eq",
    "sharpe_ratio_strategy_eq",
    "sharpe_ratio_strategy_eq_w",
    "sharpe_ratio_pnl",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_strategy_eq",
    "sortino_ratio_strategy_eq_w",
    "sortino_ratio_pnl",
    "sortino_ratio_pnl_w",
    "sterling_ratio_strategy_eq",
    "sterling_ratio_strategy_eq_w",
    "strategy_eq_recovery_days_max",
    "strategy_eq_underwater_pct_mean",
    "strategy_eq_underwater_pct_median",
    "total_wallet_exposure_max",
    "total_wallet_exposure_mean",
    "volume_pct_per_day_avg",
    *_USD_STRATEGY_EQ_ALIASES,
    *_USD_PER_EXPOSURE_METRICS,
)

# Metrics backed by additional per-fill aggregates emitted by Metal.
EXTRA_KERNEL_METRICS = ("loss_profit_ratio",)

_PNL_METRICS = {
    "adg_pnl",
    "mdg_pnl",
    "sharpe_ratio_pnl",
    "sortino_ratio_pnl",
}

_WEIGHTED_PNL_METRICS = {
    "adg_pnl_w",
    "mdg_pnl_w",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl_w",
}


_GAP_HIST_BINS = 128
_GAP_HIST_LOG_MAX = math.log(4_000_001.0)
_FILL_GAP_HISTOGRAM_METRICS = {
    "fills_gap_mean_hours",
    "fills_gap_median_hours",
    "fills_gap_p95_hours",
    "fills_gap_p99_hours",
}
_HARD_STOP_LIFECYCLE_METRICS = {
    "hard_stop_duration_minutes_max",
    "hard_stop_duration_minutes_mean",
    "hard_stop_flatten_time_minutes_mean",
    "hard_stop_post_restart_retrigger_pct",
    "hard_stop_restarts",
    "hard_stop_restarts_long",
    "hard_stop_restarts_per_year",
    "hard_stop_restarts_per_year_long",
    "hard_stop_restarts_per_year_short",
    "hard_stop_restarts_short",
    "hard_stop_time_in_orange_pct",
    "hard_stop_time_in_red_pct",
    "hard_stop_time_in_yellow_pct",
    "hard_stop_trigger_drawdown_mean",
    "hard_stop_triggers",
    "hard_stop_triggers_long",
    "hard_stop_triggers_per_year",
    "hard_stop_triggers_short",
}
_HARD_STOP_PANIC_LOSS_METRICS = {
    "hard_stop_halt_to_restart_equity_loss_pct",
    "hard_stop_panic_close_loss_drawdown_pct_max",
    "hard_stop_panic_close_loss_drawdown_pct_mean",
    "hard_stop_panic_close_loss_drawdown_pct_min",
    "hard_stop_panic_close_loss_max",
    "hard_stop_panic_close_loss_sum",
}
HARD_STOP_PROXY_METRICS = tuple(
    sorted(_HARD_STOP_LIFECYCLE_METRICS | _HARD_STOP_PANIC_LOSS_METRICS)
)


def _loss_profit_ratio(loss_sum: torch.Tensor, profit_sum: torch.Tensor):
    """Match Rust's capped gross close-fill loss/profit ratio contract."""

    loss = loss_sum.to(torch.float64)
    profit = profit_sum.to(torch.float64)
    cap = torch.full_like(profit, 1_000.0)
    neutral = torch.ones_like(profit)
    ratio = loss / profit
    finite_ratio = torch.where(
        torch.isfinite(ratio),
        ratio.clamp(max=1_000.0),
        torch.where(loss > 1.0e-12, cap, neutral),
    )
    return torch.where(
        profit <= 1.0e-12,
        torch.where(loss > 1.0e-12, cap, neutral),
        finite_ratio,
    )
# Metal classifies gaps with float32 logarithms. Expand the decoded boundary
# by 1024 unit roundoffs so a value rounded into the preceding bin cannot make
# this minimizing proxy optimistic.
_GAP_HIST_EDGE_MARGIN = 1.220703125e-4
_GAP_HIST_UPPER_STEPS = tuple(
    max(
        0,
        math.ceil(
            (
                math.exp((index + 1) * _GAP_HIST_LOG_MAX / 127.0)
                - 1.0
            )
            * (1.0 + _GAP_HIST_EDGE_MARGIN)
        )
        - 1,
    )
    for index in range(_GAP_HIST_BINS - 1)
) + (float("inf"),)


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


def _weighted_percentile(values, counts, percentile):
    """Interpolate a percentile over sorted values with integer multiplicities."""

    ordered_values, order = torch.sort(values, dim=1)
    ordered_counts = counts.gather(1, order)
    cumulative = ordered_counts.cumsum(dim=1)
    total = ordered_counts.sum(dim=1)
    rank = float(percentile) * (total - 1).clamp(min=0).to(values.dtype)
    lower_rank = torch.floor(rank).to(torch.long)
    upper_rank = torch.ceil(rank).to(torch.long)

    def gather_rank(target):
        index = (cumulative > target.unsqueeze(1)).to(torch.int64).argmax(dim=1)
        return ordered_values.gather(1, index.unsqueeze(1)).squeeze(1)

    lower = gather_rank(lower_rank)
    upper = gather_rank(upper_rank)
    weight = rank - lower_rank.to(values.dtype)
    result = lower * (1.0 - weight) + upper * weight
    return torch.where(total > 0, result, torch.zeros_like(result))


def _fill_gap_metrics(out, run):
    """Conservatively reduce coalesced fill timestamps and log-gap bins."""

    interval_ms = max(float(run.interval_ms), 1.0)
    first_eq_ts = out["first_eq_ts"].to(torch.float64)
    last_eq_ts = out["last_eq_ts"].to(torch.float64)
    first_fill_ts = out["first_fill_ts"].to(torch.float64)
    last_fill_ts = out["last_fill_ts"].to(torch.float64)
    has_equity = (
        torch.isfinite(first_eq_ts)
        & torch.isfinite(last_eq_ts)
        & (last_eq_ts >= first_eq_ts)
    )
    has_fill = (
        has_equity
        & torch.isfinite(first_fill_ts)
        & torch.isfinite(last_fill_ts)
    )
    # Metal exports integer candle indices multiplied by interval_ms through a
    # float32 scalar buffer. Recover the indices before subtracting so rounding
    # of large millisecond offsets cannot make a boundary gap optimistic.
    first_eq_step = torch.round(first_eq_ts / interval_ms)
    last_eq_step = torch.round(last_eq_ts / interval_ms)
    first_fill_step = torch.round(first_fill_ts / interval_ms)
    last_fill_step = torch.round(last_fill_ts / interval_ms)
    span_ms = torch.where(
        has_equity,
        (last_eq_step - first_eq_step).clamp(min=0.0) * interval_ms,
        torch.zeros_like(first_eq_ts),
    )
    span_steps = span_ms / interval_ms
    upper_steps = torch.tensor(
        _GAP_HIST_UPPER_STEPS,
        dtype=torch.float64,
        device=first_eq_ts.device,
    ).unsqueeze(0)
    upper_steps = torch.minimum(upper_steps, span_steps.unsqueeze(1))
    gap_values = upper_steps * interval_ms / 3_600_000.0
    gap_counts = out["gap_hist"].to(torch.long)
    gap_counts = torch.where(
        has_fill.unsqueeze(1), gap_counts, torch.zeros_like(gap_counts)
    )

    lead_hours = torch.where(
        has_fill,
        (first_fill_step - first_eq_step).clamp(min=0.0)
        * interval_ms
        / 3_600_000.0,
        span_ms / 3_600_000.0,
    )
    trail_hours = torch.where(
        has_fill,
        (last_eq_step - last_fill_step).clamp(min=0.0)
        * interval_ms
        / 3_600_000.0,
        torch.zeros_like(span_ms),
    )
    boundary_values = torch.stack((lead_hours, trail_hours), dim=1)
    boundary_counts = torch.stack(
        (
            has_equity.to(torch.long),
            has_fill.to(torch.long),
        ),
        dim=1,
    )
    values = torch.cat((gap_values, boundary_values), dim=1)
    counts = torch.cat((gap_counts, boundary_counts), dim=1)
    total = counts.sum(dim=1).clamp(min=1).to(torch.float64)
    weighted_values = torch.where(
        counts > 0, values, torch.zeros_like(values)
    )
    mean = (weighted_values * counts.to(values.dtype)).sum(dim=1) / total
    return {
        "fills_gap_mean_hours": mean,
        "fills_gap_median_hours": _weighted_percentile(values, counts, 0.50),
        "fills_gap_p95_hours": _weighted_percentile(values, counts, 0.95),
        "fills_gap_p99_hours": _weighted_percentile(values, counts, 0.99),
    }


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


def _daily_pnl_stats(day_net_pnl, day_last_fill_balance, mask):
    finite_mask = (
        mask
        & torch.isfinite(day_net_pnl)
        & torch.isfinite(day_last_fill_balance)
    )
    ratios = torch.where(
        finite_mask,
        day_net_pnl / day_last_fill_balance.abs().clamp(min=1e-12),
        torch.zeros_like(day_net_pnl),
    )
    count = finite_mask.sum(dim=1)
    adg = ratios.sum(dim=1) / count.clamp(min=1).to(ratios.dtype)
    adg = torch.where(count > 0, adg, torch.zeros_like(adg))
    mdg = _masked_median(ratios, finite_mask)
    sharpe, sortino = _sharpe_sortino(ratios, finite_mask, adg)
    return adg, mdg, sharpe, sortino, count


def _weighted_pnl_metrics(
    day_net_pnl,
    day_last_fill_balance,
    day_fill_count,
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
    requested,
):
    requested = set(requested) & _WEIGHTED_PNL_METRICS
    if not requested:
        return {}
    _, subsets = _weighted_subsets(
        active,
        first_eq_ts,
        last_eq_ts,
        first_timestamp,
        interval_ms,
    )
    fill_count = torch.where(
        active, day_fill_count, torch.zeros_like(day_fill_count)
    )
    eligible = fill_count.sum(dim=1) > 1.0
    totals = {
        name: torch.zeros(
            day_net_pnl.shape[0],
            dtype=day_net_pnl.dtype,
            device=day_net_pnl.device,
        )
        for name in requested
    }
    for subset in subsets:
        subset_fill_count = torch.where(
            subset, fill_count, torch.zeros_like(fill_count)
        ).sum(dim=1)
        adg, mdg, sharpe, sortino, _ = _daily_pnl_stats(
            day_net_pnl,
            day_last_fill_balance,
            subset & (fill_count > 0.0),
        )
        include = eligible & (subset_fill_count > 0.0)
        values = {
            "adg_pnl_w": adg,
            "mdg_pnl_w": mdg,
            "sharpe_ratio_pnl_w": sharpe,
            "sortino_ratio_pnl_w": sortino,
        }
        for name in requested:
            totals[name] += torch.where(
                include, values[name], torch.zeros_like(values[name])
            )
    return {name: value / 10.0 for name, value in totals.items()}


def _hard_stop_lifecycle_metrics(out: dict, run) -> dict:
    """Reduce directional HSL counters using the authoritative Rust formulas."""

    reference = out["max_dd"].to(torch.float64)
    zeros = torch.zeros_like(reference)
    if "hsl_triggers_long" not in out:
        raise RuntimeError(
            "MPS directional HSL lifecycle outputs are missing from proxy results"
        )

    def value(name: str):
        return out[name].to(torch.float64)

    triggers_long = value("hsl_triggers_long")
    triggers_short = value("hsl_triggers_short")
    restarts_long = value("hsl_restarts_long")
    restarts_short = value("hsl_restarts_short")
    triggers = triggers_long + triggers_short
    restarts = restarts_long + restarts_short
    sample_count = value("hsl_tier_samples_total")
    first_eq_ts = out["first_eq_ts"].to(torch.float64)
    last_eq_ts = out["last_eq_ts"].to(torch.float64)
    interval_days = float(run.interval_ms) / 86_400_000.0
    timestamp_days = ((last_eq_ts - first_eq_ts) / 86_400_000.0).clamp(
        min=interval_days
    )
    has_equity = (
        torch.isfinite(first_eq_ts)
        & torch.isfinite(last_eq_ts)
        & (last_eq_ts >= first_eq_ts)
    )
    n_days = torch.where(has_equity, timestamp_days, zeros)
    per_year_scale = torch.where(n_days > 0.0, 365.25 / n_days, zeros)

    total_samples = sample_count.clamp(min=1.0)
    duration_count = value("hsl_duration_count")
    trigger_drawdown_count = value("hsl_trigger_drawdown_count")
    flatten_count = value("hsl_flatten_time_count")
    minutes_per_step = float(run.interval_ms) / 60_000.0

    return {
        "hard_stop_triggers": triggers,
        "hard_stop_triggers_per_year": triggers * per_year_scale,
        "hard_stop_triggers_long": triggers_long,
        "hard_stop_triggers_short": triggers_short,
        "hard_stop_restarts": restarts,
        "hard_stop_restarts_per_year": restarts * per_year_scale,
        "hard_stop_restarts_per_year_long": restarts_long * per_year_scale,
        "hard_stop_restarts_per_year_short": restarts_short * per_year_scale,
        "hard_stop_restarts_long": restarts_long,
        "hard_stop_restarts_short": restarts_short,
        "hard_stop_time_in_yellow_pct": value("hsl_tier_samples_yellow")
        / total_samples,
        "hard_stop_time_in_orange_pct": value("hsl_tier_samples_orange")
        / total_samples,
        "hard_stop_time_in_red_pct": value("hsl_tier_samples_red")
        / total_samples,
        "hard_stop_duration_minutes_mean": torch.where(
            duration_count > 0.0,
            value("hsl_duration_sum_steps") / duration_count * minutes_per_step,
            zeros,
        ),
        "hard_stop_duration_minutes_max": value("hsl_duration_max_steps")
        * minutes_per_step,
        "hard_stop_trigger_drawdown_mean": torch.where(
            trigger_drawdown_count > 0.0,
            value("hsl_trigger_drawdown_sum") / trigger_drawdown_count,
            zeros,
        ),
        "hard_stop_flatten_time_minutes_mean": torch.where(
            flatten_count > 0.0,
            value("hsl_flatten_time_sum_steps") / flatten_count * minutes_per_step,
            zeros,
        ),
        "hard_stop_post_restart_retrigger_pct": torch.where(
            restarts > 0.0,
            value("hsl_restart_retrigger_count") / restarts,
            zeros,
        ),
    }


def _hard_stop_panic_loss_metrics(out: dict, run) -> dict:
    """Reduce resting-limit panic-fill losses using the Rust HSL formulas."""

    reference = out["max_dd"].to(torch.float64)
    zeros = torch.zeros_like(reference)
    if "hsl_panic_close_loss_sum" not in out:
        raise RuntimeError(
            "MPS directional HSL panic-loss outputs are missing from proxy results"
        )

    def value(name: str):
        return out[name].to(torch.float64)

    drawdown_count = value("hsl_panic_loss_drawdown_count")
    return {
        "hard_stop_halt_to_restart_equity_loss_pct": value(
            "hsl_halt_to_restart_equity_loss"
        )
        / max(float(run.starting_balance), 1.0e-12),
        "hard_stop_panic_close_loss_sum": value("hsl_panic_close_loss_sum"),
        "hard_stop_panic_close_loss_max": value("hsl_panic_close_loss_max"),
        "hard_stop_panic_close_loss_drawdown_pct_min": value(
            "hsl_panic_loss_drawdown_min"
        ),
        "hard_stop_panic_close_loss_drawdown_pct_mean": torch.where(
            drawdown_count > 0.0,
            value("hsl_panic_loss_drawdown_sum") / drawdown_count,
            zeros,
        ),
        "hard_stop_panic_close_loss_drawdown_pct_max": value(
            "hsl_panic_loss_drawdown_max"
        ),
    }


def compute_objectives(out: dict, run, data: dict, needed=None) -> dict:
    """Reduce compact Metal output into validated proxy objective metrics."""

    day_end_eq = out["day_end_eq"].to(torch.float64)
    day_min_eq = out["day_min_eq"].to(torch.float64)
    day_max_dd = out["day_max_dd"].to(torch.float64)
    day_volume = out["day_volume"].to(torch.float64)
    day_has_fill = out["day_has_fill"]
    active = _daily_series_masks(out["day_min_eq"])
    requested = set(SUPPORTED_METRICS if needed is None else needed)
    requested_sources = requested | {
        source
        for alias, source in _USD_STRATEGY_EQ_ALIASES.items()
        if alias in requested
    } | {
        source
        for metric, (source, _side) in _USD_PER_EXPOSURE_METRICS.items()
        if metric in requested
    }

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
        requested_sources,
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

    zeros = torch.zeros_like(adg)
    adg_pnl = mdg_pnl = sharpe_pnl = sortino_pnl = zeros
    weighted_pnl_metrics = {}
    if requested & (_PNL_METRICS | _WEIGHTED_PNL_METRICS):
        day_net_pnl = out["day_net_pnl"].to(torch.float64)
        day_last_fill_balance = out["day_last_fill_balance"].to(torch.float64)
        if requested & _PNL_METRICS:
            adg_pnl, mdg_pnl, sharpe_pnl, sortino_pnl, _ = _daily_pnl_stats(
                day_net_pnl,
                day_last_fill_balance,
                day_has_fill & active,
            )
        if requested & _WEIGHTED_PNL_METRICS:
            weighted_pnl_metrics = _weighted_pnl_metrics(
                day_net_pnl,
                day_last_fill_balance,
                out["day_fill_count"].to(torch.float64),
                active,
                out["first_eq_ts"],
                out["last_eq_ts"],
                data["ts0"],
                run.interval_ms,
                requested,
            )

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
    fill_gap_metrics = (
        _fill_gap_metrics(out, run)
        if requested & _FILL_GAP_HISTOGRAM_METRICS
        else {}
    )
    hard_stop_metrics = (
        _hard_stop_lifecycle_metrics(out, run)
        if requested & _HARD_STOP_LIFECYCLE_METRICS
        else {}
    )
    hard_stop_panic_loss_metrics = (
        _hard_stop_panic_loss_metrics(out, run)
        if requested & _HARD_STOP_PANIC_LOSS_METRICS
        else {}
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
        "adg_pnl": adg_pnl,
        "adg_strategy_eq": adg,
        "backtest_completion_ratio": completion,
        "calmar_ratio_strategy_eq": calmar,
        "drawdown_worst_mean_1pct_strategy_eq": worst_one_pct,
        "drawdown_worst_strategy_eq": out["max_dd"],
        "expected_shortfall_1pct_strategy_eq": expected_shortfall,
        "fills_gap_longest_days": gap_longest_days,
        "gain_strategy_eq": gain,
        "mdg_strategy_eq": mdg,
        "mdg_pnl": mdg_pnl,
        "omega_ratio_strategy_eq": omega,
        "position_held_days_max": held_days,
        "position_held_hours_max": held_days * 24.0,
        "sharpe_ratio_strategy_eq": sharpe,
        "sharpe_ratio_pnl": sharpe_pnl,
        "sortino_ratio_strategy_eq": sortino,
        "sortino_ratio_pnl": sortino_pnl,
        "sterling_ratio_strategy_eq": sterling,
        "strategy_eq_recovery_days_max": recovery_max_days,
        "peak_recovery_hours_strategy_eq": recovery_max_days * 24.0,
        "strategy_eq_underwater_pct_mean": underwater,
        "strategy_eq_underwater_pct_median": underwater_median,
        "volume_pct_per_day_avg": volume_pct,
    }
    if "loss_profit_ratio" in requested:
        objectives["loss_profit_ratio"] = _loss_profit_ratio(
            out["loss_sum"], out["profit_sum"]
        )
    for name in ("total_wallet_exposure_max", "total_wallet_exposure_mean"):
        if name in requested:
            objectives[name] = out[name].to(torch.float64)
    exposure_ratio_denominators = {
        "exposure_ratio_usd": "total_wallet_exposure_max",
        "exposure_mean_ratio_usd": "total_wallet_exposure_mean",
    }
    for name, denominator in exposure_ratio_denominators.items():
        if name in requested:
            objectives[name] = adg / out[denominator].to(torch.float64).abs().clamp(
                min=1e-12
            )
    if {"position_unchanged_days_max", "position_unchanged_hours_max"} & requested:
        position_unchanged_hours_max = (
            out["position_unchanged_max_ms"] / 3_600_000.0
        )
        objectives["position_unchanged_hours_max"] = position_unchanged_hours_max
        objectives["position_unchanged_days_max"] = position_unchanged_hours_max / 24.0
    for side in ("long", "short"):
        name = f"entry_initial_balance_pct_{side}"
        if name in requested:
            objectives[name] = out[name].to(torch.float64)
    objectives.update(fill_gap_metrics)
    objectives.update(hard_stop_metrics)
    objectives.update(hard_stop_panic_loss_metrics)
    objectives.update(weighted_metrics)
    objectives.update(weighted_pnl_metrics)
    for name, (source, side) in _USD_PER_EXPOSURE_METRICS.items():
        if name not in requested:
            continue
        denominator = out[
            f"candidate_total_wallet_exposure_limit_{side}"
        ].to(torch.float64)
        objectives[name] = torch.where(
            denominator > 0.0,
            objectives[source] / denominator,
            torch.zeros_like(objectives[source]),
        )
    for alias, source in _USD_STRATEGY_EQ_ALIASES.items():
        if alias in requested:
            objectives[alias] = objectives[source]
    if needed is None:
        return objectives
    return {key: value for key, value in objectives.items() if key in requested}
