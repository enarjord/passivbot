"""Metric reductions for the Apple MPS screening proxy.

The daily-series reduction approach was adapted from ``src/gpu/torch_metrics.py``
in RustyCZ's Passivbot GPU branch (commit 7c529bc73), then narrowed to the
metrics validated for this foundation and integrated with the current schema.
"""

from __future__ import annotations

import math

import torch

from optimization.gpu.metric_registry import (
    BTC_INTRADAY_RISK_METRICS,
    ENTRY_INTERVAL_METRICS,
    EQUITY_BALANCE_DIFF_METRICS,
    GPU_EXACT_ONLY_METRICS,
    HARD_STOP_LIFECYCLE_METRICS,
    HARD_STOP_PANIC_LOSS_METRICS,
    HARD_STOP_PROXY_METRICS,
    reject_exact_only_gpu_metric_names,
)

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

_BTC_ACCOUNT_METRICS = {
    "adg_btc",
    "adg_w_btc",
    "equity_choppiness_btc",
    "equity_choppiness_w_btc",
    "equity_jerkiness_btc",
    "equity_jerkiness_w_btc",
    "exponential_fit_error_btc",
    "exponential_fit_error_w_btc",
    "exposure_mean_ratio_btc",
    "exposure_ratio_btc",
    "gain_btc",
    "mdg_btc",
    "mdg_w_btc",
    "omega_ratio_btc",
    "omega_ratio_w_btc",
    "peak_recovery_days_equity_btc",
    "peak_recovery_hours_equity_btc",
}

_BTC_ACCOUNT_METRICS.update(BTC_INTRADAY_RISK_METRICS)
_BTC_ACCOUNT_METRICS.update(
    metric for metric in EQUITY_BALANCE_DIFF_METRICS if metric.endswith("_btc")
)

_BTC_PER_EXPOSURE_METRICS = {
    f"{metric}_per_exposure_{side}_btc"
    for metric in ("adg", "adg_w", "gain", "mdg", "mdg_w")
    for side in ("long", "short")
}

BTC_ACCOUNT_METRICS = frozenset(
    _BTC_ACCOUNT_METRICS | _BTC_PER_EXPOSURE_METRICS
)

# Keep the public proxy surface deliberately narrow. Exact Rust evaluations
# still emit the normal complete metric set; this list governs only which
# metrics may guide Metal screening or proxy-side limits.
_GPU_PROXY_METRIC_CANDIDATES = (
    "adg_strategy_eq",
    "adg_strategy_eq_w",
    "backtest_completion_ratio",
    "calmar_ratio_strategy_eq",
    "calmar_ratio_strategy_eq_w",
    "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_mean_1pct_strategy_eq_long",
    "drawdown_worst_mean_1pct_strategy_eq_short",
    "drawdown_worst_strategy_eq_long",
    "drawdown_worst_strategy_eq_short",
    "drawdown_worst_mean_1pct_ema_strategy_eq",
    "drawdown_worst_mean_1pct_ema_strategy_eq_long",
    "drawdown_worst_mean_1pct_ema_strategy_eq_short",
    "drawdown_worst_ema_strategy_eq",
    "drawdown_worst_ema_strategy_eq_long",
    "drawdown_worst_ema_strategy_eq_short",
    "drawdown_worst_strategy_eq",
    "equity_choppiness_usd",
    "equity_choppiness_w_usd",
    "equity_jerkiness_usd",
    "equity_jerkiness_w_usd",
    "expected_shortfall_1pct_strategy_eq",
    "entry_initial_balance_pct_long",
    "entry_initial_balance_pct_short",
    *sorted(ENTRY_INTERVAL_METRICS),
    "exponential_fit_error_usd",
    "exponential_fit_error_w_usd",
    "exposure_mean_ratio_usd",
    "exposure_ratio_usd",
    "fills_active_days_ratio",
    "fills_active_symbols_count",
    "fills_entry_per_close",
    "fills_gap_longest_days",
    "fills_gap_p95_hours",
    "fills_gap_time_weighted_mean_hours",
    "fills_per_day",
    "fills_per_day_per_position_slot",
    "fills_top_symbol_share",
    "hard_stop_duration_minutes_max",
    "hard_stop_duration_minutes_mean",
    "hard_stop_halt_to_restart_equity_loss_pct",
    "hard_stop_panic_close_loss_drawdown_pct_max",
    "hard_stop_panic_close_loss_drawdown_pct_mean",
    "hard_stop_post_restart_retrigger_pct",
    "hard_stop_restarts_per_year",
    "hard_stop_restarts_per_year_long",
    "hard_stop_restarts_per_year_short",
    "hard_stop_time_in_red_pct",
    "hard_stop_trigger_drawdown_mean",
    "hard_stop_triggers_per_year",
    "loss_profit_ratio",
    "loss_profit_ratio_long",
    "loss_profit_ratio_short",
    "mdg_strategy_eq",
    "mdg_strategy_eq_w",
    "omega_ratio_strategy_eq",
    "omega_ratio_strategy_eq_w",
    "pnl_ratio_long_short",
    "position_held_days_mean",
    "position_held_days_max",
    "position_held_hours_mean",
    "position_held_hours_max",
    "positions_held_per_day",
    "position_unchanged_days_max",
    "position_unchanged_hours_max",
    "peak_recovery_hours_strategy_eq_long",
    "peak_recovery_hours_strategy_eq_short",
    "peak_recovery_days_strategy_eq_long",
    "peak_recovery_days_strategy_eq_short",
    "sharpe_ratio_strategy_eq",
    "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_strategy_eq",
    "sortino_ratio_strategy_eq_w",
    "sterling_ratio_strategy_eq",
    "sterling_ratio_strategy_eq_w",
    "strategy_eq_recovery_days_mean",
    "strategy_eq_recovery_days_median",
    "strategy_eq_recovery_days_p95",
    "strategy_eq_recovery_days_p99",
    "strategy_eq_recovery_days_mean_worst_5pct",
    "strategy_eq_recovery_days_mean_worst_1pct",
    "strategy_eq_recovery_days_max",
    "strategy_eq_underwater_pct_mean",
    "total_wallet_exposure_max",
    "total_wallet_exposure_mean",
    "volume_pct_per_day_avg",
    "volume_pct_per_day_avg_w",
    *_USD_STRATEGY_EQ_ALIASES,
    *_USD_PER_EXPOSURE_METRICS,
    *sorted(BTC_ACCOUNT_METRICS),
    *sorted(
        metric
        for metric in EQUITY_BALANCE_DIFF_METRICS
        if metric.endswith("_usd")
    ),
)

SUPPORTED_METRICS = tuple(
    metric
    for metric in _GPU_PROXY_METRIC_CANDIDATES
    if metric not in GPU_EXACT_ONLY_METRICS
)

assert not set(SUPPORTED_METRICS) & GPU_EXACT_ONLY_METRICS


def validate_gpu_metric_names(metric_names) -> frozenset[str]:
    """Return canonical proxy metrics, rejecting exact-only names before aliases."""

    from config.metrics import canonicalize_metric_name

    raw = reject_exact_only_gpu_metric_names(metric_names)
    canonical = frozenset(canonicalize_metric_name(name) for name in raw)
    unsupported = sorted(canonical - set(SUPPORTED_METRICS))
    if unsupported:
        raise ValueError(
            f"GPU foundation does not implement optimizer metrics {unsupported}; "
            "use supported metrics or the CPU optimizer"
        )
    return canonical

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

_FILL_ACTIVITY_METRICS = {
    "fills_analysis_duration_days",
    "fills_active_days_count",
    "fills_active_days_ratio",
    "fills_active_symbols_count",
    "fills_count",
    "fills_count_close",
    "fills_count_entry",
    "fills_count_long",
    "fills_count_short",
    "fills_entry_per_close",
    "fills_per_day",
    "fills_per_day_close",
    "fills_per_day_entry",
    "fills_per_day_long",
    "fills_per_day_per_position_slot",
    "fills_per_day_per_position_slot_long",
    "fills_per_day_per_position_slot_short",
    "fills_per_day_short",
    "fills_top_symbol_share",
}


_GAP_HIST_BINS = 128
_GAP_HIST_LOG_MAX = math.log(4_000_001.0)
_FILL_GAP_HISTOGRAM_METRICS = {
    "fills_gap_mean_hours",
    "fills_gap_median_hours",
    "fills_gap_p95_hours",
    "fills_gap_p99_hours",
    "fills_gap_time_weighted_mean_hours",
}
_HARD_STOP_LIFECYCLE_METRICS = HARD_STOP_LIFECYCLE_METRICS
_HARD_STOP_PANIC_LOSS_METRICS = HARD_STOP_PANIC_LOSS_METRICS
_HARD_STOP_EMA_DRAWDOWN_METRICS = {
    "drawdown_worst_ema_strategy_eq",
    "drawdown_worst_ema_strategy_eq_long",
    "drawdown_worst_ema_strategy_eq_short",
}
_HARD_STOP_RAW_DRAWDOWN_METRICS = {
    "drawdown_worst_strategy_eq_long",
    "drawdown_worst_strategy_eq_short",
    "drawdown_worst_mean_1pct_strategy_eq_long",
    "drawdown_worst_mean_1pct_strategy_eq_short",
}
_HARD_STOP_EMA_TAIL_METRICS = {
    "drawdown_worst_mean_1pct_ema_strategy_eq",
    "drawdown_worst_mean_1pct_ema_strategy_eq_long",
    "drawdown_worst_mean_1pct_ema_strategy_eq_short",
}
_HARD_STOP_STRATEGY_EQ_RECOVERY_METRICS = {
    "peak_recovery_hours_strategy_eq_long",
    "peak_recovery_hours_strategy_eq_short",
    "peak_recovery_days_strategy_eq_long",
    "peak_recovery_days_strategy_eq_short",
}
_STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS = {
    "strategy_eq_recovery_days_mean",
    "strategy_eq_recovery_days_median",
    "strategy_eq_recovery_days_p95",
    "strategy_eq_recovery_days_p99",
    "strategy_eq_recovery_days_mean_worst_5pct",
    "strategy_eq_recovery_days_mean_worst_1pct",
}
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


def _directional_pnl_metrics(out: dict) -> dict:
    """Match Rust's directional gross-loss and signed-PnL ratio contracts."""

    values = {}
    for side in ("long", "short"):
        values[f"loss_profit_ratio_{side}"] = _loss_profit_ratio(
            out[f"loss_sum_{side}"], out[f"profit_sum_{side}"]
        )
    long_pnl = out["profit_sum_long"].to(torch.float64) - out[
        "loss_sum_long"
    ].to(torch.float64)
    short_pnl = out["profit_sum_short"].to(torch.float64) - out[
        "loss_sum_short"
    ].to(torch.float64)
    pnl_sum = long_pnl + short_pnl
    long_short_ratio = torch.where(
        pnl_sum != 0.0,
        long_pnl / pnl_sum,
        torch.full_like(pnl_sum, 0.5),
    )
    values["pnl_ratio_long_short"] = long_short_ratio
    return values
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


def _equity_shape_metrics(day_eq, active):
    """Match Rust's shape metrics over active daily closing equity samples."""

    batch_size, day_count = day_eq.shape
    zeros = torch.zeros(batch_size, dtype=day_eq.dtype, device=day_eq.device)
    if day_count == 0:
        return {
            "equity_choppiness_usd": zeros,
            "equity_jerkiness_usd": zeros,
            "exponential_fit_error_usd": torch.full_like(zeros, float("inf")),
        }

    indices = (
        torch.arange(day_count, device=day_eq.device)
        .unsqueeze(0)
        .expand(batch_size, day_count)
    )
    counts = active.sum(dim=1)
    # Rust builds a compact Vec of touched UTC-day closes. Preserve that order
    # when an entirely invalid day leaves a hole in the fixed proxy surface.
    compact_order = torch.argsort(
        torch.where(active, indices, indices + day_count), dim=1
    )
    day_eq = day_eq.gather(1, compact_order)
    active = indices < counts.unsqueeze(1)
    first = day_eq[:, 0]
    last_index = (counts - 1).clamp(min=0)
    last = day_eq.gather(1, last_index.unsqueeze(1)).squeeze(1)

    adjacent = active[:, :-1] & active[:, 1:]
    variation = torch.where(
        adjacent,
        (day_eq[:, 1:] - day_eq[:, :-1]).abs(),
        torch.zeros_like(day_eq[:, 1:]),
    ).sum(dim=1)
    net_gain = (last - first).abs()
    epsilon = torch.finfo(day_eq.dtype).eps
    choppiness = torch.where(
        counts < 2,
        zeros,
        torch.where(
            net_gain < epsilon,
            torch.full_like(zeros, float("inf")),
            variation / net_gain.clamp(min=epsilon),
        ),
    )

    if day_count < 3:
        jerkiness = zeros
    else:
        triples = active[:, :-2] & active[:, 1:-1] & active[:, 2:]
        numerator = (
            day_eq[:, 2:] - 2.0 * day_eq[:, 1:-1] + day_eq[:, :-2]
        ).abs()
        denominator = (
            day_eq[:, :-2] + day_eq[:, 1:-1] + day_eq[:, 2:]
        ) / 3.0
        terms = torch.where(
            triples & (denominator.abs() >= epsilon),
            numerator / denominator.abs().clamp(min=epsilon),
            torch.zeros_like(numerator),
        )
        jerkiness = terms.sum(dim=1) / (counts - 2).clamp(min=1).to(day_eq.dtype)
        jerkiness = torch.where(counts >= 3, jerkiness, zeros)

    ordinal = active.cumsum(dim=1).to(day_eq.dtype) - 1.0
    count_float = counts.to(day_eq.dtype)
    safe_equity = torch.where(
        active, day_eq.clamp(min=epsilon), torch.ones_like(day_eq)
    )
    log_equity = torch.where(active, safe_equity.log(), torch.zeros_like(day_eq))
    x = torch.where(active, ordinal, torch.zeros_like(ordinal))
    sum_x = x.sum(dim=1)
    sum_y = log_equity.sum(dim=1)
    sum_xx = (x * x).sum(dim=1)
    sum_xy = (x * log_equity).sum(dim=1)
    fit_denominator = count_float * sum_xx - sum_x * sum_x
    safe_fit_denominator = torch.where(
        fit_denominator != 0.0, fit_denominator, torch.ones_like(fit_denominator)
    )
    slope = (count_float * sum_xy - sum_x * sum_y) / safe_fit_denominator
    intercept = (sum_y - slope * sum_x) / count_float.clamp(min=1.0)
    residual = slope.unsqueeze(1) * x + intercept.unsqueeze(1) - log_equity
    fit_error = torch.where(
        active, residual * residual, torch.zeros_like(residual)
    ).sum(dim=1) / count_float.clamp(min=1.0)
    invalid_fit = (
        (counts < 2)
        | (fit_denominator == 0.0)
        | (active & (day_eq <= 0.0)).any(dim=1)
    )
    fit_error = torch.where(
        invalid_fit, torch.full_like(fit_error, float("inf")), fit_error
    )

    return {
        "equity_choppiness_usd": choppiness,
        "equity_jerkiness_usd": jerkiness,
        "exponential_fit_error_usd": fit_error,
    }


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
    span_hours = span_ms / 3_600_000.0
    time_weighted_mean = torch.where(
        span_hours > 0.0,
        (
            weighted_values.square() * counts.to(values.dtype)
        ).sum(dim=1)
        / span_hours.clamp(min=1.0e-12),
        torch.zeros_like(span_hours),
    )
    return {
        "fills_gap_mean_hours": mean,
        "fills_gap_median_hours": _weighted_percentile(values, counts, 0.50),
        "fills_gap_p95_hours": _weighted_percentile(values, counts, 0.95),
        "fills_gap_p99_hours": _weighted_percentile(values, counts, 0.99),
        "fills_gap_time_weighted_mean_hours": time_weighted_mean,
    }


def _entry_interval_metrics(out, run, strategy_kind: str):
    """Reduce per-coin/side normal-initial-entry intervals.

    Exact Rust only classifies Trailing Martingale's ``EntryInitialNormal``
    fills for this metric family. EMA Anchor therefore retains the canonical
    all-zero result without enabling an extra Metal output surface.
    """

    zeros = torch.zeros_like(out["fill_count"].to(torch.float64))
    if strategy_kind not in {"ema_anchor", "trailing_martingale"}:
        raise RuntimeError(
            "MPS entry-interval reduction requires a recognized strategy kind"
        )
    if strategy_kind != "trailing_martingale":
        return {name: zeros for name in ENTRY_INTERVAL_METRICS}

    required = (
        "entry_interval_sum_steps",
        "entry_interval_count",
        "entry_interval_max_steps",
        "entry_interval_hist",
    )
    missing = [name for name in required if name not in out]
    if missing:
        raise RuntimeError(
            "MPS entry-interval output is missing " + ", ".join(missing)
        )
    total_steps = out["entry_interval_sum_steps"].to(torch.float64)
    counts_float = out["entry_interval_count"].to(torch.float64)
    max_steps = out["entry_interval_max_steps"].to(torch.float64)
    histogram_float = out["entry_interval_hist"].to(torch.float64)
    if any(
        not bool(torch.isfinite(value).all())
        for value in (total_steps, counts_float, max_steps, histogram_float)
    ):
        raise RuntimeError("MPS entry-interval output is non-finite")
    if bool(
        (
            (total_steps < 0.0)
            | (counts_float < 0.0)
            | (max_steps < 0.0)
            | (histogram_float < 0.0).any(dim=1)
        ).any()
    ):
        raise RuntimeError("MPS entry-interval output is negative")
    rounded_counts = torch.round(counts_float)
    rounded_histogram = torch.round(histogram_float)
    if bool(
        (
            (torch.abs(counts_float - rounded_counts) > 1.0e-4)
            | (torch.abs(histogram_float - rounded_histogram) > 1.0e-4).any(
                dim=1
            )
        ).any()
    ):
        raise RuntimeError("MPS entry-interval output contains fractional counts")
    counts = rounded_counts.to(torch.long)
    histogram = rounded_histogram.to(torch.long)
    if bool((histogram.sum(dim=1) != counts).any()):
        raise RuntimeError("MPS entry-interval histogram count disagrees with totals")
    if bool(
        (
            (max_steps > total_steps + 1.0)
            | ((counts == 0) & ((total_steps != 0.0) | (max_steps != 0.0)))
        ).any()
    ):
        raise RuntimeError("MPS entry-interval totals are inconsistent")

    interval_hours = max(float(run.interval_ms), 1.0) / 3_600_000.0
    upper_steps = torch.tensor(
        _GAP_HIST_UPPER_STEPS,
        dtype=torch.float64,
        device=total_steps.device,
    ).unsqueeze(0)
    values = torch.minimum(upper_steps, max_steps.unsqueeze(1)) * interval_hours
    count_denominator = counts_float.clamp(min=1.0)
    mean = torch.where(
        counts > 0,
        total_steps / count_denominator * interval_hours,
        zeros,
    )
    maximum = torch.where(counts > 0, max_steps * interval_hours, zeros)
    return {
        "entry_interval_hours_mean": mean,
        "entry_interval_hours_median": _weighted_percentile(
            values, histogram, 0.50
        ),
        "entry_interval_hours_p95": _weighted_percentile(
            values, histogram, 0.95
        ),
        "entry_interval_hours_p99": _weighted_percentile(
            values, histogram, 0.99
        ),
        "entry_interval_hours_max": maximum,
    }


def _weighted_subset_context(
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
):
    finite_timestamps = torch.isfinite(first_eq_ts) & torch.isfinite(last_eq_ts)
    timestamp_origin = float(first_timestamp)
    interval_ms_int = int(interval_ms)
    timestamp_origin_int = int(round(timestamp_origin))

    def relative_steps(timestamps):
        relative_ms = torch.where(
            timestamps < timestamp_origin,
            timestamps,
            timestamps - timestamp_origin,
        )
        return torch.floor(relative_ms / float(interval_ms_int) + 0.5).to(
            torch.long
        )

    first_eq_steps = relative_steps(first_eq_ts)
    last_eq_steps = relative_steps(last_eq_ts)
    sample_count = torch.where(
        finite_timestamps,
        last_eq_steps - first_eq_steps + 1,
        torch.ones_like(first_eq_steps),
    ).clamp(min=1)
    eligible = finite_timestamps & (sample_count >= 2)
    subsets = [active]
    subset_start_steps = [first_eq_steps]
    subset_start_timestamps = [
        first_eq_steps * interval_ms_int + timestamp_origin_int
    ]
    first_day = timestamp_origin_int // 86_400_000
    day_ids = torch.arange(active.shape[1], device=active.device) + first_day
    for index in range(1, 10):
        fraction = 1.0 / (1.0 + index)
        start_position = torch.floor(
            sample_count.to(first_eq_ts.dtype) * (1.0 - fraction) + 0.5
        ).to(torch.long)
        subset_start_step = first_eq_steps + start_position
        subset_start_ts = (
            subset_start_step * interval_ms_int + timestamp_origin_int
        )
        subset_start_day = torch.div(
            subset_start_ts, 86_400_000, rounding_mode="floor"
        )
        subsets.append(
            active & (day_ids.unsqueeze(0) >= subset_start_day.unsqueeze(1))
        )
        subset_start_steps.append(subset_start_step)
        subset_start_timestamps.append(subset_start_ts)
    return (
        eligible,
        subsets,
        subset_start_steps,
        subset_start_timestamps,
        day_ids,
    )


def _weighted_subsets(
    active,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
):
    eligible, subsets, _, _, _ = _weighted_subset_context(
        active,
        first_eq_ts,
        last_eq_ts,
        first_timestamp,
        interval_ms,
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

WEIGHTED_DAILY_SERIES_METRICS = {
    "equity_choppiness_w_usd",
    "equity_jerkiness_w_usd",
    "exponential_fit_error_w_usd",
    "volume_pct_per_day_avg_w",
}


def _weighted_daily_series_metrics(
    day_end_eq,
    day_volume,
    day_has_fill,
    active,
    fill_count,
    last_fill_ts,
    first_eq_ts,
    last_eq_ts,
    first_timestamp,
    interval_ms,
    requested,
):
    """Reduce Rust's weighted fill-day volume and daily equity-shape metrics."""

    requested = set(requested) & WEIGHTED_DAILY_SERIES_METRICS
    if not requested:
        return {}
    (
        _,
        subsets,
        subset_start_steps,
        subset_start_timestamps,
        day_ids,
    ) = _weighted_subset_context(
        active,
        first_eq_ts,
        last_eq_ts,
        first_timestamp,
        interval_ms,
    )
    fill_eligible = fill_count.to(torch.float64) > 1.0
    timestamp_origin = float(first_timestamp)
    finite_last_fill = torch.isfinite(last_fill_ts)
    relative_last_fill_ms = torch.where(
        last_fill_ts < timestamp_origin,
        last_fill_ts,
        last_fill_ts - timestamp_origin,
    )
    last_fill_steps = torch.floor(
        relative_last_fill_ms / float(interval_ms) + 0.5
    ).to(torch.long)
    # Unlike weighted equity-return metrics, Rust's weighted shape and volume
    # analysis admits a one-sample equity run when it has multiple fills. The
    # full-run analysis contributes one tenth, then the first empty suffix ends
    # the loop. Only finite, ordered timestamps and one active sample are
    # required here.
    eligible = (
        fill_eligible
        & torch.isfinite(first_eq_ts)
        & torch.isfinite(last_eq_ts)
        & (last_eq_ts >= first_eq_ts)
        & active.any(dim=1)
    )
    totals = {
        name: torch.zeros(
            day_end_eq.shape[0],
            dtype=day_end_eq.dtype,
            device=day_end_eq.device,
        )
        for name in requested
    }
    shape_names = requested & {
        "equity_choppiness_w_usd",
        "equity_jerkiness_w_usd",
        "exponential_fit_error_w_usd",
    }
    shape_sources = {
        "equity_choppiness_w_usd": "equity_choppiness_usd",
        "equity_jerkiness_w_usd": "equity_jerkiness_usd",
        "exponential_fit_error_w_usd": "exponential_fit_error_usd",
    }
    for subset_index, (subset, subset_start_step, subset_start_ts) in enumerate(
        zip(subsets, subset_start_steps, subset_start_timestamps)
    ):
        subset_eligible = (
            eligible & finite_last_fill & (last_fill_steps >= subset_start_step)
        )
        if "volume_pct_per_day_avg_w" in requested:
            if subset_index == 0:
                volume_fill_mask = day_has_fill & subset
            else:
                subset_start_day = torch.div(
                    subset_start_ts, 86_400_000, rounding_mode="floor"
                )
                at_day_boundary = subset_start_ts.remainder(86_400_000) == 0
                complete_day_mask = (
                    day_ids.unsqueeze(0) > subset_start_day.unsqueeze(1)
                )
                complete_day_mask |= at_day_boundary.unsqueeze(1) & (
                    day_ids.unsqueeze(0) == subset_start_day.unsqueeze(1)
                )
                # Daily volume cannot distinguish fills before and after an
                # intra-day cutoff. Exclude that ambiguous boundary day rather
                # than admitting pre-cutoff fills; exact Rust validation owns
                # the partial-day contribution.
                volume_fill_mask = day_has_fill & subset & complete_day_mask
            fill_days = volume_fill_mask.sum(dim=1)
            value = torch.where(
                fill_days > 0,
                torch.where(
                    volume_fill_mask,
                    day_volume,
                    torch.zeros_like(day_volume),
                ).sum(dim=1)
                / fill_days.clamp(min=1).to(day_volume.dtype),
                torch.zeros_like(totals["volume_pct_per_day_avg_w"]),
            )
            totals["volume_pct_per_day_avg_w"] += torch.where(
                subset_eligible & (fill_days > 0),
                value,
                torch.zeros_like(value),
            )
        if shape_names:
            values = _equity_shape_metrics(day_end_eq, subset)
            for name in shape_names:
                value = values[shape_sources[name]]
                totals[name] += torch.where(
                    subset_eligible, value, torch.zeros_like(value)
                )
    result = {name: value / 10.0 for name, value in totals.items()}
    # analyze_backtest returns early for zero or one fill. Preserve the custom
    # Analysis defaults for weighted shape metrics; weighted volume defaults
    # to zero.
    for name in shape_names:
        result[name] = torch.where(
            fill_eligible, result[name], torch.ones_like(result[name])
        )
    return result


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


def _analysis_duration_days(out: dict, run) -> torch.Tensor:
    """Recover Rust's timestamp-span denominator from Metal candle indices."""

    first_eq_ts = out["first_eq_ts"].to(torch.float64)
    last_eq_ts = out["last_eq_ts"].to(torch.float64)
    has_span = (
        torch.isfinite(first_eq_ts)
        & torch.isfinite(last_eq_ts)
        & (last_eq_ts > first_eq_ts)
    )
    interval_ms = max(float(run.interval_ms), 1.0)
    # Metal exports integer candle indices multiplied by interval_ms through a
    # float32 scalar buffer. Recover the indices before subtracting so a span
    # on a whole-day boundary cannot round slightly upward and add a spurious
    # active-day denominator bucket.
    first_eq_step = torch.round(first_eq_ts / interval_ms)
    last_eq_step = torch.round(last_eq_ts / interval_ms)
    return torch.where(
        has_span,
        (last_eq_step - first_eq_step).clamp(min=0.0) * interval_ms
        / 86_400_000.0,
        torch.zeros_like(first_eq_ts),
    )


def _fill_activity_metrics(out: dict, run, requested: set[str]) -> dict:
    """Match Rust's full-run fill count and timestamp-span rate contract."""

    fill_count = out["fill_count"].to(torch.float64)
    fills_count_entry = out["fill_count_entry"].to(torch.float64)
    fills_count_long = out["fill_count_long"].to(torch.float64)
    fills_count_close = (fill_count - fills_count_entry).clamp(min=0.0)
    fills_count_short = (fill_count - fills_count_long).clamp(min=0.0)
    fills_active_days_count = out["fills_active_days_count"].to(torch.float64)
    coin_fill_counts = out.get("coin_fill_counts")
    if coin_fill_counts is None:
        coin_fill_counts = fill_count.unsqueeze(1)
    else:
        coin_fill_counts = coin_fill_counts.to(torch.float64)
    fills_active_symbols_count = (coin_fill_counts > 0.0).sum(dim=1).to(torch.float64)
    fills_top_symbol_share = torch.where(
        fill_count > 0.0,
        coin_fill_counts.max(dim=1).values / fill_count.clamp(min=1.0),
        torch.zeros_like(fill_count),
    )
    duration_days = _analysis_duration_days(out, run)
    fills_per_day = torch.where(
        duration_days > 0.0,
        fill_count / duration_days.clamp(min=1.0e-9),
        torch.zeros_like(fill_count),
    )
    def per_day(count):
        return torch.where(
            duration_days > 0.0,
            count / duration_days.clamp(min=1.0e-9),
            torch.zeros_like(count),
        )

    fills_per_day_entry = per_day(fills_count_entry)
    fills_per_day_close = per_day(fills_count_close)
    fills_per_day_long = per_day(fills_count_long)
    fills_per_day_short = per_day(fills_count_short)
    metrics = {
        "fills_active_days_count": fills_active_days_count,
        "fills_active_days_ratio": fills_active_days_count
        / duration_days.ceil().clamp(min=1.0),
        "fills_active_symbols_count": fills_active_symbols_count,
        "fills_analysis_duration_days": duration_days,
        "fills_count": fill_count,
        "fills_count_close": fills_count_close,
        "fills_count_entry": fills_count_entry,
        "fills_count_long": fills_count_long,
        "fills_count_short": fills_count_short,
        "fills_entry_per_close": fills_count_entry
        / fills_count_close.clamp(min=1.0),
        "fills_per_day": fills_per_day,
        "fills_per_day_close": fills_per_day_close,
        "fills_per_day_entry": fills_per_day_entry,
        "fills_per_day_long": fills_per_day_long,
        "fills_per_day_short": fills_per_day_short,
        "fills_top_symbol_share": fills_top_symbol_share,
    }
    slot_metrics = {
        "fills_per_day_per_position_slot",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
    }
    if requested & slot_metrics:
        slots_long = out["position_slots_long"].to(torch.float64)
        slots_short = out["position_slots_short"].to(torch.float64)
        long_slot_rate = torch.where(
            slots_long > 0.0,
            fills_per_day_long / slots_long.clamp(min=1.0),
            torch.zeros_like(fills_per_day_long),
        )
        short_slot_rate = torch.where(
            slots_short > 0.0,
            fills_per_day_short / slots_short.clamp(min=1.0),
            torch.zeros_like(fills_per_day_short),
        )
        active_slot_sides = (slots_long > 0.0).to(torch.float64) + (
            slots_short > 0.0
        ).to(torch.float64)
        combined_slot_rate = torch.where(
            active_slot_sides > 0.0,
            (long_slot_rate + short_slot_rate) / active_slot_sides.clamp(min=1.0),
            torch.zeros_like(long_slot_rate),
        )
        metrics.update(
            {
                "fills_per_day_per_position_slot": combined_slot_rate,
                "fills_per_day_per_position_slot_long": long_slot_rate,
                "fills_per_day_per_position_slot_short": short_slot_rate,
            }
        )
    return metrics


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


def _hard_stop_ema_drawdown_metrics(out: dict) -> dict:
    """Reduce per-side HSL EMA maxima using Rust's public metric contract."""

    required = {
        "hsl_drawdown_ema_max_long",
        "hsl_drawdown_ema_max_short",
    }
    if missing := required.difference(out):
        raise RuntimeError(
            "MPS directional HSL drawdown-EMA outputs are missing from proxy results: "
            + ", ".join(sorted(missing))
        )
    long_ema_max = out["hsl_drawdown_ema_max_long"].to(torch.float64)
    short_ema_max = out["hsl_drawdown_ema_max_short"].to(torch.float64)
    return {
        "drawdown_worst_ema_strategy_eq": long_ema_max.maximum(short_ema_max),
        "drawdown_worst_ema_strategy_eq_long": long_ema_max,
        "drawdown_worst_ema_strategy_eq_short": short_ema_max,
    }


def _hard_stop_raw_drawdown_metrics(out: dict) -> dict:
    """Expose per-side raw HSL strategy-equity drawdown summaries."""

    required = {
        "hsl_drawdown_raw_max_long",
        "hsl_drawdown_raw_max_short",
        "hsl_drawdown_raw_mean_worst_1pct_long",
        "hsl_drawdown_raw_mean_worst_1pct_short",
    }
    if missing := required.difference(out):
        raise RuntimeError(
            "MPS directional HSL raw-drawdown outputs are missing from proxy "
            "results: " + ", ".join(sorted(missing))
        )
    return {
        "drawdown_worst_strategy_eq_long": out[
            "hsl_drawdown_raw_max_long"
        ].to(torch.float64),
        "drawdown_worst_strategy_eq_short": out[
            "hsl_drawdown_raw_max_short"
        ].to(torch.float64),
        "drawdown_worst_mean_1pct_strategy_eq_long": out[
            "hsl_drawdown_raw_mean_worst_1pct_long"
        ].to(torch.float64),
        "drawdown_worst_mean_1pct_strategy_eq_short": out[
            "hsl_drawdown_raw_mean_worst_1pct_short"
        ].to(torch.float64),
    }


def _hard_stop_ema_tail_metrics(out: dict) -> dict:
    """Reduce bounded per-side HSL EMA tails using Rust's public contract."""

    required = {
        "hsl_drawdown_ema_mean_worst_1pct_long",
        "hsl_drawdown_ema_mean_worst_1pct_short",
    }
    if missing := required.difference(out):
        raise RuntimeError(
            "MPS directional HSL drawdown-EMA tail outputs are missing from "
            "proxy results: " + ", ".join(sorted(missing))
        )
    long_tail = out["hsl_drawdown_ema_mean_worst_1pct_long"].to(torch.float64)
    short_tail = out["hsl_drawdown_ema_mean_worst_1pct_short"].to(torch.float64)
    return {
        "drawdown_worst_mean_1pct_ema_strategy_eq": long_tail.maximum(short_tail),
        "drawdown_worst_mean_1pct_ema_strategy_eq_long": long_tail,
        "drawdown_worst_mean_1pct_ema_strategy_eq_short": short_tail,
    }


def _hard_stop_strategy_eq_recovery_metrics(out: dict) -> dict:
    """Expose per-side HSL strategy-equity maximum recovery duration."""

    required = {
        "hsl_strategy_eq_recovery_max_ms_long",
        "hsl_strategy_eq_recovery_max_ms_short",
    }
    if missing := required.difference(out):
        raise RuntimeError(
            "MPS directional HSL strategy-equity recovery outputs are missing from proxy results: "
            + ", ".join(sorted(missing))
        )
    long_recovery_ms = out["hsl_strategy_eq_recovery_max_ms_long"].to(
        torch.float64
    )
    short_recovery_ms = out["hsl_strategy_eq_recovery_max_ms_short"].to(
        torch.float64
    )
    return {
        "peak_recovery_hours_strategy_eq_long": long_recovery_ms / 3_600_000.0,
        "peak_recovery_hours_strategy_eq_short": short_recovery_ms / 3_600_000.0,
        "peak_recovery_days_strategy_eq_long": long_recovery_ms / 86_400_000.0,
        "peak_recovery_days_strategy_eq_short": short_recovery_ms / 86_400_000.0,
    }


def _strategy_eq_recovery_distribution_metrics(out: dict) -> dict:
    """Map strict uniformly sampled time-to-exceed summaries from Apple MPS."""

    if "strategy_eq_recovery_distribution" not in out:
        raise RuntimeError(
            "MPS strategy-equity recovery-distribution output is missing from proxy results"
        )
    values = out["strategy_eq_recovery_distribution"].to(torch.float64)
    names = (
        "strategy_eq_recovery_days_mean",
        "strategy_eq_recovery_days_median",
        "strategy_eq_recovery_days_p95",
        "strategy_eq_recovery_days_p99",
        "strategy_eq_recovery_days_mean_worst_5pct",
        "strategy_eq_recovery_days_mean_worst_1pct",
    )
    return {name: values[:, index] for index, name in enumerate(names)}


def _daily_peak_recovery_ms(day_end_eq, active):
    """Approximate intraday recovery from the compact UTC daily surface."""

    batch_size, day_count = day_end_eq.shape
    peak = torch.full(
        (batch_size,),
        float("-inf"),
        dtype=day_end_eq.dtype,
        device=day_end_eq.device,
    )
    peak_day = torch.zeros(
        batch_size, dtype=torch.long, device=day_end_eq.device
    )
    recovery_days = torch.zeros_like(peak)
    started = torch.zeros(
        batch_size, dtype=torch.bool, device=day_end_eq.device
    )
    for day in range(day_count):
        valid = active[:, day]
        value = day_end_eq[:, day]
        new_high = valid & (~started | (value >= peak))
        recovered = (day - peak_day).to(day_end_eq.dtype)
        recovery_days = torch.where(
            new_high & started,
            torch.maximum(recovery_days, recovered),
            recovery_days,
        )
        peak = torch.where(new_high, value, peak)
        peak_day = torch.where(
            new_high, torch.full_like(peak_day, day), peak_day
        )
        started |= valid
    return torch.where(
        started,
        recovery_days * 86_400_000.0,
        torch.zeros_like(recovery_days),
    )


def _equity_balance_diff_values(out: dict, *, suffix: str = "") -> dict:
    base_names = (
        "equity_balance_diff_pos_max",
        "equity_balance_diff_pos_mean",
        "equity_balance_diff_neg_max",
        "equity_balance_diff_neg_mean",
    )
    names = tuple(f"{name}{suffix}" for name in base_names)
    missing = [name for name in names if name not in out]
    if missing:
        raise RuntimeError(
            "MPS equity-balance-diff output is missing " + ", ".join(missing)
        )
    values = {
        base_name: out[name].to(torch.float64)
        for base_name, name in zip(base_names, names)
    }
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError("MPS equity-balance-diff output is non-finite")
    return values


def _btc_account_metrics(out: dict, run, data: dict, requested) -> dict:
    """Reduce the compact USD strategy-equity surface into BTC account metrics.

    Close-only metrics convert the compact USD surface at prepared BTC day-end
    prices. Intraday-risk metrics consume the synchronized BTC surface emitted
    by the opt-in Metal feature. Exact Rust validation remains authoritative.
    """

    requested = set(requested) & BTC_ACCOUNT_METRICS
    if not requested:
        return {}
    day_end_usd = out["day_end_eq"].to(torch.float64)
    active = _daily_series_masks(out["day_min_eq"])
    risk_requested = bool(requested & BTC_INTRADAY_RISK_METRICS)
    if risk_requested:
        missing = [
            key
            for key in ("btc_day_end_eq", "btc_day_min_eq", "btc_day_max_dd")
            if key not in out
        ]
        if missing:
            raise RuntimeError(
                "MPS synchronized BTC-risk output is missing "
                + ", ".join(missing)
            )
        risk_day_end_btc = out["btc_day_end_eq"].to(torch.float64)
        risk_day_min_btc = out["btc_day_min_eq"].to(torch.float64)
        risk_day_max_dd_btc = out["btc_day_max_dd"].to(torch.float64)

    missing = [
        key for key in ("btc_day_end_price", "btc_prices") if key not in data
    ]
    if missing:
        raise RuntimeError(
            "MPS BTC account metric context is missing " + ", ".join(missing)
        )
    day_end_price = torch.as_tensor(
        data["btc_day_end_price"],
        dtype=torch.float64,
        device=day_end_usd.device,
    ).reshape(1, -1).expand(day_end_usd.shape[0], -1)
    if day_end_price.shape[1] != day_end_usd.shape[1]:
        raise RuntimeError(
            "MPS BTC account metric day grid disagrees with Metal output"
        )
    # A liquidated candidate may stop before the prepared UTC day ends.
    btc_prices = torch.as_tensor(
        data["btc_prices"],
        dtype=torch.float64,
        device=day_end_usd.device,
    ).reshape(-1)
    origin_ms = int(data["ts0"])
    interval_ms = int(run.interval_ms)
    last_eq_ts = out["last_eq_ts"].to(torch.float64)
    relative_last_ms = torch.where(
        last_eq_ts < float(origin_ms),
        last_eq_ts,
        last_eq_ts - float(origin_ms),
    )
    last_step = torch.floor(
        relative_last_ms / float(interval_ms) + 0.5
    ).to(torch.long)
    safe_last_step = last_step.clamp(min=0, max=max(len(btc_prices) - 1, 0))
    last_day = (
        torch.div(
            safe_last_step * interval_ms + origin_ms,
            86_400_000,
            rounding_mode="floor",
        )
        - origin_ms // 86_400_000
    )
    safe_last_day = last_day.clamp(
        min=0, max=max(day_end_usd.shape[1] - 1, 0)
    )
    valid_endpoint = (
        active.any(dim=1)
        & torch.isfinite(last_eq_ts)
        & (last_step >= 0)
        & (last_step < len(btc_prices))
        & (last_day >= 0)
        & (last_day < day_end_usd.shape[1])
    )
    final_day_mask = (
        torch.arange(day_end_usd.shape[1], device=day_end_usd.device)
        .unsqueeze(0)
        .eq(safe_last_day.unsqueeze(1))
        & valid_endpoint.unsqueeze(1)
    )
    endpoint_price = btc_prices.gather(0, safe_last_step).unsqueeze(1)
    day_end_price = torch.where(final_day_mask, endpoint_price, day_end_price)
    day_end_btc = torch.where(
        active, day_end_usd / day_end_price, torch.zeros_like(day_end_usd)
    )
    gain, adg = _smoothed_gain_adg(day_end_btc, active)
    daily_changes, change_mask = _pct_change(day_end_btc, active)
    mdg = _masked_median(daily_changes, change_mask)
    omega = _omega_ratio(daily_changes, change_mask)
    has_fill = out["fill_count"].to(torch.float64) > 0.0
    zeros = torch.zeros_like(adg)

    values = {
        "adg_btc": adg,
        "gain_btc": gain,
        "mdg_btc": mdg,
        "omega_ratio_btc": omega,
    }
    btc_equity_balance_metrics = {
        metric
        for metric in EQUITY_BALANCE_DIFF_METRICS
        if metric.endswith("_btc")
    }
    if requested & btc_equity_balance_metrics:
        differences = _equity_balance_diff_values(out, suffix="_btc")
        for name, value in differences.items():
            values[f"{name}_btc"] = value
        values["paper_loss_ratio_btc"] = adg / differences[
            "equity_balance_diff_neg_max"
        ].clamp(min=1e-12)
        values["paper_loss_mean_ratio_btc"] = adg / differences[
            "equity_balance_diff_neg_mean"
        ].clamp(min=1e-12)
    if risk_requested:
        _risk_gain, risk_adg = _smoothed_gain_adg(risk_day_end_btc, active)
        min_changes, min_change_mask = _pct_change(risk_day_min_btc, active)
        sharpe, sortino = _sharpe_sortino(
            min_changes, min_change_mask, risk_adg
        )
        expected_shortfall = _mean_worst_one_pct_abs(
            min_changes, min_change_mask
        )
        max_dd = risk_day_max_dd_btc.max(dim=1).values
        worst_one_pct = _mean_worst_one_pct_largest(
            risk_day_max_dd_btc, active
        )
        values.update(
            {
                "calmar_ratio_btc": risk_adg / max_dd.clamp(min=1e-12),
                "drawdown_worst_btc": max_dd,
                "drawdown_worst_mean_1pct_btc": worst_one_pct,
                "expected_shortfall_1pct_btc": expected_shortfall,
                "sharpe_ratio_btc": sharpe,
                "sortino_ratio_btc": sortino,
                "sterling_ratio_btc": risk_adg
                / worst_one_pct.clamp(min=1e-12),
            }
        )
    drawdown_defaults = {
        "drawdown_worst_btc",
        "drawdown_worst_mean_1pct_btc",
    }
    equity_balance_diff_defaults = {
        f"equity_balance_diff_{sign}_{stat}_btc"
        for sign in ("neg", "pos")
        for stat in ("max", "mean")
    }
    for name in tuple(values):
        default = (
            torch.ones_like(adg)
            if name in drawdown_defaults | equity_balance_diff_defaults
            else zeros
        )
        values[name] = torch.where(has_fill, values[name], default)

    shape_names = {
        "equity_choppiness_btc",
        "equity_jerkiness_btc",
        "exponential_fit_error_btc",
    }
    if requested & shape_names:
        shape = _equity_shape_metrics(day_end_btc, active)
        shape_sources = {
            "equity_choppiness_btc": "equity_choppiness_usd",
            "equity_jerkiness_btc": "equity_jerkiness_usd",
            "exponential_fit_error_btc": "exponential_fit_error_usd",
        }
        for name, source in shape_sources.items():
            values[name] = torch.where(
                has_fill, shape[source], torch.ones_like(adg)
            )

    safe_weighted_sources = {
        "adg_w_btc": "adg_strategy_eq_w",
        "mdg_w_btc": "mdg_strategy_eq_w",
        "omega_ratio_w_btc": "omega_ratio_strategy_eq_w",
    }
    wanted_safe_weighted_sources = {
        source
        for name, source in safe_weighted_sources.items()
        if name in requested
    }
    if requested & {
        "adg_w_per_exposure_long_btc",
        "adg_w_per_exposure_short_btc",
    }:
        wanted_safe_weighted_sources.add("adg_strategy_eq_w")
    if requested & {
        "mdg_w_per_exposure_long_btc",
        "mdg_w_per_exposure_short_btc",
    }:
        wanted_safe_weighted_sources.add("mdg_strategy_eq_w")
    enough_fills = out["fill_count"].to(torch.float64) > 1.0
    if wanted_safe_weighted_sources:
        safe_weighted = _weighted_strategy_eq_metrics(
            day_end_btc,
            day_end_btc,
            torch.zeros_like(day_end_btc),
            active,
            out["first_eq_ts"],
            out["last_eq_ts"],
            data["ts0"],
            run.interval_ms,
            wanted_safe_weighted_sources,
        )
        for name, source in safe_weighted_sources.items():
            if source in safe_weighted:
                values[name] = torch.where(
                    enough_fills, safe_weighted[source], zeros
                )
    weighted_shape_sources = {
        "equity_choppiness_w_btc": "equity_choppiness_w_usd",
        "equity_jerkiness_w_btc": "equity_jerkiness_w_usd",
        "exponential_fit_error_w_btc": "exponential_fit_error_w_usd",
    }
    wanted_weighted_shape_sources = {
        source
        for name, source in weighted_shape_sources.items()
        if name in requested
    }
    if wanted_weighted_shape_sources:
        weighted_shape = _weighted_daily_series_metrics(
            day_end_btc,
            out["day_volume"].to(torch.float64),
            out["day_has_fill"],
            active,
            out["fill_count"],
            out["last_fill_ts"],
            out["first_eq_ts"],
            out["last_eq_ts"],
            data["ts0"],
            run.interval_ms,
            wanted_weighted_shape_sources,
        )
        for name, source in weighted_shape_sources.items():
            if source in weighted_shape:
                values[name] = weighted_shape[source]

    recovery_names = {
        "peak_recovery_hours_equity_btc",
        "peak_recovery_days_equity_btc",
    }
    if requested & recovery_names:
        recovery_ms = _daily_peak_recovery_ms(day_end_btc, active)
        values["peak_recovery_hours_equity_btc"] = torch.where(
            has_fill, recovery_ms / 3_600_000.0, zeros
        )
        values["peak_recovery_days_equity_btc"] = torch.where(
            has_fill, recovery_ms / 86_400_000.0, zeros
        )
    exposure_denominators = {
        "exposure_ratio_btc": "total_wallet_exposure_max",
        "exposure_mean_ratio_btc": "total_wallet_exposure_mean",
    }
    for name, denominator in exposure_denominators.items():
        if name in requested:
            ratio = adg / out[denominator].to(torch.float64).abs().clamp(
                min=1e-12
            )
            values[name] = torch.where(has_fill, ratio, zeros)

    per_exposure_sources = {
        "adg": adg,
        "adg_w": values.get("adg_w_btc", zeros),
        "gain": gain,
        "mdg": mdg,
        "mdg_w": values.get("mdg_w_btc", zeros),
    }
    for metric, source in per_exposure_sources.items():
        for side in ("long", "short"):
            name = f"{metric}_per_exposure_{side}_btc"
            if name not in requested:
                continue
            denominator = out[
                f"candidate_total_wallet_exposure_limit_{side}"
            ].to(torch.float64)
            values[name] = torch.where(
                has_fill & (denominator > 0.0),
                source / denominator.clamp(min=1e-12),
                zeros,
            )
    return {name: values[name] for name in requested if name in values}


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
    weighted_daily_series_metrics = {}
    if requested & WEIGHTED_DAILY_SERIES_METRICS:
        weighted_daily_series_metrics = _weighted_daily_series_metrics(
            day_end_eq,
            day_volume,
            day_has_fill,
            active,
            out["fill_count"],
            out["last_fill_ts"],
            out["first_eq_ts"],
            out["last_eq_ts"],
            data["ts0"],
            run.interval_ms,
            requested,
        )
    shape_metric_names = {
        "equity_choppiness_usd",
        "equity_jerkiness_usd",
        "exponential_fit_error_usd",
    }
    equity_shape_metrics = {}
    if requested & shape_metric_names:
        equity_shape_metrics = _equity_shape_metrics(day_end_eq, active)
        has_fills = out["fill_count"].to(torch.float64) > 0.0
        for name, value in equity_shape_metrics.items():
            equity_shape_metrics[name] = torch.where(
                has_fills, value, torch.ones_like(value)
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
    account_recovery_metrics = {
        "peak_recovery_days_equity_usd",
        "peak_recovery_hours_equity_usd",
    }
    if requested & account_recovery_metrics:
        if "account_recovery_max_ms" not in out:
            raise RuntimeError(
                "MPS account-equity recovery output is missing from proxy results"
            )
        account_recovery_max_ms = torch.where(
            out["fill_count"].to(torch.float64) > 0.0,
            out["account_recovery_max_ms"].to(torch.float64),
            torch.zeros_like(recovery_max_days),
        )
    else:
        account_recovery_max_ms = torch.zeros_like(recovery_max_days)
    pnl_recovery_metrics = {
        "peak_recovery_days_pnl",
        "peak_recovery_hours_pnl",
    }
    if requested & pnl_recovery_metrics:
        if "pnl_recovery_max_ms" not in out:
            raise RuntimeError(
                "MPS realized-PnL recovery output is missing from proxy results"
            )
        pnl_recovery_max_ms = out["pnl_recovery_max_ms"].to(torch.float64)
    else:
        pnl_recovery_max_ms = torch.zeros_like(recovery_max_days)
    held_days = out["held_max_ms"] / 86_400_000.0
    position_duration_metrics = {
        "position_held_days_mean",
        "position_held_hours_mean",
        "positions_held_per_day",
    }
    held_hours_mean = positions_held_per_day = zeros
    if requested & position_duration_metrics:
        held_count = out["held_count"].to(torch.float64)
        held_hours_mean = torch.where(
            held_count > 0.0,
            out["held_sum_ms"].to(torch.float64)
            / held_count.clamp(min=1.0)
            / 3_600_000.0,
            torch.zeros_like(held_count),
        )
        duration_days = _analysis_duration_days(out, run)
        positions_held_per_day = torch.where(
            duration_days > 0.0,
            held_count / duration_days.clamp(min=1.0e-9),
            torch.zeros_like(held_count),
        )

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
    fill_activity_metrics = (
        _fill_activity_metrics(out, run, requested)
        if requested & _FILL_ACTIVITY_METRICS
        else {}
    )
    entry_interval_metrics = (
        _entry_interval_metrics(
            out,
            run,
            str(data.get("strategy_kind", "")).strip().lower(),
        )
        if requested & ENTRY_INTERVAL_METRICS
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
    hard_stop_ema_drawdown_metrics = (
        _hard_stop_ema_drawdown_metrics(out)
        if requested & _HARD_STOP_EMA_DRAWDOWN_METRICS
        else {}
    )
    hard_stop_raw_drawdown_metrics = (
        _hard_stop_raw_drawdown_metrics(out)
        if requested & _HARD_STOP_RAW_DRAWDOWN_METRICS
        else {}
    )
    hard_stop_ema_tail_metrics = (
        _hard_stop_ema_tail_metrics(out)
        if requested & _HARD_STOP_EMA_TAIL_METRICS
        else {}
    )
    hard_stop_strategy_eq_recovery_metrics = (
        _hard_stop_strategy_eq_recovery_metrics(out)
        if requested & _HARD_STOP_STRATEGY_EQ_RECOVERY_METRICS
        else {}
    )
    strategy_eq_recovery_distribution_metrics = (
        _strategy_eq_recovery_distribution_metrics(out)
        if requested & _STRATEGY_EQ_RECOVERY_DISTRIBUTION_METRICS
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
        "position_held_days_mean": held_hours_mean / 24.0,
        "position_held_days_max": held_days,
        "position_held_hours_mean": held_hours_mean,
        "position_held_hours_max": held_days * 24.0,
        "positions_held_per_day": positions_held_per_day,
        "sharpe_ratio_strategy_eq": sharpe,
        "sharpe_ratio_pnl": sharpe_pnl,
        "sortino_ratio_strategy_eq": sortino,
        "sortino_ratio_pnl": sortino_pnl,
        "sterling_ratio_strategy_eq": sterling,
        "strategy_eq_recovery_days_max": recovery_max_days,
        "peak_recovery_hours_strategy_eq": recovery_max_days * 24.0,
        "peak_recovery_days_strategy_eq": recovery_max_days,
        "peak_recovery_hours_equity_usd": account_recovery_max_ms / 3_600_000.0,
        "peak_recovery_days_equity_usd": account_recovery_max_ms / 86_400_000.0,
        "peak_recovery_hours_pnl": pnl_recovery_max_ms / 3_600_000.0,
        "peak_recovery_days_pnl": pnl_recovery_max_ms / 86_400_000.0,
        "strategy_eq_underwater_pct_mean": underwater,
        "strategy_eq_underwater_pct_median": underwater_median,
        "volume_pct_per_day_avg": volume_pct,
    }
    usd_equity_balance_metrics = {
        metric
        for metric in EQUITY_BALANCE_DIFF_METRICS
        if metric.endswith("_usd")
    }
    if requested & usd_equity_balance_metrics:
        differences = _equity_balance_diff_values(out)
        has_fills = out["fill_count"].to(torch.float64) > 0.0
        for name, value in differences.items():
            objectives[f"{name}_usd"] = torch.where(
                has_fills, value, torch.ones_like(value)
            )
        objectives["paper_loss_ratio_usd"] = torch.where(
            has_fills,
            adg
            / differences["equity_balance_diff_neg_max"].clamp(min=1e-12),
            torch.zeros_like(adg),
        )
        objectives["paper_loss_mean_ratio_usd"] = torch.where(
            has_fills,
            adg
            / differences["equity_balance_diff_neg_mean"].clamp(min=1e-12),
            torch.zeros_like(adg),
        )
    objectives.update(hard_stop_ema_drawdown_metrics)
    objectives.update(hard_stop_raw_drawdown_metrics)
    objectives.update(hard_stop_ema_tail_metrics)
    objectives.update(hard_stop_strategy_eq_recovery_metrics)
    objectives.update(strategy_eq_recovery_distribution_metrics)
    if "loss_profit_ratio" in requested:
        objectives["loss_profit_ratio"] = _loss_profit_ratio(
            out["loss_sum"], out["profit_sum"]
        )
    directional_pnl_metrics = {
        "loss_profit_ratio_long",
        "loss_profit_ratio_short",
        "long_short_profit_ratio",
        "pnl_ratio_long_short",
    }
    if requested & directional_pnl_metrics:
        objectives.update(
            {
                name: value
                for name, value in _directional_pnl_metrics(out).items()
                if name in requested
            }
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
    objectives.update(fill_activity_metrics)
    objectives.update(entry_interval_metrics)
    objectives.update(hard_stop_metrics)
    objectives.update(hard_stop_panic_loss_metrics)
    objectives.update(weighted_metrics)
    objectives.update(weighted_daily_series_metrics)
    objectives.update(weighted_pnl_metrics)
    objectives.update(equity_shape_metrics)
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
    objectives.update(_btc_account_metrics(out, run, data, requested))
    if needed is None:
        return objectives
    return {key: value for key, value in objectives.items() if key in requested}
