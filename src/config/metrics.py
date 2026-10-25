from collections.abc import Mapping
import re
from typing import Any

CURRENCY_METRICS = {
    "adg_fraction_ew",
    "sharpe_ratio_ew",
    "sortino_ratio_ew",
    "omega_ratio_ew",
    "adg_fraction",
    "adg_fraction_per_twel_ratio_long",
    "adg_fraction_per_twel_ratio_short",
    "calmar_ratio",
    "drawdown_max_fraction",
    "drawdown_daily_worst_1pct_mean_fraction",
    "equity_balance_gap_negative_max_fraction",
    "equity_balance_gap_negative_mean_fraction",
    "equity_balance_gap_positive_max_fraction",
    "equity_balance_gap_positive_mean_fraction",
    "equity_choppiness_ratio",
    "adg_to_absolute_net_exposure_mean_ratio",
    "adg_to_absolute_net_exposure_max_ratio",
    "equity_jerkiness_ratio",
    "peak_recovery_max_days",
    "peak_recovery_max_hours",
    "expected_shortfall_1pct_fraction",
    "exponential_fit_mse_log_ratio_squared",
    "gain_multiplier",
    "gain_multiplier_per_twel_ratio_long",
    "gain_multiplier_per_twel_ratio_short",
    "mdg_fraction",
    "mdg_fraction_per_twel_ratio_long",
    "mdg_fraction_per_twel_ratio_short",
    "omega_ratio",
    "adg_to_paper_loss_mean_ratio",
    "adg_to_paper_loss_max_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "sterling_ratio",
}

EXPOSURE_RMS_METRICS = frozenset({
    "wallet_exposure_rms_ratio_long",
    "wallet_exposure_rms_ratio_short",
    "gross_wallet_exposure_rms_ratio",
    "wallet_exposure_utilization_rms_ratio_long",
    "wallet_exposure_utilization_rms_ratio_short",
    "position_exposure_utilization_rms_ratio_long",
    "position_exposure_utilization_rms_ratio_short",
})

SHARED_METRICS = {
    *EXPOSURE_RMS_METRICS,
    "adg_fraction_strategy_equity_ew",
    "sharpe_ratio_strategy_equity_ew",
    "sortino_ratio_strategy_equity_ew",
    "omega_ratio_strategy_equity_ew",
    "backtest_completion_fraction",
    "position_episodes_per_day",
    "position_held_time_weighted_mean_hours",
    "position_held_mean_hours",
    "position_held_max_hours",
    "position_held_median_hours",
    "position_unchanged_max_hours",
    "position_held_mean_days",
    "position_held_max_days",
    "position_held_median_days",
    "position_unchanged_max_days",
    "trade_win_fraction",
    "trade_loss_max_fraction",
    "trade_loss_mean_fraction",
    "trade_loss_median_fraction",
    "fills_active_days_count",
    "fills_active_day_fraction",
    "fills_active_symbols_count",
    "analysis_duration_days",
    "fills_count",
    "fills_close_count",
    "fills_entry_count",
    "fills_count_long",
    "fills_count_short",
    "entry_to_close_fill_count_ratio",
    "fills_gap_max_days",
    "fills_gap_mean_hours",
    "fills_gap_median_hours",
    "fills_gap_p95_hours",
    "fills_gap_p99_hours",
    "fills_gap_time_weighted_mean_hours",
    "fills_per_day",
    "fills_close_per_day",
    "fills_entry_per_day",
    "fills_per_day_long",
    "fills_per_day_per_position_slot",
    "fills_per_day_per_position_slot_long",
    "fills_per_day_per_position_slot_short",
    "fills_per_day_short",
    "fills_top_symbol_fraction",
    "active_day_turnover_proxy_mean_ratio",
    "loss_profit_ratio",
    "peak_recovery_max_days_realized_pnl",
    "peak_recovery_max_hours_realized_pnl",
    "high_exposure_mean_days_long",
    "high_exposure_max_days_long",
    "high_exposure_mean_days_short",
    "high_exposure_max_days_short",
    "high_exposure_mean_hours_long",
    "high_exposure_max_hours_long",
    "high_exposure_mean_hours_short",
    "high_exposure_max_hours_short",
    "daily_realized_pnl_mean_fraction",
    "daily_realized_pnl_median_fraction",
    "sharpe_ratio_realized_pnl",
    "sortino_ratio_realized_pnl",
    "gain_multiplier_strategy_equity",
    "adg_fraction_strategy_equity",
    "adg_rolling_hmean_fraction_strategy_equity",
    "adg_time_integrated_fraction_strategy_equity",
    "positive_log_return_participation_score_strategy_equity",
    "mdg_fraction_strategy_equity",
    "sharpe_ratio_strategy_equity",
    "sortino_ratio_strategy_equity",
    "omega_ratio_strategy_equity",
    "expected_shortfall_1pct_fraction_strategy_equity",
    "calmar_ratio_strategy_equity",
    "sterling_ratio_strategy_equity",
    "drawdown_max_fraction_strategy_equity",
    "hsl_signal_ema_max_fraction",
    "drawdown_daily_worst_1pct_mean_fraction_strategy_equity",
    "hsl_signal_ema_worst_1pct_mean_fraction",
    "pain_index_fraction_strategy_equity",
    "ulcer_index_fraction_strategy_equity",
    "recovery_mean_days_strategy_equity",
    "recovery_median_days_strategy_equity",
    "recovery_p95_days_strategy_equity",
    "recovery_p99_days_strategy_equity",
    "recovery_worst_5pct_mean_days_strategy_equity",
    "recovery_worst_1pct_mean_days_strategy_equity",
    "recovery_max_days_strategy_equity",
    "recovery_max_days_strategy_equity",
    "recovery_max_hours_strategy_equity",
    "hsl_triggers_per_year",
    "hsl_restarts_per_year",
    "hsl_restarts_per_year_long",
    "hsl_restarts_per_year_short",
    "hsl_panic_loss_starting_balance_fraction",
    "hsl_red_time_fraction",
    "hsl_red_duration_mean_minutes",
    "hsl_red_duration_max_minutes",
    "hsl_trigger_signal_mean_fraction",
    "hsl_panic_fill_loss_sum_usd",
    "hsl_panic_fill_loss_max_usd",
    "hsl_panic_episode_loss_min_fraction",
    "hsl_panic_episode_loss_mean_fraction",
    "hsl_panic_episode_loss_max_fraction",
    "hsl_flatten_time_mean_minutes",
    "hsl_restart_retrigger_fraction",
}


ANALYSIS_SHARED_KEYS = SHARED_METRICS | {
    "loss_profit_ratio_long",
    "loss_profit_ratio_short",
    "long_pnl_to_total_pnl_ratio",
    "long_pnl_to_total_pnl_ratio",
    "absolute_net_wallet_exposure_max_ratio",
    "absolute_net_wallet_exposure_mean_ratio",
    "absolute_net_wallet_exposure_median_ratio",
    "initial_entry_balance_fraction_long",
    "initial_entry_balance_fraction_short",
    "entry_interval_mean_hours",
    "entry_interval_median_hours",
    "entry_interval_p95_hours",
    "entry_interval_p99_hours",
    "entry_interval_max_hours",
    "drawdown_max_fraction_strategy_equity_long",
    "drawdown_max_fraction_strategy_equity_short",
    "hsl_signal_ema_max_fraction_long",
    "hsl_signal_ema_max_fraction_short",
    "drawdown_daily_worst_1pct_mean_fraction_strategy_equity_long",
    "drawdown_daily_worst_1pct_mean_fraction_strategy_equity_short",
    "hsl_signal_ema_worst_1pct_mean_fraction_long",
    "hsl_signal_ema_worst_1pct_mean_fraction_short",
    "recovery_max_days_strategy_equity_long",
    "recovery_max_days_strategy_equity_short",
    "recovery_max_hours_strategy_equity_long",
    "recovery_max_hours_strategy_equity_short",
    "hsl_trigger_count_long",
    "hsl_trigger_count_short",
    "hsl_restart_count_long",
    "hsl_restart_count_short",
}

STAT_SUFFIXES = ("min", "max", "mean", "std", "median")


def metric_with_curve(metric: str, curve: str) -> str:
    """Insert the curve before position side and the final EW modifier."""
    if curve not in {"usd", "btc", "strategy_equity"}:
        raise ValueError(f"unsupported metric curve: {curve!r}")
    core = metric
    weighting = "_ew" if core.endswith("_ew") else ""
    if weighting:
        core = core[:-len(weighting)]
    side = next((f"_{side}" for side in ("long", "short")
                 if core.endswith(f"_{side}")), "")
    if side:
        core = core[:-len(side)]
    return f"{core}_{curve}{side}{weighting}"


CURRENCY_METRIC_VARIANTS = {
    metric_with_curve(base, curve): (base, curve)
    for base in CURRENCY_METRICS for curve in ("usd", "btc")
}
# Native lifecycle outputs are described even where the existing selector owners
# do not admit them. This registry does not broaden limit/backend eligibility.
KNOWN_METRICS = frozenset(
    ANALYSIS_SHARED_KEYS | CURRENCY_METRICS | CURRENCY_METRIC_VARIANTS.keys()
    | {"hsl_trigger_count", "hsl_restart_count", "liquidated"}
)


def split_metric_curve(metric: str) -> tuple[str, str | None]:
    """Split an owned account-equity selector; shared USD amounts stay shared."""
    return CURRENCY_METRIC_VARIANTS.get(metric, (metric, None))


def split_metric_stat_suffix(name: str) -> tuple[str, str | None]:
    metric = str(name).strip()
    # Intrinsic statistics are part of the metric, including the final '_ratio'.
    if metric in KNOWN_METRICS:
        return metric, None
    for suffix in STAT_SUFFIXES:
        marker = f"_{suffix}"
        if metric.endswith(marker) and len(metric) > len(marker):
            return metric[:-len(marker)], suffix
    return metric, None


def canonical_metric_name(metric: str) -> str:
    """Current names are exact: this function performs no version conversion."""
    return str(metric).strip()


def metric_aliases(metric: str) -> tuple[str, ...]:
    """Return the single current spelling; released conversion has its own owner."""
    return (canonical_metric_name(metric),)


def resolve_metric_value(metrics: Mapping[str, Any], requested_name: str) -> Any | None:
    return metrics.get(canonical_metric_name(requested_name))


def canonicalize_metric_name(metric: str) -> str:
    metric = canonical_metric_name(metric)
    base, stat = split_metric_stat_suffix(metric)
    canonical = metric_with_curve(base, "usd") if base in CURRENCY_METRICS else base
    return f"{canonical}_{stat}" if stat else canonical


def canonicalize_limit_name(limit_key: str) -> str:
    for old, current in (("lower_bound_", "penalize_if_greater_than_"),
                         ("upper_bound_", "penalize_if_lower_than_")):
        if limit_key.startswith(old):
            return current + canonicalize_metric_name(limit_key[len(old):])
    for prefix in ("penalize_if_greater_than_", "penalize_if_lower_than_"):
        if limit_key.startswith(prefix):
            return prefix + canonicalize_metric_name(limit_key[len(prefix):])
    return canonicalize_metric_name(limit_key)


_RETIRED_WEIGHTED_METRIC = re.compile(r"_w(?:_|$)", re.IGNORECASE)


def reject_retired_metric(metric: str, *, path: str) -> None:
    if str(metric).startswith("strategy_eq_underwater_pct_"):
        raise ValueError(
            f"{path}: daily-worst underwater metric {metric!r} is removed; "
            "choose minute-sampled pain_index_fraction_strategy_equity or "
            "ulcer_index_fraction_strategy_equity and re-evaluate thresholds and results"
        )
    if _RETIRED_WEIGHTED_METRIC.search(str(metric)):
        raise ValueError(
            f"{path}: legacy weighted metric {metric!r} is retired; "
            "choose a supported metric with the intended formula explicitly"
        )
