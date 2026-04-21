from collections.abc import Mapping
from typing import Any


CURRENCY_METRICS = {
    "adg",
    "adg_per_exposure_long",
    "adg_per_exposure_short",
    "adg_w",
    "adg_w_per_exposure_long",
    "adg_w_per_exposure_short",
    "calmar_ratio",
    "calmar_ratio_w",
    "drawdown_worst",
    "drawdown_worst_mean_1pct",
    "equity_balance_diff_neg_max",
    "equity_balance_diff_neg_mean",
    "equity_balance_diff_pos_max",
    "equity_balance_diff_pos_mean",
    "equity_choppiness",
    "exposure_mean_ratio",
    "exposure_mean_ratio_w",
    "exposure_ratio",
    "exposure_ratio_w",
    "equity_choppiness_w",
    "equity_jerkiness",
    "equity_jerkiness_w",
    "peak_recovery_hours_equity",
    "expected_shortfall_1pct",
    "exponential_fit_error",
    "exponential_fit_error_w",
    "gain",
    "gain_per_exposure_long",
    "gain_per_exposure_short",
    "mdg",
    "mdg_per_exposure_long",
    "mdg_per_exposure_short",
    "mdg_w",
    "mdg_w_per_exposure_long",
    "mdg_w_per_exposure_short",
    "omega_ratio",
    "omega_ratio_w",
    "paper_loss_mean_ratio",
    "paper_loss_mean_ratio_w",
    "paper_loss_ratio",
    "paper_loss_ratio_w",
    "sharpe_ratio",
    "sharpe_ratio_w",
    "sortino_ratio",
    "sortino_ratio_w",
    "sterling_ratio",
    "sterling_ratio_w",
}

SHARED_METRICS = {
    "positions_held_per_day",
    "positions_held_per_day_w",
    "position_held_hours_mean",
    "position_held_hours_max",
    "position_held_hours_median",
    "position_unchanged_hours_max",
    "win_rate",
    "win_rate_w",
    "trade_loss_max",
    "trade_loss_mean",
    "trade_loss_median",
    "volume_pct_per_day_avg",
    "volume_pct_per_day_avg_w",
    "loss_profit_ratio",
    "loss_profit_ratio_w",
    "peak_recovery_hours_pnl",
    "high_exposure_hours_mean_long",
    "high_exposure_hours_max_long",
    "high_exposure_hours_mean_short",
    "high_exposure_hours_max_short",
    "adg_pnl",
    "adg_pnl_w",
    "mdg_pnl",
    "mdg_pnl_w",
    "sharpe_ratio_pnl",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl",
    "sortino_ratio_pnl_w",
    "gain_strategy_eq",
    "adg_strategy_eq",
    "mdg_strategy_eq",
    "sharpe_ratio_strategy_eq",
    "sortino_ratio_strategy_eq",
    "omega_ratio_strategy_eq",
    "expected_shortfall_1pct_strategy_eq",
    "calmar_ratio_strategy_eq",
    "sterling_ratio_strategy_eq",
    "adg_strategy_eq_w",
    "mdg_strategy_eq_w",
    "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_strategy_eq_w",
    "omega_ratio_strategy_eq_w",
    "calmar_ratio_strategy_eq_w",
    "sterling_ratio_strategy_eq_w",
    "drawdown_worst_strategy_eq",
    "drawdown_worst_ema_strategy_eq",
    "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_mean_1pct_ema_strategy_eq",
    "peak_recovery_hours_strategy_eq",
    "hard_stop_triggers_per_year",
    "hard_stop_restarts_per_year",
    "hard_stop_restarts_per_year_long",
    "hard_stop_restarts_per_year_short",
    "hard_stop_halt_to_restart_equity_loss_pct",
    "hard_stop_time_in_yellow_pct",
    "hard_stop_time_in_orange_pct",
    "hard_stop_time_in_red_pct",
    "hard_stop_duration_minutes_mean",
    "hard_stop_duration_minutes_max",
    "hard_stop_trigger_drawdown_mean",
    "hard_stop_panic_close_loss_sum",
    "hard_stop_panic_close_loss_max",
    "hard_stop_flatten_time_minutes_mean",
    "hard_stop_post_restart_retrigger_pct",
}


ANALYSIS_SHARED_KEYS = SHARED_METRICS | {
    "loss_profit_ratio_long",
    "loss_profit_ratio_short",
    "pnl_ratio_long_short",
    "long_short_profit_ratio",
    "total_wallet_exposure_max",
    "total_wallet_exposure_mean",
    "total_wallet_exposure_median",
    "entry_initial_balance_pct_long",
    "entry_initial_balance_pct_short",
    "drawdown_worst_strategy_eq_long",
    "drawdown_worst_strategy_eq_short",
    "drawdown_worst_ema_strategy_eq_long",
    "drawdown_worst_ema_strategy_eq_short",
    "drawdown_worst_mean_1pct_strategy_eq_long",
    "drawdown_worst_mean_1pct_strategy_eq_short",
    "drawdown_worst_mean_1pct_ema_strategy_eq_long",
    "drawdown_worst_mean_1pct_ema_strategy_eq_short",
    "peak_recovery_hours_strategy_eq_long",
    "peak_recovery_hours_strategy_eq_short",
    "hard_stop_triggers_long",
    "hard_stop_triggers_short",
    "hard_stop_restarts_long",
    "hard_stop_restarts_short",
}

STAT_SUFFIXES = ("min", "max", "mean", "std", "median")

METRIC_ALIASES = {
    "gain_strategy_pnl_rebased": "gain_strategy_eq",
    "adg_strategy_pnl_rebased": "adg_strategy_eq",
    "mdg_strategy_pnl_rebased": "mdg_strategy_eq",
    "sharpe_ratio_strategy_pnl_rebased": "sharpe_ratio_strategy_eq",
    "sortino_ratio_strategy_pnl_rebased": "sortino_ratio_strategy_eq",
    "omega_ratio_strategy_pnl_rebased": "omega_ratio_strategy_eq",
    "expected_shortfall_1pct_strategy_pnl_rebased": "expected_shortfall_1pct_strategy_eq",
    "calmar_ratio_strategy_pnl_rebased": "calmar_ratio_strategy_eq",
    "sterling_ratio_strategy_pnl_rebased": "sterling_ratio_strategy_eq",
    "adg_strategy_pnl_rebased_w": "adg_strategy_eq_w",
    "mdg_strategy_pnl_rebased_w": "mdg_strategy_eq_w",
    "sharpe_ratio_strategy_pnl_rebased_w": "sharpe_ratio_strategy_eq_w",
    "sortino_ratio_strategy_pnl_rebased_w": "sortino_ratio_strategy_eq_w",
    "omega_ratio_strategy_pnl_rebased_w": "omega_ratio_strategy_eq_w",
    "calmar_ratio_strategy_pnl_rebased_w": "calmar_ratio_strategy_eq_w",
    "sterling_ratio_strategy_pnl_rebased_w": "sterling_ratio_strategy_eq_w",
    "drawdown_worst_hsl": "drawdown_worst_strategy_eq",
    "drawdown_worst_hsl_long": "drawdown_worst_strategy_eq_long",
    "drawdown_worst_hsl_short": "drawdown_worst_strategy_eq_short",
    "drawdown_worst_ema_hsl": "drawdown_worst_ema_strategy_eq",
    "drawdown_worst_ema_hsl_long": "drawdown_worst_ema_strategy_eq_long",
    "drawdown_worst_ema_hsl_short": "drawdown_worst_ema_strategy_eq_short",
    "drawdown_worst_mean_1pct_hsl": "drawdown_worst_mean_1pct_strategy_eq",
    "drawdown_worst_mean_1pct_hsl_long": "drawdown_worst_mean_1pct_strategy_eq_long",
    "drawdown_worst_mean_1pct_hsl_short": "drawdown_worst_mean_1pct_strategy_eq_short",
    "drawdown_worst_mean_1pct_ema_hsl": "drawdown_worst_mean_1pct_ema_strategy_eq",
    "drawdown_worst_mean_1pct_ema_hsl_long": "drawdown_worst_mean_1pct_ema_strategy_eq_long",
    "drawdown_worst_mean_1pct_ema_hsl_short": "drawdown_worst_mean_1pct_ema_strategy_eq_short",
    "peak_recovery_hours_hsl": "peak_recovery_hours_strategy_eq",
    "peak_recovery_hours_hsl_long": "peak_recovery_hours_strategy_eq_long",
    "peak_recovery_hours_hsl_short": "peak_recovery_hours_strategy_eq_short",
}

CANONICAL_TO_ALIASES: dict[str, tuple[str, ...]] = {}
for _old, _new in METRIC_ALIASES.items():
    CANONICAL_TO_ALIASES.setdefault(_new, tuple())
    CANONICAL_TO_ALIASES[_new] = CANONICAL_TO_ALIASES[_new] + (_old,)


def split_metric_stat_suffix(name: str) -> tuple[str, str | None]:
    metric = str(name).strip()
    for suffix in STAT_SUFFIXES:
        marker = f"_{suffix}"
        if metric.endswith(marker) and len(metric) > len(marker):
            return metric[: -len(marker)], suffix
    return metric, None


def _with_stat_suffix(metric: str, stat: str | None) -> str:
    return f"{metric}_{stat}" if stat else metric


def canonical_metric_name(metric: str) -> str:
    base, stat = split_metric_stat_suffix(metric)
    return _with_stat_suffix(METRIC_ALIASES.get(base, base), stat)


def metric_aliases(metric: str) -> tuple[str, ...]:
    base, stat = split_metric_stat_suffix(metric)
    canonical_base = METRIC_ALIASES.get(base, base)
    candidates = [_with_stat_suffix(base, stat), _with_stat_suffix(canonical_base, stat)]
    candidates.extend(
        _with_stat_suffix(alias, stat)
        for alias in CANONICAL_TO_ALIASES.get(canonical_base, ())
    )
    return tuple(dict.fromkeys(candidates))


def resolve_metric_value(metrics: Mapping[str, Any], requested_name: str) -> Any | None:
    for candidate in metric_aliases(requested_name):
        if candidate in metrics:
            return metrics[candidate]
    return None


def canonicalize_metric_name(metric: str) -> str:
    metric = canonical_metric_name(metric)
    if metric.endswith("_usd") or metric.endswith("_btc"):
        return metric

    for prefix, suffix in (("usd_", "usd"), ("btc_", "btc")):
        if metric.startswith(prefix):
            core = canonical_metric_name(metric[len(prefix) :])
            if core in SHARED_METRICS:
                return core
            return f"{core}_{suffix}"

    if metric in SHARED_METRICS:
        return metric

    if metric in CURRENCY_METRICS:
        return f"{metric}_usd"

    return metric


def canonicalize_limit_name(limit_key: str) -> str:
    if limit_key.startswith("lower_bound_"):
        metric = limit_key[len("lower_bound_") :]
        return "penalize_if_greater_than_" + canonicalize_metric_name(metric)
    if limit_key.startswith("upper_bound_"):
        metric = limit_key[len("upper_bound_") :]
        return "penalize_if_lower_than_" + canonicalize_metric_name(metric)
    prefixes = ["penalize_if_greater_than_", "penalize_if_lower_than_"]
    for prefix in prefixes:
        if limit_key.startswith(prefix):
            metric = limit_key[len(prefix) :]
            return prefix + canonicalize_metric_name(metric)
    return canonicalize_metric_name(limit_key)
