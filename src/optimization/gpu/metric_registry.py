"""Lightweight eligibility metadata for Apple MPS optimizer metrics."""

from __future__ import annotations

import re


# These metrics remain available from exact Rust backtests and analysis, but are
# intentionally ineligible for Metal proxy objectives and proxy-side limits.
# The registry is Torch-free so config normalization can reject raw aliases
# before shared metric canonicalization erases their provenance.
GPU_EXACT_ONLY_METRICS = frozenset(
    {
        "adg_pnl",
        "adg_pnl_w",
        "equity_balance_diff_pos_max_btc",
        "equity_balance_diff_pos_max_usd",
        "equity_balance_diff_pos_mean_btc",
        "equity_balance_diff_pos_mean_usd",
        "fills_active_days_count",
        "fills_analysis_duration_days",
        "fills_count",
        "fills_count_close",
        "fills_count_entry",
        "fills_count_long",
        "fills_count_short",
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p99_hours",
        "fills_per_day_close",
        "fills_per_day_entry",
        "fills_per_day_long",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
        "fills_per_day_short",
        "gain_btc",
        "gain_per_exposure_long_btc",
        "gain_per_exposure_long_usd",
        "gain_per_exposure_short_btc",
        "gain_per_exposure_short_usd",
        "gain_strategy_eq",
        "gain_usd",
        "hard_stop_flatten_time_minutes_mean",
        "hard_stop_panic_close_loss_drawdown_pct_min",
        "hard_stop_panic_close_loss_max",
        "hard_stop_panic_close_loss_sum",
        "hard_stop_restarts",
        "hard_stop_restarts_long",
        "hard_stop_restarts_short",
        "hard_stop_time_in_orange_pct",
        "hard_stop_time_in_yellow_pct",
        "hard_stop_triggers",
        "hard_stop_triggers_long",
        "hard_stop_triggers_short",
        "long_short_profit_ratio",
        "mdg_pnl",
        "mdg_pnl_w",
        "peak_recovery_days_equity_btc",
        "peak_recovery_days_equity_usd",
        "peak_recovery_days_pnl",
        "peak_recovery_days_strategy_eq",
        "peak_recovery_hours_equity_btc",
        "peak_recovery_hours_equity_usd",
        "peak_recovery_hours_pnl",
        "peak_recovery_hours_strategy_eq",
        "sharpe_ratio_pnl",
        "sharpe_ratio_pnl_w",
        "sortino_ratio_pnl",
        "sortino_ratio_pnl_w",
        "strategy_eq_underwater_pct_median",
        *{
            f"high_exposure_{unit}_{stat}_{side}"
            for unit in ("hours", "days")
            for stat in ("mean", "max")
            for side in ("long", "short")
        },
    }
)


def reject_exact_only_gpu_metric_names(metric_names) -> frozenset[str]:
    raw = frozenset(str(name).strip() for name in metric_names)
    exact_only = sorted(raw & GPU_EXACT_ONLY_METRICS)
    if exact_only:
        raise ValueError(
            f"GPU foundation reserves optimizer metrics {exact_only} for exact "
            "Rust backtests and analysis; use proxy-eligible metrics or the CPU "
            "optimizer"
        )
    return raw


def configured_exact_only_gpu_metrics(config: dict) -> frozenset[str]:
    """Recover exact-only spellings from current and preserved raw config."""

    configured_surfaces = []
    for source in (
        config,
        config.get("_raw"),
        config.get("_raw_effective"),
    ):
        if not isinstance(source, dict):
            continue
        payload = source.get("config", source)
        if not isinstance(payload, dict):
            continue
        optimize = payload.get("optimize")
        if not isinstance(optimize, dict):
            continue
        configured_surfaces.extend(
            (optimize.get("scoring"), optimize.get("limits"))
        )

    exact_only = tuple(sorted(GPU_EXACT_ONLY_METRICS, key=len, reverse=True))
    found: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if not isinstance(value, str):
            return
        token = value.strip()
        for metric in exact_only:
            if (
                token == metric
                or token.endswith(f"_{metric}")
                or re.search(
                    rf"(?:^|[^A-Za-z0-9]|_){re.escape(metric)}"
                    r"(?![A-Za-z0-9_])",
                    token,
                )
            ):
                found.add(metric)

    for surface in configured_surfaces:
        visit(surface)
    return frozenset(found)


def reject_configured_exact_only_gpu_metrics(config: dict) -> None:
    reject_exact_only_gpu_metric_names(configured_exact_only_gpu_metrics(config))
