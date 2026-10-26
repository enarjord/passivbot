"""Lightweight eligibility metadata for Apple MPS optimizer metrics."""

from __future__ import annotations

import json

import hjson
from config.ew import EW_METRICS

EXPONENTIAL_EQUITY_METRICS = EW_METRICS
EXPONENTIAL_RAW_EQUITY_METRICS = frozenset(
    name for name in EW_METRICS if name.endswith("_strategy_equity_ew")
)
EXPONENTIAL_BTC_EQUITY_METRICS = frozenset(
    name for name in EW_METRICS if name.endswith("_btc_ew")
)

# Metric-name groups that toggle opt-in MPS proxy features. Keep this metadata
# Torch-free so backend preflight and ordinary CPU-only installs can prove the
# dispatch contract without importing the optional GPU runtime.
RAW_STRATEGY_RISK_METRICS = frozenset({
    "drawdown_max_fraction_strategy_equity",
    "drawdown_daily_worst_1pct_mean_fraction_strategy_equity",
    "pain_index_fraction_strategy_equity",
    "ulcer_index_fraction_strategy_equity",
})

RAW_STRATEGY_EQUITY_METRICS = frozenset({
    f"{stem}_strategy_equity" for stem in (
        "adg_fraction", "mdg_fraction", "sharpe_ratio", "sortino_ratio", "omega_ratio",
        "calmar_ratio", "sterling_ratio", "expected_shortfall_1pct_fraction",
        "adg_rolling_hmean_fraction", "adg_time_integrated_fraction", "positive_log_return_participation_score",
    )
}) | EXPONENTIAL_RAW_EQUITY_METRICS
RAW_STRATEGY_DRAWDOWN_CONSUMERS = RAW_STRATEGY_RISK_METRICS | frozenset({
    "calmar_ratio_strategy_equity", "sterling_ratio_strategy_equity",
})

BTC_INTRADAY_RISK_METRICS = frozenset(
    {
        "calmar_ratio_btc",
        "drawdown_max_fraction_btc",
        "drawdown_daily_worst_1pct_mean_fraction_btc",
        "expected_shortfall_1pct_fraction_btc",
        "sharpe_ratio_btc",
        "sortino_ratio_btc",
        "sterling_ratio_btc",
    }
)

EQUITY_BALANCE_DIFF_METRICS = frozenset(
    {
        f"equity_balance_gap_{sign}_{stat}_fraction_{currency}"
        for sign in ("negative", "positive")
        for stat in ("max", "mean")
        for currency in ("btc", "usd")
    }
    | {
        f"adg_to_paper_loss_{stat}_ratio_{currency}"
        for stat in ("max", "mean")
        for currency in ("btc", "usd")
    }
)

ENTRY_INTERVAL_METRICS = frozenset(
    {
        "entry_interval_mean_hours",
        "entry_interval_median_hours",
        "entry_interval_p95_hours",
        "entry_interval_p99_hours",
        "entry_interval_max_hours",
    }
)

# These metrics remain available from exact Rust backtests and analysis, but are
# intentionally ineligible for Metal proxy objectives and proxy-side limits.
# The registry is Torch-free so GPU backend preflight can inspect preserved raw
# config without coupling shared config loading to the optional GPU runtime.
GPU_EXACT_ONLY_METRICS = frozenset(
    {
        "daily_realized_pnl_mean_fraction",
        "equity_balance_gap_positive_max_fraction_btc",
        "equity_balance_gap_positive_max_fraction_usd",
        "equity_balance_gap_positive_mean_fraction_btc",
        "equity_balance_gap_positive_mean_fraction_usd",
        "fills_active_days_count",
        "analysis_duration_days",
        "fills_count",
        "fills_close_count",
        "fills_entry_count",
        "fills_count_long",
        "fills_count_short",
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p99_hours",
        "fills_close_per_day",
        "fills_entry_per_day",
        "fills_per_day_long",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
        "fills_per_day_short",
        "gain_multiplier_btc",
        "gain_multiplier_per_twel_ratio_btc_long",
        "gain_multiplier_per_twel_ratio_usd_long",
        "gain_multiplier_per_twel_ratio_btc_short",
        "gain_multiplier_per_twel_ratio_usd_short",
        "gain_multiplier_strategy_equity",
        "gain_multiplier_usd",
        "hsl_flatten_time_mean_minutes",
        "hsl_panic_episode_loss_min_fraction",
        "hsl_panic_fill_loss_max_usd",
        "hsl_panic_fill_loss_sum_usd",
        "hsl_restart_count",
        "hsl_restart_count_long",
        "hsl_restart_count_short",
        "hsl_trigger_count",
        "hsl_trigger_count_long",
        "hsl_trigger_count_short",
        "daily_realized_pnl_median_fraction",
        "peak_recovery_max_days_btc",
        "peak_recovery_max_days_usd",
        "peak_recovery_max_days_realized_pnl",
        "peak_recovery_max_hours_btc",
        "peak_recovery_max_hours_usd",
        "peak_recovery_max_hours_realized_pnl",
        "recovery_max_hours_strategy_equity",
        "sharpe_ratio_realized_pnl",
        "sortino_ratio_realized_pnl",
        *{
            f"high_exposure_{stat}_{unit}_{side}"
            for unit in ("hours", "days")
            for stat in ("mean", "max")
            for side in ("long", "short")
        },
    }
)

HARD_STOP_LIFECYCLE_METRICS = frozenset(
    {
        "hsl_red_duration_max_minutes",
        "hsl_red_duration_mean_minutes",
        "hsl_flatten_time_mean_minutes",
        "hsl_restart_retrigger_fraction",
        "hsl_restart_count",
        "hsl_restart_count_long",
        "hsl_restarts_per_year",
        "hsl_restarts_per_year_long",
        "hsl_restarts_per_year_short",
        "hsl_restart_count_short",
        "hsl_red_time_fraction",
        "hsl_trigger_signal_mean_fraction",
        "hsl_trigger_count",
        "hsl_trigger_count_long",
        "hsl_triggers_per_year",
        "hsl_trigger_count_short",
    }
)
HARD_STOP_PANIC_LOSS_METRICS = frozenset(
    {
        "hsl_panic_loss_starting_balance_fraction",
        "hsl_panic_episode_loss_max_fraction",
        "hsl_panic_episode_loss_mean_fraction",
        "hsl_panic_episode_loss_min_fraction",
        "hsl_panic_fill_loss_max_usd",
        "hsl_panic_fill_loss_sum_usd",
    }
)
HARD_STOP_PROXY_METRICS = tuple(
    sorted(
        (HARD_STOP_LIFECYCLE_METRICS | HARD_STOP_PANIC_LOSS_METRICS)
        - GPU_EXACT_ONLY_METRICS
    )
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

    configured_surfaces: list[tuple[object, str]] = []
    effective = config.get("_raw_effective")
    provenance = effective if isinstance(effective, dict) else config.get("_raw")
    for source in (config, provenance):
        if not isinstance(source, dict):
            continue
        payload = source.get("config", source)
        if not isinstance(payload, dict):
            continue
        optimize = payload.get("optimize")
        if not isinstance(optimize, dict):
            continue
        configured_surfaces.extend(
            (
                (optimize.get("scoring"), "scoring"),
                (optimize.get("limits"), "limits"),
            )
        )

    found: set[str] = set()

    def add_metric(value) -> None:
        token = str(value or "").strip()
        if token in GPU_EXACT_ONLY_METRICS:
            found.add(token)

    def add_legacy_limit_name(value) -> None:
        token = str(value or "").strip().lstrip("-")
        for prefix in (
            "lower_bound_",
            "upper_bound_",
            "penalize_if_greater_than_",
            "penalize_if_lower_than_",
        ):
            if token.startswith(prefix):
                token = token[len(prefix) :]
                break
        add_metric(token)

    def visit_scoring(value) -> None:
        if not isinstance(value, (list, tuple)):
            return
        for entry in value:
            if isinstance(entry, dict):
                add_metric(entry.get("metric"))
            elif isinstance(entry, str):
                add_metric(entry)

    def visit_limits(value) -> None:
        if isinstance(value, dict):
            if "metric" in value or "name" in value:
                if bool(value.get("enabled", True)):
                    add_metric(value.get("metric") or value.get("name"))
                return
            for key in value:
                add_legacy_limit_name(key)
            return
        if isinstance(value, (list, tuple)):
            for entry in value:
                visit_limits(entry)
            return
        if not isinstance(value, str):
            return
        stripped = value.strip()
        if stripped[:1] in {"[", "{"}:
            try:
                parsed = json.loads(stripped)
            except Exception:
                try:
                    parsed = hjson.loads(stripped)
                except Exception:
                    parsed = None
            if isinstance(parsed, (dict, list, tuple)):
                visit_limits(parsed)
                return
        for segment in value.split("--"):
            tokens = segment.strip().split()
            if not tokens:
                continue
            if any(
                token.lower()
                in {"enabled=false", "enabled=0", "enabled=no", "enabled=off"}
                for token in tokens[1:]
            ):
                continue
            add_legacy_limit_name(tokens[0])

    for surface, surface_kind in configured_surfaces:
        if surface_kind == "scoring":
            visit_scoring(surface)
        else:
            visit_limits(surface)
    return frozenset(found)


def reject_configured_exact_only_gpu_metrics(config: dict) -> None:
    reject_exact_only_gpu_metric_names(configured_exact_only_gpu_metrics(config))
