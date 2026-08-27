"""Lightweight eligibility metadata for Apple MPS optimizer metrics."""

from __future__ import annotations

# These metrics remain available from exact Rust backtests and analysis, but are
# intentionally ineligible for Metal proxy objectives and proxy-side limits.
# The registry is Torch-free so GPU backend preflight can inspect preserved raw
# config without coupling shared config loading to the optional GPU runtime.
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
