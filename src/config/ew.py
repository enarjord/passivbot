"""Exponential analysis configuration and metric selection (no return calculations)."""

import math
from collections.abc import Mapping

from .metrics import canonicalize_metric_name, split_metric_stat_suffix

EW_DEFINITION_VERSION = 1
EW_METRIC_BASES = frozenset({"adg", "sharpe_ratio", "sortino_ratio", "omega_ratio"})
EW_METRICS = frozenset(
    [f"{base}_strategy_eq_ew" for base in EW_METRIC_BASES]
    + [f"{base}_ew_{currency}" for base in EW_METRIC_BASES for currency in ("usd", "btc")]
)


def is_ew_metric(metric) -> bool:
    base, _ = split_metric_stat_suffix(canonicalize_metric_name(str(metric)))
    return base in EW_METRICS


def validate_ew_half_life_days(value):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1.0
    ):
        raise ValueError("backtest.ew_half_life_days must be null or a finite number >= 1 day")
    return float(value)


def validate_ew_candle_interval(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 1
        or value != int(value)
        or 1440 % int(value) != 0
    ):
        raise ValueError(
            "backtest.candle_interval_minutes must be a positive integer and "
            "divide a UTC day when exponential (_ew) analysis is enabled"
        )
    return int(value)


def configured_ew_metrics(config):
    from .limits import normalize_limit_entries
    from .scoring import extract_objective_specs

    metrics = [spec.metric for spec in extract_objective_specs(config)]
    try:
        limits = normalize_limit_entries(config.get("optimize", {}).get("limits", []))
    except (TypeError, ValueError) as exc:
        # Preserve the canonical loader diagnostic when discovering EW limit requests.
        raise ValueError(f"optimize.limits malformed or unsupported: {exc}") from exc
    metrics.extend(entry["metric"] for entry in limits if entry.get("enabled", True))
    return {metric for metric in metrics if is_ew_metric(metric)}


def reject_ew_scenario_overrides(overrides):
    from .scenario_overrides import normalize_scenario_overrides

    if "backtest.ew_half_life_days" in normalize_scenario_overrides(overrides):
        raise ValueError("Scenario overrides cannot set backtest.ew_half_life_days; use one global half-life")


def validate_ew_config(config):
    half_life = validate_ew_half_life_days(config["backtest"].get("ew_half_life_days"))
    config["backtest"]["ew_half_life_days"] = half_life
    if half_life is None and configured_ew_metrics(config):
        raise ValueError("Exponential (_ew) scoring and limits require backtest.ew_half_life_days")
    if half_life is not None:
        validate_ew_candle_interval(config["backtest"].get("candle_interval_minutes", 1))
    for scenario in config["backtest"].get("scenarios", []):
        if isinstance(scenario, Mapping):
            if "ew_half_life_days" in scenario:
                raise ValueError("Scenarios cannot set ew_half_life_days; use one global half-life")
            reject_ew_scenario_overrides(scenario.get("overrides"))
