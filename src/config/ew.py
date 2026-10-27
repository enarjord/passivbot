"""Exponential analysis configuration and metric selection (no return calculations)."""

import math
from collections.abc import Mapping

from .metrics import (
    canonicalize_metric_name,
    split_metric_stat_suffix,
    DEFAULT_EW_HALF_LIFE_DAYS,
)

EW_DEFINITION_VERSION = 1
EW_METRICS = frozenset({"cagr_fraction_strategy_equity_ew"})


def is_ew_metric(metric) -> bool:
    base, _ = split_metric_stat_suffix(canonicalize_metric_name(str(metric)))
    return base in EW_METRICS


def validate_ew_half_life_days(value):
    if value is None:
        return DEFAULT_EW_HALF_LIFE_DAYS
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            "backtest.ew_half_life_days must be null or a finite positive number of days"
        )
    return float(value)


def reject_ew_scenario_overrides(overrides):
    from .scenario_overrides import normalize_scenario_overrides

    if "backtest.ew_half_life_days" in normalize_scenario_overrides(overrides):
        raise ValueError(
            "Scenario overrides cannot set backtest.ew_half_life_days; use one global half-life"
        )


def validate_ew_config(config):
    half_life = validate_ew_half_life_days(config["backtest"].get("ew_half_life_days"))
    config["backtest"]["ew_half_life_days"] = half_life
    for scenario in config["backtest"].get("scenarios", []):
        if isinstance(scenario, Mapping):
            if "ew_half_life_days" in scenario:
                raise ValueError(
                    "Scenarios cannot set ew_half_life_days; use one global half-life"
                )
            reject_ew_scenario_overrides(scenario.get("overrides"))
