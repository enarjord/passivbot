"""Shared fixed bot policy validation and simulation inheritance."""

import math

from .access import require_config_value
from .universe import normalize_always_include


POSITION_SIDES = ("long", "short")


def minimum_cost_filter(config, side):
    """Return one side's authored live eligibility policy, without simulation overrides."""
    if side not in POSITION_SIDES:
        raise ValueError(f"invalid position side {side!r}")
    path = f"bot.{side}.universe.filter_by_min_effective_cost"
    value = require_config_value(config, path)
    if not isinstance(value, bool):
        raise TypeError(f"config.{path} must be a boolean")
    return value


def effective_minimum_cost_filter(config, side):
    """A nullable scalar simulation override inherits each side independently."""
    shared = minimum_cost_filter(config, side)
    override = require_config_value(config, "backtest.filter_by_min_effective_cost")
    if override is not None and not isinstance(override, bool):
        raise TypeError(
            "config.backtest.filter_by_min_effective_cost must be a boolean or null"
        )
    return shared if override is None else override


def effective_minimum_cost_filters(config):
    return {side: effective_minimum_cost_filter(config, side) for side in POSITION_SIDES}


def validate_minimum_cost_filters(value, *, path="filter_by_min_effective_cost"):
    """Validate the complete native policy pair; never coerce missing or malformed flags."""
    if not isinstance(value, dict) or set(value) != set(POSITION_SIDES):
        raise TypeError(f"{path} must contain exactly long and short boolean flags")
    for side in POSITION_SIDES:
        if not isinstance(value[side], bool):
            raise TypeError(f"{path}.{side} must be a boolean")
    return {side: value[side] for side in POSITION_SIDES}


def validate_shared_bot_policy(config):
    for path in (
        "bot.global.hedge_mode",
        "bot.global.execution.market_orders_allowed",
    ):
        if not isinstance(require_config_value(config, path), bool):
            raise TypeError(f"config.{path} must be a boolean")
    override = require_config_value(config, "backtest.filter_by_min_effective_cost")
    if override is not None and not isinstance(override, bool):
        raise TypeError(
            "config.backtest.filter_by_min_effective_cost must be a boolean or null"
        )
    for path in (
        "bot.global.universe.minimum_coin_age_days",
        "bot.global.execution.market_order_near_touch_fraction",
        "bot.global.forager.score_hysteresis_margin",
        "bot.global.warmup.span_multiplier",
    ):
        value = require_config_value(config, path)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"config.{path} must be a finite nonnegative number")
    for side in ("long", "short"):
        minimum_cost_filter(config, side)
        forager = require_config_value(config, f"bot.{side}.forager")
        forager["always_include"] = normalize_always_include(
            forager["always_include"], path=f"config.bot.{side}.forager.always_include"
        )
