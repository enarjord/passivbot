"""Shared fixed bot policy validation and simulation inheritance."""

import math

from .access import require_config_value
from .universe import normalize_always_include


def effective_minimum_cost_filter(config):
    override = require_config_value(config, "backtest.filter_by_min_effective_cost")
    shared = require_config_value(
        config, "bot.global.universe.filter_by_min_effective_cost"
    )
    if not isinstance(shared, bool):
        raise TypeError(
            "config.bot.global.universe.filter_by_min_effective_cost must be a boolean"
        )
    if override is not None and not isinstance(override, bool):
        raise TypeError(
            "config.backtest.filter_by_min_effective_cost must be a boolean or null"
        )
    return shared if override is None else override


def validate_shared_bot_policy(config):
    for path in (
        "bot.global.hedge_mode",
        "bot.global.execution.market_orders_allowed",
        "bot.global.universe.filter_by_min_effective_cost",
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
        forager = require_config_value(config, f"bot.{side}.forager")
        forager["always_include"] = normalize_always_include(
            forager["always_include"], path=f"config.bot.{side}.forager.always_include"
        )
