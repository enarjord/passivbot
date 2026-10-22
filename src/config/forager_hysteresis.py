"""Strict resolved per-side fixed policy for Forager incumbent replacement."""

from math import isfinite
from numbers import Real

from .access import require_config_value

POSITION_SIDES = ("long", "short")


def validate_forager_score_hysteresis(value):
    if not isinstance(value, dict) or set(value) != set(POSITION_SIDES):
        raise TypeError("forager_score_hysteresis must contain exactly long and short")
    result = {}
    for side in POSITION_SIDES:
        number = value[side]
        if isinstance(number, bool) or not isinstance(number, Real):
            raise TypeError(f"forager_score_hysteresis.{side} must be a number")
        number = float(number)
        if not isfinite(number) or number < 0.0:
            raise ValueError(f"forager_score_hysteresis.{side} must be finite and non-negative")
        result[side] = number
    return result


def forager_score_hysteresis(config):
    return validate_forager_score_hysteresis({
        side: require_config_value(config, f"bot.{side}.forager.score_hysteresis_margin")
        for side in POSITION_SIDES
    })
