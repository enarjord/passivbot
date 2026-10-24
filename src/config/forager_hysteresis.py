"""Strict resolved per-side fixed policy for Forager incumbent replacement."""

from math import isfinite
from numbers import Real

from .access import require_config_value

POSITION_SIDES = ("long", "short")


def validate_forager_score_hysteresis_margin(value, *, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a number")
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise ValueError(f"{path} must be finite and non-negative")
    return number


def validate_forager_score_hysteresis(value):
    if not isinstance(value, dict) or set(value) != set(POSITION_SIDES):
        raise TypeError("forager_score_hysteresis must contain exactly long and short")
    return {
        side: validate_forager_score_hysteresis_margin(
            value[side], path=f"forager_score_hysteresis.{side}",
        ) for side in POSITION_SIDES
    }


def forager_score_hysteresis(config):
    return validate_forager_score_hysteresis({
        side: require_config_value(config, f"bot.{side}.forager.score_hysteresis_margin")
        for side in POSITION_SIDES
    })
