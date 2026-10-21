"""Side-owned fixed age policy and factual native transport."""

import math

from .access import require_config_value


AGE_PATHS = frozenset(("bot", side, "universe", "minimum_coin_age_days") for side in ("long", "short"))


def validate_age_value(value, *, path):
    error = f"config.{path} must be a finite nonnegative number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(error)
    try:
        days = float(value)
    except OverflowError as exc:
        raise ValueError(error) from exc
    if not math.isfinite(days) or days < 0:
        raise ValueError(error)
    return days


def minimum_coin_age_days(config, side):
    path = f"bot.{side}.universe.minimum_coin_age_days"
    return validate_age_value(require_config_value(config, path), path=path)


def inception_timestamp_ms(value):
    """Missing/zero is unknown, never a fabricated listing timestamp."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2**64 - 1 or not math.isfinite(value) or int(value) != value:
        raise ValueError("inception_timestamp_ms must be a positive integer or null")
    return int(value) if value else None
