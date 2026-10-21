"""Canonical control values shared by authored policy and its consumers."""

import math


FORCED_MODE_PATHS = frozenset({("live", "modes", side) for side in ("long", "short")})
FORCED_MODES = ("normal", "manual", "graceful_stop", "panic", "tp_only")
WARMUP_CAP_PATH = ("bot", "global", "warmup", "max_minutes")
WARMUP_CONCURRENCY_PATH = ("live", "startup", "warmup_concurrency")


def optional_forced_mode(value, *, path):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a mode string or null")
    from .overrides import expand_PB_mode

    try:
        return expand_PB_mode(value.strip())
    except Exception as exc:
        raise ValueError(f"{path} must be a supported mode or null; empty strings are not supported") from exc


def warmup_cap(value, *, path="bot.global.warmup.max_minutes"):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a positive finite number or null")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} must be a positive finite number or null")
    return float(value)


def warmup_concurrency(value, *, path="live.startup.warmup_concurrency"):
    if value == "auto":
        return "auto"
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be 'auto' or a positive integer")
    return value


def validate_control_value(parts, value):
    path = ".".join(parts)
    if parts in FORCED_MODE_PATHS:
        optional_forced_mode(value, path=path)
    elif parts == WARMUP_CAP_PATH:
        warmup_cap(value, path=path)
    elif parts == WARMUP_CONCURRENCY_PATH:
        warmup_concurrency(value, path=path)


def normalize_control_values(config):
    modes = config["live"]["modes"]
    for side in ("long", "short"):
        modes[side] = optional_forced_mode(modes[side], path=f"live.modes.{side}")
    startup = config["live"]["startup"]
    startup["warmup_concurrency"] = warmup_concurrency(startup["warmup_concurrency"])
    warmup = config["bot"]["global"]["warmup"]
    warmup["max_minutes"] = warmup_cap(warmup["max_minutes"])


def forced_mode_cli(value):
    return optional_forced_mode(None if value.lower() == "null" else value, path="forced mode")


def warmup_cap_cli(value):
    return warmup_cap(None if value.lower() == "null" else float(value))


def warmup_concurrency_cli(value):
    return warmup_concurrency(value if value == "auto" else int(value))
