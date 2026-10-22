"""Canonical control values shared by authored policy and its consumers."""

import math
from types import MappingProxyType


FORCED_MODE_PATHS = frozenset({("live", "modes", side) for side in ("long", "short")})
MODE_ALIASES = MappingProxyType({
    "n": "normal", "normal": "normal",
    "m": "manual", "manual": "manual",
    "gs": "graceful_stop", "graceful_stop": "graceful_stop", "graceful-stop": "graceful_stop",
    "p": "panic", "panic": "panic",
    "t": "tp_only", "tp": "tp_only", "tp_only": "tp_only", "tp-only": "tp_only",
})
FORCED_MODES = tuple(dict.fromkeys(MODE_ALIASES.values()))
WARMUP_CAP_PATH = ("bot", "portfolio", "warmup", "max_minutes")
WARMUP_CONCURRENCY_PATH = ("live", "startup", "warmup_concurrency")


def forced_mode_input_pattern():
    """Portable JSON Schema pattern for the existing strip/lower input domain."""
    # Python strip includes these Unicode whitespace characters. Spell them
    # explicitly because JavaScript and Python disagree on what \s includes.
    whitespace = r"[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*"
    alternatives = ["".join(f"[{char}{char.upper()}]" if char.isalpha() else char
                            for char in alias) for alias in MODE_ALIASES]
    return "^" + whitespace + "(?:" + "|".join(alternatives) + ")" + whitespace + "$"


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


def warmup_cap(value, *, path="bot.portfolio.warmup.max_minutes"):
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
    warmup = config["bot"]["portfolio"]["warmup"]
    warmup["max_minutes"] = warmup_cap(warmup["max_minutes"])


def forced_mode_cli(value):
    return optional_forced_mode(None if value.lower() == "null" else value, path="forced mode")


def warmup_cap_cli(value):
    return warmup_cap(None if value.lower() == "null" else float(value))


def warmup_concurrency_cli(value):
    return warmup_concurrency(value if value == "auto" else int(value))
