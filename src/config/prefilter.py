"""Opt-in, optimizer-only policy for deterministic configuration probes."""

from copy import deepcopy
import math


PREFILTER_VERSION = 2
PREFILTER_METRICS = frozenset(
    {
        "entry_close_span",
        "entry_threshold",
        "entry_retracement",
        "close_retracement",
        "entry_cooldown_minutes",
        "grid_span",
    }
)
PREFILTER_DEFAULTS = {
    "checks": [],
    "exposure_ratios": [0.25, 0.5, 0.8, 0.95],
    "volatility_ema_1m": 0.0,
    "volatility_ema_1h": 0.0,
    "adverse_directionality": 0.0,
    "grid_max_entries": 1024,
    "penalty_weight": 1e6,
}


def _number(value, path, *, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise ValueError(
            f"{path} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return float(value)


def resolve_prefilter(value=None):
    if value is not None and not isinstance(value, dict):
        raise TypeError("optimize.prefilter must be an object")
    result = deepcopy(PREFILTER_DEFAULTS)
    if value is not None:
        unknown = set(value) - result.keys()
        if unknown:
            raise ValueError(f"unknown optimize.prefilter settings: {sorted(unknown)}")
        result.update(deepcopy(value))
    ratios = result["exposure_ratios"]
    if not isinstance(ratios, list) or not ratios or len(ratios) > 64:
        raise ValueError("optimize.prefilter.exposure_ratios must contain 1..64 values")
    result["exposure_ratios"] = [
        _number(x, "prefilter.exposure_ratios") for x in ratios
    ]
    if any(x >= 0.999 for x in result["exposure_ratios"]):
        raise ValueError(
            "prefilter.exposure_ratios must be below the 0.999 full-capacity boundary"
        )
    for name in (
        "volatility_ema_1m",
        "volatility_ema_1h",
        "adverse_directionality",
        "penalty_weight",
    ):
        result[name] = _number(
            result[name], f"prefilter.{name}", positive=name == "penalty_weight"
        )
    if result["adverse_directionality"] > 1:
        raise ValueError("prefilter.adverse_directionality must be in [0, 1]")
    count = result["grid_max_entries"]
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 4096:
        raise ValueError("prefilter.grid_max_entries must be an integer in [2, 4096]")
    checks = result["checks"]
    if not isinstance(checks, list) or len(checks) > 64:
        raise ValueError("prefilter.checks must be a list with at most 64 checks")
    normalized = []
    for index, raw in enumerate(checks):
        path = f"prefilter.checks[{index}]"
        if not isinstance(raw, dict) or set(raw) - {
            "metric",
            "enabled",
            "min",
            "max",
            "scale",
            "weight",
            "sides",
        }:
            raise ValueError(f"{path} must contain only documented check fields")
        check = {
            "enabled": False,
            "min": None,
            "max": None,
            "scale": 1.0,
            "weight": 1.0,
            "sides": ["long", "short"],
            **deepcopy(raw),
        }
        if not isinstance(check["enabled"], bool):
            raise ValueError(f"{path}.enabled must be a boolean")
        if (
            not isinstance(check.get("metric"), str)
            or check["metric"] not in PREFILTER_METRICS
        ):
            raise ValueError(
                f"{path}.metric must be one of {sorted(PREFILTER_METRICS)}"
            )
        sides = check["sides"]
        if (
            not isinstance(sides, list)
            or not sides
            or any(
                not isinstance(side, str) or side not in ("long", "short")
                for side in sides
            )
            or len(set(sides)) != len(sides)
        ):
            raise ValueError(f"{path}.sides must select long and/or short")
        for name in ("min", "max"):
            if check[name] is not None:
                check[name] = _number(check[name], f"{path}.{name}")
        if check["enabled"] and check["min"] is None and check["max"] is None:
            raise ValueError(f"{path} needs a min and/or max when enabled")
        if (
            check["min"] is not None
            and check["max"] is not None
            and check["min"] > check["max"]
        ):
            raise ValueError(f"{path}.min must not exceed max")
        for name in ("scale", "weight"):
            check[name] = _number(check[name], f"{path}.{name}", positive=True)
        normalized.append(check)
    result["checks"] = normalized
    return result


def prefilter_enabled(params):
    return any(
        check.get("enabled", False) for check in (params or {}).get("checks", [])
    )
