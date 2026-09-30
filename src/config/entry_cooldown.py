"""Canonical cooldown validation and bounded history requirements (no trading policy)."""

import math


def validate_entry_cooldown(cfg, *, path):
    for key in ("base_duration_minutes", "min_duration_minutes"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{path}.{key} must be finite and nonnegative")
    weights = cfg["weights_minutes"]
    if not isinstance(weights, dict) or set(weights) != {
        "exposure_ratio",
        "adverse_directionality",
    }:
        raise ValueError(
            f"{path}.weights_minutes requires exposure_ratio and adverse_directionality"
        )
    for key, value in weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{path}.weights_minutes.{key} must be finite and nonnegative")
    maximum = cfg["max_duration_minutes"]
    if maximum is None:
        if any(weights.values()):
            raise ValueError(
                f"{path}.max_duration_minutes must be finite when modifiers are enabled"
            )
    elif (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(maximum)
        or maximum < cfg["min_duration_minutes"]
    ):
        raise ValueError(f"{path}.max_duration_minutes must be finite and >= min_duration_minutes")


def constant_duration(params):
    """Input-independence check for validated, flattened cooldown settings."""
    maximum = params.get("entry_cooldown_max_duration_minutes")
    if maximum is not None and max(
        params.get("risk_entry_cooldown_minutes", 0.0),
        params.get("entry_cooldown_min_duration_minutes", 0.0),
    ) >= maximum:
        return maximum
    return None


def uses_adverse_rms(params):
    return (
        params.get("entry_cooldown_weights_minutes", {}).get("adverse_directionality", 0.0) > 0.0
        and constant_duration(params) is None
    )


def maximum_duration(cfg):
    validate_entry_cooldown(cfg, path="entry_cooldown")
    maximum = cfg["max_duration_minutes"]
    if any(cfg["weights_minutes"].values()):
        return maximum
    return min(
        max(cfg["base_duration_minutes"], cfg["min_duration_minutes"]),
        maximum if maximum is not None else math.inf,
    )
