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


def reject_gpu_adaptive(config):
    from .shared_bot import get_grouped_bot_value

    sides = list(config["bot"].values())
    sides += [
        s
        for patch in (config.get("coin_overrides") or {}).values()
        for s in patch.get("bot", {}).values()
    ]
    for side in sides:
        weights = get_grouped_bot_value(side, "entry_cooldown_weights_minutes", {})
        score_weights = get_grouped_bot_value(side, "forager_score_weights", {})
        if (
            any(weights.values())
            or score_weights.get("unilateralness", 0) != 0
            or get_grouped_bot_value(side, "entry_cooldown_min_duration_minutes", 0) != 0
            or get_grouped_bot_value(side, "entry_cooldown_max_duration_minutes") is not None
        ):
            raise ValueError(
                "Adaptive cooldown bounds/modifiers and unilateralness require the CPU backend"
            )
    from .optimize_bounds import flatten_optimize_bounds
    from optimization.bounds import Bound

    bounds = flatten_optimize_bounds(
        config.get("optimize", {}).get("bounds", {}),
        strategy_kind=config.get("live", {}).get("strategy_kind", "trailing_martingale"),
    )
    for key, value in bounds.items():
        if any(
            token in key
            for token in (
                "forager_score_weights_unilateralness",
                "entry_cooldown_weights_minutes",
                "entry_cooldown_min_duration_minutes",
                "entry_cooldown_max_duration_minutes",
            )
        ):
            bound = Bound.from_config(key, value)
            if bound.low != 0 or bound.high != 0 or "max_duration_minutes" in key:
                raise ValueError(
                    "Adaptive cooldown and unilateralness optimizer bounds require the CPU backend"
                )
