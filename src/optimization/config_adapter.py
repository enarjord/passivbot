"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

from typing import List, Tuple

from config.bot import validate_unstuck_ema_dist_value
from config.schema import get_template_config
from optimization.bounds import Bound

OPTIMIZABLE_BOT_KEY_PATHS = {
    "long_forager_volume_ema_span": ("bot", "long", "forager_volume_ema_span"),
    "long_filter_volume_ema_span": ("bot", "long", "forager_volume_ema_span"),
    "long_forager_volatility_ema_span": ("bot", "long", "forager_volatility_ema_span"),
    "long_filter_volatility_ema_span": ("bot", "long", "forager_volatility_ema_span"),
    "long_forager_score_weights_volume": ("bot", "long", "forager_score_weights", "volume"),
    "long_forager_score_weights_ema_readiness": (
        "bot",
        "long",
        "forager_score_weights",
        "ema_readiness",
    ),
    "long_forager_score_weights_volatility": (
        "bot",
        "long",
        "forager_score_weights",
        "volatility",
    ),
    "short_forager_score_weights_volume": ("bot", "short", "forager_score_weights", "volume"),
    "short_forager_score_weights_ema_readiness": (
        "bot",
        "short",
        "forager_score_weights",
        "ema_readiness",
    ),
    "short_forager_score_weights_volatility": (
        "bot",
        "short",
        "forager_score_weights",
        "volatility",
    ),
    "short_forager_volume_ema_span": ("bot", "short", "forager_volume_ema_span"),
    "short_filter_volume_ema_span": ("bot", "short", "forager_volume_ema_span"),
    "short_forager_volatility_ema_span": ("bot", "short", "forager_volatility_ema_span"),
    "short_filter_volatility_ema_span": ("bot", "short", "forager_volatility_ema_span"),
}

DEPRECATED_OPTIMIZE_BOUND_ALIASES = {
    "long_filter_volume_ema_span": "long_forager_volume_ema_span",
    "long_filter_volatility_ema_span": "long_forager_volatility_ema_span",
    "short_filter_volume_ema_span": "short_forager_volume_ema_span",
    "short_filter_volatility_ema_span": "short_forager_volatility_ema_span",
}


def _validate_standard_optimize_bound_target(bound_key: str, bot_config) -> None:
    if "_" not in bound_key:
        return
    pside, key = bound_key.split("_", 1)
    if pside not in ("long", "short"):
        return
    if key not in bot_config[pside]:
        raise KeyError(f"optimize bound {bound_key} does not map to bot.{pside}.{key}")
    value = bot_config[pside][key]
    if isinstance(value, dict):
        raise KeyError(f"optimize bound {bound_key} must map to a scalar bot.{pside}.{key}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KeyError(
            f"optimize bound {bound_key} must map to a numeric bot.{pside}.{key}, "
            f"got {type(value).__name__}"
        )


def validate_optimize_bounds_against_bot_config(bot_config, optimize_bounds) -> None:
    if not isinstance(optimize_bounds, dict):
        return
    for bound_key in optimize_bounds:
        if not isinstance(bound_key, str):
            continue
        canonical_key = DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(bound_key)
        if canonical_key is not None and canonical_key in optimize_bounds:
            continue
        if bound_key in OPTIMIZABLE_BOT_KEY_PATHS:
            continue
        _validate_standard_optimize_bound_target(bound_key, bot_config)
        if bound_key == "long_unstuck_ema_dist":
            bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
            validate_unstuck_ema_dist_value(
                bound.low,
                path="optimize.bounds.long_unstuck_ema_dist lower bound",
                pside="long",
            )
        elif bound_key == "short_unstuck_ema_dist":
            bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
            validate_unstuck_ema_dist_value(
                bound.high,
                path="optimize.bounds.short_unstuck_ema_dist upper bound",
                pside="short",
            )


def get_optimization_key_paths(config) -> List[Tuple[str, Tuple[str, ...]]]:
    key_paths: List[Tuple[str, Tuple[str, ...]]] = []
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = get_template_config()["bot"]
    optimize_bounds = config.get("optimize", {}).get("bounds", {})
    validate_optimize_bounds_against_bot_config(bot_config, optimize_bounds)
    if not optimize_bounds:
        for pside in ("long", "short"):
            pside_cfg = bot_config.get(pside, {})
            for key in sorted(pside_cfg):
                value = pside_cfg[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                key_paths.append((f"{pside}_{key}", ("bot", pside, key)))
        return key_paths
    for bound_key in sorted(optimize_bounds):
        if not isinstance(bound_key, str):
            continue
        canonical_key = DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(bound_key)
        if canonical_key is not None and canonical_key in optimize_bounds:
            continue
        if bound_key in OPTIMIZABLE_BOT_KEY_PATHS:
            key_paths.append((bound_key, OPTIMIZABLE_BOT_KEY_PATHS[bound_key]))
            continue
        pside, key = bound_key.split("_", 1)
        key_paths.append((bound_key, ("bot", pside, key)))
    return key_paths


def extract_bounds_tuple_list_from_config(config) -> List[Bound]:
    """
    Extracts list of Bound instances for bot parameters.
    Also sets all bounds to (low, low, step) if pside is not enabled.

    Supported formats:
        - [low, high]: continuous optimization (step=None)
        - [low, high, step]: discrete optimization with given step
        - [low, high, 0] or [low, high, null]: treated as continuous
        - single value: fixed parameter (low=high, step=None)
    """
    bounds = []
    optimize_bounds = config["optimize"]["bounds"]
    key_paths = get_optimization_key_paths(config)
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = get_template_config()["bot"]
    pside_enabled = {}
    for pside in ("long", "short"):
        pside_enabled[pside] = all(
            Bound.from_config(k, optimize_bounds[k]).high > 0.0
            for k in [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
        )

    for bound_key, path in key_paths:
        assert bound_key in optimize_bounds, f"bound {bound_key} missing from optimize.bounds"
        bound_vals = Bound.from_config(bound_key, optimize_bounds[bound_key])
        if len(path) >= 2 and path[:2] == ("bot", "long"):
            if pside_enabled["long"]:
                bounds.append(bound_vals)
            else:
                bounds.append(Bound(bound_vals.low, bound_vals.low, bound_vals.step))
            continue
        if len(path) >= 2 and path[:2] == ("bot", "short"):
            if pside_enabled["short"]:
                bounds.append(bound_vals)
            else:
                bounds.append(Bound(bound_vals.low, bound_vals.low, bound_vals.step))
            continue
        bounds.append(bound_vals)
    return bounds
