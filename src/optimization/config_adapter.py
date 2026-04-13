"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

from functools import lru_cache
from typing import List, Tuple

import passivbot_rust as pbr

from config.bot import validate_unstuck_ema_dist_value
from config.optimize_bounds import flatten_optimize_bounds
from config.shared_bot import flatten_shared_bot_side, resolve_shared_bot_path
from config.schema import get_template_config
from config.strategy import get_active_strategy_config, normalize_strategy_kind
from optimization.bounds import Bound

OPTIMIZABLE_BOT_KEY_PATHS = {
    "long_forager_volume_ema_span": ("bot", "long", "forager", "volume_ema_span"),
    "long_filter_volume_ema_span": ("bot", "long", "forager", "volume_ema_span"),
    "long_forager_volatility_ema_span": ("bot", "long", "forager", "volatility_ema_span"),
    "long_filter_volatility_ema_span": ("bot", "long", "forager", "volatility_ema_span"),
    "long_forager_score_weights_volume": ("bot", "long", "forager", "score_weights", "volume"),
    "long_forager_score_weights_ema_readiness": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "long_forager_score_weights_volatility": (
        "bot",
        "long",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_score_weights_volume": ("bot", "short", "forager", "score_weights", "volume"),
    "short_forager_score_weights_ema_readiness": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "ema_readiness",
    ),
    "short_forager_score_weights_volatility": (
        "bot",
        "short",
        "forager",
        "score_weights",
        "volatility",
    ),
    "short_forager_volume_ema_span": ("bot", "short", "forager", "volume_ema_span"),
    "short_filter_volume_ema_span": ("bot", "short", "forager", "volume_ema_span"),
    "short_forager_volatility_ema_span": ("bot", "short", "forager", "volatility_ema_span"),
    "short_filter_volatility_ema_span": ("bot", "short", "forager", "volatility_ema_span"),
}

DEPRECATED_OPTIMIZE_BOUND_ALIASES = {
    "long_filter_volume_ema_span": "long_forager_volume_ema_span",
    "long_filter_volatility_ema_span": "long_forager_volatility_ema_span",
    "short_filter_volume_ema_span": "short_forager_volume_ema_span",
    "short_filter_volatility_ema_span": "short_forager_volatility_ema_span",
}


def _flatten_bounds_for_config(config: dict, optimize_bounds: dict) -> dict:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    has_flat_keys = any(
        isinstance(key, str) and (key.startswith("long_") or key.startswith("short_"))
        for key in optimize_bounds
    )
    if not has_flat_keys:
        return flatten_optimize_bounds(optimize_bounds, strategy_kind=strategy_kind)

    flat_bounds = {
        key: value
        for key, value in optimize_bounds.items()
        if isinstance(key, str) and (key.startswith("long_") or key.startswith("short_"))
    }
    nested_bounds = {
        key: value
        for key, value in optimize_bounds.items()
        if not (isinstance(key, str) and (key.startswith("long_") or key.startswith("short_")))
    }
    if nested_bounds:
        flat_bounds = {
            **flatten_optimize_bounds(nested_bounds, strategy_kind=strategy_kind),
            **flat_bounds,
        }
    return flat_bounds


@lru_cache(maxsize=None)
def get_strategy_spec(strategy_kind: str = "trailing_grid") -> dict:
    return pbr.get_strategy_spec(normalize_strategy_kind(strategy_kind))


def _strategy_spec_for_config(config: dict) -> dict:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    return get_strategy_spec(strategy_kind)


def _strategy_path_map(config: dict) -> dict[str, Tuple[str, ...]]:
    spec = _strategy_spec_for_config(config)
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    mapping: dict[str, Tuple[str, ...]] = {}
    for param in spec.get("parameters", []):
        optimize_key = param.get("optimize_key")
        config_path = tuple(param.get("config_path", ()))
        if len(config_path) == 3 and config_path[0] == "strategy" and config_path[1] in ("long", "short"):
            config_path = ("bot", config_path[1], "strategy", strategy_kind, config_path[2])
        if not isinstance(optimize_key, str) or not config_path:
            continue
        mapping[optimize_key] = config_path
    return mapping


def _strategy_field_names_by_side(config: dict) -> dict[str, set[str]]:
    fields = {"long": set(), "short": set()}
    spec = _strategy_spec_for_config(config)
    for param in spec.get("parameters", []):
        side = param.get("side")
        name = param.get("name")
        if side in fields and isinstance(name, str):
            fields[side].add(name)
    return fields


def resolve_optimization_bound_path(config: dict, bound_key: str) -> Tuple[str, ...] | None:
    canonical_key = DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(bound_key, bound_key)
    strategy_path_map = _strategy_path_map(config)
    if canonical_key in strategy_path_map:
        return strategy_path_map[canonical_key]
    if canonical_key in OPTIMIZABLE_BOT_KEY_PATHS:
        return OPTIMIZABLE_BOT_KEY_PATHS[canonical_key]
    try:
        pside, key = canonical_key.split("_", 1)
    except ValueError:
        return None
    if pside not in ("long", "short"):
        return None
    bot_side = config.get("bot", {}).get(pside, {})
    return resolve_shared_bot_path(bot_side, pside, key) or ("bot", pside, key)


def validate_optimize_bounds_against_bot_config(config: dict, optimize_bounds) -> None:
    if not isinstance(optimize_bounds, dict):
        return
    bot_config = config.get("bot") or get_template_config()["bot"]
    optimize_bounds = _flatten_bounds_for_config(config, optimize_bounds)
    strategy_path_map = _strategy_path_map(config)
    for bound_key in optimize_bounds:
        if not isinstance(bound_key, str):
            continue
        canonical_key = DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(bound_key, bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        resolved = resolve_optimization_bound_path(config, bound_key)
        if resolved is None:
            raise KeyError(f"optimize bound {bound_key} does not map to a known bot parameter")
        if canonical_key in strategy_path_map or canonical_key in OPTIMIZABLE_BOT_KEY_PATHS:
            continue
        try:
            pside = resolved[1]
        except IndexError as exc:
            raise KeyError(f"optimize bound {bound_key} resolved to invalid path {resolved!r}") from exc
        flat_pside_cfg = flatten_shared_bot_side(bot_config[pside])
        key = canonical_key.split("_", 1)[1] if "_" in canonical_key else canonical_key
        if key not in flat_pside_cfg:
            raise KeyError(f"optimize bound {bound_key} does not map to bot.{pside}.{key}")
        value = flat_pside_cfg[key]
        if isinstance(value, dict):
            raise KeyError(f"optimize bound {bound_key} must map to a scalar bot.{pside}.{key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise KeyError(
                f"optimize bound {bound_key} must map to a numeric bot.{pside}.{key}, "
                f"got {type(value).__name__}"
            )
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
    template = get_template_config()
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = template["bot"]
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    strategy_config = get_active_strategy_config(config, strategy_kind=strategy_kind)
    strategy_fields = _strategy_field_names_by_side(config)
    strategy_path_map = _strategy_path_map(config)
    optimize_bounds = _flatten_bounds_for_config(config, config.get("optimize", {}).get("bounds", {}))
    validate_optimize_bounds_against_bot_config(config, optimize_bounds)
    if not optimize_bounds:
        for pside in ("long", "short"):
            scalar_paths: dict[str, Tuple[str, ...]] = {}
            pside_cfg = flatten_shared_bot_side(bot_config.get(pside, {}))
            for key in sorted(pside_cfg):
                value = pside_cfg[key]
                if key in strategy_fields.get(pside, set()):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                resolved = resolve_shared_bot_path(bot_config.get(pside, {}), pside, key) or (
                    "bot",
                    pside,
                    key,
                )
                scalar_paths[key] = resolved
            pside_strategy = strategy_config.get(pside, {}) if isinstance(strategy_config, dict) else {}
            if isinstance(pside_strategy, dict):
                for key in sorted(pside_strategy):
                    value = pside_strategy[key]
                    if key not in strategy_fields.get(pside, set()):
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    scalar_paths[key] = ("bot", pside, "strategy", strategy_kind, key)
            for key in sorted(scalar_paths):
                key_paths.append((f"{pside}_{key}", scalar_paths[key]))
        return key_paths
    for bound_key in sorted(optimize_bounds):
        if not isinstance(bound_key, str):
            continue
        canonical_key = DEPRECATED_OPTIMIZE_BOUND_ALIASES.get(bound_key, bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        resolved = resolve_optimization_bound_path(config, bound_key)
        if resolved is None:
            continue
        if bound_key in OPTIMIZABLE_BOT_KEY_PATHS:
            key_paths.append((bound_key, resolved))
            continue
        if canonical_key in strategy_path_map:
            key_paths.append((bound_key, resolved))
            continue
        pside, key = bound_key.split("_", 1)
        if pside not in ("long", "short"):
            continue
        flat_pside_cfg = flatten_shared_bot_side(bot_config[pside])
        if key not in flat_pside_cfg:
            raise KeyError(f"optimize bound {bound_key} does not map to bot.{pside}.{key}")
        if isinstance(flat_pside_cfg[key], dict):
            raise KeyError(f"optimize bound {bound_key} must map to a scalar bot.{pside}.{key}")
        key_paths.append((bound_key, resolved))
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
    optimize_bounds = _flatten_bounds_for_config(config, config["optimize"]["bounds"])
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
