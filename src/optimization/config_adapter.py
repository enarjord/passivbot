"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

from typing import List, Tuple

from config.bot import validate_unstuck_ema_dist_value
from config.param_paths import (
    OPTIMIZABLE_BOT_KEY_PATHS,
    canonical_optimizer_key,
    resolve_optimizer_key_path,
)
from config.optimize_bounds import flatten_optimize_bounds
from config.shared_bot import flatten_shared_bot_side
from config.schema import get_template_config
from config.strategy import normalize_strategy_kind
from config.strategy_spec import (
    get_strategy_spec,
    strategy_field_names_by_side,
    strategy_optimize_key_path_map,
)
from optimization.bounds import Bound


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


def _strategy_path_map(config: dict) -> dict[str, Tuple[str, ...]]:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    return strategy_optimize_key_path_map(strategy_kind)


def _strategy_field_names_by_side(config: dict) -> dict[str, set[str]]:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    return strategy_field_names_by_side(strategy_kind)


def resolve_optimization_bound_path(config: dict, bound_key: str) -> Tuple[str, ...] | None:
    return resolve_optimizer_key_path(config, bound_key)


def validate_optimize_bounds_against_bot_config(config: dict, optimize_bounds) -> None:
    if not isinstance(optimize_bounds, dict):
        return
    bot_config = config.get("bot") or get_template_config()["bot"]
    optimize_bounds = _flatten_bounds_for_config(config, optimize_bounds)
    strategy_path_map = _strategy_path_map(config)
    for bound_key in optimize_bounds:
        if not isinstance(bound_key, str):
            continue
        canonical_key = canonical_optimizer_key(bound_key)
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
        target_key = canonical_key or bound_key
        if target_key == "long_unstuck_ema_dist":
            bound = Bound.from_config(target_key, optimize_bounds[bound_key])
            validate_unstuck_ema_dist_value(
                bound.low,
                path="optimize.bounds.long_unstuck_ema_dist lower bound",
                pside="long",
            )
        elif target_key == "short_unstuck_ema_dist":
            bound = Bound.from_config(target_key, optimize_bounds[bound_key])
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
                resolved = resolve_optimizer_key_path(config, f"{pside}_{key}") or ("bot", pside, key)
                scalar_paths[key] = resolved
            for key, path in strategy_path_map.items():
                if not key.startswith(f"{pside}_"):
                    continue
                local_key = key.split("_", 1)[1]
                current = config
                found = True
                for part in path:
                    if not isinstance(current, dict) or part not in current:
                        found = False
                        break
                    current = current[part]
                if not found:
                    continue
                if isinstance(current, bool) or not isinstance(current, (int, float)):
                    continue
                scalar_paths[local_key] = path
            for key in sorted(scalar_paths):
                key_paths.append((f"{pside}_{key}", scalar_paths[key]))
        return key_paths
    for bound_key in sorted(optimize_bounds):
        if not isinstance(bound_key, str):
            continue
        canonical_key = canonical_optimizer_key(bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        resolved = resolve_optimization_bound_path(config, bound_key)
        if resolved is None:
            continue
        if canonical_key in OPTIMIZABLE_BOT_KEY_PATHS:
            key_paths.append((bound_key, resolved))
            continue
        if canonical_key in strategy_path_map:
            key_paths.append((bound_key, resolved))
            continue
        pside, key = canonical_key.split("_", 1)
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
