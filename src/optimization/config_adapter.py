"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

import math
from typing import List, Tuple

from config.bot import validate_unstuck_ema_dist_value
from config.param_paths import (
    OPTIMIZABLE_BOT_KEY_PATHS,
    canonical_optimizer_key,
    resolve_optimizer_key_path,
    require_existing_config_path,
)
from config.optimize_bounds import flatten_optimize_bounds
from config.shared_bot import flatten_shared_bot_side
from config.schema import get_template_config
from config.strategy import merge_runtime_bot_side, normalize_strategy_kind
from config.strategy_spec import (
    get_strategy_spec,
    strategy_optimize_key_path_map,
)
from optimization.bounds import Bound
from optimizer_overrides import (
    COUPLED_UNSTUCK_EMA_BOUND_KEYS,
    unstuck_ema_spans_coupled,
)


def _flatten_bounds_for_config(config: dict, optimize_bounds: dict) -> dict:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    return flatten_optimize_bounds(optimize_bounds, strategy_kind=strategy_kind)



def _flatten_required_optimize_bounds(config: dict) -> dict:
    raw_bounds = config.get("optimize", {}).get("bounds")
    if not isinstance(raw_bounds, dict):
        raise TypeError("config.optimize.bounds must be a non-empty dict")
    optimize_bounds = _flatten_bounds_for_config(config, raw_bounds)
    if not optimize_bounds:
        raise ValueError("config.optimize.bounds must contain at least one optimizer bound")
    return optimize_bounds


def _strategy_path_map(config: dict) -> dict[str, Tuple[str, ...]]:
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    return strategy_optimize_key_path_map(strategy_kind)


def resolve_optimization_bound_path(config: dict, bound_key: str) -> Tuple[str, ...] | None:
    return resolve_optimizer_key_path(config, bound_key)


def validate_optimize_bounds_against_bot_config(config: dict, optimize_bounds) -> None:
    if not isinstance(optimize_bounds, dict):
        return
    bot_config = config.get("bot") or get_template_config()["bot"]
    optimize_bounds = _flatten_bounds_for_config(config, optimize_bounds)
    strategy_path_map = _strategy_path_map(config)
    cooldown_ranges = {}
    for bound_key in optimize_bounds:
        if not isinstance(bound_key, str):
            continue
        canonical_key = canonical_optimizer_key(bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        resolved = resolve_optimization_bound_path(config, bound_key)
        if resolved is None:
            raise KeyError(f"optimize bound {bound_key} does not map to a known bot parameter")
        if config.get("live", {}).get("hsl_engine") == "revised" and "hsl" in resolved:
            from config.hsl_revised import _number
            name = resolved[-1]
            constraints = {"red_threshold": dict(minimum=0, maximum=1, strict=True),
                           "ema_span_minutes": dict(minimum=1),
                           "cooldown_minutes_after_red": dict(minimum=0)}
            if name in constraints:
                bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
                for endpoint in (bound.low, bound.high):
                    _number(endpoint, f"optimize.bounds.{bound_key}", **constraints[name])
        adaptive_domain = {
            ("forager", "unilateralness_ema_span_1m"): (1.0, 100_000.0),
            ("forager", "score_weights", "unilateralness"): (0.0, math.inf),
            ("entry_cooldown", "weights_minutes", "exposure_ratio"): (0.0, math.inf),
            ("entry_cooldown", "weights_minutes", "adverse_directionality"): (0.0, math.inf),
            ("entry_cooldown", "base_duration_minutes"): (0.0, math.inf),
            ("entry_cooldown", "min_duration_minutes"): (0.0, math.inf),
            ("entry_cooldown", "max_duration_minutes"): (0.0, math.inf),
        }.get(resolved[2:])
        if adaptive_domain is not None:
            bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
            minimum, maximum = adaptive_domain
            if any(
                not math.isfinite(value) or not minimum <= value <= maximum
                for value in (bound.low, bound.high)
            ):
                raise ValueError(
                    f"optimize.bounds.{bound_key} endpoints must be finite and in "
                    f"[{minimum}, {maximum}]"
                )
        if resolved[2:] in (
            ("entry_cooldown", "min_duration_minutes"),
            ("entry_cooldown", "max_duration_minutes"),
        ):
            cooldown_ranges[resolved[1], resolved[-1]] = bound
        if resolved[:2] == ("bot", "hsl"):
            value = bot_config["hsl"].get(resolved[-1])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KeyError(f"optimize bound {bound_key} must map to a numeric portfolio HSL parameter")
            continue
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
        if key in ("unstuck_ema_span_0", "unstuck_ema_span_1"):
            bound = Bound.from_config(target_key, optimize_bounds[bound_key])
            if (
                not math.isfinite(bound.low)
                or not math.isfinite(bound.high)
                or bound.low <= 0.0
            ):
                raise ValueError(
                    f"optimize.bounds.{target_key} must be positive and finite"
                )
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

    # Independently sampled dimensions must be valid at every corner, not just
    # the all-low/all-high configurations used to estimate warmup.
    fixed_cooldowns = {"long": {}, "short": {}}
    for dotted_path, value in (config.get("optimize", {}).get("fixed_runtime_overrides") or {}).items():
        path = require_existing_config_path(config, dotted_path)
        if len(path) == 4 and path[0] == "bot" and path[1] in fixed_cooldowns and path[2] == "entry_cooldown":
            fixed_cooldowns[path[1]][path[3]] = value
    for pside in ("long", "short"):
        floor_bound = cooldown_ranges.get((pside, "min_duration_minutes"))
        ceiling_bound = cooldown_ranges.get((pside, "max_duration_minutes"))
        if floor_bound is None and ceiling_bound is None:
            continue
        # Apply the independently reachable corner first, then the same coin
        # patches used at runtime: a pinned override wins over a searched gene.
        corner = flatten_shared_bot_side(bot_config[pside])
        if floor_bound:
            corner["entry_cooldown_min_duration_minutes"] = floor_bound.high
        if ceiling_bound:
            corner["entry_cooldown_max_duration_minutes"] = ceiling_bound.low
        corner = merge_runtime_bot_side(
            corner, pside=pside,
            override_side={"entry_cooldown": fixed_cooldowns[pside]},
        )
        effective_sides = [("bot", corner)]
        for coin, patch in (config.get("coin_overrides") or {}).items():
            override_side = patch.get("bot", {}).get(pside, {})
            effective_sides.append((
                f"coin_overrides.{coin}.bot",
                merge_runtime_bot_side(corner, pside=pside, override_side=override_side),
            ))
        for source, effective in effective_sides:
            highest_floor = effective["entry_cooldown_min_duration_minutes"]
            lowest_ceiling = effective["entry_cooldown_max_duration_minutes"]
            if lowest_ceiling is not None and highest_floor > lowest_ceiling:
                raise ValueError(
                    f"optimize.bounds.{pside}.entry_cooldown ({source}.{pside}): "
                    f"highest min_duration_minutes ({highest_floor}) must not exceed "
                    f"lowest max_duration_minutes ({lowest_ceiling})"
                )


def get_optimization_key_paths(config) -> List[Tuple[str, Tuple[str, ...]]]:
    key_paths: List[Tuple[str, Tuple[str, ...]]] = []
    template = get_template_config()
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = template["bot"]
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    strategy_path_map = _strategy_path_map(config)
    optimize_bounds = _flatten_required_optimize_bounds(config)
    validate_optimize_bounds_against_bot_config(config, optimize_bounds)
    for bound_key in sorted(optimize_bounds):
        if not isinstance(bound_key, str):
            continue
        canonical_key = canonical_optimizer_key(bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        if (
            unstuck_ema_spans_coupled(config)
            and canonical_key in COUPLED_UNSTUCK_EMA_BOUND_KEYS
        ):
            continue
        resolved = resolve_optimization_bound_path(config, bound_key)
        if resolved is None:
            continue
        if canonical_key in OPTIMIZABLE_BOT_KEY_PATHS or resolved[:2] == ("bot", "hsl"):
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
    optimize_bounds = _flatten_required_optimize_bounds(config)
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
