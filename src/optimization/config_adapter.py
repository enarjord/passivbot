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
    iter_optimizer_key_paths,
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


from config.access import get_optional_config_value

def _flatten_bounds_for_config(config: dict, optimize_bounds: dict) -> dict:
    strategy_kind = normalize_strategy_kind(get_optional_config_value(config, 'bot.global.strategy_kind'))
    return flatten_optimize_bounds(optimize_bounds, strategy_kind=strategy_kind)



def _flatten_required_optimize_bounds(config: dict) -> dict:
    raw_bounds = config.get("optimize", {}).get("bounds")
    if not isinstance(raw_bounds, dict):
        raise TypeError("config.optimize.bounds must be a non-empty dict")
    optimize_bounds = _flatten_bounds_for_config(config, raw_bounds)
    if not optimize_bounds:
        raise ValueError(
            "config.optimize.bounds must contain at least one optimizer bound"
        )
    return optimize_bounds


def _strategy_path_map(config: dict) -> dict[str, Tuple[str, ...]]:
    strategy_kind = normalize_strategy_kind(get_optional_config_value(config, 'bot.global.strategy_kind'))
    return strategy_optimize_key_path_map(strategy_kind)


def resolve_optimization_bound_path(
    config: dict, bound_key: str
) -> Tuple[str, ...] | None:
    return resolve_optimizer_key_path(config, bound_key)


def validate_optimize_bounds_against_bot_config(config: dict, optimize_bounds) -> None:
    if not isinstance(optimize_bounds, dict):
        return
    bot_config = config.get("bot") or get_template_config()["bot"]
    authored_bounds = optimize_bounds
    optimize_bounds = _flatten_bounds_for_config(config, optimize_bounds)
    strategy_path_map = _strategy_path_map(config)
    cooldown_ranges = {}
    for bound_key, resolved in iter_optimizer_key_paths(
        config, (key for key in optimize_bounds if isinstance(key, str))
    ):
        canonical_key = canonical_optimizer_key(bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        if resolved is None:
            raise KeyError(
                f"optimize bound {bound_key} does not map to a known bot parameter"
            )
        if resolved in {
            ("bot", "global", "unstuck", "memory_ema_span_minutes"),
            ("bot", "global", "loss_gate", "memory_ema_span_minutes"),
        }:
            from config.risk_memory import normalize_memory_ema_span_minutes

            bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
            for endpoint in (bound.low, bound.high):
                normalize_memory_ema_span_minutes(endpoint, field=f"optimize.bounds.{bound_key}")
            continue
        if "hsl" in resolved:
            from config.hsl import _number

            name = resolved[-1]
            constraints = {
                "red_threshold": dict(minimum=0, maximum=1, strict=True),
                "memory_ema_span_minutes": dict(minimum=1),
                "drawdown_ratio_ema_span_minutes": dict(minimum=1),
                "cooldown_minutes_after_red": dict(minimum=0),
            }
            if name in constraints:
                bound = Bound.from_config(bound_key, optimize_bounds[bound_key])
                for endpoint in (bound.low, bound.high):
                    _number(endpoint, f"optimize.bounds.{bound_key}", **constraints[name])
                    if name in {"memory_ema_span_minutes", "drawdown_ratio_ema_span_minutes"}:
                        from config.risk_memory import normalize_memory_ema_span_minutes

                        normalize_memory_ema_span_minutes(
                            endpoint, field=f"optimize.bounds.{bound_key}"
                        )
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
        if resolved[:3] == ("bot", "global", "hsl"):
            value = bot_config["global"]["hsl"].get(resolved[-1])
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KeyError(
                    f"optimize bound {bound_key} must map to a numeric portfolio HSL parameter"
                )
            continue
        if (
            canonical_key in strategy_path_map
            or canonical_key in OPTIMIZABLE_BOT_KEY_PATHS
        ):
            continue
        try:
            pside = resolved[1]
        except IndexError as exc:
            raise KeyError(
                f"optimize bound {bound_key} resolved to invalid path {resolved!r}"
            ) from exc
        key = canonical_key.split("_", 1)[1] if "_" in canonical_key else canonical_key
        value = config
        try:
            for part in resolved:
                value = value[part]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"optimize bound {bound_key} has no canonical value at {resolved!r}") from exc
        if isinstance(value, dict):
            raise KeyError(
                f"optimize bound {bound_key} must map to a scalar bot.{pside}.{key}"
            )
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


    validate_forager_weight_search(config, authored_bounds)


def validate_forager_weight_search(
    config: dict, optimize_bounds=None, *, tunable_keys=None, suite_config=None
) -> None:
    """Check normalized weight authority before candidate or dataset evaluation.

    Runtime pins precede scenario pins. Validate the base candidate stage too:
    its normalization runs before the final scenario patches are applied.
    """
    from config.param_paths import resolve_bound_selectors
    from config.scenario_overrides import normalize_scenario_overrides

    raw = (config.get("optimize", {}).get("bounds", {})
           if optimize_bounds is None else optimize_bounds)
    flat = _flatten_bounds_for_config(config, raw)
    fixed = set(resolve_bound_selectors(
        config, config.get("optimize", {}).get("fixed_params", []) or [], flat
    ))
    runtime_pins = list(normalize_scenario_overrides(
        config.get("optimize", {}).get("fixed_runtime_overrides", {})
    ).items())
    contexts = [("", runtime_pins)]
    suite = config.get("backtest", {}) if suite_config is None else suite_config
    enabled = (suite.get("suite_enabled", False)
               if suite_config is None else suite.get("enabled", False))
    if enabled:
        for index, scenario in enumerate(suite.get("scenarios") or []):
            if not isinstance(scenario, dict):
                continue  # The scenario-schema gate owns malformed rows.
            label = scenario.get("label", index + 1)
            context = f"backtest.scenarios[{index}] ({label}): "
            scenario_pins = normalize_scenario_overrides(scenario.get("overrides"))
            contexts.append((context, [*runtime_pins, *scenario_pins.items()]))
    for context, pins in contexts:
        _validate_forager_weight_context(config, flat, fixed, tunable_keys, pins, context)


def _validate_forager_weight_context(config, flat, fixed, tunable_keys, pins, context):
    def pinned_value(path, configured):
        pinned, removed = False, False
        for selector, value in pins:
            parts = tuple(selector.split("."))
            if path[:len(parts)] != parts:
                continue
            pinned, removed = True, False
            for part in path[len(parts):]:
                if not isinstance(value, dict) or part not in value:
                    removed = True
                    break
                value = value[part]
            if not removed:
                configured = value
        if removed:
            raise ValueError(
                f"{context}override removes required {'.'.join(path)}; "
                "provide the complete object or use leaf overrides"
            )
        return configured, pinned

    names = ("volume", "volatility", "ema_readiness", "unilateralness")
    for side in ("long", "short"):
        bot = config.get("bot", {}).get(side, {})
        weights = bot.get("forager", {}).get("score_weights")
        if not isinstance(weights, dict):
            continue
        missing = set(names) - set(weights)
        if missing:
            raise ValueError(
                f"{context}bot.{side}.forager.score_weights missing required weights: "
                f"{', '.join(sorted(missing))}"
            )
        active = True
        for field, key in (("max_positions", "n_positions"),
                           ("total_wallet_exposure_limit", "total_wallet_exposure_limit")):
            bound_key = f"{side}_{key}"
            value, pinned = pinned_value(
                ("bot", side, "risk", field), bot.get("risk", {}).get(field, 0.0)
            )
            if (not pinned and bound_key in flat and bound_key not in fixed
                    and (tunable_keys is None or bound_key in tunable_keys)):
                value = Bound.from_config(bound_key, flat[bound_key]).high
            if float(value) <= 0:
                active = False
        if not active:
            continue
        varied, fixed_positive, minima, fixed_values = [], [], [], {}
        for name in names:
            key = f"{side}_forager_score_weights_{name}"
            configured, pinned = pinned_value(
                ("bot", side, "forager", "score_weights", name), weights[name]
            )
            if pinned and (isinstance(configured, bool) or not isinstance(configured, (int, float))
                           or not math.isfinite(configured) or configured < 0):
                raise ValueError(
                    f"{context}bot.{side}.forager.score_weights.{name} pin "
                    "must be finite and nonnegative"
                )
            can_vary = (key in flat and key not in fixed and not pinned
                        and (tunable_keys is None or key in tunable_keys))
            bound = (Bound.from_config(key, flat[key]) if key in flat
                     else Bound(float(configured), float(configured)))
            if can_vary and bound.low != bound.high:
                varied.append(name)
                minima.append(bound.low)
            else:
                value = (float(configured) if pinned or key in fixed
                         or (tunable_keys is not None and key not in tunable_keys) else bound.low)
                fixed_values[name] = value
                minima.append(value)
                if value > 0:
                    fixed_positive.append(name)
        if not varied:
            continue
        fallback_fixed = (sum(minima) == 0 and "ema_readiness" in fixed_values)
        if fixed_positive or fallback_fixed:
            affected = sorted(fixed_positive) or ["ema_readiness (all-zero normalization fallback)"]
            raise ValueError(
                f"{context}bot.{side}.forager.score_weights: partial normalized-weight search would change "
                f"fixed positive weight(s): {', '.join(affected)}. Search every participating "
                "weight or freeze the whole score_weights group; omitted zero weights must "
                "also remain zero at the all-zero fallback."
            )


def get_optimization_key_paths(config) -> List[Tuple[str, Tuple[str, ...]]]:
    key_paths: List[Tuple[str, Tuple[str, ...]]] = []
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = get_template_config()["bot"]
    strategy_kind = normalize_strategy_kind(get_optional_config_value(config, 'bot.global.strategy_kind'))
    strategy_path_map = _strategy_path_map(config)
    optimize_bounds = _flatten_required_optimize_bounds(config)
    validate_optimize_bounds_against_bot_config(config, config["optimize"]["bounds"])
    for bound_key, resolved in iter_optimizer_key_paths(
        config, (key for key in sorted(optimize_bounds) if isinstance(key, str))
    ):
        canonical_key = canonical_optimizer_key(bound_key)
        if canonical_key != bound_key and canonical_key in optimize_bounds:
            continue
        if (
            unstuck_ema_spans_coupled(config)
            and canonical_key in COUPLED_UNSTUCK_EMA_BOUND_KEYS
        ):
            continue
        if resolved is None:
            continue
        if (canonical_key in OPTIMIZABLE_BOT_KEY_PATHS
            or resolved[:3] == ("bot", "global", "hsl")
            or resolved in {("bot", "global", "unstuck", "memory_ema_span_minutes"),
                            ("bot", "global", "loss_gate", "memory_ema_span_minutes")}):
            key_paths.append((bound_key, resolved))
            continue
        if canonical_key in strategy_path_map:
            key_paths.append((bound_key, resolved))
            continue
        value = config
        try:
            for part in resolved:
                value = value[part]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"optimize bound {bound_key} has no canonical value at {resolved!r}") from exc
        if isinstance(value, dict):
            raise KeyError(f"optimize bound {bound_key} must map to a scalar at {resolved!r}")
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
            (Bound.from_config(key, optimize_bounds[key]).high if key in optimize_bounds
             else float(bot_config[pside]["risk"][field])) > 0.0
            for key, field in ((f"{pside}_n_positions", "max_positions"),
                               (f"{pside}_total_wallet_exposure_limit", "total_wallet_exposure_limit"))
        )

    for bound_key, path in key_paths:
        assert (
            bound_key in optimize_bounds
        ), f"bound {bound_key} missing from optimize.bounds"
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
