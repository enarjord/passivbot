from copy import deepcopy
import math
from typing import Optional

from .strategy_spec import (
    BOT_POSITION_SIDES,
    get_supported_strategy_kinds,
    get_strategy_optimize_bounds,
    normalize_strategy_kind,
    strategy_optimize_key_path_map,
)


from config.access import get_optional_config_value

def _flatten_strategy_bound_items(bounds: dict, prefix: tuple[str, ...] = ()):
    for key, value in bounds.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _flatten_strategy_bound_items(value, path)
        else:
            yield "_".join(path), value


SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY = {
    "entry_cooldown": {
        "base_duration_minutes": "risk_entry_cooldown_minutes",
        "weights_minutes_exposure_ratio": "entry_cooldown_weights_minutes_exposure_ratio",
        "weights_minutes_adverse_directionality": "entry_cooldown_weights_minutes_adverse_directionality",
        "min_duration_minutes": "entry_cooldown_min_duration_minutes",
        "max_duration_minutes": "entry_cooldown_max_duration_minutes",
    },
    "forager": {
        "score_weights_unilateralness": "forager_score_weights_unilateralness",
        "unilateralness_ema_span_minutes": "unilateralness_ema_span_1m",
        "score_weights_ema_readiness": "forager_score_weights_ema_readiness",
        "score_weights_volatility": "forager_score_weights_volatility",
        "score_weights_volume": "forager_score_weights_volume",
        "volatility_ema_span_minutes": "forager_volatility_ema_span_1m",
        "volume_ema_span_minutes": "forager_volume_ema_span_1m",
    },
    "hsl": {
        "cooldown_after_red_minutes": "hsl_cooldown_minutes_after_red",
        "drawdown_ratio_ema_span_minutes": "hsl_drawdown_ratio_ema_span_minutes",
        "memory_ema_span_minutes": "hsl_memory_ema_span_minutes",
        "drawdown_ratio_threshold": "hsl_red_threshold",
    },
    "risk": {
        "max_positions": "n_positions",
        "total_exposure_enforcer_threshold_ratio": "risk_twel_enforcer_threshold",
        "we_excess_allowance_fraction": "risk_we_excess_allowance_pct",
        "position_exposure_enforcer_threshold_ratio": "risk_wel_enforcer_threshold",
        "total_wallet_exposure_limit": "total_wallet_exposure_limit",
    },
    "unstuck": {
        "close_fraction": "unstuck_close_pct",
        "ema_distance_fraction": "unstuck_ema_dist",
        "ema_span_0_minutes": "unstuck_ema_span_0",
        "ema_span_1_minutes": "unstuck_ema_span_1",
        "loss_allowance_fraction": "unstuck_loss_allowance_pct",
        "exposure_trigger_ratio": "unstuck_threshold",
    },
}
SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY = {
    group_name: {flat_key: local_key for local_key, flat_key in field_map.items()}
    for group_name, field_map in SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.items()
}
BOT_BOUND_GROUP_BY_KEY = {
    flat_key: group_name
    for group_name, field_map in SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY.items()
    for flat_key in field_map
}

SHARED_OPTIMIZE_BOUNDS_DEFAULTS = {
    "long": {
        "forager": {
            "score_weights": {
                "ema_readiness": [0, 1, 0.01],
                "volatility": [0, 1, 0.01],
                "volume": [0, 1, 0.01],
            },
            "volatility_ema_span_minutes": [10, 2880, 1],
            "volume_ema_span_minutes": [180, 2880, 1]
        },
        "hsl": {
            "cooldown_after_red_minutes": [2160, 2160, 10],
            "drawdown_ratio_ema_span_minutes": [720, 720, 1],
            "drawdown_ratio_threshold": [0.15, 0.15, 0.001]
        },
        "entry_cooldown": {"base_duration_minutes": [0, 60, 0.1]},
        "risk": {
            "max_positions": [7, 7, 1],
            "total_wallet_exposure_limit": [1.5, 1.5, 0.01],
            "we_excess_allowance_fraction": [0, 0.4, 0.01],
            "position_exposure_enforcer": {
                'threshold_ratio': [1, 1, 0.001],
            },
            "total_exposure_enforcer": {
                'threshold_ratio': [1, 1, 0.001],
            },
        },
        "unstuck": {
            "close_fraction": [0.01, 0.12, 0.001],
            "ema_distance_fraction": [-0.2, 0.01, 0.0001],
            "ema_span_0_minutes": [60, 2880, 10],
            "ema_span_1_minutes": [60, 2880, 10],
            "loss_allowance_fraction": [0.005, 0.2, 0.0001],
            "exposure_trigger_ratio": [0.4, 0.9, 0.001],
        },
    },
    "short": {
        "forager": {
            "score_weights": {
                "ema_readiness": [0, 1, 0.01],
                "volatility": [0, 1, 0.01],
                "volume": [0, 1, 0.01],
            },
            "volatility_ema_span_minutes": [10, 1440, 1],
            "volume_ema_span_minutes": [180, 2880, 1]
        },
        "hsl": {
            "cooldown_after_red_minutes": [2160, 4320, 10],
            "drawdown_ratio_ema_span_minutes": [1, 720, 1],
            "drawdown_ratio_threshold": [0.01, 0.15, 0.001]
        },
        "entry_cooldown": {"base_duration_minutes": [0, 60, 0.1]},
        "risk": {
            "max_positions": [5, 5, 1],
            "total_wallet_exposure_limit": [0, 0],
            "we_excess_allowance_fraction": [0, 0.5, 0.01],
            "position_exposure_enforcer": {
                'threshold_ratio': [0.8, 1.01, 0.001],
            },
            "total_exposure_enforcer": {
                'threshold_ratio': [0.8, 1.01, 0.001],
            },
        },
        "unstuck": {
            "close_fraction": [0.01, 0.12, 0.001],
            "ema_distance_fraction": [-0.2, 0.01, 0.0001],
            "ema_span_0_minutes": [60, 2880, 10],
            "ema_span_1_minutes": [60, 2880, 10],
            "loss_allowance_fraction": [0.005, 0.2, 0.0001],
            "exposure_trigger_ratio": [0.4, 0.9, 0.001],
        },
    },
}


def get_optimize_bounds_defaults() -> dict:
    result = {}
    for pside in BOT_POSITION_SIDES:
        result[pside] = deepcopy(SHARED_OPTIMIZE_BOUNDS_DEFAULTS[pside])
        result[pside]["strategy"] = {
            kind: deepcopy(get_strategy_optimize_bounds(kind)[pside])
            for kind in get_supported_strategy_kinds()
        }
    return result


def flatten_optimize_bounds(bounds: dict | None, *, strategy_kind: str) -> dict:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    flat = {}
    strategy_keys = {
        "_".join(path[4:]): key.split("_", 1)[1]
        for key, path in strategy_optimize_key_path_map(normalized_kind).items()
    }
    if not isinstance(bounds, dict):
        return flat
    for key, value in bounds.get("portfolio", {}).get("unstuck", {}).items():
        flat[f"unstuck_{key}"] = deepcopy(value)
    for group in ("hsl", "loss_gate"):
        for key, value in bounds.get("portfolio", {}).get(group, {}).items():
            flat_key = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.get(group, {}).get(key, f"{group}_{key}")
            flat[flat_key] = deepcopy(value)
    for pside in BOT_POSITION_SIDES:
        side_bounds = bounds.get(pside, {})
        if not isinstance(side_bounds, dict):
            continue
        for group_name, group_bounds in side_bounds.items():
            if group_name == "strategy":
                strategy_bounds = group_bounds.get(normalized_kind, {}) if isinstance(group_bounds, dict) else {}
                if isinstance(strategy_bounds, dict):
                    for key, value in _flatten_strategy_bound_items(strategy_bounds):
                        flat[f"{pside}_{strategy_keys.get(key, key)}"] = deepcopy(value)
                continue
            if not isinstance(group_bounds, dict):
                continue
            for key, value in _flatten_strategy_bound_items(group_bounds):
                flat_key = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.get(group_name, {}).get(key, key)
                flat[f"{pside}_{flat_key}"] = deepcopy(value)
    return flat



def hydrate_adaptive_optimize_bounds(config: dict, *, source_bounds=None, tracker=None) -> None:
    """Materialize omitted optimizer dimensions as fixed effective bot values.

    The complete template is an explicit starting profile, never an authority
    for missing bounds in an authored document. ``source_bounds`` must describe
    the input before template hydration. Nullable policy remains outside the
    numeric genome until an explicit numeric domain is supplied.
    """
    from .param_paths import OPTIMIZABLE_BOT_KEY_PATHS, iter_optimizer_key_paths

    kind = config["bot"]["portfolio"]["strategy_kind"]
    bounds = config["optimize"]["bounds"]
    authored = flatten_optimize_bounds(
        bounds if source_bounds is None else source_bounds, strategy_kind=kind
    )
    keys = set(strategy_optimize_key_path_map(kind)) | set(OPTIMIZABLE_BOT_KEY_PATHS)
    for side in BOT_POSITION_SIDES:
        for fields in SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.values():
            keys.update(f"{side}_{key}" for key in fields.values())
    portfolio = config["bot"]["portfolio"]["hsl"]["scope"] == "portfolio"
    if portfolio:
        keys = {key for key in keys if not key.startswith(("long_hsl_", "short_hsl_"))}
        keys.update(SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY["hsl"].values())
    rebuilt = {}
    for key, path in iter_optimizer_key_paths(config, sorted(keys | set(authored))):
        if key in authored:
            value = authored[key]
        else:
            value = config
            for part in path or ():
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{'.'.join(path)} must be numeric for an optimizer bound")
                if not math.isfinite(float(value)):
                    raise ValueError(f"{'.'.join(path)} must be finite for an optimizer bound")
                value = [value, value]
                set_flat_optimize_bound(rebuilt, kind, key, value)
            continue
        set_flat_optimize_bound(rebuilt, kind, key, value)
    if tracker is not None and bounds != rebuilt:
        tracker.update(["optimize", "bounds"], bounds, rebuilt)
    bounds.clear()
    bounds.update(rebuilt)


def preserve_optional_adaptive_bounds(template: dict, source: dict) -> None:
    """Keep explicit search dimensions without adding them to default searches."""
    kind = get_optional_config_value(source, 'bot.portfolio.strategy_kind', "trailing_martingale")
    flat = flatten_optimize_bounds(source.get("optimize", {}).get("bounds"), strategy_kind=kind)
    optional = {
        "entry_cooldown_weights_minutes_exposure_ratio",
        "entry_cooldown_weights_minutes_adverse_directionality",
        "entry_cooldown_min_duration_minutes",
        "entry_cooldown_max_duration_minutes",
        "forager_score_weights_unilateralness",
        "unilateralness_ema_span_1m",
    }
    target = template["optimize"]["bounds"]
    for key, value in flat.items():
        if key in {"unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes"} or "hsl_memory_ema_span_minutes" in key or key.split("_", 1)[-1] in optional:
            set_flat_optimize_bound(target, kind, key, value)
    # Input also accepts nested score_weights before canonical bound sorting.
    for side in BOT_POSITION_SIDES:
        scoring = (
            source.get("optimize", {})
            .get("bounds", {})
            .get(side, {})
            .get("forager", {})
            .get("score_weights", {})
        )
        if isinstance(scoring, dict) and "unilateralness" in scoring:
            target[side]["forager"].setdefault("score_weights", {})["unilateralness"] = deepcopy(
                scoring["unilateralness"]
            )


def _shared_bound_local_path(group: str, local_key: str) -> tuple[str, ...]:
    for owner, prefix in (("entry_cooldown", "weights_minutes"), ("forager", "score_weights")):
        if group == owner and local_key.startswith(prefix + "_"):
            return (prefix, local_key.removeprefix(prefix + "_"))
    if group == "risk":
        for owner in ("position_exposure_enforcer", "total_exposure_enforcer"):
            if local_key.startswith(owner + "_"):
                return (owner, local_key.removeprefix(owner + "_"))
    return (local_key,)


def set_flat_optimize_bound(bounds: dict, strategy_kind: str, flat_key: str, value) -> None:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    if flat_key == "unstuck_memory_ema_span_minutes":
        bounds.setdefault("portfolio", {}).setdefault("unstuck", {})["memory_ema_span_minutes"] = deepcopy(value)
        return
    if flat_key == "loss_gate_memory_ema_span_minutes":
        bounds.setdefault("portfolio", {}).setdefault("loss_gate", {})["memory_ema_span_minutes"] = deepcopy(value)
        return
    if flat_key.startswith("hsl_"):
        local_key = SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY["hsl"].get(flat_key, flat_key.removeprefix("hsl_"))
        bounds.setdefault("portfolio", {}).setdefault("hsl", {})[local_key] = deepcopy(value)
        return
    pside, key = flat_key.split("_", 1)
    if pside not in BOT_POSITION_SIDES:
        raise KeyError(flat_key)
    side_bounds = bounds.setdefault(pside, {})
    group = BOT_BOUND_GROUP_BY_KEY.get(key)
    if group is None:
        strategy_root = side_bounds.setdefault("strategy", {})
        current = strategy_root.setdefault(normalized_kind, {})
        canonical_path = strategy_optimize_key_path_map(normalized_kind).get(flat_key)
        if canonical_path is not None:
            for part in canonical_path[4:-1]:
                current = current.setdefault(part, {})
            current[canonical_path[-1]] = deepcopy(value)
            return
        parts = key.split("_")
        if parts[0] in {"entry", "close"} and len(parts) > 1:
            current = current.setdefault(parts[0], {})
            current["_".join(parts[1:])] = deepcopy(value)
        else:
            current[key] = deepcopy(value)
    else:
        local_key = SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY[group].get(key, key)
        target = side_bounds.setdefault(group, {})
        path = _shared_bound_local_path(group, local_key)
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = deepcopy(value)


def sort_optimize_bounds_in_place(bounds: dict, *, strategy_kind: str, portfolio_hsl: bool = False) -> None:
    flat = flatten_optimize_bounds(bounds, strategy_kind=strategy_kind)
    normalized_kind = normalize_strategy_kind(strategy_kind)
    for key, value in list(flat.items()):
        if isinstance(value, list):
            if len(value) == 1:
                flat[key] = [value[0], value[0]]
            elif len(value) == 2:
                flat[key] = sorted(value)
    # Sorting must never authorize a search dimension omitted by the author.
    rebuilt = {}
    for flat_key, value in flat.items():
        set_flat_optimize_bound(rebuilt, normalized_kind, flat_key, value)
    bounds.clear()
    bounds.update(rebuilt)


def prune_inactive_optimize_strategy_bounds(config: dict, *, tracker: Optional[object] = None) -> None:
    active_kind = normalize_strategy_kind(get_optional_config_value(config, 'bot.portfolio.strategy_kind'))
    bounds = config.get("optimize", {}).get("bounds")
    if not isinstance(bounds, dict):
        return
    for pside in BOT_POSITION_SIDES:
        side_bounds = bounds.get(pside)
        if not isinstance(side_bounds, dict):
            continue
        strategy_bounds = side_bounds.get("strategy")
        if not isinstance(strategy_bounds, dict):
            continue
        for kind in list(strategy_bounds):
            if kind == active_kind:
                continue
            removed = strategy_bounds.pop(kind)
            if tracker is not None:
                tracker.remove(["optimize", "bounds", pside, "strategy", kind], removed)
