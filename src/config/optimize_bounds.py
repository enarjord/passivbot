from copy import deepcopy
from typing import Optional

from .strategy_spec import (
    BOT_POSITION_SIDES,
    get_supported_strategy_kinds,
    get_strategy_optimize_bounds,
    normalize_strategy_kind,
)


def _flatten_strategy_bound_items(bounds: dict, prefix: tuple[str, ...] = ()):
    for key, value in bounds.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _flatten_strategy_bound_items(value, path)
        else:
            yield "_".join(path), value


SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY = {
    "forager": {
        "score_weights_ema_readiness": "forager_score_weights_ema_readiness",
        "score_weights_volatility": "forager_score_weights_volatility",
        "score_weights_volume": "forager_score_weights_volume",
        "volatility_ema_span_1m": "forager_volatility_ema_span_1m",
        "volume_drop_pct": "forager_volume_drop_pct",
        "volume_ema_span_1m": "forager_volume_ema_span_1m",
    },
    "hsl": {
        "cooldown_minutes_after_red": "hsl_cooldown_minutes_after_red",
        "ema_span_minutes": "hsl_ema_span_minutes",
        "red_threshold": "hsl_red_threshold",
    },
    "risk": {
        "entry_cooldown_minutes": "risk_entry_cooldown_minutes",
        "n_positions": "n_positions",
        "total_exposure_enforcer_threshold": "risk_twel_enforcer_threshold",
        "we_excess_allowance_pct": "risk_we_excess_allowance_pct",
        "position_exposure_enforcer_threshold": "risk_wel_enforcer_threshold",
        "total_wallet_exposure_limit": "total_wallet_exposure_limit",
    },
    "unstuck": {
        "close_pct": "unstuck_close_pct",
        "ema_dist": "unstuck_ema_dist",
        "loss_allowance_pct": "unstuck_loss_allowance_pct",
        "threshold": "unstuck_threshold",
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
            "score_weights_ema_readiness": [0, 1, 0.01],
            "score_weights_volatility": [0, 1, 0.01],
            "score_weights_volume": [0, 1, 0.01],
            "volatility_ema_span_1m": [10, 720, 1],
            "volume_drop_pct": [0.4, 1, 0.01],
            "volume_ema_span_1m": [360, 2880, 10],
        },
        "hsl": {
            "cooldown_minutes_after_red": [1, 2880, 10],
            "ema_span_minutes": [1, 2880, 10],
            "red_threshold": [0.01, 0.12, 0.001],
        },
        "risk": {
            "entry_cooldown_minutes": [0.0, 60.0, 0.1],
            "n_positions": [10, 10, 1],
            "total_exposure_enforcer_threshold": [0.95, 1.01, 0.001],
            "we_excess_allowance_pct": [0, 0.3, 0.01],
            "position_exposure_enforcer_threshold": [0.95, 1.01, 0.001],
            "total_wallet_exposure_limit": [1.25, 1.25],
        },
        "unstuck": {
            "close_pct": [0.05, 0.12, 0.001],
            "ema_dist": [-0.2, -0.07, 0.0001],
            "loss_allowance_pct": [0.005, 0.025, 0.0001],
            "threshold": [0.4, 0.9, 0.001],
        },
    },
    "short": {
        "forager": {
            "score_weights_ema_readiness": [0, 1, 0.01],
            "score_weights_volatility": [0, 1, 0.01],
            "score_weights_volume": [0, 1, 0.01],
            "volatility_ema_span_1m": [10, 720, 1],
            "volume_drop_pct": [0.4, 1, 0.01],
            "volume_ema_span_1m": [360, 2880, 10],
        },
        "hsl": {
            "cooldown_minutes_after_red": [1, 2880, 10],
            "ema_span_minutes": [1, 2880, 10],
            "red_threshold": [0.01, 0.12, 0.001],
        },
        "risk": {
            "entry_cooldown_minutes": [0.0, 60.0, 0.1],
            "n_positions": [10, 10, 1],
            "total_exposure_enforcer_threshold": [0.95, 1.01, 0.001],
            "we_excess_allowance_pct": [0, 0.3, 0.01],
            "position_exposure_enforcer_threshold": [0.95, 1.01, 0.001],
            "total_wallet_exposure_limit": [0.0, 0.0],
        },
        "unstuck": {
            "close_pct": [0.05, 0.12, 0.001],
            "ema_dist": [-0.2, -0.07, 0.0001],
            "loss_allowance_pct": [0.005, 0.025, 0.0001],
            "threshold": [0.4, 0.9, 0.001],
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
    if not isinstance(bounds, dict):
        return flat
    if any(
        isinstance(key, str) and (key.startswith("long_") or key.startswith("short_"))
        for key in bounds
    ):
        return deepcopy(bounds)
    for pside in BOT_POSITION_SIDES:
        side_bounds = bounds.get(pside, {})
        if not isinstance(side_bounds, dict):
            continue
        for group_name, group_bounds in side_bounds.items():
            if group_name == "strategy":
                strategy_bounds = group_bounds.get(normalized_kind, {}) if isinstance(group_bounds, dict) else {}
                if isinstance(strategy_bounds, dict):
                    for key, value in _flatten_strategy_bound_items(strategy_bounds):
                        flat[f"{pside}_{key}"] = deepcopy(value)
                continue
            if not isinstance(group_bounds, dict):
                continue
            for key, value in group_bounds.items():
                flat_key = SHARED_OPTIMIZE_LOCAL_TO_FLAT_KEY.get(group_name, {}).get(key, key)
                flat[f"{pside}_{flat_key}"] = deepcopy(value)
    return flat


def set_flat_optimize_bound(bounds: dict, strategy_kind: str, flat_key: str, value) -> None:
    normalized_kind = normalize_strategy_kind(strategy_kind)
    pside, key = flat_key.split("_", 1)
    if pside not in BOT_POSITION_SIDES:
        raise KeyError(flat_key)
    side_bounds = bounds.setdefault(pside, {})
    group = BOT_BOUND_GROUP_BY_KEY.get(key)
    if group is None:
        strategy_root = side_bounds.setdefault("strategy", {})
        current = strategy_root.setdefault(normalized_kind, {})
        parts = key.split("_")
        if parts[0] in {"entry", "close"} and len(parts) > 1:
            current = current.setdefault(parts[0], {})
            current["_".join(parts[1:])] = deepcopy(value)
        else:
            current[key] = deepcopy(value)
    else:
        local_key = SHARED_OPTIMIZE_FLAT_TO_LOCAL_KEY[group].get(key, key)
        side_bounds.setdefault(group, {})[local_key] = deepcopy(value)


def sort_optimize_bounds_in_place(bounds: dict, *, strategy_kind: str) -> None:
    flat = flatten_optimize_bounds(bounds, strategy_kind=strategy_kind)
    normalized_kind = normalize_strategy_kind(strategy_kind)
    for key, value in list(flat.items()):
        if isinstance(value, list):
            if len(value) == 1:
                flat[key] = [value[0], value[0]]
            elif len(value) == 2:
                flat[key] = sorted(value)
    rebuilt = get_optimize_bounds_defaults()
    for pside in BOT_POSITION_SIDES:
        rebuilt[pside]["strategy"] = {normalized_kind: deepcopy(rebuilt[pside]["strategy"][normalized_kind])}
    for flat_key, value in flat.items():
        set_flat_optimize_bound(rebuilt, normalized_kind, flat_key, value)
    bounds.clear()
    bounds.update(rebuilt)


def prune_inactive_optimize_strategy_bounds(config: dict, *, tracker: Optional[object] = None) -> None:
    active_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
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
