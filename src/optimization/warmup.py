from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Sequence

from config.bot import normalize_forager_score_weights
from config.param_paths import resolve_dotted_config_path
from config.shared_bot import flatten_shared_bot_side
from optimizer_overrides import optimizer_overrides
from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths
from optimization.fine_tune_anchors import ANCHOR_PLAN_KEY, get_anchor_plan
from optimization.shape import build_optimization_shape
from warmup_utils import compute_per_coin_warmup_minutes


def _refresh_shared_bot_runtime_aliases(config: dict) -> None:
    bot_cfg = config.get("bot")
    if not isinstance(bot_cfg, dict):
        return
    for pside in ("long", "short"):
        pside_cfg = bot_cfg.get(pside)
        if not isinstance(pside_cfg, dict):
            continue
        for key, value in flatten_shared_bot_side(pside_cfg).items():
            pside_cfg[key] = deepcopy(value)


def _apply_config_overrides(config: dict, overrides: dict) -> None:
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            continue
        resolved = resolve_dotted_config_path(config, dotted_path)
        if resolved is None:
            continue
        parts = list(resolved)
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    _refresh_shared_bot_runtime_aliases(config)


def _set_path(config: dict, path: Sequence[str], value) -> None:
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def _finalize_optimizer_vector_config(config: dict, overrides_list=None) -> dict:
    config.pop(ANCHOR_PLAN_KEY, None)
    _apply_config_overrides(
        config,
        config.get("optimize", {}).get("fixed_runtime_overrides", {}),
    )
    config = optimizer_overrides(overrides_list or [], config, None)
    for pside in ("long", "short"):
        pside_cfg = config.get("bot", {}).get(pside, {})
        if not isinstance(pside_cfg, dict):
            continue
        red_threshold = pside_cfg.get("hsl_red_threshold")
        no_restart = pside_cfg.get("hsl_no_restart_drawdown_threshold")
        if red_threshold is not None and no_restart is not None:
            if float(no_restart) < float(red_threshold):
                pside_cfg["hsl_no_restart_drawdown_threshold"] = float(red_threshold)
    for pside in sorted(config.get("bot", {})):
        config = optimizer_overrides(overrides_list or [], config, pside)
    for pside in ("long", "short"):
        pside_cfg = config.get("bot", {}).get(pside, {})
        if not isinstance(pside_cfg, dict) or "forager_score_weights" not in pside_cfg:
            continue
        pside_cfg["forager_score_weights"] = normalize_forager_score_weights(
            pside_cfg["forager_score_weights"],
            path=f"bot.{pside}.forager_score_weights",
        )
    return config


def _build_anchored_optimizer_vector_config(
    vector: Sequence[float],
    template: dict,
    *,
    overrides_list=None,
) -> dict:
    plan = get_anchor_plan(template)
    if plan is None:
        raise ValueError("anchored fine-tune plan is missing anchors")
    key_paths = [tuple(item) for item in plan.get("key_paths") or []]
    expected_len = 1 + len(key_paths)
    assert len(vector) == expected_len, (
        f"anchored individual length {len(vector)} does not match expected {expected_len}"
    )
    anchors = plan["anchors"]
    anchor_id = int(round(float(vector[0])))
    anchor_id = max(0, min(len(anchors) - 1, anchor_id))
    anchor = anchors[anchor_id]
    config = deepcopy(template)
    config.pop(ANCHOR_PLAN_KEY, None)
    for item in anchor.get("fixed_values") or []:
        path = tuple(item.get("path") or ())
        if not path:
            continue
        _set_path(config, path, deepcopy(item.get("value")))
    for value, path in zip(vector[1:], key_paths):
        _set_path(config, path, value)
    config["_optimizer_anchor"] = {
        "id": anchor_id,
        "source": anchor.get("source"),
        "tunable_keys": list(plan.get("tunable_keys") or []),
        "fixed_keys": list(plan.get("fixed_keys") or []),
    }
    return _finalize_optimizer_vector_config(config, overrides_list=overrides_list)


def build_optimizer_vector_config(
    vector: Sequence[float],
    template: dict,
    *,
    key_paths=None,
    overrides_list=None,
) -> dict:
    if get_anchor_plan(template) is not None:
        return _build_anchored_optimizer_vector_config(
            vector,
            template,
            overrides_list=overrides_list,
        )
    config = deepcopy(template)
    if key_paths is None:
        key_paths = get_optimization_key_paths(config)
    assert len(vector) == len(key_paths), (
        f"individual length {len(vector)} does not match optimization key count {len(key_paths)}"
    )
    for value, (_, path) in zip(vector, key_paths):
        _set_path(config, path, value)
    return _finalize_optimizer_vector_config(config, overrides_list=overrides_list)


def build_optimizer_max_config(config: dict) -> dict:
    bounds = extract_bounds_tuple_list_from_config(config)
    if not bounds:
        return deepcopy(config)
    overrides_list = config.get("optimize", {}).get("enable_overrides", []) or []
    shape = build_optimization_shape(config)
    max_vector = [bound.high for bound in shape.bounds]
    return build_optimizer_vector_config(
        max_vector,
        config,
        key_paths=shape.key_paths,
        overrides_list=overrides_list,
    )


def compute_optimizer_per_coin_warmup_minutes(config: dict) -> dict:
    anchor_plan = get_anchor_plan(config)
    if anchor_plan is not None:
        shape = build_optimization_shape(config)
        overrides_list = config.get("optimize", {}).get("enable_overrides", []) or []
        tunable_max_vector = [bound.high for bound in shape.bounds[1:]]
        merged: dict[str, int] = {}
        for anchor_id in range(len(anchor_plan.get("anchors") or [])):
            anchor_config = build_optimizer_vector_config(
                [float(anchor_id), *tunable_max_vector],
                config,
                key_paths=shape.key_paths,
                overrides_list=overrides_list,
            )
            for key, value in compute_per_coin_warmup_minutes(anchor_config).items():
                merged[key] = max(int(value), int(merged.get(key, 0)))
        return merged
    return compute_per_coin_warmup_minutes(build_optimizer_max_config(config))


def compute_optimizer_backtest_warmup_minutes(config: dict) -> int:
    warmup_map = compute_optimizer_per_coin_warmup_minutes(config)
    return max((int(value) for value in warmup_map.values()), default=0)


def stamp_warmup_metadata(mss: dict, coins: Sequence[str], warmup_map: dict) -> Counter:
    default_warmup = int(warmup_map.get("__default__", 0))
    stamped: Counter = Counter()
    for coin in coins:
        meta = mss.get(coin)
        if not isinstance(meta, dict):
            continue
        warmup_minutes = int(warmup_map.get(coin, default_warmup))
        first_idx = int(meta.get("first_valid_index", 0))
        last_idx = int(meta.get("last_valid_index", 0))
        if first_idx > last_idx:
            trade_start = first_idx
        else:
            trade_start = min(last_idx, first_idx + warmup_minutes)
        meta["warmup_minutes"] = warmup_minutes
        meta["trade_start_index"] = trade_start
        stamped[(warmup_minutes, trade_start)] += 1
    return stamped


__all__ = [
    "build_optimizer_max_config",
    "build_optimizer_vector_config",
    "compute_optimizer_backtest_warmup_minutes",
    "compute_optimizer_per_coin_warmup_minutes",
    "stamp_warmup_metadata",
]
