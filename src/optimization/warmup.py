from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Sequence

from config.bot import normalize_forager_score_weights
from optimizer_overrides import optimizer_overrides
from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths
from warmup_utils import compute_per_coin_warmup_minutes


def _apply_config_overrides(config: dict, overrides: dict) -> None:
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            continue
        parts = dotted_path.split(".")
        if not parts:
            continue
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value


def build_optimizer_vector_config(
    vector: Sequence[float],
    template: dict,
    *,
    key_paths=None,
    overrides_list=None,
) -> dict:
    config = deepcopy(template)
    if key_paths is None:
        key_paths = get_optimization_key_paths(config)
    assert len(vector) == len(key_paths), (
        f"individual length {len(vector)} does not match optimization key count {len(key_paths)}"
    )
    for value, (_, path) in zip(vector, key_paths):
        target = config
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    _apply_config_overrides(
        config,
        config.get("optimize", {}).get("fixed_runtime_overrides", {}),
    )
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


def build_optimizer_max_config(config: dict) -> dict:
    bounds = extract_bounds_tuple_list_from_config(config)
    if not bounds:
        return deepcopy(config)
    key_paths = get_optimization_key_paths(config)
    overrides_list = config.get("optimize", {}).get("enable_overrides", []) or []
    max_vector = [bound.high for bound in bounds]
    return build_optimizer_vector_config(
        max_vector,
        config,
        key_paths=key_paths,
        overrides_list=overrides_list,
    )


def compute_optimizer_per_coin_warmup_minutes(config: dict) -> dict:
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
