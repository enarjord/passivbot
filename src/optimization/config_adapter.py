"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

from typing import List, Tuple

from config_utils import get_template_config
from optimization.bounds import Bound


OPTIMIZABLE_COMMON_KEY_PATHS = {
    "common_equity_hard_stop_loss_cooldown_minutes_after_red": (
        "bot",
        "common",
        "equity_hard_stop_loss",
        "cooldown_minutes_after_red",
    ),
    "common_equity_hard_stop_loss_ema_span_minutes": (
        "bot",
        "common",
        "equity_hard_stop_loss",
        "ema_span_minutes",
    ),
    "common_equity_hard_stop_loss_no_restart_drawdown_threshold": (
        "bot",
        "common",
        "equity_hard_stop_loss",
        "no_restart_drawdown_threshold",
    ),
    "common_equity_hard_stop_loss_red_threshold": (
        "bot",
        "common",
        "equity_hard_stop_loss",
        "red_threshold",
    ),
}


def get_optimization_key_paths(config) -> List[Tuple[str, Tuple[str, ...]]]:
    key_paths: List[Tuple[str, Tuple[str, ...]]] = []
    bot_config = config.get("bot")
    if bot_config is None:
        bot_config = get_template_config()["bot"]
    for pside in ("long", "short"):
        pside_config = bot_config.get(pside)
        if not isinstance(pside_config, dict):
            continue
        for key in sorted(pside_config):
            key_paths.append((f"{pside}_{key}", ("bot", pside, key)))
    for bound_key in sorted(OPTIMIZABLE_COMMON_KEY_PATHS):
        if bound_key in config.get("optimize", {}).get("bounds", {}):
            key_paths.append((bound_key, OPTIMIZABLE_COMMON_KEY_PATHS[bound_key]))
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
        if not isinstance(bot_config.get(pside), dict):
            pside_enabled[pside] = False
            continue
        required_keys = [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
        if not all(key in optimize_bounds for key in required_keys):
            pside_enabled[pside] = True
            continue
        pside_enabled[pside] = all(
            Bound.from_config(key, optimize_bounds[key]).high > 0.0 for key in required_keys
        )

    for bound_key, path in key_paths:
        assert bound_key in optimize_bounds, f"bound {bound_key} missing from optimize.bounds"
        bound_vals = Bound.from_config(bound_key, optimize_bounds[bound_key])
        if len(path) >= 2 and path[:2] == ("bot", "common"):
            bounds.append(bound_vals)
            continue
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
