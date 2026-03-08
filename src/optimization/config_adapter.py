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
        for key in sorted(bot_config[pside]):
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
    for pside in ("long", "short"):
        is_enabled = all(
            [
                Bound.from_config(k, optimize_bounds[k]).high > 0.0
                for k in [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
            ]
        )
        for bound_key, path in key_paths:
            if path[:2] != ("bot", pside):
                continue
            assert bound_key in optimize_bounds, f"bound {bound_key} missing from optimize.bounds"
            bound_vals = Bound.from_config(bound_key, optimize_bounds[bound_key])
            if is_enabled:
                bounds.append(bound_vals)
            else:
                # Disabled: fix to low value, preserve step for consistency
                bounds.append(Bound(bound_vals.low, bound_vals.low, bound_vals.step))
    for bound_key, path in key_paths:
        if path[:2] != ("bot", "common"):
            continue
        assert bound_key in optimize_bounds, f"bound {bound_key} missing from optimize.bounds"
        bounds.append(Bound.from_config(bound_key, optimize_bounds[bound_key]))
    return bounds
