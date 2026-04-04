from copy import deepcopy
from typing import Optional


BOT_POSITION_SIDES = ("long", "short")
BOT_SHARED_GROUPS = ("risk", "forager", "hsl", "unstuck")

BOT_GROUP_FIELD_MAP = {
    "risk": {
        "n_positions": "n_positions",
        "total_wallet_exposure_limit": "total_wallet_exposure_limit",
        "twel_enforcer_threshold": "risk_twel_enforcer_threshold",
        "we_excess_allowance_pct": "risk_we_excess_allowance_pct",
        "wel_enforcer_threshold": "risk_wel_enforcer_threshold",
    },
    "forager": {
        "score_weights": "forager_score_weights",
        "volatility_ema_span": "forager_volatility_ema_span",
        "volume_drop_pct": "forager_volume_drop_pct",
        "volume_ema_span": "forager_volume_ema_span",
    },
    "hsl": {
        "cooldown_minutes_after_red": "hsl_cooldown_minutes_after_red",
        "ema_span_minutes": "hsl_ema_span_minutes",
        "enabled": "hsl_enabled",
        "no_restart_drawdown_threshold": "hsl_no_restart_drawdown_threshold",
        "orange_tier_mode": "hsl_orange_tier_mode",
        "panic_close_order_type": "hsl_panic_close_order_type",
        "red_threshold": "hsl_red_threshold",
        "tier_ratios": "hsl_tier_ratios",
    },
    "unstuck": {
        "close_pct": "unstuck_close_pct",
        "ema_dist": "unstuck_ema_dist",
        "loss_allowance_pct": "unstuck_loss_allowance_pct",
        "threshold": "unstuck_threshold",
    },
}

FLAT_BOT_KEY_TO_GROUP_PATH = {
    flat_key: (group_name, local_key)
    for group_name, field_map in BOT_GROUP_FIELD_MAP.items()
    for local_key, flat_key in field_map.items()
}


def get_bot_group(bot_side: dict | None, group_name: str) -> dict:
    if not isinstance(bot_side, dict):
        return {}
    group = bot_side.get(group_name)
    if isinstance(group, dict):
        return group
    return {}


def get_grouped_bot_value(bot_side: dict | None, flat_key: str, default=None):
    if not isinstance(bot_side, dict):
        return default
    group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
    if group_path is not None:
        group_name, local_key = group_path
        group = get_bot_group(bot_side, group_name)
        if local_key in group:
            return group[local_key]
    return bot_side.get(flat_key, default)


def flatten_shared_bot_side(bot_side: dict | None) -> dict:
    if not isinstance(bot_side, dict):
        return {}
    result = {}
    for flat_key in FLAT_BOT_KEY_TO_GROUP_PATH:
        value = get_grouped_bot_value(bot_side, flat_key, default=None)
        if flat_key in bot_side or value is not None:
            result[flat_key] = deepcopy(value)
    for key, value in bot_side.items():
        if key in result or key in BOT_SHARED_GROUPS or key == "strategy":
            continue
        result[key] = deepcopy(value)
    return result


def inject_flattened_shared_bot_side(bot_side: dict | None) -> None:
    if not isinstance(bot_side, dict):
        return
    for flat_key, value in flatten_shared_bot_side(bot_side).items():
        bot_side.setdefault(flat_key, deepcopy(value))


def canonical_shared_bot_path_for_flat_key(pside: str, flat_key: str) -> tuple[str, ...] | None:
    group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
    if group_path is None:
        return None
    group_name, local_key = group_path
    return ("bot", pside, group_name, local_key)


def resolve_shared_bot_path(bot_side: dict | None, pside: str, flat_key: str) -> tuple[str, ...] | None:
    group_path = FLAT_BOT_KEY_TO_GROUP_PATH.get(flat_key)
    if isinstance(bot_side, dict) and group_path is not None:
        group_name, local_key = group_path
        group_cfg = bot_side.get(group_name)
        if isinstance(group_cfg, dict) and local_key in group_cfg:
            return ("bot", pside, group_name, local_key)
    if isinstance(bot_side, dict) and flat_key in bot_side:
        return ("bot", pside, flat_key)
    return canonical_shared_bot_path_for_flat_key(pside, flat_key)


def canonicalize_shared_bot_side(
    bot_side: dict | None,
    *,
    path_prefix: tuple[str, ...] = (),
    tracker: Optional[object] = None,
    seed_missing_groups: bool = False,
) -> None:
    if not isinstance(bot_side, dict):
        return
    for group_name in BOT_SHARED_GROUPS:
        group_cfg = bot_side.get(group_name)
        if group_cfg is None:
            if seed_missing_groups:
                bot_side[group_name] = {}
                if tracker is not None:
                    tracker.add([*path_prefix, group_name], {})
            continue
        if not isinstance(group_cfg, dict):
            raise TypeError(
                f"{'.'.join(path_prefix + (group_name,))} must be a dict; "
                f"got {type(group_cfg).__name__}"
            )
    for flat_key, (group_name, local_key) in FLAT_BOT_KEY_TO_GROUP_PATH.items():
        if flat_key not in bot_side:
            continue
        moved_value = bot_side.pop(flat_key)
        group_cfg = bot_side.get(group_name)
        if group_cfg is None:
            group_cfg = {}
            bot_side[group_name] = group_cfg
            if tracker is not None:
                tracker.add([*path_prefix, group_name], {})
        if local_key not in group_cfg:
            group_cfg[local_key] = moved_value
            if tracker is not None:
                tracker.rename(
                    [*path_prefix, flat_key],
                    [*path_prefix, group_name, local_key],
                    moved_value,
                )
        elif tracker is not None:
            tracker.remove([*path_prefix, flat_key], moved_value)
