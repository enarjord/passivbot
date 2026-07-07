from copy import deepcopy

from config.access import require_config_value
from config.shared_bot import BOT_SHARED_GROUPS
from config.strategy import DEFAULT_STRATEGY_KIND, normalize_strategy_kind

TRAILING_GRID_V7_STRATEGY_KIND = "trailing_grid_v7"

KNOWN_OPTIMIZER_OVERRIDES = frozenset(
    {
        "backward_tp_grid",
        "forward_tp_grid",
        "lossless_close_trailing",
        "mirror_short_from_long",
    }
)


def _mirror_short_from_long(config):
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))

    bot_cfg = config.get("bot", {})
    long_side = bot_cfg.get("long")
    short_side = bot_cfg.get("short")
    if not isinstance(long_side, dict) or not isinstance(short_side, dict):
        return config

    for group_name in BOT_SHARED_GROUPS:
        if group_name in long_side:
            short_side[group_name] = deepcopy(long_side[group_name])

    long_strategy = long_side.get("strategy")
    if isinstance(long_strategy, dict) and strategy_kind in long_strategy:
        short_strategy = short_side.setdefault("strategy", {})
        short_strategy[strategy_kind] = deepcopy(long_strategy[strategy_kind])

    return config


def validate_optimizer_overrides(overrides_list):
    unknown = sorted(
        {override for override in overrides_list or [] if override not in KNOWN_OPTIMIZER_OVERRIDES}
    )
    if unknown:
        allowed = ", ".join(sorted(KNOWN_OPTIMIZER_OVERRIDES))
        raise ValueError(
            f"Unknown optimize.enable_overrides value(s): {', '.join(unknown)}. "
            f"Allowed values: {allowed}"
        )


def _require_trailing_martingale_side(config, pside):
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    if strategy_kind != DEFAULT_STRATEGY_KIND:
        raise ValueError(
            f"optimizer override requires live.strategy_kind = {DEFAULT_STRATEGY_KIND!r}; got {strategy_kind!r}"
        )
    strategy_cfg = require_config_value(config, f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}")
    if not isinstance(strategy_cfg, dict):
        raise TypeError(
            f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND} must be a dict; got {type(strategy_cfg).__name__}"
        )
    return strategy_cfg


def _require_trailing_grid_v7_side(config, pside):
    strategy_kind = normalize_strategy_kind(config.get("live", {}).get("strategy_kind"))
    if strategy_kind != TRAILING_GRID_V7_STRATEGY_KIND:
        raise ValueError(
            "optimizer override requires live.strategy_kind = "
            f"{TRAILING_GRID_V7_STRATEGY_KIND!r}; got {strategy_kind!r}"
        )
    strategy_cfg = require_config_value(
        config,
        f"bot.{pside}.strategy.{TRAILING_GRID_V7_STRATEGY_KIND}",
    )
    if not isinstance(strategy_cfg, dict):
        raise TypeError(
            f"bot.{pside}.strategy.{TRAILING_GRID_V7_STRATEGY_KIND} must be a dict; "
            f"got {type(strategy_cfg).__name__}"
        )
    return strategy_cfg


def _set_trailing_grid_v7_tp_grid_order(config, pside, *, forward: bool):
    strategy_cfg = _require_trailing_grid_v7_side(config, pside)
    start = require_config_value(
        config,
        f"bot.{pside}.strategy.{TRAILING_GRID_V7_STRATEGY_KIND}.close.grid_markup_start",
    )
    end = require_config_value(
        config,
        f"bot.{pside}.strategy.{TRAILING_GRID_V7_STRATEGY_KIND}.close.grid_markup_end",
    )
    lower = min(start, end)
    upper = max(start, end)
    close_cfg = strategy_cfg.setdefault("close", {})
    close_cfg["grid_markup_start"] = lower if forward else upper
    close_cfg["grid_markup_end"] = upper if forward else lower
    return config


def optimizer_overrides(overrides_list, config, pside=None):
    if not overrides_list:
        # No overrides to apply
        return config

    validate_optimizer_overrides(overrides_list)

    for override in overrides_list:
        if override == "mirror_short_from_long":
            if pside is None:
                config = _mirror_short_from_long(config)
            continue

        if pside is None:
            continue

        if override == "lossless_close_trailing":
            strategy_cfg = _require_trailing_martingale_side(config, pside)
            threshold = require_config_value(
                config,
                f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close.threshold_base_pct",
            )
            retracement = require_config_value(
                config,
                f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close.retracement_base_pct",
            )
            strategy_cfg.setdefault("close", {})["threshold_base_pct"] = max(threshold, retracement)

        elif override == "forward_tp_grid":
            config = _set_trailing_grid_v7_tp_grid_order(config, pside, forward=True)

        elif override == "backward_tp_grid":
            config = _set_trailing_grid_v7_tp_grid_order(config, pside, forward=False)

    return config
