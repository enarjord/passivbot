from copy import deepcopy

from config.access import require_config_value
from config.shared_bot import BOT_SHARED_GROUPS
from config.strategy import DEFAULT_STRATEGY_KIND, normalize_strategy_kind


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


def _require_trailing_grid_side(config, pside):
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


def optimizer_overrides(overrides_list, config, pside=None):
    if not overrides_list:
        # No overrides to apply
        return config

    for override in overrides_list:
        if override == "mirror_short_from_long":
            if pside is None:
                config = _mirror_short_from_long(config)
            continue

        if pside is None:
            continue

        if override == "lossless_close_trailing":
            strategy_cfg = _require_trailing_grid_side(config, pside)
            threshold = require_config_value(
                config,
                f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_trailing_threshold_pct",
            )
            retracement = require_config_value(
                config,
                f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_trailing_retracement_pct",
            )
            strategy_cfg["close_trailing_threshold_pct"] = max(threshold, retracement)

        elif override == "forward_tp_grid":
            strategy_cfg = _require_trailing_grid_side(config, pside)
            close_grid_markup_start = require_config_value(
                config, f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_grid_markup_start"
            )
            close_grid_markup_end = require_config_value(
                config, f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_grid_markup_end"
            )

            strategy_cfg["close_grid_markup_start"] = min(close_grid_markup_start, close_grid_markup_end)
            strategy_cfg["close_grid_markup_end"] = max(close_grid_markup_start, close_grid_markup_end)

        elif override == "backward_tp_grid":
            strategy_cfg = _require_trailing_grid_side(config, pside)
            close_grid_markup_start = require_config_value(
                config, f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_grid_markup_start"
            )
            close_grid_markup_end = require_config_value(
                config, f"bot.{pside}.strategy.{DEFAULT_STRATEGY_KIND}.close_grid_markup_end"
            )

            strategy_cfg["close_grid_markup_start"] = max(close_grid_markup_start, close_grid_markup_end)
            strategy_cfg["close_grid_markup_end"] = min(close_grid_markup_start, close_grid_markup_end)

        elif override == "example":
            # Logic for override 'example'
            pass

        else:
            print(f"Unknown override: {override}")
            return config

    return config
