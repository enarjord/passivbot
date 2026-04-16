import logging
from copy import deepcopy
from typing import Optional

from config.transform_log import ConfigTransformTracker


LEGACY_FILTER_KEYS = {
    "filter_volatility_ema_span": "forager_volatility_ema_span",
    "filter_noisiness_rolling_window": "forager_volatility_ema_span",
    "filter_noisiness_ema_span": "forager_volatility_ema_span",
    "filter_log_range_ema_span": "forager_volatility_ema_span",
    "filter_volume_ema_span": "forager_volume_ema_span",
    "filter_volume_rolling_window": "forager_volume_ema_span",
}

LEGACY_FORAGER_KEYS = {
    "filter_volume_drop_pct": "forager_volume_drop_pct",
}

OBSOLETE_BOT_KEYS = {
    "filter_volatility_drop_pct",
}

LEGACY_ENTRY_GRID_KEYS = {
    "ddown_factor": "entry_grid_double_down_factor",
    "initial_eprice_ema_dist": "entry_initial_ema_dist",
    "initial_qty_pct": "entry_initial_qty_pct",
    "rentry_pprice_dist": "entry_grid_spacing_pct",
    "rentry_pprice_dist_wallet_exposure_weighting": "entry_grid_spacing_we_weight",
    "entry_grid_spacing_weight": "entry_grid_spacing_we_weight",
    "entry_grid_spacing_log_span_hours": "entry_volatility_ema_span_hours",
    "entry_log_range_ema_span_hours": "entry_volatility_ema_span_hours",
    "entry_grid_spacing_log_weight": "entry_grid_spacing_volatility_weight",
    "entry_trailing_retracement_log_weight": "entry_trailing_retracement_volatility_weight",
    "entry_trailing_threshold_log_weight": "entry_trailing_threshold_volatility_weight",
}

LEGACY_BOUNDS_KEYS = {
    "long_min_markup": "long_close_grid_markup_start",
    "short_min_markup": "short_close_grid_markup_start",
    "long_close_grid_min_markup": "long_close_grid_markup_end",
    "short_close_grid_min_markup": "short_close_grid_markup_end",
    "long_filter_volatility_ema_span": "long_forager_volatility_ema_span",
    "long_filter_noisiness_rolling_window": "long_forager_volatility_ema_span",
    "long_filter_noisiness_ema_span": "long_forager_volatility_ema_span",
    "long_filter_volume_rolling_window": "long_forager_volume_ema_span",
    "long_filter_log_range_ema_span": "long_forager_volatility_ema_span",
    "long_filter_volume_ema_span": "long_forager_volume_ema_span",
    "short_filter_volatility_ema_span": "short_forager_volatility_ema_span",
    "short_filter_noisiness_rolling_window": "short_forager_volatility_ema_span",
    "short_filter_noisiness_ema_span": "short_forager_volatility_ema_span",
    "short_filter_volume_rolling_window": "short_forager_volume_ema_span",
    "short_filter_log_range_ema_span": "short_forager_volatility_ema_span",
    "short_filter_volume_ema_span": "short_forager_volume_ema_span",
    "long_filter_volume_drop_pct": "long_forager_volume_drop_pct",
    "short_filter_volume_drop_pct": "short_forager_volume_drop_pct",
    "long_entry_grid_spacing_weight": "long_entry_grid_spacing_we_weight",
    "short_entry_grid_spacing_weight": "short_entry_grid_spacing_we_weight",
    "long_entry_grid_spacing_log_span_hours": "long_entry_volatility_ema_span_hours",
    "short_entry_grid_spacing_log_span_hours": "short_entry_volatility_ema_span_hours",
    "long_entry_log_range_ema_span_hours": "long_entry_volatility_ema_span_hours",
    "short_entry_log_range_ema_span_hours": "short_entry_volatility_ema_span_hours",
    "long_entry_grid_spacing_log_weight": "long_entry_grid_spacing_volatility_weight",
    "short_entry_grid_spacing_log_weight": "short_entry_grid_spacing_volatility_weight",
    "long_entry_trailing_retracement_log_weight": "long_entry_trailing_retracement_volatility_weight",
    "short_entry_trailing_retracement_log_weight": "short_entry_trailing_retracement_volatility_weight",
    "long_entry_trailing_threshold_log_weight": "long_entry_trailing_threshold_volatility_weight",
    "short_entry_trailing_threshold_log_weight": "short_entry_trailing_threshold_volatility_weight",
}

OBSOLETE_BOUND_KEYS = {
    "long_filter_volatility_drop_pct",
    "short_filter_volatility_drop_pct",
}


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    if verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)


def apply_backward_compatibility_renames(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    for pside, bot_cfg in result.get("bot", {}).items():
        if not isinstance(bot_cfg, dict):
            continue
        for old, new in LEGACY_FILTER_KEYS.items():
            if old in bot_cfg:
                moved_value = bot_cfg[old]
                if new not in bot_cfg:
                    bot_cfg[new] = moved_value
                    _log_config(verbose, logging.INFO, "renaming parameter bot.%s.%s -> %s", pside, old, new)
                    if tracker is not None:
                        tracker.rename(["bot", pside, old], ["bot", pside, new], moved_value)
                del bot_cfg[old]
        for old, new in LEGACY_FORAGER_KEYS.items():
            if old in bot_cfg:
                moved_value = bot_cfg[old]
                if new not in bot_cfg:
                    bot_cfg[new] = moved_value
                    _log_config(verbose, logging.INFO, "renaming parameter bot.%s.%s -> %s", pside, old, new)
                    if tracker is not None:
                        tracker.rename(["bot", pside, old], ["bot", pside, new], moved_value)
                del bot_cfg[old]
        for old, new in LEGACY_ENTRY_GRID_KEYS.items():
            if old in bot_cfg:
                moved_value = bot_cfg[old]
                if new not in bot_cfg:
                    bot_cfg[new] = moved_value
                    _log_config(verbose, logging.INFO, "renaming parameter bot.%s.%s -> %s", pside, old, new)
                    if tracker is not None:
                        tracker.rename(["bot", pside, old], ["bot", pside, new], moved_value)
                del bot_cfg[old]
        for old in OBSOLETE_BOT_KEYS:
            if old in bot_cfg:
                removed_value = bot_cfg.pop(old)
                _log_config(verbose, logging.INFO, "dropping obsolete parameter bot.%s.%s", pside, old)
                if tracker is not None:
                    tracker.remove(["bot", pside, old], removed_value)

    bounds = result.get("optimize", {}).get("bounds", {})
    for old, new in LEGACY_BOUNDS_KEYS.items():
        if old in bounds:
            moved_value = bounds[old]
            if new not in bounds:
                bounds[new] = moved_value
                _log_config(verbose, logging.INFO, "renaming parameter optimize.bounds.%s -> %s", old, new)
                if tracker is not None:
                    tracker.rename(["optimize", "bounds", old], ["optimize", "bounds", new], moved_value)
            del bounds[old]
    for old in OBSOLETE_BOUND_KEYS:
        if old in bounds:
            removed_value = bounds.pop(old)
            _log_config(verbose, logging.INFO, "dropping obsolete parameter optimize.bounds.%s", old)
            if tracker is not None:
                tracker.remove(["optimize", "bounds", old], removed_value)

    live_cfg = result.get("live")
    logging_cfg = result.setdefault("logging", {})
    if isinstance(live_cfg, dict) and "memory_snapshot_interval_minutes" in live_cfg:
        val = live_cfg.pop("memory_snapshot_interval_minutes")
        if "memory_snapshot_interval_minutes" not in logging_cfg:
            logging_cfg["memory_snapshot_interval_minutes"] = val
            _log_config(
                verbose,
                logging.INFO,
                "moved live.memory_snapshot_interval_minutes -> logging.memory_snapshot_interval_minutes",
            )
            if tracker is not None:
                tracker.rename(
                    ["live", "memory_snapshot_interval_minutes"],
                    ["logging", "memory_snapshot_interval_minutes"],
                    val,
                )


def rename_config_keys(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    for section, src, dst in [
        ("live", "minimum_market_age_days", "minimum_coin_age_days"),
        ("live", "noisiness_rolling_mean_window_size", "ohlcv_rolling_window"),
        ("live", "ohlcvs_1m_update_after_minutes", "inactive_coin_candle_ttl_minutes"),
        ("backtest", "panic_market_slippage_pct", "market_order_slippage_pct"),
    ]:
        if src in result[section]:
            result[section][dst] = deepcopy(result[section][src])
            _log_config(verbose, logging.INFO, "renaming parameter %s %s -> %s", section, src, dst)
            if tracker is not None:
                tracker.rename([section, src], [section, dst], result[section][dst])
            del result[section][src]
    if "exchange" in result["backtest"] and isinstance(result["backtest"]["exchange"], str):
        exchange = result["backtest"]["exchange"]
        result["backtest"]["exchanges"] = [exchange]
        _log_config(verbose, logging.INFO, "changed backtest.exchange: %s -> backtest.exchanges: [%s]", exchange, exchange)
        if tracker is not None:
            tracker.rename(["backtest", "exchange"], ["backtest", "exchanges"], [exchange])
        del result["backtest"]["exchange"]
