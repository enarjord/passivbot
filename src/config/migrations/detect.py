from copy import deepcopy


BOT_POSITION_SIDES = ("long", "short")

PB_MULTI_FIELD_MAP = {
    "ddown_factor": "entry_grid_double_down_factor",
    "initial_eprice_ema_dist": "entry_initial_ema_dist",
    "initial_qty_pct": "entry_initial_qty_pct",
    "markup_range": "close_grid_markup_range",
    "min_markup": "close_grid_min_markup",
    "rentry_pprice_dist": "entry_grid_spacing_pct",
    "rentry_pprice_dist_wallet_exposure_weighting": "entry_grid_spacing_we_weight",
    "ema_span_0": "ema_span_0",
    "ema_span_1": "ema_span_1",
    "filter_noisiness_rolling_window": "forager_volatility_ema_span",
    "filter_volume_rolling_window": "forager_volume_ema_span",
}
PB_MULTI_FIELD_MAP_INV = {v: k for k, v in PB_MULTI_FIELD_MAP.items()}


def detect_flavor(config: dict, template: dict) -> str:
    pb_keys = {
        "user",
        "pnls_max_lookback_days",
        "loss_allowance_pct",
        "stuck_threshold",
        "unstuck_close_pct",
        "TWE_long",
        "TWE_short",
        "universal_live_config",
    }
    if all(k in config for k in pb_keys):
        return "pb_multi"
    required_current = {"bot", "live", "backtest", "optimize"}
    if required_current.issubset(config):
        return "current"
    if (
        "config" in config
        and isinstance(config["config"], dict)
        and required_current.issubset(config["config"])
    ):
        return "nested_current"
    if "bot" in config and "live" in config:
        return "live_only"
    return "unknown"


def build_base_config_from_flavor(config: dict, template: dict, flavor: str, verbose: bool) -> dict:
    if flavor == "pb_multi":
        result = deepcopy(template)
        for key1 in result["live"]:
            if key1 in config:
                result["live"][key1] = config[key1]
        if config.get("approved_symbols") and isinstance(config["approved_symbols"], dict):
            result["live"]["coin_flags"] = config["approved_symbols"]
        result["live"]["approved_coins"] = sorted(set(config.get("approved_symbols", [])))
        result["live"]["ignored_coins"] = sorted(set(config.get("ignored_symbols", [])))
        for pside in BOT_POSITION_SIDES:
            universal_cfg = config.get("universal_live_config", {}).get(pside, {})
            for key in result["bot"][pside]:
                inverse_key = PB_MULTI_FIELD_MAP_INV.get(key)
                if inverse_key and inverse_key in universal_cfg:
                    result["bot"][pside][key] = universal_cfg[inverse_key]
            try:
                result["bot"][pside]["close_grid_qty_pct"] = 1.0 / round(
                    universal_cfg.get("n_close_orders", 0)
                )
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            for key in (
                "close_trailing_grid_ratio",
                "close_trailing_retracement_pct",
                "close_trailing_threshold_pct",
                "entry_trailing_grid_ratio",
                "entry_trailing_retracement_pct",
                "entry_trailing_retracement_we_weight",
                "entry_trailing_retracement_volatility_weight",
                "entry_trailing_threshold_pct",
                "entry_trailing_threshold_we_weight",
                "entry_trailing_threshold_volatility_weight",
                "unstuck_ema_dist",
            ):
                result["bot"][pside][key] = 0.0
            if config.get("n_longs", 0) == 0 and config.get("n_shorts", 0) == 0:
                n_positions = len(result["live"].get("coin_flags", {}))
            else:
                n_positions = config.get(f"n_{pside}s", 0)
            result["bot"][pside]["n_positions"] = n_positions
            result["bot"][pside]["unstuck_close_pct"] = config.get("unstuck_close_pct", 0.0)
            result["bot"][pside]["unstuck_loss_allowance_pct"] = config.get(
                "loss_allowance_pct", 0.0
            )
            result["bot"][pside]["unstuck_threshold"] = config.get("stuck_threshold", 0.0)
            twe_key = f"TWE_{pside}"
            if config.get(f"{pside}_enabled", True):
                result["bot"][pside]["total_wallet_exposure_limit"] = config.get(twe_key, 0.0)
            else:
                result["bot"][pside]["total_wallet_exposure_limit"] = 0.0
        return result
    if flavor == "current":
        return deepcopy(config)
    if flavor == "nested_current":
        return deepcopy(config["config"])
    if flavor == "live_only":
        result = deepcopy(config)
        for section in ("optimize", "backtest"):
            if section not in result:
                result[section] = deepcopy(template[section])
        return result
    raise Exception("failed to format config: unknown flavor")
