import argparse
import json
import logging
import math
import os
import re
from copy import deepcopy
from typing import Any, Dict, Tuple, List, Union, Optional, Iterable

import hjson

from pure_funcs import remove_OD, sort_dict_keys, str2bool
from utils import (
    format_end_date,
    symbol_to_coin,
    normalize_coins_source,
    dump_json_streamlined,
)
from config_transform import ConfigTransformTracker, record_transform


def _log_config(verbose: bool, level: int, message: str, *args) -> None:
    prefixed_message = "[config] " + message
    if verbose or level >= logging.WARNING:
        logging.log(level, prefixed_message, *args)
    else:
        logging.debug(prefixed_message, *args)


CURRENCY_METRICS = {
    "adg",
    "adg_per_exposure_long",
    "adg_per_exposure_short",
    "adg_w",
    "adg_w_per_exposure_long",
    "adg_w_per_exposure_short",
    "calmar_ratio",
    "calmar_ratio_w",
    "drawdown_worst",
    "drawdown_worst_mean_1pct",
    "equity_balance_diff_neg_max",
    "equity_balance_diff_neg_mean",
    "equity_balance_diff_pos_max",
    "equity_balance_diff_pos_mean",
    "equity_choppiness",
    "equity_choppiness_w",
    "equity_jerkiness",
    "equity_jerkiness_w",
    "peak_recovery_hours_equity",
    "expected_shortfall_1pct",
    "exponential_fit_error",
    "exponential_fit_error_w",
    "gain",
    "gain_per_exposure_long",
    "gain_per_exposure_short",
    "mdg",
    "mdg_per_exposure_long",
    "mdg_per_exposure_short",
    "mdg_w",
    "mdg_w_per_exposure_long",
    "mdg_w_per_exposure_short",
    "omega_ratio",
    "omega_ratio_w",
    "sharpe_ratio",
    "sharpe_ratio_w",
    "sortino_ratio",
    "sortino_ratio_w",
    "sterling_ratio",
    "sterling_ratio_w",
}

SHARED_METRICS = {
    "positions_held_per_day",
    "positions_held_per_day_w",
    "position_held_hours_mean",
    "position_held_hours_max",
    "position_held_hours_median",
    "position_unchanged_hours_max",
    "volume_pct_per_day_avg",
    "volume_pct_per_day_avg_w",
    "loss_profit_ratio",
    "loss_profit_ratio_w",
    "peak_recovery_hours_pnl",
    "high_exposure_hours_mean_long",
    "high_exposure_hours_max_long",
    "high_exposure_hours_mean_short",
    "high_exposure_hours_max_short",
    "adg_pnl",
    "adg_pnl_w",
    "mdg_pnl",
    "mdg_pnl_w",
    "sharpe_ratio_pnl",
    "sharpe_ratio_pnl_w",
    "sortino_ratio_pnl",
    "sortino_ratio_pnl_w",
}


Path = Tuple[str, ...]  # ("bot", "long", "entry_grid_spacing_pct")


# Template-aligned config cleaning normally treats non-empty dict templates as closed schemas.
# These paths are exceptions: they have required template keys plus arbitrary user-defined keys.
PARTIALLY_OPEN_CONFIG_PATHS: set[Path] = {
    ("backtest", "aggregate"),
}

TEMPLATE_SYNC_PRESERVE_PATHS: tuple[Path, ...] = (
    ("coin_overrides",),
    ("backtest", "suite", "aggregate"),
    ("backtest", "suite", "scenarios"),
    ("backtest", "market_settings_sources"),
    *tuple(PARTIALLY_OPEN_CONFIG_PATHS),
)


def load_hjson_config(config_path: str) -> dict:
    try:
        with open(config_path, encoding="utf-8") as f:
            return remove_OD(hjson.load(f))
    except Exception as e:
        logging.exception("failed to load config file %s", config_path)
        raise


def load_config(filepath: str, live_only=False, verbose=True) -> dict:
    # loads hjson or json v7 config
    try:
        config_raw = load_hjson_config(filepath)
        config = format_config(
            config_raw, live_only=live_only, verbose=verbose, base_config_path=filepath
        )
        config["_raw"] = deepcopy(config_raw)
        existing_log = config.get("_transform_log", [])
        config["_transform_log"] = []
        record_transform(config, "load_config", {"path": filepath})
        config["_transform_log"].extend(existing_log)
        return config
    except Exception:
        logging.exception("failed to load config %s", filepath)
        raise


def dump_config(config: dict, filepath: str, *, clean: bool = False):
    config_copy = deepcopy(config)
    if clean:
        config_copy = clean_config(config_copy)
    sorted_config = sort_dict_keys(config_copy)
    try:
        with open(filepath, "w", encoding="utf-8") as fp:
            dump_json_streamlined(sorted_config, fp, sort_keys=False)
            fp.write("\n")
    except Exception:
        logging.exception("failed to dump config to %s", filepath)
        raise


def expand_PB_mode(mode: str) -> str:
    if mode.lower() in ["gs", "graceful_stop", "graceful-stop"]:
        return "graceful_stop"
    elif mode.lower() in ["m", "manual"]:
        return "manual"
    elif mode.lower() in ["n", "normal"]:
        return "normal"
    elif mode.lower() in ["p", "panic"]:
        return "panic"
    elif mode.lower() in ["t", "tp", "tp_only", "tp-only"]:
        return "tp_only"
    else:
        raise Exception(f"unknown passivbot mode {mode}")


def apply_allowed_modifications(src, modifications, allowed_overrides, return_full=True):
    """
    Apply `modifications` to `src`, but only where `allowed_overrides` permits.

    Args:
        src (dict): The source dictionary (remains untouched).
        modifications (dict): The requested changes.
        allowed_overrides (dict): Same shape as `modifications`, with True/False
                                  (or nested dicts) indicating what is allowed.
        return_full (bool):  True  -> full, deep-copied result of src ⊕ allowed mods
                            False -> *diff* containing only allowed & changed fields.

    Returns:
        dict: Either the fully-merged result (return_full=True) or the filtered diff.
    """

    if return_full:
        result = deepcopy(src)
        target = result
    else:
        result = {}
        target = result

    def _apply_recursive(target_dict, mod_dict, allowed_dict, src_dict=None):
        """
        Recursively walk `mod_dict`:
          • if allowed_dict[key] is True  – apply (or record) the value
          • if it is a dict              – recurse
        `src_dict` carries the corresponding subtree of the original `src`
        so we can compare values when building a *diff*.
        """
        for key, mod_value in mod_dict.items():
            # Skip keys that are not explicitly allowed
            if key not in allowed_dict:
                continue

            allowed_value = allowed_dict[key]

            # ──────────────────────────────────────────────────────────
            # Nested-dict case
            # ──────────────────────────────────────────────────────────
            if isinstance(allowed_value, dict) and isinstance(mod_value, dict):
                # Decide whether it is worth recursing (any nested True?)
                if not _has_allowed_values(allowed_value):
                    continue

                # Ensure a container exists only when needed
                if key not in target_dict:
                    if return_full:
                        target_dict[key] = {}
                    else:
                        # In diff mode we create it *lazily*; only if changes survive
                        target_dict[key] = {}

                # Recurse
                _apply_recursive(
                    target_dict[key],
                    mod_value,
                    allowed_value,
                    src_dict[key] if src_dict and key in src_dict else None,
                )

                # In diff mode, remove empty sub-dicts produced after filtering
                if not return_full and not target_dict[key]:
                    target_dict.pop(key, None)

            # ──────────────────────────────────────────────────────────
            # Scalar / non-dict case
            # ──────────────────────────────────────────────────────────
            elif allowed_value is True:
                if return_full:
                    # Always copy in full-mode
                    target_dict[key] = deepcopy(mod_value)
                else:
                    # Diff-mode: only include if value *changes* w.r.t. src
                    src_val = src_dict.get(key) if src_dict else None
                    if src_val != mod_value:
                        target_dict[key] = deepcopy(mod_value)
            # If allowed_value is False ⇒ skip

    def _has_allowed_values(allowed_subdict):
        """Return True if any nested value (recursively) is True"""
        for v in allowed_subdict.values():
            if v is True:
                return True
            if isinstance(v, dict) and _has_allowed_values(v):
                return True
        return False

    _apply_recursive(target, modifications, allowed_overrides, src if return_full else src)
    return result


def get_allowed_modifications():
    return {
        "bot": {
            "long": {
                "close_grid_markup_end": True,
                "close_grid_markup_start": True,
                "close_grid_qty_pct": True,
                "close_trailing_grid_ratio": True,
                "close_trailing_qty_pct": True,
                "close_trailing_retracement_pct": True,
                "close_trailing_threshold_pct": True,
                "ema_span_0": True,
                "ema_span_1": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_volatility_ema_span_hours": True,
                "entry_grid_spacing_volatility_weight": True,
                "entry_grid_spacing_we_weight": True,
                "entry_initial_ema_dist": True,
                "entry_initial_qty_pct": True,
                "entry_trailing_double_down_factor": True,
                "entry_trailing_grid_ratio": True,
                "entry_trailing_retracement_pct": True,
                "entry_trailing_retracement_we_weight": True,
                "entry_trailing_retracement_volatility_weight": True,
                "entry_trailing_threshold_pct": True,
                "entry_trailing_threshold_we_weight": True,
                "entry_trailing_threshold_volatility_weight": True,
                "unstuck_close_pct": True,
                "unstuck_ema_dist": True,
                "unstuck_threshold": True,
                "wallet_exposure_limit": True,
                "risk_wel_enforcer_threshold": True,
                "risk_we_excess_allowance_pct": True,
                "risk_twel_enforcer_threshold": False,
            },
            "short": {
                "close_grid_markup_end": True,
                "close_grid_markup_start": True,
                "close_grid_qty_pct": True,
                "close_trailing_grid_ratio": True,
                "close_trailing_qty_pct": True,
                "close_trailing_retracement_pct": True,
                "close_trailing_threshold_pct": True,
                "ema_span_0": True,
                "ema_span_1": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_volatility_ema_span_hours": True,
                "entry_grid_spacing_volatility_weight": True,
                "entry_grid_spacing_we_weight": True,
                "entry_initial_ema_dist": True,
                "entry_initial_qty_pct": True,
                "entry_trailing_double_down_factor": True,
                "entry_trailing_grid_ratio": True,
                "entry_trailing_retracement_pct": True,
                "entry_trailing_retracement_we_weight": True,
                "entry_trailing_retracement_volatility_weight": True,
                "entry_trailing_threshold_pct": True,
                "entry_trailing_threshold_we_weight": True,
                "entry_trailing_threshold_volatility_weight": True,
                "unstuck_close_pct": True,
                "unstuck_ema_dist": True,
                "unstuck_threshold": True,
                "wallet_exposure_limit": True,
                "risk_wel_enforcer_threshold": True,
                "risk_we_excess_allowance_pct": True,
                "risk_twel_enforcer_threshold": False,
            },
        },
        "live": {
            "forced_mode_long": True,
            "forced_mode_short": True,
            "leverage": True,
        },
    }


def set_nested_value(d: dict, p: list, v: object):
    """
    Sets a value in a nested dictionary using a path.

    Args:
        d: Dictionary to modify (modified in-place)
        p: Path as list of keys/indices to traverse
        v: Value to set at the target location

    Raises:
        KeyError: If intermediate path doesn't exist
        TypeError: If trying to index into non-dict/non-indexable object
    """
    if not p:
        raise ValueError("Path cannot be empty")

    current = d

    # Navigate to the parent of the target location
    for key in p[:-1]:
        current = current[key]

    # Set the final value
    current[p[-1]] = v


def set_nested_value_safe(d: dict, p: list, v: object, create_missing=False):
    """
    Safe version that handles missing intermediate paths.

    Args:
        d: Dictionary to modify (modified in-place)
        p: Path as list of keys/indices to traverse
        v: Value to set at the target location
        create_missing: If True, creates missing intermediate dictionaries

    Returns:
        bool: True if successful, False if path doesn't exist and create_missing=False
    """
    if not p:
        raise ValueError("Path cannot be empty")

    current = d

    # Navigate to the parent of the target location
    for i, key in enumerate(p[:-1]):
        if key not in current:
            if create_missing:
                current[key] = {}
            else:
                return False
        elif not isinstance(current[key], dict):
            if create_missing:
                # Can't traverse through non-dict, would need to overwrite
                return False
            else:
                return False
        current = current[key]

    # Set the final value
    current[p[-1]] = v
    return True


def nested_update(base_dict, update_dict):
    """Recursively update base_dict with values from update_dict"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            nested_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def parse_overrides(config, verbose=True):
    result = deepcopy(config)
    if not result.get("coin_overrides", {}):
        result["coin_overrides"] = parse_old_coin_flags(config)
        if verbose and result["coin_overrides"]:
            _log_config(
                verbose,
                logging.INFO,
                "Converted old coin_flags to coin_overrides: %s -> %s",
                config.get("live", {}).get("coin_flags"),
                result["coin_overrides"],
            )
    if "live" in result:
        result["live"].pop("coin_flags", None)
        result["live"].setdefault("coin_flags", {})
    for coin in sorted(result["coin_overrides"]):
        coinf = symbol_to_coin(coin)
        if coinf != coin:
            if coinf:
                result["coin_overrides"][coinf] = deepcopy(result["coin_overrides"][coin])
                _log_config(verbose, logging.INFO, "Renamed %s -> %s for coin_overrides", coin, coinf)
            else:
                _log_config(
                    verbose, logging.INFO, "Failed to format %s; removed from coin_overrides", coin
                )
            del result["coin_overrides"][coin]
    for coin, overrides in result["coin_overrides"].items():
        parsed_overrides = {}
        if loaded := load_override_config(result, coin):
            parsed_overrides = apply_allowed_modifications(
                result, loaded, get_allowed_modifications(), return_full=False
            )
        nested_update(
            parsed_overrides,
            apply_allowed_modifications(
                result, overrides, get_allowed_modifications(), return_full=False
            ),
        )

        result.setdefault("coin_overrides", {})[coin] = parsed_overrides
        _log_config(
            verbose,
            logging.INFO,
            "Added overrides for %s: %s",
            coin,
            sort_dict_keys(parsed_overrides),
        )
    record_transform(
        result,
        "parse_overrides",
        {"coins": sorted(result.get("coin_overrides", {}).keys())},
    )
    return result


def load_override_config(config, coin):
    try:
        path = config.get("coin_overrides", {}).get(coin, {}).get("override_config_path")
        if path and os.path.exists(path):
            return load_config(path, verbose=False)
        else:
            base_config_path = config.get("live", {}).get("base_config_path")
            if (
                path
                and base_config_path
                and os.path.exists(
                    (
                        npath := os.path.join(
                            os.path.dirname(base_config_path),
                            path,
                        )
                    )
                )
            ):
                return load_config(npath, verbose=False)
    except Exception as e:
        logging.exception("error loading config %s: %s", path, e)
    return {}


def parse_old_coin_flags(config) -> dict:
    """
    convert pre v7.3.14 coin flags to v7.3.14 dict diff style config diffs
    """
    key_map = {
        "short_mode": ["live", "forced_mode_short"],
        "long_mode": ["live", "forced_mode_long"],
        "WE_limit_long": ["bot", "long", "wallet_exposure_limit"],
        "WE_limit_short": ["bot", "short", "wallet_exposure_limit"],
        "leverage": ["live", "leverage"],
    }
    if not isinstance(config, dict) or "live" not in config or "coin_flags" not in config["live"]:
        return {}
    flags = config["live"]["coin_flags"]
    if not isinstance(flags, dict):
        return {}
    result = {}
    for coin in flags:
        result[coin] = {}
        if not isinstance(flags[coin], str):
            continue
        parser = _build_flag_argparser()
        keysvals = vars(parser.parse_args(flags[coin].split()))
        if lcp := keysvals.get("live_config_path"):
            set_nested_value_safe(
                result[coin],
                ["override_config_path"],
                lcp,
                create_missing=True,
            )
        for key, val in keysvals.items():
            if val and key in key_map:
                set_nested_value_safe(result[coin], key_map[key], val, create_missing=True)
    return result


def _build_flag_argparser() -> argparse.ArgumentParser:
    """Internal helper: returns the tiny parser that understands the *per-coin* flag strings."""

    p = argparse.ArgumentParser(prog="coin_flags", add_help=False)
    p.add_argument("-sm", type=expand_PB_mode, dest="short_mode", default=None)
    p.add_argument("-lm", type=expand_PB_mode, dest="long_mode", default=None)
    p.add_argument("-lw", type=float, dest="WE_limit_long", default=None)
    p.add_argument("-sw", type=float, dest="WE_limit_short", default=None)
    p.add_argument("-lev", type=float, dest="leverage", default=None)
    p.add_argument("-lc", type=str, dest="live_config_path", default=None)
    return p


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
    "filter_noisiness_rolling_window": "filter_volatility_ema_span",
    "filter_volume_rolling_window": "filter_volume_ema_span",
}
PB_MULTI_FIELD_MAP_INV = {v: k for k, v in PB_MULTI_FIELD_MAP.items()}


def _build_from_pb_multi(config: dict, template: dict) -> dict:
    result = deepcopy(template)
    for key1 in result["live"]:
        if key1 in config:
            result["live"][key1] = config[key1]
    if config.get("approved_symbols") and isinstance(config["approved_symbols"], dict):
        result["live"]["coin_flags"] = config["approved_symbols"]
    result["live"]["approved_coins"] = sorted(set(config.get("approved_symbols", [])))
    result["live"]["ignored_coins"] = sorted(set(config.get("ignored_symbols", [])))
    for pside in ("long", "short"):
        universal_cfg = config.get("universal_live_config", {}).get(pside, {})
        for key in result["bot"][pside]:
            inverse_key = PB_MULTI_FIELD_MAP_INV.get(key)
            if inverse_key and inverse_key in universal_cfg:
                result["bot"][pside][key] = universal_cfg[inverse_key]
        try:
            result["bot"][pside]["close_grid_qty_pct"] = 1.0 / round(
                universal_cfg.get("n_close_orders", 0)
            )
        except Exception:
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
        result["bot"][pside]["unstuck_loss_allowance_pct"] = config.get("loss_allowance_pct", 0.0)
        result["bot"][pside]["unstuck_threshold"] = config.get("stuck_threshold", 0.0)
        twe_key = f"TWE_{pside}"
        if config.get(f"{pside}_enabled", True):
            result["bot"][pside]["total_wallet_exposure_limit"] = config.get(twe_key, 0.0)
        else:
            result["bot"][pside]["total_wallet_exposure_limit"] = 0.0
    return result


def _build_from_v7_legacy(config: dict, template: dict) -> dict:
    result = deepcopy(template)
    for section in ("backtest", "live", "optimize", "bot"):
        source_section = config.get(section, {})
        for key, value in source_section.items():
            if key in result[section]:
                result[section][key] = value
    common = config.get("common", {})
    for key, value in common.items():
        if key in result["live"]:
            result["live"][key] = value
    result["live"]["approved_coins"] = common.get("approved_symbols", [])
    result["live"]["coin_flags"] = common.get("symbol_flags", {})
    return result


def _build_from_live_only(config: dict, template: dict) -> dict:
    result = deepcopy(config)
    for section in ("optimize", "backtest"):
        if section not in result:
            result[section] = deepcopy(template[section])
    return result


LEGACY_FILTER_KEYS = {
    "filter_noisiness_rolling_window": "filter_volatility_ema_span",
    "filter_noisiness_ema_span": "filter_volatility_ema_span",
    "filter_log_range_ema_span": "filter_volatility_ema_span",
    "filter_volume_rolling_window": "filter_volume_ema_span",
}

LEGACY_ENTRY_GRID_KEYS = {
    "entry_grid_spacing_weight": "entry_grid_spacing_we_weight",
    "entry_grid_spacing_log_span_hours": "entry_volatility_ema_span_hours",
    "entry_log_range_ema_span_hours": "entry_volatility_ema_span_hours",
    "entry_grid_spacing_log_weight": "entry_grid_spacing_volatility_weight",
    "entry_trailing_retracement_log_weight": "entry_trailing_retracement_volatility_weight",
    "entry_trailing_threshold_log_weight": "entry_trailing_threshold_volatility_weight",
}

LEGACY_BOUNDS_KEYS = {
    "long_filter_noisiness_rolling_window": "long_filter_volatility_ema_span",
    "long_filter_noisiness_ema_span": "long_filter_volatility_ema_span",
    "long_filter_volume_rolling_window": "long_filter_volume_ema_span",
    "long_filter_log_range_ema_span": "long_filter_volatility_ema_span",
    "short_filter_noisiness_rolling_window": "short_filter_volatility_ema_span",
    "short_filter_noisiness_ema_span": "short_filter_volatility_ema_span",
    "short_filter_volume_rolling_window": "short_filter_volume_ema_span",
    "short_filter_log_range_ema_span": "short_filter_volatility_ema_span",
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


def _apply_backward_compatibility_renames(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Translate legacy rolling_window keys to their EMA-span counterparts."""

    for pside, bot_cfg in result.get("bot", {}).items():
        if not isinstance(bot_cfg, dict):
            continue
        for old, new in LEGACY_FILTER_KEYS.items():
            if old in bot_cfg:
                moved_value = bot_cfg[old]
                if new not in bot_cfg:
                    bot_cfg[new] = moved_value
                    _log_config(
                        verbose, logging.INFO, "renaming parameter bot.%s.%s -> %s", pside, old, new
                    )
                    if tracker is not None:
                        tracker.rename(
                            ["bot", pside, old],
                            ["bot", pside, new],
                            moved_value,
                        )
                del bot_cfg[old]
        for old, new in LEGACY_ENTRY_GRID_KEYS.items():
            if old in bot_cfg:
                moved_value = bot_cfg[old]
                if new not in bot_cfg:
                    bot_cfg[new] = moved_value
                    _log_config(
                        verbose, logging.INFO, "renaming parameter bot.%s.%s -> %s", pside, old, new
                    )
                    if tracker is not None:
                        tracker.rename(
                            ["bot", pside, old],
                            ["bot", pside, new],
                            moved_value,
                        )
                del bot_cfg[old]

    bounds = result.get("optimize", {}).get("bounds", {})
    for old, new in LEGACY_BOUNDS_KEYS.items():
        if old in bounds:
            moved_value = bounds[old]
            if new not in bounds:
                bounds[new] = moved_value
                _log_config(
                    verbose, logging.INFO, "renaming parameter optimize.bounds.%s -> %s", old, new
                )
                if tracker is not None:
                    tracker.rename(
                        ["optimize", "bounds", old],
                        ["optimize", "bounds", new],
                        moved_value,
                    )
            del bounds[old]

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


def _migrate_suite_to_scenarios(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Migrate legacy backtest.suite structure to flattened scenarios structure.

    Old format:
        backtest.suite.enabled, backtest.suite.scenarios, backtest.suite.aggregate,
        backtest.suite.include_base_scenario, backtest.suite.base_label, backtest.combine_ohlcvs

    New format:
        backtest.scenarios, backtest.aggregate, backtest.volume_normalization
        (behavior derived from scenario exchange count - single = one exchange, multiple = best-per-coin)
    """
    backtest = result.setdefault("backtest", {})

    # Migrate suite.scenarios -> scenarios, suite.aggregate -> aggregate
    suite = backtest.pop("suite", None)
    if suite and isinstance(suite, dict):
        old_scenarios = suite.get("scenarios", [])
        aggregate = suite.get("aggregate", {"default": "mean"})
        include_base = suite.get("include_base_scenario", False)
        base_label = suite.get("base_label", "base")
        suite_enabled = suite.get("enabled", False)

        # Only migrate if suite was actually in use (enabled or has scenarios)
        if suite_enabled or old_scenarios:
            # Move aggregate up (merge with existing, suite values take precedence)
            existing_aggregate = backtest.get("aggregate", {})
            merged_aggregate = {**existing_aggregate, **aggregate}
            backtest["aggregate"] = merged_aggregate
            _log_config(
                verbose,
                logging.INFO,
                "migrated backtest.suite.aggregate -> backtest.aggregate",
            )
            if tracker is not None:
                tracker.rename(
                    ["backtest", "suite", "aggregate"],
                    ["backtest", "aggregate"],
                    merged_aggregate,
                )

            # Move scenarios up
            new_scenarios = list(old_scenarios)

            # If include_base_scenario was True, prepend implicit base scenario
            # even when no explicit scenarios were provided.
            if include_base:
                base_scenario = {"label": base_label}
                new_scenarios = [base_scenario] + new_scenarios
                _log_config(
                    verbose,
                    logging.INFO,
                    "prepended base scenario '%s' (from include_base_scenario=True)",
                    base_label,
                )

            if "scenarios" not in backtest or not backtest["scenarios"]:
                backtest["scenarios"] = new_scenarios
                _log_config(
                    verbose,
                    logging.INFO,
                    "migrated backtest.suite.scenarios -> backtest.scenarios (%d scenarios)",
                    len(new_scenarios),
                )
                if tracker is not None:
                    tracker.rename(
                        ["backtest", "suite", "scenarios"],
                        ["backtest", "scenarios"],
                        new_scenarios,
                    )
        else:
            # Suite existed but was disabled with no scenarios - just remove it
            if tracker is not None:
                tracker.remove(["backtest", "suite"], suite)

    # Migrate combine_ohlcvs (just remove it, behavior is now derived from exchange count)
    if "combine_ohlcvs" in backtest:
        old_value = backtest.pop("combine_ohlcvs")
        _log_config(
            verbose,
            logging.INFO,
            "removed backtest.combine_ohlcvs=%s (behavior now derived from scenario exchange count)",
            old_value,
        )
        if tracker is not None:
            tracker.remove(["backtest", "combine_ohlcvs"], old_value)

    # Ensure new defaults exist
    if "aggregate" not in backtest:
        backtest["aggregate"] = {"default": "mean"}
    if "scenarios" not in backtest:
        backtest["scenarios"] = []
    if "volume_normalization" not in backtest:
        backtest["volume_normalization"] = True


def _migrate_btc_collateral_settings(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Convert legacy bool collateral flag to fractional settings and ensure defaults."""
    backtest = result.setdefault("backtest", {})

    if "use_btc_collateral" in backtest:
        use_btc = backtest.pop("use_btc_collateral")
        try:
            use_btc_bool = bool(int(use_btc))
        except (TypeError, ValueError):
            use_btc_bool = bool(use_btc)
        if "btc_collateral_cap" not in backtest:
            backtest["btc_collateral_cap"] = 1.0 if use_btc_bool else 0.0
            _log_config(
                verbose,
                logging.INFO,
                "changed backtest.use_btc_collateral -> backtest.btc_collateral_cap = %s",
                backtest["btc_collateral_cap"],
            )
            if tracker is not None:
                tracker.rename(
                    ["backtest", "use_btc_collateral"],
                    ["backtest", "btc_collateral_cap"],
                    backtest["btc_collateral_cap"],
                )
        elif tracker is not None:
            tracker.remove(["backtest", "use_btc_collateral"], use_btc)
        if "btc_collateral_ltv_cap" not in backtest:
            backtest["btc_collateral_ltv_cap"] = None
            if tracker is not None:
                tracker.add(["backtest", "btc_collateral_ltv_cap"], None)

    cap = backtest.get("btc_collateral_cap")
    try:
        cap_float = float(cap)
        if tracker is not None and cap != cap_float:
            tracker.update(["backtest", "btc_collateral_cap"], cap, cap_float)
        backtest["btc_collateral_cap"] = cap_float
    except (TypeError, ValueError):
        if tracker is not None:
            tracker.update(["backtest", "btc_collateral_cap"], cap, 0.0)
        backtest["btc_collateral_cap"] = 0.0

    if "btc_collateral_ltv_cap" not in backtest:
        backtest["btc_collateral_ltv_cap"] = None
        if tracker is not None:
            tracker.add(["backtest", "btc_collateral_ltv_cap"], None)


def detect_flavor(config: dict, template: dict) -> str:
    """Detect incoming config flavor to drive the builder.

    Returns one of: "pb_multi", "v7_legacy", "current", "nested_current", "live_only", or "unknown".
    """
    # PB multi live config signature
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
    if "common" in config:
        return "v7_legacy"
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
    """Return a base v7-shaped config based on detected flavor.

    This function only assembles the skeleton and copies values.
    It intentionally avoids broader migrations/renames.
    """

    if flavor == "pb_multi":
        return _build_from_pb_multi(config, template)

    if flavor == "v7_legacy":
        return _build_from_v7_legacy(config, template)

    if flavor == "current":
        return deepcopy(config)

    if flavor == "nested_current":
        return deepcopy(config["config"])

    if flavor == "live_only":
        return _build_from_live_only(config, template)

    raise Exception("failed to format config: unknown flavor")


def _ensure_bot_defaults_and_bounds(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Ensure required bot defaults and optimize bounds exist for each position side."""
    bounds = result["optimize"]["bounds"]
    for pside in ("long", "short"):
        for k0, v_bt, v_opt in [
            ("close_trailing_qty_pct", 1.0, [0.05, 1.0]),
            (
                "entry_trailing_double_down_factor",
                result["bot"][pside].get("entry_grid_double_down_factor", 1.0),
                [0.01, 3.0],
            ),
            (
                "filter_volatility_ema_span",
                result["bot"][pside].get(
                    "filter_volatility_ema_span",
                    result["bot"][pside].get(
                        "filter_rolling_window",
                        result["live"].get("ohlcv_rolling_window", 60.0),
                    ),
                ),
                [10.0, 1440.0],
            ),
            (
                "filter_volume_ema_span",
                result["bot"][pside].get(
                    "filter_volume_ema_span",
                    result["bot"][pside].get(
                        "filter_rolling_window",
                        result["live"].get("ohlcv_rolling_window", 60.0),
                    ),
                ),
                [10.0, 1440.0],
            ),
            (
                "close_grid_markup_start",
                result["bot"][pside].get("close_grid_min_markup", 0.001)
                + result["bot"][pside].get("close_grid_markup_range", 0.001),
                bounds.get(f"{pside}_min_markup", [0.001, 0.03]),
            ),
            (
                "close_grid_markup_end",
                result["bot"][pside].get("close_grid_min_markup", 0.001),
                bounds.get(f"{pside}_close_grid_min_markup", [0.001, 0.03]),
            ),
            (
                "filter_volume_drop_pct",
                result["live"].get("filter_relative_volume_clip_pct", 0.5),
                [0.0, 1.0],
            ),
            (
                "filter_volatility_drop_pct",
                0.0,
                [0.0, 1.0],
            ),
        ]:
            if k0 not in result["bot"][pside]:
                result["bot"][pside][k0] = v_bt
                _log_config(
                    verbose,
                    logging.INFO,
                    "adding missing backtest parameter %s %s: %s",
                    pside,
                    k0,
                    v_bt,
                )
                if tracker is not None:
                    tracker.add(["bot", pside, k0], v_bt)
            opt_key = f"{pside}_{k0}"
            if opt_key not in bounds:
                bounds[opt_key] = v_opt
                _log_config(
                    verbose,
                    logging.INFO,
                    "adding missing optimize parameter %s %s: %s",
                    pside,
                    opt_key,
                    v_opt,
                )
                if tracker is not None:
                    tracker.add(["optimize", "bounds", opt_key], v_opt)


def _rename_config_keys(
    result: dict, verbose: bool = True, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Rename legacy keys to their current names."""
    for section, src, dst in [
        ("live", "minimum_market_age_days", "minimum_coin_age_days"),
        ("live", "noisiness_rolling_mean_window_size", "ohlcv_rolling_window"),
        ("live", "ohlcvs_1m_update_after_minutes", "inactive_coin_candle_ttl_minutes"),
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
        _log_config(
            verbose,
            logging.INFO,
            "changed backtest.exchange: %s -> backtest.exchanges: [%s]",
            exchange,
            exchange,
        )
        if tracker is not None:
            tracker.rename(
                ["backtest", "exchange"],
                ["backtest", "exchanges"],
                [exchange],
            )
        del result["backtest"]["exchange"]


def _sync_with_template(
    template: dict,
    result: dict,
    base_config_path: str,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    """Prune unused keys and enforce final template-aligned structure."""
    existing_base = result["live"].get("base_config_path") if "live" in result else None
    had_key = "live" in result and "base_config_path" in result["live"]
    if base_config_path or "base_config_path" not in result["live"]:
        result["live"]["base_config_path"] = base_config_path
        if tracker is not None:
            if not had_key:
                tracker.add(["live", "base_config_path"], base_config_path)
            elif existing_base != base_config_path:
                tracker.update(["live", "base_config_path"], existing_base, base_config_path)
    template_with_extras = deepcopy(template)
    template_with_extras.setdefault("live", {})["base_config_path"] = ""
    remove_unused_keys_recursively(
        template_with_extras,
        result,
        verbose=verbose,
        preserve=TEMPLATE_SYNC_PRESERVE_PATHS,
        tracker=tracker,
    )
    remove_unused_keys_recursively(template["bot"], result["bot"], verbose=verbose, tracker=tracker)
    remove_unused_keys_recursively(
        template["optimize"]["bounds"],
        result["optimize"]["bounds"],
        verbose=verbose,
        tracker=tracker,
    )
    remove_unused_keys_recursively(
        template.get("optimize", {}).get("limits", []),
        result["optimize"].setdefault("limits", []),
        verbose=verbose,
        tracker=tracker,
    )


def _hydrate_missing_template_fields(
    template: dict,
    result: dict,
    *,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    """Centralized schema hydration: add all missing template keys before downstream consumers."""
    add_missing_keys_recursively(template, result, verbose=verbose, tracker=tracker)


def _seed_missing_compatibility_sections(
    template: dict,
    result: dict,
    *,
    tracker: Optional[ConfigTransformTracker] = None,
) -> None:
    """Seed missing structural sections without pre-filling legacy-derived leaf values."""
    for pside in ("long", "short"):
        if pside not in result["bot"]:
            seeded = deepcopy(template["bot"][pside])
            result["bot"][pside] = seeded
            if tracker is not None:
                tracker.add(["bot", pside], seeded)
    if "bounds" not in result["optimize"]:
        seeded_bounds = deepcopy(template["optimize"]["bounds"])
        result["optimize"]["bounds"] = seeded_bounds
        if tracker is not None:
            tracker.add(["optimize", "bounds"], seeded_bounds)


def _normalize_position_counts(
    result: dict, tracker: Optional[ConfigTransformTracker] = None
) -> None:
    """Round position counts to integers for each side."""
    for pside in result["bot"]:
        current = result["bot"][pside].get("n_positions")
        rounded = int(round(current))
        if tracker is not None and current != rounded:
            tracker.update(["bot", pside, "n_positions"], current, rounded)
        result["bot"][pside]["n_positions"] = rounded


def _normalize_coin_sources(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("backtest.coin_sources must be a mapping of coin -> exchange")
    normalized: Dict[str, str] = {}
    for coin, exchange in raw.items():
        if exchange is None:
            continue
        coin_key = symbol_to_coin(str(coin), verbose=False)
        if not coin_key:
            continue
        exchange_value = str(exchange)
        existing = normalized.get(coin_key)
        if existing is not None and existing != exchange_value:
            raise ValueError(
                f"backtest.coin_sources maps conflicting exchanges for {coin_key}: "
                f"{existing} and {exchange_value}"
            )
        normalized[coin_key] = exchange_value
    return normalized


def _preserve_coin_sources(result: dict) -> None:
    """Keep track of original approved/ignored coin sources before normalization."""
    sources = result.setdefault("_coins_sources", {})
    live = result.get("live", {})
    for key in ("approved_coins", "ignored_coins"):
        if key in live and key not in sources:
            sources[key] = deepcopy(live[key])


def _apply_non_live_adjustments(
    result: dict,
    verbose: bool = True,
    tracker: Optional[ConfigTransformTracker] = None,
    raw_optimize_limits: Any = None,
    raw_optimize_limits_present: Optional[bool] = None,
) -> None:
    """Adjust live/backtest/optimize fields when not running in live-only mode."""
    for key in ("approved_coins", "ignored_coins"):
        result["live"][key] = normalize_coins_source(result["live"].get(key, ""))
    for pside in result["live"]["approved_coins"]:
        result["live"]["approved_coins"][pside] = [
            coin
            for coin in result["live"]["approved_coins"][pside]
            if coin not in result["live"]["ignored_coins"][pside]
        ]
    result["backtest"]["end_date"] = format_end_date(result["backtest"]["end_date"])
    result["backtest"]["coin_sources"] = _normalize_coin_sources(
        result["backtest"].get("coin_sources", {})
    )
    if result["backtest"].get("filter_by_min_effective_cost") is None:
        result["backtest"]["filter_by_min_effective_cost"] = bool(
            result["live"].get("filter_by_min_effective_cost", False)
        )

    canonical_scoring = []
    seen = set()
    for metric in result["optimize"].get("scoring", []):
        canon = canonicalize_metric_name(metric)
        if canon not in seen:
            canonical_scoring.append(canon)
            seen.add(canon)
    result["optimize"]["scoring"] = canonical_scoring
    backend = str(result["optimize"].get("backend", "deap") or "deap").strip().lower()
    if backend not in {"deap", "pymoo"}:
        raise ValueError(
            f"optimize.backend must be one of ['deap', 'pymoo']; got {result['optimize'].get('backend')!r}"
        )
    result["optimize"]["backend"] = backend

    current_limits = deepcopy(result["optimize"].get("limits", []))
    limits_snapshot = deepcopy(current_limits)
    if raw_optimize_limits_present is None:
        raw_optimize_limits_present = "limits" in result.get("optimize", {})
        if raw_optimize_limits is None and raw_optimize_limits_present:
            raw_optimize_limits = deepcopy(result["optimize"].get("limits"))
    template_limits = deepcopy(get_template_config()["optimize"]["limits"])
    resolved_limits, resolution = _resolve_optimize_limits_for_load(
        raw_optimize_limits=raw_optimize_limits,
        raw_optimize_limits_present=raw_optimize_limits_present,
        template_limits=template_limits,
    )
    result["optimize"]["limits"] = resolved_limits
    if resolution == "normalized_legacy":
        _log_config(
            verbose,
            logging.INFO,
            "normalized optimize.limits to canonical schema (%d entries)",
            len(resolved_limits),
        )
        if tracker is not None:
            tracker.update(["optimize", "limits"], limits_snapshot, resolved_limits)
    elif resolution == "fallback_template":
        _log_config(
            verbose,
            logging.WARNING,
            "optimize.limits malformed or unsupported; falling back to template defaults (%d entries)",
            len(template_limits),
        )
        if tracker is not None:
            tracker.update(["optimize", "limits"], limits_snapshot, resolved_limits)
    for key, value in sorted(result["optimize"]["bounds"].items()):
        if isinstance(value, list):
            if len(value) == 1:
                result["optimize"]["bounds"][key] = [value[0], value[0]]
            elif len(value) == 2:
                result["optimize"]["bounds"][key] = sorted(value)


def format_config(config: dict, verbose=True, live_only=False, base_config_path: str = "") -> dict:
    # attempts to format a config to v7 config
    raw_snapshot = deepcopy(config["_raw"]) if "_raw" in config else None
    existing_log = config.get("_transform_log")
    if isinstance(existing_log, list):
        existing_log = deepcopy(existing_log)
    else:
        existing_log = []
    tracker = ConfigTransformTracker()
    optimize_suite_defined = (
        isinstance(config.get("optimize"), dict) and "suite" in config["optimize"]
    )
    raw_optimize_limits_present = (
        isinstance(config.get("optimize"), dict) and "limits" in config["optimize"]
    )
    raw_optimize_limits = deepcopy(config.get("optimize", {}).get("limits"))
    coin_sources_input = deepcopy(config.get("backtest", {}).get("coin_sources"))
    template = get_template_config()
    flavor = detect_flavor(config, template)
    result = build_base_config_from_flavor(config, template, flavor, verbose)
    for path in ("backtest", "bot", "live", "optimize"):
        require_config_dict(result, path)
    _apply_backward_compatibility_renames(result, verbose=verbose, tracker=tracker)
    _migrate_btc_collateral_settings(result, verbose=verbose, tracker=tracker)
    _migrate_suite_to_scenarios(result, verbose=verbose, tracker=tracker)
    _seed_missing_compatibility_sections(template, result, tracker=tracker)
    for path in ("bot.long", "bot.short", "optimize.bounds"):
        require_config_dict(result, path)
    _ensure_bot_defaults_and_bounds(result, verbose=verbose, tracker=tracker)
    _hydrate_missing_template_fields(template, result, verbose=verbose, tracker=tracker)
    result["bot"] = sort_dict_keys(result["bot"])

    _rename_config_keys(result, verbose=verbose, tracker=tracker)

    _sync_with_template(template, result, base_config_path, verbose=verbose, tracker=tracker)

    _normalize_position_counts(result, tracker=tracker)
    if coin_sources_input is not None:
        result.setdefault("backtest", {})["coin_sources"] = coin_sources_input
    _preserve_coin_sources(result)

    if optimize_suite_defined:
        logging.warning(
            "Config contains optimize.suite, but suite configuration is now defined via "
            "backtest.scenarios. optimize.suite will be ignored and deleted; backtest.scenarios "
            "will be used. If you need different suite definitions, pass --suite-config with a "
            "file containing backtest.scenarios."
        )
        if isinstance(result.get("optimize"), dict) and "suite" in result["optimize"]:
            del result["optimize"]["suite"]

    if not live_only:
        # unneeded adjustments if running live
        _apply_non_live_adjustments(
            result,
            verbose=verbose,
            tracker=tracker,
            raw_optimize_limits=raw_optimize_limits,
            raw_optimize_limits_present=raw_optimize_limits_present,
        )

    result["_transform_log"] = existing_log
    details = {
        "live_only": live_only,
        "base_config_path": base_config_path,
        "flavor": flavor,
    }
    details = tracker.merge_details(details)
    record_transform(
        result,
        "format_config",
        details,
    )

    if raw_snapshot is not None and "_raw" not in result:
        result["_raw"] = deepcopy(raw_snapshot)

    return result


def _clean_dynamic_node(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, sub_value in value.items():
            if str(key).startswith("_"):
                continue
            cleaned[key] = _clean_dynamic_node(sub_value)
        return cleaned
    if isinstance(value, list):
        return [_clean_dynamic_node(item) for item in value]
    return deepcopy(value)


def _clean_with_template(template_node, source_node, path: Path = ()):
    if isinstance(template_node, dict):
        source_dict = source_node if isinstance(source_node, dict) else {}
        if path in PARTIALLY_OPEN_CONFIG_PATHS:
            cleaned = {}
            for key, value in source_dict.items():
                if key in template_node:
                    cleaned[key] = _clean_with_template(template_node[key], value, path + (key,))
                else:
                    cleaned[key] = _clean_dynamic_node(value)
            for key, tmpl_value in template_node.items():
                if key not in cleaned:
                    cleaned[key] = _clean_with_template(tmpl_value, None, path + (key,))
            return cleaned
        if not template_node:
            return _clean_dynamic_node(source_dict)
        result = {}
        for key, tmpl_value in template_node.items():
            result[key] = _clean_with_template(tmpl_value, source_dict.get(key), path + (key,))
        return result
    if isinstance(template_node, list):
        if isinstance(source_node, list):
            return [_clean_dynamic_node(item) for item in source_node]
        return deepcopy(template_node)
    if source_node is None:
        return deepcopy(template_node)
    return deepcopy(source_node)


def clean_config(config: dict) -> dict:
    """
    Return a sanitized config aligned with the template structure, stripped of helper keys,
    with dictionaries sorted recursively.
    """
    template = get_template_config()
    cleaned = _clean_with_template(template, config or {})
    return sort_dict_keys(cleaned)


def strip_config_metadata(config: dict, *, keys: Iterable[str] | None = None) -> dict:
    """
    Return a deep-copied config with the provided metadata keys removed recursively.
    Defaults to removing `_raw` and `_transform_log`.
    """

    removal = set(keys or ("_raw", "_transform_log", "_coins_sources"))

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k not in removal}
        if isinstance(node, list):
            return [_strip(item) for item in node]
        return deepcopy(node)

    return _strip(config)


def parse_limits_string(limits_str: Union[str, dict]) -> dict:
    """
    Parses a string like "--penalize_if_greater_than_drawdown_worst 0.3 --penalize_if_lower_than_gain 0.005"
    into a dictionary like:
    {
        "penalize_if_greater_than_drawdown_worst": 0.3,
        "penalize_if_lower_than_gain": 0.005,
    }
    """
    if not limits_str:
        return {}
    if isinstance(limits_str, dict):
        return limits_str
    tokens = limits_str.replace(":", "").split("--")
    result = {}
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        try:
            k, v = token.split()
            result[k] = float(v)
        except ValueError:
            raise ValueError(f"Invalid limits format for token: {token}")
    return result


def normalize_limit_entries(
    raw_limits: Union[str, List[dict], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normalizes optimize.limits into a canonical list of limit clauses.
    Accepts legacy dicts/CLI strings as well as the new list format.
    """
    if raw_limits is None:
        return []
    parsed = raw_limits
    if isinstance(raw_limits, str):
        stripped = raw_limits.strip()
        if not stripped:
            return []
        if stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = parse_limits_string(stripped)
        else:
            parsed = parse_limits_string(stripped)
    elif isinstance(raw_limits, dict):
        parsed = deepcopy(raw_limits)
    elif isinstance(raw_limits, list):
        parsed = deepcopy(raw_limits)
    else:
        raise ValueError(f"Unsupported limits format: {type(raw_limits).__name__}")

    if isinstance(parsed, dict):
        entries: List[Dict[str, Any]] = _legacy_limits_dict_to_entries(parsed)
    elif isinstance(parsed, list):
        entries = parsed
    else:
        raise ValueError(f"Unsupported parsed limits payload: {type(parsed).__name__}")

    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        normalized.append(_normalize_limit_entry_preserve_extras(entry))
    return normalized


def _normalize_limit_entry_preserve_extras(entry: Any) -> Dict[str, Any]:
    normalized = _normalize_limit_entry(entry)
    if not isinstance(entry, dict):
        return normalized
    for key, value in entry.items():
        if key not in normalized:
            normalized[key] = deepcopy(value)
    return normalized


def _is_canonical_limit_entry(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        normalized = _normalize_limit_entry(entry)
    except Exception:
        return False
    for key, norm_val in normalized.items():
        raw_val = entry.get(key)
        if key == "range":
            if not _range_equal(raw_val, norm_val):
                return False
        else:
            if not _numeric_equal(raw_val, norm_val):
                return False
    return True


def _resolve_optimize_limits_for_load(
    *,
    raw_optimize_limits: Any,
    raw_optimize_limits_present: bool,
    template_limits: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Resolve optimize.limits during config load.

    Returns (resolved_limits, resolution) where resolution is one of:
    - "template_default"
    - "preserved_canonical"
    - "normalized_legacy"
    - "fallback_template"
    """
    if not raw_optimize_limits_present:
        return deepcopy(template_limits), "template_default"

    if isinstance(raw_optimize_limits, list):
        if all(_is_canonical_limit_entry(entry) for entry in raw_optimize_limits):
            return deepcopy(raw_optimize_limits), "preserved_canonical"
        try:
            return normalize_limit_entries(raw_optimize_limits), "normalized_legacy"
        except Exception:
            return deepcopy(template_limits), "fallback_template"

    if isinstance(raw_optimize_limits, (str, dict)):
        try:
            return normalize_limit_entries(raw_optimize_limits), "normalized_legacy"
        except Exception:
            return deepcopy(template_limits), "fallback_template"

    return deepcopy(template_limits), "fallback_template"


def _legacy_limits_dict_to_entries(limits_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key in limits_dict:
        value = limits_dict[key]
        canonical_key = canonicalize_limit_name(key)
        metric: str
        penalty: str
        if canonical_key.startswith("penalize_if_greater_than_"):
            metric = canonical_key[len("penalize_if_greater_than_") :]
            penalty = "greater_than"
        elif canonical_key.startswith("penalize_if_lower_than_"):
            metric = canonical_key[len("penalize_if_lower_than_") :]
            penalty = "less_than"
        else:
            metric = canonical_key
            penalty = "auto"
        numeric_value = _ensure_float(value)
        if numeric_value is None:
            raise ValueError(f"Limit '{key}' must have a numeric value.")
        entries.append({"metric": metric, "penalize_if": penalty, "value": numeric_value})
    return entries


def _normalize_penalize_if(value: Any) -> str:
    if value is None:
        raise ValueError("limits entries must include 'penalize_if'.")
    token = str(value).strip().lower()
    mapping = {
        ">": "greater_than",
        "gt": "greater_than",
        "greater": "greater_than",
        "greater_than": "greater_than",
        "above": "greater_than",
        "<": "less_than",
        "lt": "less_than",
        "lower": "less_than",
        "less": "less_than",
        "less_than": "less_than",
        "below": "less_than",
        "outside": "outside_range",
        "outside_range": "outside_range",
        "out_of_range": "outside_range",
        "inside": "inside_range",
        "inside_range": "inside_range",
        "auto": "auto",
    }
    normalized = mapping.get(token)
    if not normalized:
        raise ValueError(f"Unsupported penalize_if value '{value}'.")
    return normalized


def _ensure_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _restore_numeric_precision(value: Optional[float]) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isfinite(value):
            rounded = round(value)
            if abs(value - rounded) < 1e-12:
                return int(rounded)
    return value


def _extract_range(payload: Any) -> Optional[Tuple[float, float]]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        low = payload.get("low")
        high = payload.get("high")
        if low is None:
            low = payload.get("min")
        if low is None:
            low = payload.get("start")
        if high is None:
            high = payload.get("max")
        if high is None:
            high = payload.get("end")
        if low is None or high is None:
            return None
        low_f = _ensure_float(low)
        high_f = _ensure_float(high)
        if low_f is None or high_f is None:
            return None
        return (min(low_f, high_f), max(low_f, high_f))
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        low_f = _ensure_float(payload[0])
        high_f = _ensure_float(payload[1])
        if low_f is None or high_f is None:
            return None
        return (min(low_f, high_f), max(low_f, high_f))
    return None


def _normalize_limit_entry(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Each limit entry must be a dict, got {type(entry).__name__}.")
    payload = deepcopy(entry)
    metric = payload.get("metric") or payload.get("name")
    if not metric:
        raise ValueError("Limit entries must include a 'metric' field.")
    metric = canonicalize_metric_name(str(metric))
    penalize_if = _normalize_penalize_if(payload.get("penalize_if"))
    stat = payload.get("stat") or payload.get("field")
    normalized_stat: Optional[str] = None
    if stat is not None:
        stat = str(stat).lower()
        if stat not in {"min", "max", "mean", "std"}:
            raise ValueError(f"Unsupported stat '{stat}' for limit on {metric}.")
        normalized_stat = stat
    result: Dict[str, Any] = {"metric": metric, "penalize_if": penalize_if}
    if normalized_stat:
        result["stat"] = normalized_stat
    if penalize_if in {"greater_than", "less_than", "auto"}:
        bound = payload.get("value")
        if bound is None:
            bound = payload.get("threshold")
        if bound is None:
            bound = payload.get("bound")
        numeric_bound = _ensure_float(bound)
        if numeric_bound is None:
            raise ValueError(f"Limit for {metric} requires a numeric 'value'.")
        result["value"] = _restore_numeric_precision(numeric_bound)
    elif penalize_if in {"outside_range", "inside_range"}:
        range_payload = payload.get("range")
        if range_payload is None:
            range_payload = payload.get("values")
        if range_payload is None:
            range_payload = payload.get("bounds")
        if range_payload is None and isinstance(payload.get("value"), (list, tuple)):
            range_payload = payload.get("value")
        bounds = _extract_range(range_payload)
        if bounds is None:
            raise ValueError(f"Limit for {metric} requires a two-value 'range'.")
        result["range"] = [
            _restore_numeric_precision(bounds[0]),
            _restore_numeric_precision(bounds[1]),
        ]
    else:
        raise ValueError(f"Unsupported penalize_if '{penalize_if}' for {metric}.")
    return result


def _numeric_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def _range_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != len(b):
        return False
    return all(_numeric_equal(x, y, tol=tol) for x, y in zip(a, b))


def _entries_equivalent(raw_entry: Any, normalized_entry: Dict[str, Any]) -> bool:
    if not isinstance(raw_entry, dict):
        return False
    raw_keys = set(raw_entry.keys())
    norm_keys = set(normalized_entry.keys())
    if raw_keys != norm_keys:
        return False
    for key, norm_val in normalized_entry.items():
        raw_val = raw_entry.get(key)
        if key == "range":
            if not _range_equal(raw_val, norm_val):
                return False
        else:
            if not _numeric_equal(raw_val, norm_val):
                return False
    return True


def _limits_structurally_equal(raw_limits: Any, normalized_limits: List[Dict[str, Any]]) -> bool:
    if not isinstance(raw_limits, list):
        return False
    if len(raw_limits) != len(normalized_limits):
        return False
    return all(_entries_equivalent(raw, norm) for raw, norm in zip(raw_limits, normalized_limits))


def add_missing_keys_recursively(src, dst, parent=None, verbose=True, tracker=None):
    if parent is None:
        parent = []
    for k in src:
        if k not in dst:
            _log_config(verbose, logging.INFO, "Added missing %s to config.", ".".join(parent + [k]))
            dst[k] = src[k]
            if tracker is not None:
                tracker.add(parent + [k], src[k])
        # --- NEW: only walk down if both sides are dicts -------------
        elif isinstance(src[k], dict) and isinstance(dst.get(k), dict):
            add_missing_keys_recursively(src[k], dst[k], parent + [k], verbose, tracker=tracker)
        # --------------------------------------------------------------
        elif isinstance(src[k], dict):
            # type clash: leave the user’s value untouched
            _log_config(
                verbose,
                logging.INFO,
                "Skipping template subtree %s (template is dict, config is %s)",
                ".".join(parent + [k]),
                type(dst.get(k)).__name__,
            )
            continue
        else:
            # previous branches already handle k not in dst; keep safe assignment
            if k not in dst:
                _log_config(
                    verbose,
                    logging.INFO,
                    "Adding missing key -> val %s -> %s to config",
                    ".".join(parent + [k]),
                    src[k],
                )
                dst[k] = src[k]
                if tracker is not None:
                    tracker.add(parent + [k], src[k])


def remove_unused_keys_recursively(
    src,
    dst,
    parent=None,
    verbose=True,
    preserve: Optional[Iterable[Iterable[str]]] = None,
    tracker=None,
):
    if parent is None:
        parent = []
        # normalize preserve spec only once at root invocation
        if preserve is None:
            preserve_set = set()
        else:
            preserve_set = {tuple(p) for p in preserve}
    else:
        preserve_set = getattr(remove_unused_keys_recursively, "_preserve_set", set())

    def _path_is_preserved(path: Iterable[str]) -> bool:
        if not preserve_set:
            return False
        path_tuple = tuple(path)
        for preserved in preserve_set:
            if path_tuple[: len(preserved)] == preserved:
                return True
        return False

    # stash preserve set on the function so recursive calls can reuse it without recomputing
    if parent == []:
        remove_unused_keys_recursively._preserve_set = preserve_set

    if _path_is_preserved(parent):
        return
    if not isinstance(dst, dict) or not isinstance(src, dict):
        return
    # Defensive: configs can contain non-string keys (e.g. from user edits or JSON coercions).
    # Template keys are always strings, so non-string keys are always unused and should be removed.
    for k in list(dst.keys()):
        if isinstance(k, str):
            continue
        removed = dst.pop(k)
        current_path = parent + [str(k)]
        _log_config(
            verbose, logging.INFO, "Removed unused key from config: %s", ".".join(current_path)
        )
        if tracker is not None:
            tracker.remove(current_path, removed)

    def _sort_key(value) -> tuple[str, str]:
        """Sort keys by type name, then by string representation."""
        return (type(value).__name__, str(value))

    for k in sorted(list(dst.keys()), key=_sort_key):
        current_path = parent + [k]
        if _path_is_preserved(current_path):
            continue
        if isinstance(k, str) and k.startswith("_"):
            continue
        if k not in src:
            removed = dst.pop(k)
            _log_config(
                verbose,
                logging.INFO,
                "Removed unused key from config: %s",
                ".".join(map(str, current_path)),
            )
            if tracker is not None:
                tracker.remove(current_path, removed)
            continue
        src_val = src[k]
        dst_val = dst[k]
        if isinstance(dst_val, dict) and isinstance(src_val, dict):
            remove_unused_keys_recursively(
                src_val, dst_val, current_path, verbose=verbose, tracker=tracker
            )

    if parent == [] and hasattr(remove_unused_keys_recursively, "_preserve_set"):
        delattr(remove_unused_keys_recursively, "_preserve_set")


def comma_separated_values_float(x):
    return [float(z) for z in x.split(",")]


def comma_separated_values(x):
    # Preserve JSON/HJSON-like strings (used for approved/ignored coin dicts)
    if isinstance(x, str):
        raw = x.strip()
        if raw and raw[0] in "[{" and raw[-1] in "]}":
            return [x]
    return [item.strip() for item in x.split(",")]


def optional_float(x):
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return float(x)


def canonicalize_metric_name(metric: str) -> str:
    if metric.endswith("_usd") or metric.endswith("_btc"):
        return metric

    for prefix, suffix in (("usd_", "usd"), ("btc_", "btc")):
        if metric.startswith(prefix):
            core = metric[len(prefix) :]
            if core in SHARED_METRICS:
                return core
            return f"{core}_{suffix}"

    if metric in SHARED_METRICS:
        return metric

    if metric in CURRENCY_METRICS:
        return f"{metric}_usd"

    return metric


def canonicalize_limit_name(limit_key: str) -> str:
    if limit_key.startswith("lower_bound_"):
        metric = limit_key[len("lower_bound_") :]
        return "penalize_if_greater_than_" + canonicalize_metric_name(metric)
    if limit_key.startswith("upper_bound_"):
        metric = limit_key[len("upper_bound_") :]
        return "penalize_if_lower_than_" + canonicalize_metric_name(metric)
    prefixes = ["penalize_if_greater_than_", "penalize_if_lower_than_"]
    for prefix in prefixes:
        if limit_key.startswith(prefix):
            metric = limit_key[len(prefix) :]
            return prefix + canonicalize_metric_name(metric)
    return canonicalize_metric_name(limit_key)


def merge_negative_cli_values(argv):
    """Allow comma-separated values that begin with '-' to be parsed as option values."""
    out = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            out.extend(argv[i:])
            break
        if token.startswith("-") and "=" not in token and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-") and "," in nxt:
                out.append(f"{token}={nxt}")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def create_acronym(full_name, acronyms=set()):
    i = 1
    while True:
        i += 1
        if i > 100:
            raise Exception(f"too many acronym duplicates for {full_name}")
        shortened_name = full_name
        for k in [
            "backtest.",
            "live.",
            "optimize.bounds.",
            "optimize.limits.",
            "optimize.",
            "bot.",
        ]:
            if shortened_name.startswith(k):
                shortened_name = shortened_name.replace(k, "")
                break

        # Split on both '_' and '.' using regex
        splitted = re.split(r"[._]+", shortened_name)
        acronym = "".join(word[0] for word in splitted if word)  # skip any empty splits

        if acronym not in acronyms:
            break
        acronym = acronym + str(i)
        if acronym not in acronyms:
            break
    return acronym


# Hard-coded CLI shortcuts for backwards compatibility and cleaner default help.
# Format:
#   config_key -> {
#       "visible": ["--preferred-name", "-x"],
#       "hidden": ["--legacy_name", "--legacy_name_with_dots"],
#       "commands": {"live", "backtest", "optimize"},
#       "group": {"live": "Coin Selection", ...},
#       "type": type_converter,
#       "metavar": "CSV|INT|...",
#       "help": "Human-facing help text",
#   }
RESERVED_CLI_ARGS = {
    "live.approved_coins": {
        "visible": ["--symbols", "-s"],
        "hidden": ["--live.approved_coins", "--live_approved_coins"],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest", "optimize"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": (
            "Approved coins. Comma-separated coins like BTC,ETH,XRP, or path to a JSON "
            "coin list file. Use coin tickers, not exchange symbols."
        ),
    },
    "live.ignored_coins": {
        "visible": ["--ignored-coins", "-ic"],
        "hidden": ["--live.ignored_coins", "--live_ignored_coins"],
        "type": comma_separated_values,
        "metavar": "CSV_OR_PATH",
        "commands": {"live", "backtest"},
        "group": {
            "live": "Coin Selection",
            "backtest": "Coin Selection",
        },
        "help": "Ignored coins. Comma-separated coins or path to a JSON coin list file.",
    },
    "live.minimum_coin_age_days": {
        "visible": ["--minimum-coin-age-days", "-mcad"],
        "hidden": ["--live.minimum_coin_age_days", "--live_minimum_coin_age_days"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Coin Selection"},
        "help": "Minimum coin age in days required before a coin is eligible to trade.",
    },
    "live.filter_by_min_effective_cost": {
        "visible": ["--filter-by-min-effective-cost", "-fbmec"],
        "hidden": [
            "--live.filter_by_min_effective_cost",
            "--live_filter_by_min_effective_cost",
        ],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Filter out coins whose minimum effective cost exceeds the configured size.",
    },
    "live.hedge_mode": {
        "visible": ["--hedge-mode", "-hm"],
        "hidden": ["--live.hedge_mode", "--live_hedge_mode"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": (
            "Enable or disable hedge mode. If the exchange does not support simultaneous "
            "long and short on the same coin, the bot will use hedge_mode=false."
        ),
    },
    "live.leverage": {
        "visible": ["--leverage", "-lev"],
        "hidden": ["--live.leverage", "--live_leverage"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Set leverage for this run.",
    },
    "live.market_orders_allowed": {
        "visible": ["--market-orders-allowed", "-moa"],
        "hidden": ["--live.market_orders_allowed", "--live_market_orders_allowed"],
        "type": str2bool,
        "metavar": "Y/N",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Allow or disallow market orders.",
    },
    "live.max_realized_loss_pct": {
        "visible": ["--max-realized-loss-pct", "-mrlp"],
        "hidden": ["--live.max_realized_loss_pct", "--live_max_realized_loss_pct"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Maximum realized loss percentage allowed before trading is halted.",
    },
    "live.price_distance_threshold": {
        "visible": ["--price-distance-threshold", "-pdt"],
        "hidden": ["--live.price_distance_threshold", "--live_price_distance_threshold"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Reject orders whose price is too far from the market.",
    },
    "live.time_in_force": {
        "visible": ["--time-in-force", "-tif"],
        "hidden": ["--live.time_in_force", "--live_time_in_force"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"live"},
        "group": {"live": "Behavior"},
        "help": "Time-in-force policy for live orders.",
    },
    "backtest.exchanges": {
        "visible": ["--exchanges", "-e"],
        "hidden": ["--backtest.exchanges", "--backtest_exchanges"],
        "type": comma_separated_values,
        "metavar": "CSV",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Coin Selection",
            "optimize": "Coin Selection",
        },
        "help": "Backtest exchanges to use, for example bybit or binance,bybit.",
    },
    "backtest.start_date": {
        "visible": ["--start-date", "-sd"],
        "hidden": ["--backtest.start_date", "--backtest_start_date"],
        "type": str,
        "metavar": "DATE",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Date Range",
            "optimize": "Date Range",
        },
        "help": "Backtest start date. Examples: 2025, 2025-01, 2025-01-15.",
    },
    "backtest.end_date": {
        "visible": ["--end-date", "-ed"],
        "hidden": ["--backtest.end_date", "--backtest_end_date"],
        "type": str,
        "metavar": "DATE",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Date Range",
            "optimize": "Date Range",
        },
        "help": 'Backtest end date. Use "-ed now" for the latest available candles.',
    },
    "backtest.candle_interval_minutes": {
        "visible": ["--candle-interval-minutes", "-cim"],
        "hidden": [
            "--backtest.candle_interval_minutes",
            "--backtest_candle_interval_minutes",
        ],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"backtest", "optimize"},
        "group": {
            "backtest": "Backtest Runtime",
            "optimize": "Date Range",
        },
        "help": "Backtest candle interval in minutes.",
    },
    "backtest.starting_balance": {
        "visible": ["--starting-balance", "-sb"],
        "hidden": ["--backtest.starting_balance", "--backtest_starting_balance"],
        "type": float,
        "metavar": "FLOAT",
        "commands": {"backtest"},
        "group": {"backtest": "Backtest Runtime"},
        "help": "Starting balance for the backtest.",
    },
    "backtest.aggregate.default": {
        "visible": ["--aggregate-default"],
        "hidden": ["--backtest.aggregate.default", "--backtest_aggregate_default"],
        "type": str,
        "metavar": "VALUE",
        "commands": {"backtest"},
        "group": {"backtest": "Suite"},
        "help": "Suite-only: default aggregation to use for scenario metrics.",
    },
    "optimize.iters": {
        "visible": ["--iters", "-i"],
        "hidden": ["--optimize.iters", "--optimize_iters"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer iteration budget.",
    },
    "optimize.n_cpus": {
        "visible": ["--cpus", "-c"],
        "hidden": ["--optimize.n_cpus", "--optimize_n_cpus"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer worker count.",
    },
    "optimize.scoring": {
        "visible": ["--scoring", "-os"],
        "hidden": ["--optimize.scoring", "--optimize_scoring"],
        "type": comma_separated_values,
        "metavar": "CSV",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer scoring metrics, for example adg_pnl,loss_profit_ratio.",
    },
    "optimize.population_size": {
        "visible": ["--population-size", "-ps"],
        "hidden": ["--optimize.population_size", "--optimize_population_size"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer population size.",
    },
    "optimize.pareto_max_size": {
        "visible": ["--pareto-max-size", "-pms"],
        "hidden": ["--optimize.pareto_max_size", "--optimize_pareto_max_size"],
        "type": int,
        "metavar": "INT",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Maximum persisted Pareto set size.",
    },
    "optimize.backend": {
        "visible": ["--backend", "-ob"],
        "hidden": ["--optimize.backend", "--optimize_backend", "--optimizer-backend"],
        "type": str,
        "metavar": "BACKEND",
        "commands": {"optimize"},
        "group": {"optimize": "Optimizer"},
        "help": "Optimizer backend to use. Supported values: deap or pymoo.",
    },
}

CLI_HELP_GROUPS = {
    "live": [
        "Coin Selection",
        "Behavior",
        "Runtime",
        "Logging",
        "Advanced Overrides",
    ],
    "backtest": [
        "Coin Selection",
        "Date Range",
        "Backtest Runtime",
        "Suite",
        "Output / Analysis",
        "Logging",
        "Advanced Overrides",
    ],
    "optimize": [
        "Coin Selection",
        "Date Range",
        "Optimizer",
        "Suite",
        "Logging",
        "Backtest Runtime",
        "Optimize Common",
        "Optimize Bounds",
        "Optimize DEAP",
        "Optimize Pymoo",
        "Advanced Overrides",
    ],
}


def _register_argument(container, visible_names, hidden_names, **kwargs):
    container.add_argument(*visible_names, **kwargs)
    hidden_kwargs = dict(kwargs)
    hidden_kwargs["help"] = argparse.SUPPRESS
    for alias in hidden_names:
        container.add_argument(alias, **hidden_kwargs)


def _argument_metavar(type_, full_name: str, value):
    if type_ is comma_separated_values:
        return "CSV"
    if type_ is comma_separated_values_float:
        return "MIN,MAX[,STEP]"
    if type_ is str2bool:
        return "Y/N"
    if type_ is optional_float:
        return "FLOAT"
    if type_ is int:
        return "INT"
    if type_ is float:
        return "FLOAT"
    if type_ is str and full_name.endswith("_date"):
        return "DATE"
    if type_ is str and full_name.endswith("_dir"):
        return "PATH"
    if type_ is str and value is None:
        return "VALUE"
    return "VALUE"


def _argument_help_text(full_name: str, appendix: str) -> str:
    base = f"Override {full_name}."
    if appendix:
        return f"{base} {appendix}".strip()
    return base


def _classify_live_argument(full_name: str, help_all: bool) -> Optional[str]:
    coin_selection = {
        "live.approved_coins",
        "live.ignored_coins",
        "live.minimum_coin_age_days",
    }
    behavior = {
        "live.filter_by_min_effective_cost",
        "live.forced_mode_long",
        "live.forced_mode_short",
        "live.hedge_mode",
        "live.leverage",
        "live.market_orders_allowed",
        "live.max_realized_loss_pct",
        "live.order_match_tolerance_pct",
        "live.price_distance_threshold",
    }
    runtime = {
        "live.execution_delay_seconds",
        "live.max_concurrent_api_requests",
        "live.max_n_restarts_per_day",
        "live.max_ohlcv_fetches_per_minute",
        "live.recv_window_ms",
        "live.time_in_force",
        "live.user",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in behavior:
        return "Behavior"
    if full_name in runtime:
        return "Runtime"
    return "Advanced Overrides" if help_all else None


def _classify_backtest_argument(full_name: str, help_all: bool) -> Optional[str]:
    coin_selection = {
        "backtest.exchanges",
        "live.approved_coins",
        "live.ignored_coins",
        "live.minimum_coin_age_days",
    }
    date_range = {
        "backtest.end_date",
        "backtest.max_warmup_minutes",
        "backtest.start_date",
    }
    runtime = {
        "backtest.aggregate.default",
        "backtest.balance_sample_divider",
        "backtest.btc_collateral_cap",
        "backtest.btc_collateral_ltv_cap",
        "backtest.candle_interval_minutes",
        "backtest.compress_cache",
        "backtest.dynamic_wel_by_tradability",
        "backtest.filter_by_min_effective_cost",
        "backtest.gap_tolerance_ohlcvs_minutes",
        "backtest.maker_fee_override",
        "backtest.ohlcv_source_dir",
        "backtest.starting_balance",
    }
    output = {
        "backtest.base_dir",
        "backtest.volume_normalization",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in date_range:
        return "Date Range"
    if full_name in runtime:
        return "Backtest Runtime"
    if full_name in output:
        return "Output / Analysis"
    if full_name.startswith("backtest.scenarios"):
        return "Suite"
    return "Advanced Overrides" if help_all else None


def _classify_optimize_argument(full_name: str, help_all: bool) -> Optional[str]:
    coin_selection = {
        "backtest.exchanges",
        "live.approved_coins",
    }
    date_range = {
        "backtest.end_date",
        "backtest.start_date",
        "backtest.candle_interval_minutes",
    }
    optimizer = {
        "optimize.iters",
        "optimize.backend",
        "optimize.n_cpus",
        "optimize.pareto_max_size",
        "optimize.population_size",
        "optimize.scoring",
    }
    backtest_runtime = {
        "backtest.aggregate.default",
        "backtest.balance_sample_divider",
        "backtest.btc_collateral_cap",
        "backtest.btc_collateral_ltv_cap",
        "backtest.compress_cache",
        "backtest.dynamic_wel_by_tradability",
        "backtest.filter_by_min_effective_cost",
        "backtest.gap_tolerance_ohlcvs_minutes",
        "backtest.maker_fee_override",
        "backtest.max_warmup_minutes",
        "backtest.ohlcv_source_dir",
        "backtest.starting_balance",
        "backtest.volume_normalization",
    }
    optimize_common = {
        "optimize.compress_results_file",
        "optimize.enable_overrides",
        "optimize.fixed_params",
        "optimize.fixed_runtime_overrides",
        "optimize.limits",
        "optimize.max_pending_starting_evals_per_cpu",
        "optimize.round_to_n_significant_digits",
        "optimize.starting_config_twe_multiplier",
        "optimize.write_all_results",
    }
    optimize_deap = {
        "optimize.crossover_eta",
        "optimize.crossover_probability",
        "optimize.mutation_eta",
        "optimize.mutation_indpb",
        "optimize.mutation_probability",
        "optimize.offspring_multiplier",
    }
    if full_name in coin_selection:
        return "Coin Selection"
    if full_name in date_range:
        return "Date Range"
    if full_name in optimizer:
        return "Optimizer"
    if full_name in backtest_runtime:
        return "Backtest Runtime"
    if full_name.startswith("backtest.scenarios"):
        return "Suite"
    if full_name.startswith("optimize.bounds."):
        return "Optimize Bounds" if help_all else None
    if full_name in optimize_deap or full_name.startswith("optimize.deap.shared."):
        return "Optimize DEAP" if help_all else None
    if full_name.startswith("optimize.pymoo."):
        return "Optimize Pymoo" if help_all else None
    if full_name in optimize_common:
        return "Optimize Common" if help_all else None
    return "Advanced Overrides" if help_all else None


def classify_config_argument(
    full_name: str, command: Optional[str], help_all: bool
) -> Optional[str]:
    if command == "live":
        return _classify_live_argument(full_name, help_all)
    if command == "backtest":
        return _classify_backtest_argument(full_name, help_all)
    if command == "optimize":
        return _classify_optimize_argument(full_name, help_all)
    return None


def add_reserved_arguments(
    parser,
    *,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
) -> Tuple[set, set]:
    """Add hard-coded CLI arguments for backwards compatibility.

    Returns the set of reserved acronyms and config keys that should be
    skipped by add_arguments_recursively().
    """
    reserved_acronyms = set()
    reserved_keys = set()

    for config_key, spec in RESERVED_CLI_ARGS.items():
        commands = spec.get("commands")
        if command is not None and commands is not None and command not in commands:
            continue
        visible_group = (
            spec.get("group", {}).get(command)
            if command is not None
            else None
        )
        container = group_map.get(visible_group, parser) if group_map and visible_group else parser

        register_kwargs = dict(
            type=spec["type"],
            dest=config_key,
            required=False,
            default=None,
            metavar=spec["metavar"],
            help=spec["help"] if visible_group is not None or command is None else argparse.SUPPRESS,
        )
        if "choices" in spec:
            register_kwargs["choices"] = spec["choices"]

        _register_argument(
            container,
            spec["visible"],
            spec["hidden"],
            **register_kwargs,
        )
        visible_shorts = [name[1:] for name in spec["visible"] if name.startswith("-") and not name.startswith("--")]
        for short_name in visible_shorts:
            reserved_acronyms.add(short_name)
        reserved_keys.add(config_key)

    return reserved_acronyms, reserved_keys


def add_config_arguments(parser, config, *, command: Optional[str] = None, help_all: bool = False, group_map=None):
    """Add all CLI arguments for config parameters.

    This is the main entry point for adding config-based arguments.
    It first adds hard-coded reserved arguments (for backwards compat),
    then recursively adds remaining config parameters.

    Args:
        parser: argparse.ArgumentParser
        config: Config dict (typically from get_template_config())
    """
    reserved_acronyms, reserved_keys = add_reserved_arguments(
        parser, command=command, help_all=help_all, group_map=group_map
    )
    add_arguments_recursively(
        parser,
        config,
        prefix="",
        acronyms=reserved_acronyms,
        skip_keys=reserved_keys,
        command=command,
        help_all=help_all,
        group_map=group_map,
    )


def add_arguments_recursively(
    parser,
    config,
    prefix="",
    acronyms=None,
    skip_keys=None,
    command: Optional[str] = None,
    help_all: bool = False,
    group_map=None,
):
    """Recursively add CLI arguments for config parameters.

    Args:
        parser: argparse.ArgumentParser
        config: Config dict to process
        prefix: Current key prefix (e.g., "live.")
        acronyms: Set of already-used acronyms to avoid collisions
        skip_keys: Set of full config keys to skip (already added by reserved args)
    """
    if acronyms is None:
        acronyms = set()
    if skip_keys is None:
        skip_keys = set()

    for key in sorted(config):
        value = config[key]
        full_name = f"{prefix}{key}"

        # Skip if this key was already added as a reserved argument
        if full_name in skip_keys:
            continue

        if isinstance(value, dict):
            if any(full_name.endswith(x) for x in ["approved_coins", "ignored_coins"]):
                # Handle coin dict configs as comma-separated values
                acronym = create_acronym(full_name, acronyms)
                visible_group = classify_config_argument(full_name, command, help_all)
                container = (
                    group_map.get(visible_group, parser) if group_map and visible_group else parser
                )
                _register_argument(
                    container,
                    [f"--{full_name}"],
                    [f"--{full_name.replace('.', '_')}", f"-{acronym}"],
                    type=comma_separated_values,
                    dest=full_name,
                    required=False,
                    default=None,
                    metavar="CSV",
                    help=(
                        "Override "
                        f"{full_name}."
                        if help_all or command is None
                        else argparse.SUPPRESS
                    ),
                )
                acronyms.add(acronym)
                continue
            add_arguments_recursively(
                parser,
                value,
                f"{full_name}.",
                acronyms=acronyms,
                skip_keys=skip_keys,
                command=command,
                help_all=help_all,
                group_map=group_map,
            )
            continue
        else:
            acronym = create_acronym(full_name, acronyms)
            appendix = ""
            type_ = type(value)
            if "bounds" in full_name:
                type_ = comma_separated_values_float
            if "limits" in full_name:
                type_ = str
                appendix = 'Example: "--loss_profit_ratio 0.5 --drawdown_worst 0.3333"'
            elif any([x in full_name for x in ["approved_coins", "ignored_coins", "exchanges"]]):
                type_ = comma_separated_values
                appendix = "item1,item2,item3,..."
            elif "scoring" in full_name:
                type_ = comma_separated_values
                appendix = "Examples: adg,sharpe_ratio; mdg,sortino_ratio; ..."
            elif value is None:
                if full_name == "backtest.btc_collateral_ltv_cap":
                    type_ = optional_float
                else:
                    type_ = str
            elif type_ == bool:
                type_ = str2bool
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                type_ = float
            if "combine_ohlcvs" in full_name:
                appendix = (
                    "If true, combine ohlcvs data from all exchanges into single numpy array, otherwise backtest each exchange separately. "
                    + appendix
                )
            visible_group = classify_config_argument(full_name, command, help_all)
            container = group_map.get(visible_group, parser) if group_map and visible_group else parser
            _register_argument(
                container,
                [f"--{full_name}"],
                [f"--{full_name.replace('.', '_')}", f"-{acronym}"],
                type=type_,
                dest=full_name,
                required=False,
                default=None,
                metavar=_argument_metavar(type_, full_name, value),
                help=(
                    _argument_help_text(full_name, appendix)
                    if help_all or command is None
                    else argparse.SUPPRESS
                ),
            )
            acronyms.add(acronym)


def recursive_config_update(config, key, value, path=None, verbose=False):
    if path is None:
        path = []

    def _coerce_value(original, new_value):
        if isinstance(original, bool):
            return bool(new_value)
        if isinstance(original, int) and not isinstance(original, bool):
            if isinstance(new_value, (int, float)):
                if isinstance(new_value, float) and not float(new_value).is_integer():
                    return float(new_value)
                return int(round(new_value))
        if isinstance(original, float):
            if isinstance(new_value, (int, float)):
                return float(new_value)
        return new_value

    if key in config:
        coerced_value = _coerce_value(config[key], value)
        if coerced_value != config[key]:
            full_path = ".".join(path + [key])
            old_value = deepcopy(config[key])
            _log_config(
                verbose, logging.INFO, "changed %s %s -> %s", full_path, config[key], coerced_value
            )
            config[key] = coerced_value
            return {"path": full_path, "old": old_value, "new": deepcopy(coerced_value)}
        return None

    key_split = key.split(".")
    if key_split[0] in config:
        new_path = path + [key_split[0]]
        return recursive_config_update(
            config[key_split[0]], ".".join(key_split[1:]), value, new_path, verbose=verbose
        )

    return None


def update_config_with_args(config, args, verbose=False):
    changed_keys = []
    diffs = []
    for key, value in vars(args).items():
        if value is None:
            continue
        if key in {"live.approved_coins", "live.ignored_coins"}:
            normalized = normalize_coins_source(value)
            change = recursive_config_update(config, key, normalized, verbose=verbose)
            source_key = key.split(".")[-1]
            config.setdefault("_coins_sources", {})[source_key] = deepcopy(normalized)
            if change:
                changed_keys.append(key)
                diffs.append(change)
            continue
        change = recursive_config_update(config, key, value, verbose=verbose)
        if change:
            changed_keys.append(key)
            diffs.append(change)
    if changed_keys:
        details = {"keys": changed_keys}
        if diffs:
            details["diffs"] = diffs
        record_transform(config, "update_config_with_args", details)


def require_config_value(config: dict, dotted_path: str):
    parts = dotted_path.split(".")
    if not parts:
        raise KeyError("empty dotted_path")
    current = config
    traversed = []
    for part in parts:
        traversed.append(part)
        if not isinstance(current, dict):
            raise KeyError(
                f"config path {'/'.join(traversed[:-1])} is not a dict (required for '{dotted_path}')"
            )
        if part not in current:
            raise KeyError(f"config missing required key '{'.'.join(traversed)}'")
        current = current[part]
    return current


def require_config_dict(config: dict, dotted_path: str) -> dict:
    value = require_config_value(config, dotted_path)
    if not isinstance(value, dict):
        raise TypeError(f"config.{dotted_path} must be a dict; got {type(value).__name__}")
    return value


def get_optional_config_value(config: dict, dotted_path: str, default=None):
    parts = dotted_path.split(".")
    if not parts:
        return default
    current = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def require_live_value(config: dict, key: str):
    return require_config_value(config, f"live.{key}")


def get_optional_live_value(config: dict, key: str, default=None):
    return get_optional_config_value(config, f"live.{key}", default)


def get_template_config():
    return {
        "backtest": {
            "aggregate": {"default": "mean"},
            "balance_sample_divider": 60,
            "base_dir": "backtests",
            "btc_collateral_cap": 1.0,
            "btc_collateral_ltv_cap": None,
            "compress_cache": True,
            "coin_sources": {},
            "ohlcv_source_dir": None,
            "market_settings_sources": {},
            "end_date": "now",
            "exchanges": ["binance", "bybit", "gateio", "bitget"],
            "filter_by_min_effective_cost": None,
            "dynamic_wel_by_tradability": True,
            "candle_interval_minutes": 1,
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "maker_fee_override": None,
            "max_warmup_minutes": 0.0,
            "scenarios": [],
            "start_date": "2021-04-01",
            "starting_balance": 100000.0,
            "suite_enabled": True,
            "volume_normalization": True,
        },
        "bot": {
            "long": {
                "close_grid_markup_end": 0.0089,
                "close_grid_markup_start": 0.0344,
                "close_grid_qty_pct": 0.125,
                "close_trailing_grid_ratio": 0.5,
                "close_trailing_qty_pct": 0.125,
                "close_trailing_retracement_pct": 0.002,
                "close_trailing_threshold_pct": 0.008,
                "ema_span_0": 1318.0,
                "ema_span_1": 1435.0,
                "entry_grid_double_down_factor": 0.894,
                "entry_grid_spacing_pct": 0.04,
                "entry_grid_spacing_volatility_weight": 0.0,
                "entry_grid_spacing_we_weight": 0.697,
                "entry_initial_ema_dist": -0.00738,
                "entry_initial_qty_pct": 0.00592,
                "entry_trailing_double_down_factor": 0.894,
                "entry_trailing_grid_ratio": 0.5,
                "entry_trailing_retracement_pct": 0.01,
                "entry_trailing_retracement_volatility_weight": 0.0,
                "entry_trailing_retracement_we_weight": 0.0,
                "entry_trailing_threshold_pct": 0.05,
                "entry_trailing_threshold_volatility_weight": 0.0,
                "entry_trailing_threshold_we_weight": 0.0,
                "entry_volatility_ema_span_hours": 72,
                "filter_volatility_drop_pct": 0.0,
                "filter_volatility_ema_span": 60.0,
                "filter_volume_drop_pct": 0.95,
                "filter_volume_ema_span": 60.0,
                "n_positions": 10.0,
                "risk_twel_enforcer_threshold": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "risk_wel_enforcer_threshold": 1.0,
                "total_wallet_exposure_limit": 1.7,
                "unstuck_close_pct": 0.001,
                "unstuck_ema_dist": 0.0,
                "unstuck_loss_allowance_pct": 0.03,
                "unstuck_threshold": 0.916,
            },
            "short": {
                "close_grid_markup_end": 0.0089,
                "close_grid_markup_start": 0.0344,
                "close_grid_qty_pct": 0.125,
                "close_trailing_grid_ratio": 0.5,
                "close_trailing_qty_pct": 0.125,
                "close_trailing_retracement_pct": 0.002,
                "close_trailing_threshold_pct": 0.008,
                "ema_span_0": 1318.0,
                "ema_span_1": 1435.0,
                "entry_grid_double_down_factor": 0.894,
                "entry_grid_spacing_pct": 0.04,
                "entry_grid_spacing_volatility_weight": 0.0,
                "entry_grid_spacing_we_weight": 0.697,
                "entry_initial_ema_dist": -0.00738,
                "entry_initial_qty_pct": 0.00592,
                "entry_trailing_double_down_factor": 0.894,
                "entry_trailing_grid_ratio": 0.5,
                "entry_trailing_retracement_pct": 0.01,
                "entry_trailing_retracement_volatility_weight": 0.0,
                "entry_trailing_retracement_we_weight": 0.0,
                "entry_trailing_threshold_pct": 0.05,
                "entry_trailing_threshold_volatility_weight": 0.0,
                "entry_trailing_threshold_we_weight": 0.0,
                "entry_volatility_ema_span_hours": 72,
                "filter_volatility_drop_pct": 0.0,
                "filter_volatility_ema_span": 60.0,
                "filter_volume_drop_pct": 0.95,
                "filter_volume_ema_span": 60.0,
                "n_positions": 10.0,
                "risk_twel_enforcer_threshold": 1.0,
                "risk_we_excess_allowance_pct": 0.0,
                "risk_wel_enforcer_threshold": 1.0,
                "total_wallet_exposure_limit": 1.7,
                "unstuck_close_pct": 0.001,
                "unstuck_ema_dist": 0.0,
                "unstuck_loss_allowance_pct": 0.03,
                "unstuck_threshold": 0.916,
            },
        },
        "coin_overrides": {},
        "live": {
            "approved_coins": {"long": [], "short": []},
            "auto_gs": True,
            "balance_hysteresis_snap_pct": 0.02,
            "balance_override": None,
            "candle_lock_timeout_seconds": 10.0,
            "empty_means_all_approved": False,
            "enable_archive_candle_fetch": False,
            "execution_delay_seconds": 2.0,
            "filter_by_min_effective_cost": True,
            "forced_mode_long": "",
            "forced_mode_short": "",
            "hedge_mode": True,
            "ignored_coins": {"long": [], "short": []},
            "inactive_coin_candle_ttl_minutes": 10.0,
            "leverage": 10.0,
            "market_orders_allowed": True,
            "max_disk_candles_per_symbol_per_tf": 2000000,
            "max_memory_candles_per_symbol": 20000,
            "max_n_cancellations_per_batch": 5,
            "max_n_creations_per_batch": 3,
            "max_n_restarts_per_day": 10,
            "max_ohlcv_fetches_per_minute": 30,
            "max_realized_loss_pct": 1.0,
            "equity_hard_stop_loss": {
                "enabled": False,
                "threshold": 0.25,
                "ema_span_minutes": 60.0,
                "cooldown_minutes_after_red": 0.0,
                "no_restart_threshold": 1.0,
                "tier_ratios": {
                    "yellow": 0.5,
                    "orange": 0.75,
                },
                "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                "panic_close_order_type": "market",
            },
            "max_warmup_minutes": 0.0,
            "minimum_coin_age_days": 7.0,
            "order_match_tolerance_pct": 0.0002,
            "pnls_max_lookback_days": 30.0,
            "price_distance_threshold": 0.002,
            "recv_window_ms": 5000,
            "time_in_force": "good_till_cancelled",
            "user": "bybit_01",
            "warmup_jitter_seconds": 30.0,
            "warmup_ratio": 0.2,
            "warmup_concurrency": 0,
            "max_concurrent_api_requests": None,
        },
        "logging": {
            "level": 1,
            "memory_snapshot_interval_minutes": 30.0,
            "volume_refresh_info_threshold_seconds": 30.0,
        },
        "optimize": {
            "bounds": {
                "long_close_grid_markup_end": [0.001, 0.03],
                "long_close_grid_markup_start": [0.001, 0.03],
                "long_close_grid_qty_pct": [0.05, 1.0],
                "long_close_trailing_grid_ratio": [-1.0, 1.0],
                "long_close_trailing_qty_pct": [0.05, 1.0],
                "long_close_trailing_retracement_pct": [0.001, 0.1],
                "long_close_trailing_threshold_pct": [0.001, 0.1],
                "long_ema_span_0": [200.0, 1440.0],
                "long_ema_span_1": [200.0, 1440.0],
                "long_entry_grid_double_down_factor": [0.1, 3.0],
                "long_entry_grid_spacing_pct": [0.005, 0.12],
                "long_entry_grid_spacing_volatility_weight": [0.0, 400.0],
                "long_entry_grid_spacing_we_weight": [0.0, 20.0],
                "long_entry_initial_ema_dist": [-0.1, 0.002],
                "long_entry_initial_qty_pct": [0.005, 0.1],
                "long_entry_trailing_double_down_factor": [0.1, 3.0],
                "long_entry_trailing_grid_ratio": [-1.0, 1.0],
                "long_entry_trailing_retracement_pct": [0.001, 0.1],
                "long_entry_trailing_retracement_volatility_weight": [0.0, 400.0],
                "long_entry_trailing_retracement_we_weight": [0.0, 20.0],
                "long_entry_trailing_threshold_pct": [0.001, 0.1],
                "long_entry_trailing_threshold_volatility_weight": [0.0, 400.0],
                "long_entry_trailing_threshold_we_weight": [0.0, 20.0],
                "long_entry_volatility_ema_span_hours": [24.0, 336.0],
                "long_filter_volatility_drop_pct": [0.0, 1.0],
                "long_filter_volatility_ema_span": [10.0, 1440.0],
                "long_filter_volume_drop_pct": [0.0, 1.0],
                "long_filter_volume_ema_span": [10.0, 1440.0],
                "long_n_positions": [1.0, 20.0],
                "long_risk_twel_enforcer_threshold": [0.9, 1.01],
                "long_risk_we_excess_allowance_pct": [0.0, 0.5],
                "long_risk_wel_enforcer_threshold": [0.8, 1.05],
                "long_total_wallet_exposure_limit": [0.0, 2.0],
                "long_unstuck_close_pct": [0.001, 0.1],
                "long_unstuck_ema_dist": [-0.1, 0.01],
                "long_unstuck_loss_allowance_pct": [0.001, 0.05],
                "long_unstuck_threshold": [0.4, 0.95],
                "short_close_grid_markup_end": [0.001, 0.03],
                "short_close_grid_markup_start": [0.001, 0.03],
                "short_close_grid_qty_pct": [0.05, 1.0],
                "short_close_trailing_grid_ratio": [-1.0, 1.0],
                "short_close_trailing_qty_pct": [0.05, 1.0],
                "short_close_trailing_retracement_pct": [0.001, 0.1],
                "short_close_trailing_threshold_pct": [0.001, 0.1],
                "short_ema_span_0": [200.0, 1440.0],
                "short_ema_span_1": [200.0, 1440.0],
                "short_entry_grid_double_down_factor": [0.1, 3.0],
                "short_entry_grid_spacing_pct": [0.005, 0.12],
                "short_entry_grid_spacing_volatility_weight": [0.0, 400.0],
                "short_entry_grid_spacing_we_weight": [0.0, 20.0],
                "short_entry_initial_ema_dist": [-0.1, 0.002],
                "short_entry_initial_qty_pct": [0.005, 0.1],
                "short_entry_trailing_double_down_factor": [0.1, 3.0],
                "short_entry_trailing_grid_ratio": [-1.0, 1.0],
                "short_entry_trailing_retracement_pct": [0.001, 0.1],
                "short_entry_trailing_retracement_volatility_weight": [0.0, 400.0],
                "short_entry_trailing_retracement_we_weight": [0.0, 20.0],
                "short_entry_trailing_threshold_pct": [0.001, 0.1],
                "short_entry_trailing_threshold_volatility_weight": [0.0, 400.0],
                "short_entry_trailing_threshold_we_weight": [0.0, 20.0],
                "short_entry_volatility_ema_span_hours": [24.0, 336.0],
                "short_filter_volatility_drop_pct": [0.0, 1.0],
                "short_filter_volatility_ema_span": [10.0, 1440.0],
                "short_filter_volume_drop_pct": [0.0, 1.0],
                "short_filter_volume_ema_span": [10.0, 1440.0],
                "short_n_positions": [1.0, 20.0],
                "short_risk_twel_enforcer_threshold": [0.9, 1.01],
                "short_risk_we_excess_allowance_pct": [0.0, 0.5],
                "short_risk_wel_enforcer_threshold": [0.8, 1.05],
                "short_total_wallet_exposure_limit": [0.0, 2.0],
                "short_unstuck_close_pct": [0.001, 0.1],
                "short_unstuck_ema_dist": [-0.1, 0.01],
                "short_unstuck_loss_allowance_pct": [0.001, 0.05],
                "short_unstuck_threshold": [0.4, 0.95],
            },
            "backend": "deap",
            "compress_results_file": True,
            "crossover_eta": 20.0,
            "crossover_probability": 0.7,
            "enable_overrides": [],
            "iters": 30000,
            "limits": [
                {
                    "metric": "drawdown_worst_usd",
                    "penalize_if": "greater_than",
                    "value": 0.85,
                    "enabled": True
                },
                {
                    "metric": "loss_profit_ratio",
                    "penalize_if": "greater_than",
                    "value": 0.85,
                    "enabled": True
                },
                {
                    "metric": "adg_pnl",
                    "penalize_if": "less_than",
                    "stat": "mean",
                    "value": 0,
                    "enabled": False
                },
                {
                    "metric": "peak_recovery_hours_pnl",
                    "penalize_if": "greater_than",
                    "value": 1680,
                    "enabled": False
                },
                {
                    "metric": "position_held_hours_max",
                    "penalize_if": "greater_than",
                    "value": 840,
                    "enabled": False
                }
            ],
            "mutation_eta": 20.0,
            "mutation_indpb": 0.0,
            "mutation_probability": 0.45,
            "n_cpus": 5,
            "offspring_multiplier": 1.0,
            "pareto_max_size": 300,
            "population_size": 1000,
            "round_to_n_significant_digits": 5,
            "scoring": ["adg", "sharpe_ratio"],
            "write_all_results": True,
        },
    }
