import os
import re
from copy import deepcopy
from typing import Any, Dict, Tuple, List, Union, Optional, Iterable
import argparse
import logging
import hjson
from pure_funcs import remove_OD, sort_dict_keys, str2bool
from procedures import dump_pretty_json
from utils import format_end_date, symbol_to_coin, normalize_coins_source


Path = Tuple[str, ...]  # ("bot", "long", "entry_grid_spacing_pct")


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
        config = load_hjson_config(filepath)
        config = format_config(
            config, live_only=live_only, verbose=verbose, base_config_path=filepath
        )
        return config
    except Exception:
        logging.exception("failed to load config %s", filepath)
        raise


def dump_config(config: dict, filepath: str):
    config_ = deepcopy(config)
    try:
        dump_pretty_json(config_, filepath)
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
                "enforce_exposure_limit": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_grid_spacing_log_span_hours": True,
                "entry_grid_spacing_log_weight": True,
                "entry_grid_spacing_we_weight": True,
                "entry_initial_ema_dist": True,
                "entry_initial_qty_pct": True,
                "entry_trailing_double_down_factor": True,
                "entry_trailing_grid_ratio": True,
                "entry_trailing_retracement_pct": True,
                "entry_trailing_threshold_pct": True,
                "unstuck_close_pct": True,
                "unstuck_ema_dist": True,
                "unstuck_threshold": True,
                "wallet_exposure_limit": True,
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
                "enforce_exposure_limit": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_grid_spacing_log_span_hours": True,
                "entry_grid_spacing_log_weight": True,
                "entry_grid_spacing_we_weight": True,
                "entry_initial_ema_dist": True,
                "entry_initial_qty_pct": True,
                "entry_trailing_double_down_factor": True,
                "entry_trailing_grid_ratio": True,
                "entry_trailing_retracement_pct": True,
                "entry_trailing_threshold_pct": True,
                "unstuck_close_pct": True,
                "unstuck_ema_dist": True,
                "unstuck_threshold": True,
                "wallet_exposure_limit": True,
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
        if verbose:
            if result["coin_overrides"]:
                logging.info(
                    "Converted old coin_flags to coin_overrides: %s -> %s",
                    config.get("live", {}).get("coin_flags"),
                    result["coin_overrides"],
                )
    result["live"].pop("coin_flags", None) if "live" in result else None
    for coin in sorted(result["coin_overrides"]):
        coinf = symbol_to_coin(coin)
        if coinf != coin:
            if coinf:
                result["coin_overrides"][coinf] = deepcopy(result["coin_overrides"][coin])
                if verbose:
                    logging.info("Renamed %s -> %s for coin_overrides", coin, coinf)
            else:
                if verbose:
                    logging.info("Failed to format %s; removed from coin_overrides", coin)
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
        if verbose:
            logging.info("Added overrides for %s: %s", coin, sort_dict_keys(parsed_overrides))
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
    "filter_noisiness_rolling_window": "filter_log_range_ema_span",
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
            "entry_trailing_threshold_pct",
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
    "filter_noisiness_rolling_window": "filter_log_range_ema_span",
    "filter_noisiness_ema_span": "filter_log_range_ema_span",
    "filter_volume_rolling_window": "filter_volume_ema_span",
}

LEGACY_ENTRY_GRID_KEYS = {
    "entry_grid_spacing_weight": "entry_grid_spacing_we_weight",
}

LEGACY_BOUNDS_KEYS = {
    "long_filter_noisiness_rolling_window": "long_filter_log_range_ema_span",
    "long_filter_noisiness_ema_span": "long_filter_log_range_ema_span",
    "long_filter_volume_rolling_window": "long_filter_volume_ema_span",
    "short_filter_noisiness_rolling_window": "short_filter_log_range_ema_span",
    "short_filter_noisiness_ema_span": "short_filter_log_range_ema_span",
    "short_filter_volume_rolling_window": "short_filter_volume_ema_span",
    "long_entry_grid_spacing_weight": "long_entry_grid_spacing_we_weight",
    "short_entry_grid_spacing_weight": "short_entry_grid_spacing_we_weight",
}


def _apply_backward_compatibility_renames(result: dict, verbose: bool = True) -> None:
    """Translate legacy rolling_window keys to their EMA-span counterparts."""

    for pside, bot_cfg in result.get("bot", {}).items():
        if not isinstance(bot_cfg, dict):
            continue
        for old, new in LEGACY_FILTER_KEYS.items():
            if old in bot_cfg:
                if new not in bot_cfg:
                    bot_cfg[new] = bot_cfg[old]
                    if verbose:
                        print(f"renaming parameter bot.{pside}.{old}: {new}")
                del bot_cfg[old]
        for old, new in LEGACY_ENTRY_GRID_KEYS.items():
            if old in bot_cfg:
                if new not in bot_cfg:
                    bot_cfg[new] = bot_cfg[old]
                    if verbose:
                        print(f"renaming parameter bot.{pside}.{old}: {new}")
                del bot_cfg[old]

    bounds = result.get("optimize", {}).get("bounds", {})
    for old, new in LEGACY_BOUNDS_KEYS.items():
        if old in bounds:
            if new not in bounds:
                bounds[new] = bounds[old]
                if verbose:
                    print(f"renaming parameter optimize.bounds.{old}: {new}")
            del bounds[old]


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
    if all(k in config for k in template):
        return "current"
    if "config" in config and all(k in config["config"] for k in template):
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


def _ensure_bot_defaults_and_bounds(result: dict, verbose: bool = True) -> None:
    """Ensure required bot defaults and optimize bounds exist for each position side."""
    for pside in ("long", "short"):
        for k0, v_bt, v_opt in [
            ("close_trailing_qty_pct", 1.0, [0.05, 1.0]),
            (
                "entry_trailing_double_down_factor",
                result["bot"][pside].get("entry_grid_double_down_factor", 1.0),
                [0.01, 3.0],
            ),
            (
                "filter_log_range_ema_span",
                result["bot"][pside].get(
                    "filter_log_range_ema_span",
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
                result["optimize"]["bounds"].get(f"{pside}_min_markup", [0.001, 0.03]),
            ),
            (
                "close_grid_markup_end",
                result["bot"][pside].get("close_grid_min_markup", 0.001),
                result["optimize"]["bounds"].get(f"{pside}_close_grid_min_markup", [0.001, 0.03]),
            ),
            (
                "filter_volume_drop_pct",
                result["live"].get("filter_relative_volume_clip_pct", 0.5),
                [0.0, 1.0],
            ),
        ]:
            if k0 not in result["bot"][pside]:
                result["bot"][pside][k0] = v_bt
                if verbose:
                    print(f"adding missing backtest parameter {pside} {k0}: {v_bt}")
            opt_key = f"{pside}_{k0}"
            if opt_key not in result["optimize"]["bounds"]:
                result["optimize"]["bounds"][opt_key] = v_opt
                if verbose:
                    print(f"adding missing optimize parameter {pside} {opt_key}: {v_opt}")


def _rename_config_keys(result: dict, verbose: bool = True) -> None:
    """Rename legacy keys to their current names."""
    for section, src, dst in [
        ("live", "minimum_market_age_days", "minimum_coin_age_days"),
        ("live", "noisiness_rolling_mean_window_size", "ohlcv_rolling_window"),
        ("live", "ohlcvs_1m_update_after_minutes", "inactive_coin_candle_ttl_minutes"),
    ]:
        if src in result[section]:
            result[section][dst] = deepcopy(result[section][src])
            if verbose:
                print(f"renaming parameter {section} {src}: {dst}")
            del result[section][src]
    if "exchange" in result["backtest"] and isinstance(result["backtest"]["exchange"], str):
        exchange = result["backtest"]["exchange"]
        result["backtest"]["exchanges"] = [exchange]
        if verbose:
            print(f"changed backtest.exchange: {exchange} -> backtest.exchanges: [{exchange}]")
        del result["backtest"]["exchange"]


def _sync_with_template(
    template: dict, result: dict, base_config_path: str, verbose: bool = True
) -> None:
    """Synchronize the config with the template structure and prune unused keys."""
    add_missing_keys_recursively(template, result, verbose=verbose)
    if base_config_path or "base_config_path" not in result["live"]:
        result["live"]["base_config_path"] = base_config_path
    template_with_extras = deepcopy(template)
    template_with_extras.setdefault("live", {})["base_config_path"] = ""
    remove_unused_keys_recursively(
        template_with_extras, result, verbose=verbose, preserve=[("coin_overrides",)]
    )
    remove_unused_keys_recursively(template["bot"], result["bot"], verbose=verbose)
    remove_unused_keys_recursively(
        template["optimize"]["bounds"], result["optimize"]["bounds"], verbose=verbose
    )
    remove_unused_keys_recursively(
        template.get("optimize", {}).get("limits", {}),
        result["optimize"].setdefault("limits", {}),
        verbose=verbose,
    )


def _normalize_position_counts(result: dict) -> None:
    """Round position counts to integers for each side."""
    for pside in result["bot"]:
        result["bot"][pside]["n_positions"] = int(round(result["bot"][pside]["n_positions"]))


def _apply_non_live_adjustments(result: dict, verbose: bool = True) -> None:
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
    result["optimize"]["scoring"] = sorted(result["optimize"]["scoring"])
    result["optimize"]["limits"] = parse_limits_string(result["optimize"]["limits"])
    for key, value in sorted(result["optimize"]["limits"].items()):
        if key.startswith("lower_bound_"):
            new_key = key.replace("lower_bound_", "penalize_if_greater_than_")
            result["optimize"]["limits"][new_key] = value
            if verbose:
                print(f"changed config.optimize.limits.{key} -> {new_key}")
            del result["optimize"]["limits"][key]
    if not result["backtest"]["use_btc_collateral"]:
        for idx, value in enumerate(result["optimize"]["scoring"]):
            if value.startswith("btc_"):
                new_value = value[len("btc_") :]
                if verbose:
                    print(f"changed config.optimize.scoring.{value} -> {new_value}")
                result["optimize"]["scoring"][idx] = new_value
        for key in sorted(result["optimize"]["limits"]):
            if key.startswith("btc_"):
                new_key = key[len("btc_") :]
                limit_value = result["optimize"]["limits"][key]
                if verbose:
                    print(f"changed config.optimize.limits.{key} -> {new_key}")
                result["optimize"]["limits"][new_key] = limit_value
                del result["optimize"]["limits"][key]
    for key, value in sorted(result["optimize"]["bounds"].items()):
        if isinstance(value, list):
            if len(value) == 1:
                result["optimize"]["bounds"][key] = [value[0], value[0]]
            elif len(value) == 2:
                result["optimize"]["bounds"][key] = sorted(value)


def _ensure_enforce_exposure_limit_bool(result: dict) -> None:
    """Ensure enforce_exposure_limit is a bool for each side."""
    for pside in result["bot"]:
        result["bot"][pside]["enforce_exposure_limit"] = bool(
            result["bot"][pside]["enforce_exposure_limit"]
        )


def format_config(config: dict, verbose=True, live_only=False, base_config_path: str = "") -> dict:
    # attempts to format a config to v7 config
    template = get_template_config("v7")
    flavor = detect_flavor(config, template)
    result = build_base_config_from_flavor(config, template, flavor, verbose)
    _apply_backward_compatibility_renames(result, verbose=verbose)
    _ensure_bot_defaults_and_bounds(result, verbose=verbose)
    result["bot"] = sort_dict_keys(result["bot"])

    _rename_config_keys(result, verbose=verbose)

    _sync_with_template(template, result, base_config_path, verbose=verbose)

    _normalize_position_counts(result)

    if not live_only:
        # unneeded adjustments if running live
        _apply_non_live_adjustments(result, verbose=verbose)

    _ensure_enforce_exposure_limit_bool(result)
    return result


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


def add_missing_keys_recursively(src, dst, parent=None, verbose=True):
    if parent is None:
        parent = []
    for k in src:
        if k not in dst:
            if verbose:
                logging.info("Added missing %s to config.", ".".join(parent + [k]))
            dst[k] = src[k]
        # --- NEW: only walk down if both sides are dicts -------------
        elif isinstance(src[k], dict) and isinstance(dst.get(k), dict):
            add_missing_keys_recursively(src[k], dst[k], parent + [k], verbose)
        # --------------------------------------------------------------
        elif isinstance(src[k], dict):
            # type clash: leave the user’s value untouched
            if verbose:
                logging.info(
                    "Skipping template subtree %s (template is dict, config is %s)",
                    ".".join(parent + [k]),
                    type(dst.get(k)).__name__,
                )
            continue
        else:
            # previous branches already handle k not in dst; keep safe assignment
            if k not in dst:
                if verbose:
                    logging.info(
                        "Adding missing key -> val %s -> %s to config",
                        ".".join(parent + [k]),
                        src[k],
                    )
                dst[k] = src[k]


def remove_unused_keys_recursively(
    src, dst, parent=None, verbose=True, preserve: Optional[Iterable[Iterable[str]]] = None
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
    for k in sorted(list(dst.keys())):
        current_path = parent + [k]
        if _path_is_preserved(current_path):
            continue
        if k not in src:
            del dst[k]
            if verbose:
                logging.info("Removed unused key from config: %s", ".".join(current_path))
            continue
        src_val = src[k]
        dst_val = dst[k]
        if isinstance(dst_val, dict) and isinstance(src_val, dict):
            remove_unused_keys_recursively(src_val, dst_val, current_path, verbose=verbose)

    if parent == [] and hasattr(remove_unused_keys_recursively, "_preserve_set"):
        delattr(remove_unused_keys_recursively, "_preserve_set")


def comma_separated_values_float(x):
    return [float(z) for z in x.split(",")]


def comma_separated_values(x):
    return x.split(",")


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


def add_arguments_recursively(parser, config, prefix="", acronyms=set()):

    for key, value in config.items():
        full_name = f"{prefix}{key}"

        if isinstance(value, dict):
            if any(full_name.endswith(x) for x in ["approved_coins", "ignored_coins"]):
                acronym = "s" if full_name.endswith("approved_coins") else "i"
                if acronym in acronyms:
                    acronym = create_acronym(full_name, acronyms)
                parser.add_argument(
                    f"--{full_name}",
                    f"--{full_name.replace('.', '_')}",
                    f"-{acronym}",
                    type=comma_separated_values,
                    dest=full_name,
                    required=False,
                    default=None,
                    metavar="",
                    help=f"Override {full_name}: comma_separated_values",
                )
                acronyms.add(acronym)
                continue
            add_arguments_recursively(parser, value, f"{full_name}.", acronyms=acronyms)
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
            elif "approved_coins" in full_name:
                acronym = "s"
                type_ = comma_separated_values
            elif any([x in full_name for x in ["ignored_coins", "exchanges"]]):
                type_ = comma_separated_values
                appendix = "item1,item2,item3,..."
            elif "scoring" in full_name:
                type_ = comma_separated_values
                acronym = "os"
                appendix = "Examples: adg,sharpe_ratio; mdg,sortino_ratio; ..."
            elif "cpus" in full_name:
                acronym = "c"
            elif "iters" in full_name:
                acronym = "i"
            elif type_ == bool:
                type_ = str2bool
                appendix = "[y/n]"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                type_ = float
            if "combine_ohlcvs" in full_name:
                appendix = (
                    "If true, combine ohlcvs data from all exchanges into single numpy array, otherwise backtest each exchange separately. "
                    + appendix
                )
            parser.add_argument(
                f"--{full_name}",
                f"--{full_name.replace('.', '_')}",
                f"-{acronym}",
                type=type_,
                dest=full_name,
                required=False,
                default=None,
                metavar="",
                help=f"Override {full_name}: {str(type_.__name__)} " + appendix,
            )
            acronyms.add(acronym)


def recursive_config_update(config, key, value, path=None):
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
            print(f"changed {full_path} {config[key]} -> {coerced_value}")
            config[key] = coerced_value
        return True

    key_split = key.split(".")
    if key_split[0] in config:
        new_path = path + [key_split[0]]
        return recursive_config_update(config[key_split[0]], ".".join(key_split[1:]), value, new_path)

    return False


def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is None:
            continue
        if key in {"live.approved_coins", "live.ignored_coins"}:
            normalized = normalize_coins_source(value)
            recursive_config_update(config, key, normalized)
            continue
        recursive_config_update(config, key, value)


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


def get_template_config(passivbot_mode="v7"):
    return {
        "logging": {
            "level": 1,
        },
        "backtest": {
            "base_dir": "backtests",
            "combine_ohlcvs": True,
            "compress_cache": True,
            "end_date": "now",
            "exchanges": ["binance", "bybit", "gateio", "bitget"],
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "start_date": "2021-04-01",
            "starting_balance": 100000.0,
            "use_btc_collateral": False,
            "max_warmup_minutes": 0.0,
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
                "enforce_exposure_limit": True,
                "entry_grid_double_down_factor": 0.894,
                "entry_grid_spacing_log_span_hours": 72,
                "entry_grid_spacing_log_weight": 0.0,
                "entry_grid_spacing_pct": 0.04,
                "entry_grid_spacing_we_weight": 0.697,
                "entry_initial_ema_dist": -0.00738,
                "entry_initial_qty_pct": 0.00592,
                "entry_trailing_double_down_factor": 0.894,
                "entry_trailing_grid_ratio": 0.5,
                "entry_trailing_retracement_pct": 0.01,
                "entry_trailing_threshold_pct": 0.05,
                "filter_log_range_ema_span": 60.0,
                "filter_volume_drop_pct": 0.95,
                "filter_volume_ema_span": 60.0,
                "n_positions": 10.0,
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
                "enforce_exposure_limit": True,
                "entry_grid_double_down_factor": 0.894,
                "entry_grid_spacing_log_span_hours": 72,
                "entry_grid_spacing_log_weight": 0.0,
                "entry_grid_spacing_pct": 0.04,
                "entry_grid_spacing_we_weight": 0.697,
                "entry_initial_ema_dist": -0.00738,
                "entry_initial_qty_pct": 0.00592,
                "entry_trailing_double_down_factor": 0.894,
                "entry_trailing_grid_ratio": 0.5,
                "entry_trailing_retracement_pct": 0.01,
                "entry_trailing_threshold_pct": 0.05,
                "filter_log_range_ema_span": 60.0,
                "filter_volume_drop_pct": 0.95,
                "filter_volume_ema_span": 60.0,
                "n_positions": 10.0,
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
            "inactive_coin_candle_ttl_minutes": 10.0,
            "empty_means_all_approved": False,
            "execution_delay_seconds": 2.0,
            "filter_by_min_effective_cost": True,
            "forced_mode_long": "",
            "forced_mode_short": "",
            "ignored_coins": {"long": [], "short": []},
            "leverage": 10.0,
            "market_orders_allowed": True,
            "max_memory_candles_per_symbol": 20000,
            "max_disk_candles_per_symbol_per_tf": 2000000,
            "max_n_cancellations_per_batch": 5,
            "max_n_creations_per_batch": 3,
            "max_n_restarts_per_day": 10,
            "minimum_coin_age_days": 7.0,
            "pnls_max_lookback_days": 30.0,
            "price_distance_threshold": 0.002,
            "time_in_force": "good_till_cancelled",
            "warmup_ratio": 0.2,
            "user": "bybit_01",
        },
        "optimize": {
            "bounds": {
                "long_close_grid_markup_end": [0.001, 0.03],
                "long_close_grid_markup_start": [0.001, 0.03],
                "long_close_grid_qty_pct": [0.05, 1.0],
                "long_close_trailing_grid_ratio": [-1.0, 1.0],
                "long_close_trailing_qty_pct": [0.05, 1.0],
                "long_close_trailing_retracement_pct": [0.0, 0.1],
                "long_close_trailing_threshold_pct": [-0.1, 0.1],
                "long_ema_span_0": [200.0, 1440.0],
                "long_ema_span_1": [200.0, 1440.0],
                "long_entry_grid_double_down_factor": [0.1, 3.0],
                "long_entry_grid_spacing_pct": [0.005, 0.12],
                "long_entry_grid_spacing_log_span_hours": [24.0, 336.0],
                "long_entry_grid_spacing_log_weight": [0.0, 400.0],
                "long_entry_grid_spacing_we_weight": [0.0, 2.0],
                "long_entry_initial_ema_dist": [-0.1, 0.002],
                "long_entry_initial_qty_pct": [0.005, 0.1],
                "long_entry_trailing_double_down_factor": [0.1, 3.0],
                "long_entry_trailing_grid_ratio": [-1.0, 1.0],
                "long_entry_trailing_retracement_pct": [0.0, 0.1],
                "long_entry_trailing_threshold_pct": [-0.1, 0.1],
                "long_filter_log_range_ema_span": [10.0, 1440.0],
                "long_filter_volume_drop_pct": [0.0, 1.0],
                "long_filter_volume_ema_span": [10.0, 1440.0],
                "long_n_positions": [1.0, 20.0],
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
                "short_close_trailing_retracement_pct": [0.0, 0.1],
                "short_close_trailing_threshold_pct": [-0.1, 0.1],
                "short_ema_span_0": [200.0, 1440.0],
                "short_ema_span_1": [200.0, 1440.0],
                "short_entry_grid_double_down_factor": [0.1, 3.0],
                "short_entry_grid_spacing_pct": [0.005, 0.12],
                "short_entry_grid_spacing_log_span_hours": [24.0, 336.0],
                "short_entry_grid_spacing_log_weight": [0.0, 400.0],
                "short_entry_grid_spacing_we_weight": [0.0, 2.0],
                "short_entry_initial_ema_dist": [-0.1, 0.002],
                "short_entry_initial_qty_pct": [0.005, 0.1],
                "short_entry_trailing_double_down_factor": [0.1, 3.0],
                "short_entry_trailing_grid_ratio": [-1.0, 1.0],
                "short_entry_trailing_retracement_pct": [0.0, 0.1],
                "short_entry_trailing_threshold_pct": [-0.1, 0.1],
                "short_filter_log_range_ema_span": [10.0, 1440.0],
                "short_filter_volume_drop_pct": [0.0, 1.0],
                "short_filter_volume_ema_span": [10.0, 1440.0],
                "short_n_positions": [1.0, 20.0],
                "short_total_wallet_exposure_limit": [0.0, 2.0],
                "short_unstuck_close_pct": [0.001, 0.1],
                "short_unstuck_ema_dist": [-0.1, 0.01],
                "short_unstuck_loss_allowance_pct": [0.001, 0.05],
                "short_unstuck_threshold": [0.4, 0.95],
            },
            "compress_results_file": True,
            "crossover_probability": 0.7,
            "crossover_eta": 20.0,
            "enable_overrides": [],
            "iters": 30000,
            "limits": "--drawdown_worst 0.333 --loss_profit_ratio: 0.9 --position_unchanged_hours_max 300.0",
            "mutation_probability": 0.45,
            "mutation_eta": 20.0,
            "mutation_indpb": 0.0,
            "n_cpus": 5,
            "offspring_multiplier": 1.0,
            "population_size": 1000,
            "round_to_n_significant_digits": 5,
            "scoring": ["adg", "sharpe_ratio"],
            "write_all_results": True,
        },
    }
