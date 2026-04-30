import argparse
import logging
import os
from copy import deepcopy
from typing import Callable

from pure_funcs import sort_dict_keys
from utils import symbol_to_coin

from .load import load_prepared_config
from .log_output import log_config_message
from .transform_log import record_transform


def apply_allowed_modifications(src, modifications, allowed_overrides, return_full=True):
    if return_full:
        result = deepcopy(src)
        target = result
    else:
        result = {}
        target = result

    def _has_allowed_values(allowed_subdict):
        for value in allowed_subdict.values():
            if value is True:
                return True
            if isinstance(value, dict) and _has_allowed_values(value):
                return True
        return False

    def _apply_recursive(target_dict, mod_dict, allowed_dict, src_dict=None):
        for key, mod_value in mod_dict.items():
            if key not in allowed_dict:
                continue
            allowed_value = allowed_dict[key]
            if isinstance(allowed_value, dict) and isinstance(mod_value, dict):
                if not _has_allowed_values(allowed_value):
                    continue
                if key not in target_dict:
                    target_dict[key] = {}
                _apply_recursive(
                    target_dict[key],
                    mod_value,
                    allowed_value,
                    src_dict[key] if src_dict and key in src_dict else None,
                )
                if not return_full and not target_dict[key]:
                    target_dict.pop(key, None)
            elif allowed_value is True:
                if return_full:
                    target_dict[key] = deepcopy(mod_value)
                else:
                    src_val = src_dict.get(key) if src_dict else None
                    if src_val != mod_value:
                        target_dict[key] = deepcopy(mod_value)

    _apply_recursive(target, modifications, allowed_overrides, src if return_full else src)
    return result


def get_allowed_modifications():
    return {
        "bot": {
            "long": {
                "close_grid_markup_end": True,
                "close_grid_markup_start": True,
                "close_grid_qty_pct": True,
                "grid_close_price_anchor": True,
                "close_trailing_grid_ratio": True,
                "close_trailing_qty_pct": True,
                "close_trailing_retracement_pct": True,
                "close_trailing_threshold_pct": True,
                "close_weight_volatility_1h": True,
                "close_weight_volatility_1m": True,
                "ema_span_0": True,
                "ema_span_1": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_volatility_ema_span_hours": True,
                "entry_volatility_ema_span_minutes": True,
                "entry_weight_volatility_1h": True,
                "entry_weight_volatility_1m": True,
                "entry_we_weight": True,
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
                "risk_wel_enforcer_threshold": True,
                "risk_we_excess_allowance_pct": True,
                "risk_twel_enforcer_threshold": False,
            },
            "short": {
                "close_grid_markup_end": True,
                "close_grid_markup_start": True,
                "close_grid_qty_pct": True,
                "grid_close_price_anchor": True,
                "close_trailing_grid_ratio": True,
                "close_trailing_qty_pct": True,
                "close_trailing_retracement_pct": True,
                "close_trailing_threshold_pct": True,
                "close_weight_volatility_1h": True,
                "close_weight_volatility_1m": True,
                "ema_span_0": True,
                "ema_span_1": True,
                "entry_grid_double_down_factor": True,
                "entry_grid_spacing_pct": True,
                "entry_volatility_ema_span_hours": True,
                "entry_volatility_ema_span_minutes": True,
                "entry_weight_volatility_1h": True,
                "entry_weight_volatility_1m": True,
                "entry_we_weight": True,
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
    if not p:
        raise ValueError("Path cannot be empty")
    current = d
    for key in p[:-1]:
        current = current[key]
    current[p[-1]] = v


def set_nested_value_safe(d: dict, p: list, v: object, create_missing=False):
    if not p:
        raise ValueError("Path cannot be empty")
    current = d
    for key in p[:-1]:
        if key not in current:
            if create_missing:
                current[key] = {}
            else:
                return False
        elif not isinstance(current[key], dict):
            return False
        current = current[key]
    current[p[-1]] = v
    return True


def nested_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            nested_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def load_override_config(
    config,
    coin,
    *,
    config_loader: Callable[[str], dict] | None = None,
):
    if config_loader is None:
        config_loader = lambda path: load_prepared_config(path, verbose=False, log_info=False)
    path = None
    try:
        path = config.get("coin_overrides", {}).get(coin, {}).get("override_config_path")
        if path and os.path.exists(path):
            return config_loader(path)
        base_config_path = config.get("live", {}).get("base_config_path")
        if path and base_config_path:
            npath = os.path.join(os.path.dirname(base_config_path), path)
            if os.path.exists(npath):
                return config_loader(npath)
    except Exception as exc:
        logging.exception("error loading config %s: %s", path, exc)
    return {}


def parse_old_coin_flags(config) -> dict:
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
        if live_config_path := keysvals.get("live_config_path"):
            set_nested_value_safe(
                result[coin],
                ["override_config_path"],
                live_config_path,
                create_missing=True,
            )
        for key, value in keysvals.items():
            if value and key in key_map:
                set_nested_value_safe(result[coin], key_map[key], value, create_missing=True)
    return result


def parse_overrides(
    config,
    *,
    verbose=True,
    override_loader: Callable[[dict, str], dict] | None = None,
    symbol_normalizer: Callable[[str], str] | None = None,
):
    if override_loader is None:
        override_loader = load_override_config
    if symbol_normalizer is None:
        symbol_normalizer = symbol_to_coin
    result = deepcopy(config)
    if not result.get("coin_overrides", {}):
        result["coin_overrides"] = parse_old_coin_flags(config)
        if verbose and result["coin_overrides"]:
            log_config_message(
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
        formatted_coin = symbol_normalizer(coin)
        if formatted_coin != coin:
            if formatted_coin:
                result["coin_overrides"][formatted_coin] = deepcopy(result["coin_overrides"][coin])
                log_config_message(
                    verbose,
                    logging.INFO,
                    "Renamed %s -> %s for coin_overrides",
                    coin,
                    formatted_coin,
                )
            else:
                log_config_message(
                    verbose,
                    logging.INFO,
                    "Failed to format %s; removed from coin_overrides",
                    coin,
                )
            del result["coin_overrides"][coin]
    for coin, overrides in result["coin_overrides"].items():
        parsed_overrides = {}
        loaded = override_loader(result, coin)
        if loaded:
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
        log_config_message(
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


def _build_flag_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coin_flags", add_help=False)
    parser.add_argument("-sm", type=expand_PB_mode, dest="short_mode", default=None)
    parser.add_argument("-lm", type=expand_PB_mode, dest="long_mode", default=None)
    parser.add_argument("-lw", type=float, dest="WE_limit_long", default=None)
    parser.add_argument("-sw", type=float, dest="WE_limit_short", default=None)
    parser.add_argument("-lev", type=float, dest="leverage", default=None)
    parser.add_argument("-lc", type=str, dest="live_config_path", default=None)
    return parser


def expand_PB_mode(mode: str) -> str:
    lowered = mode.lower()
    if lowered in ["gs", "graceful_stop", "graceful-stop"]:
        return "graceful_stop"
    if lowered in ["m", "manual"]:
        return "manual"
    if lowered in ["n", "normal"]:
        return "normal"
    if lowered in ["p", "panic"]:
        return "panic"
    if lowered in ["t", "tp", "tp_only", "tp-only"]:
        return "tp_only"
    raise Exception(f"unknown passivbot mode {mode}")
