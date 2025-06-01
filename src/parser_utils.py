import os
from copy import deepcopy
from typing import Any, Dict, Tuple, List
from procedures import load_config
import argparse
import pprint


Path = Tuple[str, ...]  # ("bot", "long", "entry_grid_spacing_pct")


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
    Returns modifications applied to src where allowed, either as full result or diff only.

    Args:
        src: Source dictionary to modify
        modifications: Dictionary of modifications to apply
        allowed_overrides: Dictionary specifying which modifications are allowed
                          (True = allow, False = don't allow, missing = don't allow)
        return_full: If True, returns full deepcopy of src with modifications applied.
                    If False, returns only the allowed modifications (filtered diff).

    Returns:
        If return_full=True: Deep copy of src with allowed modifications applied
        If return_full=False: Dictionary containing only the allowed modifications
    """
    if return_full:
        result = deepcopy(src)
        target = result
    else:
        result = {}
        target = result

    def _apply_recursive(target_dict, mod_dict, allowed_dict, src_dict=None):
        for key, mod_value in mod_dict.items():
            # Check if this key is allowed to be modified
            if key not in allowed_dict:
                continue  # Skip if not explicitly allowed

            allowed_value = allowed_dict[key]

            if isinstance(allowed_value, dict) and isinstance(mod_value, dict):
                # Recursive case: both modification and allowed are dicts
                if return_full:
                    # Full mode: work with existing structure or create new
                    if key in target_dict and isinstance(target_dict[key], dict):
                        _apply_recursive(
                            target_dict[key],
                            mod_value,
                            allowed_value,
                            src_dict[key] if src_dict and key in src_dict else None,
                        )
                    elif _has_allowed_values(
                        allowed_value
                    ):  # Only create if some nested value is allowed
                        target_dict[key] = {}
                        _apply_recursive(target_dict[key], mod_value, allowed_value, None)
                else:
                    # Diff mode: only include if there are allowed nested modifications
                    if _has_allowed_values(allowed_value):
                        if key not in target_dict:
                            target_dict[key] = {}
                        _apply_recursive(
                            target_dict[key],
                            mod_value,
                            allowed_value,
                            src_dict[key] if src_dict and key in src_dict else None,
                        )
            elif allowed_value is True:
                # Base case: modification is explicitly allowed
                target_dict[key] = deepcopy(mod_value)
            # If allowed_value is False, skip this modification

    def _has_allowed_values(allowed_dict):
        """Helper to check if any nested value in allowed_dict is True"""
        for value in allowed_dict.values():
            if value is True:
                return True
            elif isinstance(value, dict) and _has_allowed_values(value):
                return True
        return False

    _apply_recursive(target, modifications, allowed_overrides, src if return_full else None)
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
                "entry_grid_spacing_weight": True,
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
                "entry_grid_spacing_weight": True,
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


def parse_overrides(config):
    result = deepcopy(config)
    if not result.get("coin_overrides", {}):
        result["coin_overrides"] = parse_old_coin_flags(config)
    result["live"].pop("coin_flags", None) if "live" in result else None
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
    return result


def load_override_config(config, coin):
    try:
        path = config.get("coin_overrides", {}).get(coin, {}).get("override_config_path")
        if path and os.path.exists(path):
            return load_config(path)
        else:
            base_config_path = config.get("live", {}).get("base_config_path")
            if base_config_path and os.path.exists(
                (
                    npath := os.path.join(
                        os.path.dirname(base_config_path),
                        path,
                    )
                )
            ):
                return load_config(npath)
    except Exception as e:
        print(f"error loading config {path} {e}")
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
