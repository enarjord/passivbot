import os
from copy import deepcopy
from typing import Any, Dict, Tuple, List
from procedures import load_config
import argparse


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
            },
        },
        "live": {},
        "backtest": {},
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
        print(keysvals)
        if "live_config_path" in keysvals:
            val = keysvals["live_config_path"]
            lcfg = None
            try:
                if val and os.path.exists(val):
                    lcfg = load_config(val)
                elif "base_config_path" in config["live"] and os.path.exists(
                    (
                        npath := os.path.join(
                            os.path.dirname(config["live"]["base_config_path"]),
                            val,
                        )
                    )
                ):
                    lcfg = load_config(npath)
            except Exception as e:
                print(f"error loading config {val} {e}")
            if lcfg:
                result[coin] = apply_allowed_modifications(
                    config, lcfg, get_allowed_modifications(), return_full=False
                )
        for key, val in keysvals.items():
            if key in key_map:
                set_nested_value_safe(result[coin], key_map[key], val, create_missing=True)
    return result


# ──────────────────────────────────────────────────────────────────────────
#  Coin-flags handling (ex-Passivbot.init_flags)
# ──────────────────────────────────────────────────────────────────────────


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


def _all_paths(d: Dict[str, Any], prefix: Tuple[str, ...] = ()) -> List[Path]:
    """Return every leaf-path inside *d* as a tuple of keys."""
    paths = []
    for k, v in d.items():
        new_prefix = prefix + (k,)
        if isinstance(v, dict):
            paths.extend(_all_paths(v, new_prefix))
        else:
            paths.append(new_prefix)
    return paths


def _allowed(path: Path, allowed_mask: Dict[str, Any]) -> bool:
    """Check if a path is allowed to be overridden."""
    cur = allowed_mask
    for p in path:
        if p not in cur:
            return False
        cur = cur[p]
    return bool(cur)  # must be True at the leaf


def _set_by_path(root: Dict[str, Any], path: Path, value: Any) -> None:
    """Set deep value in nested dict, creating missing levels if needed."""
    for key in path[:-1]:
        root = root.setdefault(key, {})
    root[path[-1]] = value


def apply_coin_flags(
    cfg: Dict[str, Any], allowed_coin_overrides: Dict[str, Any], quiet: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Produce a dict {coin: merged_cfg}.
    Prints what was (or was not) overridden for each coin.

    Example: merged_cfgs = apply_coin_flags(cfg, allowed_mask)
    """
    coin_flags = cfg.get("live", {}).get("coin_flags", {})
    if not coin_flags:  # nothing to do – return a single entry
        return {"_GLOBAL_": cfg}

    merged = {}
    for coin, overrides in coin_flags.items():
        c_cfg = deepcopy(cfg)  # per-coin working copy
        for section, section_over in overrides.items():
            for path in _all_paths(section_over, prefix=(section,)):
                new_val = section_over
                for p in path[1:]:  # drill down in overrides
                    new_val = new_val[p]

                if _allowed(path, allowed_coin_overrides):
                    _set_by_path(c_cfg, path, new_val)
                    if not quiet:
                        print(f"{coin}: set {'.'.join(path)} -> {new_val!r}")
                else:
                    if not quiet:
                        print(f"{coin}: NOT allowed to override " f"{'.'.join(path)}")
        merged[coin] = c_cfg

    return merged
