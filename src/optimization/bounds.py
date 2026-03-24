"""
Configuration adapter for optimization.

This module bridges the gap between the general configuration system and the
optimization-specific bounds logic.
"""

import logging
from copy import deepcopy
from typing import List, Tuple

import numpy as np

def parse_bound(key: str, val) -> Tuple[float, float]:
    """
    Parse a bound config value into (low, high).

    Supported formats:
    - Single value: fixed (low=high)
    - [low, high]: continuous
    - [low, high, step]: step is ignored, continuous
    - [low, high, 0/null]: continuous
    """
    if isinstance(val, (float, int)):
        return (float(val), float(val))

    if isinstance(val, (tuple, list)):
        if len(val) == 0:
            raise ValueError(f"malformed bound {key}: empty array")
        if len(val) == 1:
            return (float(val[0]), float(val[0]))
        low, high = sorted([float(val[0]), float(val[1])])
        if len(val) > 2:
            logging.debug("Bound %s: step value ignored (pymoo uses continuous bounds)", key)
        return (low, high)

    raise ValueError(f"malformed bound {key}: {val}")


def extract_bounds_arrays(config: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract bounds from config as (xl, xu, keys) arrays for pymoo.

    Iterates bot params in sorted order (long then short, keys alphabetical).
    For each param, looks up the matching bound in optimize.bounds.
    If no bound exists, the param is fixed at its current bot value.

    Returns:
        xl: lower bounds array
        xu: upper bounds array
        keys: list of param keys in order (e.g. "long_ema_span_0")
    """
    bounds_cfg = config.get("optimize", {}).get("bounds", {})
    xl_list = []
    xu_list = []
    keys = []

    for pside in sorted(config["bot"]):
        for key in sorted(config["bot"][pside]):
            full_key = f"{pside}_{key}"
            keys.append(full_key)

            if full_key in bounds_cfg:
                low, high = parse_bound(full_key, bounds_cfg[full_key])
            else:
                val = float(config["bot"][pside][key])
                low, high = val, val

            xl_list.append(low)
            xu_list.append(high)

    return np.array(xl_list), np.array(xu_list), keys


def config_to_individual(config, xl, xu):
    """Convert a bot config dict to a flat parameter vector, clamped to bounds."""
    values = [
        config["bot"][pside][key]
        for pside in sorted(config["bot"])
        for key in sorted(config["bot"][pside])
    ]
    return np.clip(values, xl, xu).tolist()


def individual_to_config(individual, overrides_fn, overrides_list, template):
    """
    Convert a flat parameter vector back to a bot config dict.

    Iterates bot params in sorted order (matching extract_bounds_arrays)
    and applies optimizer overrides per position side.
    """
    config = deepcopy(template)
    i = 0
    for pside in sorted(config["bot"]):
        for key in sorted(config["bot"][pside]):
            config["bot"][pside][key] = individual[i]
            i += 1
        if overrides_fn:
            config = overrides_fn(overrides_list, config, pside)

    return config


def apply_config_overrides(config: dict, overrides: dict) -> None:
    """Apply dotted-path overrides to a config dict in-place."""
    if not overrides:
        return
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str):
            continue
        parts = dotted_path.split(".")
        if not parts:
            continue
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value


def validate_array(arr, name, allow_nan=True):
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf values")
    if allow_nan and np.isnan(arr).all():
        raise ValueError(f"{name} is entirely NaN")


def apply_fine_tune_bounds(
    config: dict,
    fine_tune_params: list[str],
    cli_overridden_bounds: set[str],
) -> None:
    bounds = config.get("optimize", {}).get("bounds", {})
    bot_cfg = config.get("bot", {})
    # First, normalize any CLI overrides such that single values mean fixed bounds
    for key in cli_overridden_bounds:
        if key not in bounds:
            continue
        raw_val = bounds[key]
        if isinstance(raw_val, (list, tuple)):
            if len(raw_val) == 1:
                bounds[key] = [float(raw_val[0]), float(raw_val[0])]
        else:
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            bounds[key] = [val, val]

    if not fine_tune_params:
        return

    fine_tune_set = set(fine_tune_params)

    for key in list(bounds.keys()):
        if key in fine_tune_set:
            continue
        try:
            pside, param = key.split("_", 1)
        except ValueError:
            logging.warning(f"fine-tune bounds: unable to parse key '{key}', skipping")
            continue
        side_cfg = bot_cfg.get(pside)
        if not isinstance(side_cfg, dict) or param not in side_cfg:
            logging.warning(
                f"fine-tune bounds: missing bot value for '{key}', leaving bounds unchanged"
            )
            continue
        value = side_cfg[param]
        try:
            value_float = float(value)
            bounds[key] = [value_float, value_float]
        except (TypeError, ValueError):
            bounds[key] = [value, value]

    missing = [key for key in fine_tune_set if key not in bounds]
    if missing:
        logging.warning(
            "fine-tune bounds: requested keys not found in optimize bounds: %s",
            ",".join(sorted(missing)),
        )
