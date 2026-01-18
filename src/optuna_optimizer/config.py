"""Configuration loading for Optuna optimizer."""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable

from .models import Bound, Constraint, Objective, OptunaConfig


def _get_optimize_section(config: dict) -> dict:
    """Extract the optimize section from config."""
    return config.get("optimize", {})


def extract_bounds(config: dict) -> dict[str, Bound]:
    """Extract parameter bounds from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        Dictionary mapping parameter names to Bound objects.
    """
    bounds_raw = _get_optimize_section(config).get("bounds", {})
    return {name: Bound.from_config(value) for name, value in bounds_raw.items()}


def extract_constraints(config: dict) -> list[Constraint]:
    """Extract metric constraints from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        List of Constraint objects.
    """
    constraints_raw = _get_optimize_section(config).get("constraints", [])
    return [Constraint.model_validate(c) for c in constraints_raw]


def extract_optuna_config(config: dict) -> OptunaConfig:
    """Extract Optuna-specific configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        OptunaConfig object with optimizer settings.
    """
    optuna_raw = _get_optimize_section(config).get("optuna", {})
    return OptunaConfig.model_validate(optuna_raw)


def extract_objectives(config: dict) -> list[Objective]:
    """Extract optimization objectives from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        List of Objective objects with metric names and directions.
    """
    objectives_raw = _get_optimize_section(config).get("objectives", [])
    return [Objective.model_validate(o) for o in objectives_raw]


def extract_params_from_config(config: dict, bounds: dict[str, Bound]) -> dict[str, float] | None:
    """Extract flat params from config, filtered to bounds.

    Reads config.bot.long.* -> long_*
    Reads config.bot.short.* -> short_*

    Args:
        config: Full passivbot config dict
        bounds: Parameter bounds to filter against

    Returns:
        Dict of param name -> value, or None if config has no bot section.
    """
    bot = config.get("bot")
    if not bot:
        return None

    params = {}
    for pside in ("long", "short"):
        side_params = bot.get(pside, {})
        for key, value in side_params.items():
            flat_key = f"{pside}_{key}"
            if flat_key in bounds:
                bound = bounds[flat_key]
                clamped = max(bound.low, min(bound.high, value))
                params[flat_key] = clamped

    return params if params else None


def load_seed_configs(path: Path, config_loader: Callable[[str], dict]) -> list[dict]:
    """Recursively load configs from file or directory.

    Args:
        path: Path to JSON/HJSON file or directory containing config files
        config_loader: Function that loads a config file path and returns a dict

    Returns:
        List of parsed config dicts. Invalid/missing files are skipped.
    """
    if not path.exists():
        return []

    if path.is_file():
        if path.suffix not in (".json", ".hjson"):
            return []
        try:
            return [config_loader(str(path))]
        except Exception as e:
            logging.warning(f"Failed to load seed config {path}: {e}")
            return []

    # Directory: recurse
    configs = []
    for child in sorted(path.iterdir()):
        configs.extend(load_seed_configs(child, config_loader))
    return configs


def apply_params_to_config(params: dict[str, float], base_config: dict) -> dict:
    """Apply sampled parameters to a config.

    Routes parameters based on prefix:
    - long_* -> config.bot.long.*
    - short_* -> config.bot.short.*
    - other -> config.live.*

    Args:
        params: Dictionary of parameter names to values
        base_config: Base configuration to merge into

    Returns:
        New config dict with params applied (does not mutate base_config)
    """
    config = deepcopy(base_config)
    bot = config.setdefault("bot", {})

    for name, value in params.items():
        if name.startswith("long_"):
            bot.setdefault("long", {})[name[5:]] = value
        elif name.startswith("short_"):
            bot.setdefault("short", {})[name[6:]] = value
        else:
            config.setdefault("live", {})[name] = value

    return config
