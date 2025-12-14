"""
Utilities for enforcing parameter bounds and step constraints.

This module provides reusable functions for quantizing parameter values
to discrete steps and enforcing bounds on configuration dictionaries.
"""

import logging
from typing import Dict


def quantize_to_step(value: float, low: float, high: float, step: float = None) -> float:
    """
    Quantize a value to the nearest step within bounds [low, high].

    Args:
        value: value to quantize
        low: lower bound
        high: upper bound
        step: step size (if None or <= 0, no quantization)

    Returns:
        float: quantized value clamped to [low, high]
    """
    if step is None or step <= 0:
        return value

    # Clamp to bounds first
    clamped = max(low, min(high, value))

    # Find nearest step (using int + 0.5 for proper rounding, not banker's rounding)
    n_steps_from_low = int((clamped - low) / step + 0.5)
    quantized = low + n_steps_from_low * step

    # Ensure we're still within bounds after quantization
    return max(low, min(high, quantized))


def enforce_bounds_on_config(config: dict, sig_digits: int = None) -> dict:
    """
    Re-apply step bounds to bot parameters after rounding to ensure
    all values respect the step constraints defined in optimize.bounds.

    This is necessary because round_floats() can cause values to drift
    off the step grid when rounding to N significant digits.

    Args:
        config: Configuration dictionary with bot parameters and bounds
        sig_digits: Significant digits used for rounding (not used in this function,
                   but kept for API compatibility)

    Returns:
        dict: Modified config with quantized parameter values
    """
    try:
        bounds_dict = config.get("optimize", {}).get("bounds", {})
        if not bounds_dict:
            return config

        bot = config.get("bot", {})
        if not bot:
            return config

        keys_ignored = {"enforce_exposure_limit"}

        # Process each pside
        for pside in sorted(bot.keys()):
            pside_params = bot[pside]
            if not isinstance(pside_params, dict):
                continue

            # Process each parameter
            for param_key in sorted(pside_params.keys()):
                if param_key in keys_ignored:
                    continue

                # Look up bounds for this parameter
                bound_key = f"{pside}_{param_key}"
                if bound_key not in bounds_dict:
                    continue

                bound_value = bounds_dict[bound_key]
                if not isinstance(bound_value, (list, tuple)) or len(bound_value) < 3:
                    continue

                low, high, step = bound_value[0], bound_value[1], bound_value[2]

                # Quantize if step is defined
                if step is not None and step > 0:
                    current_value = pside_params[param_key]
                    quantized_value = quantize_to_step(current_value, low, high, step)
                    config["bot"][pside][param_key] = quantized_value

        return config
    except Exception as e:
        logging.debug(f"Could not enforce step bounds on config: {e}")
        return config


def get_bound_keys_ignored():
    """Return set of parameter keys that should be ignored when processing bounds."""
    return {"enforce_exposure_limit"}
