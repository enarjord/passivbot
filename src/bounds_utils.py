"""
Utilities for enforcing parameter bounds and step constraints.

This module provides reusable functions for quantizing parameter values
to discrete steps and enforcing bounds on configuration dictionaries.
"""

import logging
from typing import Dict, Optional, Tuple


# Bound: (low, high) for continuous, or (low, high, step) for stepped params
# step=None or step=0 means continuous (no quantization)
Bound = Tuple[float, float, Optional[float]]  # (low, high, step)


def quantize_to_step(value: float, low: float, high: float, step: Optional[float] = None) -> float:
    """
    Quantize a value to the nearest step within bounds [low, high].

    Args:
        value: value to quantize
        low: lower bound
        high: upper bound
        step: step size (if None or <= 0, no quantization)

    Returns:
        float: quantized value clamped to the valid grid within [low, high]
    """
    if step is None or step <= 0:
        return max(low, min(high, value))

    # Clamp to bounds first
    clamped = max(low, min(high, value))

    # Find nearest step (using int + 0.5 for proper rounding, not banker's rounding)
    n_steps_from_low = int((clamped - low) / step + 0.5)

    # Calculate max valid index: largest n where low + n*step <= high
    # Use floor to ensure we don't exceed high bound
    # Add small epsilon to handle floating point precision issues
    max_index = int((high - low + 1e-9) / step)

    # Clamp index to valid range
    clamped_index = max(0, min(max_index, n_steps_from_low))

    quantized = low + clamped_index * step

    # Final safety clamp (should be redundant but kept for safety)
    return max(low, min(high, quantized))


def value_to_index(value: float, low: float, high: float, step: Optional[float]) -> float:
    """
    Convert a parameter value to index space for stepped parameters.

    For continuous parameters (step=None or step<=0), returns the value unchanged.
    For stepped parameters, returns the index (0-based) of the step.

    Args:
        value: parameter value
        low: lower bound
        high: upper bound
        step: step size

    Returns:
        float: index in step space, or original value if continuous
    """
    if step is None or step <= 0:
        return value
    return (value - low) / step


def index_to_value(index: float, low: float, high: float, step: Optional[float]) -> float:
    """
    Convert an index back to a parameter value for stepped parameters.

    For continuous parameters (step=None or step<=0), returns the index unchanged.
    For stepped parameters, converts the index to a value and quantizes to grid.

    Args:
        index: index in step space (or original value if continuous)
        low: lower bound
        high: upper bound
        step: step size

    Returns:
        float: parameter value, quantized to step if applicable
    """
    if step is None or step <= 0:
        return max(low, min(high, index))
    # Round to nearest integer index and convert back to value
    rounded_index = int(index + 0.5)
    # Calculate max valid index: largest n where low + n*step <= high
    # Add small epsilon to handle floating point precision issues
    max_index = int((high - low + 1e-9) / step)
    clamped_index = max(0, min(max_index, rounded_index))
    return low + clamped_index * step


def get_index_bounds(low: float, high: float, step: Optional[float]) -> Tuple[float, float]:
    """
    Get the bounds in index space for a parameter.

    For continuous parameters, returns (low, high).
    For stepped parameters, returns (0, max_index).

    Args:
        low: lower bound
        high: upper bound
        step: step size

    Returns:
        Tuple[float, float]: (low_index, high_index)
    """
    if step is None or step <= 0:
        return (low, high)
    # Calculate max valid index: largest n where low + n*step <= high
    # Add small epsilon to handle floating point precision issues
    max_index = int((high - low + 1e-9) / step)
    return (0.0, float(max_index))


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
