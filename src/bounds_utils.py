"""
Utilities for enforcing parameter bounds and step constraints.

This module provides reusable functions for quantizing parameter values
to discrete steps and enforcing bounds on configuration dictionaries.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np


# Bound: (low, high) for continuous, or (low, high, step) for stepped params
# step=None or step=0 means continuous (no quantization)
Bound = Tuple[float, float, Optional[float]]  # (low, high, step)


def get_max_index(low: float, high: float, step: float) -> int:
    """
    Get the maximum valid index for a stepped parameter.

    Calculates the largest n where low + n*step <= high.

    Args:
        low: lower bound
        high: upper bound
        step: step size (must be > 0)

    Returns:
        int: maximum valid index
    """
    # Add small epsilon to handle floating point precision issues
    return int((high - low + 1e-9) / step)


def is_stepped(step: Optional[float]) -> bool:
    """
    Check if step represents a stepped (non-continuous) parameter.

    Args:
        step: step size or None

    Returns:
        bool: True if step is defined and positive
    """
    return step is not None and step > 0


def random_on_grid(low: float, high: float, step: Optional[float]) -> float:
    """
    Generate a random value respecting step constraints.

    For continuous parameters (step is None or <= 0), returns a uniform random
    value in [low, high]. For stepped parameters, returns a random value on
    the grid: low, low + step, low + 2*step, ..., up to high.

    Args:
        low: lower bound
        high: upper bound
        step: step size or None for continuous

    Returns:
        float: random value on the grid (or uniform if continuous)
    """
    if not is_stepped(step):
        return np.random.uniform(low, high)
    max_idx = get_max_index(low, high, step)
    random_idx = np.random.randint(0, max_idx + 1)
    return low + random_idx * step


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
    if not is_stepped(step):
        return max(low, min(high, value))

    # Clamp to bounds first
    clamped = max(low, min(high, value))

    # Find nearest step (using int + 0.5 for proper rounding, not banker's rounding)
    n_steps_from_low = int((clamped - low) / step + 0.5)

    # Clamp index to valid range
    max_index = get_max_index(low, high, step)
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
    if not is_stepped(step):
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
    if not is_stepped(step):
        return max(low, min(high, index))
    # Round to nearest integer index and convert back to value
    rounded_index = int(index + 0.5)
    max_index = get_max_index(low, high, step)
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
    if not is_stepped(step):
        return (low, high)
    max_index = get_max_index(low, high, step)
    return (0.0, float(max_index))


# === Index-space conversion helpers for genetic operators ==================


def extract_steps(bounds, n: int) -> List[Optional[float]]:
    """
    Extract step values from a bounds list.

    Args:
        bounds: List of bound tuples, or None
        n: Expected number of parameters

    Returns:
        List of step values (None for continuous parameters)
    """
    if bounds is None or len(bounds) != n:
        return [None] * n
    return [b[2] for b in bounds]


def to_index_space(
    values: List[float],
    bounds,
    steps: List[Optional[float]],
    low: List[float],
    up: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convert values to index space for stepped parameters.

    For stepped parameters, values are converted to their index in the discrete grid.
    For continuous parameters, values and bounds are passed through unchanged.

    Args:
        values: Parameter values to convert
        bounds: List of bound tuples (used for stepped parameters)
        steps: List of step values (from extract_steps)
        low: Original low bounds (used for continuous parameters)
        up: Original high bounds (used for continuous parameters)

    Returns:
        Tuple of (index_values, index_low, index_up)
    """
    index_values = []
    index_low = []
    index_up = []

    for i, (val, step) in enumerate(zip(values, steps)):
        if is_stepped(step):
            low_val, high_val, _ = bounds[i]
            index_values.append(value_to_index(val, low_val, high_val, step))
            idx_low, idx_up = get_index_bounds(low_val, high_val, step)
            index_low.append(idx_low)
            index_up.append(idx_up)
        else:
            index_values.append(val)
            index_low.append(low[i])
            index_up.append(up[i])

    return index_values, index_low, index_up


def prepare_bounds_for_deap(
    index_low: List[float],
    index_up: List[float],
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Prepare bounds for DEAP genetic operators, handling equal bounds.

    DEAP operators fail when low == high, so we temporarily adjust such bounds
    by a small epsilon. The equal_bounds_mask is returned so callers can reset
    those values after the operation.

    Args:
        index_low: Lower bounds in index space
        index_up: Upper bounds in index space

    Returns:
        Tuple of (temp_low, temp_up, equal_bounds_mask)
    """
    low_array = np.array(index_low)
    up_array = np.array(index_up)
    equal_bounds_mask = low_array == up_array
    temp_low = np.where(equal_bounds_mask, low_array - 1e-6, low_array)
    temp_up = np.where(equal_bounds_mask, up_array + 1e-6, up_array)
    return list(temp_low), list(temp_up), equal_bounds_mask


def from_index_space(
    index_values: List[float],
    target: List[float],
    bounds,
    steps: List[Optional[float]],
    low: List[float],
    equal_mask: np.ndarray,
) -> None:
    """
    Convert index-space values back to parameter space, modifying target in-place.

    For stepped parameters, indices are converted back to values on the grid.
    For continuous parameters, values are copied directly.
    Parameters with equal bounds are reset to their low value.

    Args:
        index_values: Values in index space (modified by DEAP operator)
        target: Target list to write results into (modified in-place)
        bounds: List of bound tuples (used for stepped parameters)
        steps: List of step values (from extract_steps)
        low: Original low bounds (used for equal-bounds reset)
        equal_mask: Boolean mask indicating which parameters have equal bounds
    """
    for i, step in enumerate(steps):
        if equal_mask[i]:
            target[i] = low[i]
        elif is_stepped(step):
            low_val, high_val, _ = bounds[i]
            target[i] = index_to_value(index_values[i], low_val, high_val, step)
        else:
            target[i] = index_values[i]


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

        keys_ignored = get_bound_keys_ignored()

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
                if is_stepped(step):
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
