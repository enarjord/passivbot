"""
Utilities for enforcing parameter bounds and step constraints.

This module provides reusable functions for quantizing parameter values
to discrete steps and enforcing bounds on configuration dictionaries.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
try:
    from deap import tools as deap_tools
except ImportError:  # pragma: no cover - allow import in minimal test envs
    deap_tools = None

from config_utils import get_template_config
from opt_utils import round_floats

# Epsilon for floating-point comparisons in step calculations
STEP_EPSILON = 1e-9

# Epsilon for temporarily adjusting equal bounds in DEAP operators
DEAP_EQUAL_BOUNDS_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class Bound:
    """
    Represents parameter bounds for optimization.

    For continuous parameters, step is None.
    For stepped (discrete) parameters, step defines the grid spacing.

    Args:
        low: Lower bound
        high: Upper bound
        step: Step size for discrete parameters (None for continuous)
    """
    low: float
    high: float
    step: Optional[float] = None

    @property
    def is_stepped(self) -> bool:
        """Check if this represents a stepped (discrete) parameter."""
        return self.step is not None and self.step > 0

    @property
    def max_index(self) -> int:
        """
        Get the maximum valid index for a stepped parameter.

        Returns the largest n where low + n*step <= high.
        Raises ValueError if parameter is not stepped.
        """
        if not self.is_stepped:
            raise ValueError("max_index only valid for stepped parameters")
        return int((self.high - self.low + STEP_EPSILON) / self.step)

    def quantize(self, value: float) -> float:
        """
        Quantize a value to the nearest step within bounds.

        For continuous parameters, just clamps to [low, high].
        For stepped parameters, snaps to the nearest grid point.

        Args:
            value: Value to quantize

        Returns:
            Quantized value clamped to valid bounds
        """
        if not self.is_stepped:
            return max(self.low, min(self.high, value))

        # Clamp to bounds first
        clamped = max(self.low, min(self.high, value))

        # Find nearest step (using int + 0.5 for proper rounding)
        n_steps_from_low = int((clamped - self.low) / self.step + 0.5)

        # Clamp index to valid range
        clamped_index = max(0, min(self.max_index, n_steps_from_low))

        quantized = self.low + clamped_index * self.step

        # Final safety clamp
        return max(self.low, min(self.high, quantized))

    def random_on_grid(self) -> float:
        """
        Generate a random value respecting step constraints.

        For continuous parameters, returns uniform random in [low, high].
        For stepped parameters, returns random value on the grid.

        Returns:
            Random value
        """
        if not self.is_stepped:
            return np.random.uniform(self.low, self.high)
        random_idx = np.random.randint(0, self.max_index + 1)
        return self.low + random_idx * self.step

    def value_to_index(self, value: float) -> float:
        """
        Convert a parameter value to index space.

        For continuous parameters, returns value unchanged.
        For stepped parameters, returns the index (0-based).

        Args:
            value: Parameter value

        Returns:
            Index in step space, or original value if continuous
        """
        if not self.is_stepped:
            return value
        return (value - self.low) / self.step

    def index_to_value(self, index: float) -> float:
        """
        Convert an index back to a parameter value.

        For continuous parameters, returns index clamped to bounds.
        For stepped parameters, converts index to grid value.

        Args:
            index: Index in step space (or value if continuous)

        Returns:
            Parameter value
        """
        if not self.is_stepped:
            return max(self.low, min(self.high, index))
        # Round to nearest integer index and convert
        rounded_index = int(index + 0.5)
        clamped_index = max(0, min(self.max_index, rounded_index))
        return self.low + clamped_index * self.step

    def get_index_bounds(self) -> Tuple[float, float]:
        """
        Get the bounds in index space.

        For continuous parameters, returns (low, high).
        For stepped parameters, returns (0, max_index).

        Returns:
            (low_index, high_index)
        """
        if not self.is_stepped:
            return (self.low, self.high)
        return (0.0, float(self.max_index))

    @classmethod
    def from_config(cls, key: str, val) -> "Bound":
        """
        Extract and validate a Bound from config value.

        Supported formats:
        - Single value: fixed parameter (low=high)
        - [low, high]: continuous optimization
        - [low, high, step]: discrete optimization with step
        - [low, high, 0] or [low, high, null]: continuous

        Args:
            key: Parameter key (for error messages)
            val: Config value (number, list, or tuple)

        Returns:
            Validated Bound instance

        Raises:
            Exception: If bound specification is malformed
        """
        if isinstance(val, (float, int)):
            return cls(float(val), float(val), None)

        if isinstance(val, (tuple, list)):
            if len(val) == 0:
                raise Exception(f"malformed bound {key}: empty array")
            if len(val) == 1:
                return cls(float(val[0]), float(val[0]), None)
            if len(val) == 2:
                low, high = sorted([float(val[0]), float(val[1])])
                return cls(low, high, None)
            if len(val) >= 3:
                low, high = sorted([float(val[0]), float(val[1])])
                if len(val) > 3:
                    logging.warning(
                        "Bound %s has %d elements; expected 1, 2, or 3. Ignoring step and using sig_digits.",
                        key, len(val),
                    )
                    return cls(low, high, None)

                step_raw = val[2]
                if step_raw is None:
                    logging.warning(
                        "Bound %s step is null; treating as continuous and using sig_digits.", key
                    )
                    return cls(low, high, None)

                try:
                    step = float(step_raw)
                except Exception:
                    logging.warning(
                        "Bound %s step is not a number (%r); treating as continuous and using sig_digits.",
                        key, step_raw,
                    )
                    return cls(low, high, None)

                if step <= 0:
                    logging.warning(
                        "Bound %s step must be > 0 (got %s); treating as continuous and using sig_digits.",
                        key, step,
                    )
                    return cls(low, high, None)

                if high != low and step > (high - low):
                    logging.warning(
                        "Bound %s step=%s is larger than range [%s, %s]; treating as continuous and using sig_digits.",
                        key, step, low, high,
                    )
                    return cls(low, high, None)

                return cls(low, high, step)

        raise Exception(f"malformed bound {key}: {val}")


# === Index-space conversion helpers for genetic operators ==================


def extract_steps(bounds: Optional[Sequence[Bound]], n: int) -> List[Optional[float]]:
    """
    Extract step values from a bounds list.

    Args:
        bounds: List of Bound instances, or None
        n: Expected number of parameters

    Returns:
        List of step values (None for continuous parameters)
    """
    if bounds is None or len(bounds) != n:
        return [None] * n
    return [b.step for b in bounds]


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

    for i, val in enumerate(values):
        if bounds[i].is_stepped:
            bound = bounds[i]
            index_values.append(bound.value_to_index(val))
            idx_low, idx_up = bound.get_index_bounds()
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
    temp_low = np.where(equal_bounds_mask, low_array - DEAP_EQUAL_BOUNDS_EPSILON, low_array)
    temp_up = np.where(equal_bounds_mask, up_array + DEAP_EQUAL_BOUNDS_EPSILON, up_array)
    return list(temp_low), list(temp_up), equal_bounds_mask


def from_index_space(
    index_values: List[float],
    bounds,
    steps: List[Optional[float]],
    low: List[float],
    equal_mask: np.ndarray,
) -> List[float]:
    """
    Convert index-space values back to parameter space.

    For stepped parameters, indices are converted back to values on the grid.
    For continuous parameters, values are copied directly.
    Parameters with equal bounds are reset to their low value.

    Args:
        index_values: Values in index space (modified by DEAP operator)
        bounds: List of bound tuples (used for stepped parameters)
        steps: List of step values (from extract_steps)
        low: Original low bounds (used for equal-bounds reset)
        equal_mask: Boolean mask indicating which parameters have equal bounds

    Returns:
        List[float]: Values converted back to parameter space
    """
    result = []
    for i in range(len(index_values)):
        if equal_mask[i]:
            result.append(low[i])
        elif bounds[i].is_stepped:
            bound = bounds[i]
            result.append(bound.index_to_value(index_values[i]))
        else:
            result.append(index_values[i])
    return result


def enforce_bounds_on_config(config: dict) -> dict:
    """
    Re-apply step bounds to bot parameters after rounding to ensure
    all values respect the step constraints defined in optimize.bounds.

    This is necessary because round_floats() can cause values to drift
    off the step grid when rounding to N significant digits.

    Args:
        config: Configuration dictionary with bot parameters and bounds

    Returns:
        dict: Modified config with quantized parameter values

    Raises:
        ValueError: If bounds cannot be enforced due to malformed config
    """
    try:
        bounds_dict = config.get("optimize", {}).get("bounds", {})
        if not bounds_dict:
            return config

        bot = config.get("bot", {})
        if not bot:
            return config

        # Process each pside
        for pside in sorted(bot.keys()):
            pside_params = bot[pside]
            if not isinstance(pside_params, dict):
                continue

            # Process each parameter
            for param_key in sorted(pside_params.keys()):
                # Look up bounds for this parameter
                bound_key = f"{pside}_{param_key}"
                if bound_key not in bounds_dict:
                    continue

                bound_value = bounds_dict[bound_key]
                if not isinstance(bound_value, (list, tuple)):
                    continue

                # Parse bound and quantize if stepped
                try:
                    bound = Bound.from_config(bound_key, bound_value)
                    if bound.is_stepped:
                        current_value = pside_params[param_key]
                        quantized_value = bound.quantize(current_value)
                        config["bot"][pside][param_key] = quantized_value
                except KeyError:
                    # Parameter not found in config - skip silently as this is expected
                    continue
                except Exception as e:
                    raise ValueError(
                        f"Failed to enforce bounds for {bound_key}: {e}"
                    ) from e

        return config
    except Exception as e:
        raise ValueError(f"Could not enforce step bounds on config: {e}") from e


def enforce_bounds(
    values: Sequence[float], bounds: Sequence[Bound], sig_digits: int = None
) -> List[float]:
    """
    Clamp each value to its corresponding [low, high] interval and quantize to step if defined.
    Also round to significant digits (optional).

    Args:
        values : iterable of floats (length == len(bounds))
        bounds : iterable of Bound instances
        sig_digits: int

    Returns
        List[float]  – clamped and quantized copy (original is *not* modified)
    """
    if len(values) != len(bounds):
        raise ValueError(
            f"values/bounds length mismatch: got {len(values)} values but {len(bounds)} bounds"
        )
    result = []
    for v, bound in zip(values, bounds):
        if bound.is_stepped:
            result.append(bound.quantize(v))
        else:
            rounded = v if sig_digits is None else round_floats(v, sig_digits)
            result.append(bound.high if rounded > bound.high else bound.low if rounded < bound.low else rounded)
    return result


def extract_bounds_tuple_list_from_config(config) -> List[Bound]:
    """
    Extracts list of Bound instances for bot parameters.
    Also sets all bounds to (low, low, step) if pside is not enabled.

    Supported formats:
        - [low, high]: continuous optimization (step=None)
        - [low, high, step]: discrete optimization with given step
        - [low, high, 0] or [low, high, null]: treated as continuous
        - single value: fixed parameter (low=high, step=None)
    """
    template_config = get_template_config()
    bounds = []
    for pside in sorted(template_config["bot"]):
        is_enabled = all(
            [
                Bound.from_config(k, config["optimize"]["bounds"][k]).high > 0.0
                for k in [f"{pside}_n_positions", f"{pside}_total_wallet_exposure_limit"]
            ]
        )
        for key in sorted(template_config["bot"][pside]):
            bound_key = f"{pside}_{key}"
            assert (
                bound_key in config["optimize"]["bounds"]
            ), f"bound {bound_key} missing from optimize.bounds"
            bound_vals = Bound.from_config(bound_key, config["optimize"]["bounds"][bound_key])
            if is_enabled:
                bounds.append(bound_vals)
            else:
                # Disabled: fix to low value, preserve step for consistency
                bounds.append(Bound(bound_vals.low, bound_vals.low, bound_vals.step))
    return bounds


# === DEAP genetic operator wrappers =========================================


def mutPolynomialBoundedWrapper(individual, eta, low, up, indpb, bounds=None):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    For stepped parameters (where bounds[i][2] is defined and > 0), mutation
    is performed in index space to ensure offspring values stay on the grid.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        indpb: Independent probability for each attribute to be mutated.
        bounds: Optional list of (low, high, step) tuples. If provided, stepped
                parameters are mutated in index space.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    """
    if deap_tools is None:  # pragma: no cover
        raise ModuleNotFoundError("deap is required for optimizer mutation operators")
    n = len(individual)
    steps = extract_steps(bounds, n)

    index_ind, index_low, index_up = to_index_space(individual, bounds, steps, low, up)
    temp_low, temp_up, equal_mask = prepare_bounds_for_deap(index_low, index_up)

    deap_tools.mutPolynomialBounded(index_ind, eta, temp_low, temp_up, indpb)

    individual[:] = from_index_space(index_ind, bounds, steps, low, equal_mask)
    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, low, up, bounds=None):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    For stepped parameters (where bounds[i][2] is defined and > 0), crossover
    is performed in index space to ensure offspring values stay on the grid.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        low: A value or sequence of values that is the lower bound of the search space.
        up: A value or sequence of values that is the upper bound of the search space.
        bounds: Optional list of (low, high, step) tuples. If provided, stepped
                parameters are crossed over in index space.

    Returns:
        A tuple of two individuals after crossover operation.
    """
    if deap_tools is None:  # pragma: no cover
        raise ModuleNotFoundError("deap is required for optimizer crossover operators")
    n = len(ind1)
    steps = extract_steps(bounds, n)

    index_ind1, index_low, index_up = to_index_space(ind1, bounds, steps, low, up)
    index_ind2, _, _ = to_index_space(ind2, bounds, steps, low, up)
    temp_low, temp_up, equal_mask = prepare_bounds_for_deap(index_low, index_up)

    deap_tools.cxSimulatedBinaryBounded(index_ind1, index_ind2, eta, temp_low, temp_up)

    ind1[:] = from_index_space(index_ind1, bounds, steps, low, equal_mask)
    ind2[:] = from_index_space(index_ind2, bounds, steps, low, equal_mask)
    return ind1, ind2
