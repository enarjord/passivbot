"""
DEAP adapter functions for optimization.

This module contains wrappers and helpers to adapt the domain-specific Bound logic
to the DEAP evolutionary algorithm library.
"""

from typing import List, Sequence, Tuple
import numpy as np

try:
    from deap import tools as deap_tools
except ImportError:  # pragma: no cover
    deap_tools = None

from optimization.bounds import Bound

# Epsilon for temporarily adjusting equal bounds in DEAP operators
DEAP_EQUAL_BOUNDS_EPSILON = 1e-6


# === Index-space conversion helpers for genetic operators ==================


def to_index_space(
    values: List[float],
    bounds: Sequence[Bound],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convert values to index space for stepped parameters.

    For stepped parameters, values are converted to their index in the discrete grid.
    For continuous parameters, values and bounds are passed through unchanged.

    Args:
        values: Parameter values to convert
        bounds: List of Bound instances

    Returns:
        Tuple of (index_values, index_low, index_up)
    """
    index_values = []
    index_low = []
    index_up = []

    for i, val in enumerate(values):
        bound = bounds[i]
        if bound.is_stepped:
            index_values.append(bound.value_to_index(val))
            idx_low, idx_up = bound.get_index_bounds()
            index_low.append(idx_low)
            index_up.append(idx_up)
        else:
            index_values.append(val)
            index_low.append(bound.low)
            index_up.append(bound.high)

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
    bounds: Sequence[Bound],
    equal_mask: np.ndarray,
) -> List[float]:
    """
    Convert index-space values back to parameter space.

    For stepped parameters, indices are converted back to values on the grid.
    For continuous parameters, values are copied directly.
    Parameters with equal bounds are reset to their low value.

    Args:
        index_values: Values in index space (modified by DEAP operator)
        bounds: List of Bound instances
        equal_mask: Boolean mask indicating which parameters have equal bounds

    Returns:
        List[float]: Values converted back to parameter space
    """
    result = []
    for i in range(len(index_values)):
        bound = bounds[i]
        if equal_mask[i]:
            result.append(bound.low)
        elif bound.is_stepped:
            result.append(bound.index_to_value(index_values[i]))
        else:
            result.append(index_values[i])
    return result


# === DEAP genetic operator wrappers =========================================


def mutPolynomialBoundedWrapper(individual, eta, indpb, bounds: Sequence[Bound]):
    """
    A wrapper around DEAP's mutPolynomialBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    For stepped parameters, mutation is performed in index space to ensure
    offspring values stay on the grid.

    Args:
        individual: Sequence individual to be mutated.
        eta: Crowding degree of the mutation.
        indpb: Independent probability for each attribute to be mutated.
        bounds: List of Bound instances defining parameter constraints.

    Returns:
        A tuple of one individual, mutated with consideration for equal lower and upper bounds.
    """
    if deap_tools is None:  # pragma: no cover
        raise ModuleNotFoundError("deap is required for optimizer mutation operators")

    index_ind, index_low, index_up = to_index_space(individual, bounds)
    temp_low, temp_up, equal_mask = prepare_bounds_for_deap(index_low, index_up)

    deap_tools.mutPolynomialBounded(index_ind, eta, temp_low, temp_up, indpb)

    individual[:] = from_index_space(index_ind, bounds, equal_mask)
    return (individual,)


def cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta, bounds: Sequence[Bound]):
    """
    A wrapper around DEAP's cxSimulatedBinaryBounded function to pre-process
    bounds and handle the case where lower and upper bounds may be equal.

    For stepped parameters, crossover is performed in index space to ensure
    offspring values stay on the grid.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.
        eta: Crowding degree of the crossover.
        bounds: List of Bound instances defining parameter constraints.

    Returns:
        A tuple of two individuals after crossover operation.
    """
    if deap_tools is None:  # pragma: no cover
        raise ModuleNotFoundError("deap is required for optimizer crossover operators")

    index_ind1, index_low, index_up = to_index_space(ind1, bounds)
    index_ind2, _, _ = to_index_space(ind2, bounds)
    temp_low, temp_up, equal_mask = prepare_bounds_for_deap(index_low, index_up)

    deap_tools.cxSimulatedBinaryBounded(index_ind1, index_ind2, eta, temp_low, temp_up)

    ind1[:] = from_index_space(index_ind1, bounds, equal_mask)
    ind2[:] = from_index_space(index_ind2, bounds, equal_mask)
    return ind1, ind2
