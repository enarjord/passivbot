"""
Utilities for enforcing parameter bounds and step constraints.

This module provides reusable functions for quantizing parameter values
to discrete steps and enforcing bounds on configuration dictionaries.
"""

import logging
from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Epsilon for floating-point comparisons in step calculations
STEP_EPSILON = 1e-9


def round_to_sig_digits(value: float, sig_digits: int) -> float:
    """
    Pure-python significant-digit rounding.

    This intentionally avoids depending on passivbot_rust so unit tests can run
    in environments where the native extension is stubbed or unavailable.
    """
    if sig_digits is None:
        return value
    if not isinstance(value, (int, float)):
        raise TypeError(f"value must be numeric, got {type(value).__name__}")
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    digits = sig_digits - 1 - int(math.floor(math.log10(abs(value))))
    return round(value, digits)


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

        # Clean up floating-point artifacts by rounding to step precision
        # e.g., step=0.0002 -> 4 decimal places, step=0.01 -> 2 decimal places
        if self.step < 1:
            decimal_places = -int(math.floor(math.log10(self.step)))
            quantized = round(quantized, decimal_places)

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
                        key,
                        len(val),
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
                        key,
                        step_raw,
                    )
                    return cls(low, high, None)

                if step <= 0:
                    logging.warning(
                        "Bound %s step must be > 0 (got %s); treating as continuous and using sig_digits.",
                        key,
                        step,
                    )
                    return cls(low, high, None)

                if high != low and step > (high - low):
                    logging.warning(
                        "Bound %s step=%s is larger than range [%s, %s]; treating as continuous and using sig_digits.",
                        key,
                        step,
                        low,
                        high,
                    )
                    return cls(low, high, None)

                return cls(low, high, step)

        raise Exception(f"malformed bound {key}: {val}")


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
            rounded = v if sig_digits is None else round_to_sig_digits(v, sig_digits)
            result.append(
                bound.high if rounded > bound.high else bound.low if rounded < bound.low else rounded
            )
    return result
