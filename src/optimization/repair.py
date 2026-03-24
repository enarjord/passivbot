"""
pymoo Repair operator for significant-digit rounding.
"""

import numpy as np
from pymoo.core.repair import Repair


def _round_to_sig_digits_array(X, sig_digits):
    """Vectorized significant-digit rounding for a 2D numpy array."""
    mask = (X != 0.0) & np.isfinite(X)
    out = X.copy()
    digits = sig_digits - 1 - np.floor(np.log10(np.abs(X[mask]))).astype(int)
    factors = 10.0 ** digits
    out[mask] = np.round(X[mask] * factors) / factors
    return out


class SignificantDigitsRepair(Repair):
    """
    Rounds all decision variables to N significant digits after each
    genetic operation (crossover, mutation), then clamps to bounds.
    """

    def __init__(self, sig_digits: int):
        super().__init__()
        self.sig_digits = sig_digits

    def _do(self, problem, X, **kwargs):
        X = _round_to_sig_digits_array(X, self.sig_digits)
        X = np.clip(X, problem.xl, problem.xu)
        return X
