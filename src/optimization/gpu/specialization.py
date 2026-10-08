"""Dispatch feature proofs over packed candidates and immutable coin overrides.

These decisions belong to execution, independently of search and device scheduling.
"""

import numpy as np


def unstuck_ema_required(matrix, keys, side_overrides):
    """Keep EMA state if any candidate/coin can use unstuck EMA gating.

    Override columns are enabled/gating, in shader order. Nonfinite overrides
    inherit the candidate value exactly as coin_override_or does. Unknown base
    flags are conservatively retained; producer validation remains separate.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    keys = tuple(keys)
    side_overrides = tuple(side_overrides)
    if matrix.ndim != 2 or not side_overrides or matrix.shape[1] != len(keys) * len(side_overrides):
        raise ValueError("unstuck EMA proof requires one packed parameter block per side")
    if not len(matrix):
        return True
    enabled = keys.index("unstuck_enabled")
    gating = keys.index("unstuck_ema_gating_enabled")
    for side, overrides in enumerate(side_overrides):
        overrides = np.asarray(overrides, dtype=np.float32)
        if overrides.ndim != 2 or overrides.shape[1] != 2 or not len(overrides):
            raise ValueError("unstuck EMA proof requires enabled/gating override columns")
        flags = matrix[:, side * len(keys) + np.asarray([enabled, gating])]
        if not np.isfinite(flags).all():
            return True
        effective = np.where(np.isfinite(overrides)[None, :, :],
                             overrides[None, :, :], flags[:, None, :])
        if np.any((effective[:, :, 0] > 0.5) & (effective[:, :, 1] > 0.5)):
            return True
    return False
