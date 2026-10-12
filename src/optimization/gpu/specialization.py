"""Dispatch feature proofs over packed candidates and immutable coin overrides.

These decisions belong to execution, independently of search and device scheduling.
"""

import numpy as np


def unstuck_required(matrix, keys, side_overrides):
    """Keep unstuck work if any effective candidate/coin/side enables it."""
    return _unstuck_consumer_required(matrix, keys, side_overrides, ema_gating=False)


def unstuck_ema_required(matrix, keys, side_overrides):
    """Keep EMA state if any effective unstuck consumer requires EMA gating."""
    return _unstuck_consumer_required(matrix, keys, side_overrides, ema_gating=True)


def _unstuck_consumer_required(matrix, keys, side_overrides, *, ema_gating):
    # Override columns are enabled/gating, in shader order. Nonfinite overrides
    # inherit the candidate value exactly as coin_override_or does. Unknown base
    # flags retain the consuming work; producer validation remains separate.
    matrix = np.asarray(matrix, dtype=np.float32)
    keys = tuple(keys)
    side_overrides = tuple(side_overrides)
    label = "unstuck EMA proof" if ema_gating else "unstuck proof"
    if matrix.ndim != 2 or not side_overrides or matrix.shape[1] != len(keys) * len(side_overrides):
        raise ValueError(f"{label} requires one packed parameter block per side")
    if not len(matrix):
        return True
    fields = ["unstuck_enabled"]
    if ema_gating:
        fields.append("unstuck_ema_gating_enabled")
    columns = np.asarray([keys.index(name) for name in fields])
    for side, overrides in enumerate(side_overrides):
        overrides = np.asarray(overrides, dtype=np.float32)
        if overrides.ndim != 2 or overrides.shape[1] != 2 or not len(overrides):
            raise ValueError(f"{label} requires enabled/gating override columns")
        flags = matrix[:, side * len(keys) + columns]
        if not np.isfinite(flags).all():
            return True
        overrides = overrides[:, :len(fields)]
        effective = np.where(np.isfinite(overrides)[None, :, :],
                             overrides[None, :, :], flags[:, None, :])
        if np.any(np.all(effective > 0.5, axis=2)):
            return True
    return False
