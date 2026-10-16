"""Original-precision TM decimal operands for the private shader ABI.

This module deliberately needs no PyTorch or device runtime.
"""

from __future__ import annotations

import numpy as np

from optimization.gpu.model import encode_tm_retracement_base_pct


# Private shader ABI metadata follows the public candidate/override columns.
_TM_DECIMAL_KEYS = ("entry_double_down_factor", "entry_threshold_base_pct",
                    "entry_retracement_base_pct", "close_threshold_base_pct",
                    "close_retracement_base_pct")
_TM_DECIMAL_VALUE_WORDS = 4 * len(_TM_DECIMAL_KEYS)
_TM_DECIMAL_WORDS = _TM_DECIMAL_VALUE_WORDS + 4
_TM_DECIMAL_MODIFIER_GROUPS = tuple(
    tuple(prefix + "_" + suffix for suffix in suffixes)
    for prefix, suffixes in (
        ("entry_threshold", ("we_weight", "volatility_1h_weight", "volatility_1m_weight")),
        ("entry_retracement", ("we_weight", "volatility_1h_weight", "volatility_1m_weight")),
        ("close_threshold", ("we_weight", "volatility_1h_weight", "volatility_1m_weight")),
        ("close_retracement", ("volatility_1h_weight", "volatility_1m_weight")),
    )
)


def _tm_modifier_masks(originals, keys, *, overrides=False):
    masks = []
    for group in _TM_DECIMAL_MODIFIER_GROUPS:
        values = originals[..., [keys.index(key) for key in group]]
        bits = 1 << np.arange(len(group), dtype=np.uint32)
        present = np.isfinite(values)
        nonzero = np.sum((present & (values != 0.0)) * bits, axis=-1)
        # Low three bits identify explicit pins; next three preserve their
        # original nonzero values, including values lost in float32 conversion.
        masks.append(np.sum(present * bits, axis=-1) + (nonzero << 3)
                     if overrides else nonzero)
    return np.stack(masks, axis=-1).astype(np.float32)


def _tm_decimal_metadata(values):
    import passivbot_rust

    flat = np.asarray(values, dtype=np.float64)
    metadata = passivbot_rust.gpu_decimal_metadata_py(flat.reshape(-1).tolist())
    return np.asarray(metadata, dtype=np.float32).reshape(*flat.shape, 4)


def _pack_tm_parameter_matrix(
    params: np.ndarray, keys: tuple[str, ...], *, sides: int
) -> np.ndarray:
    """Pack TM rows while preserving each candidate's retracement mode sign."""

    matrix = np.ascontiguousarray(params, dtype=np.float32)
    side_width = len(keys)
    for side_index in range(sides):
        offset = side_index * side_width
        for key in (
            "entry_retracement_base_pct",
            "close_retracement_base_pct",
        ):
            column = offset + keys.index(key)
            positive_underflow = (params[:, column] > 0.0) & (
                matrix[:, column] == np.float32(0.0)
            )
            if np.any(positive_underflow):
                matrix[positive_underflow, column] = encode_tm_retracement_base_pct(
                    np.finfo(np.float64).tiny
                )
    public = matrix.reshape(len(matrix), sides, side_width)
    originals = np.asarray(params, dtype=np.float64).reshape(len(matrix), sides, side_width)
    columns = [keys.index(key) for key in _TM_DECIMAL_KEYS]
    metadata = _tm_decimal_metadata(originals[:, :, columns]).reshape(len(matrix), sides, _TM_DECIMAL_VALUE_WORDS)
    flags = _tm_modifier_masks(originals, keys)
    return np.ascontiguousarray(np.concatenate((public, metadata, flags), axis=2).reshape(len(matrix), sides * (side_width + _TM_DECIMAL_WORDS)))
