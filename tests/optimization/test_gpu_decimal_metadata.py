"""Canonical TM metadata remains usable without the optional device runtime."""

from decimal import Decimal
import subprocess
import sys

import numpy as np

from optimization.gpu.decimal import _tm_decimal_metadata, _tm_modifier_masks
from optimization.gpu.model import TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS as KEYS


def test_decimal_metadata_preserves_authored_near_half_values_without_torch():
    subprocess.run(
        [sys.executable, "-c", """
import sys
sys.modules['torch'] = None
from optimization.gpu.decimal import _tm_decimal_metadata
values = _tm_decimal_metadata([1.15, 1.15 - 1e-12, 1.15 + 1e-12])
assert len({tuple(row) for row in values}) == 3
"""],
        check=True,
    )


def test_decimal_metadata_finite_limb_roundtrip_and_overflow_sentinel():
    values = [1.15, 1.15 - 1e-12, 1.15 + 1e-12, -0.044, 0.0]
    metadata = _tm_decimal_metadata(values)
    assert np.isfinite(metadata).all()
    for original, (low, middle, high, encoded) in zip(values, metadata):
        coefficient = int(low) | int(middle) << 20 | int(high) << 40
        negative = 512 <= encoded < 1536
        exponent = int(encoded) - (1024 if negative else 0)
        actual = Decimal(-coefficient if negative else coefficient).scaleb(exponent)
        assert actual == Decimal(str(original))
    assert _tm_decimal_metadata([1e-100, 1e100])[:, 3].tolist() == [2048, 2048]


def test_modifier_masks_distinguish_pinned_zero_inheritance_and_underflow():
    global_values = np.zeros((1, len(KEYS)), dtype=np.float64)
    global_values[0, KEYS.index("entry_threshold_we_weight")] = 0.25
    global_values[0, KEYS.index("entry_threshold_volatility_1h_weight")] = 1e-100
    inherited = np.full_like(global_values, np.nan)
    pinned = inherited.copy()
    pinned[0, KEYS.index("entry_threshold_we_weight")] = 0.0
    pinned[0, KEYS.index("entry_threshold_volatility_1h_weight")] = 0.0
    tiny_pin = pinned.copy()
    tiny_pin[0, KEYS.index("entry_threshold_volatility_1m_weight")] = 1e-100
    global_mask = int(_tm_modifier_masks(global_values, KEYS)[0, 0])
    assert global_mask == 3
    encoded = _tm_modifier_masks(np.concatenate((inherited, pinned, tiny_pin)), KEYS,
                                 overrides=True)[:, 0].astype(int)
    assert encoded.tolist() == [0, 3, 39]
    effective = [(global_mask & ~(value & 7)) | (value >> 3) for value in encoded]
    assert effective == [3, 0, 4]
