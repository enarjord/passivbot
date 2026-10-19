"""Original-f64 candidate/coin transport is independent of float32 device packing."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu import model
from optimization.gpu.ema_bootstrap import pack_bootstrap_counts


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension
    assert not getattr(pbr, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return pbr


def test_native_count_buffer_preserves_original_f64_windows(require_real_passivbot_rust_module):
    import passivbot_rust as pbr
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    near = np.nextafter(3.0, np.inf)
    spans = np.array([[near] * 8, [3.0] * 8, [np.finfo(np.float64).max] * 8])
    counts = pbr.gpu_ema_bootstrap_counts_py(spans, 1.0, 73)
    assert counts.dtype == np.int32
    assert np.array_equal(counts, np.array([[7] * 10, [6] * 10, [73] * 10]))
    a = np.nextafter(1.0, np.inf)
    derived = pbr.gpu_ema_bootstrap_counts_py(np.array([[a, 9., a, 9., near * 3, near, near * 3, near * 3]]), 3., 73)
    assert derived.tolist() == [[1, 3, 6, 1, 6, 3, 7, 7, 7, 7]]


@pytest.mark.parametrize("kind", ["EMA_ANCHOR", "TRAILING_MARTINGALE"])
@pytest.mark.parametrize("sides", [1, 2])
def test_original_float64_candidate_and_coin_pins_reach_rust_before_scaling(monkeypatch, kind, sides):
    keys = getattr(model, kind + "_MULTICOIN_PARAM_KEYS")
    cols = getattr(model, kind + "_COIN_OVERRIDE_COLS")
    unstuck = getattr(model, kind + "_COIN_OVERRIDE_UNSTUCK_EMA_START_COLUMN")
    strategy = (model.EMA_ANCHOR_COIN_OVERRIDE_STRATEGY_KEYS if kind == "EMA_ANCHOR" else
                tuple(key for key, _ in model.TRAILING_MARTINGALE_COIN_OVERRIDE_PATHS))
    minute = "offset_volatility_ema_span_1m" if kind == "EMA_ANCHOR" else "volatility_ema_span_1m"
    hour = minute.removesuffix("1m") + "1h"
    near = np.nextafter(3.0, np.inf)
    matrix = np.full((2, len(keys) * sides), near, dtype=np.float64)
    overrides = tuple(np.full((2, cols), np.nan, dtype=np.float64) for _ in range(sides))
    for side, pins in enumerate(overrides):
        matrix[1, side * len(keys) + keys.index("ema_span_1")] = 9.0 + side
        pins[0, strategy.index("ema_span_0")] = np.nextafter(1.0, np.inf)
        pins[1, unstuck] = near * 3.0
        pins[1, strategy.index(minute)] = 12.0 + side
        pins[0, strategy.index(hour)] = 15.0 + side
    seen = []
    def native(spans, interval, available):
        seen.append((spans.copy(), interval, available))
        return np.tile(np.arange(1, 11, dtype=np.int32), (len(spans), 1))
    monkeypatch.setitem(sys.modules, "passivbot_rust", SimpleNamespace(gpu_ema_bootstrap_counts_py=native))
    counts = pack_bootstrap_counts(matrix, keys, overrides, interval_minutes=3, available=73)
    spans, interval, available = seen[0]
    spans = spans.reshape(2, sides, 2, 8)
    assert (interval, available) == (3.0, 73)
    assert spans.dtype == np.float64
    assert spans[0, 0, 0, 0] == np.nextafter(1.0, np.inf)
    assert spans[0, 0, 1, 0] == near  # Inherited authored candidate, still above 3.
    assert spans[0, 0, 1, 2] == near * 3.0  # Independent unstuck pin.
    assert spans[1, 0, 1, 1] == 9.0
    assert spans[0, 0, 1, 4] == 12.0
    assert spans[0, 0, 0, 5] == 15.0  # Hours are not divided in Python.
    assert np.all(spans[..., 6:] == near)
    if sides == 2:
        assert spans[1, 1, 1, 1] == 10.0
        assert spans[0, 1, 1, 4] == 13.0
    assert counts.shape == (2, sides, 2, 10)
    assert counts.dtype == np.int32 and counts.flags.c_contiguous
    assert np.array_equal(counts[1, -1, 1], np.arange(1, 11))
    assert matrix[0, 0] == near  # Transport does not mutate candidate or pin views.


def test_stale_native_api_is_rejected_without_recovering_counts_from_float32(monkeypatch):
    monkeypatch.setitem(sys.modules, "passivbot_rust", SimpleNamespace())
    with pytest.raises(RuntimeError, match="rebuilt native EMA bootstrap-count API"):
        pack_bootstrap_counts(None, (), (), interval_minutes=1, available=73)


@pytest.mark.parametrize("result", [np.ones((1, 10), dtype=np.float32),
    np.zeros((1, 10), dtype=np.int32), np.full((1, 10), 74, dtype=np.int32),
    np.ones((1, 9), dtype=np.int32)])
def test_malformed_native_integer_count_transport_is_rejected(monkeypatch, result):
    monkeypatch.setitem(sys.modules, "passivbot_rust", SimpleNamespace(
        gpu_ema_bootstrap_counts_py=lambda *_: result))
    keys = model.EMA_ANCHOR_MULTICOIN_PARAM_KEYS
    with pytest.raises(ValueError, match="private int32 ABI"):
        pack_bootstrap_counts(np.ones((1, len(keys))), keys,
            (np.full((1, model.EMA_ANCHOR_COIN_OVERRIDE_COLS), np.nan),),
            interval_minutes=1, available=73)


@pytest.mark.parametrize("sides,coins", [(1, 2), (2, 3)])
def test_bootstrap_count_device_bytes_are_included_in_admission(sides, coins):
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _MulticoinReplayRunner
    runner = SimpleNamespace(_hsl_history_bytes_per_candidate=lambda: 0,
        _temporal_replay_enabled=lambda: False, raw_strategy_risk_enabled=False,
        raw_strategy_growth_enabled=False, recovery_distribution_enabled=False,
        _hsl_ema_tail_samples_enabled=lambda: False, _exact_entry_intervals_enabled=lambda: False,
        replay_sides=sides, n_coins=coins)
    assert _MulticoinReplayRunner._history_bytes_per_candidate(runner) == 40 * sides * coins
