"""Sparse hourly GPU transport preserves every canonical float32 observation."""

import ast
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu import model, residency, runtime


ROOT = Path(__file__).resolve().parents[2]


def _case(n, minute, interval=60_000):
    timestamps = (1_700_000_000_000 // 3_600_000) * 3_600_000 + minute * 60_000
    timestamps = timestamps + np.arange(n, dtype=np.int64) * interval
    closes = 100.0 + np.arange(n)[:, None] * .002 + np.arange(3)[None, :] * 7.0
    values = np.stack((closes * 1.002, closes * .998, closes, closes * .25), axis=2)
    ranges = [(0, n - 1), (min(5, n - 2), n - 1), (0, max(0, n - 9))]
    runs = [model.ProxyRun(1000., 0, first, int(timestamps[0]), 0,
                          int(timestamps[0]), interval, .05, first, last)
            for first, last in ranges]
    for coin, (first, last) in enumerate(ranges):
        values[:first, coin] = np.nan
        values[last + 1:, coin] = np.nan
    markets = [model.ProxyMarket(.001, .01, .001, 1., 1., .0002)] * 3
    return values, timestamps, runs, markets


def _decode(compact, n, minute):
    result = np.full((n, compact.shape[1]), -1.0, dtype=np.float32)
    boundaries = np.arange(60 - minute, n, 60, dtype=np.int64)
    result[boundaries] = compact[:len(boundaries)]
    return result


@pytest.mark.parametrize("n,minute", [(3, 0), (3, 59), (61, 0), (61, 1), (185, 23), (1537, 59)])
def test_compact_spill_decodes_to_every_dense_canonical_byte(tmp_path, n, minute):
    values, timestamps, runs, markets = _case(n, minute)
    directories = [tmp_path / mode for mode in ("dense", "compact")]
    for directory in directories:
        directory.mkdir()
    packed = [model._spill_multicoin_arrays(
        values, timestamps, runs, markets, np.int32, .001, .0005, True, directory,
        compact_hourly_ranges=compact,
    ) for compact, directory in zip((False, True), directories)]
    dense, compact = [item[0] for item in packed]
    assert dense.keys() == compact.keys()
    np.testing.assert_array_equal(packed[0][1], packed[1][1])
    np.testing.assert_array_equal(packed[0][2], packed[1][2])
    for name in dense:
        expected, actual = np.load(dense[name]), np.load(compact[name])
        if name == "hour_log_ranges":
            assert actual.shape == (residency.compact_hourly_rows(n, start_minute_of_hour=minute), 3)
            actual = _decode(actual, n, minute)
        np.testing.assert_array_equal(actual, expected, err_msg=name)
    assert np.load(compact["hour_log_ranges"]).dtype == np.float32


def test_compact_hour_output_obeys_preparation_chunk_bound(tmp_path, monkeypatch):
    values, timestamps, runs, markets = _case(185, 23)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 1024)
    outputs = model._spill_multicoin_arrays(
        values, timestamps, runs, markets, np.int32, 0.0, 0.0, True, tmp_path,
        compact_hourly_ranges=True,
    )
    expected = np.column_stack([
        np.where(valid, ranges, np.float32(-1.0))
        for ranges, valid in [model._build_hourly_log_range(
            values[:, coin, 0], values[:, coin, 1], timestamps, run,
        ) for coin, run in enumerate(runs)]
    ])
    actual = _decode(np.load(outputs[0]["hour_log_ranges"]), len(values), 23)
    np.testing.assert_array_equal(actual, expected)
    assert not list(tmp_path.glob("hourly-*"))


@pytest.mark.parametrize("interval,stream,requested,expected", [
    (60_000, True, True, True), (120_000, True, True, False),
    (60_000, False, True, False), (60_000, True, False, False),
])
def test_packer_compacts_only_streamed_one_minute_metal(
    tmp_path, monkeypatch, interval, stream, requested, expected,
):
    fake_torch = SimpleNamespace(float32=np.float32,
        mps=SimpleNamespace(recommended_max_memory=lambda: 2**40,
                            driver_allocated_memory=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime, "gpu_device", lambda *args: "mps")
    values, timestamps, runs, markets = _case(185, 23, interval)
    packed = model.build_mps_multicoin_data(
        values, timestamps, runs, markets, spill_dir=tmp_path,
        stream_tick_inputs=stream, compact_hourly_ranges=requested,
    )
    assert packed["compact_hourly_ranges"] is expected
    n, coins = values.shape[:2]
    hourly_rows = residency.compact_hourly_rows(n, start_minute_of_hour=23) if expected else n
    assert np.load(packed["hour_log_ranges"]).shape == (hourly_rows, coins)
    assert packed["hourly_range_rows"] == hourly_rows
    assert packed["invariant_bytes"] == n * coins * 52 + hourly_rows * coins * 4
    assert packed["resident_bytes"] == residency.multicoin_resident_bytes(
        n, coins, device="mps", stream_tick_inputs=stream,
        compact_hourly_ranges=expected, start_minute_of_hour=23,
    )


def test_compact_estimator_bounds_every_start_phase_and_keeps_selection_pure():
    n, coins = 3_000_000, 40
    header = residency.multicoin_resident_bytes(
        n, coins, device="mps", stream_tick_inputs=True, compact_hourly_ranges=True,
    )
    dense = residency.multicoin_resident_bytes(n, coins, device="mps", stream_tick_inputs=True)
    assert header < dense
    for minute in range(60):
        actual = residency.multicoin_resident_bytes(
            n, coins, device="mps", stream_tick_inputs=True,
            compact_hourly_ranges=True, start_minute_of_hour=minute,
        )
        assert actual <= header
        assert actual == n * coins * 24 + min(n, 8194) * coins * 28 + (
            residency.compact_hourly_rows(n, start_minute_of_hour=minute) * coins * 4
        )
    # No telemetry reads are needed after the run-start budget has been frozen.
    budget = residency.MpsInputMemoryBudget(0, 0)
    assert residency.prefer_streamed_ticks(1, coins, budget, interval_ms=60_000) is False
    assert residency.prefer_streamed_ticks(n, coins, budget, interval_ms=60_000) is True
    assert residency.prefer_streamed_ticks(n, coins, budget, interval_ms=120_000) is True


def test_compact_hour_shader_preserves_pointer_abi_and_arbitrary_bootstrap_indexing():
    parsed = ast.parse((ROOT / "src/optimization/gpu/mps_kernel.py").read_text())
    function = next(node for node in parsed.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_with_compact_metal_hourly_ranges")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT), "exec"), namespace)
    source = (ROOT / "passivbot-rust/src/gpu/mps_multicoin_common.metal").read_text()
    source += (ROOT / "passivbot-rust/src/gpu/mps_trailing_martingale_multicoin.metal").read_text()
    result = namespace["_with_compact_metal_hourly_ranges"](source)
    assert result.count("constant float* compact_hour_log_ranges") == 3
    assert "constant float* hour_log_ranges" not in result
    assert "hour_log_ranges[j * coin_count + coin]" in result  # Arbitrary bootstrap history.
    assert "hour_log_ranges[k * coin_count + c]" in result
    assert "k <= 0 || (start_minute_of_hour + k) % 60 != 0" in result
    assert "const int hour = (start_minute_of_hour + k) / 60 - 1;" in result
    assert result.count("{compact_hour_log_ranges, sizes[2], sizes[7]}") == 3
    for signature in re.findall(r"kernel void \w+\((.*?)\) \{", result, flags=re.DOTALL):
        assert "PassivbotHourlyRangeView" not in signature
    with pytest.raises(RuntimeError, match="declaration contract"):
        namespace["_with_compact_metal_hourly_ranges"](source.replace(
            "constant float* hour_log_ranges", "constant float* renamed", 1,
        ))
