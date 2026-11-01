"""CUDA input transport fits long histories without changing packed values."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu import model, residency, runtime
from optimize import _gpu_cached_input_preflight


def _cuda(monkeypatch, *, free=8 * 2**30, available=8 * 2**30):
    observations = dict(free=free, available=available, device_calls=0, host_calls=0)

    def device():
        observations["device_calls"] += 1
        return observations["free"], 8 * 2**30

    def host():
        observations["host_calls"] += 1
        return SimpleNamespace(available=observations["available"])

    torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=device), float32=np.float32)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=host))
    monkeypatch.setattr(runtime, "gpu_device", lambda *args: "cuda")
    return torch, observations


@pytest.mark.parametrize("available", [0, 2**29, 2**30, 8 * 2**30])
def test_cuda_budget_captures_original_headroom_once(monkeypatch, available):
    torch, observations = _cuda(monkeypatch, free=7 * 2**30, available=available)
    budget = residency.capture_cuda_input_memory_budget(torch)
    assert budget.initial_free_bytes == 7 * 2**30
    assert budget.device_bytes == int(7 * 2**30 * .45)
    assert budget.host_bytes == max(0, min(int(available * .7), available - 2**30))
    assert observations["device_calls"] == observations["host_calls"] == 1
    with pytest.raises(FrozenInstanceError):
        budget.initial_free_bytes = 0


def test_cuda_resident_estimate_counts_byte_relation_and_bounded_halo():
    n, coins = 3_000_000, 40
    for minute in range(60):
        resident = residency.multicoin_resident_bytes(
            n, coins, device="cuda", stream_tick_inputs=True,
            compact_hourly_ranges=True, start_minute_of_hour=minute,
        )
        assert resident == n * coins * 24 + 8194 * coins * 25 + (
            residency.compact_hourly_rows(n, start_minute_of_hour=minute) * coins * 4
        )
        assert resident < int(7 * 2**30 * .45)
    assert residency.multicoin_invariant_bytes(n, coins, device="cuda") > int(8 * 2**30 * .45)
    for n in (3, 8192, 8194):
        assert residency.multicoin_resident_bytes(n, coins, device="cuda", stream_tick_inputs=True) == (
            residency.multicoin_invariant_bytes(n, coins, device="cuda")
        )


def test_cuda_transport_preference_is_pure_and_keeps_small_histories_resident(monkeypatch):
    torch, observations = _cuda(monkeypatch, free=7 * 2**30)
    budget = residency.capture_cuda_input_memory_budget(torch)
    assert residency.prefer_streamed_ticks(3_000_000, 40, budget, device="cuda", interval_ms=60_000)
    assert not residency.prefer_streamed_ticks(32_768, 40, budget, device="cuda", interval_ms=60_000)
    observations.update(free=0, available=0)
    assert residency.prefer_streamed_ticks(3_000_000, 40, budget, device="cuda", interval_ms=60_000)
    assert observations["device_calls"] == observations["host_calls"] == 1


def test_cuda_header_preflight_admits_streaming_but_keeps_guard_and_live_check(monkeypatch):
    torch, observations = _cuda(monkeypatch, free=7 * 2**30)
    budget = residency.capture_cuda_input_memory_budget(torch)
    shape = (3_000_000, 40, 4)
    with pytest.raises(MemoryError, match="45%"):
        _gpu_cached_input_preflight(None, shape, np.float64, input_memory_budget=budget)
    _gpu_cached_input_preflight(None, shape, np.float64,
                                stream_tick_inputs=True, input_memory_budget=budget)
    observations["free"] = 2**30
    with pytest.raises(MemoryError, match="currently free"):
        _gpu_cached_input_preflight(None, shape, np.float64,
                                    stream_tick_inputs=True, input_memory_budget=budget)


def test_cuda_preparation_uses_original_budget_while_activation_checks_live_free(monkeypatch):
    torch, observations = _cuda(monkeypatch, free=7 * 2**30)
    budget = residency.capture_cuda_input_memory_budget(torch)
    manager = residency.CudaSuiteResidency(input_memory_budget=budget)
    observations["free"] = 2**30
    manager.check_preparation_memory(torch, 2 * 2**30)
    assert observations["device_calls"] == 1  # Prior scenario is not a smaller preparation budget.
    with pytest.raises(MemoryError, match="currently free"):
        manager._check_memory(torch, 2 * 2**30)
    with pytest.raises(MemoryError, match="45%"):
        manager.check_preparation_memory(torch, 4 * 2**30)


def test_cuda_compact_packing_preserves_dense_inputs_and_int8_relation(monkeypatch, tmp_path):
    _cuda(monkeypatch)
    n, coins, minute = 185, 3, 23
    timestamps = (1_700_000_000_000 // 3_600_000) * 3_600_000 + minute * 60_000
    timestamps = timestamps + np.arange(n, dtype=np.int64) * 60_000
    closes = 100.0 + np.arange(n)[:, None] * .002 + np.arange(coins)[None, :] * 7.0
    values = np.stack((closes * 1.002, closes * .998, closes, closes * .25), axis=2)
    runs = [model.ProxyRun(1000., 0, 0, int(timestamps[0]), 0,
                           int(timestamps[0]), 60_000, .05, 0, n - 1)] * coins
    markets = [model.ProxyMarket(.001, .01, .001, 1., 1., .0002)] * coins
    outputs = []
    for stream in (False, True):
        directory = tmp_path / str(stream)
        directory.mkdir()
        outputs.append(model.build_mps_multicoin_data(
            values, timestamps, runs, markets, spill_dir=directory,
            stream_tick_inputs=stream, compact_hourly_ranges=stream,
        ))
    dense, streamed = outputs
    assert streamed["stream_tick_inputs"] and streamed["compact_hourly_ranges"]
    assert np.load(streamed["touch_min_qty_relation"]).dtype == np.int8
    assert streamed["resident_bytes"] == residency.multicoin_resident_bytes(
        n, coins, device="cuda", stream_tick_inputs=True, compact_hourly_ranges=True,
        start_minute_of_hour=minute,
    )
    for name, path in dense.items():
        if not isinstance(path, Path):
            continue
        expected, actual = np.load(path), np.load(streamed[name])
        if name == "hour_log_ranges":
            decoded = np.full(expected.shape, -1.0, dtype=np.float32)
            boundaries = np.arange(60 - minute, n, 60)
            decoded[boundaries] = actual[:len(boundaries)]
            actual = decoded
        np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_cuda_residency_uploads_only_nonstreamed_fields(monkeypatch, tmp_path):
    torch, _ = _cuda(monkeypatch)
    uploads = []
    torch.cuda.synchronize = lambda: None
    torch.cuda.empty_cache = lambda: None
    torch.as_tensor = lambda array, **kwargs: SimpleNamespace(
        contiguous=lambda: uploads.append(Path(array.filename).stem) or object(),
    )

    class Owner:
        def __init__(self, data):
            self.data, self.runners, self.fused_runner = data, {}, None

        def _create_runners(self):
            assert all(isinstance(self.data[name], Path) for name in residency.METAL_STREAMED_TICK_FIELDS)
            self.runners["long"] = object()

    manager = residency.CudaSuiteResidency()

    def build(directory):
        data = dict(stream_tick_inputs=True, invariant_bytes=100, resident_bytes=50)
        for name in (*residency.METAL_STREAMED_TICK_FIELDS, "bars", "market_execution_ticks", "hour_log_ranges"):
            data[name] = directory / (name + ".npy")
            np.save(data[name], np.arange(5, dtype=np.int8))
        return data

    try:
        data = manager.prepare("input", build)
        owner = Owner(data)
        manager.register(owner)
        manager.activate(owner)
        assert set(uploads) == {"bars", "market_execution_ticks", "hour_log_ranges"}
    finally:
        manager.close()
