"""CUDA input transport fits long histories without changing packed values."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu import model, residency, runtime
from optimize import _gpu_cached_input_preflight


def test_rocm_max_is_rejected_before_memory_preparation():
    torch = SimpleNamespace(version=SimpleNamespace(hip="7.0"))
    with pytest.raises(ValueError, match="max.*unavailable on AMD ROCm"):
        residency.capture_cuda_input_memory_budget(torch, max_throughput=True)


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


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cuda_header_preflight_admits_streaming_but_keeps_guard_and_live_check(monkeypatch, strategy):
    torch, observations = _cuda(monkeypatch, free=7 * 2**30)
    budget = residency.capture_cuda_input_memory_budget(torch)
    shape = (3_000_000, 40, 4)
    with pytest.raises(MemoryError, match="45%"):
        _gpu_cached_input_preflight(None, shape, np.float64, input_memory_budget=budget)
    _gpu_cached_input_preflight(None, shape, np.float64,
                                strategy_kind=strategy, input_memory_budget=budget)
    observations["free"] = 2**30
    with pytest.raises(MemoryError, match="currently free"):
        _gpu_cached_input_preflight(None, shape, np.float64,
                                    strategy_kind=strategy, input_memory_budget=budget)


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


def test_max_resident_ceiling_threads_through_header_preflight_and_activation(monkeypatch):
    gib = 2**30
    torch, observations = _cuda(monkeypatch, free=7 * gib, available=8 * gib)
    conservative = residency.capture_cuda_input_memory_budget(torch)
    maximum = residency.capture_cuda_input_memory_budget(torch, max_throughput=True)
    assert maximum.device_bytes == int(7 * gib * 0.70)
    assert maximum.host_bytes == int(8 * gib * 0.85)
    shape = (2_000_000, 40, 4)  # 3.95 GiB dense, inside the new resident limit.
    assert residency.prefer_streamed_ticks(*shape[:2], conservative, device="cuda")
    assert not residency.prefer_streamed_ticks(*shape[:2], maximum, device="cuda")
    _gpu_cached_input_preflight(None, shape, np.float64, input_memory_budget=maximum)
    resident = residency.multicoin_resident_bytes(*shape[:2], device="cuda")
    manager = residency.CudaSuiteResidency(input_memory_budget=maximum)
    observations["free"] = 4 * gib
    manager.check_preparation_memory(torch, resident)
    with pytest.raises(MemoryError, match="currently free.*reserve"):
        manager._check_memory(torch, resident)
    with pytest.raises(MemoryError, match="max-mode input budget"):
        manager.check_preparation_memory(torch, 5 * gib)


def test_max_cannot_spend_vram_needed_by_the_reserve_and_replay(monkeypatch):
    torch, _ = _cuda(monkeypatch, free=2**30, available=2**28)
    budget = residency.capture_cuda_input_memory_budget(torch, max_throughput=True)
    assert budget.device_bytes == budget.host_bytes == 0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_max_header_preflight_keeps_streamable_dense_input_under_live_pressure(monkeypatch, strategy):
    gib = 2**30
    torch, observations = _cuda(monkeypatch, free=7 * gib)
    budget = residency.capture_cuda_input_memory_budget(torch, max_throughput=True)
    shape = (2_300_000, 40, 4)
    assert not residency.prefer_streamed_ticks(*shape[:2], budget, device="cuda")
    observations["free"] = 4 * gib
    _gpu_cached_input_preflight(None, shape, np.float64, strategy_kind=strategy, input_memory_budget=budget)
    with pytest.raises(residency.CudaReplayMemoryPressure, match="currently free"):
        _gpu_cached_input_preflight(None, shape, np.float64, input_memory_budget=budget)
    observations["free"] = 2 * gib
    with pytest.raises(residency.CudaReplayMemoryPressure, match="currently free"):
        _gpu_cached_input_preflight(None, shape, np.float64, strategy_kind=strategy, input_memory_budget=budget)


@pytest.mark.parametrize("pressure", ["upload_admission", "first_runner", "after_replay", "streamed_fails", "untyped"])
def test_max_falls_back_once_using_same_packed_tick_files(monkeypatch, pressure):
    from weakref import ref

    gib = 2**30
    torch, observations = _cuda(monkeypatch, free=7 * gib)
    budget = residency.capture_cuda_input_memory_budget(torch, max_throughput=True)
    torch.cuda.synchronize = lambda: None
    torch.cuda.empty_cache = lambda: None
    manager = residency.CudaSuiteResidency(input_memory_budget=budget)
    uploads, created = [], []
    tick_fields = residency.METAL_STREAMED_TICK_FIELDS
    invariant = residency.multicoin_invariant_bytes(2_300_000, 40, device="cuda")

    class Tensor:
        pass

    def upload(_torch, path):
        uploads.append(path.stem)
        return Tensor()

    monkeypatch.setattr(manager, "_upload_file", upload)
    dense_tensors = []

    class Owner:
        _max_throughput = True
        temporal_chunking = True
        strategy_kind = "ema_anchor"

        def __init__(self, data):
            self.data, self.runners, self.fused_runner = data, {}, None

        def _create_runners(self):
            stream = self.data["stream_tick_inputs"]
            created.append(stream)
            if stream:
                assert all(tensor() is None for tensor in dense_tensors)
                assert all(isinstance(self.data[name], Path) for name in tick_fields)
                if pressure == "streamed_fails":
                    raise residency.CudaReplayMemoryPressure("cannot admit streamed runner")
            else:
                dense_tensors.extend(ref(self.data[name]) for name in tick_fields)
                if pressure == "untyped":
                    raise MemoryError("actual CUDA allocation failed")
                if pressure in {"first_runner", "streamed_fails"}:
                    dense_reference = self.data[next(iter(tick_fields))]
                    assert isinstance(dense_reference, Tensor)
                    raise residency.CudaReplayMemoryPressure("cannot admit dense runner")
            self.fused_runner = Tensor()

    def build(directory):
        names = (*tick_fields, "bars", "market_execution_ticks", "hour_log_ranges")
        return dict(n=2_300_000, n_coins=40, stream_tick_inputs=False, compact_hourly_ranges=False,
                    invariant_bytes=invariant, resident_bytes=invariant,
                    **{name: directory / (name + ".npy") for name in names})

    try:
        data = manager.prepare("history", build)
        owner = Owner(data)
        manager.register(owner)
        if pressure == "upload_admission":
            observations["free"] = 4 * gib
        if pressure == "untyped":
            with pytest.raises(MemoryError, match="actual CUDA allocation failed"):
                manager.activate(owner)
            assert created == [False] and not data["stream_tick_inputs"]
            return
        if pressure == "streamed_fails":
            with pytest.raises(residency.CudaReplayMemoryPressure, match="cannot admit streamed runner"):
                manager.activate(owner)
            assert created == [False, True] and manager._active is None
            return
        manager.activate(owner)
        if pressure == "after_replay":
            manager.stream_on_pressure(owner)
        assert created == ([True] if pressure == "upload_admission" else [False, True])
        assert not data["compact_hourly_ranges"]
        assert data["resident_bytes"] == residency.multicoin_resident_bytes(
            2_300_000, 40, device="cuda", stream_tick_inputs=True,
        )
        assert data["invariant_bytes"] == invariant
        assert not manager.can_stream_on_pressure(owner)
        assert len(manager._entries) == 1 and manager.prepare("history", build) is data
        assert set(uploads[-3:]) == {"bars", "market_execution_ticks", "hour_log_ranges"}
    finally:
        manager.close()


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
