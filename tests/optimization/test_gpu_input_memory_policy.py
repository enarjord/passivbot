from dataclasses import FrozenInstanceError
import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gpu import residency


def _telemetry(monkeypatch, *, recommended=12 * 2**30, driver=2**30, available=8 * 2**30):
    values = dict(recommended=recommended, driver=driver, available=available, total=16 * 2**30)
    calls = dict(recommended=0, driver=0, available=0)
    def read(name):
        calls[name] += 1
        return values[name]
    torch = SimpleNamespace(mps=SimpleNamespace(
        recommended_max_memory=lambda: read("recommended"),
        driver_allocated_memory=lambda: read("driver"),
    ))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=read("available"), total=values["total"]),
    ))
    return torch, values, calls


@pytest.mark.parametrize("available", [0, 2**29, 2**30, 2 * 2**30, 8 * 2**30])
def test_budget_captures_device_and_host_once_with_reserved_headroom(monkeypatch, available):
    torch, _values, calls = _telemetry(monkeypatch, available=available)
    budget = residency.capture_mps_input_memory_budget(torch)
    assert budget.device_bytes == int((12 * 2**30 - 2**30) * 0.45)
    assert budget.host_bytes == max(0, min(int(available * 0.7), available - 2**30))
    assert calls == dict(recommended=1, driver=1, available=1)
    with pytest.raises(FrozenInstanceError):
        budget.host_bytes = 0


def test_budget_clamps_negative_device_headroom(monkeypatch):
    torch, _values, _calls = _telemetry(monkeypatch, recommended=1, driver=2)
    assert residency.capture_mps_input_memory_budget(torch).device_bytes == 0


def test_max_input_preflight_uses_frozen_allowance_without_live_probes(monkeypatch):
    torch, values, calls = _telemetry(monkeypatch)
    budget = residency.capture_mps_input_memory_budget(torch, max_throughput=True)
    assert budget.max_throughput
    values.update(driver=values["recommended"], available=0)

    residency.check_mps_invariant_memory(
        torch, budget.device_bytes, input_memory_budget=budget, require_free=False,
    )
    with pytest.raises(MemoryError, match="initial max-mode input budget") as failure:
        residency.check_mps_invariant_memory(
            torch, budget.device_bytes + 1, input_memory_budget=budget, require_free=False,
        )
    assert type(failure.value) is MemoryError
    assert calls == dict(recommended=1, driver=1, available=1)


@pytest.mark.parametrize(
    "driver,available,pending,expected",
    [
        (7, 10, 1, 1 * 2**30),  # MPS working set has only a GiB left above reserve.
        (1, 3, 1, 2 * 2**30 - (16 * 2**30) // 10),  # macOS reserve and pending host work.
        (1, 1, 1, 0),
    ],
)
def test_max_live_admission_uses_tighter_of_mps_and_system_headroom(
    monkeypatch, driver, available, pending, expected,
):
    torch, _values, calls = _telemetry(
        monkeypatch, recommended=10 * 2**30, driver=driver * 2**30, available=available * 2**30,
    )
    assert residency.mps_max_available_bytes(torch, pending_host_bytes=pending * 2**30) == expected
    assert calls == dict(recommended=1, driver=1, available=1)


def test_unavailable_system_telemetry_only_declines_optional_expansion(monkeypatch):
    torch, values, _calls = _telemetry(monkeypatch)
    values["available"] = -1
    assert residency.mps_max_available_bytes(torch, require_system_telemetry=False) == 0
    with pytest.raises(RuntimeError, match="requires valid system memory telemetry"):
        residency.mps_max_available_bytes(torch)

    def broken_device():
        raise RuntimeError("driver unavailable")

    torch.mps.driver_allocated_memory = broken_device
    with pytest.raises(RuntimeError, match="driver unavailable"):
        residency.mps_max_available_bytes(torch, require_system_telemetry=False)


def test_max_invariant_live_admission_counts_upload_source_pages(monkeypatch):
    torch, values, _calls = _telemetry(monkeypatch)
    budget = residency.capture_mps_input_memory_budget(torch, max_throughput=True)
    resident = 2**30
    values["available"] = residency.mps_system_reserve_bytes(values["total"]) + resident
    with pytest.raises(residency.MetalReplayMemoryPressure, match="live unified-memory headroom"):
        residency.check_mps_invariant_memory(torch, resident, input_memory_budget=budget)

    values["available"] += residency.PREPARATION_CHUNK_BYTES
    residency.check_mps_invariant_memory(torch, resident, input_memory_budget=budget)


@pytest.mark.parametrize("candles", [1, 8192, 8194])
def test_small_histories_keep_resident_inputs_when_streaming_saves_nothing(candles):
    assert residency.multicoin_resident_bytes(candles, 8, device="mps", stream_tick_inputs=True) == (
        residency.multicoin_invariant_bytes(candles, 8, device="mps")
    )
    assert residency.prefer_streamed_ticks(candles, 8, residency.MpsInputMemoryBudget(0, 0)) is False


@pytest.mark.parametrize("limiting", ["device", "host"])
@pytest.mark.parametrize("difference,stream", [(1, False), (0, False), (-1, True)])
def test_transport_choice_exact_comfort_boundary(limiting, difference, stream):
    candles, coins = 32_768, 8
    full = residency.multicoin_invariant_bytes(candles, coins, device="mps")
    budgets = dict(device_bytes=full * 2, host_bytes=full * 2)
    budgets[f"{limiting}_bytes"] = full + difference
    assert residency.prefer_streamed_ticks(candles, coins, residency.MpsInputMemoryBudget(**budgets)) is stream


def test_large_history_prefers_streaming_even_when_separate_admission_may_refuse():
    # Selection does not weaken or replace the later live memory check.
    assert residency.prefer_streamed_ticks(3_000_000, 40, residency.MpsInputMemoryBudget(1, 1)) is True


def test_preflight_and_effective_shapes_share_frozen_budget_despite_cache_page_growth(monkeypatch):
    torch, values, calls = _telemetry(monkeypatch)
    budget = residency.capture_mps_input_memory_budget(torch)
    full_shape, cropped_shape = (3_000_000, 40), (32_768, 8)
    assert residency.prefer_streamed_ticks(*full_shape, budget) is True
    assert residency.prefer_streamed_ticks(*cropped_shape, budget) is False
    values.update(available=0, driver=values["recommended"])
    # An exact header and its effective service shape choose the same transport;
    # suite/override callers apply the same rule after taking their actual slice.
    assert residency.prefer_streamed_ticks(*full_shape, budget) is True
    assert residency.prefer_streamed_ticks(*cropped_shape, budget) is False
    assert calls == dict(recommended=1, driver=1, available=1)


def _raw_mapping(tmp_path, *, mode="r"):
    path = tmp_path / "hlcvs.npy"
    expected = np.arange(16_384, dtype=np.float64).reshape(2048, 2, 4)
    np.save(path, expected)
    return np.load(path, mmap_mode=mode, allow_pickle=False), expected


def test_discarded_readonly_pages_preserve_mapping_and_exact_readback(tmp_path):
    raw, expected = _raw_mapping(tmp_path)
    try:
        values = np.asarray(raw)[5:1000]
        supported = (hasattr(residency.mmap, "MADV_DONTNEED")
                     and callable(getattr(raw._mmap, "madvise", None)))
        assert residency.discard_readonly_file_pages(values) is supported
        assert not raw._mmap.closed
        np.testing.assert_array_equal(raw, expected)
        np.testing.assert_array_equal(values, expected[5:1000])
    finally:
        raw._mmap.close()


@pytest.mark.parametrize("mode", ["r+", "c"])
def test_writable_and_copy_on_write_mappings_are_never_advised(tmp_path, mode):
    raw, _expected = _raw_mapping(tmp_path, mode=mode)
    try:
        assert residency.discard_readonly_file_pages(raw) is False
        raw.flags.writeable = False
        assert residency.discard_readonly_file_pages(raw) is False
    finally:
        raw._mmap.close()


def test_ordinary_arrays_are_never_advised():
    values = np.arange(10)
    assert residency.discard_readonly_file_pages(values) is False
    values.flags.writeable = False
    assert residency.discard_readonly_file_pages(values) is False


@pytest.mark.parametrize("support", ["supported", "no_constant", "no_method", "os_error"])
def test_readonly_mapping_advice_is_optional(tmp_path, monkeypatch, caplog, support):
    raw, expected = _raw_mapping(tmp_path)
    original = raw._mmap
    advised = []
    def advise(value):
        advised.append(value)
        if support == "os_error":
            raise OSError("optional advice failed")
    raw._mmap = (SimpleNamespace() if support == "no_method"
                 else SimpleNamespace(madvise=advise))
    monkeypatch.setattr(residency, "mmap", (SimpleNamespace() if support == "no_constant"
                                          else SimpleNamespace(MADV_DONTNEED=123)))
    try:
        with caplog.at_level(logging.DEBUG):
            result = residency.discard_readonly_file_pages(np.asarray(raw))
        assert result is (support == "supported")
        assert advised == ([123] if support in {"supported", "os_error"} else [])
        if support == "os_error":
            assert "Could not discard read-only raw input pages" in caplog.text
        np.testing.assert_array_equal(raw, expected)
    finally:
        raw._mmap = original
        original.close()
