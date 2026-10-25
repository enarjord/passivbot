"""Metal disk preparation does not count the previous GPU owner twice."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from optimization.gpu import model, residency


@pytest.fixture
def metal_memory(monkeypatch):
    memory = {"recommended": 1000, "used": 100, "external": 100}
    uploads, events = [], []

    class Array(np.ndarray):
        def contiguous(self):
            return self

    def recommended():
        events.append("recommended")
        return memory["recommended"]

    def allocated():
        events.append("allocated")
        return memory["used"]

    def empty_cache():
        events.append("release")
        memory["used"] = memory["external"]

    def upload(array, *, device):
        assert device == "mps"
        uploads.append(np.array(array, copy=True).view(Array))
        return uploads[-1]

    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        float32=np.float32,
        as_tensor=upload,
        mps=SimpleNamespace(
            recommended_max_memory=recommended, driver_allocated_memory=allocated,
            synchronize=lambda: events.append("synchronize"), empty_cache=empty_cache,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, memory, uploads, events


def build(directory, candles=6):
    values = np.tile([101.0, 99.0, 100.0, 1.0], (candles, 1, 1))
    run = model.ProxyRun(1000.0, 0, 0, 0, 0, 0, 60000, 0.05, 0, candles - 1)
    market = model.ProxyMarket(0.001, 0.01, 0.001, 1.0, 1.0, 0.0002)
    return model.build_mps_multicoin_data(
        values, np.arange(candles) * 60000, [run], [market], spill_dir=directory,
    )


class Owner:
    def __init__(self, manager, data, memory):
        self.data = data
        self.memory = memory
        self.runners = {}
        self.fused_runner = None
        manager.register(self)

    def _create_runners(self):
        self.runners["scratch"] = object()
        self.memory["used"] = 700


def test_second_spill_reuses_initial_budget_and_activation_still_checks_live_free(metal_memory):
    _torch, memory, uploads, events = metal_memory
    with residency.metal_residency_scope() as manager:
        first = Owner(manager, manager.prepare("first", build), memory)
        assert manager._working_set == 900
        assert first.data["invariant_bytes"] == 336
        manager.activate(first)
        assert memory["used"] == 700
        first_scratch = first.runners["scratch"]
        before = list(events)

        # The second payload fits 45% of the initial budget, but not even the
        # current free space while the first payload and scratch remain active.
        second = Owner(manager, manager.prepare("second", build), memory)
        assert events == before
        assert manager._working_set == 900
        assert first.runners["scratch"] is first_scratch
        assert manager._active["data"] is first.data
        assert isinstance(second.data["bars"], Path)

        manager.activate(second)
        assert not first.runners and isinstance(first.data["bars"], Path)
        assert second.runners and manager._active["data"] is second.data
        assert events.count("release") == 1

        # Files can still be prepared under unrelated pressure. Activation
        # releases the old owner first, then rejects insufficient live space.
        memory["used"], memory["external"] = 980, 950
        third = Owner(manager, manager.prepare("third", build), memory)
        prior_uploads = len(uploads)
        with pytest.raises(MemoryError, match="available recommended working set"):
            manager.activate(third)
        assert memory["used"] == 950
        assert not second.runners and not third.runners
        assert manager._active is None and len(uploads) == prior_uploads
        assert manager._working_set == 900


def test_preparation_keeps_initial_budget_when_more_memory_later_becomes_free(metal_memory, monkeypatch):
    _torch, memory, _uploads, _events = metal_memory
    with residency.metal_residency_scope() as manager:
        manager.prepare("first", build)
        assert manager._working_set == 900
        memory["used"] = 0

        def forbidden(*_args, **_kwargs):
            pytest.fail("oversized payload must fail before packing")

        monkeypatch.setattr(model, "_spill_multicoin_arrays", forbidden)
        with pytest.raises(MemoryError, match="45%"):
            manager.prepare("large", lambda directory: build(directory, candles=8))
        assert manager._working_set == 900


def test_direct_spill_uses_nominal_budget_without_a_live_free_requirement(metal_memory, monkeypatch, tmp_path):
    torch, memory, uploads, _events = metal_memory
    memory["used"] = 900

    def forbidden():
        pytest.fail("direct disk packing must not query live device allocations")

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", forbidden)
    packed = build(tmp_path)
    assert isinstance(packed["bars"], Path)
    assert packed["invariant_bytes"] == 336 and not uploads


def test_direct_spill_still_rejects_the_nominal_invariant_limit(metal_memory, monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("oversized disk payload must fail before packing")

    monkeypatch.setattr(model, "_spill_multicoin_arrays", forbidden)
    with pytest.raises(MemoryError, match="45%"):
        build(tmp_path, candles=9)


def test_direct_device_build_keeps_current_live_admission(metal_memory, monkeypatch):
    _torch, memory, uploads, _events = metal_memory
    memory["used"] = 900

    def forbidden(*_args, **_kwargs):
        pytest.fail("device payload must fail before packing")

    monkeypatch.setattr(model, "_pack_multicoin_price_arrays", forbidden)
    with pytest.raises(MemoryError, match="45%"):
        build(None)
    assert not uploads
