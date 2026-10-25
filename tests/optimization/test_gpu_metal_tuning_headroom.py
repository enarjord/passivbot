"""Automatic growth requires both device and unified host memory headroom."""

import math
import sys
from threading import Event
from types import SimpleNamespace
import time

import pytest

from optimization.gpu import native


MIB = 1024**2
GIB = 1024**3


def runtime(monkeypatch, *, device="mps", free=8 * GIB):
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: device == "mps")),
        mps=SimpleNamespace(recommended_max_memory=lambda: 10 * GIB,
                            driver_allocated_memory=lambda: 10 * GIB - free),
        cuda=SimpleNamespace(is_available=lambda: device == "cuda",
                             mem_get_info=lambda: (free, 10 * GIB)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.parametrize("total,available,expected", [
    (16 * GIB, GIB, False),
    (16 * GIB, 2 * GIB, True),
    (16 * GIB, math.ceil(16 * GIB * .1) - 1, False),
    (16 * GIB, math.ceil(16 * GIB * .1), True),
    (2 * GIB, 256 * MIB - 1, False),
    (2 * GIB, 256 * MIB, True),
])
def test_metal_growth_requires_live_os_headroom(monkeypatch, total, available, expected):
    runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=total, available=available),
    ))
    assert native.NativeBacktestService._headroom() is expected


@pytest.mark.parametrize("snapshot", [
    SimpleNamespace(total=0, available=0),
    SimpleNamespace(total=GIB, available=-1),
    SimpleNamespace(total=GIB, available=2 * GIB),
    SimpleNamespace(total=None, available=GIB),
    SimpleNamespace(total=True, available=GIB),
    SimpleNamespace(total=GIB, available=float("nan")),
    SimpleNamespace(total=GIB),
])
def test_unusable_os_telemetry_conservatively_denies_growth(monkeypatch, snapshot):
    runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: snapshot))
    assert native.NativeBacktestService._headroom() is False


@pytest.mark.parametrize("missing", [False, True])
def test_os_telemetry_failure_denies_growth_without_raising(monkeypatch, missing):
    runtime(monkeypatch)

    def unavailable():
        raise OSError("memory telemetry unavailable")

    monkeypatch.setitem(sys.modules, "psutil", None if missing else SimpleNamespace(
        virtual_memory=unavailable,
    ))
    assert native.NativeBacktestService._headroom() is False


@pytest.mark.parametrize("device,free,expected", [
    ("mps", GIB, False), ("cuda", GIB, False), ("cuda", 8 * GIB, True),
])
def test_device_guard_and_cuda_do_not_depend_on_os_telemetry(monkeypatch, device, free, expected):
    runtime(monkeypatch, device=device, free=free)

    def forbidden():
        pytest.fail("OS telemetry must not be queried on this path")

    monkeypatch.setattr(native, "_metal_system_headroom", forbidden)
    assert native.NativeBacktestService._headroom() is expected


def test_unavailable_os_memory_denies_trial_without_failing_production_results(monkeypatch):
    from optimization.gpu import executor
    from optimization.gpu.autotune import record_compilation
    from optimization.gpu.execution_tuning import ExecutionBatchTuner

    runtime(monkeypatch)

    def unavailable():
        raise OSError("memory telemetry unavailable")

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(virtual_memory=unavailable))
    queries = []

    def headroom():
        queries.append(True)
        return native.NativeBacktestService._headroom()

    policy = ExecutionBatchTuner(initial=2, headroom=headroom)
    policy.constrain("market", 4)
    clock, started, release = [0.0], Event(), Event()
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        perf_counter=lambda: clock[0], monotonic=time.monotonic,
    ))

    class Replay:
        def evaluate(self, candidates):
            record_compilation(False)
            clock[0] += 10
            started.set()
            assert release.wait(3)
            return [executor.ReplayResult({"value": row["value"]}, False) for row in candidates]

    service = executor.GpuBacktestService(batch_size=4, max_pending=8, max_batch_delay=.05,
                                          batch_policy=policy)
    service.register_dataset("market", Replay())

    def submit(value):
        return service.submit(executor.BacktestRequest(str(value), "market", {"value": value}))

    try:
        futures = [submit(0), submit(1)]
        assert started.wait(3)
        futures.extend(submit(value) for value in range(2, 6))
        release.set()
        assert [future.result(timeout=3).metrics for future in futures] == [
            {"value": value} for value in range(6)
        ]
        assert queries == [True]
        assert policy.controllers["market"].width == 1
    finally:
        release.set()
        service.close(cancel_pending=True)
