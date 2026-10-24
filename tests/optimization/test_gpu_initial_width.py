"""Device-specific initial width stays on the execution owner and inside limits."""

import builtins
from contextlib import contextmanager
from threading import Event, get_ident

import pytest

from optimization.gpu.executor import BacktestRequest, ReplayResult
from optimization.gpu.native import CudaBacktestService, NativeBacktestService


@pytest.mark.parametrize("setting,mode", [(None, "auto"), ("auto", "refresh"), (4, "auto"), (None, "off")])
def test_native_construction_never_imports_device_runtime(monkeypatch, setting, mode):
    original = builtins.__import__
    def cpu_only(name, *args, **kwargs):
        if name in {"torch", "optimization.gpu.runtime", "optimization.gpu.residency"}:
            pytest.fail(f"producer construction imported {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", cpu_only)
    with NativeBacktestService(batch_size=setting, tuning_mode=mode) as service:
        assert service._executor._thread is None
        assert not service._batch_policy.controllers


@pytest.mark.parametrize(
    "service_class,device,setting,mode,max_pending,prepared_ceiling,expected",
    [
        (NativeBacktestService, "mps", None, "auto", 320, 512, 256),
        (NativeBacktestService, "mps", "auto", "refresh", 320, 512, 256),
        (NativeBacktestService, "cuda", None, "auto", 320, 512, 64),
        (CudaBacktestService, "cuda", None, "auto", 320, 512, 64),
        (NativeBacktestService, "mps", None, "auto", 2, 512, 2),
        (NativeBacktestService, "mps", None, "auto", 1, 512, 1),
        (NativeBacktestService, "mps", None, "auto", 320, 3, 3),
        (NativeBacktestService, "mps", 4, "auto", 320, 512, 4),
        (NativeBacktestService, "mps", 4, "auto", 2, 512, 2),
        (NativeBacktestService, "cuda", 4, "auto", 320, 512, 4),
        (NativeBacktestService, "mps", None, "off", 320, 512, 64),
        (NativeBacktestService, "mps", None, "off", 1, 512, 1),
        (NativeBacktestService, "cuda", None, "off", 320, 512, 64),
    ],
)
def test_owner_selects_initial_width_before_prepared_admission_without_extra_replay(
    monkeypatch, service_class, device, setting, mode, max_pending, prepared_ceiling, expected
):
    from optimization.gpu import residency, runtime

    producer = get_ident()
    ownership, calls = [], []
    preparing, release = Event(), Event()
    def owner_device():
        assert get_ident() != producer
        ownership.append(("device", get_ident()))
        return device
    @contextmanager
    def metal_scope():
        ownership.append(("mps", get_ident()))
        yield None
    @contextmanager
    def cuda_scope():
        ownership.append(("cuda", get_ident()))
        yield None
    monkeypatch.setattr(runtime, "gpu_device", owner_device)
    monkeypatch.setattr(residency, "metal_residency_scope", metal_scope)
    monkeypatch.setattr(residency, "cuda_residency_scope", cuda_scope)
    service = service_class(batch_size=setting, tuning_mode=mode, max_pending=max_pending,
                            max_batch_delay=0)
    class Replay:
        def evaluate(self, candidates):
            calls.append([row["value"] for row in candidates])
            return [ReplayResult({"value": row["value"]}, False) for row in candidates]
    @contextmanager
    def factory():
        assert get_ident() == ownership[0][1] != producer
        # The worker first claims one unprepared request; only preparation owns
        # the work/scratch ceiling and lets the executor fill that dispatch.
        assert service._batch_policy.width("market", service._batch_size) == 1
        assert not service._batch_policy.controllers
        if service._batch_tuner is not None:
            assert service._batch_tuner.initial == (256 if device == "mps" else 64)
        preparing.set()
        assert release.wait(3)
        service._batch_policy.constrain("market", prepared_ceiling)
        yield Replay()
    service._executor.register_dataset_factory("market", factory)
    try:
        first = service.submit(BacktestRequest("0", "market", {"value": 0}))
        assert preparing.wait(3)
        assert first.running()
        futures = [first, *(service.submit(BacktestRequest(str(index), "market", {"value": index}))
                            for index in range(1, expected))]
        release.set()
        assert [future.result(timeout=3).metrics for future in futures] == [
            {"value": index} for index in range(expected)
        ]
        assert service._batch_policy.width("market", service._batch_size) == expected
    finally:
        release.set()
        service.close(cancel_pending=True)
    assert ownership == [("device", ownership[0][1]), (device, ownership[0][1])]
    assert calls == [list(range(expected))]
