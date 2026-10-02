"""CUDA suite residency preserves packed inputs and bounds simultaneous allocation."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from optimization.gpu.residency import (
    CudaSuiteResidency, cuda_suite_residency_scope, current_cuda_residency,
)


@pytest.fixture
def runtime(monkeypatch):
    uploads = []
    cuda = SimpleNamespace(
        mem_get_info=lambda: (1000, 2000), synchronize=lambda: None,
        empty_cache=lambda: None,
    )
    def upload(array, **kwargs):
        assert kwargs["device"] == "cuda"
        uploads.append(array.copy())
        return SimpleNamespace(contiguous=lambda: uploads[-1])
    torch = SimpleNamespace(cuda=cuda, as_tensor=upload)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch, uploads


def proxy(manager, key, value, invariant_bytes=100):
    def build(directory):
        path = directory / "bars.npy"
        np.save(path, np.array([value], dtype=np.int8), allow_pickle=False)
        return {"bars": path, "invariant_bytes": invariant_bytes}
    data = manager.prepare(key, build)
    owner = SimpleNamespace(data=data, runners={}, fused_runner=None)
    # SimpleNamespace cannot be weak-referenced.
    class Owner:
        def __init__(self):
            self.__dict__.update(owner.__dict__)
        def _create_runners(self):
            self.runners["long"] = self.data["bars"]
    result = Owner()
    manager.register(result)
    return result


def test_switching_evicts_all_owners_and_preserves_identity_and_values(runtime):
    _, uploads = runtime
    manager = CudaSuiteResidency()
    try:
        first = proxy(manager, "a", 12)
        compatible = proxy(manager, "a", 99)
        other = proxy(manager, "b", -7)
        identity = id(first.data)
        assert not uploads
        assert compatible.data is first.data
        manager.activate(first)
        first.runners["scratch"] = object()
        first.fused_runner = object()
        manager.activate(compatible)
        assert len(uploads) == 1
        assert first.runners == {}
        assert first.fused_runner is None
        assert compatible.runners
        manager.activate(first)
        assert compatible.runners == {}
        assert len(uploads) == 1  # Owner changes never reupload shared market inputs.
        manager.activate(other)
        assert first.runners == compatible.runners == {}
        assert isinstance(first.data["bars"], Path)
        manager.activate(first)
        assert other.runners == {}
        assert id(first.data) == identity
        np.testing.assert_array_equal(first.data["bars"], np.array([12], dtype=np.int8))
        assert [a.dtype for a in uploads] == [np.dtype("int8")] * 3
    finally:
        directory = manager._directory.name
        manager.close()
    assert not Path(directory).exists()
    assert first.runners == {}


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_backend_failure_survives_cleanup_error(runtime, error, caplog):
    torch, _ = runtime
    failure = error("original backend failure")
    directories = []
    def fail_cleanup():
        raise RuntimeError("cleanup synchronization failed")
    @cuda_suite_residency_scope
    def run():
        manager = current_cuda_residency()
        owner = proxy(manager, "a", 1)
        manager.activate(owner)
        directories.append(manager._directory.name)
        torch.cuda.synchronize = fail_cleanup
        raise failure
    with pytest.raises(error) as caught:
        run()
    assert caught.value is failure
    assert "CUDA suite cleanup failed" in caplog.text
    assert "cleanup synchronization failed" in caplog.text
    assert current_cuda_residency() is None
    assert not Path(directories[0]).exists()


def test_successful_backend_propagates_cleanup_error(runtime):
    torch, _ = runtime
    directories = []
    def fail_cleanup():
        raise RuntimeError("cleanup synchronization failed")
    @cuda_suite_residency_scope
    def run():
        manager = current_cuda_residency()
        owner = proxy(manager, "a", 1)
        manager.activate(owner)
        directories.append(manager._directory.name)
        torch.cuda.synchronize = fail_cleanup
        return "success"
    with pytest.raises(RuntimeError, match="cleanup synchronization failed"):
        run()
    assert current_cuda_residency() is None
    assert not Path(directories[0]).exists()


def test_switching_retains_original_memory_limit(runtime):
    manager = CudaSuiteResidency()
    try:
        first = proxy(manager, "a", 1)
        large = proxy(manager, "b", 2, invariant_bytes=451)
        manager.activate(first)
        with pytest.raises(MemoryError, match="45% safety limit.*free VRAM"):
            manager.activate(large)
        assert first.runners == {}
        assert manager._active is None
    finally:
        manager.close()


@pytest.mark.parametrize("stage", ["upload", "runner"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_activation_failure_releases_partial_resources(runtime, stage, error):
    torch, _ = runtime
    manager = CudaSuiteResidency()
    owner = proxy(manager, "a", 1)
    def fail(*args, **kwargs):
        if stage == "runner":
            owner.runners["long"] = owner.data["bars"]
            owner.fused_runner = owner.data["bars"]
        raise error("failed")
    if stage == "upload":
        torch.as_tensor = fail
    else:
        owner._create_runners = fail
    try:
        with pytest.raises(error, match="failed"):
            manager.activate(owner)
        assert manager._active is None
        assert owner.runners == {}
        assert owner.fused_runner is None
        assert isinstance(owner.data["bars"], Path)
    finally:
        manager.close()


@pytest.mark.parametrize("stage", ["upload", "runner"])
def test_activation_failure_survives_cleanup_error(runtime, stage, caplog):
    torch, _ = runtime
    manager = CudaSuiteResidency()
    owner = proxy(manager, "a", 1)
    failure = ValueError("original activation failure")
    def fail_cleanup():
        raise RuntimeError("cleanup synchronization failed")
    def fail(*args, **kwargs):
        torch.cuda.synchronize = fail_cleanup
        raise failure
    if stage == "upload":
        torch.as_tensor = fail
    else:
        owner._create_runners = fail
    try:
        with pytest.raises(ValueError) as caught:
            manager.activate(owner)
        assert caught.value is failure
        assert "CUDA suite cleanup failed" in caplog.text
        assert manager._active is None
        assert manager._active_owner is None
        assert owner.runners == {}
    finally:
        manager.close()


def test_failed_packing_can_be_retried(runtime):
    manager = CudaSuiteResidency()
    def fail(directory):
        (directory / "partial").write_text("partial")
        raise ValueError("invalid")
    try:
        with pytest.raises(ValueError, match="invalid"):
            manager.prepare("a", fail)
        assert manager._entries == {}
        assert list(Path(manager._directory.name).iterdir()) == []
        proxy(manager, "a", 1)
    finally:
        manager.close()


def test_backend_scope_cleans_files_and_restores_context_on_interrupt(runtime):
    directories = []
    @cuda_suite_residency_scope
    def run():
        manager = current_cuda_residency()
        owner = proxy(manager, "a", 1)
        manager.activate(owner)
        directories.append(manager._directory.name)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run()
    assert current_cuda_residency() is None
    assert not Path(directories[0]).exists()


def test_close_cleans_files_even_when_cuda_synchronization_fails(runtime):
    torch, _ = runtime
    manager = CudaSuiteResidency()
    owner = proxy(manager, "a", 1)
    manager.activate(owner)
    directory = manager._directory.name
    def fail():
        raise RuntimeError("CUDA failed")
    torch.cuda.synchronize = fail
    with pytest.raises(RuntimeError, match="CUDA failed"):
        manager.close()
    assert owner.runners == {}
    assert manager._active is None
    assert not Path(directory).exists()


def test_zero_free_vram_does_not_bypass_guard(runtime):
    torch, _ = runtime
    manager = CudaSuiteResidency()
    try:
        owner = proxy(manager, "a", 1)
        torch.cuda.mem_get_info = lambda: (0, 2000)
        with pytest.raises(MemoryError, match="45% safety limit"):
            manager.activate(owner)
        assert manager._active is None
    finally:
        manager.close()


def test_driver_workspace_does_not_shrink_budget_but_current_free_is_checked(runtime):
    torch, _ = runtime
    manager = CudaSuiteResidency()
    try:
        first = proxy(manager, "a", 1)
        second = proxy(manager, "b", 2, invariant_bytes=400)
        manager.activate(first)  # Initial free VRAM is 1000; invariant limit is 450.
        torch.cuda.mem_get_info = lambda: (500, 2000)
        manager.activate(second)  # A retained driver workspace consumes 500.
        manager.activate(first)
        torch.cuda.mem_get_info = lambda: (300, 2000)
        with pytest.raises(MemoryError, match="currently free"):
            manager.activate(second)
        assert manager._active is None
    finally:
        manager.close()


@pytest.mark.parametrize("hourly", [True, False])
def test_disk_packing_and_activation_preserve_every_eager_array(runtime, hourly):
    from optimization.gpu.model import ProxyMarket, ProxyRun, build_mps_multicoin_data

    torch, uploads = runtime
    torch.float32, torch.int32, torch.int8 = np.float32, np.int32, np.int8
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda.is_available = lambda: True
    torch.cuda.mem_get_info = lambda: (1_000_000, 2_000_000)
    count = 121
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    close = 100 + np.sin(np.arange(count) / 7)
    values = np.stack([close + .019, close - .021, close, close * .5], axis=1)
    values = np.stack([values, values * 1.5], axis=1)
    run = ProxyRun(1000, 1, 1, int(timestamps[1]), 0, int(timestamps[0]), 60_000, .05, 0, count - 1)
    market = ProxyMarket(.001, .01, .001, 5, 1, .0002)
    def build(directory=None):
        return build_mps_multicoin_data(
            values, timestamps, runs=[run] * 2, markets=[market] * 2,
            include_hourly_ranges=hourly, spill_dir=directory,
        )
    manager = CudaSuiteResidency()
    try:
        data = manager.prepare("a", build)
        assert uploads == []
        expected = build()
        class Owner:
            def _create_runners(self):
                pass
        owner = Owner()
        owner.data, owner.runners, owner.fused_runner = data, {}, None
        manager.register(owner)
        manager.activate(owner)
        assert data.keys() == expected.keys()
        for name, value in expected.items():
            if isinstance(value, np.ndarray):
                assert data[name].dtype == value.dtype
                np.testing.assert_array_equal(data[name], value, err_msg=name)
            else:
                assert data[name] == value
        assert data["touch_min_qty_relation"].dtype == np.int8
    finally:
        manager.close()
