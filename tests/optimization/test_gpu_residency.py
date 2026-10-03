"""CUDA suite residency preserves packed inputs and bounds simultaneous allocation."""

from dataclasses import replace
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
@pytest.mark.parametrize("fill_buffer", [0.0, 0.0001])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_disk_packing_and_activation_preserve_every_eager_array(runtime, hourly, fill_buffer, dtype, monkeypatch):
    from optimization.gpu.model import ProxyMarket, ProxyRun, build_mps_multicoin_data

    torch, uploads = runtime
    torch.float32, torch.int32, torch.int8 = np.float32, np.int32, np.int8
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda.is_available = lambda: True
    torch.cuda.mem_get_info = lambda: (1_000_000, 2_000_000)
    count = 121
    # Force several packing chunks, including a one-row tail and hour boundaries.
    monkeypatch.setattr("optimization.gpu.residency.PREPARATION_CHUNK_BYTES", 13 * (2 * 45 + 256))
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    close = 100 + np.sin(np.arange(count) / 7)
    values = np.stack([close + .019, close - .021, close, close * .5], axis=1)
    values = np.stack([values, values * 1.5], axis=1)
    values = values.astype(dtype)
    values[:3, 1] = np.nan
    values[-2:, 1] = np.nan
    values.flags.writeable = False  # Disk-backed scenario subsets are read-only.
    run = ProxyRun(1000, 1, 1, int(timestamps[1]), 0, int(timestamps[0]), 60_000, .05, 0, count - 1)
    market = ProxyMarket(.001, .01, .001, 5, 1, .0002)
    def build(directory=None):
        return build_mps_multicoin_data(
            values, timestamps, runs=[run, replace(run, first_valid_idx=3, last_valid_idx=count - 3)], markets=[market] * 2,
            include_hourly_ranges=hourly, spill_dir=directory,
            limit_order_fill_buffer_pct=fill_buffer,
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


@pytest.mark.parametrize("indices", [[3, 0, 2], [2], [1, 1, 0]])
def test_coin_subsets_stream_exact_order_without_full_private_copy(runtime, monkeypatch, indices):
    from optimization.gpu import residency

    values = np.arange(31 * 4 * 4, dtype=np.float64).reshape(31, 4, 4)
    # A strided source must preserve its selected row/coin/channel order as well.
    values = values[::2, :, ::2]
    values[1, 2, 0] = np.nan
    expected = np.take(values, indices, axis=1)
    original_take = np.take
    row_counts = []
    def bounded_take(array, *args, **kwargs):
        row_counts.append(len(array))
        assert len(array) <= 3
        return original_take(array, *args, **kwargs)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 3 * len(indices) * values.shape[2] * values.dtype.itemsize)
    monkeypatch.setattr(residency.np, "take", bounded_take)
    manager = CudaSuiteResidency()
    try:
        selected = manager.prepare_coin_subset(values, indices)
        assert isinstance(selected, np.memmap)
        assert not selected.flags.writeable
        assert selected.flags.c_contiguous
        np.testing.assert_array_equal(selected, expected)
        assert len(row_counts) > 1
        directory = Path(manager._directory.name)
    finally:
        manager.close()
    assert selected._mmap.closed
    assert not directory.exists()


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_failed_subset_stream_removes_partial_file_and_can_retry(runtime, monkeypatch, error):
    from optimization.gpu import residency

    values = np.ones((10, 3, 4))
    original_take = np.take
    def fail(*args, **kwargs):
        raise error("selection failed")
    manager = CudaSuiteResidency()
    try:
        monkeypatch.setattr(residency.np, "take", fail)
        with pytest.raises(error, match="selection failed"):
            manager.prepare_coin_subset(values, [2, 0])
        assert list(Path(manager._directory.name).iterdir()) == []
        monkeypatch.setattr(residency.np, "take", original_take)
        selected = manager.prepare_coin_subset(values, [2, 0])
        np.testing.assert_array_equal(selected, values[:, [2, 0]])
    finally:
        manager.close()


def test_spilled_price_packing_never_allocates_whole_history(runtime, monkeypatch):
    from optimization.gpu import model, residency

    torch, _ = runtime
    torch.float32, torch.int32, torch.int8 = np.float32, np.int32, np.int8
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda.is_available = lambda: True
    values = np.tile([101., 99., 100., 1.], (109, 3, 1))
    timestamps = np.arange(len(values), dtype=np.int64) * 60_000
    run = model.ProxyRun(1000, 0, 0, 0, 0, 0, 60_000, .05, 0, len(values) - 1)
    market = model.ProxyMarket(.001, .01, .001, 1, 1, .0002)
    original_pack = model._pack_multicoin_price_arrays
    row_counts = []
    def bounded_pack(chunk, *args):
        row_counts.append(len(chunk))
        assert len(chunk) <= 7
        return original_pack(chunk, *args)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 7 * (3 * 45 + 256))
    monkeypatch.setattr(model, "_pack_multicoin_price_arrays", bounded_pack)
    manager = CudaSuiteResidency()
    try:
        data = manager.prepare("market", lambda directory: model.build_mps_multicoin_data(
            values, timestamps, [run] * 3, [market] * 3, spill_dir=directory,
        ))
        assert len(row_counts) > 1
        for name in ("bars", "fill_ticks", "touch_ticks", "hour_log_ranges"):
            assert np.load(data[name]).shape[0] == len(values)
    finally:
        manager.close()


def test_late_chunk_tick_failure_removes_all_partial_market_files(runtime, monkeypatch):
    from optimization.gpu import model, residency

    torch, uploads = runtime
    torch.float32, torch.int32, torch.int8 = np.float32, np.int32, np.int8
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch.cuda.is_available = lambda: True
    values = np.tile([101., 99., 100., 1.], (29, 2, 1))
    values[-1, 0, 2] = 1e12
    timestamps = np.arange(len(values), dtype=np.int64) * 60_000
    run = model.ProxyRun(1000, 0, 0, 0, 0, 0, 60_000, .05, 0, len(values) - 1)
    market = model.ProxyMarket(.001, .01, .001, 1, 1, .0002)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 7 * (2 * 45 + 256))
    manager = CudaSuiteResidency()
    try:
        with pytest.raises(ValueError, match="touch ticks exceed"):
            manager.prepare("market", lambda directory: model.build_mps_multicoin_data(
                values, timestamps, [run] * 2, [market] * 2, spill_dir=directory,
            ))
        assert not uploads
        assert list(Path(manager._directory.name).iterdir()) == []
    finally:
        manager.close()
