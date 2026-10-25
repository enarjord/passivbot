"""Metal immutable uploads bound source-page residency and retain queued aliases."""

import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from optimization.gpu import residency


class _Tensor:
    def __init__(self, values, state, device, storage_bytes=None):
        self.values, self.state, self.device = values, state, device
        self.dtype = values.dtype
        self.storage_bytes = values.nbytes if storage_bytes is None else storage_bytes

    def contiguous(self):
        assert self.values.flags.c_contiguous
        return self

    def reshape(self, *shape):
        return _Tensor(self.values.reshape(*shape), self.state, self.device, self.storage_bytes)

    def __getitem__(self, indices):
        return _Tensor(self.values[indices], self.state, self.device, self.storage_bytes)

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.storage_bytes)

    def copy_(self, source):
        assert self.device == "mps" and source.device == "cpu"
        assert source.values.flags.c_contiguous
        self.state.source_storages.append(source.untyped_storage().nbytes())
        self.state.copies.append(source.values.nbytes)
        self.state.queue.append((self.values, source.values))
        if self.state.copy_failure is not None:
            failure, self.state.copy_failure = self.state.copy_failure, None
            raise failure  # Simulate submission followed by a producer failure.
        return self


class _Mapping:
    def __init__(self, real, state):
        self.real, self.state = real, state

    @property
    def closed(self):
        return self.real.closed

    def __len__(self):
        return len(self.real)

    def madvise(self, option, start, length):
        assert not self.state.queue
        self.state.aliases_at_advice.append(sum(reference() is not None
                                                for reference in self.state.alias_refs))
        assert start % residency.mmap.PAGESIZE == 0
        assert length % residency.mmap.PAGESIZE == 0
        self.state.advice.append((start, length))
        if self.state.advice_failure:
            raise OSError("advice unavailable")

    def close(self):
        assert not self.state.queue, "A queued source must not be unmapped"
        self.state.events.append("close")
        if self.state.close_failure is not None:
            failure, self.state.close_failure = self.state.close_failure, None
            raise failure
        self.real.close()


@pytest.fixture
def uploads(monkeypatch):
    assert not residency._mps_upload_quarantine
    state = SimpleNamespace(queue=[], copies=[], allocations=[], mappings=[],
        aliases=[], alias_refs=[], aliases_at_advice=[], source_storages=[],
        direct=[], events=[], advice=[], sync_failures=[],
        copy_failure=None, close_failure=None, advice_failure=False)
    original_load = np.load

    def load(*args, **kwargs):
        array = original_load(*args, **kwargs)
        assert kwargs["mmap_mode"] == "c"
        array._mmap = _Mapping(array._mmap, state)
        state.mappings.append(array)
        return array

    def alias(array):
        state.aliases.append(array)
        assert array.size == 0 or isinstance(array, np.memmap)
        tensor = _Tensor(array, state, "cpu")
        state.alias_refs.append(weakref.ref(tensor))
        return tensor

    def empty(shape, *, dtype, device):
        assert device == "mps"
        state.allocations.append((shape, dtype))
        return _Tensor(np.empty(shape, dtype=dtype), state, device)

    def direct(array, *, device):
        state.direct.append(array.nbytes)
        output = _Tensor(np.empty_like(array), state, device)
        state.queue.append((output.values, array))
        return output

    def synchronize():
        state.events.append("sync")
        if state.sync_failures:
            failure = state.sync_failures.pop(0)
            if failure is not None:
                raise failure
        for destination, source in state.queue:
            mapping = getattr(source, "_mmap", None)
            assert mapping is None or not mapping.closed
            np.copyto(destination, source)
        state.queue.clear()

    torch = SimpleNamespace(from_numpy=alias, empty=empty, as_tensor=direct,
        mps=SimpleNamespace(synchronize=synchronize, empty_cache=lambda: None))
    state.torch = torch
    monkeypatch.setattr(residency.np, "load", load)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 4096)
    monkeypatch.setitem(sys.modules, "torch", torch)
    yield state
    state.sync_failures.clear()
    residency._synchronize_mps_uploads(torch)
    assert not residency._mps_upload_quarantine
    for array in state.mappings:
        if not array._mmap.closed:
            array._mmap.close()


def _input(tmp_path, shape=(53, 257), dtype=np.int32):
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    path = tmp_path / "bars.npy"
    np.save(path, values, allow_pickle=False)
    return path, values


def test_large_upload_allocates_one_destination_and_bounds_alias_copies(tmp_path, uploads):
    path, values = _input(tmp_path)
    actual = residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    np.testing.assert_array_equal(actual.values, values)
    assert len(uploads.allocations) == 1
    assert len(uploads.aliases) == len(uploads.copies) + 1
    assert uploads.aliases[0].nbytes == 0  # Dtype inference owns no history storage.
    assert max(array.nbytes for array in uploads.aliases) <= 4096
    assert max(uploads.source_storages) <= 4096
    assert all(reference() is None for reference in uploads.alias_refs)
    assert all(count == 0 for count in uploads.aliases_at_advice)
    assert not uploads.direct and len(uploads.copies) > 1
    assert max(uploads.copies) <= 4096
    assert sum(uploads.copies) == values.nbytes
    assert all(array._mmap.closed for array in uploads.mappings)
    assert not uploads.queue and not residency._mps_upload_quarantine
    assert uploads.advice  # Complete pages are discarded after their copy finishes.
    assert uploads.events[-1] == "close"


def test_one_oversized_row_is_split_without_exceeding_the_copy_budget(tmp_path, uploads):
    path, values = _input(tmp_path, shape=(2, 4097))
    actual = residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    np.testing.assert_array_equal(actual.values, values)
    assert len(uploads.allocations) == 1 and max(uploads.copies) <= 4096
    assert max(array.nbytes for array in uploads.aliases) <= 4096
    assert max(uploads.source_storages) <= 4096


def test_real_cpu_storage_is_bounded_before_any_device_upload(tmp_path, monkeypatch):
    """A small tensor slice of a whole-history CPU storage must fail this gate."""
    torch = pytest.importorskip("torch")
    path, values = _input(tmp_path)
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 4096)
    aliases, source_storages, destinations = [], [], []

    class Destination:
        def __init__(self, array):
            self.array = array

        def reshape(self, *shape):
            return Destination(self.array.reshape(*shape))

        def __getitem__(self, indices):
            return Destination(self.array[indices])

        def copy_(self, source):
            source_storages.append(source.untyped_storage().nbytes())
            np.copyto(self.array, source.numpy())
            return self

    def alias(array):
        tensor = torch.from_numpy(array)
        aliases.append(tensor.untyped_storage().nbytes())
        return tensor

    def allocate(shape, *, dtype, device):
        assert device == "mps"
        assert dtype == torch.int32
        result = Destination(np.empty(shape, dtype=np.int32))
        destinations.append(result)
        return result

    # Actual Torch CPU storage metadata is checked; the destination and wait are
    # CPU fakes, so this regression does not execute any GPU work.
    fake = SimpleNamespace(from_numpy=alias, empty=allocate,
                           mps=SimpleNamespace(synchronize=lambda: None))
    result = residency.MetalSuiteResidency()._upload_file(fake, path)
    assert len(destinations) == 1 and aliases[0] == 0
    assert len(source_storages) > 1
    assert max(aliases) <= 4096 and max(source_storages) <= 4096
    np.testing.assert_array_equal(result.array, values)


def test_small_upload_keeps_direct_fastpath_but_waits_before_unmapping(tmp_path, uploads):
    path, values = _input(tmp_path, shape=(19,))
    actual = residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    np.testing.assert_array_equal(actual.values, values)
    assert uploads.direct == [values.nbytes]
    assert not uploads.aliases and not uploads.allocations and not uploads.copies
    assert uploads.events == ["sync", "close"]
    assert uploads.mappings[0]._mmap.closed


@pytest.mark.parametrize("failure", [RuntimeError("copy failed"), KeyboardInterrupt("interrupted upload")])
def test_failed_copy_drains_before_close_and_preserves_original(tmp_path, uploads, failure):
    path, _ = _input(tmp_path)
    uploads.copy_failure = failure
    with pytest.raises(type(failure)) as caught:
        residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    assert caught.value is failure
    assert uploads.events == ["sync", "close"]
    assert uploads.mappings[0]._mmap.closed
    assert not residency._mps_upload_quarantine


def test_failed_cleanup_sync_quarantines_until_successful_service_close(tmp_path, uploads, caplog):
    path, _ = _input(tmp_path)
    original = RuntimeError("copy failed")
    uploads.copy_failure = original
    uploads.sync_failures = [RuntimeError("queue status unknown")]
    manager = residency.MetalSuiteResidency()
    with pytest.raises(RuntimeError) as caught:
        manager._upload_file(uploads.torch, path)
    assert caught.value is original
    assert len(residency._mps_upload_quarantine) == 1
    assert not uploads.mappings[0]._mmap.closed and uploads.queue
    assert "cleanup synchronization failed" in caplog.text
    manager.close()
    assert not residency._mps_upload_quarantine and not uploads.queue
    assert uploads.mappings[0]._mmap.closed


def test_cleanup_close_error_does_not_replace_the_original_failure(tmp_path, uploads, caplog):
    path, _ = _input(tmp_path)
    original = RuntimeError("copy failed")
    uploads.copy_failure = original
    uploads.close_failure = OSError("mapping close failed")
    with pytest.raises(RuntimeError) as caught:
        residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    assert caught.value is original
    assert "mapping close failed" in caplog.text
    assert not uploads.queue


def test_quarantine_close_failure_keeps_ownership_for_another_confirmed_wait(tmp_path, uploads, caplog):
    path, _ = _input(tmp_path)
    uploads.copy_failure = RuntimeError("copy failed")
    uploads.sync_failures = [RuntimeError("drain failed")]
    with pytest.raises(RuntimeError, match="copy failed"):
        residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    uploads.close_failure = OSError("close still unavailable")
    residency._synchronize_mps_uploads(uploads.torch)
    assert len(residency._mps_upload_quarantine) == 1
    assert not uploads.mappings[0]._mmap.closed
    residency._synchronize_mps_uploads(uploads.torch)
    assert not residency._mps_upload_quarantine and uploads.mappings[0]._mmap.closed


def test_optional_page_advice_failure_does_not_change_the_upload(tmp_path, uploads, caplog):
    path, values = _input(tmp_path)
    uploads.advice_failure = True
    with caplog.at_level("DEBUG"):
        actual = residency.MetalSuiteResidency()._upload_file(uploads.torch, path)
    np.testing.assert_array_equal(actual.values, values)
    assert "Could not discard completed Metal upload pages" in caplog.text
    assert uploads.mappings[0]._mmap.closed


def test_cuda_upload_keeps_whole_file_path_without_new_waits(tmp_path, monkeypatch):
    path, values = _input(tmp_path)
    calls = []

    def upload(array, *, device):
        assert device == "cuda"
        calls.append(array.shape)
        return SimpleNamespace(contiguous=lambda: np.array(array, copy=True))

    actual = residency.CudaSuiteResidency()._upload_file(SimpleNamespace(as_tensor=upload), path)
    np.testing.assert_array_equal(actual, values)
    assert calls == [values.shape]


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_actual_metal_chunked_upload_is_bit_identical(tmp_path, monkeypatch, dtype):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal required")
    monkeypatch.setattr(residency, "PREPARATION_CHUNK_BYTES", 128)
    bits = np.resize(np.array([0, 0x80000000, 0x3F800000, 0x7F800000,
                              0xFF800000, 0x7FC00123], dtype=np.uint32), 291)
    values = bits.view(dtype).reshape(97, 3)
    path = tmp_path / "exact.npy"
    np.save(path, values, allow_pickle=False)
    from_numpy = torch.from_numpy
    storage_bytes = []

    def bounded_alias(array):
        tensor = from_numpy(array)
        storage_bytes.append(tensor.untyped_storage().nbytes())
        return tensor

    monkeypatch.setattr(torch, "from_numpy", bounded_alias)
    actual = residency.MetalSuiteResidency()._upload_file(torch, path)
    torch.mps.synchronize()
    assert actual.is_contiguous() and tuple(actual.shape) == values.shape
    assert storage_bytes[0] == 0 and max(storage_bytes) <= 128
    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint32), values.view(np.uint32))
