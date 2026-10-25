import gzip
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import shared_arrays
from hlcvs_manifest import (
    HlcvsManifestError,
    hash_json_value,
    hash_logical_array,
    load_numpy_artifact,
    numpy_artifact_loader,
    verify_hlcvs_manifest,
)
from shared_arrays import SharedArrayManager, SharedArraySpec, attach_shared_array


def _save(path, values):
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
    else:
        np.save(path, values, allow_pickle=False)


@pytest.mark.parametrize("compressed", [False, True])
def test_file_loader_maps_read_only_and_adopts_slices_without_shared_memory(
    tmp_path, monkeypatch, compressed,
):
    values = np.arange(2000, dtype=np.float64).reshape(100, 5, 4)
    path = tmp_path / ("hlcvs.npy.gz" if compressed else "hlcvs.npy")
    _save(path, values)
    original_load = np.load
    def mapped_load(*args, **kwargs):
        assert kwargs["mmap_mode"] == "r"
        assert kwargs["allow_pickle"] is False
        return original_load(*args, **kwargs)
    monkeypatch.setattr(shared_arrays.np, "load", mapped_load)
    monkeypatch.setattr(shared_arrays, "FILE_ARRAY_CHUNK_BYTES", 1024)
    original_copy = shared_arrays.shutil.copyfileobj
    copied = []
    def copy(source, destination, *, length):
        assert length == 1024
        copied.append(length)
        return original_copy(source, destination, length=length)
    monkeypatch.setattr(shared_arrays.shutil, "copyfileobj", copy)
    def forbidden(*args, **kwargs):
        pytest.fail("file transport must not allocate shared memory")
    monkeypatch.setattr(shared_arrays.shared_memory, "SharedMemory", forbidden)
    manager = SharedArrayManager()
    try:
        array = manager.load_numpy_artifact(path)
        mapped_path = Path(array.filename)
        assert isinstance(array, np.memmap) and not array.flags.writeable
        spec, view = manager.create_file_from(np.asarray(array)[7:29])
        assert spec.storage == "file"
        assert spec.offset == array.offset + 7 * values[0].nbytes
        assert np.shares_memory(view, array)
        np.testing.assert_array_equal(manager.view(spec), values[7:29])
        assert not manager.view(spec).flags.writeable
        # Descriptors can cross a spawned-process/pickle boundary without payload copies.
        attachment = attach_shared_array(pickle.loads(pickle.dumps(spec)))
        try:
            np.testing.assert_array_equal(attachment.array, values[7:29])
            with pytest.raises(ValueError, match="read-only"):
                attachment.array[0, 0, 0] = 1
        finally:
            attachment.close()
            attachment.close()
        assert copied == ([1024] if compressed else [])
    finally:
        manager.cleanup()
        manager.cleanup()
    assert path.exists()
    assert mapped_path.exists() is (not compressed)


def test_file_copy_bounds_strided_input_and_does_not_adopt_materializer_files(tmp_path, monkeypatch):
    source_path = tmp_path / "hlcvs.dat"
    source = np.memmap(source_path, mode="w+", shape=(12, 3, 4), dtype=np.float64)
    source[:] = np.arange(source.size).reshape(source.shape)
    manager = SharedArrayManager()
    monkeypatch.setattr(shared_arrays, "FILE_ARRAY_CHUNK_BYTES", 32)
    expected = np.asarray(source)[:, ::2].copy()
    try:
        spec, view = manager.create_file_from(source[:, ::2])
        assert Path(spec.name) != source_path
        assert view.flags.c_contiguous and not view.flags.writeable
        source._mmap.close()
        source_path.unlink()
        attachment = attach_shared_array(spec)
        try:
            np.testing.assert_array_equal(attachment.array, expected)
        finally:
            attachment.close()
        owned_path = Path(spec.name)
    finally:
        manager.cleanup()
    assert not owned_path.exists()


@pytest.mark.parametrize("compressed", [False, True])
def test_file_preflight_prevents_payload_loading_and_allocation(tmp_path, monkeypatch, compressed):
    path = tmp_path / ("hlcvs.npy.gz" if compressed else "hlcvs.npy")
    values = np.ones((100, 3, 4), dtype=np.float64)
    _save(path, values)
    inspected = []
    def preflight(actual_path, shape, dtype):
        inspected.append((actual_path, shape, dtype))
        raise MemoryError("workload exceeds budget")
    def forbidden(*args, **kwargs):
        pytest.fail("preflight must run before reading/mapping the payload")
    monkeypatch.setattr(shared_arrays.np, "load", forbidden)
    monkeypatch.setattr(shared_arrays.shutil, "copyfileobj", forbidden)
    manager = SharedArrayManager(file_preflight=preflight)
    with pytest.raises(MemoryError, match="budget"):
        manager.load_numpy_artifact(path)
    assert inspected == [(path, values.shape, values.dtype)]
    assert manager._file_directory is None
    assert not manager._file_arrays
    manager.cleanup()


def test_truncated_gzip_does_not_leave_a_partial_owned_artifact(tmp_path):
    path = tmp_path / "hlcvs.npy.gz"
    _save(path, np.arange(1000, dtype=np.float64))
    path.write_bytes(path.read_bytes()[:-8])
    manager = SharedArrayManager()
    try:
        with pytest.raises(EOFError):
            manager.load_numpy_artifact(path)
        assert not manager._file_arrays
        assert not list(Path(manager._file_directory.name).iterdir())
    finally:
        manager.cleanup()


def test_file_replacement_during_preflight_is_rejected_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "hlcvs.npy"
    _save(path, np.ones((3, 1, 4)))
    def preflight(actual_path, shape, dtype):
        replacement = tmp_path / "replacement.npy"
        _save(replacement, np.ones((30, 1, 4)))
        replacement.replace(actual_path)
    def forbidden(*args, **kwargs):
        pytest.fail("changed source must fail before payload loading")
    monkeypatch.setattr(shared_arrays.np, "load", forbidden)
    manager = SharedArrayManager(file_preflight=preflight)
    with pytest.raises(ValueError, match="changed since preparation"):
        manager.load_numpy_artifact(path)
    assert manager._file_directory is None
    manager.cleanup()


def test_file_copy_failure_removes_partial_artifact(tmp_path, monkeypatch):
    manager = SharedArrayManager()
    def broken(*args, **kwargs):
        raise ValueError("cannot map completed artifact")
    monkeypatch.setattr(shared_arrays.np, "load", broken)
    try:
        with pytest.raises(ValueError, match="cannot map"):
            manager.create_file_from(np.arange(20))
        assert not manager._file_arrays
        assert not list(Path(manager._file_directory.name).iterdir())
    finally:
        manager.cleanup()


@pytest.mark.parametrize("replacement", ["atomic", "same_inode"])
def test_borrowed_file_changes_fail_closed_on_attachment(tmp_path, replacement):
    path = tmp_path / "hlcvs.npy"
    original = np.arange(20, dtype=np.float64)
    _save(path, original)
    manager = SharedArrayManager()
    try:
        array = manager.load_numpy_artifact(path)
        spec, _view = manager.create_file_from(array)
        before = path.stat()
        if replacement == "atomic":
            changed = tmp_path / "replacement.npy"
            _save(changed, original + 100)
            changed.replace(path)
            # The parent's mapping remains pinned to its verified original inode.
            np.testing.assert_array_equal(array, original)
        else:
            _save(path, original + 100)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(ValueError, match="changed since preparation"):
            attach_shared_array(spec)
        with pytest.raises(ValueError, match="changed since preparation"):
            manager.load_numpy_artifact(path)
    finally:
        manager.cleanup()
    assert path.exists()


def test_default_shared_memory_transport_is_unchanged():
    manager = SharedArrayManager()
    try:
        spec, _view = manager.create_from(np.arange(5, dtype=np.float64))
        assert spec.storage == "shared_memory" and spec.offset == 0
        attached = attach_shared_array(spec)
        try:
            np.testing.assert_array_equal(attached.array, np.arange(5))
        finally:
            attached.close()
    finally:
        manager.cleanup()


def test_file_backed_manager_mode_routes_existing_create_calls_to_files(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("file-backed manager must not allocate shared memory")
    monkeypatch.setattr(shared_arrays.shared_memory, "SharedMemory", forbidden)
    manager = SharedArrayManager(file_backed=True)
    try:
        spec, view = manager.create_from(np.arange(5, dtype=np.float64))
        assert spec.storage == "file" and not view.flags.writeable
        np.testing.assert_array_equal(manager.view(spec), np.arange(5))
    finally:
        manager.cleanup()


def test_loader_context_is_nested_and_resets_after_failure(tmp_path):
    path = tmp_path / "hlcvs.npy"
    _save(path, np.arange(4))
    manager = SharedArrayManager()
    try:
        with numpy_artifact_loader(manager.load_numpy_artifact):
            assert isinstance(load_numpy_artifact(path), np.memmap)
            def refused(actual_path, shape, dtype):
                assert (actual_path, shape, dtype) == (path, (4,), np.dtype("int64"))
                raise MemoryError("preflight refusal")
            with pytest.raises(MemoryError):
                with numpy_artifact_loader(lambda _path: pytest.fail("loader ran"), preflight=refused):
                    load_numpy_artifact(path)
            assert isinstance(load_numpy_artifact(path), np.memmap)
        assert not isinstance(load_numpy_artifact(path), np.memmap)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("compressed", [False, True])
def test_file_loading_preserves_manifest_verification_and_rejects_corruption(tmp_path, compressed):
    arrays = {
        "hlcvs": np.arange(24, dtype=np.float64).reshape(6, 1, 4),
        "timestamps": np.arange(6, dtype=np.int64) * 60_000,
        "btc_usd_prices": np.ones(6, dtype=np.float64),
    }
    files = {}
    for name, array in arrays.items():
        path = tmp_path / (f"{name}.npy.gz" if compressed else f"{name}.npy")
        _save(path, array)
        files[name] = dict(path=path.name, sha256=hash_logical_array(array),
                           shape=list(array.shape), dtype=str(array.dtype))
    for name, value in (("coins", ["A"]), ("market_specific_settings", {"A": {}})):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        files[name] = dict(path=path.name, sha256=hash_json_value(value))
    manifest = dict(schema_version=1, materialization_schema_version=1, files=files)
    manager = SharedArrayManager()
    try:
        with numpy_artifact_loader(manager.load_numpy_artifact):
            loaded = {}
            verify_hlcvs_manifest(tmp_path, manifest, out_arrays=loaded)
        assert all(isinstance(value, np.memmap) for value in loaded.values())
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)
        manifest["files"]["hlcvs"]["sha256"] = "invalid"
        with numpy_artifact_loader(manager.load_numpy_artifact):
            with pytest.raises(HlcvsManifestError, match="hash mismatch"):
                verify_hlcvs_manifest(tmp_path, manifest)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("kwargs", [{"storage": "unknown"}, {"offset": -1},
                                    {"offset": True}, {"offset": 8}])
def test_invalid_storage_descriptor_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SharedArraySpec("array", (5,), "<f8", **kwargs)
