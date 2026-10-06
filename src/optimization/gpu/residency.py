"""Run-local, disk-backed market packing with one active CUDA dataset."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import logging
import weakref

import numpy as np


PREPARATION_CHUNK_BYTES = 8 * 2**20


@contextmanager
def spill_array_file(path, shape, dtype):
    """Write a standard contiguous .npy array without mapping its output pages."""
    dtype = np.dtype(dtype)
    with path.open("wb") as handle:
        np.lib.format.write_array_header_1_0(handle, {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": tuple(shape),
        })
        expected_end = handle.tell() + int(np.prod(shape)) * dtype.itemsize
        yield handle
        if handle.tell() != expected_end:
            raise ValueError(f"Incomplete CUDA preparation array: {path.name}")


_current = ContextVar("cuda_suite_residency", default=None)


def current_cuda_residency():
    return _current.get()


def _cleanup_after_failure(cleanup, *args, **kwargs):
    try:
        cleanup(*args, **kwargs)
    except BaseException:
        logging.exception("CUDA suite cleanup failed while handling an earlier exception")


def check_cuda_invariant_memory(torch, invariant_bytes, *, working_set=None):
    free = int(torch.cuda.mem_get_info()[0])
    budget = free if working_set is None else working_set
    if invariant_bytes > int(budget * 0.45):
        raise MemoryError(
            f"CUDA multicoin invariant tensors would consume {invariant_bytes / 2**30:.2f} "
            f"GiB, above the 45% safety limit of {budget / 2**30:.2f} GiB "
            f"{'free VRAM' if working_set is None else 'initial free VRAM'}"
        )
    if invariant_bytes > free:
        raise MemoryError(
            f"CUDA multicoin invariant tensors require {invariant_bytes / 2**30:.2f} GiB "
            f"but only {free / 2**30:.2f} GiB VRAM is currently free"
        )


class CudaSuiteResidency:
    def __init__(self):
        self._directory = None
        self._entries = {}
        self._active = None
        self._active_owner = None
        self._working_set = None
        self._subset_arrays = []

    def _spill_directory(self):
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="passivbot-cuda-suite-")
        return Path(self._directory.name)

    def prepare_coin_subset(self, values, indices):
        """Keep raw scenario selections on disk instead of in private host RAM."""
        path = self._spill_directory() / f"subset-{len(self._subset_arrays)}.npy"
        shape = (len(values), len(indices), values.shape[2])
        row_bytes = max(1, int(np.prod(shape[1:])) * values.dtype.itemsize)
        chunk_rows = max(1, PREPARATION_CHUNK_BYTES // row_bytes)
        try:
            with spill_array_file(path, shape, values.dtype) as handle:
                for start in range(0, len(values), chunk_rows):
                    chunk = np.take(values[start:start + chunk_rows], indices, axis=1)
                    handle.write(memoryview(chunk).cast("B"))
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except BaseException:
            _cleanup_after_failure(path.unlink, missing_ok=True)
            raise
        self._subset_arrays.append(array)
        return array

    def prepare(self, key, builder):
        if key in self._entries:
            return self._entries[key]["data"]
        directory = self._spill_directory() / str(len(self._entries))
        directory.mkdir()
        try:
            data = builder(directory)
        except BaseException:
            _cleanup_after_failure(shutil.rmtree, directory)
            raise
        files = {name: value for name, value in data.items() if isinstance(value, Path)}
        self._entries[key] = {"data": data, "files": files, "owners": []}
        return data

    def owns(self, data):
        return any(entry["data"] is data for entry in self._entries.values())

    def register(self, proxy):
        entry = next(entry for entry in self._entries.values() if entry["data"] is proxy.data)
        entry["owners"].append(weakref.ref(proxy))

    @staticmethod
    def _clear_runners(entry):
        for owner in entry["owners"]:
            proxy = owner()
            if proxy is not None:
                proxy.runners.clear()
                proxy.fused_runner = None

    def _release(self, torch):
        if self._active is None:
            return
        entry = self._active
        try:
            torch.cuda.synchronize()
        finally:
            self._clear_runners(entry)
            # Keep dictionary identity for compatible suite grouping.
            entry["data"].update(entry["files"])
            self._active = None
            self._active_owner = None
            torch.cuda.empty_cache()

    def activate(self, proxy):
        import torch

        entry = next(entry for entry in self._entries.values() if entry["data"] is proxy.data)
        if self._active is not entry:
            self._release(torch)
            # CUDA driver/kernel workspace can remain resident after tensors are
            # released. It belongs to the reserved 55%, not a shrinking budget.
            if self._working_set is None:
                self._working_set = int(torch.cuda.mem_get_info()[0])
            check_cuda_invariant_memory(
                torch, entry["data"]["invariant_bytes"], working_set=self._working_set,
            )
            self._active = entry
            try:
                for name, path in entry["files"].items():
                    array = np.load(path, mmap_mode="c", allow_pickle=False)
                    entry["data"][name] = torch.as_tensor(array, device="cuda").contiguous()
                    del array
            except BaseException:
                _cleanup_after_failure(self._release, torch)
                raise
        elif self._active_owner is None or self._active_owner() is not proxy:
            # Incompatible proxies and screening representatives can share the
            # market entry. Keep its tensors, but retain only one owner's scratch.
            try:
                torch.cuda.synchronize()
            finally:
                self._clear_runners(entry)
                self._active_owner = None
                torch.cuda.empty_cache()
        if not proxy.runners and proxy.fused_runner is None:
            try:
                proxy._create_runners()
            except BaseException:
                _cleanup_after_failure(self._release, torch)
                raise
        self._active_owner = weakref.ref(proxy)

    def close(self):
        try:
            if self._active is not None:
                import torch
                self._release(torch)
        finally:
            self._entries.clear()
            for array in self._subset_arrays:
                array._mmap.close()
            self._subset_arrays.clear()
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None


@contextmanager
def cuda_residency_scope():
    """Own run-local packing/residency on the calling execution thread."""
    residency = CudaSuiteResidency()
    token = _current.set(residency)
    try:
        try:
            yield residency
        except BaseException:
            _cleanup_after_failure(residency.close)
            raise
        else:
            residency.close()
    finally:
        _current.reset(token)


def cuda_suite_residency_scope(func):
    @wraps(func)
    def scoped(*args, **kwargs):
        with cuda_residency_scope():
            return func(*args, **kwargs)
    return scoped
