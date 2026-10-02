"""Run-local, disk-backed market packing with one active CUDA dataset."""

from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import weakref

import numpy as np


_current = ContextVar("cuda_suite_residency", default=None)


def current_cuda_residency():
    return _current.get()


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
        self._working_set = None

    def prepare(self, key, builder):
        if key in self._entries:
            return self._entries[key]["data"]
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="passivbot-cuda-suite-")
        directory = Path(self._directory.name) / str(len(self._entries))
        directory.mkdir()
        try:
            data = builder(directory)
        except BaseException:
            shutil.rmtree(directory)
            raise
        files = {name: value for name, value in data.items() if isinstance(value, Path)}
        self._entries[key] = {"data": data, "files": files, "owners": []}
        return data

    def owns(self, data):
        return any(entry["data"] is data for entry in self._entries.values())

    def register(self, proxy):
        entry = next(entry for entry in self._entries.values() if entry["data"] is proxy.data)
        entry["owners"].append(weakref.ref(proxy))

    def _release(self, torch):
        if self._active is None:
            return
        entry = self._active
        try:
            torch.cuda.synchronize()
        finally:
            for owner in entry["owners"]:
                proxy = owner()
                if proxy is not None:
                    proxy.runners.clear()
                    proxy.fused_runner = None
            # Keep dictionary identity for compatible suite grouping.
            entry["data"].update(entry["files"])
            self._active = None
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
                self._release(torch)
                raise
        if not proxy.runners and proxy.fused_runner is None:
            try:
                proxy._create_runners()
            except BaseException:
                self._release(torch)
                raise

    def close(self):
        try:
            if self._active is not None:
                import torch
                self._release(torch)
        finally:
            self._entries.clear()
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None


def cuda_suite_residency_scope(func):
    @wraps(func)
    def scoped(*args, **kwargs):
        residency = CudaSuiteResidency()
        token = _current.set(residency)
        try:
            return func(*args, **kwargs)
        finally:
            try:
                residency.close()
            finally:
                _current.reset(token)
    return scoped
