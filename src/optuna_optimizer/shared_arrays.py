"""Shared memory utilities for multiprocess optimization.

Provides SharedArrayManager (parent process) and attachment utilities (workers)
for sharing NumPy arrays between processes via multiprocessing.shared_memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    """Descriptor for attaching to a shared memory array.

    Carries the name, shape, and dtype needed to reconstruct a numpy view
    of the shared memory buffer.
    """

    name: str
    shape: tuple[int, ...]
    dtype: str


class SharedArrayAttachment:
    """RAII wrapper for attached shared memory.

    Attaches to an existing shared memory block and provides a numpy array view.
    Call close() when done to release the attachment.
    """

    def __init__(self, spec: SharedArraySpec):
        self.spec = spec
        self._shm = shared_memory.SharedMemory(name=spec.name)
        self.array: np.ndarray = np.ndarray(
            spec.shape, dtype=np.dtype(spec.dtype), buffer=self._shm.buf
        )

    def close(self) -> None:
        """Close the shared memory attachment."""
        self._shm.close()


class SharedArrayManager:
    """Manages shared memory allocations owned by the parent process.

    Creates shared memory blocks, copies data into them, and provides specs
    that workers can use to attach to the same memory.
    """

    def __init__(self) -> None:
        self._owned_blocks: dict[str, shared_memory.SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}

    def create_from(self, array: np.ndarray) -> tuple[SharedArraySpec, np.ndarray]:
        """Allocate shared memory sized to `array`, copy data, return spec and view."""
        contiguous = np.ascontiguousarray(array)
        shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
        view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
        np.copyto(view, contiguous)
        spec = SharedArraySpec(name=shm.name, shape=contiguous.shape, dtype=contiguous.dtype.str)
        self._owned_blocks[spec.name] = shm
        self._arrays[spec.name] = view
        return spec, view

    def view(self, spec: SharedArraySpec) -> np.ndarray:
        """Return the NumPy view for a spec owned by this manager."""
        return self._arrays[spec.name]

    def cleanup(self, specs: Iterable[SharedArraySpec] | None = None) -> None:
        """Close and unlink all owned shared memory segments."""
        to_cleanup = (
            specs
            if specs is not None
            else [
                SharedArraySpec(name, array.shape, array.dtype.str)
                for name, array in self._arrays.items()
            ]
        )
        names = {spec.name for spec in to_cleanup}
        for name in list(self._owned_blocks.keys()):
            if name in names:
                shm = self._owned_blocks.pop(name)
                shm.close()
                shm.unlink()
                self._arrays.pop(name, None)


def attach_shared_array(spec: SharedArraySpec) -> SharedArrayAttachment:
    """Attach to existing shared memory described by spec."""
    return SharedArrayAttachment(spec)
