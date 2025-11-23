"""
Utilities for sharing NumPy arrays between processes via multiprocessing.shared_memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    """
    Lightweight descriptor used to attach to a shared-memory backed NumPy array.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str


class SharedArrayAttachment:
    """
    RAII wrapper for an attached shared memory block.
    """

    def __init__(self, spec: SharedArraySpec):
        self.spec = spec
        self._shm = shared_memory.SharedMemory(name=spec.name)
        self.array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=self._shm.buf)

    def close(self) -> None:
        self._shm.close()


class SharedArrayManager:
    """
    Manages shared memory allocations owned by the parent process.
    """

    def __init__(self) -> None:
        self._owned_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

    def create_from(self, array: np.ndarray) -> Tuple[SharedArraySpec, np.ndarray]:
        """
        Allocate shared memory sized to `array`, copy the data into it, and
        return both the descriptor and a NumPy view backed by the shared segment.
        """
        contiguous = np.ascontiguousarray(array)
        shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
        view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
        np.copyto(view, contiguous)
        spec = SharedArraySpec(name=shm.name, shape=contiguous.shape, dtype=contiguous.dtype.str)
        self._owned_blocks[spec.name] = shm
        self._arrays[spec.name] = view
        return spec, view

    def view(self, spec: SharedArraySpec) -> np.ndarray:
        """
        Return the NumPy view for a spec owned by this manager.
        """
        return self._arrays[spec.name]

    def cleanup(self, specs: Iterable[SharedArraySpec] | None = None) -> None:
        """
        Close and unlink all owned shared memory segments. Optionally limit to a subset.
        """
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
    """
    Attach to an existing shared-memory backed array described by `spec`.
    """
    return SharedArrayAttachment(spec)
