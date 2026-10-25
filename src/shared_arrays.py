"""
Utilities for borrowing NumPy arrays from shared memory or read-only files.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from io import BytesIO
import logging
import mmap
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, Tuple

import numpy as np


FILE_ARRAY_CHUNK_BYTES = 8 * 2**20
_NPY_HEADER_MAX_BYTES = 10_000


def _file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _require_file_identity(actual, expected):
    if expected is not None and actual != expected:
        raise ValueError("file-backed array changed since preparation")


def numpy_artifact_header(path):
    """Read only an NPY header, including for a compressed artifact."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version not in {(1, 0), (2, 0), (3, 0)}:
            raise ValueError(f"unsupported NPY version: {version!r}")
        length_size = 2 if version == (1, 0) else 4
        length_bytes = handle.read(length_size)
        if len(length_bytes) != length_size:
            raise ValueError("incomplete NPY header length")
        header_size = int.from_bytes(length_bytes, "little")
        if header_size > _NPY_HEADER_MAX_BYTES:
            raise ValueError(f"NPY header exceeds {_NPY_HEADER_MAX_BYTES} bytes")
        header = handle.read(header_size)
        if len(header) != header_size:
            raise ValueError("incomplete NPY header")
        if version == (3, 0):
            # V3 differs from V2 only in its UTF-8 header encoding. Escape
            # non-ASCII characters inside its Python literal for the public
            # V2 reader, retaining Unicode field names without private APIs.
            header = header.decode("utf-8").encode("ascii", "backslashreplace")
            length_bytes = len(header).to_bytes(4, "little")
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        # ASCII escaping expands a UTF-8 header by at most three times.
        limit = _NPY_HEADER_MAX_BYTES * (3 if version == (3, 0) else 1)
        shape, fortran_order, dtype = reader(BytesIO(length_bytes + header), max_header_size=limit)
    return tuple(shape), np.dtype(dtype), fortran_order


@dataclass(frozen=True)
class SharedArraySpec:
    """
    Lightweight descriptor for shared memory or a read-only contiguous file view.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str
    storage: str = "shared_memory"
    offset: int = 0
    file_identity: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        if self.storage not in {"shared_memory", "file"}:
            raise ValueError(f"unsupported array storage: {self.storage!r}")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("array file offset must be a nonnegative integer")
        if self.storage == "shared_memory" and self.offset:
            raise ValueError("shared-memory arrays cannot have a file offset")
        if self.file_identity is not None:
            if (self.storage != "file" or not isinstance(self.file_identity, tuple)
                    or len(self.file_identity) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in self.file_identity)):
                raise ValueError("file identity must contain device, inode, size and modification time")


class SharedArrayAttachment:
    """
    RAII wrapper for an attached shared memory block or read-only file mapping.
    """

    def __init__(self, spec: SharedArraySpec):
        self.spec = spec
        self._shm = None
        self._mapping = None
        if spec.storage == "file":
            try:
                with open(spec.name, "rb") as handle:
                    _require_file_identity(_file_identity(os.fstat(handle.fileno())), spec.file_identity)
                    self._mapping = np.memmap(
                        handle, mode="r", shape=spec.shape, dtype=np.dtype(spec.dtype),
                        offset=spec.offset, order="C",
                    )
                    _require_file_identity(_file_identity(os.fstat(handle.fileno())), spec.file_identity)
            except BaseException:
                if self._mapping is not None:
                    self._mapping._mmap.close()
                raise
            self.array = self._mapping
        else:
            self._shm = shared_memory.SharedMemory(name=spec.name)
            try:
                self.array = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=self._shm.buf)
            except BaseException:
                self._shm.close()
                raise

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping._mmap.close()
            self._mapping = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None


class SharedArrayManager:
    """
    Own parent allocations and file mappings until all borrowers have closed.
    """

    def __init__(self, *, file_backed=False, file_preflight=None) -> None:
        self._owned_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        self._file_arrays: Dict[str, np.memmap] = {}
        self._file_identities = {}
        self._file_directory = None
        self._file_preflight = file_preflight
        self._file_backed = file_backed
        self._next_file_id = 0

    @property
    def file_backed(self):
        return self._file_backed

    def discard_file_pages(self):
        """Hint that prepared read-only parent mappings may be reclaimed.

        The GPU owner calls this after packing, while the parent retains file
        ownership. Views, descriptors and subsequent reads stay valid.
        """
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if advice is None:
            return 0
        advised = 0
        for array in self._file_arrays.values():
            if array.mode != "r" or array.flags.writeable:
                continue
            advise = getattr(array._mmap, "madvise", None)
            if callable(advise):
                try:
                    advise(advice)
                    advised += 1
                except OSError:
                    logging.debug("Could not discard prepared input file pages", exc_info=True)
        return advised

    def _file_root(self):
        if self._file_directory is None:
            self._file_directory = TemporaryDirectory(prefix="passivbot-array-")
        return Path(self._file_directory.name)

    def _new_file_path(self, prefix):
        path = self._file_root() / f"{prefix}-{self._next_file_id}.npy"
        self._next_file_id += 1
        return path

    def load_numpy_artifact(self, path):
        """Map a plain NPY, or stream a gzip NPY into owned run-local storage."""
        path = Path(path)
        source_identity = _file_identity(path.stat())
        shape, dtype, _fortran_order = numpy_artifact_header(path)
        _require_file_identity(_file_identity(path.stat()), source_identity)
        if dtype.hasobject:
            raise ValueError("Object arrays cannot be loaded with allow_pickle=False")
        if self._file_preflight is not None:
            self._file_preflight(path, shape, dtype)
        _require_file_identity(_file_identity(path.stat()), source_identity)
        if path.suffix != ".gz" and str(path.resolve()) in self._file_arrays:
            name = str(path.resolve())
            _require_file_identity(_file_identity(path.stat()), self._file_identities[name])
            return self._file_arrays[name]
        owned_path = None
        array = None
        try:
            if path.suffix == ".gz":
                owned_path = self._new_file_path("loaded")
                with gzip.open(path, "rb") as source, owned_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=FILE_ARRAY_CHUNK_BYTES)
                _require_file_identity(_file_identity(path.stat()), source_identity)
                path = owned_path
            identity = _file_identity(path.stat())
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            _require_file_identity(_file_identity(path.stat()), identity)
            name = str(path.resolve())
            self._file_arrays[name] = array
            self._file_identities[name] = identity
            return array
        except BaseException:
            if array is not None:
                array._mmap.close()
            if owned_path is not None:
                owned_path.unlink(missing_ok=True)
            raise

    def create_file_from(self, array: np.ndarray) -> Tuple[SharedArraySpec, np.ndarray]:
        """Borrow an owned/loaded NPY mapping, or copy in bounded chunks to an owned file.

        Raw materializer .dat files are copied so their independent owner may
        release its run without invalidating this manager's descriptors.
        """
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise ValueError("file-backed arrays must not contain Python objects")
        if array.flags.c_contiguous:
            for name, mapping in self._file_arrays.items():
                if np.shares_memory(array, mapping):
                    offset = (int(mapping.offset) + array.ctypes.data - mapping.ctypes.data)
                    spec = SharedArraySpec(name, array.shape, array.dtype.str, "file", offset,
                                           self._file_identities[name])
                    return spec, array
        path = self._new_file_path("copied")
        mapped = None
        try:
            with path.open("wb") as handle:
                np.lib.format.write_array_header_1_0(handle, {
                    "descr": np.lib.format.dtype_to_descr(array.dtype),
                    "fortran_order": False,
                    "shape": array.shape,
                })
                chunk_elements = max(1, FILE_ARRAY_CHUNK_BYTES // max(1, array.dtype.itemsize))
                for start in range(0, array.size, chunk_elements):
                    chunk = array.flat[start:start + chunk_elements]
                    handle.write(memoryview(chunk).cast("B"))
            identity = _file_identity(path.stat())
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            _require_file_identity(_file_identity(path.stat()), identity)
            name = str(path.resolve())
            self._file_arrays[name] = mapped
            self._file_identities[name] = identity
        except BaseException:
            if mapped is not None:
                mapped._mmap.close()
            path.unlink(missing_ok=True)
            raise
        return self.create_file_from(mapped)

    def create_from(self, array: np.ndarray) -> Tuple[SharedArraySpec, np.ndarray]:
        """
        Allocate shared memory sized to `array`, copy the data into it, and
        return both the descriptor and a NumPy view backed by the shared segment.
        """
        if self._file_backed:
            return self.create_file_from(array)
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
        if spec.storage == "file":
            mapping = self._file_arrays[spec.name]
            return np.ndarray(spec.shape, dtype=spec.dtype, buffer=mapping,
                              offset=spec.offset - int(mapping.offset))
        return self._arrays[spec.name]

    def cleanup(self, specs: Iterable[SharedArraySpec] | None = None) -> None:
        """
        Close mappings before deleting owned files; leave borrowed cache files intact.

        Call only after every attachment has closed. Optional subsets select
        shared-memory blocks or file mappings by name.
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
        if specs is None:
            names.update(self._file_arrays)
        for name in list(self._owned_blocks.keys()):
            if name in names:
                shm = self._owned_blocks.pop(name)
                shm.close()
                shm.unlink()
                self._arrays.pop(name, None)
        for name in list(self._file_arrays):
            if name in names:
                mapping = self._file_arrays.pop(name)
                mapping._mmap.close()
                self._file_identities.pop(name)
        if not self._file_arrays and self._file_directory is not None:
            self._file_directory.cleanup()
            self._file_directory = None


def attach_shared_array(spec: SharedArraySpec) -> SharedArrayAttachment:
    """
    Attach to an existing shared-memory backed array described by `spec`.
    """
    return SharedArrayAttachment(spec)
