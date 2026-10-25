"""Run-local, disk-backed market packing with one active GPU dataset."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import logging
import mmap
from threading import Lock
import weakref

import numpy as np


PREPARATION_CHUNK_BYTES = 8 * 2**20


@dataclass
class _MpsUploadSource:
    """Own aliases until the device has finished reading a mapped upload source."""

    mapped: object
    cpu: object = None
    destination: object = None

    def close(self):
        self.cpu = None
        self.destination = None
        self.mapped._mmap.close()


# A failed synchronization cannot establish that queued copies released their
# source pointers. Retain ownership until a later confirmed device quiescence;
# never drop an older unsafe mapping merely to bound this exceptional list.
_mps_upload_quarantine = []
_mps_upload_quarantine_lock = Lock()


def _quarantine_mps_upload_sources(sources):
    with _mps_upload_quarantine_lock:
        _mps_upload_quarantine.extend(sources)


def _close_mps_upload_sources(sources, *, preserve_error=False):
    first_error = None
    for source in sources:
        try:
            source.close()
        except BaseException as exc:
            if preserve_error or first_error is not None:
                logging.exception("Could not close a completed Metal upload source")
            else:
                first_error = exc
    if first_error is not None:
        raise first_error


def _mps_upload_quarantine_snapshot():
    with _mps_upload_quarantine_lock:
        return tuple(_mps_upload_quarantine)


def _release_mps_upload_sources_after_sync(pending):
    """Release only sources snapshotted before a confirmed successful wait."""
    with _mps_upload_quarantine_lock:
        claimed = [source for source in pending
                   if any(source is other for other in _mps_upload_quarantine)]
        _mps_upload_quarantine[:] = [source for source in _mps_upload_quarantine
                                    if not any(source is other for other in claimed)]
    for source in claimed:
        try:
            source.close()
        except BaseException as exc:
            # Preserve ownership even if a mapping-close failure prevents this
            # confirmed-quiescent source from being released immediately.
            _quarantine_mps_upload_sources((source,))
            if not isinstance(exc, Exception):
                raise
            logging.exception("Could not close a completed Metal upload source")


def _synchronize_mps_uploads(torch):
    # A different worker may quarantine newer queued work after this wait
    # completes. The pre-wait snapshot excludes those still-unsafe sources.
    pending = _mps_upload_quarantine_snapshot()
    torch.mps.synchronize()
    _release_mps_upload_sources_after_sync(pending)


def _discard_completed_upload_pages(array, completed_bytes, discarded_bytes=0):
    """Advise only complete consumed pages of an internal, unmodified mapping."""
    advise = getattr(array._mmap, "madvise", None)
    advice = getattr(mmap, "MADV_DONTNEED", None)
    page = mmap.PAGESIZE
    data_offset = int(array.offset) % mmap.ALLOCATIONGRANULARITY
    end = min(len(array._mmap), data_offset + int(completed_bytes))
    end -= end % page
    if end <= discarded_bytes:
        return discarded_bytes
    if callable(advise) and advice is not None:
        try:
            advise(advice, discarded_bytes, end - discarded_bytes)
        except OSError:
            logging.debug("Could not discard completed Metal upload pages", exc_info=True)
    return end


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
            raise ValueError(f"Incomplete GPU preparation array: {path.name}")


_current = ContextVar("cuda_suite_residency", default=None)


def current_gpu_residency():
    return _current.get()


def current_cuda_residency():
    return current_gpu_residency()


def _cleanup_after_failure(cleanup, *args, **kwargs):
    try:
        cleanup(*args, **kwargs)
    except BaseException:
        logging.exception("GPU suite cleanup failed while handling an earlier exception")


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


def compact_hourly_rows(candles, *, start_minute_of_hour=None):
    """Boundary rows for 1m hourly inputs; unknown phase uses a safe header bound."""
    candles = int(candles)
    if candles < 1:
        raise ValueError("Hourly GPU inputs require at least one candle")
    if start_minute_of_hour is None:
        return (candles + 59) // 60 + 1
    minute = int(start_minute_of_hour)
    if not 0 <= minute < 60:
        raise ValueError("Hourly GPU start minute must be in 0..59")
    # The canonical hourly builder never marks row zero as an observation.
    # Keep one sentinel row for a bindable device buffer when no boundary occurs.
    return max(1, (minute + candles - 1) // 60)


def multicoin_invariant_bytes(candles, coins, *, device, include_hourly_ranges=True,
                             compact_hourly_ranges=False, start_minute_of_hour=None):
    """Packed payload size, excluding candidate state and driver scratch."""
    # bars, execution ticks, fill ticks, touch ticks, nearest ticks, min-qty
    # bits/relation and (when requested) hourly ranges. CUDA packs relation as i8.
    per_coin = 16 + 8 + 8 + 8 + 4 + 4 + (1 if device == "cuda" else 4)
    total = int(candles) * int(coins) * per_coin
    if include_hourly_ranges:
        if compact_hourly_ranges:
            if device != "mps":
                raise ValueError("Compact hourly GPU inputs require Metal")
            return total + compact_hourly_rows(
                candles, start_minute_of_hour=start_minute_of_hour,
            ) * int(coins) * 4
        total += int(candles) * int(coins) * 4
    return total


METAL_STREAMED_TICK_FIELDS = frozenset((
    "fill_ticks", "touch_ticks", "touch_nearest_ticks", "touch_min_qty_bits", "touch_min_qty_relation",
))


def multicoin_resident_bytes(candles, coins, *, device, include_hourly_ranges=True,
                            stream_tick_inputs=False, compact_hourly_ranges=False,
                            start_minute_of_hour=None):
    if compact_hourly_ranges and not stream_tick_inputs:
        raise ValueError("Compact hourly GPU inputs require streamed ticks")
    full = multicoin_invariant_bytes(candles, coins, device=device,
                                    include_hourly_ranges=include_hourly_ranges,
                                    compact_hourly_ranges=compact_hourly_ranges,
                                    start_minute_of_hour=start_minute_of_hour)
    if not stream_tick_inputs:
        return full
    if device != "mps":
        raise ValueError("Streamed tick inputs require Metal")
    # Previous-touch and next-fill context add one row each to a replay window.
    from optimization.gpu.model import MPS_TM_MULTICOIN_CHUNK_BARS
    streamed_per_row = int(coins) * 28
    window_rows = min(int(candles), MPS_TM_MULTICOIN_CHUNK_BARS + 2)
    return full - int(candles) * streamed_per_row + window_rows * streamed_per_row


@dataclass(frozen=True)
class MpsInputMemoryBudget:
    """Run-start transport preferences, fixed before file-cache pages are touched."""

    device_bytes: int
    host_bytes: int


def capture_mps_input_memory_budget(torch):
    """Sample device and system headroom once, before preparing input histories."""
    import psutil

    recommended = int(torch.mps.recommended_max_memory())
    driver = int(torch.mps.driver_allocated_memory())
    available = int(psutil.virtual_memory().available)
    return MpsInputMemoryBudget(
        device_bytes=int(max(0, recommended - driver) * 0.45),
        host_bytes=max(0, min(int(available * 0.7), available - 2**30)),
    )


def prefer_streamed_ticks(candles, coins, budget, *, interval_ms=None):
    """Prefer resident Metal inputs when comfortable, streaming only if it saves RAM.

    This choice is pure and separate from live allocation admission. Reuse the
    same budget for header preflight and the effective prepared dataset.
    """
    full = multicoin_invariant_bytes(candles, coins, device="mps")
    streamed = multicoin_resident_bytes(
        candles, coins, device="mps", stream_tick_inputs=True,
        compact_hourly_ranges=interval_ms == 60_000,
    )
    return streamed < full and full > min(budget.device_bytes, budget.host_bytes)


def discard_readonly_file_pages(values):
    """Optionally evict clean raw input pages after independent packed outputs exist.

    The mapping remains open and valid for future reads. Writable, copy-on-write
    and shared-memory arrays do not qualify for this advisory optimization.
    """
    if not isinstance(values, np.ndarray) or values.flags.writeable:
        return False
    owner = values
    while owner is not None:
        if isinstance(owner, np.memmap):
            if owner.mode != "r":
                return False
            advice = getattr(mmap, "MADV_DONTNEED", None)
            advise = getattr(owner._mmap, "madvise", None)
            if advice is None or not callable(advise):
                return False
            try:
                advise(advice)
            except OSError:
                logging.debug("Could not discard read-only raw input pages", exc_info=True)
                return False
            return True
        owner = getattr(owner, "base", None)
    return False


def check_mps_invariant_memory(torch, invariant_bytes, *, working_set=None, require_free=True):
    """Check invariant admission, optionally deferring live device availability."""
    recommended_fn = getattr(torch.mps, "recommended_max_memory", None)
    if require_free or working_set is None:
        if not callable(recommended_fn):
            return  # Lightweight packer callers may not expose memory telemetry.
        total = int(recommended_fn())
    if require_free:
        allocated_fn = getattr(torch.mps, "driver_allocated_memory", None)
        free = max(0, total - (int(allocated_fn()) if callable(allocated_fn) else 0))
        budget = free if working_set is None else working_set
    else:
        budget = total if working_set is None else working_set
    if invariant_bytes > int(budget * 0.45):
        budget_name = ("initial available" if working_set is not None else
                       "available" if require_free else "nominal")
        raise MemoryError(
            f"Metal multicoin invariant tensors require {invariant_bytes / 2**30:.2f} GiB; "
            f"the {budget_name} recommended working set is {budget / 2**30:.2f} GiB "
            f"and its 45% safety limit is {int(budget * 0.45) / 2**30:.2f} GiB. "
            "Rejected before packing or device allocation."
        )
    if require_free and invariant_bytes > free:
        raise MemoryError(
            f"Metal multicoin invariant tensors require {invariant_bytes / 2**30:.2f} GiB; "
            f"the available recommended working set is {free / 2**30:.2f} GiB. "
            "Rejected before device allocation."
        )


class CudaSuiteResidency:
    device = "cuda"

    @staticmethod
    def _device_api(torch):
        return torch.cuda

    def _check_memory(self, torch, invariant_bytes):
        if self._working_set is None:
            self._working_set = int(torch.cuda.mem_get_info()[0])
        check_cuda_invariant_memory(torch, invariant_bytes, working_set=self._working_set)

    def _synchronize(self, torch):
        self._device_api(torch).synchronize()

    def _upload_file(self, torch, path):
        # Preserve CUDA's existing whole-file device-copy behavior.
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        return torch.as_tensor(array, device=self.device).contiguous()

    def __init__(self):
        self._directory = None
        self._entries = {}
        self._active = None
        self._active_owner = None
        self._working_set = None
        self._subset_arrays = []

    def _spill_directory(self):
        if self._directory is None:
            self._directory = TemporaryDirectory(prefix="passivbot-gpu-suite-")
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
            self._synchronize(torch)
        finally:
            self._clear_runners(entry)
            # Keep dictionary identity for compatible suite grouping.
            entry["data"].pop("organillo_buffers", None)
            entry["data"].update(entry["files"])
            self._active = None
            self._active_owner = None
            self._device_api(torch).empty_cache()

    def activate(self, proxy):
        import torch

        entry = next(entry for entry in self._entries.values() if entry["data"] is proxy.data)
        if self._active is not entry:
            self._release(torch)
            # CUDA driver/kernel workspace can remain resident after tensors are
            # released. It belongs to the reserved 55%, not a shrinking budget.
            self._check_memory(torch, entry["data"].get("resident_bytes", entry["data"]["invariant_bytes"]))
            self._active = entry
            try:
                for name, path in entry["files"].items():
                    if (entry["data"].get("stream_tick_inputs", False)
                            and name in METAL_STREAMED_TICK_FIELDS):
                        continue
                    entry["data"][name] = self._upload_file(torch, path)
            except BaseException:
                _cleanup_after_failure(self._release, torch)
                raise
        elif self._active_owner is None or self._active_owner() is not proxy:
            # Incompatible proxies and screening representatives can share the
            # market entry. Keep its tensors, but retain only one owner's scratch.
            try:
                self._synchronize(torch)
            finally:
                self._clear_runners(entry)
                self._active_owner = None
                self._device_api(torch).empty_cache()
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


class MetalSuiteResidency(CudaSuiteResidency):
    """One active dataset/scratch owner inside Metal's recommended working set."""

    device = "mps"

    @staticmethod
    def _device_api(torch):
        return torch.mps

    def _initial_working_set(self, torch):
        if self._working_set is None:
            total = int(torch.mps.recommended_max_memory())
            self._working_set = max(0, total - int(torch.mps.driver_allocated_memory()))
        return self._working_set

    def check_preparation_memory(self, torch, invariant_bytes):
        # Packing writes files only. The previous entry and its scratch remain
        # resident until activation, so they must not shrink the initial budget.
        check_mps_invariant_memory(
            torch, invariant_bytes, working_set=self._initial_working_set(torch),
            require_free=False,
        )

    def _check_memory(self, torch, invariant_bytes):
        check_mps_invariant_memory(
            torch, invariant_bytes, working_set=self._initial_working_set(torch),
        )

    def _synchronize(self, torch):
        _synchronize_mps_uploads(torch)

    def _upload_file(self, torch, path):
        """Upload one packed file without keeping all touched CPU pages active.

        A single destination owns the complete immutable device array. Large
        inputs copy bounded contiguous slices from a CPU alias, then discard
        consumed clean source pages after each completed copy. These startup
        waits do not change replay dispatch synchronization.
        """
        source = _MpsUploadSource(np.load(path, mmap_mode="c", allow_pickle=False))
        queued = failed = quarantined = False
        try:
            array = source.mapped
            if not array.flags.c_contiguous:
                raise ValueError("Packed Metal input must be C-contiguous")
            if PREPARATION_CHUNK_BYTES < array.dtype.itemsize:
                raise ValueError("Metal upload chunk budget is smaller than one input value")
            if array.nbytes <= PREPARATION_CHUNK_BYTES:
                queued = True
                source.destination = torch.as_tensor(array, device=self.device).contiguous()
                self._synchronize(torch)
                queued = False
                _discard_completed_upload_pages(array, array.nbytes)
            else:
                # Slice NumPy before exposing it to Torch. Slicing a full CPU
                # tensor would retain its history-sized storage even when its
                # numel is small, allowing MPS to stage the complete source.
                flat = array.reshape(-1)  # A NumPy view, never a full CPU copy.
                dtype = torch.from_numpy(flat[:0]).dtype
                source.destination = torch.empty(
                    array.shape, dtype=dtype, device=self.device,
                )
                destination = source.destination.reshape(-1)
                row_values = int(np.prod(array.shape[1:])) if array.ndim else 1
                chunk_values = PREPARATION_CHUNK_BYTES // array.dtype.itemsize
                if row_values <= chunk_values:
                    chunk_values = (chunk_values // row_values) * row_values
                discarded = 0
                for start in range(0, array.size, chunk_values):
                    end = min(start + chunk_values, array.size)
                    source.cpu = torch.from_numpy(flat[start:end])
                    queued = True
                    destination[start:end].copy_(source.cpu)
                    self._synchronize(torch)
                    queued = False
                    source.cpu = None
                    discarded = _discard_completed_upload_pages(
                        array, end * array.dtype.itemsize, discarded,
                    )
            return source.destination
        except BaseException:
            failed = True
            if queued:
                try:
                    self._synchronize(torch)
                except BaseException:
                    # Unknown queue ownership must outlive this failed helper
                    # and its service. Recover only after a later successful wait.
                    _quarantine_mps_upload_sources((source,))
                    quarantined = True
                    logging.exception("Metal upload cleanup synchronization failed")
            raise
        finally:
            if not quarantined:
                if failed:
                    _cleanup_after_failure(source.close)
                else:
                    source.close()

    def close(self):
        try:
            # Failed activation may already have cleared _active while an
            # unreadable device queue still owns its quarantined source.
            if self._active is None and _mps_upload_quarantine:
                import torch
                self._synchronize(torch)
        finally:
            super().close()


@contextmanager
def metal_residency_scope():
    """Keep Metal preparation, replay and cleanup on the owning worker."""
    residency = MetalSuiteResidency()
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
