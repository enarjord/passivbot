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


_cuda_upload_quarantine = []
_cuda_upload_quarantine_lock = Lock()


def _quarantine_cuda_upload_sources(sources):
    with _cuda_upload_quarantine_lock:
        _cuda_upload_quarantine.extend(sources)


def _cuda_upload_quarantine_snapshot():
    with _cuda_upload_quarantine_lock:
        return tuple(_cuda_upload_quarantine)


def _release_cuda_upload_sources_after_sync(pending):
    """Only a confirmed wait may release sources of an interrupted CUDA copy."""
    with _cuda_upload_quarantine_lock:
        claimed = [source for source in pending
                   if any(source is other for other in _cuda_upload_quarantine)]
        _cuda_upload_quarantine[:] = [source for source in _cuda_upload_quarantine
                                     if not any(source is other for other in claimed)]
    for source in claimed:
        try:
            source.close()
        except BaseException as exc:
            _quarantine_cuda_upload_sources((source,))
            if not isinstance(exc, Exception):
                raise
            logging.exception("Could not close a completed CUDA upload source")


def _synchronize_cuda_uploads(torch):
    pending = _cuda_upload_quarantine_snapshot()
    torch.cuda.synchronize()
    _release_cuda_upload_sources_after_sync(pending)


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


def cuda_max_vram_reserve_bytes(total):
    """Leave room for CUDA allocations outside Passivbot's replay estimates."""
    return max(512 * 2**20, int(total * 0.10))


class CudaReplayMemoryPressure(MemoryError):
    """Live max-mode admission failed before candidate replay or allocations."""


def check_cuda_invariant_memory(
    torch, invariant_bytes, *, working_set=None, require_free=True, input_memory_budget=None,
):
    max_throughput = input_memory_budget is not None and input_memory_budget.max_throughput
    if input_memory_budget is not None:
        working_set = input_memory_budget.initial_free_bytes
    if require_free or working_set is None:
        free, total = map(int, torch.cuda.mem_get_info())
    budget = free if working_set is None else working_set
    allowance = input_memory_budget.device_bytes if max_throughput else int(budget * 0.45)
    if invariant_bytes > allowance:
        if max_throughput:
            raise MemoryError(
                f"CUDA multicoin invariant tensors require {invariant_bytes / 2**30:.2f} GiB, "
                f"above the max-mode input budget of {allowance / 2**30:.2f} GiB"
            )
        raise MemoryError(
            f"CUDA multicoin invariant tensors would consume {invariant_bytes / 2**30:.2f} "
            f"GiB, above the 45% safety limit of {budget / 2**30:.2f} GiB "
            f"{'free VRAM' if working_set is None else 'initial free VRAM'}"
        )
    if require_free and max_throughput:
        reserve = cuda_max_vram_reserve_bytes(total)
        if invariant_bytes > free - reserve:
            raise CudaReplayMemoryPressure(
                f"CUDA multicoin invariant tensors require {invariant_bytes / 2**30:.2f} GiB "
                f"with {free / 2**30:.2f} GiB currently free and a {reserve / 2**30:.2f} GiB reserve"
            )
    elif require_free and invariant_bytes > free:
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
            if device not in {"mps", "cuda"}:
                raise ValueError("Compact hourly GPU inputs require Metal or CUDA")
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
    if device not in {"mps", "cuda"}:
        raise ValueError("Streamed tick inputs require Metal or CUDA")
    # Previous-touch and next-fill context add one row each to a replay window.
    from optimization.gpu.model import MPS_TM_MULTICOIN_CHUNK_BARS
    streamed_per_row = int(coins) * (25 if device == "cuda" else 28)
    window_rows = min(int(candles), MPS_TM_MULTICOIN_CHUNK_BARS + 2)
    return full - int(candles) * streamed_per_row + window_rows * streamed_per_row


@dataclass(frozen=True)
class MpsInputMemoryBudget:
    """Run-start transport preferences, fixed before file-cache pages are touched."""

    device_bytes: int
    host_bytes: int
    max_throughput: bool = False


def capture_mps_input_memory_budget(torch, *, max_throughput=False):
    """Sample device and system headroom once, before preparing input histories."""
    import psutil

    recommended = int(torch.mps.recommended_max_memory())
    driver = int(torch.mps.driver_allocated_memory())
    available = int(psutil.virtual_memory().available)
    return MpsInputMemoryBudget(
        device_bytes=int(max(0, recommended - driver) * 0.45),
        host_bytes=max(0, min(int(available * 0.7), available - 2**30)),
        max_throughput=max_throughput,
    )


class MetalReplayMemoryPressure(MemoryError):
    """Valid telemetry refused a Metal max allocation before it started."""


def mps_system_reserve_bytes(total):
    """Keep space for macOS and the desktop in Metal max mode."""
    return max(2**30, total // 10)


def mps_system_headroom_bytes(*, pending_host_bytes=0, require_system_telemetry=True):
    """Bytes available after the system reserve and as-yet-unallocated host work."""
    if type(pending_host_bytes) is not int or pending_host_bytes < 0:
        raise ValueError("Metal pending host memory must be a nonnegative byte count")
    try:
        import psutil

        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
        if type(total) is not int or type(available) is not int or not 0 <= available <= total or total == 0:
            raise ValueError("Invalid system memory telemetry")
    except Exception as exc:
        if require_system_telemetry:
            raise RuntimeError("Metal max admission requires valid system memory telemetry") from exc
        # Optional observations may decline expansion. Live admission requires
        # valid telemetry and does not convert probe errors into pressure.
        return 0
    return max(0, available - mps_system_reserve_bytes(total) - pending_host_bytes)


def mps_max_available_bytes(torch, *, pending_host_bytes=0, require_system_telemetry=True):
    """Additional device bytes that fit both MPS and system headroom."""
    recommended = torch.mps.recommended_max_memory()
    driver = torch.mps.driver_allocated_memory()
    if type(recommended) is not int or recommended <= 0 or type(driver) is not int or driver < 0:
        raise RuntimeError("Metal max admission requires valid MPS memory telemetry")
    system_headroom = mps_system_headroom_bytes(
        pending_host_bytes=pending_host_bytes, require_system_telemetry=require_system_telemetry,
    )
    device_headroom = max(0, recommended - driver - max(256 * 2**20, recommended // 5))
    # Current MPS and worker allocations are already visible in system memory.
    # Compare the two remaining limits instead of charging GPU allocations twice.
    return min(device_headroom, system_headroom)


@dataclass(frozen=True)
class CudaInputMemoryBudget:
    """Freeze CUDA transport preference and its original invariant allowance."""

    device_bytes: int
    host_bytes: int
    initial_free_bytes: int
    max_throughput: bool = False


def capture_cuda_input_memory_budget(torch, *, max_throughput=False):
    """Capture CUDA and host headroom before history preparation touches pages."""
    if max_throughput and getattr(getattr(torch, "version", None), "hip", None):
        raise ValueError(
            "optimize.gpu.tuning_mode='max' is unavailable on AMD ROCm. "
            "Use optimize.gpu.tuning_mode='conservative'."
        )

    import psutil

    free, total = map(int, torch.cuda.mem_get_info())
    available = int(psutil.virtual_memory().available)
    if max_throughput:
        # The rest is available for replay state, workspaces and upload staging.
        device_bytes = max(0, min(int(free * 0.70), free - cuda_max_vram_reserve_bytes(total) - 512 * 2**20))
        host_bytes = max(0, min(int(available * 0.85), available - 512 * 2**20))
        return CudaInputMemoryBudget(device_bytes, host_bytes, free, max_throughput=True)
    return CudaInputMemoryBudget(
        device_bytes=int(free * 0.45),
        host_bytes=max(0, min(int(available * 0.7), available - 2**30)),
        initial_free_bytes=free,
    )


def supports_streamed_ticks(strategy_kind, device):
    return (
        strategy_kind == "trailing_martingale" and device in {"mps", "cuda"}
        or strategy_kind == "ema_anchor" and device == "cuda"
    )


def prefer_streamed_ticks(candles, coins, budget, *, interval_ms=None, device="mps"):
    """Prefer resident inputs when comfortable, streaming only if it saves memory.

    This choice is pure and separate from live allocation admission. Reuse the
    same budget for header preflight and the effective prepared dataset.
    """
    full = multicoin_invariant_bytes(candles, coins, device=device)
    streamed = multicoin_resident_bytes(
        candles, coins, device=device, stream_tick_inputs=True,
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


def check_mps_invariant_memory(
    torch, invariant_bytes, *, working_set=None, require_free=True, input_memory_budget=None,
):
    """Check invariant admission, optionally deferring live device availability."""
    if input_memory_budget is not None and input_memory_budget.max_throughput:
        allowance = input_memory_budget.device_bytes
        if invariant_bytes > allowance:
            raise MemoryError(
                f"Metal multicoin invariant tensors require {invariant_bytes / 2**20:.1f} MiB, "
                f"above the initial max-mode input budget of {allowance / 2**20:.1f} MiB. "
                "Rejected before packing or device allocation."
            )
        if require_free:
            # Large active histories copy one bounded source slice at a time.
            available = mps_max_available_bytes(
                torch, pending_host_bytes=min(PREPARATION_CHUNK_BYTES, invariant_bytes),
            )
            if invariant_bytes > available:
                raise MetalReplayMemoryPressure(
                    f"Metal multicoin invariant tensors require {invariant_bytes / 2**20:.1f} MiB, "
                    f"with {available / 2**20:.1f} MiB of live unified-memory headroom. "
                    "Rejected before device allocation."
                )
        return

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

    def _initial_working_set(self, torch):
        if self._working_set is None:
            self._working_set = int(torch.cuda.mem_get_info()[0])
        return self._working_set

    def check_preparation_memory(self, torch, invariant_bytes):
        # Preparation writes files while the previous scenario may remain on
        # device. Activation releases it before checking live availability.
        check_cuda_invariant_memory(
            torch, invariant_bytes, working_set=self._initial_working_set(torch),
            require_free=False, input_memory_budget=self._input_memory_budget,
        )

    def _check_memory(self, torch, invariant_bytes):
        check_cuda_invariant_memory(
            torch, invariant_bytes, working_set=self._initial_working_set(torch),
            input_memory_budget=self._input_memory_budget,
        )

    def _synchronize(self, torch):
        _synchronize_cuda_uploads(torch)

    def _upload_file(self, torch, path):
        # Preserve CUDA's existing whole-file device-copy behavior.
        array = np.load(path, mmap_mode="c", allow_pickle=False)
        return torch.as_tensor(array, device=self.device).contiguous()

    def __init__(self, *, input_memory_budget=None):
        self._directory = None
        self._entries = {}
        self._active = None
        self._active_owner = None
        self._input_memory_budget = input_memory_budget
        self._working_set = (None if input_memory_budget is None
                             else input_memory_budget.initial_free_bytes)
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

    def _entry(self, proxy):
        return next(entry for entry in self._entries.values() if entry["data"] is proxy.data)

    def register(self, proxy):
        self._entry(proxy)["owners"].append(weakref.ref(proxy))

    def _streamed_fallback_bytes(self, proxy):
        data = proxy.data
        if (self.device != "cuda" or not getattr(proxy, "_max_throughput", False)
                or not proxy.temporal_chunking or not supports_streamed_ticks(proxy.strategy_kind, self.device)
                or data["stream_tick_inputs"]):
            return None
        entry = self._entry(proxy)
        if not METAL_STREAMED_TICK_FIELDS <= entry["files"].keys():
            return None
        streamed = multicoin_resident_bytes(data["n"], data["n_coins"], device=self.device,
                                           stream_tick_inputs=True)
        return streamed if streamed < data["resident_bytes"] else None

    def can_stream_on_pressure(self, proxy):
        return self._streamed_fallback_bytes(proxy) is not None

    def stream_on_pressure(self, proxy):
        """Switch packed tick files to windowed inputs after a live admission failure."""
        import torch

        streamed = self._streamed_fallback_bytes(proxy)
        if streamed is None:
            raise RuntimeError("CUDA packed input cannot switch to streamed ticks")
        if self._active is self._entry(proxy):
            self._release(torch)
        data = proxy.data
        logging.info("GPU max input pressure | transport=streamed_ticks coins=%d bars=%d "
                     "previous_mib=%.1f resident_mib=%.1f", data["n_coins"], data["n"],
                     data["resident_bytes"] / 2**20, streamed / 2**20)
        # Dense and streamed packers write identical tick files. Keep the
        # existing dense hourly table and cache identity for every owner.
        data.update(stream_tick_inputs=True, resident_bytes=streamed)
        self._activate(proxy, torch)

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

        try:
            self._activate(proxy, torch)
        except CudaReplayMemoryPressure:
            if not self.can_stream_on_pressure(proxy):
                raise
        else:
            return
        self.stream_on_pressure(proxy)

    def _activate(self, proxy, torch):
        entry = self._entry(proxy)
        if self._active is not entry:
            self._release(torch)
            # Driver and kernel allocations can remain resident after tensors
            # are released. Keep the original allowance and check live free space.
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
            except CudaReplayMemoryPressure:
                if not self.can_stream_on_pressure(proxy):
                    _cleanup_after_failure(self._release, torch)
                # The fallback releases this entry after the failed constructor
                # and its device references have left the traceback.
                raise
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
def cuda_residency_scope(*, input_memory_budget=None):
    """Own run-local packing/residency on the calling execution thread."""
    residency = CudaSuiteResidency(input_memory_budget=input_memory_budget)
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

    def __init__(self, *, input_memory_budget=None):
        super().__init__()
        if input_memory_budget is not None and not isinstance(input_memory_budget, MpsInputMemoryBudget):
            raise ValueError("Metal residency requires a Metal input memory budget")
        self._input_memory_budget = input_memory_budget

    def resolve_max_input_memory_budget(self, torch, input_memory_budget):
        """Freeze one input allowance before the first max-mode dataset is packed."""
        current = self._input_memory_budget
        if input_memory_budget is None:
            input_memory_budget = current or capture_mps_input_memory_budget(torch, max_throughput=True)
        if (not isinstance(input_memory_budget, MpsInputMemoryBudget)
                or not input_memory_budget.max_throughput):
            raise ValueError("Metal input memory budget does not match the GPU tuning mode")
        if current is None and not self._entries:
            self._input_memory_budget = input_memory_budget
        elif current != input_memory_budget:
            raise ValueError("Metal max replay and its residency owner require the same input memory budget")
        return self._input_memory_budget

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
        budget = self._input_memory_budget
        if budget is not None and budget.max_throughput:
            check_mps_invariant_memory(
                torch, invariant_bytes, require_free=False, input_memory_budget=budget,
            )
            return
        check_mps_invariant_memory(
            torch, invariant_bytes, working_set=self._initial_working_set(torch),
            require_free=False,
        )

    def _check_memory(self, torch, invariant_bytes):
        budget = self._input_memory_budget
        if budget is not None and budget.max_throughput:
            check_mps_invariant_memory(torch, invariant_bytes, input_memory_budget=budget)
            return
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
def metal_residency_scope(*, input_memory_budget=None):
    """Keep Metal preparation, replay and cleanup on the owning worker."""
    residency = MetalSuiteResidency(input_memory_budget=input_memory_budget)
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
