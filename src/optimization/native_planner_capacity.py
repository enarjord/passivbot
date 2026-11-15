"""Choose how many host processes can prepare candidates for CUDA max."""

from math import prod
import os
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

from optimization.gpu.residency import multicoin_invariant_bytes


_MIB = 2**20
# Leave CPU time for the optimizer and the CUDA owner. Only consider worker
# counts exercised in throughput qualification.
_CPU_RESERVE = 2
_WORKER_COUNTS = (8, 4)
_MINIMUM_CANDIDATES_PER_WORKER = 16


def _mount_path(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_root(lines, mounts, *, filesystem, controller):
    for line in lines:
        _group, names, path = line.split(":", 2)
        if controller in names.split(","):
            member = PurePosixPath(path)
            if ".." in member.parts:
                raise OSError(f"Cannot resolve the current process's {filesystem} {controller} hierarchy")
            break
    else:
        return None
    for line in mounts:
        before, after = line.split(" - ", 1)
        fs_type, _source, options = after.split()
        if fs_type != filesystem or (controller and controller not in options.split(",")):
            continue
        _id, _parent, _device, root, mount, *_ = before.split()
        root = PurePosixPath(_mount_path(root))
        if member.is_relative_to(root):
            mount = Path(_mount_path(mount))
            return mount, mount.joinpath(*member.relative_to(root).parts)
    raise OSError(f"Cannot resolve the current process's {filesystem} {controller} hierarchy")


def _ancestors(root):
    if root is None:
        return
    mount, current = root
    while True:
        yield current
        if current == mount:
            return
        current = current.parent


def _read_optional(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None  # An undelegated controller has no limit at this level.


def _read_cgroup_limits(proc):
    lines = (proc / "cgroup").read_text().splitlines()
    mounts = (proc / "mountinfo").read_text().splitlines()
    unified = _cgroup_root(lines, mounts, filesystem="cgroup2", controller="")
    cpu_v1 = _cgroup_root(lines, mounts, filesystem="cgroup", controller="cpu")
    memory_v1 = _cgroup_root(lines, mounts, filesystem="cgroup", controller="memory")
    cpu_limits, memory_limits = [], []
    for directory in _ancestors(unified):
        cpu = _read_optional(directory / "cpu.max")
        if cpu is not None:
            quota, period = cpu.split()
            if quota != "max":
                cpu_limits.append(int(quota) / int(period))
        limits = [_read_optional(directory / name) for name in ("memory.max", "memory.high")]
        numeric = [int(limit) for limit in limits if limit not in (None, "max")]
        if numeric:
            memory_limits.append(min(numeric) - int((directory / "memory.current").read_text()))
    for directory in _ancestors(cpu_v1):
        quota = _read_optional(directory / "cpu.cfs_quota_us")
        if quota is not None and int(quota) >= 0:
            cpu_limits.append(int(quota) / int((directory / "cpu.cfs_period_us").read_text()))
    for directory in _ancestors(memory_v1):
        limit = _read_optional(directory / "memory.limit_in_bytes")
        if limit is not None:
            memory_limits.append(int(limit) - int((directory / "memory.usage_in_bytes").read_text()))
    return (min(cpu_limits) if cpu_limits else None,
            max(0, min(memory_limits)) if memory_limits else None)


def _host_capacity():
    """Return current scheduler/RAM headroom, or None if telemetry is unavailable."""
    import psutil

    proc = Path("/proc/self")
    try:
        if hasattr(os, "sched_getaffinity"):
            affinity = len(os.sched_getaffinity(0))
        else:
            affinity = os.process_cpu_count() if hasattr(os, "process_cpu_count") else os.cpu_count()
        available = int(psutil.virtual_memory().available)
        quota, cgroup_available = _read_cgroup_limits(proc) if sys.platform.startswith("linux") else (None, None)
        if not affinity or available < 0 or (quota is not None and not 0 < quota):
            return None
        cpu = min(affinity, int(quota)) if quota is not None else affinity
        memory = min(available, cgroup_available) if cgroup_available is not None else available
        return cpu, memory
    except (OSError, psutil.Error, ValueError, ZeroDivisionError):
        return None


def cuda_plan_preparation_memory(planner, admission_slots):
    # Input packing is owned by one CUDA worker. Use the largest input span,
    # including its source pages and full packed view, across registered datasets.
    packing = 0
    dataset_metadata = 0
    for binding in planner.bindings:
        dataset = binding.dataset
        rows = dataset.time_range[1] - dataset.time_range[0]
        coins = len(dataset.coin_indices)
        source = rows * prod(dataset.hlcvs.shape[1:]) * np.dtype(dataset.hlcvs.dtype).itemsize
        packed = multicoin_invariant_bytes(rows, coins, device="cuda")
        packing = max(packing, source + packed)
        dataset_metadata += len(dataset.config_json) + len(dataset.markets_json)
    # Account for parent request/config/result objects and each worker's
    # overlapping wire representations. These are allowances, not allocations.
    candidate_bytes = max(256 * 1024, dataset_metadata * 2)
    parent_bytes = packing + admission_slots * candidate_bytes + 1024 * _MIB
    worker_bytes = 320 * _MIB + dataset_metadata * 8 + candidate_bytes * 4
    return parent_bytes, worker_bytes


def cuda_plan_worker_count(planner, *, admission_slots, available_rows):
    """Admit a measured worker count within current host and scheduler headroom."""
    if available_rows < min(_WORKER_COUNTS) * _MINIMUM_CANDIDATES_PER_WORKER:
        return 0
    capacity = _host_capacity()
    if capacity is None:
        return 0
    cpus, available = capacity
    parent_bytes, worker_bytes = cuda_plan_preparation_memory(planner, admission_slots)
    memory_workers = max(0, available - parent_bytes) // worker_bytes
    cpu_workers = max(0, cpus - _CPU_RESERVE)
    demand_workers = available_rows // _MINIMUM_CANDIDATES_PER_WORKER
    feasible = min(memory_workers, cpu_workers, demand_workers)
    return next((workers for workers in _WORKER_COUNTS if workers <= feasible), 0)


def cuda_plan_pool_fits(*, parent_bytes, workers):
    """Check an existing idle pool before submitting work for the next cohort."""
    capacity = _host_capacity()
    if capacity is None:
        return False
    cpus, available = capacity
    # Available memory already excludes resident worker pages. Reserve the
    # remaining parent/replay budget without charging those pages a second time.
    return workers <= cpus - _CPU_RESERVE and available >= parent_bytes
