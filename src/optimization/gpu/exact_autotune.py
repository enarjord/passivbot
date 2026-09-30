"""Initial CPU resource sizing and slow exact-queue tuning during real work."""

from collections import deque
import logging
import math
from pathlib import Path
import statistics
import time

import psutil

from optimization.gpu.autotune import (
    CalibrationCache,
    _digest,
    hardware_identity,
    workload_contract,
)

WINDOW = 24
MIN_SECONDS = 30.0
MIN_GENERATIONS = 4
MIB = 1024**2


def _cgroup_limits(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Read cgroup-v2 constraints at the process group and every ancestor."""
    try:
        relative = next(
            line[3:] for line in membership.read_text().splitlines() if line.startswith("0::")
        )
    except (OSError, StopIteration):
        return None, None
    group = (root / relative.lstrip("/")).resolve()
    root = root.resolve()
    if not group.is_relative_to(root):
        return None, None
    cpus = memory = None
    while True:
        try:
            quota, period = (group / "cpu.max").read_text().split()
            if quota != "max":
                count = max(1, int(quota) // int(period))
                cpus = count if cpus is None else min(cpus, count)
        except (OSError, ValueError, ZeroDivisionError):
            pass  # Optional resource hints; host/affinity limits still apply.
        try:
            maximum = (group / "memory.max").read_text().strip()
            if maximum != "max":
                available = max(0, int(maximum) - int((group / "memory.current").read_text()))
                memory = available if memory is None else min(memory, available)
        except (OSError, ValueError):
            pass
        if group == root:
            break
        group = group.parent
    return cpus, memory


def resource_snapshot():
    process = psutil.Process()
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    logical = psutil.cpu_count() or 1
    try:
        logical = min(logical, len(process.cpu_affinity()))
    except (AttributeError, OSError, psutil.Error):
        pass  # macOS has no CPU affinity API.
    quota, memory = _cgroup_limits()
    cores = min(physical, logical, quota if quota is not None else logical)
    available = int(psutil.virtual_memory().available)
    if memory is not None:
        available = min(available, memory)
    return dict(cores=max(1, cores), available=available, rss=process.memory_info().rss)


def prepared_bytes(evaluator):
    # Shared candle mappings are not copied for each suite scenario. Reserve for
    # the largest exact replay, not the sum of shared views of the same data.
    contexts = [evaluator, *getattr(evaluator, "contexts", [])]
    return max(
        (
            sum(int(array.nbytes) for array in getattr(ctx, "shared_hlcvs_np", {}).values())
            for ctx in contexts
        ),
        default=0,
    )


def initial_workers(requested, inherited, evaluator, *, mode, pending=None):
    if requested is not None:
        return int(requested) or int(inherited)
    if mode == "off":
        return int(inherited)
    try:
        resources = resource_snapshot()
    except (OSError, psutil.Error) as error:
        logging.warning("GPU exact auto-sizing unavailable; using optimize.n_cpus: %s", error)
        return int(inherited)
    # RSS is deliberately conservative: interpreter/library costs plus room for
    # private Rust replay state. Keep 40% of available RAM for GPU/host growth.
    per_worker = max(512 * MIB, resources["rss"] + 2 * prepared_bytes(evaluator))
    memory_workers = max(1, int(resources["available"] * 0.6) // per_worker)
    workers = max(1, min(max(1, resources["cores"] - 1), memory_workers))
    if pending:
        workers = min(workers, pending)
    logging.info(
        "GPU exact auto-sizing | workers=%d cores=%d available_mib=%d estimated_worker_mib=%d",
        workers,
        resources["cores"],
        resources["available"] // MIB,
        per_worker // MIB,
    )
    return workers


class ExactQueueController:
    """Bounded queue trials; complete validation allocations remain unchanged."""

    def __init__(
        self,
        workers,
        validations,
        proxies,
        *,
        mode="auto",
        cache_dir=None,
        clock=time.perf_counter,
        context=None,
        hardware=None
    ):
        self.floor = max(workers, validations)
        self.ceiling = 4 * self.floor
        self.step = self.floor
        self.limit = 2 * self.floor
        self.proxies = proxies
        self.clock = clock
        self.cache = CalibrationCache(cache_dir)
        self.key = _digest(
            dict(
                kind="exact_queue_v1",
                workers=workers,
                validations=validations,
                hardware=(
                    hardware if hardware is not None else hardware_identity(proxies[0]._torch)
                ),
                context=context,
                proxies=[
                    dict(
                        execution=workload_contract(item.checkpoint_contract),
                        metrics=sorted(item.needed_metrics),
                        implementation=getattr(
                            getattr(item, "batch_tuner", None), "identity", None
                        ),
                    )
                    for item in proxies
                ],
            )
        )
        self.samples = deque(maxlen=WINDOW)
        self.collected = deque(maxlen=self.ceiling)
        self.baseline = None
        self.direction = 1
        self.cooldown = 0
        self.warmed = False
        if mode != "refresh":
            cached = self.cache.read(self.key, "max_pending_exact", self.floor, self.ceiling)
            if cached is not None:
                self.limit = cached
        for item in proxies:
            tuner = getattr(item, "batch_tuner", None)
            if tuner is not None:
                tuner.allow_trial = lambda: self.baseline is None
        self.reset(0)
        logging.info(
            "GPU exact queue auto-tune starting | pending=%d floor=%d ceiling=%d",
            self.limit,
            self.floor,
            self.ceiling,
        )

    def proxy_state(self):
        tuners = [getattr(item, "batch_tuner", None) for item in self.proxies]
        revision = tuple(getattr(tuner, "revision", 0) for tuner in tuners)
        trial = any(
            controller.baseline is not None
            for tuner in tuners
            if tuner is not None
            for controller in tuner.controllers.values()
        )
        return revision, trial

    def reset(self, generation):
        self.samples.clear()
        self.collected.clear()
        self.completed = 0
        self.started = self.clock()
        self.generation = generation
        self.revision = self.proxy_state()[0]

    def record(self, worker_seconds, queue_seconds):
        # The collector only appends timing. The main thread consumes it after
        # joining the collector, before admission of the next generation.
        self.collected.append((worker_seconds, queue_seconds))

    def update(self, generation):
        revision, proxy_trial = self.proxy_state()
        if proxy_trial or revision != self.revision:
            self.reset(generation)
            return
        while self.collected:
            worker_seconds, queue_seconds = self.collected.popleft()
            if not (
                math.isfinite(worker_seconds)
                and worker_seconds > 0
                and math.isfinite(queue_seconds)
                and queue_seconds >= 0
            ):
                continue
            self.samples.append((worker_seconds, queue_seconds))
            self.completed += 1
        elapsed = self.clock() - self.started
        if (
            len(self.samples) < WINDOW
            or elapsed < MIN_SECONDS
            or generation - self.generation < MIN_GENERATIONS
        ):
            return
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        if rate <= 0:
            return
        work = statistics.median(item[0] for item in self.samples)
        wait = statistics.median(item[1] for item in self.samples)
        self.reset(generation)
        # First completed window excludes worker startup and cold exact replay.
        if not self.warmed:
            self.warmed = True
            return
        if self.baseline is not None:
            old_limit, old_rate = self.baseline
            self.baseline = None
            accepted = rate >= old_rate * (0.98 if self.limit < old_limit else 1.05)
            if accepted:
                self.cache.write(self.key, "max_pending_exact", self.limit, rate, elapsed)
                self.cooldown = 1
            else:
                self.limit = old_limit
                self.direction *= -1
                self.cooldown = 3
            logging.info(
                "GPU exact queue auto-tune %s | pending=%d validations/s=%.3f median_work=%.3f median_wait=%.3f",
                "accepted" if accepted else "retained",
                self.limit,
                rate,
                work,
                wait,
            )
            return
        self.cache.write(self.key, "max_pending_exact", self.limit, rate, elapsed)
        if self.cooldown:
            self.cooldown -= 1
            return
        trial = max(self.floor, min(self.ceiling, self.limit + self.direction * self.step))
        if trial == self.limit:
            self.direction *= -1
            self.cooldown = 1
            return
        self.baseline = (self.limit, rate)
        self.limit = trial
        logging.info(
            "GPU exact queue auto-tune trial | pending=%d previous=%d validations/s=%.3f",
            self.limit,
            self.baseline[0],
            rate,
        )
