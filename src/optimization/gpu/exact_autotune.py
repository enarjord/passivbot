"""Initial CPU sizing and bounded exact-queue tuning during production validation."""

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
LONG_WINDOW_SECONDS = 120.0
LONG_WINDOW_SAMPLES = 4
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


def _affinity_physical_cores(allowed, physical, logical, root=Path("/sys/devices/system/cpu")):
    """Count package/core pairs within Linux affinity, with a conservative fallback."""
    try:
        pairs = {
            (
                (root / f"cpu{cpu}/topology/physical_package_id").read_text().strip(),
                (root / f"cpu{cpu}/topology/core_id").read_text().strip(),
            )
            for cpu in allowed
        }
        if pairs and all(package and core for package, core in pairs):
            return min(physical, len(pairs))
    except OSError:
        pass  # Optional topology; do not assume each allowed SMT sibling is a core.
    return max(1, min(physical, len(allowed) * physical // logical))


def resource_snapshot():
    process = psutil.Process()
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    logical = psutil.cpu_count() or 1
    try:
        allowed = process.cpu_affinity()
        if allowed:
            physical = _affinity_physical_cores(allowed, physical, logical)
            logical = min(logical, len(allowed))
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
    contexts = list(getattr(evaluator, "contexts", []))
    prepared = getattr(evaluator, "get_prepared_context_data", None)
    if contexts and prepared is not None:
        # Lazy suite contexts do not populate shared_hlcvs_np. Use the exact
        # prepared time views; coin subsetting remains a Rust index mapping.
        # Attaching/shared views does not copy each scenario's master dataset.
        return max(
            sum(int(prepared(ctx, exchange)[0].nbytes) for exchange in ctx.exchanges)
            for ctx in contexts
        )
    contexts = [evaluator, *contexts]
    return max(
        (
            sum(int(array.nbytes) for array in getattr(ctx, "shared_hlcvs_np", {}).values())
            for ctx in contexts
        ),
        default=0,
    )


def capture_worker_rss(requested, *, mode):
    """Capture the CPU coordinator baseline before GPU proxies allocate history."""
    if requested is not None or mode == "off":
        return None
    try:
        return psutil.Process().memory_info().rss
    except (OSError, psutil.Error) as error:
        logging.warning("GPU exact worker memory baseline unavailable: %s", error)
        return None


def initial_workers(requested, inherited, evaluator, *, mode, pending=None, baseline_rss=None):
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
    worker_rss = resources["rss"] if baseline_rss is None else baseline_rss
    per_worker = max(512 * MIB, worker_rss + 2 * prepared_bytes(evaluator))
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
        self.workers = workers
        self.floor = max(workers, validations)
        self.ceiling = 4 * self.floor
        self.step = self.floor
        self.limit = 2 * self.floor
        self.proxies = proxies
        self.clock = clock
        self.cache = CalibrationCache(cache_dir)
        self.key = _digest(
            dict(
                kind="exact_queue_v4",
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
        self.epoch = 0
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

    def reset(self, generation, *, invalidate=False):
        if invalidate:
            self.epoch += 1
        self.samples.clear()
        self.collected.clear()
        self.completed = 0
        self.worker_seconds = 0.0
        self.started = self.clock()
        self.generation = generation
        self.revision = self.proxy_state()[0]

    def finish_seed_screen(self, generation):
        # Seed-only GPU classes may never be visited by evolution. Retire their
        # unfinished trials before seed CPU admission can gather queue evidence.
        for proxy in self.proxies:
            tuner = getattr(proxy, "batch_tuner", None)
            if tuner is not None:
                retire = getattr(tuner, "retire_trials", None)
                if retire is not None:
                    retire()
        self.update(generation)

    def finish_bootstrap(self, generation):
        if self.baseline is not None:
            old_limit, _ = self.baseline
            self.baseline = None
            self.limit = old_limit
            self.epoch += 1
            self.cooldown = 1
            logging.info(
                "GPU exact queue auto-tune retained | pending=%d reason=seed_phase_end",
                self.limit,
            )
        self.reset(generation)

    def record(self, worker_seconds, queue_seconds, started, finished, *, epoch,
               admission_stall=None):
        # The collector only appends timing. The main thread consumes it after
        # joining the collector, before admission of the next generation.
        self.collected.append((worker_seconds, queue_seconds, started, finished, epoch,
                               admission_stall))

    def update(self, generation):
        revision, proxy_trial = self.proxy_state()
        if proxy_trial or revision != self.revision:
            self.reset(generation, invalidate=True)
            return
        while self.collected:
            worker_seconds, queue_seconds, started, finished, epoch, stall = self.collected.popleft()
            if epoch != self.epoch:
                continue  # Work admitted before the current queue/batch trial is not evidence.
            if not (
                math.isfinite(worker_seconds)
                and worker_seconds > 0
                and math.isfinite(queue_seconds)
                and queue_seconds >= 0
                and math.isfinite(started)
                and math.isfinite(finished)
                and finished > started
            ):
                continue
            if stall is not None and not (
                len(stall) == 2 and all(math.isfinite(value) for value in stall)
                and stall[0] <= stall[1] <= started
            ):
                continue
            self.samples.append((worker_seconds, queue_seconds, started, finished, stall))
            self.completed += 1
            self.worker_seconds += worker_seconds
        # Include admission stalls caused by queue backpressure, through the next
        # submission (including its GPU pass). Otherwise a shallow queue can hide
        # the CPU idle gaps it causes. Unrelated GPU pauses/collector delays stay
        # excluded. Duplicate/overlapping stalls in a submission wave count once.
        active = [(item[2], item[3]) for item in self.samples]
        intervals = active + [item[4] for item in self.samples if item[4] is not None]
        elapsed = 0.0
        end = float("-inf")
        for start, stop in sorted(intervals):
            elapsed += max(0.0, stop - max(start, end))
            end = max(end, stop)
        if not self.samples:
            return
        work = statistics.median(item[0] for item in self.samples)
        wait = statistics.median(item[1] for item in self.samples)
        # Accumulate measured work even when very fast tasks roll out of the
        # bounded rate window. Divide by pool capacity so parallel jobs cannot
        # manufacture a 30-second observation from a few wall-clock seconds.
        regular = (
            len(self.samples) >= WINDOW
            and self.worker_seconds >= MIN_SECONDS * self.workers
        )
        active_seconds = 0.0
        end = float("-inf")
        for start, stop in sorted(active):
            active_seconds += max(0.0, stop - max(start, end))
            end = max(end, stop)
        long_work = (
            len(self.samples) >= LONG_WINDOW_SAMPLES
            and work >= MIN_SECONDS
            and active_seconds >= LONG_WINDOW_SECONDS
        )
        if not (regular or long_work):
            return
        rate = len(self.samples) / elapsed
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
                self.epoch += 1
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
        self.epoch += 1
        logging.info(
            "GPU exact queue auto-tune trial | pending=%d previous=%d validations/s=%.3f",
            self.limit,
            self.baseline[0],
            rate,
        )
