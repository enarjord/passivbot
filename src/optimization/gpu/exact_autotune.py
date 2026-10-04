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
    # A nullable inherited count delegates sizing to this GPU controller. If
    # sizing is disabled or host hints fail, one worker is a safe fallback.
    inherited = 1 if inherited is None else inherited
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
        self.allow_trial = lambda: True
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
        if not self.allow_trial():
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


class ExactWorkerController:
    """Propose pool sizes from complete jobs; the caller drains before resizing."""

    def __init__(
        self,
        workers,
        ceiling,
        proxies,
        *,
        per_worker,
        mode="auto",
        context=None,
        hardware=None,
        cache_dir=None,
    ):
        self.workers = self.target = workers
        self.ceiling = max(1, ceiling)
        self.proxies = proxies
        self.per_worker = per_worker
        self.private_worker_peak = 0
        self.cache = CalibrationCache(cache_dir)
        self.key = _digest(
            dict(
                kind="exact_workers_v2",
                hardware=hardware,
                context=context,
                ceiling=self.ceiling,
                worker_memory_class=(per_worker + 128 * MIB - 1) // (128 * MIB),
                workloads=[
                    dict(
                        execution=workload_contract(p.checkpoint_contract),
                        metrics=sorted(p.needed_metrics),
                    )
                    for p in proxies
                ],
            )
        )
        self.samples = deque(maxlen=WINDOW)
        self.epoch = 0
        self.revision = self.proxy_state()[0]
        self.queue_epoch = lambda: None
        self.queue_revision = None
        self.blocked = False
        self.baseline = None
        self.warmed = False
        self.cooldown = 0
        self.direction = 1 if workers < self.ceiling else -1
        self.allow_trial = lambda: True
        if mode != "refresh":
            cached = self.cache.read(self.key, "exact_workers", 1, self.ceiling)
            if cached is not None and (
                cached <= workers or self.can_grow(cached, startup=True)
            ):
                self.workers = self.target = cached
        self.direction = 1 if self.workers < self.ceiling else -1
        self.reset()

    def proxy_state(self):
        revisions, trials = [], False
        for proxy in self.proxies:
            tuner = getattr(proxy, "batch_tuner", None)
            revisions.append(getattr(tuner, "revision", 0))
            trials |= any(
                c.baseline is not None
                for c in getattr(tuner, "controllers", {}).values()
            )
        return tuple(revisions), trials

    def reset(self):
        self.samples.clear()
        self.work_seconds = 0.0

    def observe_worker_memory(self, workers):
        """Refine startup's copy reserve using resident private replay memory.

        Shared candle mappings and inherited libraries are paid once by the host.
        Keep twice the largest observed private footprint plus 256 MiB for replay
        growth; unavailable private accounting retains the startup estimate.
        """
        try:
            private = [psutil.Process(w.pid).memory_full_info().uss for w in workers]
        except (AttributeError, OSError, psutil.Error):
            return
        if private:
            self.private_worker_peak = max(self.private_worker_peak, *private)
            estimate = max(512 * MIB, 2 * self.private_worker_peak + 256 * MIB)
            if estimate != self.per_worker:
                logging.info(
                    "GPU exact worker memory | private_peak_mib=%d estimated_worker_mib=%d",
                    self.private_worker_peak // MIB, estimate // MIB,
                )
            self.per_worker = estimate
            self.recheck_pending_growth()

    def recheck_pending_growth(self):
        """Retire an unapplied increase if current resource headroom no longer fits."""
        if self.target <= self.workers or self.can_grow(self.target):
            return
        self.target = self.workers
        self.baseline = None
        self.epoch += 1
        self.reset()
        self.warmed = False
        self.cooldown = 1
        logging.info(
            "GPU exact worker auto-tune trial retired | workers=%d reason=resource_headroom",
            self.workers,
        )

    def can_grow(self, count, *, startup=False):
        try:
            resources = resource_snapshot()
            available = resources["available"]
            if count > max(1, resources.get("cores", self.ceiling + 1) - 1):
                return False
        except (OSError, psutil.Error):
            return False
        if startup:
            # No existing workers are resident yet. Preserve the same host/GPU
            # reserve as initial sizing when an advisory cache requests more.
            return count * self.per_worker <= int(available * 0.6)
        return available >= max(
            2 * self.per_worker, (count - self.workers + 1) * self.per_worker
        )

    def record(
        self,
        work,
        wait,
        started,
        finished,
        *,
        epoch,
        admission_stall=None,
        queue_epoch=None,
    ):
        if (
            epoch != self.epoch
            or self.target != self.workers
            or queue_epoch != self.queue_epoch()
        ):
            return
        if (
            not all(math.isfinite(v) for v in (work, started, finished))
            or work <= 0
            or finished <= started
        ):
            return
        if admission_stall is not None and not (
            len(admission_stall) == 2
            and all(math.isfinite(v) for v in admission_stall)
            and admission_stall[0] <= admission_stall[1] <= started
        ):
            return
        self.samples.append((work, started, finished, admission_stall))
        self.work_seconds += work

    def update(self):
        revision, gpu_trial = self.proxy_state()
        queue_revision = self.queue_epoch()
        changed = revision != self.revision or queue_revision != self.queue_revision
        blocked = gpu_trial or not self.allow_trial()
        if blocked or changed:
            if not self.blocked or changed:
                self.reset()
                self.epoch += 1
                if changed and self.baseline is not None:
                    # Retire incomparable trials toward the smaller pool;
                    # future growth still requires current memory headroom.
                    self.target = min(self.workers, self.baseline[0])
                    self.baseline = None
                    logging.info(
                        "GPU exact worker auto-tune trial retired | workers=%d reason=%s",
                        self.target,
                        (
                            "queue_epoch_changed"
                            if queue_revision != self.queue_revision
                            else "gpu_revision_changed"
                        ),
                    )
            self.blocked = blocked
            self.revision = revision
            self.queue_revision = queue_revision
            return
        self.blocked = False
        if self.target != self.workers or not self.samples:
            return
        active = sorted((s[1], s[2]) for s in self.samples)
        intervals = sorted(active + [s[3] for s in self.samples if s[3] is not None])
        elapsed, end = 0.0, float("-inf")
        for begin, stop in intervals:
            elapsed += max(0.0, stop - max(begin, end))
            end = max(end, stop)
        median = statistics.median(s[0] for s in self.samples)
        regular = (
            len(self.samples) >= WINDOW
            and self.work_seconds >= MIN_SECONDS * self.workers
        )
        active_seconds, end = 0.0, float("-inf")
        for begin, stop in active:
            active_seconds += max(0.0, stop - max(begin, end))
            end = max(end, stop)
        expensive = (
            len(self.samples) >= max(4, 2 * self.workers)
            and median >= MIN_SECONDS
            and active_seconds >= LONG_WINDOW_SECONDS
        )
        if not (regular or expensive):
            return
        rate = len(self.samples) / elapsed
        self.reset()
        if not self.warmed:
            self.warmed = True
            return
        if self.baseline is not None:
            previous, old_rate = self.baseline
            self.baseline = None
            accepted = rate >= old_rate * (0.98 if self.workers < previous else 1.05)
            if accepted:
                self.cache.write(self.key, "exact_workers", self.workers, rate, elapsed)
                self.cooldown = 1
            else:
                self.target = (
                    previous
                    if previous <= self.workers or self.can_grow(previous)
                    else self.workers
                )
                self.direction *= -1
                self.cooldown = 3
            logging.info(
                "GPU exact worker auto-tune %s | workers=%d validations/s=%.3f previous_validations/s=%.3f",
                "accepted" if accepted else "retained",
                self.target,
                rate,
                old_rate,
            )
        else:
            self.cache.write(self.key, "exact_workers", self.workers, rate, elapsed)
            if self.cooldown:
                self.cooldown -= 1
                return
            trial = max(1, min(self.ceiling, self.workers + self.direction))
            if trial == self.workers:
                self.direction *= -1
                self.cooldown = 1
                return
            if trial > self.workers and not self.can_grow(trial):
                self.direction = -1
                self.cooldown = 1
                return
            self.baseline = (self.workers, rate)
            self.target = trial
            logging.info(
                "GPU exact worker auto-tune trial | workers=%d previous=%d validations/s=%.3f",
                trial,
                self.workers,
                rate,
            )
        if self.target != self.workers:
            self.epoch += 1

    def applied(self):
        self.workers = self.target
        self.revision = self.proxy_state()[0]
        self.queue_revision = self.queue_epoch()
        self.epoch += 1
        self.reset()
        self.warmed = False  # The replacement pool's first window is cold.
