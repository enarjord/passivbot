"""Transient operator progress; never checkpointed or used for selection."""
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
import json
from threading import RLock


_work_context = ContextVar("gpu_optimizer_work", default=(None, "gpu_backtest", None))


def duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def log_tokens(prefix, tokens, *, logger=None, level=logging.INFO):
    """Split optimizer facts into timestamped records without hiding metrics."""
    logger = logger or logging.getLogger()

    def emit(line):
        try:
            logger.log(level, line)
        except Exception:
            # Console delivery is optional; result persistence is not.
            return False
        return True

    prefix = "".join(c if c.isprintable() else "_" for c in prefix)[:120]
    line = prefix
    delivered = True
    for token in tokens:
        token = "".join(c if c.isprintable() else "_" for c in str(token))
        if len(line) + len(token) + 1 > 240 and line != prefix:
            delivered = emit(line) and delivered
            line = prefix
        # Preserve unusually long objective names across bounded records too.
        width = 239 - len(prefix)
        while len(token) > width:
            delivered = emit(prefix + " " + token[:width]) and delivered
            token = token[width:]
        line += " " + token
    return emit(line) and delivered


@contextmanager
def gpu_work_context(generation, phase, callback=None):
    token = _work_context.set((generation, phase, callback))
    try:
        yield
    finally:
        _work_context.reset(token)


def work_scope():
    generation, phase, _ = _work_context.get()
    return (f"gen={generation} " if generation is not None else "") + f"phase={phase}"


def publish_progress():
    callback = _work_context.get()[2]
    if callback is not None:
        try:
            callback()
        except Exception as error:
            log_tokens("GPU progress callback unavailable |", [f"error_type={type(error).__name__}"], level=logging.DEBUG)


class OptimizerProgress:
    """Transient operator summaries; snapshots cannot influence optimization."""
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.started = time.monotonic()
        self.phase_started = self.started
        self.last_log = None
        self.phase = "startup"
        self.rate_baselines = {}
        self._lock = RLock()

    def transition(self, phase):
        with self._lock:
            if phase != self.phase:
                self.phase = phase
                self.phase_started = time.monotonic()
                self.report(force=True)

    def report(self, *, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and self.last_log is not None and now - self.last_log < 60:
                return
            try:
                snapshot = dict(self.snapshot())
                generation = snapshot.pop("gen")
            except Exception as error:
                # A progress snapshot is optional presentation, never evidence.
                generation = "unknown"
                snapshot = dict(snapshot="unavailable", error_type=type(error).__name__)
            rates = []
            for key in ("completed", "screened"):
                value = snapshot.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    baseline = self.rate_baselines.setdefault(key, value)
                    elapsed = now - self.started
                    rate = (value - baseline) / elapsed if elapsed > 0 and value >= baseline else None
                    rates.append(f"{key}/s={rate:.4f}" if rate is not None else f"{key}/s=unknown")
            delivered = log_tokens(f"GPU optimizer progress | gen={generation} phase={self.phase} |", [
                *(f"{key}={value}" for key, value in snapshot.items()),
                *rates,
                f"phase_elapsed={duration(now - self.phase_started)}",
                f"run_elapsed={duration(now - self.started)}",
            ])
            if delivered:
                self.last_log = now


@contextmanager
def search_boundary(generation, operation):
    """Time serial ask/tell/promotion outside evaluation-stage wall spans."""
    if not logging.getLogger().isEnabledFor(logging.INFO):
        yield
        return
    try:
        started = time.perf_counter(), time.thread_time()
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            log_tokens(f"GPU search boundary | gen={generation} operation={operation} |", [
                f"wall_s={time.perf_counter() - started[0]:.6f}",
                f"main_thread_cpu_s={time.thread_time() - started[1]:.6f}",
            ], level=logging.DEBUG)
        except Exception:
            pass  # Optional presentation must preserve the original search error.


class CohortDiagnostics:
    """Bounded presentation of one CPU-owned evaluation stage.

    Main-thread prepare, poll and consume spans are disjoint. Poll wall time
    includes blocking waits; thread CPU time does not. Owner replay/accumulation
    spans overlap this wall time and must never be added to the CPU spans.
    No diagnostic value enters search, checkpoints, cadence or execution policy.
    """

    def __init__(self, *, generation, phase, counts, session):
        self.enabled = logging.getLogger().isEnabledFor(logging.INFO)
        self.generation, self.phase = generation, phase
        self.counts, self.session = counts, session
        self.spans = {name: [0.0, 0.0] for name in ("prepare", "poll", "consume")}
        if self.enabled:
            try:
                self.started = time.perf_counter(), time.thread_time()
                self.before = dict(counts())
                self.transport = self._snapshot(session)
                self.owner = self._snapshot(session.service)
            except Exception:
                self.enabled = False

    @staticmethod
    def _snapshot(owner):
        snapshot = getattr(owner, "diagnostics_snapshot", None)
        if snapshot is None:
            return None
        try:
            return snapshot()
        except Exception:
            return None  # Optional telemetry cannot change a future's failure.

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        try:
            started = time.perf_counter(), time.thread_time()
        except Exception:
            self.enabled = False
            yield
            return
        try:
            yield
        finally:
            try:
                self.spans[name][0] += time.perf_counter() - started[0]
                self.spans[name][1] += time.thread_time() - started[1]
            except Exception:
                self.enabled = False

    @staticmethod
    def _delta(after, before):
        if after is None or before is None:
            return None
        return {name: value - before[name] for name, value in after.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and name in before and isinstance(before[name], (int, float))}

    def finish(self, status):
        if not self.enabled:
            return
        try:
            wall, cpu = time.perf_counter() - self.started[0], time.thread_time() - self.started[1]
            counts = self._delta(dict(self.counts()), self.before)
            transport = self._delta(self._snapshot(self.session), self.transport)
            owner_after = self._snapshot(self.session.service)
            owner = self._delta(owner_after, self.owner)
            attempts = counts["completed"] + counts["screened"]
            rows = None if transport is None else transport["replay_rows"]
            rate = lambda value: f"{value / wall:.4f}" if value is not None and wall > 0 else "unknown"
            prefix = f"GPU cohort | gen={self.generation} phase={self.phase} status={status} |"
            log_tokens(prefix, [f"attempts={attempts}", f"prefilter={counts['prefilter_rejected']}",
                                f"gpu_rows={rows if rows is not None else 'unknown'}", f"wall_s={wall:.3f}",
                                f"attempts/s={rate(attempts)}", f"gpu_rows/s={rate(rows)}"])
            details = [f"full_attempts={counts['completed']}", f"screened_attempts={counts['screened']}",
                       f"main_thread_cpu_s={cpu:.6f}"]
            for name, (seconds, cpu_seconds) in self.spans.items():
                details.extend((f"{name}_wall_s={seconds:.6f}", f"{name}_cpu_s={cpu_seconds:.6f}"))
            details.extend((f"other_wall_s={max(0.0, wall - sum(v[0] for v in self.spans.values())):.6f}",
                            f"other_cpu_s={max(0.0, cpu - sum(v[1] for v in self.spans.values())):.6f}"))
            if transport is None:
                details.append("transport=unavailable")
            else:
                details.extend(f"{name}={value}" for name, value in transport.items())
            if owner is None:
                details.append("owner=unavailable")
            else:
                details.extend(f"overlapping_owner_{name}={value:.6f}" for name, value in owner.items()
                               if name.endswith("seconds"))
                details.extend(f"owner_{name}={value}" for name, value in owner.items()
                               if name in {"batches", "rows"})
            log_tokens(prefix, details, level=logging.DEBUG)
            tuner = None if owner_after is None else owner_after.get("tuner")
            if tuner is not None:
                log_tokens(prefix, ["tuner_since_construction=" + json.dumps(tuner, sort_keys=True)],
                           level=logging.DEBUG)
        except Exception as error:
            log_tokens("GPU cohort diagnostics unavailable |", [f"error_type={type(error).__name__}"],
                       level=logging.DEBUG)
