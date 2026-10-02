"""Transient operator progress; never checkpointed or used for selection."""
from collections import deque
import logging
import math
import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock


_work_context = ContextVar("gpu_optimizer_work", default=(None, "gpu_proxy", None))


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
            # Console delivery is optional; exact-result persistence is not.
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
            delivered = log_tokens(f"GPU optimizer progress | gen={generation} phase={self.phase} |", [
                *(f"{key}={value}" for key, value in snapshot.items()),
                f"phase_elapsed={duration(now - self.phase_started)}",
                f"run_elapsed={duration(now - self.started)}",
            ])
            if delivered:
                self.last_log = now


class ExactValidationProgress:
    """Show CPU queue activity without changing durable submission order.

    Estimates use worker-reported completion latency, excluding time a finished
    result waits behind another result or while the coordinator runs a GPU pass.
    An overdue job has an unknown ETA rather than a misleading zero-second ETA.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.pending = {}
        self.latencies = deque(maxlen=24)
        self._lock = RLock()

    def submitted(self, result):
        with self._lock:
            self.pending[result] = self.clock()

    def completed(self, result, *, worker_seconds, queue_seconds):
        with self._lock:
            if result not in self.pending:
                return
            del self.pending[result]
            if (
                math.isfinite(worker_seconds) and worker_seconds > 0
                and math.isfinite(queue_seconds) and queue_seconds >= 0
            ):
                self.latencies.append(worker_seconds + queue_seconds)

    def reset_estimate(self):
        with self._lock:
            self.latencies.clear()

    def snapshot(self):
        with self._lock:
            pending = list(self.pending.items())
            typical = statistics.median(self.latencies) if len(self.latencies) >= 4 else None
        if not pending:
            return {}
        now = self.clock()
        ready = [result.ready() for result, _ in pending]
        age = max(0.0, now - pending[0][1])
        remaining = typical - age if typical is not None else None
        eta = "0s" if ready[0] else (
            duration(remaining) if remaining is not None and remaining >= 1 else "unknown"
        )
        return {
            "exact_unfinished": len(ready) - sum(ready),
            "exact_ready_unrecorded": sum(ready),
            "oldest_exact_age": duration(age),
            "eta_next_exact": eta,
        }


class GenerationMilestone:
    """Scoped estimate from recent finished ask/tell generations only."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = None
        self.completed = deque(maxlen=5)

    def begin(self):
        self.started = self.clock()

    def finish(self):
        if self.started is not None:
            self.completed.append(max(0.0, self.clock() - self.started))
            self.started = None

    def eta(self):
        if self.started is None or not self.completed:
            return "unknown"
        remaining = statistics.median(self.completed) - (self.clock() - self.started)
        # An overrun is new evidence, not a perpetually false zero-second ETA.
        return duration(remaining) if remaining >= 1 else "unknown"


class DriftProgress:
    """Coalesce numeric warning churn without changing the drift decision."""
    def __init__(self, *, rank_halt, constraint_halt, objective_tolerance):
        self.thresholds = (rank_halt, constraint_halt, objective_tolerance)
        self.signature = None
        self.last_log = None

    def update(self, status):
        reason = status["halt_reason"] or status["warn_reason"]
        if reason is None:
            if self.signature is not None:
                if not log_tokens("GPU proxy quality |", ["state=recovered", "action=continue"]):
                    return
            self.signature = self.last_log = None
            return
        state = "halt" if status["halt_reason"] else "warning"
        signature = (state, reason.split(" (", 1)[0])
        log_tokens("GPU proxy quality detail |", [reason], level=logging.DEBUG)
        now = time.monotonic()
        if signature == self.signature and self.last_log is not None and now - self.last_log < 60:
            return
        def number(key):
            value = status[key]
            return f"{value:.6g}" if value == value else "unknown"
        delivered = log_tokens(f"GPU proxy quality | state={state} |", [
            f"reason={signature[1]}",
            f"action={'stop' if state == 'halt' else 'continue_exact_validation'}",
            f"samples={status['samples']}", f"rank_probes={status['probe_rank_samples']}",
            f"rho={number('rho')}", f"probe_rho={number('probe_rho')}",
            f"constraint_agreement={number('constraint_agreement')}",
            f"rank_halt={self.thresholds[0]:.3g}", f"constraint_halt={self.thresholds[1]:.3g}",
            f"objective_tolerance={self.thresholds[2]:.3g}",
        ], level=logging.ERROR if state == "halt" else logging.WARNING)
        if delivered:
            self.signature, self.last_log = signature, now


class SeedBootstrapProgress:
    def __init__(self, total, *, completed=0, workers):
        self.total = total
        self.initial_completed = completed
        self.started = time.monotonic()
        self.last_log = self.started
        logging.info(
            "GPU seed exact start | completed=%d/%d workers=%d",
            completed, total, workers,
        )

    def update(self, completed, *, inflight, queued, force=False):
        now = time.monotonic()
        if not force and now - self.last_log < 60.0:
            return
        elapsed = now - self.started
        newly_completed = completed - self.initial_completed
        eta = (
            duration(max(0, self.total - completed) * elapsed / newly_completed)
            if newly_completed > 0 and elapsed > 0 else "unknown"
        )
        logging.info(
            "GPU seed exact %s | completed=%d/%d inflight=%d queued=%d elapsed=%s eta_seed=%s",
            "complete" if force else "progress", completed, self.total,
            inflight, queued, duration(elapsed), eta,
        )
        self.last_log = now
