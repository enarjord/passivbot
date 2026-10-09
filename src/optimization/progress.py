"""Transient operator progress; never checkpointed or used for selection."""
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
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
