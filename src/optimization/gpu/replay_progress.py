"""Bounded, scoped context for sequential GPU temporal replay logs.

Context is local to the evaluation call, including nested history-window proxies;
it is never attached to cached runners or persisted in optimizer checkpoints.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
import logging


_replay_context = ContextVar("gpu_replay_context", default="")
_replay_ids = count(1)


def _label(value):
    return "".join(c if c.isprintable() else "_" for c in str(value))[:24].replace(" ", "_")


@contextmanager
def suite_replay_context(*, pass_index, pass_count, labels, exchanges, evaluation_stage):
    names = ",".join(_label(label) for label in labels[:3])
    if len(labels) > 3:
        names += f",+{len(labels) - 3}"
    token = _replay_context.set(
        f"suite_pass={pass_index}/{pass_count} scenarios={names} "
        f"exchange={_label(','.join(dict.fromkeys(exchanges)))} stage={_label(evaluation_stage)} "
    )
    try:
        yield
    finally:
        _replay_context.reset(token)


class TemporalReplayProgress:
    """Identify every replay so a new candidate batch cannot resemble a reset."""

    def __init__(self, candidates, total_bars):
        self.replay_id = next(_replay_ids)
        self.context = _replay_context.get()
        self.candidates = candidates
        self.total_bars = total_bars
        self.last_info_elapsed = 0.0
        self.log("start", 0, 0.0)

    def log(self, state, completed_bars, elapsed):
        # Context belongs on the start record; the replay ID correlates compact
        # updates with that record without repeating every scenario name.
        if state == "start":
            logging.info(
                "GPU replay start | %sreplay=%d candidates=%d bars=%d",
                self.context, self.replay_id, self.candidates, self.total_bars,
            )
            return
        percent = 100.0 * completed_bars / self.total_bars if self.total_bars else 100.0
        rate = completed_bars / elapsed if elapsed > 0.0 else 0.0
        eta = f"{max(0, self.total_bars - completed_bars) / rate:.0f}s" if rate > 0 else "unknown"
        emit = logging.info
        if state == "progress":
            if elapsed - self.last_info_elapsed < 60.0:
                emit = logging.debug
            else:
                self.last_info_elapsed = elapsed
        emit(
            "GPU replay %s | replay=%d progress=%.1f%% bars=%d/%d elapsed=%.0fs rate=%.0f bars/s eta_replay=%s",
            state, self.replay_id, percent, completed_bars, self.total_bars,
            elapsed, rate, eta,
        )
