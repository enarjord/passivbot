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
        self.log("start", 0, 0.0)

    def log(self, state, completed_bars, elapsed):
        logging.info(
            "GPU temporal replay %s | %sreplay=%d candidates=%d bars=%d/%d elapsed=%.1fs",
            state, self.context, self.replay_id, self.candidates,
            completed_bars, self.total_bars, elapsed,
        )
