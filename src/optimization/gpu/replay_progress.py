"""Bounded, scoped context for sequential GPU temporal replay logs.

Context is local to the evaluation call, including nested history-window proxies;
it is never attached to cached runners or persisted in optimizer checkpoints.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
import logging
from optimization.progress import duration, log_tokens, work_scope, publish_progress


_replay_context = ContextVar("gpu_replay_context", default="")
_replay_ids = count(1)
_group_context = ContextVar("gpu_replay_group", default="group=1/1")


def _label(value):
    return "".join(c if c.isprintable() else "_" for c in str(value))[:24].replace(" ", "_")


@contextmanager
def suite_replay_context(*, pass_index, pass_count, labels, exchanges, evaluation_stage):
    names = ",".join(_label(label) for label in labels[:3])
    if len(labels) > 3:
        names += f",+{len(labels) - 3}"
    group_token = _group_context.set(f"group={pass_index}/{pass_count}")
    token = _replay_context.set(
        f"group={pass_index}/{pass_count} scenarios={names} "
        f"exchange={_label(','.join(dict.fromkeys(exchanges)))} stage={_label(evaluation_stage)} "
    )
    try:
        yield
    finally:
        _replay_context.reset(token)
        _group_context.reset(group_token)


class TemporalReplayProgress:
    """Identify every replay so a new candidate batch cannot resemble a reset."""

    def __init__(self, candidates, total_bars, *, history_chunk_bars=None):
        self.replay_id = next(_replay_ids)
        self.context = _replay_context.get()
        self.scope = work_scope() + " " + _group_context.get()
        self.candidates = candidates
        self.total_bars = total_bars
        self.history_chunk_bars = history_chunk_bars
        self.last_info_elapsed = 0.0
        self.log("start", 0, 0.0)

    def log(self, state, completed_bars, elapsed, *, kernel_dispatches=None):
        # Context belongs on the start record; the replay ID correlates compact
        # updates with that record without repeating every scenario name.
        if state == "start":
            log_tokens(f"GPU replay start | {self.scope} replay={self.replay_id} |", [
                *(token for token in self.context.split() if not token.startswith("group=")),
                f"batch_candidates={self.candidates}", f"history_bars={self.total_bars}",
                *([f"history_chunk_bars={self.history_chunk_bars}"] if self.history_chunk_bars else []),
            ])
            publish_progress()
            return
        percent = 100.0 * completed_bars / self.total_bars if self.total_bars else 100.0
        rate = completed_bars / elapsed if elapsed > 0.0 else 0.0
        eta = duration(max(0, self.total_bars - completed_bars) / rate) if rate > 0 else "unknown"
        level = logging.INFO
        if state == "progress":
            if elapsed - self.last_info_elapsed < 60.0:
                level = logging.DEBUG
        delivered = log_tokens(f"GPU replay {state} | {self.scope} replay={self.replay_id} |", [
            f"progress={percent:.1f}%",
            f"bars={completed_bars}/{self.total_bars}", f"elapsed={duration(elapsed)}",
            f"rate={rate:.0f} bars/s", f"eta_batch={eta}",
            *([f"kernel_dispatches={kernel_dispatches}"] if kernel_dispatches is not None else []),
        ], level=level)
        if level == logging.INFO:
            if state == "progress" and delivered:
                self.last_info_elapsed = elapsed
            publish_progress()


def replay_scope():
    return work_scope() + " " + _group_context.get()
