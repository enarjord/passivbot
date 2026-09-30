"""Operator progress for exact seed validation; never checkpointed."""
import logging
import time


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
            f"{max(0, self.total - completed) * elapsed / newly_completed:.0f}s"
            if newly_completed > 0 and elapsed > 0 else "unknown"
        )
        logging.info(
            "GPU seed exact %s | completed=%d/%d inflight=%d queued=%d elapsed=%.0fs eta_seed=%s",
            "complete" if force else "progress", completed, self.total,
            inflight, queued, elapsed, eta,
        )
        self.last_log = now
