"""Run-local CPU result cadence, independent of device and search policy."""

import math


class ResultCadence:
    # A latency target, not a candidate count or a generation policy. Preparation
    # and result servicing alternate within this soft budget; individual atomic
    # CPU operations may take longer. No advisory state belongs in checkpoints.
    budget_seconds = 0.05

    def __init__(self):
        self.limit = 1
        self._cost = None

    def observe(self, count, seconds):
        """Adapt completion grouping to measured CPU work, excluding idle waits."""
        if not count or not math.isfinite(seconds) or seconds <= 0:
            return
        cost = seconds / count
        # React immediately to increased cost; grow cautiously after faster work.
        self._cost = (cost if self._cost is None or cost > self._cost
                      else self._cost * 0.8 + cost * 0.2)
        desired = max(1, min(256, int(self.budget_seconds / self._cost)))
        self.limit = min(desired, self.limit * 2)
