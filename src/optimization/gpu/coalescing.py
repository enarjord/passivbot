"""Run-local queue coalescing from arrival cadence and successful warm work.

Only the service owns this advisory state. No device, candidate scoring, search
policy or checkpoint state belongs here; losing the observations changes timing.
The caller serializes access with its admission/queue condition.
"""

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class _Stream:
    last_arrival: float | None = None
    gaps: deque = field(default_factory=lambda: deque(maxlen=32))
    seen_counts: set[int] = field(default_factory=set)
    replay_seconds: float | None = None


class BatchCoalescer:
    minimum_delay = 0.005
    maximum_delay = 0.5

    def __init__(self):
        self._streams: dict[str, _Stream] = {}

    def arrived(self, dataset_id, now, *, active):
        stream = self._streams.get(dataset_id)
        if stream is None:
            stream = self._streams[dataset_id] = _Stream()
        # Exclude gaps between completed cohorts or unrelated idle periods.
        # Accepted requests in one CPU preparation burst can be nearly adjacent.
        if active and stream.last_arrival is not None and now > stream.last_arrival:
            stream.gaps.append(now - stream.last_arrival)
        stream.last_arrival = now

    def observe(self, dataset_id, count, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            return
        stream = self._streams[dataset_id]
        # Do not learn compilation or first allocation of a replay batch shape.
        if count not in stream.seen_counts:
            stream.seen_counts.add(count)
            return
        stream.replay_seconds = (
            seconds if stream.replay_seconds is None
            else stream.replay_seconds * 0.8 + seconds * 0.2
        )

    def deadline(self, dataset_id, started):
        stream = self._streams[dataset_id]
        if stream.replay_seconds is None or len(stream.gaps) < 3:
            return started + self.minimum_delay
        ceiling = min(self.maximum_delay, max(self.minimum_delay, stream.replay_seconds))
        # A high quantile sees gaps between CPU preparation bursts instead of
        # treating their near-zero intra-burst gaps as the whole arrival cadence.
        gaps = sorted(stream.gaps)
        gap = gaps[math.ceil(len(gaps) * 0.9) - 1]
        idle = min(ceiling, max(self.minimum_delay, gap * 2.0))
        # Keep a hard accumulation bound even while arrivals continue. Stop early
        # after a stream stalls, or when work already waited through another replay.
        return min(started + ceiling, stream.last_arrival + idle)
