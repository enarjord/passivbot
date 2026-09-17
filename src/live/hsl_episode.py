"""Immutable fill evidence shared by HSL coverage, replay, and live boundaries.

This reconstructs exchange quantities and PnL prefixes, not risk decisions. Rust
remains the owner of drawdown, RED, and restart policy evaluation. Consumers must
build evidence from the authoritative tape before trimming it for price replay.
"""
from dataclasses import dataclass
from bisect import bisect_left, bisect_right

from live.state_refresh import AuthoritativeSurfaceUnavailable


class EpisodeEvidenceUnavailable(AuthoritativeSurfaceUnavailable):
    def __init__(self, cause, *, pside, symbol):
        self.details = {"cause": cause, "pside": pside, "symbol": symbol}
        description = {
            "ambiguous_fill_order_or_values": "fill tape has ambiguous boundaries",
            "missing_opening_fill": "fill tape has ambiguous boundaries: missing opening fill",
            "position_mismatch": "fill tape does not match position",
        }.get(cause, cause)
        super().__init__("hsl_episode_boundaries", description)


@dataclass(frozen=True)
class EpisodeEvidence:
    rows: tuple[tuple[int, str, float, float], ...]
    ending_size: float
    sizes: tuple[float, ...]
    # Indices retain the order of a proven same-millisecond close/re-entry.
    flatten_indices: tuple[int, ...]
    episodes: tuple[tuple[int, int | None], ...]
    realized_prefix: tuple[float, ...]
    unavailable: str | None
    epsilon: float

    @classmethod
    def reconstruct(cls, rows, *, ambiguous=False, epsilon=1e-12):
        rows = tuple(rows)
        size = 0.0
        start = None
        flats, episodes, prefix, sizes = [], [], [0.0], [0.0]
        reason = "ambiguous_fill_order_or_values" if ambiguous else None
        for index, (timestamp, action, qty, delta) in enumerate(rows):
            was_flat = size <= epsilon
            if action == "increase":
                size += qty
                if was_flat and size > epsilon:
                    start = timestamp
            else:
                if qty > size + epsilon:
                    reason = reason or "missing_opening_fill"
                size = max(0.0, size - qty)
                if not was_flat and size <= epsilon:
                    flats.append(index)
                    episodes.append((start, timestamp))
                    start = None
            prefix.append(prefix[-1] + delta)
            sizes.append(size)
        if start is not None:
            episodes.append((start, None))
        return cls(rows, size, tuple(sizes), tuple(flats), tuple(episodes), tuple(prefix), reason, epsilon)

    def window(self, start_ms, end_ms):
        """Project proven evidence without assuming a truncated tape starts flat."""
        timestamps = [row[0] for row in self.rows]
        first = 0 if start_ms is None else bisect_left(timestamps, start_ms)
        end = bisect_right(timestamps, end_ms)
        end = max(first, end)
        baseline = self.realized_prefix[first]
        return EpisodeEvidence(
            rows=self.rows[first:end], ending_size=self.sizes[end],
            sizes=self.sizes[first:end + 1],
            flatten_indices=tuple(i - first for i in self.flatten_indices if first <= i < end),
            episodes=tuple((start, flat) for start, flat in self.episodes
                           if start <= end_ms and (flat is None or flat >= (start_ms or 0))),
            realized_prefix=tuple(value - baseline for value in self.realized_prefix[first:end + 1]),
            unavailable=self.unavailable, epsilon=self.epsilon,
        )

    def matches_position(self, size):
        return abs(self.ending_size - abs(size)) <= max(self.epsilon, abs(size) * 1e-12)

    def require_position(self, size, *, pside, symbol):
        reason = self.unavailable
        if reason is None and not self.matches_position(size):
            reason = "position_mismatch"
        if reason is not None:
            raise EpisodeEvidenceUnavailable(reason, pside=pside, symbol=symbol)

    def required_start(self, size, cooldown_ms):
        if self.unavailable or not self.matches_position(size) or abs(size) <= self.epsilon:
            return None
        if not self.episodes or self.episodes[-1][1] is not None:
            return None
        start = self.episodes[-1][0]
        for previous_start, flatten in reversed(self.episodes[:-1]):
            if flatten + cooldown_ms <= start:
                break
            start = previous_start
        return start
