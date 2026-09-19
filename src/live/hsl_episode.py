"""Immutable fill evidence shared by HSL coverage, replay, and live boundaries.

This reconstructs exchange quantities and PnL prefixes, not risk decisions. Rust
remains the owner of drawdown, RED, and restart policy evaluation. Consumers must
build evidence from the authoritative tape before trimming it for price replay.
"""
from dataclasses import dataclass, replace
from bisect import bisect_left, bisect_right
import math

from live.state_refresh import AuthoritativeSurfaceUnavailable


class EpisodeEvidenceUnavailable(AuthoritativeSurfaceUnavailable):
    def __init__(self, cause, *, pside, symbol):
        self.details = {"cause": cause, "pside": pside, "symbol": symbol}
        description = {
            "ambiguous_fill_order_or_values": "fill tape has ambiguous boundaries",
            "missing_opening_fill": "fill tape has ambiguous boundaries: missing opening fill",
            "position_mismatch": "fill tape does not match position",
            "scope_boundaries_unavailable": "fill tape cannot prove episode boundaries",
            "canonical_flatten_replay_unavailable": "canonical replay unavailable for flatten",
            "reset_fill_cohort_ambiguous": "reset fill cohort is ambiguous",
            "reset_fill_cohort_missing_flatten": "reset fill cohort cannot prove flatten",
            "reset_tail_ambiguous": "reset tail is ambiguous",
            "coin_replay_symbol_unavailable": "symbol is unavailable for coin replay",
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
    start_ms: int | None = None
    degraded_reason: str = ""
    degraded_timestamps: tuple[int, ...] = ()
    recovered_from_flat_ms: int | None = None

    @classmethod
    def reconstruct(cls, rows, *, ambiguous=False, epsilon=1e-12, degraded_reason="", degraded_timestamps=()):
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
        return cls(rows, size, tuple(sizes), tuple(flats), tuple(episodes), tuple(prefix), reason, epsilon, degraded_reason=degraded_reason, degraded_timestamps=tuple(degraded_timestamps))

    def recover_closed_prefix(self, current_size, cooldown_ms, *, now_ms=None):
        """Recover a later episode from current quantity and its complete suffix.

        Only a missing opening in an older episode is recoverable here. Walk
        backward without clamping: a reduction ending at zero anchors the later
        tape independently of the corrupt forward prefix. Coverage/freshness
        remain caller requirements. No price or PnL for the old opening is invented.
        """
        if self.unavailable != "missing_opening_fill":
            return self
        size = abs(float(current_size))
        if not math.isfinite(size):
            return self
        next_open = None
        for index in range(len(self.rows) - 1, -1, -1):
            ts, action, qty, _ = self.rows[index]
            before = size - qty if action == "increase" else size + qty
            if before < -self.epsilon:
                return self
            if action == "increase" and before <= self.epsilon:
                next_open = ts
            if (action == "decrease" and size <= self.epsilon and before > self.epsilon
                    and next_open is not None and ts < next_open
                    and ts + cooldown_ms <= next_open):
                # Earlier episodes may own cooldown until this gap. Find the
                # separating gap in one reverse pass, then replay the suffix once.
                suffix = self.reconstruct(self.rows[index + 1:], epsilon=self.epsilon)
                if suffix.required_start(current_size, cooldown_ms, now_ms=now_ms) is not None:
                    return replace(suffix,
                        recovered_from_flat_ms=ts,
                        degraded_reason="position_anchored_episode_suffix",
                        degraded_timestamps=(self.rows[index + 1][0],))
                return self
            size = max(0.0, before)  # arithmetic epsilon only, never an over-close
        return self

    def window(self, start_ms, end_ms):
        """Project proven evidence without assuming a truncated tape starts flat."""
        timestamps = [row[0] for row in self.rows]
        first = 0 if start_ms is None else bisect_left(timestamps, start_ms)
        end = bisect_right(timestamps, end_ms)
        end = max(first, end)
        baseline = self.realized_prefix[first]
        degraded_timestamps = tuple(ts for ts in self.degraded_timestamps
                                    if (start_ms is None or ts >= start_ms) and ts <= end_ms)
        return EpisodeEvidence(
            rows=self.rows[first:end], ending_size=self.sizes[end],
            sizes=self.sizes[first:end + 1],
            flatten_indices=tuple(i - first for i in self.flatten_indices if first <= i < end),
            episodes=tuple((start, flat) for start, flat in self.episodes
                           if start <= end_ms and (flat is None or flat >= (start_ms or 0))),
            realized_prefix=tuple(value - baseline for value in self.realized_prefix[first:end + 1]),
            unavailable=self.unavailable, epsilon=self.epsilon, start_ms=start_ms,
            degraded_reason=self.degraded_reason if degraded_timestamps else "",
            degraded_timestamps=degraded_timestamps,
            recovered_from_flat_ms=self.recovered_from_flat_ms,
        )

    def matches_position(self, size):
        return abs(self.ending_size - abs(size)) <= max(self.epsilon, abs(size) * 1e-12)

    def require_position(self, size, *, pside, symbol):
        reason = self.unavailable
        if reason is None and not self.matches_position(size):
            reason = "position_mismatch"
        if reason is not None:
            raise EpisodeEvidenceUnavailable(reason, pside=pside, symbol=symbol)

    def required_start(self, size, cooldown_ms, *, now_ms=None):
        if self.unavailable or not self.matches_position(size) or not self.episodes:
            return None
        start, flatten = self.episodes[-1]
        if abs(size) <= self.epsilon:
            # A closed episode can still own a RED cooldown. Its proven opening
            # and every cooldown-connected predecessor remain required.
            if now_ms is None or flatten is None or flatten + cooldown_ms < now_ms:
                return None
        elif flatten is not None:
            return None
        for previous_start, flatten in reversed(self.episodes[:-1]):
            if flatten + cooldown_ms <= start:
                break
            start = previous_start
        return start
