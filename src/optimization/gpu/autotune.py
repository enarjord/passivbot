"""Bounded execution tuning from completed GPU work only.

This is execution policy, never evolutionary or drift policy. Cache loss changes
performance only; exceptions from evaluation are deliberately not intercepted.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import statistics
from optimization.progress import log_tokens

WINDOW = 24
MIN_SECONDS = 30.0
_REPLAY_SAMPLES = ContextVar("gpu_tuning_replay_samples", default=None)
_COMPILATION_EVIDENCE = ContextVar("gpu_compilation_evidence", default=None)


@dataclass
class CompilationEvidence:
    observed: bool = False
    cold: bool = False


@contextmanager
def collect_compilation():
    """Collect actual library cache misses without enabling replay profiling."""
    evidence = CompilationEvidence()
    parent = _COMPILATION_EVIDENCE.get()
    token = _COMPILATION_EVIDENCE.set(evidence)
    try:
        yield evidence
    finally:
        _COMPILATION_EVIDENCE.reset(token)
        if parent is not None:
            parent.observed |= evidence.observed
            parent.cold |= evidence.cold


def record_compilation(cold):
    evidence = _COMPILATION_EVIDENCE.get()
    if evidence is not None:
        evidence.observed = True
        evidence.cold |= bool(cold)


def compilation_cold():
    """Return None when a caller does not supply compilation evidence."""
    evidence = _COMPILATION_EVIDENCE.get()
    return None if evidence is None or not evidence.observed else evidence.cold


class ReplayDurationController:
    """Adjust history chunks from completed work, inside a fixed safety ceiling.

    Duration is an execution target, not a preemption guarantee. Tails and
    invalid timing samples do not train the controller. No extra replay is run.
    """

    def __init__(self, ceiling, *, target_seconds=1.0):
        if type(ceiling) is not int or ceiling < 1:
            raise ValueError("replay chunk ceiling must be a positive integer")
        if not math.isfinite(target_seconds) or target_seconds <= 0:
            raise ValueError("replay duration target must be positive and finite")
        self.ceiling = self.bars = ceiling
        self.target_seconds = target_seconds
        self.fast_chunks = 0

    def observe(self, bars, seconds):
        if bars != self.bars or not math.isfinite(seconds) or seconds <= 0:
            return
        if seconds > self.target_seconds * 1.25:
            self.bars = max(1, int(bars * self.target_seconds * 0.9 / seconds))
            self.fast_chunks = 0
        elif seconds < self.target_seconds * 0.5:
            self.fast_chunks += 1
            if self.fast_chunks >= 3:
                self.bars = min(self.ceiling, self.bars * 2)
                self.fast_chunks = 0
        else:
            self.fast_chunks = 0


class ReplayEvidence(deque):
    def __init__(self):
        super().__init__(maxlen=128)
        self.kernel_seconds = 0.0


def record_replay_chunk(count, bars, total_bars, seconds, *, eligible=True):
    """Observe completed production dispatches; never launch calibration work."""
    samples = _REPLAY_SAMPLES.get()
    if (
        samples is not None and count > 0 and bars > 0 and total_bars > 0
        and math.isfinite(seconds) and seconds > 0
    ):
        samples.kernel_seconds += seconds
        if eligible:
            samples.append((count, seconds * total_bars / bars, seconds))


def is_auto(value):
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


class BatchController:
    """Bounded hill climbing over completed work, with median smoothing.

    An optional ``growth_limit`` callback caps the next doubled trial. It does
    not replace the physical ceiling or the no-argument ``can_grow`` guard.
    A long window can finish with fewer sustained, expensive observations;
    callers that omit it retain the normal sample and duration requirements.
    """

    def __init__(
        self,
        ceiling,
        initial,
        *,
        save=lambda *args: None,
        can_grow=lambda: True,
        growth_limit=None,
        can_trial=lambda: True,
        allow_partial_batches=False,
        window_samples=None,
        min_seconds=None,
        rate_aggregation="median",
        long_window_samples=None,
        long_window_seconds=None,
    ):
        window_samples = WINDOW if window_samples is None else window_samples
        min_seconds = MIN_SECONDS if min_seconds is None else min_seconds
        if type(window_samples) is not int or window_samples < 1:
            raise ValueError("batch evidence samples must be a positive integer")
        if (isinstance(min_seconds, bool) or not isinstance(min_seconds, (int, float))
                or not math.isfinite(min_seconds) or min_seconds < 0):
            raise ValueError("batch evidence duration must be finite and non-negative")
        if not isinstance(rate_aggregation, str) or rate_aggregation not in {"median", "per_count"}:
            raise ValueError("batch rate aggregation must be median or per_count")
        if growth_limit is not None and not callable(growth_limit):
            raise TypeError("batch growth limit must be callable")
        if (long_window_samples is None) != (long_window_seconds is None):
            raise ValueError("long batch evidence requires both samples and duration")
        if long_window_samples is not None:
            if (type(long_window_samples) is not int
                    or not 1 <= long_window_samples <= window_samples):
                raise ValueError("long batch evidence samples must fit the evidence window")
            if (isinstance(long_window_seconds, bool)
                    or not isinstance(long_window_seconds, (int, float))
                    or not math.isfinite(long_window_seconds)
                    or long_window_seconds < min_seconds):
                raise ValueError("long batch evidence duration must be finite and at least the minimum")
        self.ceiling = max(1, int(ceiling))
        self.width = max(1, min(int(initial), self.ceiling))
        self.save = save
        self.can_grow = can_grow
        self.growth_limit = growth_limit
        self.can_trial = can_trial
        self.allow_partial_batches = allow_partial_batches
        self.window_samples = window_samples
        self.min_seconds = min_seconds
        self.rate_aggregation = rate_aggregation
        self.long_window_samples = long_window_samples
        self.long_window_seconds = long_window_seconds
        self.last_window_rate = None
        self.samples = deque(maxlen=window_samples)
        self._replay_durations = deque(maxlen=window_samples)
        # Only execution tuning groups actual integer batch sizes. Frequencies
        # cover the complete window; retained durations stay bounded per count.
        self._durations_by_count = {}
        self._count_frequencies = {}
        self.seconds = 0.0
        self.seen = set()
        self.baseline = None
        self.direction = -1 if self.width == self.ceiling else 1
        self.cooldown = 0

    def observe(self, count, seconds, *, evidence_seconds=None, cold=None):
        """Return true when a complete evidence window has been consumed."""
        # Callers can require full batches. The async service learns actual
        # dispatch shapes, including warm partial request cohorts.
        if (not 1 <= count <= self.width
                or (not self.allow_partial_batches and count != self.width)
                or not math.isfinite(seconds) or seconds <= 0):
            return
        if cold is not None and type(cold) is not bool:
            raise TypeError("batch compilation evidence must be a boolean or None")
        new_count = count not in self.seen
        self.seen.add(count)
        # Explicit evidence identifies actual compiled feature variants. Legacy
        # callers without that evidence retain first-use rejection per count.
        if cold is True or (cold is None and new_count):
            return
        self.samples.append(count / seconds)
        self._replay_durations.append(seconds)
        if self.rate_aggregation == "per_count":
            durations = self._durations_by_count.get(count)
            if durations is None:
                durations = self._durations_by_count[count] = deque(maxlen=self.window_samples)
            durations.append(seconds)
            self._count_frequencies[count] = self._count_frequencies.get(count, 0) + 1
        self.seconds += seconds if evidence_seconds is None else evidence_seconds
        complete = len(self.samples) >= self.window_samples and self.seconds >= self.min_seconds
        long_complete = (self.long_window_samples is not None
                         and len(self.samples) >= self.long_window_samples
                         and self.seconds >= self.long_window_seconds
                         # A lone scheduling stall cannot make an otherwise
                         # fast workload consume its smoothing window early.
                         and statistics.median(self._replay_durations)
                         >= self.long_window_seconds / self.long_window_samples)
        if not (complete or long_complete):
            return
        if self.rate_aggregation == "per_count":
            candidates = sum(count * frequency for count, frequency in self._count_frequencies.items())
            duration = math.fsum(statistics.median(self._durations_by_count[count]) * frequency
                                 for count, frequency in self._count_frequencies.items())
            rate = candidates / duration
        else:
            rate = statistics.median(self.samples)
        self.last_window_rate = rate
        evidence_seconds = self.seconds
        self.samples.clear()
        self._replay_durations.clear()
        self._durations_by_count.clear()
        self._count_frequencies.clear()
        self.seconds = 0.0
        if self.baseline is not None:
            old_width, old_rate = self.baseline
            self.baseline = None
            # Prefer a smaller allocation on a plateau; larger ones must pay off.
            accepted = rate >= old_rate * (0.98 if self.width < old_width else 1.05)
            if accepted:
                log_tokens("GPU auto-tune accepted |", [
                    f"batch={self.width}", f"candidates/s={rate:.3f}",
                    f"previous_candidates/s={old_rate:.3f}",
                    "reason=smaller_plateau" if self.width < old_width else "reason=throughput_gain",
                ])
                self.save(self.width, rate, evidence_seconds)
                self.cooldown = 1
            else:
                log_tokens("GPU auto-tune retained |", [
                    f"batch={old_width}", f"trial_batch={self.width}",
                    f"candidates/s={old_rate:.3f}", f"trial_candidates/s={rate:.3f}",
                    "reason=insufficient_gain",
                ])
                self.width = old_width
                self.direction *= -1
                self.cooldown = 3
            return True
        self.save(self.width, rate, evidence_seconds)
        if self.cooldown:
            self.cooldown -= 1
            return True
        if not self.can_trial():
            return True
        trial = (
            min(self.ceiling, self.width * 2) if self.direction > 0 else max(1, self.width // 2)
        )
        if trial == self.width:
            self.direction *= -1
            self.cooldown = 1
            return True
        if trial > self.width and self.growth_limit is not None:
            # A producer may have useful demand between the current width and
            # its doubled target. Bound that trial without changing the physical
            # ceiling or requiring a full doubled cohort to become available.
            growth_limit = self.growth_limit()
            if type(growth_limit) is not int or growth_limit < 1:
                raise ValueError("batch growth limit must be a positive integer")
            trial = max(self.width, min(trial, growth_limit))
        if self.direction > 0 and (trial == self.width or not self.can_grow()):
            if not self.allow_partial_batches or self.width == 1:
                self.cooldown = 1
                return True
            # A bounded producer may never queue enough work for growth. Try a
            # smaller allocation using subsequent real requests instead.
            self.direction = -1
            trial = max(1, self.width // 2)
        self.baseline = (self.width, rate)
        self.width = trial
        log_tokens("GPU auto-tune trial |", [
            f"batch={trial}", f"previous={self.baseline[0]}",
            f"rolling_candidates/s={rate:.3f}", "reason=throughput_probe",
        ])
        return True


def history_dispatch_ceiling(proxy, ceiling):
    """Bound replay plus reduction history before claiming an outer request batch."""
    single = getattr(proxy, "runner", None)
    fused = getattr(proxy, "fused_runner", None)
    runners = ([single] if single is not None else [fused] if fused is not None
               else list(getattr(proxy, "runners", {}).values()))
    for runner in runners:
        batch_limit = getattr(runner, "_history_batch_limit", None)
        if batch_limit is not None:
            ceiling = min(ceiling, max(1, batch_limit()))
            continue
        history_size = getattr(runner, "_history_bytes_per_candidate", None)
        history_bytes = history_size() if history_size is not None else 0
        if history_bytes:
            ceiling = min(ceiling, max(1, runner.hsl_scratch_budget_bytes // history_bytes))
    return ceiling


def cuda_history_budget(runner, free, total, *, owners=1):
    """Bound one new replay owner's histories and shared reduction workspace.

    Immutable market tensors are already resident when free VRAM is sampled.
    Directional owners share the allowance because both can retain scratch.
    """
    reserve = max(1024**3, int(total * 0.2))
    minimum = runner._history_batch_bytes(1)
    if minimum == 0:
        return runner.hsl_scratch_budget_bytes
    if minimum > free // owners:
        raise MemoryError("CUDA replay history and workspace exceed available VRAM")
    allowance = (free - reserve if free - reserve >= minimum * owners
                 else int(free * 0.35)) // owners
    # This is an allowance, not an allocation. Keep room for factual-history
    # growth within this owner; dispatch demand bounds actual lazy allocation.
    return max(minimum, allowance)


def proxy_batches(proxy, candidates, ceiling, *, end_step=None, clock=None):
    """Bound physical replay histories; service scheduling owns adaptive width.

    Direct replay callers share the same history budget. No hidden tuner,
    calibration replay, hardware cache or search state lives in this iterator.
    """
    width = history_dispatch_ceiling(proxy, ceiling)
    for start in range(0, len(candidates), width):
        yield start, candidates[start:start + width]
