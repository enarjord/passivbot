"""Bounded execution tuning from completed GPU work only.

This is execution policy, never evolutionary or drift policy. Cache loss changes
performance only; exceptions from evaluation are deliberately not intercepted.
"""

from __future__ import annotations

from collections import deque
from contextvars import ContextVar
import math
import statistics
from optimization.progress import log_tokens

WINDOW = 24
MIN_SECONDS = 30.0
_REPLAY_SAMPLES = ContextVar("gpu_tuning_replay_samples", default=None)


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
    """Bounded hill climbing over completed work, with median smoothing."""

    def __init__(
        self,
        ceiling,
        initial,
        *,
        save=lambda *args: None,
        can_grow=lambda: True,
        can_trial=lambda: True,
        allow_partial_batches=False,
    ):
        self.ceiling = max(1, int(ceiling))
        self.width = max(1, min(int(initial), self.ceiling))
        self.save = save
        self.can_grow = can_grow
        self.can_trial = can_trial
        self.allow_partial_batches = allow_partial_batches
        self.samples = deque(maxlen=WINDOW)
        self.seconds = 0.0
        self.seen = set()
        self.baseline = None
        self.direction = -1 if self.width == self.ceiling else 1
        self.cooldown = 0

    def observe(self, count, seconds, *, evidence_seconds=None):
        """Return true when a complete evidence window has been consumed."""
        # Callers can require full batches. The async service learns actual
        # dispatch shapes, including warm partial request cohorts.
        if (not 1 <= count <= self.width
                or (not self.allow_partial_batches and count != self.width)
                or not math.isfinite(seconds) or seconds <= 0):
            return
        if count not in self.seen:
            self.seen.add(count)
            return
        self.samples.append(count / seconds)
        self.seconds += seconds if evidence_seconds is None else evidence_seconds
        if len(self.samples) < WINDOW or self.seconds < MIN_SECONDS:
            return
        rate = statistics.median(self.samples)
        evidence_seconds = self.seconds
        self.samples.clear()
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
        if trial > self.width and not self.can_grow():
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
        history_size = getattr(runner, "_history_bytes_per_candidate", None)
        history_bytes = history_size() if history_size is not None else 0
        if history_bytes:
            ceiling = min(ceiling, max(1, runner.hsl_scratch_budget_bytes // history_bytes))
    return ceiling


def proxy_batches(proxy, candidates, ceiling, *, end_step=None, clock=None):
    """Bound physical replay histories; service scheduling owns adaptive width.

    Direct replay callers share the same history budget. No hidden tuner,
    calibration replay, hardware cache or search state lives in this iterator.
    """
    width = history_dispatch_ceiling(proxy, ceiling)
    for start in range(0, len(candidates), width):
        yield start, candidates[start:start + width]
