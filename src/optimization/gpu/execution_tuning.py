"""Service-owned batch adaptation using successful production work only.

The existing evidence controller supplies smoothing, cold-shape rejection,
cooldowns and rollback. Dataset classes do not share timing evidence; no search
policy, simulations, device imports or checkpoint state live here.
"""

import math
import time

from optimization.gpu.autotune import BatchController


EVIDENCE_SAMPLES = 6
EVIDENCE_SECONDS = 3.0


class ExecutionBatchTuner:
    def __init__(self, *, initial=64, headroom=lambda: True, enabled=True, diagnostics=False):
        self.initial = initial
        self.headroom = headroom
        self.enabled = enabled
        self.controllers = {}
        self._ceilings = {}
        self._demand = {}
        self._allow_trial = False
        self._diagnostic_started = self._diagnostics = None
        if diagnostics:
            try:
                self._diagnostic_started = time.perf_counter()
                self._diagnostics = dict(observations=0, rows=0, replay_seconds=0.0,
                                         cold_observations=0, completed_windows=0,
                                         untrained_observations=0,
                                         last_decision=None, first_trial=None,
                                         latest=None)
            except Exception:
                pass  # Optional host telemetry cannot prevent service startup.

    def diagnostics_snapshot(self):
        """Bounded completed-work context, never input to the controller.

        Elapsed time starts at tuner construction, includes supply gaps, and is
        host wall time. Replay seconds exclude CPU preparation and queue waits.
        The latest class has its own evidence; classes never pool tuner scores.
        """
        value = self._diagnostics
        if value is None:
            return None
        return {name: dict(item) if isinstance(item, dict) else item
                for name, item in value.items()}

    def constrain(self, dataset_id, ceiling):
        """Apply the prepared replay's dispatch bound on its owning worker."""
        if self._ceilings.get(dataset_id) != ceiling:
            # Preparation/retry may change capacity before this replay's
            # completion is observed. Its old allocation-shape evidence expires.
            self.controllers.pop(dataset_id, None)
            self._demand.pop(dataset_id, None)
        self._ceilings[dataset_id] = ceiling

    def width(self, dataset_id, ceiling):
        # Claim one initial request, prepare on its owner, then discover the
        # physical limit. Never hide multiple serial replays behind a cold batch.
        if dataset_id not in self._ceilings:
            return 1
        limit = min(ceiling, self._ceilings.get(dataset_id, ceiling))
        if not self.enabled:
            return min(self.initial, limit)
        controller = self.controllers.get(dataset_id)
        if controller is None or controller.ceiling != limit:
            self._demand[dataset_id] = 0
            controller = BatchController(
                limit, min(self.initial, limit),
                allow_partial_batches=True,
                window_samples=EVIDENCE_SAMPLES,
                min_seconds=EVIDENCE_SECONDS,
                rate_aggregation="per_count",
                growth_limit=lambda: self._demand[dataset_id],
                can_grow=lambda: self.headroom(),
                can_trial=lambda: self._allow_trial,
            )
            self.controllers[dataset_id] = controller
        return controller.width

    def observe(self, dataset_id, count, seconds, *, backlog, closing):
        if not self.enabled:
            return
        controller = self.controllers.get(dataset_id)
        diagnostic = self._diagnostics
        if controller is None:
            if diagnostic is not None:
                self._record_diagnostics(dataset_id, count, seconds, None, None, False,
                                         diagnostic, 0, backlog, closing)
            return  # Capacity changed during replay; next claim creates fresh evidence.
        if diagnostic is not None:
            before = (controller.width, controller.baseline, controller.cooldown,
                      count not in controller.seen)
        # A finite cohort's final replay may have no remaining backlog even
        # when this window repeatedly had enough compatible work for growth.
        if 1 <= count <= controller.width and math.isfinite(seconds) and seconds > 0:
            self._demand[dataset_id] = max(self._demand[dataset_id], count + backlog)
        peak_demand = self._demand[dataset_id] if diagnostic is not None else 0
        self._allow_trial = not closing
        completed_window = controller.observe(count, seconds)
        if completed_window:
            self._demand[dataset_id] = 0
        if diagnostic is not None:
            self._record_diagnostics(dataset_id, count, seconds, controller, before,
                                     bool(completed_window), diagnostic, peak_demand, backlog, closing)

    def _record_diagnostics(self, *args):
        # Keep optional presentation failures outside policy/future delivery.
        try:
            self._observe_diagnostics(*args)
        except Exception:
            self._diagnostics = None

    def _observe_diagnostics(self, dataset_id, count, seconds, controller, before,
                             completed_window, previous, peak_demand, backlog, closing):
        if count < 1 or not math.isfinite(seconds) or seconds <= 0:
            return
        value = dict(previous)
        value["observations"] += 1
        value["rows"] += count
        value["replay_seconds"] += seconds
        if before is None or count > before[0]:
            # A prepared-capacity change can invalidate the controller during
            # this replay. Count its actual work, without inventing evidence.
            value["untrained_observations"] += 1
            value["latest"] = dict(dataset_id=dataset_id, count=count, evidence_available=False,
                                    ceiling=self._ceilings.get(dataset_id), backlog_rows=backlog,
                                    closing=closing)
            self._diagnostics = value
            return
        old_width, old_baseline, old_cooldown, cold = before
        value["cold_observations"] += int(cold)
        if completed_window:
            value["completed_windows"] += 1
            decision = ("trial_selected" if controller.baseline is not None else
                        ("accepted" if controller.width == old_width else "rollback")
                        if old_baseline is not None else
                        "cooldown" if old_cooldown else "window_completed")
            event = dict(kind=decision, elapsed_seconds=time.perf_counter() - self._diagnostic_started,
                         rows=value["rows"], replay_seconds=value["replay_seconds"],
                         dataset_id=dataset_id, peak_demand=peak_demand,
                         previous_width=old_width, width=controller.width,
                         window_rows_per_second=controller.last_window_rate)
            value["last_decision"] = event
            if decision == "trial_selected" and value["first_trial"] is None:
                value["first_trial"] = dict(event)
        value["latest"] = dict(
            dataset_id=dataset_id, dataset_classes=len(self.controllers),
            evidence_available=True,
            count=count, width=controller.width, ceiling=controller.ceiling,
            trial=controller.baseline is not None, retained_samples=len(controller.samples),
            required_samples=controller.window_samples, evidence_seconds=controller.seconds,
            required_seconds=controller.min_seconds, peak_demand=self._demand[dataset_id],
            backlog_rows=backlog, closing=closing,
        )
        # One replacement publishes a coherent snapshot to the CPU reader.
        self._diagnostics = value
