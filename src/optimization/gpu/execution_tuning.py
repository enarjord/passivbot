"""Service-owned batch adaptation using successful production work only.

The existing evidence controller supplies smoothing, cold-compilation rejection,
cooldowns and rollback. Dataset classes do not share timing evidence; no search
policy, simulations, device imports or checkpoint state live here.
"""

import math
import time

from optimization.gpu.autotune import BatchController, compilation_cold


EVIDENCE_SAMPLES = 6
EVIDENCE_SECONDS = 3.0
LONG_EVIDENCE_SAMPLES = 3
LONG_EVIDENCE_SECONDS = 30.0


class ExecutionBatchTuner:
    def __init__(self, *, initial=64, headroom=lambda: True, enabled=True, diagnostics=False,
                 expandable_producer=False, max_throughput=False):
        self.initial = initial
        self.headroom = headroom
        self.enabled = enabled
        self.controllers = {}
        self._ceilings = {}
        self._transports = {}
        self._restart_widths = {}
        self._demand = {}
        self._split_observation = None
        self._allow_trial = False
        self._expandable_producer = expandable_producer
        self._max_throughput = max_throughput or expandable_producer
        self.probe_inner_growth = False
        # The device owner publishes trial demand separately from the baseline
        # that the optimizer can use to restart with fresh observations.
        self.cohort_width = initial
        self.accepted_cohort_width = initial
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

    def constrain(self, dataset_id, ceiling, *, split=False, transport=None):
        """Apply the prepared replay's dispatch bound on its owning worker."""
        transport_changed = self.enabled and transport is not None and (
            self._transports.get(dataset_id, transport) != transport
        )
        if self.enabled and transport is not None:
            self._transports[dataset_id] = transport
        if transport_changed or self._ceilings.get(dataset_id) != ceiling:
            # Preparation/retry may change capacity before this replay's
            # completion is observed. Its old allocation-shape evidence expires.
            controller = self.controllers.pop(dataset_id, None)
            self._demand.pop(dataset_id, None)
            if controller is not None and (transport_changed or self._max_throughput):
                self._restart_widths[dataset_id] = self._accepted_width(controller)
        self._ceilings[dataset_id] = ceiling
        if dataset_id in self._restart_widths:
            self._restart_widths[dataset_id] = min(ceiling, self._restart_widths[dataset_id])
            if self._expandable_producer:
                self._update_cohort_widths()
        if self.enabled and (split or transport_changed):
            # Transition and split timings cannot train a single replay shape.
            self._split_observation = dataset_id

    @staticmethod
    def _accepted_width(controller):
        return controller.width if controller.baseline is None else controller.baseline[0]

    def _update_cohort_widths(self):
        restarting = max(self._restart_widths.values(), default=0)
        self.cohort_width = max(restarting, max((value.width for value in self.controllers.values()), default=0))
        self.accepted_cohort_width = max(
            restarting, max((self._accepted_width(value) for value in self.controllers.values()), default=0),
        )

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
                limit, min(self._restart_widths.pop(dataset_id, self.initial), limit),
                allow_partial_batches=True,
                window_samples=EVIDENCE_SAMPLES,
                min_seconds=EVIDENCE_SECONDS,
                long_window_samples=LONG_EVIDENCE_SAMPLES,
                long_window_seconds=LONG_EVIDENCE_SECONDS,
                rate_aggregation="per_count" if self._max_throughput else "largest_count",
                shrink_min_rate_ratio=1.0,
                growth_min_rate_ratio=1.03 if self._expandable_producer else 1.0,
                require_repeatable_width=self._max_throughput,
                probe_inner_growth=self.probe_inner_growth,
                growth_limit=lambda: (
                    controller.ceiling
                    if (self._expandable_producer and controller.last_comparison_count == controller.width
                        and controller.last_comparison_frequency >= 2)
                    else self._demand[dataset_id]
                ),
                can_grow=lambda: self.headroom(),
                can_trial=lambda: self._allow_trial,
            )
            self.controllers[dataset_id] = controller
        return controller.width

    def observe(self, dataset_id, count, seconds, *, backlog, closing, cold=None):
        if not self.enabled:
            return
        if cold is None:
            cold = compilation_cold()
        controller = self.controllers.get(dataset_id)
        diagnostic = self._diagnostics
        split = self._split_observation == dataset_id
        if split:
            self._split_observation = None
        if controller is None or split:
            if diagnostic is not None:
                self._record_diagnostics(dataset_id, count, seconds, None, None, False,
                                         diagnostic, 0, backlog, closing, cold)
            return  # Capacity changed during replay; next claim creates fresh evidence.
        if diagnostic is not None:
            before = (controller.width, controller.baseline, controller.cooldown,
                      cold if cold is not None else count not in controller.seen)
        # A finite cohort's final replay may have no remaining backlog even
        # when this window repeatedly had enough compatible work for growth.
        if 1 <= count <= controller.width and math.isfinite(seconds) and seconds > 0:
            self._demand[dataset_id] = max(self._demand[dataset_id], count + backlog)
        peak_demand = self._demand[dataset_id] if diagnostic is not None else 0
        self._allow_trial = not closing
        completed_window = controller.observe(count, seconds, cold=cold)
        if completed_window:
            self._demand[dataset_id] = 0
        if self._expandable_producer:
            self._update_cohort_widths()
        if diagnostic is not None:
            self._record_diagnostics(dataset_id, count, seconds, controller, before,
                                     bool(completed_window), diagnostic, peak_demand, backlog, closing, cold)

    def _record_diagnostics(self, *args):
        # Keep optional presentation failures outside policy/future delivery.
        try:
            self._observe_diagnostics(*args)
        except Exception:
            self._diagnostics = None

    def _observe_diagnostics(self, dataset_id, count, seconds, controller, before,
                             completed_window, previous, peak_demand, backlog, closing,
                             reported_cold=None):
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
                                    cold_observation=None, cold_compilation=reported_cold,
                                    compilation_evidence_available=reported_cold is not None,
                                    cold_source="reported" if reported_cold is not None else "unknown",
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
                         window_rows_per_second=controller.last_window_actual_rate,
                         comparison_rows_per_second=controller.last_window_rate,
                         comparison_count=controller.last_comparison_count,
                         comparison_samples=controller.last_comparison_samples,
                         comparison_frequency=controller.last_comparison_frequency,
                         comparison_full_width=controller.last_comparison_count == old_width,
                         rate_aggregation=controller.rate_aggregation)
            value["last_decision"] = event
            if decision == "trial_selected" and value["first_trial"] is None:
                value["first_trial"] = dict(event)
        value["latest"] = dict(
            dataset_id=dataset_id, dataset_classes=len(self.controllers),
            evidence_available=True,
            cold_observation=cold, cold_compilation=reported_cold,
            compilation_evidence_available=reported_cold is not None,
            cold_source="reported" if reported_cold is not None else "count_history",
            count=count, width=controller.width, ceiling=controller.ceiling,
            trial=controller.baseline is not None, retained_samples=len(controller.samples),
            required_samples=controller.window_samples, evidence_seconds=controller.seconds,
            required_seconds=controller.min_seconds, peak_demand=self._demand[dataset_id],
            long_required_samples=controller.long_window_samples,
            long_required_seconds=controller.long_window_seconds,
            long_required_median_seconds=controller.long_window_seconds / controller.long_window_samples,
            backlog_rows=backlog, closing=closing,
        )
        # One replacement publishes a coherent snapshot to the CPU reader.
        self._diagnostics = value
