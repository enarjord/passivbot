"""Completed replay evidence remains usable across counts and long histories."""

from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace
import time

import pytest

from optimization.gpu.autotune import (
    BatchController,
    collect_compilation,
    compilation_cold,
    record_compilation,
)
from optimization.gpu.execution_tuning import ExecutionBatchTuner
from optimization.gpu.executor import BacktestRequest, GpuBacktestService, ReplayResult


def policy(initial=4, ceiling=16, **kwargs):
    result = ExecutionBatchTuner(initial=initial, **kwargs)
    result.constrain("market", ceiling)
    result.width("market", ceiling)
    return result


def observed(result, count, seconds, *, cold=False, backlog=100):
    with collect_compilation():
        record_compilation(cold)
        return result.observe("market", count, seconds, backlog=backlog, closing=False)


def test_warm_new_counts_train_after_actual_cold_compilation_is_excluded():
    result = policy(initial=256, ceiling=512, diagnostics=True)
    observed(result, 75, 200, cold=True, backlog=0)
    controller = result.controllers["market"]
    assert not controller.samples and controller.seconds == 0
    for count, seconds, backlog in ((1, 20, 499), (256, 30, 243)):
        observed(result, count, seconds, backlog=backlog)
        assert controller.width == 256
    observed(result, 243, 40, backlog=0)
    assert controller.width == 500
    assert controller.baseline == (256, 500 / 90)
    snapshot = result.diagnostics_snapshot()
    assert snapshot["cold_observations"] == 1
    assert snapshot["completed_windows"] == 1
    assert snapshot["latest"]["long_required_samples"] == 3
    assert snapshot["latest"]["long_required_seconds"] == 30
    assert snapshot["latest"]["long_required_median_seconds"] == 10
    assert snapshot["latest"]["cold_compilation"] is False


def test_long_gate_requires_three_observations_as_well_as_thirty_seconds():
    result = policy()
    controller = result.controllers["market"]
    observed(result, 4, 100)
    observed(result, 4, 100)
    assert controller.width == 4 and len(controller.samples) == 2
    observed(result, 4, 100)
    assert controller.width == 8 and controller.baseline == (4, .04)

    result = policy()
    controller = result.controllers["market"]
    for seconds in (10, 10, 9):
        observed(result, 4, seconds)
    assert controller.width == 4 and controller.seconds == 29
    observed(result, 4, 10)
    assert controller.width == 8 and controller.baseline == (4, .4)


def test_new_feature_compilation_is_excluded_even_for_a_seen_batch_count():
    result = policy()
    observed(result, 4, 10)
    observed(result, 4, 1000, cold=True)
    controller = result.controllers["market"]
    assert len(controller.samples) == 1 and controller.seconds == 10
    observed(result, 4, 10)
    observed(result, 4, 10)
    assert controller.baseline == (4, .4)


def test_long_growth_still_needs_headroom_and_never_borrows_another_dataset():
    queries = []
    result = policy(headroom=lambda: queries.append(1) or False)
    result.constrain("other", 16)
    result.width("other", 16)
    for _ in range(3):
        observed(result, 4, 10)
    assert result.controllers["market"].width == 2
    assert queries == [1]
    assert result.controllers["other"].width == 4
    assert not result.controllers["other"].samples


def test_short_gate_keeps_six_samples_and_one_stall_is_still_smoothed():
    result = policy()
    for _ in range(5):
        observed(result, 4, .5)
    assert result.controllers["market"].width == 4
    observed(result, 4, .5)
    assert result.controllers["market"].baseline == (4, 8)

    result = policy()
    for seconds in (1, 28, 1):
        observed(result, 4, seconds)
    assert result.controllers["market"].width == 4
    for _ in range(3):
        observed(result, 4, 1)
    assert result.controllers["market"].baseline == (4, 4)


@pytest.mark.parametrize("trial_seconds,width,cooldown", [(15, 8, 1), (30, 4, 3)])
def test_long_trials_keep_acceptance_and_rollback(trial_seconds, width, cooldown):
    result = policy()
    for _ in range(3):
        observed(result, 4, 10)
    assert result.controllers["market"].width == 8
    for _ in range(3):
        observed(result, 8, trial_seconds)
    controller = result.controllers["market"]
    assert controller.width == width
    assert controller.baseline is None and controller.cooldown == cooldown


def test_generic_controller_keeps_sample_gate_and_legacy_count_rejection():
    controller = BatchController(16, 4, window_samples=6, min_seconds=3)
    controller.observe(4, 100)
    assert not controller.samples
    for _ in range(3):
        controller.observe(4, 100)
    assert controller.width == 4 and len(controller.samples) == 3
    assert controller.long_window_samples is controller.long_window_seconds is None


@pytest.mark.parametrize("kwargs", [
    {"long_window_samples": 3},
    {"long_window_seconds": 30},
    {"long_window_samples": 0, "long_window_seconds": 30},
    {"long_window_samples": 7, "long_window_seconds": 30},
    {"long_window_samples": True, "long_window_seconds": 30},
    {"long_window_samples": 3, "long_window_seconds": float("inf")},
    {"long_window_samples": 3, "long_window_seconds": 2},
])
def test_invalid_long_evidence_gate_is_rejected(kwargs):
    with pytest.raises(ValueError, match="long batch evidence"):
        BatchController(16, 4, window_samples=6, min_seconds=3, **kwargs)


def test_compilation_context_aggregates_nested_work_and_resets_on_exception():
    assert compilation_cold() is None
    record_compilation(True)  # No observation exists outside a completed replay.
    with collect_compilation() as first:
        assert compilation_cold() is None and not first.observed
        with collect_compilation():
            record_compilation(True)
            record_compilation(False)
        assert first.observed and first.cold and compilation_cold()
    assert compilation_cold() is None
    with pytest.raises(RuntimeError, match="failed replay"):
        with collect_compilation():
            record_compilation(True)
            raise RuntimeError("failed replay")
    assert compilation_cold() is None
    with collect_compilation():
        assert compilation_cold() is None
        with collect_compilation():
            record_compilation(False)
        assert compilation_cold() is False
        with collect_compilation():
            pass  # Unobserved nested work cannot erase an actual warm report.
        assert compilation_cold() is False


def test_executor_reports_actual_compilation_without_changing_policy_protocol():
    observations, cleanup = [], []

    class Policy:
        def width(self, _dataset, _ceiling):
            return 1

        def observe(self, _dataset, _count, _seconds, *, backlog, closing):
            observations.append(compilation_cold())

    class Replay:
        def evaluate(self, candidates):
            value = candidates[0]["value"]
            record_compilation(value == 1)
            record_compilation(False)
            return [ReplayResult({"value": value}, False)]

    @contextmanager
    def worker():
        try:
            yield
        finally:
            cleanup.append(compilation_cold())

    with GpuBacktestService(batch_size=1, max_batch_delay=0, batch_policy=Policy(),
                            worker_context=worker) as service:
        service.register_dataset("market", Replay())
        for value in (1, 2):
            future = service.submit(BacktestRequest(str(value), "market", {"value": value}))
            assert future.result(timeout=3).metrics == {"value": value}
    assert observations == [True, False]
    assert cleanup == [None]


@pytest.mark.parametrize("failure", ["raise", "invalid"])
def test_failed_replay_cannot_train_and_compilation_context_is_released(failure):
    observations, cleanup = [], []

    class Policy:
        def width(self, _dataset, _ceiling):
            return 1

        def observe(self, *_args, **_kwargs):
            observations.append(compilation_cold())

    class Replay:
        def evaluate(self, _candidates):
            with collect_compilation():
                record_compilation(True)
                if failure == "raise":
                    raise RuntimeError("failed replay")
            return [ReplayResult({}, False)]  # Producer output must validate first.

    @contextmanager
    def worker():
        try:
            yield
        finally:
            cleanup.append(compilation_cold())

    with GpuBacktestService(batch_size=1, max_batch_delay=0, batch_policy=Policy(),
                            worker_context=worker) as service:
        service.register_dataset("market", Replay())
        future = service.submit(BacktestRequest("failed", "market", {}))
        with pytest.raises(RuntimeError, match="failed replay" if failure == "raise" else "non-empty"):
            future.result(timeout=3)
    assert observations == []
    assert cleanup == [None]


def test_cached_library_reports_actual_feature_misses_across_batch_counts(monkeypatch):
    pytest.importorskip("torch")
    from optimization.gpu import mps_kernel

    def forbidden(*_args, **_kwargs):
        pytest.fail("compilation evidence must not add device work or synchronization")

    monkeypatch.setattr(mps_kernel, "synchronize", forbidden)
    monkeypatch.setattr(mps_kernel, "compile_shader", forbidden)

    @lru_cache(maxsize=4)
    def library(feature):
        return object()

    result = policy(diagnostics=True)
    for feature, count, expected_cold in (
        ("first", 4, True), ("first", 3, False), ("second", 3, True),
        ("first", 2, False), ("first", 4, False),
    ):
        with collect_compilation():
            _value, cold = mps_kernel._cached_library_with_miss(library, feature)
            assert cold is expected_cold
            result.observe("market", count, 10, backlog=100, closing=False)
        latest = result.diagnostics_snapshot()["latest"]
        assert latest["cold_compilation"] is expected_cold
        assert latest["cold_observation"] is expected_cold
        assert latest["compilation_evidence_available"] is True
        assert latest["cold_source"] == "reported"
    snapshot = result.diagnostics_snapshot()
    assert snapshot["cold_observations"] == 2
    assert snapshot["completed_windows"] == 1
    assert result.controllers["market"].baseline == (4, 9 / 30)


def test_legacy_diagnostics_distinguish_first_count_rejection_from_reported_compilation():
    result = policy(diagnostics=True)
    result.observe("market", 4, 1, backlog=100, closing=False)
    latest = result.diagnostics_snapshot()["latest"]
    assert latest["cold_compilation"] is None and latest["cold_observation"] is True
    assert latest["compilation_evidence_available"] is False
    assert latest["cold_source"] == "count_history"
    observed(result, 4, 1)
    latest = result.diagnostics_snapshot()["latest"]
    assert latest["cold_compilation"] is False and latest["cold_observation"] is False


@pytest.mark.parametrize("reported", [None, False, True])
def test_capacity_change_retains_reported_compilation_without_inventing_training_evidence(reported):
    result = policy(diagnostics=True)
    result.constrain("market", 2)
    with collect_compilation():
        if reported is not None:
            record_compilation(reported)
        result.observe("market", 2, 10, backlog=100, closing=False)
    latest = result.diagnostics_snapshot()["latest"]
    assert latest["evidence_available"] is False
    assert latest["cold_compilation"] is reported and latest["cold_observation"] is None
    assert latest["compilation_evidence_available"] is (reported is not None)
    assert latest["cold_source"] == ("reported" if reported is not None else "unknown")
    assert result.diagnostics_snapshot()["untrained_observations"] == 1


def test_generic_service_without_compilation_hook_keeps_legacy_rejection_and_valid_futures(monkeypatch):
    import optimization.gpu.executor as executor

    result = policy(initial=1, ceiling=2, diagnostics=True)
    clock = [0.0]
    monkeypatch.setattr(executor, "time", SimpleNamespace(
        perf_counter=lambda: clock[0], monotonic=time.monotonic,
    ))

    class Replay:
        def evaluate(self, candidates):
            assert compilation_cold() is None
            clock[0] += 10
            return [ReplayResult({"value": row["value"]}, False) for row in candidates]

    with GpuBacktestService(batch_size=2, max_batch_delay=0, batch_policy=result) as service:
        service.register_dataset("market", Replay())
        for value in range(3):
            future = service.submit(BacktestRequest(str(value), "market", {"value": value}))
            assert future.result(timeout=3).metrics == {"value": value}
            latest = result.diagnostics_snapshot()["latest"]
            assert latest["cold_compilation"] is None
            assert latest["compilation_evidence_available"] is False
            assert latest["cold_source"] == "count_history"
            assert latest["cold_observation"] is (value == 0)
    controller = result.controllers["market"]
    assert len(controller.samples) == 2 and controller.seconds == 20
    assert controller.baseline is None and controller.width == 1
    assert result.diagnostics_snapshot()["cold_observations"] == 1
    assert compilation_cold() is None
