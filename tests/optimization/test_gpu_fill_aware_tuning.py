"""Execution tuning compares observed fill shapes without crediting partial tails."""

import pytest

from optimization.gpu.autotune import BatchController
from optimization.gpu.execution_tuning import ExecutionBatchTuner


def policy(initial=16, ceiling=16):
    result = ExecutionBatchTuner(initial=initial, headroom=lambda: True, diagnostics=True)
    result.constrain("market", ceiling)
    result.width("market", ceiling)
    return result


def feed(result, count, seconds, repeats=3, *, cold=False, backlog=16):
    for _ in range(repeats):
        result.observe("market", count, seconds, backlog=backlog, closing=False, cold=cold)


def test_mixed_partial_fills_cannot_make_a_slower_smaller_trial_win():
    result = policy()
    for count in (4, 1, 16):
        feed(result, count, 10, repeats=1)
    controller = result.controllers["market"]
    assert controller.width == 8 and controller.baseline == (16, 1.6)
    decision = result.diagnostics_snapshot()["last_decision"]
    assert decision["window_rows_per_second"] == pytest.approx(21 / 30)
    assert decision["comparison_rows_per_second"] == 1.6
    assert decision["comparison_count"] == 16 and decision["comparison_full_width"]
    # Three mixed observations supplied only one full-width observation. Expose
    # its actual support rather than describing three stable full-width samples.
    assert decision["comparison_samples"] == decision["comparison_frequency"] == 1
    feed(result, 8, 10)
    assert controller.width == 16 and controller.baseline is None
    assert controller.cooldown == 3
    decision = result.diagnostics_snapshot()["last_decision"]
    assert decision["kind"] == "rollback" and decision["comparison_count"] == 8
    assert decision["comparison_samples"] == decision["comparison_frequency"] == 3


def test_partial_only_bounded_cohorts_still_train_and_can_accept_a_faster_trial():
    result = policy()
    feed(result, 3, 12, backlog=0)
    controller = result.controllers["market"]
    assert controller.width == 8 and controller.baseline == (16, .25)
    decision = result.diagnostics_snapshot()["last_decision"]
    assert decision["comparison_count"] == 3 and not decision["comparison_full_width"]
    feed(result, 3, 10, backlog=0)
    assert controller.width == 8 and controller.baseline is None
    decision = result.diagnostics_snapshot()["last_decision"]
    assert decision["kind"] == "accepted" and decision["comparison_count"] == 3
    assert not decision["comparison_full_width"]


def test_cold_full_width_shape_cannot_supply_comparison_evidence():
    result = policy()
    feed(result, 16, 1, repeats=1, cold=True)
    feed(result, 4, 10)
    controller = result.controllers["market"]
    assert controller.last_comparison_count == 4
    assert controller.last_comparison_samples == 3
    assert controller.baseline == (16, .4)
    assert not result.diagnostics_snapshot()["last_decision"]["comparison_full_width"]


def test_largest_observed_shape_retains_within_shape_stall_smoothing():
    result = policy()
    for seconds in (10, 80, 10):
        feed(result, 16, seconds, repeats=1)
    controller = result.controllers["market"]
    assert controller.baseline == (16, 1.6)
    assert controller.last_comparison_samples == 3
    assert controller._durations_by_count == controller._count_frequencies == {}


def test_smaller_execution_trial_must_not_accept_a_small_measured_slowdown():
    result = policy()
    feed(result, 16, 10)
    feed(result, 8, 5.05, repeats=6)
    assert result.controllers["market"].width == 16
    assert result.diagnostics_snapshot()["last_decision"]["kind"] == "rollback"


def test_larger_execution_trial_can_accept_a_small_measured_throughput_gain():
    result = policy(initial=8)
    feed(result, 8, 10)
    feed(result, 16, 19.9)
    assert result.controllers["market"].width == 16
    assert result.diagnostics_snapshot()["last_decision"]["kind"] == "accepted"


def test_equal_execution_throughput_can_keep_a_smaller_trial():
    result = policy()
    feed(result, 16, 10)
    feed(result, 8, 5, repeats=6)
    assert result.controllers["market"].width == 8
    assert result.diagnostics_snapshot()["last_decision"]["kind"] == "accepted"


@pytest.mark.parametrize("accept", [False, True])
def test_tuning_logs_identify_actual_shape_and_support_for_conditional_rates(caplog, accept):
    import logging

    caplog.set_level(logging.INFO)
    result = policy()
    for count in (4, 1, 16):
        feed(result, count, 10, repeats=1)
    trial = " ".join(message for message in caplog.messages
                     if message.startswith("GPU auto-tune trial |"))
    assert "batch=8" in trial and "rate_aggregation=largest_count" in trial
    assert "baseline_comparison_count=16" in trial
    assert "baseline_comparison_samples=1" in trial
    assert "baseline_comparison_frequency=1" in trial
    caplog.clear()
    feed(result, 8, 5 if accept else 10, repeats=6 if accept else 3)
    prefix = "GPU auto-tune accepted |" if accept else "GPU auto-tune retained |"
    decision = " ".join(message for message in caplog.messages if message.startswith(prefix))
    support_prefix = "" if accept else "trial_"
    samples = 6 if accept else 3
    assert "rate_aggregation=largest_count" in decision
    assert f"{support_prefix}comparison_count=8" in decision
    assert f"{support_prefix}comparison_samples={samples}" in decision
    assert f"{support_prefix}comparison_frequency={samples}" in decision


def test_largest_count_evidence_stays_bounded_until_duration_gate():
    controller = BatchController(16, 16, allow_partial_batches=True,
                                 window_samples=6, min_seconds=100,
                                 rate_aggregation="largest_count", can_trial=lambda: False)
    for _ in range(20):
        for count in (1, 16):
            controller.observe(count, 1, cold=False)
    assert len(controller.samples) == 6
    assert {count: len(values) for count, values in controller._durations_by_count.items()} == {1: 6, 16: 6}
    assert controller._count_frequencies == {1: 20, 16: 20}
    for _ in range(30):
        for count in (1, 16):
            controller.observe(count, 1, cold=False)
    assert controller.last_window_rate == 16
    assert controller.last_window_actual_rate == 8.5
    assert controller.last_comparison_count == 16
    assert controller.last_comparison_samples == 6
    assert controller.last_comparison_frequency == 50
    assert controller._durations_by_count == controller._count_frequencies == {}


@pytest.mark.parametrize("field", ["shrink_min_rate_ratio", "growth_min_rate_ratio"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "1", None])
def test_invalid_acceptance_ratio_is_rejected(field, value):
    with pytest.raises(ValueError, match="batch acceptance rate ratios"):
        BatchController(16, 4, **{field: value})


def test_generic_controller_preserves_legacy_acceptance_ratios():
    controller = BatchController(16, 4)
    assert controller.rate_aggregation == "median"
    assert controller.shrink_min_rate_ratio == .98
    assert controller.growth_min_rate_ratio == 1.05
