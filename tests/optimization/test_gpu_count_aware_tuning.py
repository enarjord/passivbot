"""Mixed batch counts must not invert execution-width throughput comparisons."""

import pytest

from optimization.gpu.autotune import BatchController
from optimization.gpu.execution_tuning import ExecutionBatchTuner
from tools.gpu_cohort_benchmark import _observe_batches


WIDE = [(1, .1), (255, 1.2)] * 3
NARROW = [(1, .1), (128, 1.0), (127, 1.0)] * 2


def policy(initial):
    result = ExecutionBatchTuner(initial=initial, headroom=lambda: True)
    result.constrain("a", 256)
    result.width("a", 256)
    return result


def window(result, samples):
    controller = result.controllers["a"]
    for count, _seconds in samples:
        if count not in controller.seen:
            result.observe("a", count, 1000, backlog=1024, closing=False)
    for count, seconds in samples:
        result.observe("a", count, seconds, backlog=1024, closing=False)
    assert not controller.samples and controller.seconds == 0
    assert controller._durations_by_count == controller._count_frequencies == {}
    return controller.last_window_rate


def test_slower_smaller_width_is_rejected_despite_larger_unweighted_median():
    result = policy(256)
    assert window(result, WIDE) == pytest.approx(256 / 1.3)
    assert result.width("a", 256) == 128
    assert window(result, NARROW) == pytest.approx(256 / 2.1)
    controller = result.controllers["a"]
    assert controller.width == 256 and controller.baseline is None
    assert controller.cooldown == 3


def test_faster_larger_width_is_accepted_despite_smaller_unweighted_median():
    result = policy(128)
    assert window(result, NARROW) == pytest.approx(256 / 2.1)
    assert result.width("a", 256) == 256
    assert window(result, WIDE) == pytest.approx(256 / 1.3)
    controller = result.controllers["a"]
    assert controller.width == 256 and controller.baseline is None
    assert controller.cooldown == 1


def test_isolated_stalls_are_smoothed_within_the_same_actual_count():
    result = policy(256)
    samples = [(1, .1), (255, 1.2), (1, .1), (255, 50), (1, .1), (255, 1.2)]
    assert window(result, samples) == pytest.approx(256 / 1.3)


def test_each_count_is_weighted_by_its_warm_occurrence_frequency():
    controller = BatchController(256, 256, allow_partial_batches=True,
                                 window_samples=6, min_seconds=0, rate_aggregation="per_count",
                                 can_trial=lambda: False)
    for count in (1, 255):
        controller.observe(count, 1000)
    for count, seconds in [(1, .1)] * 5 + [(255, 1.2)]:
        controller.observe(count, seconds)
    assert controller.last_window_rate == pytest.approx(260 / 1.7)


def test_grouped_samples_are_bounded_until_duration_gate_and_reset_after_consumption():
    controller = BatchController(8, 8, allow_partial_batches=True, window_samples=6,
                                 min_seconds=100, rate_aggregation="per_count", can_trial=lambda: False)
    for count in (1, 2):
        controller.observe(count, 1000)
    for _ in range(20):
        controller.observe(1, 1)
        controller.observe(2, 1)
    assert len(controller.samples) == 6
    assert {count: len(values) for count, values in controller._durations_by_count.items()} == {1: 6, 2: 6}
    assert controller._count_frequencies == {1: 20, 2: 20}
    for _ in range(30):
        controller.observe(1, 1)
        controller.observe(2, 1)
    assert controller.last_window_rate == pytest.approx(1.5)
    assert controller._durations_by_count == controller._count_frequencies == {}
    assert not controller.samples and controller.seconds == 0


def test_invalid_and_cold_samples_cannot_train_or_complete_a_grouped_window():
    result = policy(64)
    controller = result.controllers["a"]
    for count, seconds in [(0, 10), (65, 10), (1, 0), (1, float("nan")), (1, float("inf"))]:
        result.observe("a", count, seconds, backlog=1024, closing=False)
    assert not controller.seen
    for count in (1, 63):
        result.observe("a", count, 1000, backlog=1024, closing=False)
    assert controller._durations_by_count == controller._count_frequencies == {}
    for _ in range(5):
        result.observe("a", 63, 1, backlog=1024, closing=False)
    assert controller.width == 64 and controller.last_window_rate is None
    result.observe("a", 63, 1, backlog=1024, closing=False)
    assert controller.last_window_rate == 63 and controller.width == 128


def test_grouped_evidence_is_dataset_owned_and_new_ceiling_discards_it():
    result = policy(64)
    result.constrain("b", 256)
    result.width("b", 256)
    result.observe("a", 63, 1000, backlog=1024, closing=False)
    result.observe("a", 63, 1, backlog=1024, closing=False)
    assert result.controllers["a"]._count_frequencies == {63: 1}
    assert not result.controllers["b"]._count_frequencies
    result.constrain("a", 32)
    assert result.width("a", 256) == 32
    fresh = result.controllers["a"]
    assert fresh._durations_by_count == fresh._count_frequencies == {}
    assert not fresh.seen and fresh.last_window_rate is None


def test_generic_controller_retains_its_existing_median_policy():
    controller = BatchController(256, 256, allow_partial_batches=True, window_samples=6,
                                 min_seconds=3, can_trial=lambda: False)
    assert controller.rate_aggregation == "median"
    for count in (1, 255):
        controller.observe(count, 1000)
    for count, seconds in WIDE:
        controller.observe(count, seconds)
    assert controller.last_window_rate == pytest.approx(111.25)


def test_cohort_report_exposes_actual_decision_rate_separately_from_global_median():
    result = policy(256)
    evidence = _observe_batches(result, [])
    window(result, WIDE)
    completed = evidence["completed_windows"][0]
    assert completed["median_candidates_per_second"] == pytest.approx(111.25)
    assert completed["candidates_per_second"] == pytest.approx(256 / 1.3)
    assert completed["rate_aggregation"] == "per_count"


@pytest.mark.parametrize("value", ["unknown", None, False, []])
def test_unknown_rate_aggregation_is_rejected(value):
    with pytest.raises(ValueError, match="batch rate aggregation"):
        BatchController(16, 4, rate_aggregation=value)
