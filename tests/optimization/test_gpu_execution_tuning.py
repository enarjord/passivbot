import pytest

from optimization.gpu.execution_tuning import EVIDENCE_SAMPLES, ExecutionBatchTuner
from optimization.gpu.native import CudaBacktestService


def prepared_policy(**kwargs):
    policy = ExecutionBatchTuner(**kwargs)
    for identity in ("a", "b"):
        policy.constrain(identity, 16)
    return policy


def evidence(policy, dataset="a", *, seconds=2, backlog=64, closing=False):
    width = policy.width(dataset, 16)
    controller = policy.controllers[dataset]
    for _ in range(EVIDENCE_SAMPLES + (width not in controller.seen)):
        policy.observe(dataset, width, seconds, backlog=backlog, closing=closing)


def test_execution_window_needs_six_warm_samples_even_after_three_seconds():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    controller = policy.controllers["a"]
    assert controller.window_samples == 6 and controller.min_seconds == 3.0
    policy.observe("a", 4, 1000, backlog=64, closing=False)
    assert not controller.samples and controller.seconds == 0
    for _ in range(5):
        policy.observe("a", 4, 1, backlog=64, closing=False)
    assert controller.width == 4 and controller.seconds == 5
    policy.observe("a", 4, 1, backlog=64, closing=False)
    assert controller.width == 8 and controller.baseline == (4, 4.0)


def test_execution_window_waits_for_three_seconds_after_six_samples():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    controller = policy.controllers["a"]
    policy.observe("a", 4, 1000, backlog=64, closing=False)
    for _ in range(11):
        policy.observe("a", 4, .25, backlog=64, closing=False)
    assert controller.width == 4 and controller.seconds == 2.75
    assert len(controller.samples) == 6
    policy.observe("a", 4, .25, backlog=64, closing=False)
    assert controller.width == 8 and controller.baseline == (4, 16.0)


def test_execution_short_window_keeps_median_smoothing_for_an_isolated_stall():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    policy.observe("a", 4, 1000, backlog=64, closing=False)
    for seconds in (1, 1, 20, 1, 1, 1):
        policy.observe("a", 4, seconds, backlog=64, closing=False)
    assert policy.controllers["a"].baseline == (4, 4.0)


def test_growth_uses_demand_and_headroom_then_rolls_back_slow_trial():
    policy = prepared_policy(initial=4, headroom=lambda: True)
    evidence(policy, backlog=64)
    assert policy.width("a", 16) == 8
    evidence(policy, seconds=5)
    assert policy.width("a", 16) == 4
    assert policy.controllers["a"].cooldown == 3
    other = prepared_policy(initial=4, headroom=lambda: False)
    evidence(other, backlog=64)
    assert other.width("a", 16) == 2  # Probe smaller work; never grow without headroom.


@pytest.mark.parametrize("trial_seconds,retained_width,cooldown", [(3, 500, 1), (5, 256, 3)])
def test_bounded_cohort_can_try_non_power_of_two_growth_and_accept_or_roll_back(
    trial_seconds, retained_width, cooldown,
):
    queries = []
    policy = ExecutionBatchTuner(initial=256, headroom=lambda: queries.append(1) or True)
    policy.constrain("a", 512)
    assert policy.width("a", 512) == 256
    # Four finite 500-request cohorts. The final tail has no backlog; earlier
    # compatible demand in this same evidence window still permits a 500 trial.
    for count in (256, 244) * 4:
        policy.observe("a", count, 2, backlog=244 if count == 256 else 0, closing=False)
    controller = policy.controllers["a"]
    assert controller.width == 500 and controller.ceiling == 512
    assert controller.baseline == (256, 128.0)
    assert queries == [1]
    assert policy._demand["a"] == 0
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", 500, trial_seconds, backlog=0, closing=False)
    assert policy.width("a", 512) == retained_width
    assert controller.baseline is None and controller.cooldown == cooldown
    assert queries == [1]  # Acceptance/rollback does not query headroom again.


@pytest.mark.parametrize("demand, expected_width", [(1, 256), (128, 256), (129, 128), (256, 128)])
def test_demand_at_or_below_current_width_only_probes_a_binding_smaller_width(
    demand, expected_width,
):
    def forbidden():
        pytest.fail("no feasible growth must not query device headroom")

    policy = ExecutionBatchTuner(initial=256, headroom=forbidden)
    policy.constrain("a", 512)
    policy.width("a", 512)
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", demand, 2, backlog=0, closing=False)
    assert policy.width("a", 512) == expected_width
    assert policy.controllers["a"].ceiling == 512


def test_growth_remains_available_after_an_underfilled_window_skips_shrinking():
    policy = ExecutionBatchTuner(initial=256)
    policy.constrain("a", 512)
    assert policy.width("a", 512) == 256
    for _ in range(EVIDENCE_SAMPLES):
        policy.observe("a", 128, 2, backlog=0, closing=False, cold=False)
    assert policy.width("a", 512) == 256
    for _ in range(EVIDENCE_SAMPLES):
        policy.observe("a", 256, 2, backlog=256, closing=False, cold=False)
    assert policy.width("a", 512) == 512


def test_expandable_cuda_producer_can_supply_a_trial_then_retain_the_throughput_plateau():
    policy = prepared_policy(initial=4, expandable_producer=True)
    evidence(policy, seconds=1, backlog=0)
    assert policy.cohort_width == policy.width("a", 16) == 8
    assert policy.accepted_cohort_width == 4
    evidence(policy, seconds=1.98, backlog=0)
    assert policy.cohort_width == policy.width("a", 16) == 4
    assert policy.accepted_cohort_width == 4


def test_expandable_cuda_producer_keeps_a_faster_warm_width():
    policy = prepared_policy(initial=4, expandable_producer=True)
    evidence(policy, seconds=1, backlog=0)
    assert policy.accepted_cohort_width == 4
    evidence(policy, seconds=1.5, backlog=0)
    assert policy.cohort_width == policy.width("a", 16) == 8
    assert policy.accepted_cohort_width == 8
    assert policy.controllers["a"].baseline is None


@pytest.mark.parametrize("max_throughput,expected", [(False, 4096), (True, 2048)])
def test_wide_trial_cannot_persist_on_one_fast_partial_dispatch_in_cuda_max(max_throughput, expected):
    policy = ExecutionBatchTuner(initial=2048, max_throughput=max_throughput, diagnostics=True)
    policy.constrain("a", 8192)
    policy.width("a", 8192)
    for count, seconds in [(2048, 45)] * 3:
        policy.observe("a", count, seconds, backlog=2048, closing=False, cold=False)
    assert policy.width("a", 8192) == 4096
    for count, seconds in [(2500, 49), (1024, 35), (1024, 35)]:
        policy.observe("a", count, seconds, backlog=0, closing=False, cold=False)

    assert policy.width("a", 8192) == expected
    decision = policy.diagnostics_snapshot()["last_decision"]
    assert decision["comparison_count"] == 2500 and decision["comparison_frequency"] == 1
    assert not decision["comparison_full_width"]
    assert decision["window_rows_per_second"] == pytest.approx((2500 + 2048) / 119)
    rate = (2500 + 2048) / 119 if max_throughput else 2500 / 49
    assert decision["comparison_rows_per_second"] == pytest.approx(rate)


@pytest.mark.parametrize("samples,expected", [
    ([(4096, 50), (4096, 54), (2048, 40)], 4096),
    ([(4096, 45), (4096, 45), (256, 120)], 2048),
    ([(4096, 45), (3500, 45), (3500, 45)], 2048),
])
def test_cuda_max_persists_only_repeatable_width_with_faster_representative_rate(samples, expected):
    policy = ExecutionBatchTuner(initial=2048, expandable_producer=True, diagnostics=True)
    policy.constrain("a", 8192)
    policy.width("a", 8192)
    for _ in range(3):
        policy.observe("a", 2048, 45, backlog=0, closing=False, cold=False)
    assert policy.cohort_width == 4096 and policy.accepted_cohort_width == 2048
    # A real compilation or a split must not supply a second full-width sample.
    policy.observe("a", 4096, 1, backlog=0, closing=False, cold=True)
    policy.constrain("a", 8192, split=True)
    policy.observe("a", 4096, 1, backlog=0, closing=False, cold=False)
    for count, seconds in samples:
        policy.observe("a", count, seconds, backlog=0, closing=False, cold=False)

    assert policy.cohort_width == policy.accepted_cohort_width == expected
    decision = policy.diagnostics_snapshot()["last_decision"]
    expected_rate = sum(count for count, _ in samples) / sum(seconds for _, seconds in samples)
    assert decision["comparison_rows_per_second"] == pytest.approx(expected_rate)
    assert decision["comparison_frequency"] == sum(count == 4096 for count, _ in samples)


def test_cuda_max_retries_a_repeatable_intermediate_width_without_persisting_a_partial_trial():
    policy = ExecutionBatchTuner(initial=2048, expandable_producer=True)
    policy.constrain("a", 8192)
    policy.width("a", 8192)
    for _ in range(3):
        policy.observe("a", 2048, 45, backlog=0, closing=False, cold=False)
    assert policy.cohort_width == 4096
    for count in (3000, 2500, 2500):
        policy.observe("a", count, 49, backlog=0, closing=False, cold=False)
    assert policy.cohort_width == policy.accepted_cohort_width == 2048
    for _ in range(12):
        policy.observe("a", 2048, 45, backlog=0, closing=False, cold=False)
    assert policy.cohort_width == 2500 and policy.accepted_cohort_width == 2048
    for _ in range(3):
        policy.observe("a", 2500, 49, backlog=0, closing=False, cold=False)
    assert policy.cohort_width == policy.accepted_cohort_width == 2500


@pytest.mark.parametrize("seconds,trial_width", [(1.0, 2), (8.0, 8)])
def test_cuda_max_grows_beyond_parallel_window_only_for_sustained_slow_replay(seconds, trial_width):
    policy = prepared_policy(initial=4, expandable_producer=True, host_preparation=(4, 0.5))
    evidence(policy, seconds=seconds, backlog=0)
    assert policy.cohort_width == trial_width
    assert policy.accepted_cohort_width == 4


def test_cuda_max_host_growth_uses_full_width_rate_when_partial_batches_are_slow():
    policy = prepared_policy(initial=4, expandable_producer=True, host_preparation=(4, 0.5))
    policy.width("a", 16)
    for count, seconds in [(4, 1)] * 2 + [(2, 8)] * 4:
        policy.observe("a", count, seconds, backlog=0, closing=False, cold=False)
    assert policy.controllers["a"].last_window_rate < 1
    assert policy.controllers["a"].last_comparison_shape_rate == 4
    assert policy.cohort_width == 2
    assert policy.accepted_cohort_width == 4


def test_expandable_cuda_producer_persists_a_physical_clamp_without_a_trial():
    policy = prepared_policy(initial=8, expandable_producer=True)
    policy.constrain("a", 5)
    assert policy.width("a", 16) == 5
    policy.observe("a", 3, 1, backlog=0, closing=False, cold=True)
    assert policy.cohort_width == policy.accepted_cohort_width == 5


@pytest.mark.parametrize("ceiling,expected", [(12, 8), (5, 5)])
def test_cuda_max_restarts_at_accepted_width_when_physical_admission_changes(ceiling, expected):
    policy = prepared_policy(initial=4, expandable_producer=True)
    evidence(policy, seconds=1, backlog=0)
    evidence(policy, seconds=1.5, backlog=0)
    evidence(policy, seconds=1.5, backlog=0)
    evidence(policy, seconds=1.5, backlog=0)
    assert policy.cohort_width == 16 and policy.accepted_cohort_width == 8
    policy.observe("a", 16, 2, backlog=0, closing=False, cold=False)

    policy.constrain("a", ceiling, split=True)
    assert policy.cohort_width == policy.accepted_cohort_width == expected
    assert policy.width("a", 16) == expected
    policy.observe("a", 16, 2, backlog=0, closing=False, cold=False)
    assert not policy.controllers["a"].samples


def test_non_power_of_two_growth_requires_headroom_and_never_changes_other_dataset():
    queries = []
    policy = ExecutionBatchTuner(initial=256, headroom=lambda: queries.append(1) or False)
    for dataset in ("a", "b"):
        policy.constrain(dataset, 512)
        policy.width(dataset, 512)
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", 256, 2, backlog=244, closing=False)
    assert policy.width("a", 512) == 128
    assert policy.width("b", 512) == 256
    assert not policy.controllers["b"].seen
    assert queries == [1]


def test_warm_partial_work_is_actual_count_evidence_and_dataset_owned():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    policy.width("b", 16)
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", 3, 5, backlog=100, closing=False)
    assert policy.controllers["a"].baseline == (4, 3 / 5)
    assert not policy.controllers["b"].samples
    assert policy.width("a", 16) == 8
    assert policy.width("b", 16) == 4


def test_underfilled_cohort_can_probe_smaller_then_reject_a_slow_trial():
    def no_growth_query():
        pytest.fail("insufficient queued demand must not query device headroom")
    policy = ExecutionBatchTuner(initial=64, headroom=no_growth_query)
    policy.constrain("scenario", 128)
    assert policy.width("scenario", 128) == 64
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("scenario", 63, 2, backlog=0, closing=False)
    controller = policy.controllers["scenario"]
    assert controller.baseline == (64, 63 / 2)
    assert controller.width == 32
    # Distinct allocation shapes get distinct cold-use rejection. All warm
    # trial work is slower per actual candidate, so the old width is restored.
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("scenario", 31, 2, backlog=0, closing=False)
    assert controller.width == 64
    assert controller.baseline is None
    assert controller.cooldown == 3


def test_partial_shapes_reject_cold_use_and_invalid_or_oversized_observations():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    controller = policy.controllers["a"]
    for count in (3, 2):
        policy.observe("a", count, 1000, backlog=0, closing=False)
        policy.observe("a", count, 1, backlog=0, closing=False)
    assert list(controller.samples) == [3, 2]
    assert controller.seconds == 2
    assert controller.seen == {2, 3}
    for count, seconds in ((0, 1), (5, 1), (1, 0), (1, -1), (1, float("nan")),
                           (1, float("inf"))):
        policy.observe("a", count, seconds, backlog=0, closing=False)
    assert list(controller.samples) == [3, 2]
    assert controller.seconds == 2
    assert controller.seen == {2, 3}


def test_underfilled_shutdown_records_evidence_without_probing():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", 3, 2, backlog=0, closing=True)
    assert policy.controllers["a"].seconds == 0  # The complete warm window was consumed.
    assert policy.width("a", 16) == 4
    assert policy.controllers["a"].baseline is None


def test_one_candidate_width_cannot_shrink_to_zero_when_growth_is_blocked():
    policy = prepared_policy(initial=1, headroom=lambda: False)
    evidence(policy, backlog=0)
    assert policy.width("a", 16) == 1
    assert policy.controllers["a"].baseline is None


def test_prepared_ceiling_clamps_policy_without_reusing_other_shape_evidence():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    evidence(policy)
    assert policy.width("a", 16) == 8
    policy.constrain("a", 2)
    assert policy.width("a", 16) == 2
    assert not policy.controllers["a"].seen
    assert not policy.controllers["a"].samples
    evidence(policy, backlog=64)
    assert policy.width("a", 16) <= 2


def test_shutdown_records_completed_evidence_without_starting_trials():
    policy = prepared_policy(initial=4)
    evidence(policy, closing=True)
    assert policy.width("a", 16) == 4
    assert policy.controllers["a"].baseline is None


def test_unprepared_dataset_claims_one_request_then_honors_known_capacity():
    for enabled in (False, True):
        policy = ExecutionBatchTuner(initial=64, enabled=enabled)
        assert policy.width("cold", 128) == 1
        assert not policy.controllers
        policy.constrain("cold", 3)
        assert policy.width("cold", 128) == 3
        policy.observe("cold", 1, 5, backlog=100, closing=False)
        assert policy.width("cold", 128) == 3


@pytest.mark.parametrize("setting,mode,automatic,width", [
    (None, "auto", True, 128), ("auto", "refresh", True, 128),
    (None, "off", False, 64), (4, "auto", False, 4),
    (None, "max", True, 128), (4, "max", False, 4),
    (None, "conservative", True, 128),
])
def test_cuda_facade_tuning_is_lazy_and_explicit_width_disables_it(setting, mode, automatic, width):
    with CudaBacktestService(batch_size=setting, tuning_mode=mode, max_pending=128) as service:
        assert (service._batch_tuner is not None) is automatic
        assert service._executor.batch_size == width
        assert service._executor._thread is None
        if automatic:
            tuner = service._batch_tuner
            tuner.constrain("a", width)
            tuner.width("a", width)
            controller = tuner.controllers["a"]
            assert controller.require_repeatable_width is (mode == "max")
            assert controller.rate_aggregation == ("per_count" if mode == "max" else "largest_count")


def test_cuda_facade_rejects_invalid_tuning_mode():
    with pytest.raises(ValueError, match="tuning_mode"):
        CudaBacktestService(tuning_mode="invalid")


@pytest.mark.parametrize("setting,expected", [(None, 32), (64, 64)])
def test_cuda_max_automatic_width_limit_is_independent_of_admission(setting, expected):
    with CudaBacktestService(
            batch_size=setting, tuning_mode="max", max_pending=128,
            auto_request_capacity=192, auto_batch_size_limit=32,
    ) as service:
        assert service._executor.batch_size == expected
        assert service._executor.max_pending == 192
        assert service._executor._accumulate_after_preparation


@pytest.mark.parametrize("mode,delay", [("conservative", None), ("max", 0.005)])
def test_initial_reaccumulation_is_only_enabled_in_max_with_adaptive_delay(mode, delay):
    with CudaBacktestService(tuning_mode=mode, max_batch_delay=delay) as service:
        assert not service._executor._accumulate_after_preparation


@pytest.mark.parametrize("mode,delay,automatic", [
    ("auto", None, True), ("refresh", None, True), ("off", None, False),
    ("auto", 0, False), ("auto", 0.02, False), ("off", 0.02, False),
])
def test_cuda_coalescing_is_independent_of_explicit_width_and_honors_delay(mode, delay, automatic):
    with CudaBacktestService(batch_size=4, tuning_mode=mode, max_batch_delay=delay) as service:
        assert service._batch_tuner is None
        assert (service._executor._coalescing is not None) is automatic
        assert service._executor._thread is None
        assert service._executor.max_batch_delay == (
            None if automatic else 0.005 if delay is None else delay
        )


@pytest.mark.parametrize("trial_seconds,decision,retained_width,elapsed,replay_seconds", [
    (1, "accepted", 8, 91, 21), (5, "rollback", 4, 119, 49),
])
def test_diagnostics_expose_supply_time_and_actual_rows_without_changing_decisions(
    monkeypatch, trial_seconds, decision, retained_width, elapsed, replay_seconds,
):
    from optimization.gpu import execution_tuning as module

    tick = [0.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: tick[0])
    queries = [0, 0]
    def headroom(index):
        queries[index] += 1
        return True
    plain = prepared_policy(initial=4, headroom=lambda: headroom(0))
    observed = prepared_policy(initial=4, headroom=lambda: headroom(1), diagnostics=True)
    for policy in (plain, observed):
        policy.width("a", 16)
    for count, seconds in [(4, 2)] * 7 + [(8, trial_seconds)] * 7:
        tick[0] += seconds + 5  # CPU supply gaps are absent from replay seconds.
        for policy in (plain, observed):
            policy.observe("a", count, seconds, backlog=64, closing=False)
        left, right = plain.controllers["a"], observed.controllers["a"]
        assert (left.width, left.baseline, left.cooldown, left.seconds, left.seen, list(left.samples)) == (
            right.width, right.baseline, right.cooldown, right.seconds, right.seen, list(right.samples))
    assert queries == [1, 1]
    assert plain.diagnostics_snapshot() is None
    snapshot = observed.diagnostics_snapshot()
    assert snapshot["observations"] == 14 and snapshot["rows"] == 84
    assert snapshot["replay_seconds"] == replay_seconds and snapshot["cold_observations"] == 2
    assert snapshot["completed_windows"] == 2
    assert snapshot["first_trial"]["kind"] == "trial_selected"
    assert snapshot["first_trial"]["elapsed_seconds"] == 49
    assert snapshot["first_trial"]["rows"] == 28
    assert snapshot["last_decision"]["kind"] == decision
    assert snapshot["last_decision"]["elapsed_seconds"] == elapsed
    assert snapshot["last_decision"]["rows"] == 84
    assert snapshot["last_decision"]["window_rows_per_second"] == 8 / trial_seconds
    snapshot["latest"]["width"] = -1
    snapshot["last_decision"]["width"] = -1
    assert observed.diagnostics_snapshot()["latest"]["width"] == retained_width
    assert observed.diagnostics_snapshot()["last_decision"]["width"] == retained_width


def test_optional_diagnostic_failure_does_not_replace_policy_or_future_failure(monkeypatch):
    policy = prepared_policy(initial=4, diagnostics=True)
    policy.width("a", 16)
    def unavailable(*_args, **_kwargs):
        raise OSError("observer unavailable")
    monkeypatch.setattr(policy, "_observe_diagnostics", unavailable)
    evidence(policy)
    assert policy.diagnostics_snapshot() is None
    assert policy.width("a", 16) == 8


def test_disabled_tuner_diagnostics_take_no_extra_clocks(monkeypatch):
    from optimization.gpu import execution_tuning as module

    def forbidden():
        pytest.fail("disabled diagnostic clocks must not run")
    monkeypatch.setattr(module.time, "perf_counter", forbidden)
    policy = prepared_policy(initial=4)
    evidence(policy)
    assert policy.width("a", 16) == 8
    assert policy.diagnostics_snapshot() is None


def test_diagnostics_count_completed_rows_when_capacity_change_discards_tuner_evidence():
    policy = prepared_policy(initial=4, diagnostics=True)
    policy.width("a", 16)
    policy.constrain("a", 2)  # A replay may learn a new limit before observation.
    policy.observe("a", 4, 2, backlog=0, closing=False)
    snapshot = policy.diagnostics_snapshot()
    assert snapshot["rows"] == 4 and snapshot["observations"] == 1
    assert snapshot["untrained_observations"] == 1
    assert snapshot["cold_observations"] == snapshot["completed_windows"] == 0
    assert not snapshot["latest"]["evidence_available"]
    assert snapshot["latest"]["ceiling"] == 2
    assert policy.width("a", 16) == 2
    assert not policy.controllers["a"].seen


def test_split_batch_counts_completed_rows_without_training_on_logical_width():
    policy = prepared_policy(initial=4, diagnostics=True)
    policy.width("a", 16)
    controller = policy.controllers["a"]
    policy.constrain("a", 16, split=True)
    policy.observe("a", 4, 2, backlog=0, closing=False, cold=False)
    snapshot = policy.diagnostics_snapshot()
    assert snapshot["rows"] == 4 and snapshot["untrained_observations"] == 1
    assert not controller.samples and policy.controllers["a"] is controller
    policy.observe("a", 4, 2, backlog=0, closing=False, cold=False)
    assert controller.samples and policy.diagnostics_snapshot()["rows"] == 8


@pytest.mark.parametrize("ceiling,expected", [(16, 8), (5, 5)])
def test_transport_change_discards_live_trial_and_restarts_at_accepted_width(ceiling, expected):
    policy = prepared_policy(initial=4, expandable_producer=True, diagnostics=True)
    policy.constrain("a", 16, transport=(False, False))
    evidence(policy, seconds=1, backlog=0)
    evidence(policy, seconds=1.5, backlog=0)
    assert policy.accepted_cohort_width == 8
    # Consume cooldown, then start a trial and retain part of its evidence.
    evidence(policy, seconds=1.5, backlog=0)
    evidence(policy, seconds=1.5, backlog=0)
    controller = policy.controllers["a"]
    assert controller.width == 16 and controller.baseline == (8, 8 / 1.5)
    policy.observe("a", 16, 2, backlog=100, closing=False, cold=False)
    assert controller.samples and policy.cohort_width == 16

    policy.constrain("a", ceiling, transport=(True, False))
    assert policy.cohort_width == policy.accepted_cohort_width == expected
    assert policy.width("a", 16) == expected
    restarted = policy.controllers["a"]
    assert restarted is not controller and restarted.baseline is None
    assert not restarted.samples and not restarted.seen and restarted.cooldown == 0
    policy.observe("a", expected, 2, backlog=100, closing=False, cold=False)
    assert not restarted.samples and not policy.diagnostics_snapshot()["latest"]["evidence_available"]
    policy.constrain("a", ceiling, transport=(True, False))
    policy.observe("a", expected, 2, backlog=100, closing=False, cold=False)
    assert restarted.samples and policy.controllers["a"] is restarted


def test_compact_transport_change_resets_only_the_dataset_that_observes_it():
    policy = prepared_policy(initial=4, expandable_producer=True)
    shared = {"transport": (True, False)}
    for dataset in ("a", "b", "unrelated"):
        policy.constrain(dataset, 16, transport=shared["transport"])
        policy.width(dataset, 16)
        policy.observe(dataset, 4, 2, backlog=0, closing=False, cold=False)
    before = dict(policy.controllers)

    shared["transport"] = (True, True)
    for dataset, still_unchanged in (("a", ("b", "unrelated")), ("b", ("unrelated",))):
        policy.constrain(dataset, 16, transport=shared["transport"])
        assert policy.width(dataset, 16) == 4
        assert policy.controllers[dataset] is not before[dataset]
        for other in still_unchanged:
            assert policy.controllers[other] is before[other] and policy.controllers[other].samples


def test_cuda_facade_reports_transport_from_an_already_registered_sibling(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from optimization.gpu import service as replay_module
    from optimization.gpu.datasets import PreparedGpuDataset

    packed = {"stream_tick_inputs": False, "compact_hourly_ranges": False}

    class Replay:
        data = packed
        last_evaluation_split = False

        def __init__(self, **_kwargs):
            pass

        def evaluate_results(self, candidates):
            if candidates[0].get("pressure"):
                self.data["stream_tick_inputs"] = True
            return candidates

        def _remember_factual_capacities(self):
            pass

    @contextmanager
    def attach(*_args):
        yield None, None, None

    monkeypatch.setattr(replay_module, "SharedAccountReplay", Replay)
    monkeypatch.setattr(CudaBacktestService, "_dispatch_ceiling", staticmethod(lambda replay: 16))
    monkeypatch.setattr(PreparedGpuDataset, "attach", attach)
    monkeypatch.setattr(PreparedGpuDataset, "attach_selection", attach)
    monkeypatch.setattr(PreparedGpuDataset, "execution_config", lambda _dataset: {})
    dataset = object.__new__(PreparedGpuDataset)
    for key, value in {"hlcvs": SimpleNamespace(shape=(1, 1, 4)), "coin_indices": (0,),
                       "markets_json": "{}", "exchange": "binance", "metrics": ()}.items():
        object.__setattr__(dataset, key, value)

    with CudaBacktestService(batch_size=None, tuning_mode="max", max_pending=16) as service:
        service._residency = SimpleNamespace(activate=lambda _replay: None)
        policy = service._batch_tuner
        for identity in ("a", "b"):
            service.register_dataset(identity, dataset)
        # Drive the registered factory callbacks directly with no GPU or wall-clock training.
        with service._executor._factories["a"]() as first, service._executor._factories["b"]() as sibling:
            controllers = dict(policy.controllers)
            first.evaluate([{"pressure": True}])
            assert "a" not in policy.controllers and policy.controllers["b"] is controllers["b"]
            policy.observe("a", 1, 1, backlog=0, closing=False, cold=False)
            sibling.evaluate([{}])
            assert "b" not in policy.controllers
            policy.observe("b", 1, 1, backlog=0, closing=False, cold=False)
            assert all(policy.width(identity, 16) == 16 for identity in ("a", "b"))
