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


@pytest.mark.parametrize("demand", [1, 255, 256])
def test_demand_at_or_below_current_width_probes_smaller_without_headroom_query(demand):
    def forbidden():
        pytest.fail("no feasible growth must not query device headroom")

    policy = ExecutionBatchTuner(initial=256, headroom=forbidden)
    policy.constrain("a", 512)
    policy.width("a", 512)
    for _ in range(EVIDENCE_SAMPLES + 1):
        policy.observe("a", demand, 2, backlog=0, closing=False)
    assert policy.width("a", 512) == 128
    assert policy.controllers["a"].ceiling == 512


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
])
def test_cuda_facade_tuning_is_lazy_and_explicit_width_disables_it(setting, mode, automatic, width):
    with CudaBacktestService(batch_size=setting, tuning_mode=mode, max_pending=128) as service:
        assert (service._batch_tuner is not None) is automatic
        assert service._executor.batch_size == width
        assert service._executor._thread is None


def test_cuda_facade_rejects_invalid_tuning_mode():
    with pytest.raises(ValueError, match="tuning mode"):
        CudaBacktestService(tuning_mode="invalid")


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
