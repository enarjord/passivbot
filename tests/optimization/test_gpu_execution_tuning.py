import pytest

from optimization.gpu import autotune
from optimization.gpu.execution_tuning import ExecutionBatchTuner
from optimization.gpu.native import CudaBacktestService


def prepared_policy(**kwargs):
    policy = ExecutionBatchTuner(**kwargs)
    for identity in ("a", "b"):
        policy.constrain(identity, 16)
    return policy


def evidence(policy, dataset="a", *, seconds=2, backlog=64, closing=False):
    width = policy.width(dataset, 16)
    controller = policy.controllers[dataset]
    for _ in range(autotune.WINDOW + (width not in controller.seen)):
        policy.observe(dataset, width, seconds, backlog=backlog, closing=closing)


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


def test_warm_partial_work_is_actual_count_evidence_and_dataset_owned():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    policy.width("b", 16)
    for _ in range(autotune.WINDOW + 1):
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
    for _ in range(autotune.WINDOW + 1):
        policy.observe("scenario", 63, 2, backlog=0, closing=False)
    controller = policy.controllers["scenario"]
    assert controller.baseline == (64, 63 / 2)
    assert controller.width == 32
    # Distinct allocation shapes get distinct cold-use rejection. All warm
    # trial work is slower per actual candidate, so the old width is restored.
    for _ in range(autotune.WINDOW + 1):
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
    for _ in range(autotune.WINDOW + 1):
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
