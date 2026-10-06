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
    evidence(policy, backlog=0)
    assert policy.width("a", 16) == 4
    evidence(policy, backlog=64)
    assert policy.width("a", 16) == 4  # Retain the demand-limited trial cooldown.
    evidence(policy, backlog=64)
    assert policy.width("a", 16) == 8
    evidence(policy, seconds=5)
    assert policy.width("a", 16) == 4
    assert policy.controllers["a"].cooldown == 3
    other = prepared_policy(initial=4, headroom=lambda: False)
    evidence(other, backlog=64)
    assert other.width("a", 16) == 4


def test_evidence_is_dataset_owned_and_partial_work_does_not_tune():
    policy = prepared_policy(initial=4)
    policy.width("a", 16)
    policy.width("b", 16)
    for _ in range(100):
        policy.observe("a", 3, 5, backlog=100, closing=False)
    assert not policy.controllers["a"].samples
    assert not policy.controllers["b"].samples
    evidence(policy, "a")
    assert policy.width("a", 16) == 8
    assert policy.width("b", 16) == 4


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
