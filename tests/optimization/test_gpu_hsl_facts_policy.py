"""Host-only v9 lifecycle checks: forward state has no fact tape or retries."""

import numpy as np
import pytest

pytest.importorskip("torch")
from optimization.gpu.mps_kernel import MpsEmaAnchorMulticoinRunner


def _runner():
    runner = object.__new__(MpsEmaAnchorMulticoinRunner)
    runner.native_factual_hsl = True
    runner.hsl_capacity = 2
    runner.hsl_scopes = 52
    runner.hsl_fact_capacity = 8192
    runner.hsl_factual_replay = True
    return runner


def test_forward_preparation_owns_no_history_or_reconstruction_cache():
    runner = _runner()
    runner._prepare_native_factual_hsl(np.zeros((1, 1)))
    assert runner.hsl_fact_capacity == 0
    assert not runner.hsl_factual_replay
    assert not runner.dispatch_native_hsl_cache_eligible
    assert not runner.dispatch_native_hsl_coin_cache_eligible
    assert runner._hsl_history_bytes_per_candidate() == 0
    assert runner._hsl_native_cache_capacity() == 0


def test_initial_preparation_never_queries_resources_or_learns_a_history():
    runner = _runner()
    runner._prepare_native_factual_hsl(np.zeros((1, 1)))
    runner._seed_initial_factual_capacity(64)
    assert runner.hsl_fact_capacity == 0
    assert not hasattr(runner, "hsl_fact_capacity_learned")


def test_success_executes_one_attempt_and_discards_retired_tape_controls():
    runner = _runner()
    calls = []
    result = {"identified_fake": "complete producer result"}

    def attempt(params, *, profile, end_steps):
        calls.append((runner.hsl_fact_capacity, profile, end_steps))
        return result

    runner._run_factual_attempt = attempt
    assert runner.run(np.zeros((2, 1)), profile=True) is result
    assert calls == [(0, True, None)]
    assert runner.last_hsl_fact_retries == 0


@pytest.mark.parametrize("error", [RuntimeError("producer rejection"), KeyboardInterrupt()])
def test_failure_and_interrupt_propagate_without_another_attempt(error):
    runner = _runner()
    calls = []

    def attempt(*args, **kwargs):
        calls.append(1)
        raise error

    runner._run_factual_attempt = attempt
    with pytest.raises(type(error)) as caught:
        runner.run(np.zeros((2, 1)))
    assert caught.value is error
    assert calls == [1]


@pytest.mark.parametrize("marker", [-2., -3., -4., -5., -7., -8.])
def test_fatal_native_status_rejects_the_whole_batch(marker):
    import torch
    from optimization.gpu.mps_kernel import _require_available_held_valuation

    scalars = torch.zeros((2, 10))
    scalars[1, 9] = marker
    with pytest.raises((RuntimeError, ValueError)):
        _require_available_held_valuation(scalars)
