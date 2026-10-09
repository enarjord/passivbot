"""Execution-side capacity learning only; attempts below are identified fakes."""

import numpy as np
import pytest

pytest.importorskip("torch")
from optimization.gpu.mps_kernel import HslFactHistoryOverflow, MpsEmaAnchorMulticoinRunner


def _runner(capacity=1):
    runner = object.__new__(MpsEmaAnchorMulticoinRunner)
    runner.max_dispatch_candidate_bars = None
    runner._replay_state_bytes = None
    runner.native_factual_hsl = False
    runner.hsl_fact_capacity = capacity
    runner.hsl_capacity = 64
    runner.hsl_scopes = 2
    runner.unstuck_pnl_capacity = 0
    runner.n = 64
    runner.n_days = 1
    runner.raw_strategy_risk_enabled = runner.raw_strategy_growth_enabled = False
    runner.weighted_equity_metrics = ()
    runner.recovery_distribution_enabled = False
    runner.weighted_volume_enabled = False
    runner.hsl_scratch_budget_bytes = 1024*1024
    runner._hsl_scratch_buffers = {"old allocation": object()}
    runner.interrupt_check = lambda: None
    runner.last_profile = {}
    return runner


def test_native_storage_budget_excludes_legacy_window_and_disabled_storage():
    runner = _runner(256)
    runner.hsl_capacity = 90 * 1440 + 2
    runner.hsl_scopes = 2 * (25 + 1)
    legacy = runner._hsl_history_bytes_per_candidate()
    runner.native_factual_hsl = True
    compact = runner._hsl_history_bytes_per_candidate()
    assert compact == 52 * (2 + 256 + 128) * 32
    assert legacy - compact == 52 * ((4096 + 32401) * 32 + 129602 * 8)
    assert (512 * 1024**2) // legacy == 4
    assert (512 * 1024**2) // compact == 835
    runner.hsl_fact_capacity = 0
    assert runner._hsl_history_bytes_per_candidate() == 0
    runner.native_factual_hsl = False
    assert runner._hsl_history_bytes_per_candidate() == legacy - compact


def test_capacity_learning_repeats_only_rejected_attempts_and_keeps_successful_size():
    runner = _runner()
    attempts = []
    result = {"identified_fake": "complete metrics"}
    def attempt(params, *, profile, end_steps):
        attempts.append(runner.hsl_fact_capacity)
        if runner.hsl_fact_capacity < 4:
            raise HslFactHistoryOverflow("identified fake overflow")
        return result
    runner._run_factual_attempt = attempt
    assert runner.run(np.zeros((2, 1)), profile=True) is result
    assert attempts == [1, 2, 4]
    assert runner.hsl_fact_capacity == 4
    assert runner.last_hsl_fact_retries == 2
    assert runner.last_profile["hsl_fact_retry_count"] == 2
    assert runner.last_profile["hsl_fact_retry_seconds"] > 0
    assert not runner._hsl_scratch_buffers
    assert runner.run(np.zeros((1, 1))) is result
    assert attempts == [1, 2, 4, 4]
    assert runner.last_hsl_fact_retries == 0


def test_one_candidate_budget_exhaustion_rejects_without_another_attempt():
    runner = _runner()
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    attempts = []
    def attempt(*args, **kwargs):
        attempts.append(runner.hsl_fact_capacity)
        raise HslFactHistoryOverflow("identified fake overflow")
    runner._run_factual_attempt = attempt
    with pytest.raises(HslFactHistoryOverflow, match="single-candidate scratch budget"):
        runner.run(np.zeros((1, 1)))
    assert attempts == [1]
    assert runner.hsl_fact_capacity == 1
    assert not hasattr(runner, "hsl_fact_retry_count_total")


def test_malformed_facts_keep_original_failure_and_do_not_trigger_growth():
    runner = _runner()
    original = RuntimeError("GPU HSL malformed factual history")
    def attempt(*args, **kwargs):
        raise original
    runner._run_factual_attempt = attempt
    with pytest.raises(RuntimeError) as caught:
        runner.run(np.zeros((1, 1)))
    assert caught.value is original
    assert runner.hsl_fact_capacity == 1
    assert not hasattr(runner, "hsl_fact_retry_count_total")


def test_interrupt_between_attempts_propagates_before_capacity_or_storage_changes():
    runner = _runner()
    def attempt(*args, **kwargs):
        raise HslFactHistoryOverflow("identified fake overflow")
    def interrupt():
        raise KeyboardInterrupt
    runner._run_factual_attempt = attempt
    runner.interrupt_check = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run(np.zeros((1, 1)))
    assert runner.hsl_fact_capacity == 1
    assert runner._hsl_scratch_buffers


def test_an_overflow_without_enabled_fact_storage_is_an_invariant_failure():
    runner = _runner(0)
    original = HslFactHistoryOverflow("identified impossible overflow")
    def attempt(*args, **kwargs):
        raise original
    runner._run_factual_attempt = attempt
    with pytest.raises(HslFactHistoryOverflow) as caught:
        runner.run(np.zeros((1, 1)))
    assert caught.value is original
    assert runner.hsl_fact_capacity == 0


@pytest.mark.parametrize("fatal_marker", [-2., -3., -4., -5., -7.])
def test_mixed_batch_fatal_error_precedes_capacity_growth(fatal_marker):
    import torch
    from optimization.gpu.mps_kernel import _require_available_held_valuation
    runner = _runner()
    # The attempt is a fake transport; marker admission uses the actual decoder.
    scalars = torch.zeros((2, 10))
    scalars[:, 9] = torch.tensor([-6., fatal_marker])
    calls = []
    def attempt(*args, **kwargs):
        calls.append(runner.hsl_fact_capacity)
        _require_available_held_valuation(scalars)
        raise AssertionError("malformed batch was admitted")
    runner._run_factual_attempt = attempt
    with pytest.raises((RuntimeError, ValueError)) as caught:
        runner.run(np.zeros((2, 1)))
    assert not isinstance(caught.value, HslFactHistoryOverflow)
    assert calls == [1]
    assert runner.hsl_fact_capacity == 1
    assert not hasattr(runner, "hsl_fact_retry_count_total")
