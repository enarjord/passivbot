"""Host-only admission tests: no extension, GPU allocation or replay execution."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from optimization.gpu import mps_kernel as kernel
from optimization.gpu.service import MpsMulticoinProxy


MIB = 1024**2
GIB = 1024**3


def owner(*, device="cuda", width=1, overrides=None):
    value = object.__new__(kernel._MulticoinReplayRunner)
    value.bars = SimpleNamespace(device=SimpleNamespace(type=device))
    value.native_factual_hsl = value.hsl_factual_replay = True
    value.hsl_fact_capacity = value.hsl_fact_capacity_learned = 256
    value.initial_factual_capacity_batch_width = width
    value.hsl_capacity = 1442
    value.pnl_lookback_bars = 1440
    value.hsl_scopes = 2
    value.hsl_replay_sides = value.replay_sides = value.n_coins = 1
    value.n, value.n_days = 3000, 3
    value.hsl_scratch_budget_bytes = 8 * MIB
    value._hsl_scratch_buffers = {}
    value.unstuck_pnl_capacity = 0
    value.max_dispatch_candidate_bars = 500_000_000
    value._replay_state_bytes = 65536
    value._replay_states = {}
    value._replay_state_sizes = {}
    value.raw_strategy_risk_enabled = value.raw_strategy_growth_enabled = False
    value.weighted_equity_metrics = frozenset()
    value.recovery_distribution_enabled = False
    value.entry_interval_enabled = value.hsl_ema_tail_enabled = False
    value.hsl_native_cache_enabled = value.hsl_native_coin_cache_enabled = False
    value.parameter_keys = ("hsl_enabled", "hsl_signal_mode")
    value._coin_hsl_enabled_overrides = (np.array([np.nan] if overrides is None else overrides),)
    value._library_cache_call = lambda: ("library", (bool(value.hsl_fact_capacity),))
    value._history_workspace_bytes = lambda: 32768
    value._disable_native_cache_over_budget = lambda: False
    value.interrupt_check = lambda: None
    return value


def resource_snapshot(monkeypatch, value, *, free=8 * GIB, total=10 * GIB):
    calls = []
    def cuda():
        calls.append("cuda")
        return free, total
    def recommended():
        calls.append("mps")
        return total
    monkeypatch.setattr(kernel.torch.cuda, "mem_get_info", cuda)
    monkeypatch.setattr(kernel.torch.mps, "recommended_max_memory", recommended)
    monkeypatch.setattr(kernel.torch.mps, "driver_allocated_memory", lambda: total - free)
    return calls


def retained_key(value, batch_size=1):
    return (batch_size, value.hsl_fact_capacity, value._hsl_native_cache_bytes_per_candidate())


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_seed_is_input_derived_and_never_becomes_a_learned_hint(monkeypatch, device):
    value = owner(device=device)
    calls = resource_snapshot(monkeypatch, value)
    original_identity = value._library_cache_call()
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 2048  # Public one-day window, not a supplied hint.
    assert value.hsl_fact_capacity_learned == 256
    assert value._library_cache_call() == original_identity
    assert calls == [device]
    proxy = object.__new__(MpsMulticoinProxy)
    proxy._factual_capacity_hints = {}
    proxy.fused_runner = None
    proxy.runners = {"long": value}
    proxy._remember_factual_capacities()
    assert proxy._factual_capacity_hints == {"long": 256}


def test_admission_includes_maximum_width_shared_workspace_and_actual_abi(monkeypatch):
    value = owner(width=8)
    value.hsl_fact_capacity = 512
    value.hsl_scratch_budget_bytes = value._history_batch_bytes(8)
    value.hsl_fact_capacity = 256
    resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)  # One current request must not justify a wider guess.
    assert value.hsl_fact_capacity == 512
    assert value._history_batch_limit() >= 8
    assert value._history_batch_bytes(8) == value.hsl_scratch_budget_bytes
    value.hsl_fact_capacity = 1024
    assert value._history_batch_bytes(8) > value.hsl_scratch_budget_bytes


def test_budget_fallback_performs_no_resource_query(monkeypatch):
    value = owner(width=8)
    value.hsl_scratch_budget_bytes = value._history_batch_bytes(8)
    calls = resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == value.hsl_fact_capacity_learned == 256
    assert calls == []


def test_guess_cannot_change_the_compiled_cache_identity(monkeypatch):
    value = owner()
    value._library_cache_call = lambda: ("library", (value.hsl_fact_capacity > 512,))
    identity = value._library_cache_call()
    resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 512
    assert value._library_cache_call() == identity


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("temporal", [False, True])
def test_warm_active_preparation_retains_buffer_identity_without_resource_work(monkeypatch, device, temporal):
    value = owner(device=device)
    if not temporal:
        value.max_dispatch_candidate_bars = None
    resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)
    buffers = object()
    key = retained_key(value)
    value._hsl_scratch_buffers[key] = buffers
    states = object()
    if temporal:
        value._replay_states[(1, value._replay_state_bytes)] = states
    calls = resource_snapshot(monkeypatch, value, free=1)  # No replacement allocation is required.
    value._prepare_native_factual_hsl(np.array([[1.0, 0.0]], dtype=np.float32))
    assert value._hsl_scratch_buffers[key] is buffers
    value._seed_initial_factual_capacity(1)
    assert value._hsl_scratch_buffers[key] is buffers
    assert value.hsl_fact_capacity == 2048
    assert value.hsl_fact_capacity_learned == 256
    assert calls == []
    assert value._hsl_buffers(1) is buffers  # Exercise the actual allocator reuse boundary.
    if temporal:
        assert value._replay_states[(1, value._replay_state_bytes)] is states


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_off_on_releases_guesses_and_rechecks_pressure(monkeypatch, device):
    value = owner(device=device)
    resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)
    value._hsl_scratch_buffers[retained_key(value)] = object()
    calls = resource_snapshot(monkeypatch, value, free=2 * GIB)
    value._prepare_native_factual_hsl(np.array([[0.0, 0.0]], dtype=np.float32))
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 0 and value._hsl_scratch_buffers == {}
    assert value.hsl_fact_capacity_learned == 256 and calls == []
    value._prepare_native_factual_hsl(np.array([[1.0, 0.0]], dtype=np.float32))
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 256 and calls == [device]


def test_explicit_coin_override_uses_effective_enablement(monkeypatch):
    value = owner(overrides=[1.0])
    calls = resource_snapshot(monkeypatch, value)
    value._prepare_native_factual_hsl(np.array([[0.0, 2.0]], dtype=np.float32))
    value._seed_initial_factual_capacity(1)
    assert value.hsl_factual_replay and value.hsl_fact_capacity == 2048
    assert calls == ["cuda"]
    value._coin_hsl_enabled_overrides = (np.array([0.0]),)
    value._prepare_native_factual_hsl(np.array([[1.0, 2.0]], dtype=np.float32))
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 0 and calls == ["cuda"]


@pytest.mark.parametrize("automatic", [False, True])
def test_service_binding_leaves_auto_unchanged_and_restore_keeps_only_learning(monkeypatch, automatic):
    first = owner()
    resource_snapshot(monkeypatch, first)
    first._seed_initial_factual_capacity(1)
    proxy = object.__new__(MpsMulticoinProxy)
    proxy.run = proxy.data = None
    proxy._factual_capacity_hints = {}
    proxy._automatic_scratch_budget = automatic
    proxy._torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (8 * GIB, 10 * GIB)))
    proxy.shared_account_fused = False
    proxy.sides = ["long"]
    proxy.dispatch_batch_size = 1
    proxy.fused_runner = None
    proxy.runners = {"long": first}
    proxy._remember_factual_capacities()
    restored = owner()
    del restored.initial_factual_capacity_batch_width
    result = proxy._construct_runner("long", lambda *_a, **_kw: restored, {})
    assert result.hsl_fact_capacity_learned == result.hsl_fact_capacity == 256
    assert getattr(result, "initial_factual_capacity_batch_width", 0) == int(not automatic)
    calls = resource_snapshot(monkeypatch, result, free=2 * GIB)
    result._seed_initial_factual_capacity(1)
    assert result.hsl_fact_capacity == 256
    assert calls == ([] if automatic else ["cuda"])
    assert proxy._factual_capacity_hints == {"long": 256}


def test_policy_validation_precedes_new_headroom_query(monkeypatch):
    value = owner()
    calls = resource_snapshot(monkeypatch, value)
    value._pack_params = np.asarray
    value._use_disabled_hsl_specialization = lambda _p: False
    value._unstuck_ema_required = value._unstuck_required = lambda _p: False
    value._prepare_replay_state_layout = lambda *_a: None
    value._validate_hsl_params = lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("malformed policy"))
    monkeypatch.setattr(kernel, "_cached_library_with_miss", lambda *_a: (None, False))
    with pytest.raises(ValueError, match="malformed policy"):
        value._run_factual_attempt(np.array([[1.0, 0.0]], dtype=np.float32))
    assert calls == [] and value.hsl_fact_capacity_learned == 256


@pytest.mark.parametrize("fatal,error", [
    (-2, ValueError), (-4, ValueError), (-3, RuntimeError), (-5, RuntimeError), (-7, RuntimeError),
])
def test_mixed_fatal_and_overflow_remains_fatal_without_learning(monkeypatch, fatal, error):
    value = owner(width=2)
    calls = resource_snapshot(monkeypatch, value)
    statuses = torch.zeros((2, 10), device="cpu")
    statuses[:, 9] = torch.tensor([-6, fatal], device="cpu")
    attempts = []
    def rejected(params, **_kw):
        value._seed_initial_factual_capacity(len(params))
        attempts.append(True)
        kernel._require_available_held_valuation(statuses)
    value._run_factual_attempt = rejected
    with pytest.raises(error):
        value.run(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    assert len(attempts) == 1 and calls == ["cuda"]
    assert value.hsl_fact_capacity == 2048 and value.hsl_fact_capacity_learned == 256
    assert not getattr(value, "hsl_fact_retry_count_total", 0)


def test_real_overflow_can_still_promote_learning(monkeypatch):
    value = owner()
    resource_snapshot(monkeypatch, value)
    attempts = []
    accepted = object()
    def attempt(params, **_kw):
        value._seed_initial_factual_capacity(len(params))
        attempts.append(value.hsl_fact_capacity)
        if len(attempts) == 1:
            assert value.hsl_fact_capacity_learned == 256
            raise kernel.HslFactHistoryOverflow("observed overflow")
        return accepted
    value._run_factual_attempt = attempt
    assert value.run(np.array([[1.0, 0.0]], dtype=np.float32)) is accepted
    assert attempts == [2048, 4096]
    assert value.hsl_fact_capacity_learned == 4096
    assert value.last_hsl_fact_retries == 1


@pytest.mark.parametrize("mode", [0.0, 2.0])
@pytest.mark.parametrize("active_side", [0, 1])
def test_dual_side_enablement_and_all_off_skip_resource_work(monkeypatch, mode, active_side):
    value = owner()
    value.hsl_replay_sides = value.replay_sides = 2
    value.hsl_scopes = 4
    value._coin_hsl_enabled_overrides = (np.array([np.nan]), np.array([np.nan]))
    params = np.array([[0.0, mode, 0.0, mode]], dtype=np.float32)
    calls = resource_snapshot(monkeypatch, value)
    value._prepare_native_factual_hsl(params)
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 0 and calls == []
    params[0, active_side * 2] = 1.0
    value._prepare_native_factual_hsl(params)
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == 2048 and calls == ["cuda"]
    assert value.hsl_fact_capacity_learned == 256


def test_resource_query_failure_propagates_and_does_not_promote_learning(monkeypatch):
    value = owner()
    def unavailable():
        raise RuntimeError("memory producer unavailable")
    monkeypatch.setattr(kernel.torch.cuda, "mem_get_info", unavailable)
    with pytest.raises(RuntimeError, match="memory producer unavailable"):
        value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == value.hsl_fact_capacity_learned == 256


@pytest.mark.parametrize("n,estimate", [(3000, 512), (3000, 1024), (3000, 4096), (64, 128)])
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_restored_useful_estimate_and_accepted_buffers_bypass_all_seed_work(monkeypatch, n, estimate, device):
    value = owner(device=device)
    value.n = n
    value.hsl_capacity = min(n + 2, value.pnl_lookback_bars + 2)
    value.hsl_fact_capacity = value.hsl_fact_capacity_learned = min(256, max(1, n))
    del value.initial_factual_capacity_batch_width
    proxy = object.__new__(MpsMulticoinProxy)
    proxy.run = proxy.data = None
    proxy._factual_capacity_hints = {"long": estimate}
    proxy._automatic_scratch_budget = False
    proxy.dispatch_batch_size = 1
    proxy.fused_runner = None
    proxy.runners = {"long": value}
    restored = proxy._construct_runner("long", lambda *_a, **_kw: value, {})
    assert restored.hsl_fact_capacity == restored.hsl_fact_capacity_learned == estimate
    buffers = object()
    key = retained_key(restored)
    restored._hsl_scratch_buffers[key] = buffers
    calls = resource_snapshot(monkeypatch, restored, free=1)
    def unexpected():
        raise AssertionError("useful learning must bypass seed calculation and admission")
    restored._history_batch_limit = restored._library_cache_call = unexpected
    restored._prepare_native_factual_hsl(np.array([[1.0, 0.0]], dtype=np.float32))
    restored._seed_initial_factual_capacity(1)
    assert restored.hsl_fact_capacity == restored.hsl_fact_capacity_learned == estimate
    assert restored._hsl_scratch_buffers[key] is buffers
    assert calls == []
    proxy._remember_factual_capacities()
    assert proxy._factual_capacity_hints == {"long": estimate}


@pytest.mark.parametrize("inactive", ["unowned", "legacy", "disabled"])
def test_inactive_paths_do_no_seed_admission_or_resource_work(monkeypatch, inactive):
    value = owner()
    if inactive == "unowned":
        del value.initial_factual_capacity_batch_width
    elif inactive == "legacy":
        value.native_factual_hsl = False
    else:
        value.hsl_factual_replay = False
    calls = resource_snapshot(monkeypatch, value)
    del value.pnl_lookback_bars  # None of these paths consumes a seed window.
    def unexpected():
        raise AssertionError("inactive path entered seed admission")
    value._history_batch_limit = value._library_cache_call = unexpected
    value._seed_initial_factual_capacity(1)
    assert value.hsl_fact_capacity == value.hsl_fact_capacity_learned == 256
    assert value._hsl_scratch_buffers == {} and calls == []


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("runner_class", [kernel.MpsEmaAnchorMulticoinRunner, kernel.MpsTrailingMartingaleMulticoinRunner])
def test_changed_compiled_state_abi_rechecks_headroom_before_reusing_guess(monkeypatch, device, runner_class):
    value = object.__new__(runner_class)
    value.__dict__.update(owner(device=device).__dict__)
    value._library_cache_call = lambda: ("library", (bool(value.hsl_fact_capacity),))
    resource_snapshot(monkeypatch, value)
    value._seed_initial_factual_capacity(1)
    factual_key = retained_key(value)
    factual_storage = object()
    value._hsl_scratch_buffers[factual_key] = factual_storage
    state_storage = object()
    value._replay_states[(1, value._replay_state_bytes)] = state_storage
    old_state_bytes = value._replay_state_bytes
    value._prepare_native_factual_hsl(np.array([[1.0, 0.0]], dtype=np.float32))
    assert value._hsl_scratch_buffers[factual_key] is factual_storage

    # This CPU fake is only the compiled size-query transport. The actual layout
    # method handles the ABI change and drops the now-invalid state storage.
    monkeypatch.setattr(kernel, "gpu_device", lambda *_a: "cpu")
    sizes = []
    def compiled_size(output, **_kw):
        sizes.append(True)
        output[0] = old_state_bytes * 2
    library = SimpleNamespace(**{value.replay_state_size_kernel_name: compiled_size})
    value._prepare_replay_state_layout(library, ("unstuck_ema_enabled", True))
    assert sizes == [True] and value._replay_state_bytes == old_state_bytes * 2
    assert value._replay_states == {}
    assert value._hsl_scratch_buffers[factual_key] is factual_storage
    calls = resource_snapshot(monkeypatch, value, free=2 * GIB)
    value._seed_initial_factual_capacity(1)
    assert calls == [device]
    assert value.hsl_fact_capacity == value.hsl_fact_capacity_learned == 256
    assert value._hsl_scratch_buffers == {}
    assert value._replay_states == {}
    assert value._replay_state_bytes == old_state_bytes * 2
