"""Disposable coin history arithmetic preserves independent factual replay."""
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def host_runner(device="mps", coins=2, sides=2):
    pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner
    runner = object.__new__(MpsTrailingMartingaleMulticoinRunner)
    runner.bars = SimpleNamespace(device=SimpleNamespace(type=device))
    runner.native_factual_hsl = True
    runner.hsl_capacity = 1442
    runner.hsl_fact_capacity = runner.hsl_fact_capacity_learned = 32
    runner.hsl_replay_sides, runner.n_coins = sides, coins
    runner._coin_hsl_enabled_overrides = tuple(np.full(coins, np.nan) for _ in range(sides))
    runner._hsl_scratch_buffers = {}
    return runner


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_coin_cache_has_independent_budgeted_owners(device):
    from optimization.gpu.mps_kernel import _hsl_native_cache_bytes
    runner = host_runner(device)
    keys = runner.parameter_keys
    matrix = np.zeros((3, len(keys)*2), dtype=np.float32)
    for side in range(2):
        matrix[:, side*len(keys)+keys.index("hsl_enabled")] = 1
        matrix[:, side*len(keys)+keys.index("hsl_signal_mode")] = 2
    runner._prepare_native_factual_hsl(matrix)
    capacity = runner._hsl_native_cache_capacity()
    assert capacity > runner.hsl_capacity and capacity % 64 == 0
    expected = 4 * _hsl_native_cache_bytes(capacity, 1, 32)
    assert runner._hsl_native_cache_bytes_per_candidate() == expected
    runner.hsl_native_cache_budget_bytes = expected - 1
    assert runner._hsl_native_cache_capacity() == 0
    runner.hsl_native_cache_budget_bytes = expected
    assert runner._hsl_native_cache_capacity() == capacity
    runner.hsl_native_coin_cache_enabled = False
    assert runner._hsl_native_cache_capacity() == 0


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_coin_cache_declines_mixed_modes_and_inactive_policies(mode):
    runner = host_runner()
    keys = runner.parameter_keys
    matrix = np.zeros((2, len(keys)*2), dtype=np.float32)
    for side in range(2):
        matrix[:, side*len(keys)+keys.index("hsl_enabled")] = 1
        matrix[:, side*len(keys)+keys.index("hsl_signal_mode")] = mode
    runner._prepare_native_factual_hsl(matrix)
    assert bool(runner._hsl_native_cache_capacity()) is (mode == 2)
    matrix[0, keys.index("hsl_signal_mode")] = 0
    runner._prepare_native_factual_hsl(matrix)
    assert runner._hsl_native_cache_capacity() == 0
    matrix[:] = 0
    runner._prepare_native_factual_hsl(matrix)
    assert runner.hsl_fact_capacity == 0 and runner._hsl_native_cache_capacity() == 0


@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("scaled", [False, True])
def test_coin_cache_preserves_real_native_hsl_and_temporal_boundaries(monkeypatch, sides, coins, scaled):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("Metal or CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    from test_gpu_hsl_multicoin import make_proxy, raw, compare
    import backtest
    monkeypatch.setattr(backtest, "execute_backtest", lambda *_a, **_k: pytest.fail("CPU fallback"))
    candidate = {}
    for side in sides:
        candidate.update({f"{side}_hsl_red_threshold": 0.0001,
                          f"{side}_entry_initial_qty_pct": 0.3,
                          f"{side}_entry_initial_ema_dist": -0.02,
                          f"{side}_hsl_ema_span_minutes": 3.5,
                          f"{side}_hsl_scale_budget_with_excess_allowance": float(scaled),
                          f"{side}_we_excess_allowance_pct": 0.3 if scaled else 0.0})
    outputs = []
    for enabled in (False, True):
        proxy = make_proxy("coin", minutes=2200, sides=sides, coin_count=coins, factual_hsl=True)
        runner = proxy.fused_runner or proxy.runners[sides[0]]
        runner.hsl_native_coin_cache_enabled = enabled
        runner.temporal_chunk_bar_limit = 127
        _, result = raw(proxy, [candidate])
        outputs.append(result)
        assert bool(runner._hsl_native_cache_capacity()) is enabled
        if enabled:
            _, repeated = raw(proxy, [candidate])
            compare(result, repeated)
    assert outputs[0]["fill_count"].item() > 0
    assert sum(outputs[0][f"hsl_triggers_{s}"].item() for s in sides) > 0
    compare(outputs[0], outputs[1])
    for key in ["alive", "fill_count", *(f"hsl_triggers_{s}" for s in sides)]:
        np.testing.assert_array_equal(outputs[0][key].cpu().numpy(), outputs[1][key].cpu().numpy())


def test_coin_fact_growth_drops_optional_cache_without_dropping_facts():
    from optimization.gpu.mps_kernel import HslFactHistoryOverflow
    runner = host_runner(coins=1)
    runner.hsl_scopes = 4
    runner.n, runner.n_days = 2200, 2
    runner.unstuck_pnl_capacity = 0
    runner.max_dispatch_candidate_bars = 1_000_000
    runner._replay_state_bytes = 999
    runner._replay_states = {"old": object()}
    runner.raw_strategy_risk_enabled = runner.raw_strategy_growth_enabled = False
    runner.weighted_equity_metrics = frozenset()
    runner.recovery_distribution_enabled = runner.hsl_ema_tail_enabled = False
    runner.entry_interval_enabled = False
    runner.hsl_scratch_budget_bytes = 512*1024**2
    runner.interrupt_check = lambda: None
    keys = runner.parameter_keys
    matrix = np.zeros((1, 2*len(keys)), dtype=np.float32)
    for side in range(2):
        matrix[:, side*len(keys)+keys.index("hsl_enabled")] = 1
        matrix[:, side*len(keys)+keys.index("hsl_signal_mode")] = 2
    runner._prepare_native_factual_hsl(matrix)
    runner.hsl_native_cache_budget_bytes = runner._hsl_native_cache_bytes_per_candidate()
    attempts = []
    def overflow_then_accept(_params, **_kwargs):
        attempts.append(runner.hsl_fact_capacity)
        if len(attempts) == 1:
            assert runner._hsl_native_cache_capacity() > 0
            raise HslFactHistoryOverflow("rejected attempt")
        assert runner._hsl_native_cache_capacity() == 0
        assert runner._replay_state_bytes is None and not runner._replay_states
        assert not runner._hsl_scratch_buffers
        return {"accepted": True}
    runner._run_factual_attempt = overflow_then_accept
    assert runner.run(matrix) == {"accepted": True}
    assert attempts == [32, 64] and runner.hsl_fact_capacity_learned == 64
    assert runner.last_hsl_fact_retries == 1


def test_coin_native_owner_busy_wrap_rebudget_invalid_mark_and_cache_loss():
    from test_gpu_hsl_native_cache import _native_caller_ablation
    _native_caller_ablation("trailing_martingale", coin=True)
