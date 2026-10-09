"""Full-unstuck inactivity proofs retain any effective consuming request."""
from itertools import product

import numpy as np
import pytest

from optimization.gpu.specialization import unstuck_required

KEYS = ("unstuck_enabled", "other", "unstuck_ema_gating_enabled")


def test_effective_enablement_matches_independent_scalar_inheritance():
    for enabled in (0.0, 1.0):
        rows = np.array([[enabled, 9.0, np.nan], [1.0 - enabled, 9.0, 0.0]])
        for values in product((np.nan, 0.0, 1.0, np.inf), repeat=2):
            overrides = np.column_stack((values, [np.nan, 0.0]))
            expected = any((coin if np.isfinite(coin) else row[0]) > 0.5
                           for row in rows for coin in values)
            assert unstuck_required(rows, KEYS, (overrides,)) == expected


def test_fused_sides_and_mixed_candidates_keep_any_consuming_side():
    inherited = np.full((2, 2), np.nan)
    matrix = np.array([[0.0, 9.0, 1.0, 0.0, 9.0, 1.0],
                       [0.0, 9.0, 0.0, 1.0, 9.0, 0.0]])
    assert unstuck_required(matrix, KEYS, (inherited, inherited))
    disabled_short = inherited.copy()
    disabled_short[:, 0] = 0.0
    assert not unstuck_required(matrix, KEYS, (inherited, disabled_short))
    enabled_long = inherited.copy()
    enabled_long[1, 0] = 1.0
    assert unstuck_required(matrix, KEYS, (enabled_long, disabled_short))


def test_full_proof_uses_float32_enabled_flag_and_ignores_unused_gating():
    inherited = np.full((2, 2), np.nan)
    assert not unstuck_required([[0.500000001, 0.0, np.nan]], KEYS, (inherited,))
    assert unstuck_required([[np.nan, 0.0, 0.0]], KEYS, (inherited,))
    assert unstuck_required(np.empty((0, 3)), KEYS, (inherited,))


@pytest.mark.parametrize("matrix,overrides", [
    ([[0.0, 0.0]], (np.full((2, 2), np.nan),)),
    ([[0.0, 0.0, 0.0]], ()),
    ([[0.0, 0.0, 0.0]], (np.full((2, 1), np.nan),)),
    ([[0.0, 0.0, 0.0]], (np.empty((0, 2)),)),
])
def test_full_proof_rejects_misaligned_views(matrix, overrides):
    with pytest.raises(ValueError, match="unstuck proof"):
        unstuck_required(matrix, KEYS, overrides)


@pytest.fixture(scope='module')
def require_real_passivbot_rust_module():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA device required')
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    info = verify_loaded_runtime_extension()
    assert not info.get('skipped')
    assert info['runtime_compiled_source_stamp'] == info['expected_source_fingerprint']
    return passivbot_rust


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('sides', [('long',), ('short',), ('long', 'short')])
@pytest.mark.parametrize('policy', ['disabled', 'enabled', 'coin_enabled', 'coins_disabled'])
def test_full_unstuck_ablation_preserves_raw_outputs_and_cache_transitions(
    require_real_passivbot_rust_module, strategy, sides, policy,
):
    import torch
    from optimization.gpu.service import MpsMulticoinProxy
    from test_gpu_unstuck_lookback import make_proxy
    from test_gpu_hsl_multicoin import raw

    original, inputs = make_proxy(sides, strategy=strategy)
    del original
    candles, markets, config, exchange, btc, timestamps = inputs
    for side in sides:
        config['bot'][side]['unstuck']['enabled'] = policy in {'enabled', 'coins_disabled'}
    if policy == 'coin_enabled':
        config['coin_overrides'] = {'ETH': {'bot': {sides[-1]: {'unstuck': {'enabled': True}}}}}
    elif policy == 'coins_disabled':
        config['coin_overrides'] = {coin: {'bot': {
            side: {'unstuck': {'enabled': False}} for side in sides
        }} for coin in ('BTC', 'ETH')}
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange=exchange, batch_size=2, factual_hsl=True, needed_metrics={
            'adg_strategy_eq', 'adg_strategy_eq_w', 'volume_pct_per_day_avg_w',
            'strategy_eq_recovery_days_p95', 'drawdown_worst_strategy_eq',
            'drawdown_worst_mean_1pct_strategy_eq', 'hard_stop_time_in_red_pct',
        },
    )
    runner, output = raw(proxy, [{}, {}])
    enabled = policy in {'enabled', 'coin_enabled'}
    assert runner.dispatch_unstuck_enabled == enabled
    assert output['fill_count'].min().item() >= len(sides)
    expected = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in output.items()}

    def equal(left, right):
        assert left.keys() == right.keys()
        for key, value in left.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, right[key], rtol=0, atol=0, equal_nan=True, msg=key)
            else:
                assert value == right[key]

    if strategy == 'trailing_martingale':
        runner.max_dispatch_candidate_bars = 2 * 2 * len(sides) * 7
        _, temporal = raw(proxy, [{}, {}])
        equal(expected, temporal)
        specialized_bytes = runner._replay_state_bytes
    runner.unstuck_specialization = False
    _, general = raw(proxy, [{}, {}])
    assert runner.dispatch_unstuck_enabled
    equal(expected, general)
    if strategy == 'trailing_martingale' and not enabled:
        assert runner._replay_state_bytes > specialized_bytes
    runner.unstuck_specialization = True
    _, restored = raw(proxy, [{}])
    assert runner.dispatch_unstuck_enabled == enabled
    equal({key: value[:1] if isinstance(value, torch.Tensor) else value
           for key, value in expected.items()}, restored)
    if policy == 'disabled':
        candidates = [{}, {f'{sides[-1]}_unstuck_enabled': 1.0}]
        _, mixed = raw(proxy, candidates)
        assert runner.dispatch_unstuck_enabled
        runner.unstuck_specialization = False
        _, mixed_general = raw(proxy, candidates)
        equal(mixed, mixed_general)
        runner.unstuck_specialization = True
        _, disabled_again = raw(proxy, [{}, {}])
        assert not runner.dispatch_unstuck_enabled
        equal(expected, disabled_again)


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('sides', [('long',), ('short',), ('long', 'short')])
def test_disabled_unstuck_preserves_active_factual_hsl_panics(
    require_real_passivbot_rust_module, strategy, sides,
):
    import torch
    from test_gpu_hsl_multicoin import make_proxy, raw

    proxy = make_proxy('unified', strategy, sides, minutes=3000, factual_hsl=True)
    candidate = {f'{side}_unstuck_enabled': 0.0 for side in sides}
    candidate['hsl_red_threshold'] = 1e-6
    runner, specialized = raw(proxy, [candidate])
    assert not runner.dispatch_unstuck_enabled
    assert not runner.dispatch_hsl_disabled
    assert (specialized['hsl_triggers_long'] + specialized['hsl_triggers_short']).item() > 0
    runner.unstuck_specialization = False
    _, general = raw(proxy, [candidate])
    assert runner.dispatch_unstuck_enabled
    assert specialized.keys() == general.keys()
    for key, value in specialized.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, general[key], rtol=0, atol=0, equal_nan=True, msg=key)
        else:
            assert value == general[key]


@pytest.mark.asyncio
@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('suite', [False, True])
async def test_disabled_unstuck_native_cli_persists_and_resumes_without_cpu(
    require_real_passivbot_rust_module, monkeypatch, tmp_path, strategy, suite,
):
    from tools import gpu_parity
    from optimization.gpu.service import MpsMulticoinProxy
    from test_native_backend_cuda import test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu

    fixture_inputs = gpu_parity.fixture_inputs
    def disabled_inputs(*args, **kwargs):
        inputs = fixture_inputs(*args, **kwargs)
        for side in ('long', 'short'):
            inputs[0]['bot'][side]['unstuck']['enabled'] = False
        return inputs
    monkeypatch.setattr(gpu_parity, 'fixture_inputs', disabled_inputs)
    evaluate = MpsMulticoinProxy.evaluate_results
    observations = []
    def observed(proxy, candidates):
        results = evaluate(proxy, candidates)
        runners = [proxy.fused_runner] if proxy.fused_runner else [proxy.runners[s] for s in proxy.sides]
        assert all(not runner.dispatch_unstuck_enabled for runner in runners)
        observations.append(len(results))
        return results
    monkeypatch.setattr(MpsMulticoinProxy, 'evaluate_results', observed)
    await test_native_optimizer_cli_runs_cuda_and_resumes_without_cpu(
        monkeypatch, tmp_path, suite, True, True, screening=suite, strategy_kind=strategy,
    )
    assert observations and sum(observations) > 0
