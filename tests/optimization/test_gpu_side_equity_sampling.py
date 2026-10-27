"""Side strategy-equity metrics exist independently of HSL protection."""
import pytest


def _side_equity_inputs(strategy, sides, hsl):
    from tools.gpu_parity import build_parser, fixture_inputs
    inputs = fixture_inputs(build_parser().parse_args([
        '--fixture', strategy, '--sides', sides, '--coins', '2',
        '--bars', '3000', '--seed', '43', '--hsl', hsl,
    ]))
    if hsl != 'disabled':
        for side in ('long', 'short'):
            inputs[0]['bot'][side]['hsl'].update(
                drawdown_threshold_fraction=.002, drawdown_fraction_ema_span_minutes=2.5,
                cooldown_after_red_minutes=5,
            )
        if hsl == 'portfolio':
            inputs[0]['bot']['portfolio']['hsl'].update(
                drawdown_threshold_fraction=.002, drawdown_fraction_ema_span_minutes=2.5,
                cooldown_after_red_minutes=5,
            )
    inputs[1][1500:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    return inputs


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('hsl', ['coin', 'position_side', 'portfolio'])
def test_side_equity_stress_fixture_lowers_current_hsl_policy(strategy, hsl):
    from backtest import prep_backtest_args
    from config.hsl import native_hsl_mode, native_hsl_policy

    config, _candles, markets, _btc, _timestamps = _side_equity_inputs(strategy, 'both', hsl)
    owners = [config['bot']['portfolio']['hsl']] if hsl == 'portfolio' else [
        config['bot'][side]['hsl'] for side in ('long', 'short')
    ]
    for policy in owners:
        assert policy['drawdown_threshold_fraction'] == .002
        assert policy['drawdown_fraction_ema_span_minutes'] == 2.5
        assert policy['cooldown_after_red_minutes'] == 5
    payload = prep_backtest_args(config, markets, 'binance')[3]['equity_hard_stop_loss']
    transported = [payload['portfolio']] if hsl == 'portfolio' else payload['sides']
    assert payload['mode'] == native_hsl_mode(hsl)
    expected = [native_hsl_policy(policy) for policy in owners]
    for policy in expected:
        policy.pop('scope', None)
    assert transported == expected


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('sides', ['long', 'short', 'both'])
@pytest.mark.parametrize('hsl', ['disabled', 'coin', 'position_side', 'portfolio'])
def test_native_side_equity_metrics_independent_of_hsl(strategy, sides, hsl):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import run_comparison
    inputs = _side_equity_inputs(strategy, sides, hsl)
    metrics = ["drawdown_max_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity",
               "pain_index_fraction_strategy_equity"]
    policies = {name: MetricTolerance(1e-4, 1e-3) for name in metrics}
    report = run_comparison(inputs, 'binance', metrics, policies, gpu_engine='native')
    assert report['metrics']['drawdown_max_fraction_strategy_equity']['cpu'] > 0
    assert report['passed'], report['metrics']
