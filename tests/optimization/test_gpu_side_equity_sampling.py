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
    metrics = [
        f'{name}_strategy_equity_{side}'
        for side in ('long', 'short')
        for name in ('recovery_max_days', 'drawdown_max_fraction', 'drawdown_daily_worst_1pct_mean_fraction')
    ]
    policies = {name: MetricTolerance(1e-6, 1e-4) for name in metrics}
    if strategy == 'trailing_martingale' and sides == 'both' and hsl in ('disabled', 'portfolio'):
        # These two public shock fixtures retain existing trading differences.
        # At step 1501 a .060-.042 partial-entry subtraction floors to .017
        # in Rust and .018 under GPU float32 rounding. The long drawdown gaps
        # are 5.35e-5 and 5.94e-6; bound this fixture's risk error to one basis
        # point. Recovery, short drawdown and all other cases stay strict.
        for name in ('drawdown_max_fraction_strategy_equity_long',
                     'drawdown_daily_worst_1pct_mean_fraction_strategy_equity_long'):
            policies[name] = MetricTolerance(1e-4, 0)
    report = run_comparison(inputs, 'binance', metrics, policies, gpu_engine='native')
    for side in ('long', 'short'):
        assert report['metrics'][f'recovery_max_days_strategy_equity_{side}']['cpu'] > 0
        if sides in (side, 'both'):
            assert report['metrics'][f'drawdown_max_fraction_strategy_equity_{side}']['cpu'] > 0
    assert report['passed'], report['metrics']


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("days", [2, 199])
def test_single_day_drawdown_tail_uses_existing_maximum(strategy, days):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()

    source = ("#define PASSIVBOT_HSL_CAPACITY 64\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
              "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
              "#define PASSIVBOT_HSL_RAW_TAIL_ENABLED 1\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void side_daily_tail_probe(constant int* days, device float* out,
    uint b [[thread_position_in_grid]]) {
    HslStrategyEquityStats stats = init_hsl_strategy_equity_stats();
    update_hsl_strategy_equity_stats(stats, 1000, 0);
    update_hsl_strategy_equity_stats(stats, 985, 0);
    update_hsl_strategy_equity_stats(stats, 983, 1);
    for (int day = 2; day < days[0]; ++day) {
        update_hsl_strategy_equity_stats(stats, 985, day);
    }
    out[0] = hsl_strategy_equity_drawdown_max(stats);
    out[1] = hsl_strategy_equity_drawdown_mean_worst_1pct(stats);
}
""")
    # Both daily maxima fall in one log bin. The CPU's worst floor(1%) is
    # one sample up to 199 days, so it selects .017 rather than the bin mean.
    device = gpu_device()
    count = torch.tensor([days], dtype=torch.int32, device=device)
    out = torch.empty(2, dtype=torch.float32, device=device)
    compile_shader(source).side_daily_tail_probe(count, out, threads=1)
    assert out.cpu().tolist() == pytest.approx([.017, .017], abs=1e-7)
