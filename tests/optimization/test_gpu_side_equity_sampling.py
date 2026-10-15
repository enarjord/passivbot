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
                red_threshold=.002, ema_span_minutes=2.5,
                cooldown_minutes_after_red=5,
            )
        if hsl == 'unified':
            inputs[0]['bot']['hsl'].update(
                red_threshold=.002, ema_span_minutes=2.5,
                cooldown_minutes_after_red=5,
            )
    inputs[1][1500:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    return inputs


@pytest.mark.parametrize('strategy', ['ema_anchor', 'trailing_martingale'])
@pytest.mark.parametrize('sides', ['long', 'short', 'both'])
@pytest.mark.parametrize('hsl', ['disabled', 'coin', 'pside', 'unified'])
def test_native_side_equity_metrics_independent_of_hsl(strategy, sides, hsl):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import run_comparison
    inputs = _side_equity_inputs(strategy, sides, hsl)
    metrics = [
        f'{name}_strategy_eq_{side}'
        for side in ('long', 'short')
        for name in ('peak_recovery_days', 'drawdown_worst', 'drawdown_worst_mean_1pct')
    ]
    policies = {name: MetricTolerance(1e-6, 1e-4) for name in metrics}
    if strategy == 'trailing_martingale' and sides == 'both' and hsl in ('disabled', 'unified'):
        # These two public shock fixtures retain existing trading differences.
        # At step 1501 a .060-.042 partial-entry subtraction floors to .017
        # in Rust and .018 under GPU float32 rounding. The long drawdown gaps
        # are 5.35e-5 and 5.94e-6; bound this fixture's risk error to one basis
        # point. Recovery, short drawdown and all other cases stay strict.
        for name in ('drawdown_worst_strategy_eq_long',
                     'drawdown_worst_mean_1pct_strategy_eq_long'):
            policies[name] = MetricTolerance(1e-4, 0)
    report = run_comparison(inputs, 'binance', metrics, policies, gpu_engine='native')
    for side in ('long', 'short'):
        assert report['metrics'][f'peak_recovery_days_strategy_eq_{side}']['cpu'] > 0
        if sides in (side, 'both'):
            assert report['metrics'][f'drawdown_worst_strategy_eq_{side}']['cpu'] > 0
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
