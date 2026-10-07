"""Portfolio HSL metrics retain the joint clock of observed scope signals."""

import pytest


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", [0, 1, 2], ids=["unified", "pside", "coin"])
def test_portfolio_tail_reduces_per_bar_scope_maximum(strategy, mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = ("#define PASSIVBOT_HSL_CAPACITY 64\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
              "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void portfolio_ema_tail_probe(constant float* params,
    constant float* samples, device float* out, uint b [[thread_position_in_grid]]) {
    HslState long_signal = load_hsl(params, 0, 0);
    HslState short_signal = load_hsl(params, 0, 0);
    HslState long_coins[2];
    HslState short_coins[2];
    long_coins[0] = long_signal;
    short_coins[0] = short_signal;
    long_coins[1] = long_signal;
    short_coins[1] = short_signal;
    long_coins[1].enabled = false;
    short_coins[1].enabled = false;
    long_coins[1].drawdown_ema = 99.0f;
    short_coins[1].drawdown_ema = 99.0f;
    // Cached current signals remain observations during a halted interval.
    long_signal.halted = true;
    short_signal.halted = true;
    HslDrawdownEmaTailStats portfolio = init_hsl_drawdown_ema_tail_stats();
    HslDrawdownEmaTailStats long_tail = init_hsl_drawdown_ema_tail_stats();
    HslDrawdownEmaTailStats short_tail = init_hsl_drawdown_ema_tail_stats();
    for (int k = 0; k < 200; ++k) {
        long_signal.drawdown_ema = samples[k * 2];
        short_signal.drawdown_ema = samples[k * 2 + 1];
        long_coins[0].drawdown_ema = samples[k * 2];
        short_coins[0].drawdown_ema = samples[k * 2 + 1];
        float l = observed_multicoin_hsl_ema(long_signal, long_coins, 2, 1);
        float s = observed_multicoin_hsl_ema(short_signal, short_coins, 2, 1);
        update_hsl_drawdown_ema_tail_stats(portfolio, fmax(l, s));
        update_hsl_drawdown_ema_tail_stats(long_tail, l);
        update_hsl_drawdown_ema_tail_stats(short_tail, s);
    }
    out[0] = hsl_drawdown_ema_mean_worst_1pct(portfolio);
    out[1] = hsl_drawdown_ema_mean_worst_1pct(long_tail);
    out[2] = hsl_drawdown_ema_mean_worst_1pct(short_tail);
    out[3] = portfolio.sample_count;
    out[4] = observed_multicoin_hsl_ema(long_signal, long_coins, 2, 0);
}
""")
    device = gpu_device()
    params = torch.tensor([1, .1, 1, 5, 0, mode, 1], dtype=torch.float32, device=device)
    samples = torch.zeros((200, 2), dtype=torch.float32, device=device)
    samples[0, 0], samples[1, 0] = 1, .2
    samples[2, 1], samples[3, 1] = .8, .1
    out = torch.empty(5, dtype=torch.float32, device=device)
    compile_shader(source).portfolio_ema_tail_probe(params, samples, out, threads=1)
    # Actual Rust hsl_strategy_metrics exercises this same controlled series.
    assert out.cpu().tolist() == pytest.approx([.9, .6, .45, 200, 0], abs=1e-6)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("feature", ["BASE", "EMA_TAIL", "RAW_DRAWDOWN", "ALL"])
def test_portfolio_tail_scalar_does_not_alias_optional_side_metrics(fused, feature):
    torch = pytest.importorskip("torch")
    from optimization.gpu import mps_kernel as kernel

    prefix = "MPS_MULTICOIN_FUSED" if fused else "MPS_MULTICOIN"
    suffix = "SCALAR_COLS" if feature == "ALL" else f"{feature}_SCALAR_COLS"
    width = getattr(kernel, f"{prefix}_{suffix}")
    scalars = torch.zeros((1, width))
    scalars[:, -3], scalars[:, -2], scalars[:, -1] = .9, 123, 456
    daily = torch.zeros((1, 1, 9))
    daily[:, :, 1] = float("inf")
    gaps = torch.zeros((1, 512), dtype=torch.int32)
    decode = kernel._decode_multicoin_fused_outputs if fused else kernel._decode_outputs
    result = decode(daily, scalars, gaps)
    assert result["hsl_drawdown_ema_mean_worst_1pct_portfolio"].item() == pytest.approx(.9)
    for key in ("hsl_drawdown_ema_mean_worst_1pct_long",
                "hsl_drawdown_ema_mean_worst_1pct_short",
                "hsl_drawdown_raw_max_long", "hsl_drawdown_raw_max_short",
                "hsl_drawdown_raw_mean_worst_1pct_long",
                "hsl_drawdown_raw_mean_worst_1pct_short"):
        assert result[key].item() == 0
    assert result["held_sum_squared_hours"].item() == 123
    assert result["gap_sum_squared_hours"].item() == 456


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_ema_tail_capture_preserves_shared_shock_replay(strategy, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from test_gpu_weighted_equity_capture import _runner_context

    baseline, params = _runner_context(strategy, sides, shock=True)
    enabled, _ = _runner_context(strategy, sides, shock=True, hsl_tail=True)
    expected, actual = baseline.run(params), enabled.run(params)
    tails = {"hsl_drawdown_ema_mean_worst_1pct_portfolio",
             "hsl_drawdown_ema_mean_worst_1pct_long",
             "hsl_drawdown_ema_mean_worst_1pct_short"}
    assert set(actual) == set(expected)
    assert actual["hsl_drawdown_ema_mean_worst_1pct_portfolio"].min().item() > 0
    for name, value in expected.items():
        if name not in tails:
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True)
            else:
                assert actual[name] == value


@pytest.mark.parametrize("side", ["long", "short"])
def test_temporal_portfolio_tail_preserves_partial_histories(side):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import numpy as np
    from test_gpu_weighted_equity_capture import _runner_context

    generic, params = _runner_context("trailing_martingale", side, shock=True, hsl_tail=True)
    temporal, _ = _runner_context("trailing_martingale", side, shock=True, hsl_tail=True, chunked=True)
    ends = np.asarray([2, 799, 1512], dtype=np.int32)
    expected = {k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in generic.run(params, end_steps=ends).items()}
    actual = temporal.run(params, end_steps=ends)
    assert actual["hsl_drawdown_ema_mean_worst_1pct_portfolio"][1].item() > 0
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert actual[name] == value


@pytest.mark.parametrize("strategy,relative_bound", [
    ("ema_anchor", .025), ("trailing_martingale", .002),
])
def test_observed_portfolio_tail_on_public_twenty_day_shock(strategy, relative_bound):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "28800",
        "--seed", "43", "--hsl", "coin", "--unstuck",
    ]))
    for side in ("long", "short"):
        inputs[0]["bot"][side]["hsl"].update(
            red_threshold=.002, ema_span_minutes=2.5, cooldown_minutes_after_red=5,
        )
    inputs[1][1440:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    metric = "drawdown_worst_mean_1pct_ema_strategy_eq"
    # Local bounds distinguish the corrected joint observation from the former
    # max-of-tails result. They retain existing f32 trajectory/bin approximation;
    # broader tail materiality and general comparison policy remain separate.
    report = run_comparison(inputs, "binance", [metric],
                            {metric: MetricTolerance(1e-6, relative_bound)}, gpu_engine="native")
    assert report["metrics"][metric]["cpu"] > 0
    assert report["passed"], report["metrics"]
