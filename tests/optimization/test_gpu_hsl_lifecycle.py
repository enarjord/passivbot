"""GPU lifecycle reporting includes open RED, GREEN recovery and retriggers."""

import pytest




@pytest.mark.parametrize("strategy,mode,sides", [
    (strategy, mode, sides)
    for strategy in ("ema_anchor", "trailing_martingale")
    for mode, sides in (("coin", "long"), ("coin", "short"), ("coin", "both"),
                        ("position_side", "both"), ("portfolio", "both"))
])
def test_native_hsl_lifecycle_metrics_match_cpu(strategy, mode, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2",
        "--bars", "3000", "--seed", "43", "--hsl", mode,
    ]))
    for side in ("long", "short"):
        inputs[0]["bot"][side]["hsl"].update(
            drawdown_ratio_threshold=.002, ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,
        )
    if mode == "portfolio":
        inputs[0]["bot"]["hsl"].update(
            drawdown_ratio_threshold=.002, ema_span_minutes=2.5,
            cooldown_minutes_after_red=5,
        )
    inputs[1][1500:, 0, :3] *= .7
    inputs[1][1800:, 1, :3] *= 1.3
    # Flatten-time remains CPU-analysis-only at the public native API; the
    # shader probes above verify its raw reporting/censoring separately.
    metrics = [
        "hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
        "hard_stop_post_restart_retrigger_pct", "hard_stop_duration_minutes_mean",
        "hard_stop_duration_minutes_max", "hard_stop_trigger_drawdown_mean",
        "hard_stop_time_in_red_pct", "hard_stop_restarts_per_year_long",
        "hard_stop_restarts_per_year_short",
    ]
    report = run_comparison(
        inputs, "binance", metrics,
        {name: MetricTolerance(1e-5, 1e-3) for name in metrics}, gpu_engine="native",
    )
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["open-panic", "open-halt"])
def test_native_unfinished_hsl_duration_uses_final_reporting_time(strategy, mode, sides, terminal_fill):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from config.hsl import generated_template
    from test_gpu_entry_sizing_parity import _fixture
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import run_comparison

    inputs = list(_fixture("long", 2, "initial"))
    config = generated_template(inputs[0], mode)
    inputs[0] = config
    config['live'].update(lookback_ema_span_minutes=1440.0)
    config['bot']["portfolio"]['strategy_kind'] = strategy
    config['bot']["portfolio"]['hedge_mode'] = sides == 'both'
    policy = dict(drawdown_ratio_threshold=.002, ema_span_minutes=1,
                  cooldown_minutes_after_red=10000,

                  panic_close_order_type="market" if terminal_fill else "limit")
    for side in ("long", "short"):
        bot = config["bot"][side]
        enabled = side == "long" or sides == "both"
        bot["risk"].update(n_positions=2 if enabled else 0,
                           total_wallet_exposure_limit=(5.0 if side == "long" else .001) if enabled else 0)
        bot["hsl"].update(enabled=enabled, **policy)
        bot["strategy"]["ema_anchor"]["base_qty_pct"] = .8
        bot["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = .8
    if mode == "portfolio":
        config["bot"]["hsl"].update(enabled=True, **policy)
    # Entry at 3; RED is observed at the close of 4. The limit panic cannot
    # cross the next gap, while the market panic flattens into a long cooldown.
    # Row 6 is lookahead: both remain RED at the final simulated close of 5.
    # Neither history nor cooldown expires.
    inputs[1][4, :, :3] = [100, 99, 99.5]
    terminal_mark = 98.5 if terminal_fill else 93.0
    inputs[1][5:, :, :3] = [terminal_mark + .1, terminal_mark - .1, terminal_mark]
    metrics = ["hard_stop_duration_minutes_mean", "hard_stop_duration_minutes_max"]
    report = run_comparison(
        tuple(inputs), "bybit", metrics,
        {name: MetricTolerance(1e-6, 0) for name in metrics},
        diagnostics=True, gpu_engine="native",
    )
    assert not report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    remaining_long = report["diagnostics"]["cpu"]["absolute_position_quantity"]["long"]
    if terminal_fill:
        assert remaining_long == pytest.approx(0)
    else:
        assert remaining_long > 0
    for name in metrics:
        assert report["metrics"][name]["cpu"] == pytest.approx(1)
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("gpu_engine", ["native"])
def test_forced_delist_liquidation_hsl_duration_reaches_bar_close(strategy, gpu_engine):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import numpy as np
    from config.hsl import generated_template
    from test_gpu_entry_sizing_parity import _fixture
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import run_comparison

    inputs = list(_fixture("long", 1, "initial"))
    config = generated_template(inputs[0], "coin")
    inputs[0] = config
    config['live'].update(lookback_ema_span_minutes=1440.0)
    config['bot']["portfolio"]['strategy_kind'] = strategy
    config['bot']["portfolio"]['hedge_mode'] = False
    config["bot"]["long"]["risk"].update(n_positions=1, total_wallet_exposure_limit=5.0)
    config["bot"]["long"]["hsl"].update(
        enabled=True, drawdown_ratio_threshold=.002, ema_span_minutes=1,
        cooldown_minutes_after_red=10000,
        panic_close_order_type="limit",
    )
    config["bot"]["long"]["strategy"]["ema_anchor"]["base_qty_fraction"] = .8
    config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_fraction"] = .8
    # The panic limit cannot cross the gap. Forced delisting realizes the loss
    # after Rust's ordinary-fill liquidation check, so reporting reaches close 6.
    candles = np.repeat(inputs[1][-1:], 1410, axis=0)
    candles[:7] = inputs[1]
    candles[4, :, :3] = [100, 99, 99.5]
    candles[5, :, :3] = [93.1, 92.9, 93]
    candles[6, :, :3] = [20.1, 19.9, 20]
    candles[7:, :, :3] = np.nan
    inputs[1] = candles
    inputs[2]["BTC"]["last_valid_index"] = 6
    inputs[3] = np.full(1410, 50_000.0)
    inputs[4] = inputs[4][0] + np.arange(1410, dtype=np.int64) * 60_000
    metrics = ["hard_stop_duration_minutes_mean", "hard_stop_duration_minutes_max"]
    report = run_comparison(
        tuple(inputs), "bybit", metrics,
        {name: MetricTolerance(1e-6, 0) for name in metrics},
        diagnostics=True, gpu_engine=gpu_engine,
    )
    fills = report["diagnostics"]["cpu"]["last_fills"]
    assert [fill["step"] for fill in fills] == [3, 6]
    assert report["diagnostics"]["cpu"]["absolute_position_quantity"]["long"] == 0
    if gpu_engine == "native":
        assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    else:
        assert report["diagnostics"]["gpu"]["single"]["last_eq_ts"] == 6 * 60_000
    for name in metrics:
        assert report["metrics"][name]["cpu"] == pytest.approx(2)
    assert report["passed"], report["metrics"]
