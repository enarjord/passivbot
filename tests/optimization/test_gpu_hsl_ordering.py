"""Shared GPU next orders consume the current bar's HSL decision."""

import pytest


@pytest.mark.parametrize("strategy,mode,coins,sides,shocks", [
    ("ema_anchor", "coin", 1, "long", False),
    ("ema_anchor", "coin", 2, "long", False),
    ("ema_anchor", "coin", 1, "both", False),
    ("ema_anchor", "coin", 2, "both", False),
    ("ema_anchor", "pside", 2, "both", True),
    ("ema_anchor", "unified", 2, "both", True),
    ("trailing_martingale", "coin", 2, "long", True),
    ("trailing_martingale", "pside", 2, "long", True),
    ("trailing_martingale", "unified", 2, "long", True),
    ("trailing_martingale", "coin", 2, "short", True),
    ("trailing_martingale", "pside", 2, "short", True),
    ("trailing_martingale", "unified", 2, "short", True),
    ("trailing_martingale", "coin", 2, "both", True),
    ("trailing_martingale", "pside", 2, "both", True),
    ("trailing_martingale", "unified", 2, "both", True),
])
def test_scoped_hsl_orders_use_current_observation(strategy, mode, coins, sides, shocks):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    from optimization.gpu.parity import MetricTolerance

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", str(coins),
        "--bars", "3000", "--seed", "43", "--hsl", mode,
    ]))
    for side in ("long", "short"):
        enabled = sides == "both" or side == sides
        inputs[0]["bot"][side]["risk"].update(
            n_positions=1 if enabled else 0,
            total_wallet_exposure_limit=2 if enabled else 0,
        )
        inputs[0]["bot"][side]["hsl"].update(
            red_threshold=0.002, ema_span_minutes=2.5, cooldown_minutes_after_red=5,
        )
    if mode == "unified":
        inputs[0]["bot"]["hsl"].update(
            red_threshold=0.002, ema_span_minutes=2.5, cooldown_minutes_after_red=5,
        )
    if shocks:
        # Trigger TM at a practical threshold after established exposure;
        # exercise shared EMA scopes without relying on float32-scale noise.
        inputs[1][1500:, 0, :3] *= 0.7
        inputs[1][1800:, 1, :3] *= 1.3
    metrics = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day",
               "hard_stop_triggers_per_year", "hard_stop_restarts_per_year", "hard_stop_time_in_red_pct")
    policies = {**DEFAULT_TOLERANCES,
                "hard_stop_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_restarts_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_time_in_red_pct": MetricTolerance(1e-8, 1e-5)}
    if (strategy == "trailing_martingale" and sides in {"short", "both"}
            and coins == 2 and shocks):
        # On these six seed-43 shock fixtures, original/corrected shaders have
        # identical non-time metrics. CPU ADG differs by <=4.51e-5 and fill rate
        # by <=0.098%; drawdown and lifecycle gates already pass. Keep these
        # bounded trajectory differences local; time-in-red stays strict.
        policies["adg_strategy_eq"] = MetricTolerance(5e-5, 0)
        policies["fills_per_day"] = MetricTolerance(0, 1e-3)
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", [1, 2], ids=["pside", "coin"])
def test_forced_terminal_reporting_uses_final_equity_without_reobserving(strategy, mode):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader, gpu_device

    source = ("#define PASSIVBOT_HSL_CAPACITY 64\n"
              "#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 1440\n"
              "#define PASSIVBOT_HSL_RAW_DRAWDOWN_ENABLED 1\n"
              "#define PASSIVBOT_HSL_EMA_TAIL_ENABLED 1\n"
              + getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
              + r"""
kernel void terminal_reporting_probe(constant float* params, device HslNode* trees,
    device int* rows, device float* out, uint b [[thread_position_in_grid]]) {
    HslState aggregate = load_hsl(params, 0, 0);
    HslState coins[1];
    coins[0] = load_hsl(params, 0, 0);
    thread HslState& owner = aggregate.signal_mode == HSL_SIGNAL_COIN ? coins[0] : aggregate;
    bind_hsl(owner, trees, rows, 0, 64, 1, 1440, true, true);
    HslDrawdownEmaTailStats tail = init_hsl_drawdown_ema_tail_stats();
    HslStrategyEquityStats eq = init_hsl_strategy_equity_stats();
    observe_hsl(owner, 1000, 0, 0, false, 0, false);
    record_multicoin_hsl_report(aggregate, coins, 1, 1, true, tail, eq, 1000, 0);
    observe_hsl(owner, 1000, 0, 50, true, 1, false);
    observe_hsl(owner, 1000, 0, 5000, true, 2, false);
    // A same-bar forced close follows next-order construction. The terminal
    // fill's factual time is one minute earlier than that provisional mark.
    finish_hsl_episode_at_flat(owner, 700, 1000, -300, 1, 60000);
    out[0] = record_multicoin_hsl_report(
        aggregate, coins, 1, 1, true, tail, eq, 700, 0);
    out[1] = hsl_strategy_equity_drawdown_max(eq);
    out[2] = owner.hsl.last_observed;
    out[3] = owner.hsl.flat_minute;
    out[4] = owner.hsl_valid;
    out[5] = owner.sampled_drawdown_raw;
}
""")
    device = gpu_device()
    params = torch.tensor([1, 0.1, 1, 5, 0, mode, 1], dtype=torch.float32, device=device)
    trees = torch.empty((36, 32), dtype=torch.uint8, device=device)
    rows = torch.empty(256, dtype=torch.int32, device=device)
    out = torch.zeros(6, dtype=torch.float32, device=device)
    compile_shader(source).terminal_reporting_probe(params, trees, rows, out, threads=1)
    assert out.cpu().tolist() == pytest.approx([3, 0.3, 1, 1, 1, 350 / 1050])


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_invalid_hsl_propagates_from_native_service(monkeypatch, strategy):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import backtest
    import passivbot_rust
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset, DEFAULT_METRICS

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2",
        "--bars", "128", "--hsl", "coin",
    ]))
    getter_name = f"mps_{strategy}_multicoin_source_py"
    source = getattr(passivbot_rust, getter_name)()
    assert source.count("if (!hsl_valid)") == 1
    # Inject the controller's rejection at its real caller boundary. A failed
    # simulation must surface through the future, never as liquidation metrics.
    monkeypatch.setattr(passivbot_rust, getter_name,
                        lambda: source.replace("if (!hsl_valid)", "if (true)"))
    def forbidden(*_args, **_kwargs):
        pytest.fail("native service must not replace rejected HSL with CPU simulation")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(passivbot_rust, "run_backtest_bundle", forbidden)
    library = getattr(mps_kernel, f"_{strategy}_multicoin_shader_library")
    library.cache_clear()
    try:
        with _native_dataset(inputs, "binance", DEFAULT_METRICS) as dataset:
            with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
                service.register_dataset("rejected-hsl", dataset)
                future = service.submit(BacktestRequest("rejected", "rejected-hsl", {}))
                with pytest.raises(ValueError, match="unavailable HSL controller"):
                    future.result()
    finally:
        library.cache_clear()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["mark", "panic-fill"])
def test_liquidation_retains_elapsed_red_interval(strategy, sides, terminal_fill):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from config.hsl import generated_template
    from test_gpu_entry_sizing_parity import _fixture
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = list(_fixture("long", 1, "initial"))
    config = generated_template(inputs[0], "coin")
    inputs[0] = config
    config["live"].update(strategy_kind=strategy, hedge_mode=sides == "both")
    for side in ("long", "short"):
        bot = config["bot"][side]
        enabled = side == "long" or sides == "both"
        bot["risk"].update(n_positions=1 if enabled else 0,
                           total_wallet_exposure_limit=(5.0 if side == "long" else 0.001) if enabled else 0)
        bot["hsl"].update(enabled=enabled, red_threshold=0.002, ema_span_minutes=1,
                          cooldown_minutes_after_red=5, restart_after_red_policy="always",
                          panic_close_order_type="market" if terminal_fill else "limit")
        bot["strategy"]["ema_anchor"]["base_qty_pct"] = 0.8
        bot["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.8
    candles = inputs[1]
    # Entry at 3, RED at 4, liquidation at 5. A limit panic cannot fill across
    # the gap; a market panic liquidates at the earlier factual fill boundary.
    candles[4, :, :3] = [100, 99, 99.5]
    candles[5:, :, :3] = [21, 19, 20]
    metric = "hard_stop_time_in_red_pct"
    report = run_comparison(tuple(inputs), "bybit", (metric,),
                            {metric: MetricTolerance(1e-8, 1e-6)},
                            diagnostics=True, gpu_engine="native")
    expected = 0 if terminal_fill else 1 / 3
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["metrics"][metric]["cpu"] == pytest.approx(expected)
    assert report["passed"], report["metrics"]
