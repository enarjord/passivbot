"""Actual CUDA qualification of the full core catalog and signed contributions."""

from functools import lru_cache
import json

import numpy as np
import pytest


@pytest.fixture
def cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("actual NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    runtime = verify_loaded_runtime_extension()
    assert runtime["runtime_compiled_source_stamp"] == runtime["expected_source_fingerprint"]
    return torch


def _flat_inputs(strategy, sides):
    from tools.gpu_parity import build_parser, fixture_inputs

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", "2", "--bars", "8640",
    ]))
    config, candles, *_ = inputs
    candles[:, :, :3] = 100.0
    candles[:, :, 3] = 1000.0
    config["bot"]["portfolio"]["execution"]["market_orders_allowed"] = False
    for side in ("long", "short"):
        # Resting entries stay strictly outside every flat candle. No fills are
        # an explicit witness, rather than an assumption about strategy defaults.
        bot = config["bot"][side]
        bot["unstuck"]["enabled"] = False
        if strategy == "ema_anchor":
            bot["strategy"][strategy]["offset_fraction"] = .01
        else:
            bot["strategy"][strategy]["entry"]["initial_ema_distance_fraction"] = .01
    return inputs


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_full_cuda_catalog_matches_flat_cpu_window_and_chunk_boundaries(cuda_runtime, strategy, sides):
    from backtest import build_backtest_payload, execute_backtest
    from config.metrics import CORE_METRIC_CATALOG
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    inputs = _flat_inputs(strategy, sides)
    config, candles, markets, btc, timestamps = inputs
    names = tuple(CORE_METRIC_CATALOG)
    payload = build_backtest_payload(candles, markets, config, "binance", btc, timestamps,
                                     metrics_only=False, metric_requests=names)
    fills, equities, cpu = execute_backtest(payload, config)
    assert len(fills) == 0
    np.testing.assert_array_equal(equities[:, 3], np.full(len(equities), 1000.0))
    metadata = cpu["metric_metadata"]
    assert metadata["daily_return_count"] >= 4
    assert metadata["equity_observation_count"] > 4 * 1440
    assert not cpu["liquidated"]
    expected = {name: cpu[name] for name in names}
    # These exact zero/count-derived values and undefined statuses complement
    # the independently specified finite-return observer fixtures. They need no
    # relaxed trading-path tolerance or special treatment for missing values.
    with _native_dataset(inputs, "binance", names) as dataset:
        for budget in (500_000_000, 1024):
            with CudaBacktestService(batch_size=1, tuning_mode="off",
                                     max_dispatch_candidate_bars=budget) as service:
                service.register_dataset("flat", dataset)
                actual = service.submit(BacktestRequest("flat", "flat", {})).result()
            assert not actual.liquidated
            assert actual.metrics == expected
            assert actual.metadata["metric_status"] == cpu["metric_status"]
            assert actual.metadata["metric_metadata"] == metadata
            json.dumps({**actual.metrics, **actual.metadata}, allow_nan=False)


_PROFIT_PROBE = r"""
kernel void core_profit_contribution_probe(constant float* profits, device float* output,
    uint b [[thread_position_in_grid]]) {
    if (b != 0) return;
    CoreGpuState state = init_core_gpu(1000.0f, 0);
    core_observe_equity(state, 1000.0f + profits[0] + profits[1], 1, 0,
                        0.0f, 0.0f, 0.0f, false);
    state.net_profit_long = profits[0];
    state.net_profit_short = profits[1];
    write_core_gpu(state, output, 0, 0);
}
"""


@lru_cache(maxsize=1)
def _profit_library():
    import passivbot_rust
    from optimization.gpu.core_metrics import metric_mask, with_core_metrics
    from optimization.gpu.mps_kernel import _with_hsl
    from optimization.gpu.runtime import compile_shader

    name = "short_net_profit_contribution_fraction"
    source = ("#define PASSIVBOT_HSL_LOOKBACK 0\n"
              + _with_hsl(passivbot_rust.mps_ema_anchor_multicoin_source_py(), 1)
              + _PROFIT_PROBE)
    return compile_shader(with_core_metrics(source, metric_mask([name]), 180.0),
                          mps_coin_capacity=2, cuda_coin_capacity=2)


@pytest.mark.parametrize("long,short,expected", [
    (100., 0., 0.), (0., 100., 1.), (100., 100., .5),
    (100., -25., -1 / 3), (-25., 100., 4 / 3),
    (0., 0., None), (-100., -100., None),
])
def test_cuda_short_contribution_preserves_signed_losses_and_unavailable_totals(cuda_runtime, long, short, expected):
    from optimization.gpu.core_metrics import observation_catalog, finalize_observations

    name = "short_net_profit_contribution_fraction"
    packet = cuda_runtime.zeros((1, len(observation_catalog()["fields"])),
                                dtype=cuda_runtime.float32, device="cuda")
    profits = cuda_runtime.tensor([long, short], dtype=cuda_runtime.float32, device="cuda")
    _profit_library().core_profit_contribution_probe(profits, packet, threads=1)
    row = finalize_observations(packet, [name], origin_timestamp_ms=0,
                               observation_interval_ms=60_000, intended_end_timestamp_ms=60_000,
                               ew_half_life_days=180.0)[0]
    assert row[name] == expected
    if expected is None:
        assert row["metric_status"][name] == {
            "status": "undefined", "reason": "nonpositive_combined_net_profit"}
    else:
        assert name not in row["metric_status"]
    json.dumps(row, allow_nan=False)


def _red_delist_inputs(strategy, side, prior_partial):
    from config.hsl import generated_template
    from hsl_fixture import set_risk_memory_spans
    from test_gpu_entry_sizing_parity import _fixture

    coin_count = 2 if prior_partial else 1
    config, _, markets, _, timestamps = _fixture(side, coin_count, "initial")
    config = generated_template(config, "portfolio")
    config["bot"]["portfolio"]["strategy_kind"] = strategy
    config["bot"]["portfolio"]["execution"]["market_orders_allowed"] = False
    config["bot"]["portfolio"]["loss_gate"]["max_drawdown_fraction"] = 1.0
    config["backtest"].update(hsl_detailed_report=True, market_order_slippage_fraction=0.0,
                              maker_fee_override_fraction=0.0, taker_fee_override_fraction=0.0)
    config["bot"]["portfolio"]["hsl"].update(
        enabled=True, drawdown_threshold_fraction=.002,
        drawdown_fraction_ema_span_minutes=1.0, cooldown_after_red_minutes=10000.0,
        panic_close_order_type="limit")
    set_risk_memory_spans(config, 1440.0)
    for direction in ("long", "short"):
        bot = config["bot"][direction]
        bot["hsl"]["enabled"] = False
        bot["risk"]["total_exposure_enforcer"]["enabled"] = False
        bot["risk"]["position_exposure_enforcer"]["enabled"] = False
        bot["entry_cooldown"]["base_duration_minutes"] = 10000.0
        bot["strategy"]["ema_anchor"].update(
            base_qty_fraction=.5, offset_fraction=.01, ema_span_0_minutes=2.0,
            ema_span_1_minutes=3.0, inventory_offset_exposure_coefficient=0.0,
            entry_qty_exposure_coefficient=0.0)
        bot["strategy"]["trailing_martingale"]["entry"].update(
            initial_qty_fraction=.5, initial_ema_distance_fraction=.01,
            reentry_qty_multiplier=0.0, threshold_base_fraction=.5,
            retracement_base_fraction=0.0)
        bot["strategy"]["trailing_martingale"]["close"].update(
            threshold_base_fraction=.5, retracement_base_fraction=0.0)
    count = 1410
    candles = np.full((count, coin_count, 4), 100.0)
    candles[:, :, 3] = 1.0
    candles[3, :, 0], candles[3, :, 1] = 101.0, 99.0
    adverse, gap, final = (90.0, 80.0, 75.0) if side == "long" else (110.0, 120.0, 125.0)
    candles[4:, :, :3] = adverse
    # The first coin can execute a real panic at k=5. The last coin gaps beyond
    # its pending panic limit twice, then is force-closed at the close of k=6.
    terminal = coin_count - 1
    candles[5, terminal, :3], candles[6, terminal, :3] = gap, final
    candles[7:, terminal, :3] = np.nan
    terminal_coin = config["backtest"]["coins"]["bybit"][terminal]
    for coin, market in markets.items():
        if coin == "__meta__":
            continue
        market.update(last_valid_index=6 if coin == terminal_coin else count - 1,
                      qty_step=1.0, min_qty=1.0)
    timestamps = timestamps[0] + np.arange(count, dtype=np.int64) * 60_000
    return config, candles, markets, np.full(count, 50_000.0), timestamps


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("prior_partial", [False, True])
def test_cuda_forced_delist_while_red_keeps_only_actual_hsl_episode_losses(cuda_runtime, strategy, side, prior_partial):
    from backtest import build_backtest_payload, execute_backtest
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    inputs = _red_delist_inputs(strategy, side, prior_partial)
    config, candles, markets, btc, timestamps = inputs
    names = ("hsl_panic_episode_loss_max_fraction", "hsl_panic_episode_loss_mean_fraction")
    payload = build_backtest_payload(candles, markets, config, "bybit", btc, timestamps,
                                     metrics_only=False, metric_requests=names)
    fills, _, cpu = execute_backtest(payload, config)
    closes = [fill for fill in fills if str(fill[13]).startswith("close_")]
    assert len(closes) == 1 + int(prior_partial)
    assert all(str(fill[13]) == f"close_panic_{side}" for fill in closes)
    assert int(closes[-1][0]) == 6
    assert int(closes[-1][1]) == int(timestamps[6]) + 60_000
    report = payload.hard_stop_plot_data["hsl"]
    assert report["summary"]["triggers"] == 1
    assert report["summary"]["panic_close_fills"] == int(prior_partial)
    assert report["summary"]["red_minutes"] > 0
    assert any(sample["timestamp"] == int(timestamps[6]) + 60_000
               and sample["action"] in {"panic", "halted"} for sample in report["samples"])
    if prior_partial:
        assert int(closes[0][0]) == 5
        assert int(closes[0][1]) == int(timestamps[5])
        expected = -(float(closes[0][3]) + float(closes[0][4])) / 1000.0
        assert expected > 0
        # The forced-delisting loss is larger than the actual HSL loss. Including
        # it would change both aggregates rather than hiding below tolerance.
        assert -(float(closes[-1][3]) + float(closes[-1][4])) / 1000.0 > expected
    else:
        expected = 0.0
    assert cpu["metric_metadata"]["hsl_execution_episode_count"] == int(prior_partial)
    assert all(cpu[name] == expected for name in names)
    with _native_dataset(inputs, "bybit", names) as dataset:
        with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
            service.register_dataset("red-delist", dataset)
            gpu = service.submit(BacktestRequest("red-delist", "red-delist", {})).result()
    assert not gpu.liquidated
    assert gpu.metadata["metric_metadata"]["hsl_trigger_count"] == 1
    assert gpu.metadata["metric_metadata"]["hsl_execution_episode_count"] == int(prior_partial)
    for name in names:
        # Integer fills, zero fees and one observed episode leave one f32 ratio
        # rounding; this bound does not permit an extra fill or changed anchor.
        assert gpu.metrics[name] == pytest.approx(expected, abs=0.0, rel=np.finfo(np.float32).eps)
