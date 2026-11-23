"""Compare a full-width fused Trailing Martingale replay with Rust."""

from copy import deepcopy
import json

from test_gpu_close_sizing_replay import close_sizing_device


COINS = 64
BARS = 256
SIDES = ("long", "short")
CAPS = (0.0, 0.001, 0.01, 0.002)


def wide_tm_inputs():
    import numpy as np

    from config import prepare_config
    from config_utils import clean_config
    from tools.gpu_parity import build_parser, fixture_inputs

    args = build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", str(COINS),
        "--bars", str(BARS), "--seed", "73", "--hsl", "disabled",
    ])
    effective, candles, markets, btc, timestamps = fixture_inputs(args)
    coins = effective["backtest"]["coins"]
    config = clean_config(effective)
    config["optimize"]["bounds"] = {}
    loss = config["bot"]["portfolio"]["risk"]["loss_realization"]
    loss.update(
        memory_ema_span_minutes=96.0,
        loss_gate={"enabled": True, "max_realized_drawdown_fraction": CAPS[0]},
    )
    for side in SIDES:
        params = config["bot"][side]
        params["risk"]["unstuck"].update(
            enabled=True, ema_gating_enabled=False,
            exposure_trigger_ratio=0.4, loss_allowance_fraction=0.08,
        )
        params["entry_cooldown"]["base_duration_minutes"] = float(BARS)
        strategy = params["strategy"]["trailing_martingale"]
        strategy["entry"].update(
            ema_gate_mode="disabled", initial_qty_fraction=0.8,
            threshold_base_fraction=10.0, threshold_log_range_1h_coefficient=0.0,
            retracement_log_range_1h_coefficient=0.0,
        )
        strategy["close"].update(
            qty_fraction=0.15, threshold_base_fraction=0.06,
            threshold_exposure_ratio_coefficient=0.015,
            threshold_log_range_1h_coefficient=0.0,
            retracement_base_fraction=0.0, retracement_log_range_1h_coefficient=0.0,
        )
    config = prepare_config(config, verbose=False, target="canonical", runtime=None)
    config["backtest"]["coins"] = coins

    # All coins trade before two changes in which side is exposed to losses.
    price = np.broadcast_to(100.0 + 0.1 * np.arange(COINS), (BARS, COINS)).copy()
    price[96:, ::2] *= 0.86
    price[96:, 1::2] *= 1.14
    price[192:, ::2] *= 1.12
    price[192:, 1::2] *= 0.88
    candles[:, :, 0] = price * 1.004
    candles[:, :, 1] = price * 0.996
    candles[:, :, 2] = price
    return config, candles, markets, btc, timestamps


def test_fused_tm_close_sizing_matches_rust_64_coins(close_sizing_device):
    from backtest import run_backtest
    from metrics_schema import decode_metric_value
    from optimization.gpu.parity import compare_metrics
    from optimization.gpu.runtime import gpu_device
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import DEFAULT_METRICS, DEFAULT_TOLERANCES

    config, candles, markets, btc, timestamps = wide_tm_inputs()
    oracle = {}
    for cap in CAPS:
        candidate = deepcopy(config)
        loss = candidate["bot"]["portfolio"]["risk"]["loss_realization"]["loss_gate"]
        loss["max_realized_drawdown_fraction"] = cap
        fills, _, analysis = run_backtest(candles, markets, candidate, "binance", btc, timestamps)
        assert any(str(fill[13]).startswith("close_unstuck_") for fill in fills) == (cap > 0.0)
        oracle[cap] = (
            len(fills), {metric: decode_metric_value(analysis, metric) for metric in DEFAULT_METRICS},
        )

    replay = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange="binance", batch_size=len(CAPS), needed_metrics=DEFAULT_METRICS,
        max_dispatch_candidate_bars=len(CAPS) * COINS * len(SIDES) * 96,
        max_throughput=gpu_device() == "cuda",
    )
    assert replay.shared_account_fused
    results = replay.evaluate_results([{"loss_gate_max_drawdown_fraction": cap} for cap in CAPS])
    assert replay.fused_runner.n_coins == COINS
    assert replay.fused_runner.loss_gate_enabled and replay.fused_runner.dispatch_unstuck_enabled

    tolerances = {metric: DEFAULT_TOLERANCES[metric] for metric in DEFAULT_METRICS}
    failures = {}
    for cap, result in zip(CAPS, results, strict=True):
        cpu_fills, cpu_metrics = oracle[cap]
        gpu_fills = result.metadata["metric_metadata"]["fill_count"]
        comparison = compare_metrics(cpu_metrics, result.metrics, tolerances)
        if gpu_fills != cpu_fills or not comparison["passed"] or result.liquidated:
            failures[cap] = {
                "cpu_fills": cpu_fills,
                "gpu_fills": gpu_fills,
                "gpu_liquidated": result.liquidated,
                "metrics": comparison["metrics"],
            }
    assert not failures, json.dumps(failures, indent=2)
