"""Bounded native replay qualification against the v9 CPU contract."""

import pytest


def _require_device():
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")


def _save_evidence(name, report, inputs):
    import hashlib
    import json
    import os
    from pathlib import Path

    if destination := os.environ.get("PASSIVBOT_V9_EVIDENCE_DIR"):
        import numpy as np
        from rust_utils import verify_loaded_runtime_extension

        config, candles, markets, btc, timestamps = inputs
        report = dict(report)
        runtime = verify_loaded_runtime_extension()
        report["evidence_hashes"] = {
            "config": hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "markets": hashlib.sha256(json.dumps(markets, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "runtime_artifact": runtime["runtime_compiled_sha256"],
            "rust_source_fingerprint": runtime["runtime_compiled_source_stamp"],
            "data": {
                name: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                for name, array in (("candles", candles), ("btc", btc), ("timestamps", timestamps))
            },
        }
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("mode", ["coin", "pside", "portfolio"])
def test_v9_native_risk_scopes_use_the_real_cpu_contract(mode):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("actual supported Metal/CUDA execution required")
    from rust_utils import verify_loaded_runtime_extension
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    from optimization.gpu.parity import MetricTolerance

    verify_loaded_runtime_extension()
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "2",
        "--bars", "256", "--seed", "43", "--hsl", mode,
        "--hsl-red-threshold", ".01", "--hsl-drawdown-ratio-ema-span-minutes", "2.5",
        "--hsl-memory-ema-span-minutes", "30.5", "--hsl-cooldown-minutes", "2.0000083",
        "--price-shock", "0", "120", ".8", "--price-shock", "1", "170", "1.2",
        "--unstuck",
    ]))
    metrics = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day",
               "hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
               "hard_stop_time_in_red_pct")
    policies = {**DEFAULT_TOLERANCES,
                "hard_stop_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_restarts_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_time_in_red_pct": MetricTolerance(1e-8, 1e-5)}
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    _save_evidence(mode, report, inputs)
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["passed"], report


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
@pytest.mark.parametrize("loss_pct", [2e38, 1e36], ids=["product", "fused-result"])
def test_v9_native_allowance_overflow_rejects_the_candidate(strategy, sides, loss_pct):
    """Both single-side generation and fused preselection must publish failure."""
    _require_device()
    from copy import deepcopy
    import numpy as np
    from backtest import run_backtest
    from rust_utils import verify_loaded_runtime_extension
    from test_gpu_hsl_multicoin import raw
    from test_gpu_unstuck_lookback import make_proxy

    verify_loaded_runtime_extension()
    proxy, inputs = make_proxy(sides, strategy=strategy, coins=2)
    # The first exposure is filled at k=3. End this actual production dispatch
    # there: a missing post-generation status check cannot defer failure to k=4.
    end = np.array([4], dtype=np.int32)
    _, ordinary = raw(proxy, [{}], end_steps=end)
    assert ordinary["fill_count"].item() > 0, "the allowance consumer requires actual exposure"
    # The unchanged fixture has TWEL=2 and a positive ordinary balance. Each
    # supplied percentage is finite f32; the authored product or FMA overflows.
    with pytest.raises(ValueError, match="realized-loss memory or allowance"):
        raw(proxy, [{f"{sides[0]}_unstuck_loss_allowance_pct": loss_pct}], end_steps=end)
    candles, markets, config, exchange, btc, timestamps = inputs
    config = deepcopy(config)
    config["bot"][sides[0]]["unstuck"]["loss_allowance_pct"] = loss_pct
    with pytest.raises(
        ValueError,
        match=r"v9 (?:unstuck|realized-loss allowance) .*finite risk domain|global\.(?:unstuck_allowance|realized_pnl_drawdown_or_allowance)",
    ):
        run_backtest(candles, markets, config, exchange, btc, timestamps)


@pytest.mark.parametrize("strategy,mode,sides,coins,rebate,expected_stop", [
    ("ema_anchor", "coin", "long", 2, False, True),
    ("ema_anchor", "pside", "short", 2, False, False),
    ("ema_anchor", "portfolio", "both", 2, False, False),
    ("trailing_martingale", "coin", "long", 1, False, False),
    ("trailing_martingale", "pside", "short", 1, True, True),
    ("trailing_martingale", "disabled", "long", 2, False, False),
])
def test_v9_native_selected_shapes(strategy, mode, sides, coins, rebate, expected_stop):
    _require_device()
    from rust_utils import verify_loaded_runtime_extension
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    from optimization.gpu.parity import MetricTolerance

    verify_loaded_runtime_extension()
    recipe = [
        "--fixture", strategy, "--sides", sides, "--coins", str(coins),
        "--bars", "256", "--seed", "43", "--hsl", mode,
        "--hsl-red-threshold", ".01", "--hsl-drawdown-ratio-ema-span-minutes", "2.5",
        "--hsl-memory-ema-span-minutes", "30.5", "--hsl-cooldown-minutes", "2.0000083",
        "--price-shock", "0", "120", "1.2" if sides == "short" else ".8",
        "--unstuck",
    ]
    if coins == 2:
        recipe += ["--price-shock", "1", "170", "1.2"]
    inputs = fixture_inputs(build_parser().parse_args(recipe))
    if rebate:
        for market in inputs[2].values():
            if isinstance(market, dict) and "maker" in market:
                market["maker"] = -.0002
    metrics = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day",
               "hard_stop_triggers_per_year", "hard_stop_restarts_per_year",
               "hard_stop_time_in_red_pct")
    policies = {**DEFAULT_TOLERANCES,
                "hard_stop_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_restarts_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_time_in_red_pct": MetricTolerance(1e-8, 1e-5)}
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    _save_evidence(f"{strategy}-{mode}-{sides}-c{coins}-rebate{int(rebate)}", report, inputs)
    assert (report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0) == expected_stop
    assert report["passed"], report


@pytest.mark.parametrize("mode,sides", [("pside", "short"), ("portfolio", "both")])
def test_v9_native_ema_entry_fee_forces_a_stop_and_restart(mode, sides):
    """A small threshold makes fee economics an explicit forcing witness."""
    _require_device()
    from rust_utils import verify_loaded_runtime_extension
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison, DEFAULT_TOLERANCES
    from optimization.gpu.parity import MetricTolerance

    verify_loaded_runtime_extension()
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", sides, "--coins", "2",
        "--bars", "128", "--seed", "43", "--hsl", mode,
        "--hsl-red-threshold", "1e-8", "--hsl-drawdown-ratio-ema-span-minutes", "1",
        "--hsl-memory-ema-span-minutes", "30.5", "--hsl-cooldown-minutes", "2.0000083",
        "--unstuck",
    ]))
    metrics = ("drawdown_worst_strategy_eq", "fills_per_day", "hard_stop_triggers_per_year",
               "hard_stop_restarts_per_year", "hard_stop_time_in_red_pct")
    policies = {**DEFAULT_TOLERANCES,
                "hard_stop_triggers_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_restarts_per_year": MetricTolerance(1e-6, 1e-6),
                "hard_stop_time_in_red_pct": MetricTolerance(1e-8, 1e-5)}
    report = run_comparison(inputs, "binance", metrics, policies, gpu_engine="native")
    _save_evidence(f"ema-fee-forcing-{mode}-{sides}", report, inputs)
    assert report["metrics"]["hard_stop_triggers_per_year"]["cpu"] > 0
    assert report["metrics"]["hard_stop_restarts_per_year"]["cpu"] > 0
    assert report["passed"], report
