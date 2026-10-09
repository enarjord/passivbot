"""Independent native-service parity for EMA's minimum-remainder boundary."""
import pytest
from optimization.gpu.parity import MetricTolerance
from tools import gpu_parity


def test_native_ema_minimum_remainder_has_matching_fills_and_practical_metrics():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for independent native parity")
    inputs = gpu_parity.fixture_inputs(gpu_parity.build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", "both", "--coins", "25",
        "--bars", "5760", "--seed", "7", "--hsl", "disabled",
    ]))
    config = inputs[0]
    for side in ("long", "short"):
        config["bot"][side]["strategy"]["ema_anchor"].update(
            base_qty_pct=0.01, ema_span_0=5.0,
        )
    policies = {
        "adg_strategy_eq": MetricTolerance(1e-7, 0.0),
        "drawdown_worst_strategy_eq": MetricTolerance(1e-6, 0.0),
        "fills_per_day": MetricTolerance(0.0, 0.0),
        "backtest_completion_ratio": MetricTolerance(0.0, 0.0),
        "hard_stop_time_in_red_pct": MetricTolerance(0.0, 0.0),
        "volume_pct_per_day_avg_w": MetricTolerance(1e-6, 0.0),
        "adg_strategy_eq_w": MetricTolerance(1e-7, 0.0),
    }
    # This nearly flat four-day curve has discontinuous strict recovery times.
    # Use a scoped five-minute measurement gate, not bitwise f32/f64 equality
    # or a new global tolerance for other strategies/scenarios.
    for metric in ("strategy_eq_recovery_days_mean", "strategy_eq_recovery_days_p95",
                   "strategy_eq_recovery_days_mean_worst_1pct"):
        policies[metric] = MetricTolerance(5.0 / 1440.0, 0.0)
    report = gpu_parity.run_comparison(
        inputs, "binance", tuple(policies), policies,
        diagnostics=True, gpu_engine="native",
    )
    assert report["diagnostics"]["cpu"]["fill_count"] == 1020
    assert report["gpu_engine"] == "native"
    assert report["gpu_replay"] == "shared_account"
    assert not report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert all(row["status"] == "match" for row in report["metrics"].values()), report
