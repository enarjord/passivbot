"""Per-fill percentile semantics without fill histories or CPU service fallbacks."""

import numpy as np
import pytest

from test_gpu_service_acceptance_cuda import cuda_runtime


@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coins", [2, 4])
def test_native_ema_fill_gap_p95_matches_cpu_when_candles_have_multiple_fills(
    cuda_runtime, sides, coins,
):
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", sides, "--coins", str(coins),
        "--bars", "2880", "--seed", "43",
    ]))
    metric = "fills_gap_p95_hours"
    report = run_comparison(inputs, "binance", (metric,),
                            {metric: MetricTolerance(1e-9, 1e-7)}, gpu_engine="native")
    assert report["metrics"][metric]["cpu"] > 0
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("coins", [1, 3])
def test_native_fill_gap_results_reuse_compact_counts_without_cpu_simulation(
    cuda_runtime, monkeypatch, strategy, coins,
):
    import backtest
    from optimization.gpu import metrics
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    def forbidden(*args, **kwargs):
        pytest.fail("native GPU requests must not run CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", str(coins),
        "--bars", "512", "--seed", "43",
    ]))
    observed = []
    original = metrics._fill_gap_metrics

    def capture(output, run):
        # Existing compact counts suffice: no per-fill or per-bar history added.
        assert output["gap_hist"].shape[1] == 128
        counts = output["gap_hist"].clone()
        result = original(output, run)
        cuda_runtime.testing.assert_close(output["gap_hist"], counts, rtol=0, atol=0)
        observed.extend((output["fill_count"] - counts.sum(1) - 1).tolist())
        return result
    monkeypatch.setattr(metrics, "_fill_gap_metrics", capture)
    names = ("fills_gap_p95_hours", "fills_gap_time_weighted_mean_hours")
    with _native_dataset(inputs, "binance", names) as dataset:
        with CudaBacktestService(batch_size=2, tuning_mode="off", max_batch_delay=0) as service:
            service.register_dataset("gaps", dataset)
            first = service.submit(BacktestRequest("first", "gaps", {})).result()
            repeated = [service.submit(BacktestRequest(str(i), "gaps", {}))
                        for i in range(3)]
            for future in repeated:
                assert future.result().metrics == first.metrics
            assert set(first.metrics) == set(names)
            assert all(np.isfinite(value) for value in first.metrics.values())
    assert observed and min(observed) >= 0
    assert max(observed) > 0, "fixture must exercise multiple fills in a candle"
