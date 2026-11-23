"""Exercise active close sizing through the CUDA service and its candidate transport."""

from copy import deepcopy

import pytest


@pytest.fixture
def cuda_service():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    from optimization.gpu.native import CudaBacktestService

    return CudaBacktestService


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cuda_service_preserves_close_risk_candidates_without_cpu_fallback(
    strategy, cuda_service, monkeypatch
):
    import backtest
    from metrics_schema import decode_metric_value
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.parity import compare_metrics
    from rust_utils import verify_loaded_runtime_extension
    from test_gpu_unstuck_lookback import make_inputs
    from tools.gpu_parity import DEFAULT_METRICS, DEFAULT_TOLERANCES, _native_dataset

    fingerprint = verify_loaded_runtime_extension()
    assert not fingerprint.get("skipped")
    caps = (0.0, 0.005, 0.0)
    candles, markets, config, exchange, btc, timestamps = make_inputs(
        ("long", "short"), strategy=strategy, coins=2, loss_gate_cap=caps[0],
    )
    oracle = {}
    for cap in dict.fromkeys(caps):
        candidate = deepcopy(config)
        loss = candidate["bot"]["portfolio"]["risk"]["loss_realization"]
        loss["loss_gate"]["max_realized_drawdown_fraction"] = cap
        fills, _, analysis = backtest.run_backtest(candles, markets, candidate, exchange, btc, timestamps)
        assert any(str(fill[13]).startswith("close_unstuck_") for fill in fills)
        oracle[cap] = (
            len(fills), {metric: decode_metric_value(analysis, metric) for metric in DEFAULT_METRICS},
        )
    assert oracle[caps[0]][0] < oracle[caps[1]][0]

    def forbidden(*_args, **_kwargs):
        pytest.fail("CUDA service execution must not run a CPU backtest")

    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)

    inputs = config, candles, markets, btc, timestamps
    with _native_dataset(inputs, exchange, DEFAULT_METRICS) as dataset:
        with cuda_service(batch_size=len(caps), tuning_mode="off", max_dispatch_candidate_bars=96) as service:
            service.register_dataset("risk", dataset)
            futures = [
                service.submit(BacktestRequest(str(index), "risk", {"loss_gate_max_drawdown_fraction": cap}))
                for index, cap in enumerate(caps)
            ]
            results = [future.result(timeout=300) for future in futures]

    tolerances = {metric: DEFAULT_TOLERANCES[metric] for metric in DEFAULT_METRICS}
    for index, (cap, result) in enumerate(zip(caps, results, strict=True)):
        count, cpu_metrics = oracle[cap]
        assert result.request_id == str(index)
        assert result.dataset_id == "risk"
        assert not result.liquidated
        assert result.metadata["metric_metadata"]["fill_count"] == count
        comparison = compare_metrics(cpu_metrics, result.metrics, tolerances)
        assert comparison["passed"], comparison
    assert results[0].metrics == results[2].metrics
