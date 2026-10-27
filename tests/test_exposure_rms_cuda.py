"""Real CUDA acceptance, including dispatch-resume and existing metric parity."""
import pytest

from test_exposure_rms import NAMES, fixture_inputs, run_cpu


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    return torch


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("scenario", ["ordinary", "budgets", "flat"])
def test_native_cuda_rms_matches_cpu_and_temporal_chunks(cuda, strategy, sides, scenario):
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset
    inputs = fixture_inputs(strategy, sides, scenario)
    _, _, _, analysis = run_cpu(inputs)
    metrics = (*NAMES, "gross_turnover_ratio_per_day", "cagr_fraction_strategy_equity", "backtest_completion_fraction")
    results = []
    for budget in (500_000_000, 1024):
        with _native_dataset(inputs, "binance", metrics) as dataset:
            with CudaBacktestService(batch_size=1, tuning_mode="off",
                                     max_dispatch_candidate_bars=budget) as service:
                service.register_dataset("rms", dataset)
                result = service.submit(BacktestRequest("rms", "rms", {})).result()
                results.append(result.metrics)
    for name in NAMES:
        assert results[0][name] == pytest.approx(analysis[name], abs=1e-5, rel=1e-4), name
    assert results[0]["backtest_completion_fraction"] == pytest.approx(analysis["backtest_completion_fraction"], abs=1e-6)
    # Same encoded arithmetic in the same order: dispatch boundaries add no drift.
    assert results[1] == results[0]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_cuda_omits_unrequested_rms_without_changing_existing_metrics(cuda, strategy, sides):
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset
    inputs = fixture_inputs(strategy, sides, "ordinary")
    existing = ("gross_turnover_ratio_per_day", "cagr_fraction_strategy_equity", "backtest_completion_fraction")
    results = []
    for metrics in (existing, (*existing, *NAMES)):
        with _native_dataset(inputs, "binance", metrics) as dataset:
            with CudaBacktestService(batch_size=1, tuning_mode="off",
                                     max_dispatch_candidate_bars=1024) as service:
                service.register_dataset("optional", dataset)
                result = service.submit(BacktestRequest("optional", "optional", {})).result()
                results.append(result.metrics)
    assert not set(NAMES) & results[0].keys()
    assert set(NAMES) <= results[1].keys()
    assert results[0] == {name: results[1][name] for name in results[0]}
