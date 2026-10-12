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
    metrics = (*NAMES, "fills_per_day", "adg_strategy_eq", "backtest_completion_ratio")
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
    assert results[0]["backtest_completion_ratio"] == pytest.approx(analysis["backtest_completion_ratio"], abs=1e-6)
    # Same encoded arithmetic in the same order: dispatch boundaries add no drift.
    assert results[1] == results[0]


_PROBE = r"""
kernel void rms_state_bytes(device int* result) {
    result[0] = sizeof(ExposureRmsState);
}
kernel void rms_trace(constant float* values, device ExposureRmsState* saved,
    device float* output, constant int& begin, constant int& end,
    uint b [[thread_position_in_grid]]) {
    if (b > 0) return;
    ExposureRmsState state = begin == 0 ? init_exposure_rms() : saved[0];
    for (int i = begin; i < end; ++i) {
        float e = values[i];
        record_exposure_rms(state, e, 0.5f * e,
            exposure_rms_position_moment(e, 0.5f, 2.0f),
            exposure_rms_position_moment(0.5f * e, 0.25f, 1.0f), 2.0f, 1.0f);
    }
    saved[0] = state;
    finish_exposure_rms(state, output, 0);
}
"""


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_cuda_compensated_rms_trace_matches_encoded_reference_and_resume(cuda, strategy):
    import numpy as np
    import passivbot_rust
    from optimization.gpu.runtime import compile_shader
    source = getattr(passivbot_rust, f"mps_{strategy}_multicoin_source_py")()
    source = ("#define PASSIVBOT_EXPOSURE_RMS_ENABLED 1\n"
              "#define PASSIVBOT_HSL_CAPACITY 1\n#define PASSIVBOT_HSL_TREE_SIZE 1\n"
              "#define PASSIVBOT_HSL_LOOKBACK 0\n" + source + _PROBE)
    library = compile_shader(source, cuda_coin_capacity=2)
    # Large samples followed by tiny samples expose float32 sum stagnation.
    values = np.r_[np.ones(1000), np.full(100_000, 0.001)].astype(np.float32)
    payload = cuda.tensor(values, device="cuda")
    size = cuda.empty(1, dtype=cuda.int32, device="cuda")
    library.rms_state_bytes(size, threads=1)
    saved = cuda.empty(int(size.item()), dtype=cuda.uint8, device="cuda")
    output = cuda.empty(7, device="cuda")
    library.rms_trace(payload, saved, output, 0, len(values), threads=1)
    whole = output.clone()
    expected = np.sqrt(np.mean(values.astype(np.float64) ** 2))
    factors = np.array([1, 0.5, 1.5, 0.5, 0.5, 1, 1])
    assert whole.cpu().numpy() == pytest.approx(expected * factors, rel=2e-7, abs=1e-8)
    cuts = [0, 97, 1001, 30_000, len(values)]
    for begin, end in zip(cuts, cuts[1:]):
        library.rms_trace(payload, saved, output, begin, end, threads=1)
    assert cuda.equal(output, whole)
    library.rms_trace(payload, saved, output, 0, 0, threads=1)
    assert cuda.isnan(output).all()
