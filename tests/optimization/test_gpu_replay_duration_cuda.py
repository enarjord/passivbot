"""Actual native CUDA replays remain identical when duration changes boundaries."""
import numpy as np
import pytest


@pytest.fixture
def cuda_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    import backtest
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU simulation forbidden during adaptive CUDA replay")
    for obj, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                      (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(obj, name, forbidden)
    return torch


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", [("long",), ("short",), ("long", "short")])
def test_adaptive_boundaries_preserve_every_output_and_cache_reset(
    cuda_runtime, monkeypatch, strategy, sides,
):
    from optimization.gpu import mps_kernel
    from optimization.gpu.autotune import ReplayDurationController
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    proxy = make_proxy("unified", strategy, sides, minutes=256, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    assert runner.max_dispatch_candidate_bars is not None
    assert proxy.temporal_chunking
    candidates = [{"hsl_red_threshold": 1e-6}, {"hsl_red_threshold": 1e-6}]
    runner.max_dispatch_candidate_bars = None
    _, expected = raw(proxy, candidates)
    # A deliberately tiny execution target forces actual, successful GPU work
    # to shrink the next chunk. It is diagnostic policy, never simulator precision.
    monkeypatch.setattr(mps_kernel, "ReplayDurationController",
                        lambda ceiling: ReplayDurationController(ceiling, target_seconds=1e-9))
    runner.max_dispatch_candidate_bars = len(candidates) * runner.n_coins * len(sides) * 31
    _, adaptive = raw(proxy, candidates, profile=True)
    equal_outputs(expected, adaptive)
    assert runner.last_profile["adaptive_chunk_adjustments"] > 0
    assert runner.last_profile["temporal_chunk_bars_min"] == 1
    assert runner.last_profile["temporal_chunk_bars_max"] == 31
    assert runner.last_profile["dispatch_count"] > 8
    runner.max_dispatch_candidate_bars = None
    _, restored = raw(proxy, candidates[:1])
    equal_outputs({key: value[:1] if isinstance(value, cuda_runtime.Tensor) else value
                   for key, value in expected.items()}, restored)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_zero_work_candidate_finalizes_once_in_adaptive_native_replay(cuda_runtime, strategy):
    from test_gpu_hsl_multicoin import make_proxy, raw
    proxy = make_proxy("unified", strategy, ("long", "short"), minutes=128, factual_hsl=True)
    runner, output = raw(proxy, [{}], profile=True, end_steps=np.array([1], dtype=np.int32))
    assert runner.last_profile["dispatch_count"] == 1
    assert runner.last_profile["kernel_candidate_steps"] == 0
    assert runner.last_profile["temporal_chunk_bars_max"] == 0
    assert output["fill_count"].item() == 0


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_adaptive_disabled_hsl_preserves_unequal_candidate_endpoints(
    cuda_runtime, monkeypatch, strategy,
):
    from optimization.gpu import mps_kernel
    from optimization.gpu.autotune import ReplayDurationController
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs

    proxy = make_proxy("unified", strategy, ("long", "short"), minutes=256,
                       factual_hsl=True, enabled=False)
    runner = proxy.fused_runner
    candidates = [{}, {}, {}]
    ends = np.array([1, 73, runner.n - 1], dtype=np.int32)
    runner.max_dispatch_candidate_bars = None
    _, expected = raw(proxy, candidates, end_steps=ends)
    assert runner.hsl_fact_capacity == 0
    monkeypatch.setattr(mps_kernel, "ReplayDurationController",
                        lambda ceiling: ReplayDurationController(ceiling, target_seconds=1e-9))
    runner.max_dispatch_candidate_bars = len(candidates) * runner.n_coins * 2 * 31
    _, adaptive = raw(proxy, candidates, profile=True, end_steps=ends)
    equal_outputs(expected, adaptive)
    assert runner.last_profile["adaptive_chunk_adjustments"] > 0
    assert runner.last_profile["kernel_candidate_steps"] == int((ends - 1).sum())
    assert runner.last_profile["temporal_chunk_bars_min"] == 1
