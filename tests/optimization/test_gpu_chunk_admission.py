"""Chunk failure admission and scheduling without a GPU or simulated trading."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from optimization.gpu import mps_kernel
from optimization.gpu.autotune import history_dispatch_ceiling


def _dispatch(monkeypatch, markers, device="cuda"):
    runner = object.__new__(mps_kernel.MpsTrailingMartingaleMulticoinRunner)
    runner.bars = SimpleNamespace(device=SimpleNamespace(type=device))
    runner.native_factual_hsl = False
    runner.max_dispatch_candidate_bars = 4
    runner.n_coins = 1
    runner._replay_state_bytes = 16
    runner._replay_states = {}
    runner.hsl_scratch_budget_bytes = 1024
    runner._history_batch_bytes = lambda count: count * 16
    runner.interrupt_check = lambda: None
    monkeypatch.setattr(mps_kernel, "gpu_device", lambda *_: "cpu")
    monkeypatch.setattr(mps_kernel, "synchronize", lambda: None)
    monkeypatch.setattr(mps_kernel, "record_replay_chunk", lambda *args, **kwargs: None)
    scalars = torch.zeros((2, 10))
    scalars[:, 9] = -1
    calls = []
    def kernel(*args, **kwargs):
        calls.append(kwargs)
        scalars[:, 9] = torch.tensor(markers)
    library = SimpleNamespace(**{runner.replay_kernel_name: kernel})
    def run():
        runner._dispatch_replay(library, (), torch.tensor([101, 101]),
                                batch_size=2, scalars=scalars)
    return runner, calls, run


def test_overflow_rejects_at_first_completed_chunk(monkeypatch):
    runner, calls, run = _dispatch(monkeypatch, [-6., -1.])
    with pytest.raises(mps_kernel.HslFactHistoryOverflow):
        run()
    assert len(calls) == 1
    assert runner._last_temporal_dispatch["rejected_at_bar"] == 2
    assert calls[0]["group_size"] == (1, 1, 1)


@pytest.mark.parametrize("fatal", [-2., -3., -4., -5., -7.])
def test_current_mixed_batch_failure_precedes_early_capacity_retry(monkeypatch, fatal):
    _, calls, run = _dispatch(monkeypatch, [-6., fatal])
    with pytest.raises((RuntimeError, ValueError)) as error:
        run()
    assert not isinstance(error.value, mps_kernel.HslFactHistoryOverflow)
    assert len(calls) == 1


@pytest.mark.parametrize("device,block", [("cuda", 1), ("mps", 2)])
def test_chunk_geometry_preserves_apple_and_spreads_cuda(monkeypatch, device, block):
    _, calls, run = _dispatch(monkeypatch, [-1., -1.], device)
    run()
    assert len(calls) == 50
    assert all(call["group_size"] == (block, 1, 1) for call in calls)


def test_outer_admission_reserves_shared_workspace_once():
    runner = object.__new__(mps_kernel.MpsTrailingMartingaleMulticoinRunner)
    runner.n, runner.n_days = 2_000_000, 1400
    runner.weighted_equity_metrics = {"adg_strategy_eq_w"}
    runner._history_bytes_per_candidate = lambda: runner.n * 4 + 8
    runner.hsl_scratch_budget_bytes = 512 * 1024**2
    limit = runner._history_batch_limit()
    assert limit == 47
    assert runner._history_batch_bytes(limit) <= runner.hsl_scratch_budget_bytes
    assert runner._history_batch_bytes(limit + 1) > runner.hsl_scratch_budget_bytes
    assert history_dispatch_ceiling(SimpleNamespace(runner=runner), 500) == limit


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("mode", ["coin", "unified"])
@pytest.mark.parametrize("sides", [("long",), ("long", "short")])
def test_cuda_independent_blocks_preserve_every_output(monkeypatch, strategy, mode, sides):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import numpy as np
    from rust_utils import verify_loaded_runtime_extension
    from optimization.gpu.cuda_kernel import CudaShaderLibrary
    from test_gpu_hsl_multicoin import make_proxy, raw
    from test_gpu_side_specialization import equal_outputs
    verify_loaded_runtime_extension()
    proxy = make_proxy(mode, strategy, sides, minutes=256, factual_hsl=True)
    runner = proxy.fused_runner or proxy.runners[sides[0]]
    candidates = [{}, {"long_entry_initial_qty_pct": .03}, {}]
    ends = np.array([1, 137, 256], dtype=np.int32)
    _, expected = raw(proxy, candidates, end_steps=ends)
    expected = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in expected.items()}
    # Reject an actual producer overflow between chunks, grow worker storage,
    # and reconstruct from the beginning before publishing accepted results.
    runner.hsl_fact_capacity = runner.hsl_fact_capacity_learned = 1
    _, recovered = raw(proxy, candidates, end_steps=ends)
    assert runner.last_hsl_fact_retries > 0
    equal_outputs(expected, recovered)
    original = CudaShaderLibrary.__getattr__
    def prior_geometry(self, name):
        launch = original(self, name)
        if name != runner.replay_kernel_name:
            return launch
        def packed(*args, **kwargs):
            kwargs["group_size"] = (min(len(candidates), 32), 1, 1)
            return launch(*args, **kwargs)
        return packed
    monkeypatch.setattr(CudaShaderLibrary, "__getattr__", prior_geometry)
    _, previous = raw(proxy, candidates, end_steps=ends)
    equal_outputs(expected, previous)
