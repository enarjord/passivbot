"""Real CUDA input streaming preserves replay values, state and ownership."""

from copy import deepcopy
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from test_gpu_metal_tick_windows import (
    TICK_WINDOW_METRICS,
    _assert_tick_window_output_exact,
    _tick_window_fixture,
)


@pytest.fixture
def cuda_streaming_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    from rust_utils import verify_loaded_runtime_extension
    import backtest

    verified = verify_loaded_runtime_extension()
    assert not verified.get("skipped")

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA input transport qualification must not invoke CPU simulations")

    for owner, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                        (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(owner, name, forbidden)
    return torch


@pytest.mark.parametrize("strategy,sides,hsl,chunk_bars", [
    ("trailing_martingale", "long", "disabled", 17),
    ("trailing_martingale", "short", "disabled", 61),
    ("trailing_martingale", "both", "disabled", 17),
    ("trailing_martingale", "long", "coin", 61),
    ("trailing_martingale", "both", "coin", 17),
    ("trailing_martingale", "short", "portfolio", 17),
    ("trailing_martingale", "both", "portfolio", 61),
    ("ema_anchor", "long", "disabled", 17),
    ("ema_anchor", "short", "coin", 61),
    ("ema_anchor", "both", "portfolio", 17),
])
def test_cuda_streamed_ticks_and_compact_hours_preserve_every_output_and_reset(
    cuda_streaming_runtime, monkeypatch, strategy, sides, hsl, chunk_bars,
):
    from optimization.gpu import mps_kernel, residency, service as replay_module
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    torch = cuda_streaming_runtime
    inputs = _tick_window_fixture(sides, hsl, strategy)
    assert int(inputs[4][0] // 60_000 % 60) != 0
    snapshots = [array.copy() for array in (inputs[1], inputs[3], inputs[4])]
    original_prepare = replay_module._prepared_multicoin_data
    original_run = mps_kernel._MulticoinReplayRunner.run
    mode = {"streamed": False}
    prepared_modes, raw_outputs, dispatches, spill_directories = [], [], [], []

    def prepare(*args, **kwargs):
        # Force the constrained-memory choice on this small, identical fixture.
        # The reference disables transport only, retaining every replay input.
        assert kwargs["stream_tick_inputs"] is True
        if not mode["streamed"]:
            kwargs["stream_tick_inputs"] = False
            kwargs["compact_hourly_ranges"] = False
        result = original_prepare(*args, **kwargs)
        prepared_modes.append(bool(result["stream_tick_inputs"]))
        relation = np.load(result["touch_min_qty_relation"], mmap_mode="r", allow_pickle=False)
        try:
            assert relation.dtype == np.int8
        finally:
            relation._mmap.close()
        return result

    def capture(runner, *args, **kwargs):
        assert runner.bars.device.type == "cuda"
        result = original_run(runner, *args, **kwargs)
        raw_outputs.append({key: (value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                                 else deepcopy(value)) for key, value in result.items()})
        dispatches.append(deepcopy(runner._last_temporal_dispatch))
        assert bool(runner._streamed_tick_paths) is mode["streamed"]
        assert runner.compact_hourly_ranges is mode["streamed"]
        owner = residency.current_cuda_residency()
        spill_directories.append(owner._directory.name)
        return result

    monkeypatch.setattr(residency, "prefer_streamed_ticks", lambda *args, **kwargs: True)
    monkeypatch.setattr(replay_module, "_prepared_multicoin_data", prepare)
    monkeypatch.setattr(mps_kernel._MulticoinReplayRunner, "run", capture)
    sides_count = 2 if sides == "both" else 1
    results = []
    with _native_dataset(inputs, "binance", TICK_WINDOW_METRICS) as dataset:
        for streamed in (False, True):
            mode["streamed"] = streamed
            with CudaBacktestService(
                batch_size=1, max_pending=1, tuning_mode="off",
                max_dispatch_candidate_bars=chunk_bars * 2 * sides_count,
            ) as worker:
                worker.register_dataset("market", dataset)
                for cycle in range(2):
                    results.append(worker.submit(BacktestRequest(
                        f"{streamed}-{cycle}", "market", {},
                    )).result(timeout=180))
            assert worker._residency is None and not worker._prepared_cache
            assert not worker._executor._thread.is_alive()
        with dataset.attach() as arrays:
            for observed, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(observed, snapshot)
    assert prepared_modes == [False, True]
    assert len(raw_outputs) == 4
    for other in raw_outputs[1:]:
        _assert_tick_window_output_exact(torch, raw_outputs[0], other)
    for other in results[1:]:
        assert other.liquidated is results[0].liquidated
        assert {key: float(value).hex() for key, value in other.metrics.items()} == {
            key: float(value).hex() for key, value in results[0].metrics.items()
        }
        assert other.metadata == results[0].metadata
    assert all(item["dispatch_count"] > 1 for item in dispatches)
    assert all(item["temporal_chunk_bars"] == chunk_bars for item in dispatches)
    assert all(not item.get("streamed_tick_inputs", False) for item in dispatches[:2])
    assert all(item["streamed_tick_inputs"] is True for item in dispatches[2:])
    bytes_per_coin = 16 if strategy == "ema_anchor" else 25
    assert all(item["tick_window_bytes"] == (chunk_bars + 2) * 2 * bytes_per_coin
               for item in dispatches[2:])
    assert all(not Path(directory).exists() for directory in spill_directories)


def test_cuda_streaming_interruption_preserves_completed_result_and_releases_owner(
    cuda_streaming_runtime, monkeypatch,
):
    from optimization.gpu import mps_kernel, residency
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    requested = Event()
    active = False
    completed_chunks = 0
    spill_directory = None
    original_observe = mps_kernel.record_replay_chunk

    def interrupt_check():
        if requested.is_set():
            raise KeyboardInterrupt("completed CUDA input-window boundary")

    def observe(*args, **kwargs):
        nonlocal completed_chunks, spill_directory
        original_observe(*args, **kwargs)
        if active:
            completed_chunks += 1
            owner = residency.current_cuda_residency()
            spill_directory = owner._directory.name
            requested.set()

    monkeypatch.setattr(residency, "prefer_streamed_ticks", lambda *args, **kwargs: True)
    monkeypatch.setattr(mps_kernel, "record_replay_chunk", observe)
    inputs = _tick_window_fixture("both", "portfolio")
    snapshots = [array.copy() for array in (inputs[1], inputs[3], inputs[4])]
    with _native_dataset(inputs, "binance", TICK_WINDOW_METRICS) as dataset:
        worker = CudaBacktestService(
            batch_size=1, max_pending=1, tuning_mode="off", max_batch_delay=0,
            max_dispatch_candidate_bars=17 * 2 * 2, interrupt_check=interrupt_check,
        )
        try:
            worker.register_dataset("market", dataset)
            first = worker.submit(BacktestRequest("completed", "market", {}))
            retained = first.result(timeout=180)
            saved_metrics = dict(retained.metrics)
            active = True
            interrupted = worker.submit(BacktestRequest("interrupted", "market", {}))
            with pytest.raises(KeyboardInterrupt, match="input-window boundary"):
                interrupted.result(timeout=180)
            assert requested.is_set() and completed_chunks == 1
            assert isinstance(interrupted.exception(), KeyboardInterrupt)
            assert first.result() is retained and dict(retained.metrics) == saved_metrics
        finally:
            requested.set()
            worker.close(cancel_pending=True)
        assert not worker._executor._thread.is_alive()
        assert worker._residency is None and not worker._prepared_cache
        assert spill_directory is not None and not Path(spill_directory).exists()
        with dataset.attach() as arrays:
            for observed, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(observed, snapshot)
