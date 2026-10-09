"""Real-device acceptance of incremental admission and bounded replay reuse."""

from copy import deepcopy
import gc
import json
import signal
import time
from concurrent.futures import TimeoutError
from threading import Event

import numpy as np
import pytest


@pytest.fixture
def cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    assert not getattr(passivbot_rust, "__is_stub__", False)
    verify_loaded_runtime_extension()
    return torch


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_incremental_cuda_admission_preserves_inputs_results_and_memory(
    cuda_runtime, monkeypatch, strategy
):
    import backtest
    from optimization.gpu import service as replay_module
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.executor import BacktestQueueFull, BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    torch = cuda_runtime
    def forbidden(*_args, **_kwargs):
        pytest.fail("native service acceptance must not run CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "512",
        "--unstuck",
    ]))
    config["live"].update(max_realized_loss_pct=0.1, pnls_max_lookback_days=8 / 1440)
    key = "long_base_qty_pct" if strategy == "ema_anchor" else "long_entry_initial_qty_pct"
    candidates = [{key: value} for value in (0.005, 0.015, 0.03, 0.045, 0.005)]
    reference = replay_module.MpsMulticoinProxy(
        config=config, mss=markets, hlcvs=candles, btc=btc, timestamps=timestamps,
        exchange="binance", needed_metrics=DEFAULT_METRICS, batch_size=1,
    )
    expected = reference.evaluate_results(candidates)
    assert expected[0].metrics["fills_per_day"] > 0
    assert expected[0].metrics != expected[3].metrics
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    # These pauses bracket real CUDA replays, without depending on kernel speed.
    # Holding the next replay proves a completed result is visible independently
    # of the rest of the cohort and frees admission capacity immediately.
    entered = [Event(), Event()]
    released = [Event(), Event()]
    calls = 0
    original_evaluate = replay_module.MpsMulticoinProxy.evaluate_results
    def evaluate(self, parameters):
        nonlocal calls
        index = calls
        calls += 1
        if index < 2:
            entered[index].set()
            if not released[index].wait(120):
                raise TimeoutError("acceptance replay gate was not released")
        return original_evaluate(self, parameters)
    monkeypatch.setattr(replay_module.MpsMulticoinProxy, "evaluate_results", evaluate)

    def matches(actual, index):
        assert actual.dataset_id == "fixture"
        assert actual.liquidated == expected[index].liquidated
        assert actual.metrics.keys() == expected[index].metrics.keys()
        for name, value in expected[index].metrics.items():
            np.testing.assert_array_equal(actual.metrics[name], value, err_msg=name)

    manager = SharedArrayManager()
    sources = [candles, btc, timestamps]
    snapshots = [array.copy() for array in sources]
    original_config, original_markets = deepcopy(config), deepcopy(markets)
    try:
        owned = [manager.create_from(array) for array in sources]
        specs = [item[0] for item in owned]
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", hlcvs=specs[0],
            btc=specs[1], timestamps=specs[2], candle_coins=("COIN00", "COIN01", "COIN02"),
            metrics=DEFAULT_METRICS,
        )
        config["backtest"]["starting_balance"] = 17
        markets["COIN00"]["price_step"] = 10
        assert json.loads(dataset.config_json) == original_config
        assert json.loads(dataset.markets_json) == original_markets
        with dataset.attach() as arrays:
            assert all(not array.flags.writeable for array in arrays)
            with pytest.raises(ValueError, match="read-only"):
                arrays[0][0, 0, 0] = 1

        with CudaBacktestService(batch_size=2, max_pending=4, max_batch_delay=0) as service:
            service.register_dataset("fixture", dataset)
            try:
                futures = [service.submit(BacktestRequest("first", "fixture", candidates[0]))]
                assert entered[0].wait(120)
                queued_parameters = dict(candidates[1])
                futures.append(service.submit(BacktestRequest("1", "fixture", queued_parameters)))
                futures += [service.submit(BacktestRequest(str(i), "fixture", candidates[i]))
                            for i in range(2, 4)]
                with pytest.raises(BacktestQueueFull):
                    service.submit(BacktestRequest("overflow", "fixture", {}))
                queued_parameters[key] = 0.9  # Mutate before this queued replay can be claimed.
                released[0].set()
                matches(futures[0].result(timeout=120), 0)
                assert entered[1].wait(120)
                assert all(not future.done() for future in futures[1:])
                futures.append(service.submit(BacktestRequest("replacement", "fixture", candidates[4])))
                with pytest.raises(BacktestQueueFull):
                    service.submit(BacktestRequest("still-full", "fixture", {}))
            finally:
                for event in released:
                    event.set()
            for index, future in enumerate(futures):
                result = future.result(timeout=120)
                assert result.request_id == ("first" if index == 0 else
                                             "replacement" if index == 4 else str(index))
                matches(result, index)

            def cycle(repeat):
                for count in (1, 4, 2, 3):
                    indices = [(repeat + index) % len(candidates) for index in range(count)]
                    pending = [service.submit(BacktestRequest(
                        f"{repeat}:{count}:{index}", "fixture", candidates[index],
                    )) for index in indices]
                    for index, future in zip(indices, pending, strict=True):
                        matches(future.result(timeout=120), index)

            # Warm all admitted shapes first. Later traffic must remain inside
            # that measured Torch allocation envelope, regardless of request count.
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(4):
                cycle(repeat)
            warm_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(4, 16):
                cycle(repeat)
            assert torch.cuda.max_memory_allocated() <= warm_peak

        for (_spec, array), snapshot in zip(owned, snapshots, strict=True):
            np.testing.assert_array_equal(array, snapshot)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_sigint_stops_large_clipped_hsl_request_without_losing_completed_results(
    cuda_runtime, monkeypatch, strategy,
):
    import backtest
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.residency import current_cuda_residency
    from optimization.interrupts import OptimizerInterruptLatch
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    def forbidden(*_args, **_kwargs):
        pytest.fail("large cancellation acceptance must not run CPU simulations")
    for owner, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                        (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(owner, name, forbidden)
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "long", "--coins", "25",
        "--bars", "11520", "--seed", "7", "--hsl", "unified",
        "--hsl-red-threshold", ".99", "--hsl-ema-span-minutes", "2.5",
        "--hsl-lookback-days", "1",
    ]))
    config, candles, _markets, _btc, _timestamps = inputs
    candles[:] = [100.1, 99.9, 100., 100.]
    candles[64, :, 1] = 97.
    for side in ("long", "short"):
        bot = config["bot"][side]["strategy"][strategy]
        if strategy == "ema_anchor":
            bot.update(ema_span_0=10., ema_span_1=20., offset=.02,
                       offset_psize_weight=0., offset_volatility_1m_weight=0.,
                       offset_volatility_1h_weight=0., base_qty_pct=.02,
                       entry_double_down_factor=1.)
        else:
            bot["entry"].update(ema_span_0=10., ema_span_1=20., initial_ema_dist=.01,
                                initial_qty_pct=.02, double_down_factor=1.,
                                threshold_base_pct=.9, threshold_we_weight=0.,
                                threshold_volatility_1m_weight=0.,
                                threshold_volatility_1h_weight=0.)
            bot["close"].update(qty_pct=1., threshold_base_pct=.5,
                                threshold_we_weight=0., threshold_volatility_1m_weight=0.,
                                threshold_volatility_1h_weight=0.)
    metrics = ("adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day",
               "position_held_hours_max", "hard_stop_time_in_red_pct")
    snapshots = [value.copy() for value in (candles, inputs[3], inputs[4])]
    active = False
    completed_bars = 0
    signalled = Event()
    spill_directory = None
    original_observe = mps_kernel.record_replay_chunk

    def observe(count, bars, total_bars, seconds, **kwargs):
        nonlocal completed_bars, spill_directory
        original_observe(count, bars, total_bars, seconds, **kwargs)
        if active:
            completed_bars += bars
            # Observe completed production commands, then send a real SIGINT
            # after the exposed episode's one-day boundary has started sliding.
            if completed_bars >= 1664 and not signalled.is_set():
                assert count == 1 and completed_bars < total_bars
                directory = current_cuda_residency()._directory
                spill_directory = directory.name if directory is not None else None
                signalled.set()
                signal.raise_signal(signal.SIGINT)
    monkeypatch.setattr(mps_kernel, "record_replay_chunk", observe)

    with _native_dataset(inputs, "binance", metrics) as dataset, OptimizerInterruptLatch() as latch:
        service = CudaBacktestService(batch_size=1, tuning_mode="off", max_batch_delay=0,
                                      interrupt_check=latch.raise_if_requested)
        try:
            service.register_dataset("held", dataset)
            first = service.submit(BacktestRequest("completed", "held", {"long_hsl_enabled": 0.}))
            retained = first.result(timeout=600)
            saved_metrics = dict(retained.metrics)
            assert retained.metrics["fills_per_day"] > 0
            active = True
            interrupted = service.submit(BacktestRequest("interrupted", "held", {}))
            deadline = time.monotonic() + 600
            with pytest.raises(KeyboardInterrupt):
                while True:
                    try:
                        interrupted.result(timeout=.05)
                    except TimeoutError:
                        if time.monotonic() >= deadline:
                            raise AssertionError("large GPU request did not reach its cancellation boundary")
                        continue
                    pytest.fail("interrupted replay supplied successful metrics")
            assert signalled.is_set() and latch.requested
            assert isinstance(interrupted.exception(), KeyboardInterrupt)
            assert first.result() is retained and dict(retained.metrics) == saved_metrics
        finally:
            # Also stop active work if an earlier harness assertion failed.
            if not latch.requested:
                signal.raise_signal(signal.SIGINT)
            service.close(cancel_pending=True)
        assert not service._executor._thread.is_alive()
        assert service._residency is None
        if spill_directory is not None:
            from pathlib import Path
            assert not Path(spill_directory).exists()
        with dataset.attach() as arrays:
            for observed, snapshot in zip(arrays, snapshots, strict=True):
                np.testing.assert_array_equal(observed, snapshot)
