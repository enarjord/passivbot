"""Real-device acceptance of incremental admission and bounded replay reuse."""

from copy import deepcopy
import gc
import json
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
