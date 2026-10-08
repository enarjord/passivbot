import numpy as np
import pytest
import gc
import weakref
from threading import Event


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_underfilled_cuda_cohorts_learn_without_repeating_or_changing_results(monkeypatch, strategy):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    import backtest
    from optimization.gpu import autotune, execution_tuning
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    monkeypatch.setattr(autotune, "WINDOW", 2)
    monkeypatch.setattr(autotune, "MIN_SECONDS", 0)
    observations = []
    original = execution_tuning.ExecutionBatchTuner
    class ObservedTuner(original):
        def __init__(self, **kwargs):
            kwargs.setdefault("initial", 8)
            super().__init__(**kwargs)
        def observe(self, dataset_id, count, seconds, **kwargs):
            if self.enabled:
                observations.append((dataset_id, self.controllers[dataset_id].width, count))
            super().observe(dataset_id, count, seconds, **kwargs)
    monkeypatch.setattr(execution_tuning, "ExecutionBatchTuner", ObservedTuner)
    def forbidden(*_args, **_kwargs):
        pytest.fail("underfilled tuning must never run CPU simulations")
    for obj, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                      (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(obj, name, forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    parameter = "long_base_qty_pct" if strategy == "ema_anchor" else "long_entry_initial_qty_pct"
    cohorts = [[BacktestRequest(str(cohort * 3 + i), "b" if 6 <= cohort < 9 else "a",
                               {parameter: .005 + ((cohort + i) % 4) * .003})
                for i in range(3)] for cohort in range(12)]
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", hlcvs=specs[0], btc=specs[1],
            timestamps=specs[2], candle_coins=("COIN00", "COIN01", "COIN02"), metrics=DEFAULT_METRICS,
        )
        def evaluate(width):
            results = []
            with CudaBacktestService(batch_size=width, max_pending=16, max_batch_delay=.02) as service:
                for identity in ("a", "b"):
                    service.register_dataset(identity, dataset)
                for cohort in cohorts:
                    futures = [service.submit(request) for request in cohort]
                    results.extend(future.result(timeout=120) for future in futures)
            return results
        expected = evaluate(8)
        actual = evaluate(None)
        assert sum(count for _dataset, _width, count in observations) == 36
        for identity in ("a", "b"):
            stream = [(width, count) for dataset, width, count in observations if dataset == identity]
            assert (8, 3) in stream
            # Three-cohort production never supplies a full initial-width batch.
            # A completed warm window must still trigger a smaller-width probe.
            if identity == "a":
                assert any(width < 8 for width, _count in stream)
        for left, right in zip(expected, actual, strict=True):
            assert (left.request_id, left.dataset_id, left.liquidated) == (
                right.request_id, right.dataset_id, right.liquidated
            )
            assert left.metrics.keys() == right.metrics.keys()
            for name, value in left.metrics.items():
                np.testing.assert_allclose(right.metrics[name], value, rtol=0, atol=0, err_msg=name)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_adaptive_cuda_service_changes_width_without_changing_or_repeating_work(monkeypatch, strategy):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import backtest
    from optimization.gpu import autotune, execution_tuning
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    # Accelerate policy evidence windows only; execute the real CUDA kernels,
    # admission, shared-input preparation, reductions and host results.
    monkeypatch.setattr(autotune, "WINDOW", 2)
    monkeypatch.setattr(autotune, "MIN_SECONDS", 0)
    samples = []
    original = execution_tuning.ExecutionBatchTuner
    class ObservedTuner(original):
        def __init__(self, **kwargs):
            kwargs.setdefault("initial", 2)
            super().__init__(**kwargs)
        def observe(self, dataset_id, count, seconds, **kwargs):
            if self.enabled:
                samples.append((dataset_id, count))
            super().observe(dataset_id, count, seconds, **kwargs)
    monkeypatch.setattr(execution_tuning, "ExecutionBatchTuner", ObservedTuner)
    def forbidden(*_args, **_kwargs):
        pytest.fail("execution tuning must never run CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", hlcvs=specs[0], btc=specs[1],
            timestamps=specs[2], candle_coins=("COIN00", "COIN01", "COIN02"), metrics=DEFAULT_METRICS,
        )
        parameter = "long_base_qty_pct" if strategy == "ema_anchor" else "long_entry_initial_qty_pct"
        requests = [BacktestRequest(str(i), "b" if 32 <= i < 64 else "a",
                                   {parameter: 0.005 + (i % 4) * 0.003}) for i in range(96)]
        def evaluate(width):
            with CudaBacktestService(batch_size=width, max_pending=128, max_batch_delay=0.02) as service:
                for identity in ("a", "b"):
                    service.register_dataset(identity, dataset)
                futures = [service.submit(request) for request in requests]
                return [future.result(timeout=120) for future in futures]
        expected = evaluate(4)
        actual = evaluate(None)
        assert sum(count for _dataset, count in samples) == len(requests)
        for identity in ("a", "b"):
            widths = {count for dataset_id, count in samples if dataset_id == identity}
            assert 2 in widths and any(width >= 4 for width in widths)
        for left, right in zip(expected, actual, strict=True):
            assert (left.request_id, left.dataset_id, left.liquidated) == (
                right.request_id, right.dataset_id, right.liquidated
            )
            assert left.metrics.keys() == right.metrics.keys()
            for name, value in left.metrics.items():
                np.testing.assert_allclose(right.metrics[name], value, rtol=0, atol=0, err_msg=name)
    finally:
        manager.cleanup()


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("bound", ["work", "scratch"])
def test_prepared_service_bounds_actual_dispatch_and_releases_inactive_runners(monkeypatch, strategy, bound):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from optimization.gpu import mps_kernel, service as replay_module
    from optimization.gpu.datasets import PreparedGpuDataset
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from shared_arrays import SharedArrayManager
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    if bound == "scratch":
        cls = (mps_kernel.MpsEmaAnchorMulticoinFusedRunner if strategy == "ema_anchor"
               else mps_kernel.MpsTrailingMartingaleMulticoinFusedRunner)
        original_init = cls.__init__
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.hsl_scratch_budget_bytes = 2 * self._history_bytes_per_candidate()
        monkeypatch.setattr(cls, "__init__", initialize)
    original_evaluate = replay_module.MpsMulticoinProxy.evaluate_results
    observed, references = [], []
    submitted = Event()
    def evaluate(self, candidates):
        observed.append(len(candidates))
        return original_evaluate(self, candidates)
    monkeypatch.setattr(replay_module.MpsMulticoinProxy, "evaluate_results", evaluate)
    original_ceiling = CudaBacktestService._dispatch_ceiling
    def ceiling(replay):
        references.append(weakref.ref(replay.fused_runner))
        assert submitted.wait(10)
        return original_ceiling(replay)
    monkeypatch.setattr(CudaBacktestService, "_dispatch_ceiling", staticmethod(ceiling))
    manager = SharedArrayManager()
    try:
        specs = [manager.create_from(array)[0] for array in (candles, btc, timestamps)]
        dataset = PreparedGpuDataset(
            config=config, markets=markets, exchange="binance", hlcvs=specs[0], btc=specs[1],
            timestamps=specs[2], candle_coins=("COIN00", "COIN01", "COIN02"), metrics=DEFAULT_METRICS,
        )
        with CudaBacktestService(batch_size=8, max_pending=16, max_batch_delay=0.02,
                                 max_dispatch_candidate_bars=1536 if bound == "work" else 500_000_000) as service:
            service.register_dataset("a", dataset)
            service.register_dataset("b", dataset)
            futures = [service.submit(BacktestRequest(str(i), "a", {})) for i in range(7)]
            submitted.set()
            results = [future.result(timeout=120) for future in futures]
            assert observed == [2, 2, 2, 1]
            assert all(result.metrics == results[0].metrics for result in results)
            assert references[0]() is not None
            next_result = service.submit(BacktestRequest("b", "b", {})).result(timeout=120)
            assert next_result.metrics == results[0].metrics
            gc.collect()
            assert references[0]() is None
            assert references[1]() is not None
        gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        manager.cleanup()
