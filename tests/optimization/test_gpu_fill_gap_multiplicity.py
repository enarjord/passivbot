"""Per-fill percentile semantics without fill histories or CPU service fallbacks."""

import numpy as np
import pytest

from test_gpu_service_acceptance_cuda import cuda_runtime


@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coins", [2, 4])
def test_native_ema_max_fill_gap_matches_cpu_when_candles_have_multiple_fills(
    cuda_runtime, sides, coins,
):
    from optimization.gpu.parity import MetricTolerance
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "ema_anchor", "--sides", sides, "--coins", str(coins),
        "--bars", "2880", "--seed", "43",
    ]))
    metric = "fills_gap_max_days"
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
    cache_interaction = strategy == "trailing_martingale" and coins == 3
    physical_batches = []
    logical_batches = []
    admitted_caches = []
    if cache_interaction:
        from config import prepare_config
        from config.hsl import generated_template
        from optimization.gpu.mps_kernel import MpsTrailingMartingaleMulticoinRunner
        inputs = list(inputs)
        config = inputs[0]
        candle_coins = list(config["backtest"]["coins"]["binance"])
        config = generated_template(config, "portfolio")
        config["bot"]["portfolio"]["hsl"].update(enabled=True,
            drawdown_threshold_fraction=1., drawdown_fraction_ema_span_minutes=2.5,
            cooldown_after_red_minutes=5)
        inputs[0] = prepare_config(config,verbose=False,target="canonical",runtime=None)
        # Prepared candle columns retain their original order across canonical
        # configuration normalization, which removes this derived metadata.
        inputs[0]["backtest"]["coins"] = {"binance": candle_coins}
        layout = MpsTrailingMartingaleMulticoinRunner._prepare_replay_state_layout
        dispatch = MpsTrailingMartingaleMulticoinRunner._dispatch_replay
        history_batches = MpsTrailingMartingaleMulticoinRunner._run_history_batches

        def one_candidate_scratch(self,library,args):
            layout(self,library,args)
            # Include the compiled state and exact-entry reduction workspace;
            # two logical requests must split without reducing their histories.
            self.hsl_scratch_budget_bytes = 2*self._history_bytes_per_candidate()-1

        def capture_dispatch(self,*args,**kwargs):
            physical_batches.append(kwargs["batch_size"])
            admitted_caches.append((self.hsl_native_cache_enabled,
                                    self._hsl_native_cache_capacity() > 0))
            assert kwargs["batch_size"] == 1
            return dispatch(self,*args,**kwargs)

        def capture_history_batches(self, params, **kwargs):
            logical_batches.append(len(params))
            return history_batches(self, params, **kwargs)

        monkeypatch.setattr(MpsTrailingMartingaleMulticoinRunner,
            "_run_history_batches", capture_history_batches)
        monkeypatch.setattr(MpsTrailingMartingaleMulticoinRunner,
            "_prepare_replay_state_layout",one_candidate_scratch)
        monkeypatch.setattr(MpsTrailingMartingaleMulticoinRunner,
            "_dispatch_replay",capture_dispatch)
    # A repeated batched request must preserve the selected compact packet and
    # real fill counts without invoking CPU simulation or allocating histories.
    from optimization.gpu.mps_kernel import _MulticoinReplayRunner
    observed = []
    original_run = _MulticoinReplayRunner.run
    def capture(self, *args, **kwargs):
        output = original_run(self, *args, **kwargs)
        assert "gap_hist" not in output
        assert "entry_interval_native_metrics" not in output
        assert output["core_observations"].ndim == 2
        observed.extend(output["fill_count"].detach().cpu().tolist())
        return output
    monkeypatch.setattr(_MulticoinReplayRunner, "run", capture)
    names = ("fills_gap_max_days", "gross_turnover_ratio_per_day", "backtest_completion_fraction")
    baseline = None
    for cache in ((False,True) if cache_interaction else (None,)):
        if cache_interaction:
            monkeypatch.setattr(MpsTrailingMartingaleMulticoinRunner,"hsl_native_cache_enabled",cache)
        with _native_dataset(inputs, "binance", names) as dataset:
            with CudaBacktestService(batch_size=2, tuning_mode="off",
                    max_batch_delay=.01 if cache_interaction else 0) as service:
                service.register_dataset("gaps", dataset)
                first_future = service.submit(BacktestRequest("first", "gaps", {}))
                if cache_interaction:
                    # Queue both initial requests before the first compiled ABI
                    # teaches service admission the one-candidate scratch limit.
                    initial_peer = service.submit(BacktestRequest("initial-peer", "gaps", {}))
                first = first_future.result()
                if cache_interaction:
                    assert initial_peer.result().metrics == first.metrics
                repeated = [service.submit(BacktestRequest(str(i), "gaps", {}))
                            for i in range(3)]
                for future in repeated:
                    assert future.result().metrics == first.metrics
                assert set(first.metrics) == set(names)
                assert all(np.isfinite(value) for value in first.metrics.values())
        if baseline is None:
            baseline = first.metrics
        else:
            assert first.metrics == baseline
    if cache_interaction:
        assert physical_batches and set(physical_batches)=={1}
        assert 2 in logical_batches, "two logical requests must exercise scratch splitting"
        assert (False,False) in admitted_caches and (True,True) in admitted_caches
        assert all(enabled == admitted for enabled,admitted in admitted_caches)
    assert observed and min(observed) > 0
