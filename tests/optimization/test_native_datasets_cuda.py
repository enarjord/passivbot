from copy import deepcopy
import json
import time

import numpy as np
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("screen_first", [False, True])
async def test_canonical_lazy_suite_registry_drives_cuda_without_cpu_simulation(monkeypatch, screen_first):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import backtest
    import optimize_suite
    from optimize import Evaluator, SuiteEvaluator, config_to_individual
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    from optimization.native_datasets import NativeDatasetRegistry
    from optimization.native_session import NativeEvaluationSession
    from shared_arrays import SharedArrayManager
    from suite_runner import ExchangeDataset
    from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS
    from utils import ts_to_date

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "512",
    ]))
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    config["optimize"]["limits"] = []
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "base"},
        {"label": "window", "coins": ["COIN00", "COIN02"],
         "start_date": ts_to_date(int(timestamps[192])), "end_date": ts_to_date(int(timestamps[-1]) + 60_000)},
    ]
    manager = SharedArrayManager()
    try:
        # The source columns deliberately differ from canonical scenario coin order.
        order = [2, 0, 1]
        source = np.ascontiguousarray(candles[:, order, :])
        columns = [f"COIN{i:02d}" for i in order]
        candle_spec = manager.create_from(source)[0]
        btc_spec = manager.create_from(btc)[0]
        master = ExchangeDataset(
            exchange="combined", coins=columns, coin_index={coin: i for i, coin in enumerate(columns)},
            coin_exchange={coin: "binance" for coin in columns}, available_exchanges=["binance"],
            hlcvs=source, mss=markets, btc_usd_prices=btc, timestamps=timestamps, cache_dir="",
            hlcvs_spec=candle_spec, btc_spec=btc_spec,
        )
        async def offline(*_args, **_kwargs):
            return None
        async def prepared(*_args, **_kwargs):
            return {"combined": master}
        monkeypatch.setattr(optimize_suite, "load_markets", offline)
        monkeypatch.setattr(optimize_suite, "format_approved_ignored_coins", offline)
        monkeypatch.setattr(optimize_suite, "reject_cross_exchange_market_identifier_collisions", offline)
        monkeypatch.setattr(optimize_suite, "prepare_master_datasets", prepared)
        contexts, reducers = await optimize_suite.prepare_suite_contexts(
            config, optimize_suite.extract_suite_config(config, None), shared_array_manager=manager,
        )
        base_config = deepcopy(config)
        base_config["backtest"]["coins"] = deepcopy(contexts[0].config["backtest"]["coins"])
        base = Evaluator(contexts[0].hlcvs_specs, contexts[0].btc_usd_specs,
                         contexts[0].msss, base_config, timestamps=contexts[0].timestamps)
        suite = SuiteEvaluator(base, contexts, reducers)
        def forbidden(*_args, **_kwargs):
            pytest.fail("canonical registry/session execution must never run CPU simulations")
        monkeypatch.setattr(backtest, "execute_backtest", forbidden)
        monkeypatch.setattr(backtest, "run_backtest", forbidden)
        monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
        monkeypatch.setattr(base, "evaluate", forbidden)
        monkeypatch.setattr(suite, "evaluate", forbidden)
        with NativeDatasetRegistry(suite, metrics=DEFAULT_METRICS) as registry:
            reference = {}
            for binding, ctx in zip(registry.bindings, contexts, strict=True):
                assert binding.dataset.candle_coins == tuple(columns)
                start, end = ctx.time_slice["combined"]
                selected = ctx.config["backtest"]["coins"]["combined"]
                expected_columns = [int(coin.removeprefix("COIN")) for coin in selected]
                reference[binding.dataset_id] = MpsMulticoinProxy(
                    config=json.loads(binding.dataset.config_json), mss=ctx.msss["combined"], exchange="combined",
                    hlcvs=candles[start:end, expected_columns, :], btc=btc[start:end],
                    timestamps=timestamps[start:end], needed_metrics=DEFAULT_METRICS, batch_size=1,
                ).evaluate_results([{}])[0]
            assert registry.bindings[1].dataset.time_range[0] > 0
            assert registry.bindings[1].dataset.timestamp_range[0] == 0
            vector = config_to_individual(base_config, base.bounds, optimization_shape=base.optimization_shape)
            plan = registry.planner.prepare("candidate", vector)
            observed = []
            with CudaBacktestService(batch_size=2, max_pending=2) as service:
                registry.register(service)
                original_submit = service.submit
                def submit(request):
                    future = original_submit(request)
                    observed.append((request, future))
                    return future
                monkeypatch.setattr(service, "submit", submit)
                session = NativeEvaluationSession(service, registry.scorer)
                if screen_first:
                    session.admit(registry.planner.prepare("screen", vector, scenarios=["base"]))
                    screened = []
                    deadline = time.monotonic() + 60
                    while not screened and time.monotonic() < deadline:
                        screened = session.poll(timeout=0.05)
                    assert len(screened) == 1 and screened[0].stage == "screening"
                    with pytest.raises(ValueError, match="screening"):
                        screened[0].require_full()
                session.admit(plan)
                completed = []
                deadline = time.monotonic() + 60
                while not completed and time.monotonic() < deadline:
                    completed = session.poll(timeout=0.05)
                assert len(completed) == 1
                assert completed[0].require_full()["fitness"]
                assert session.pending_request_count == 0
                assert len(observed) == 2
                assert len({request.dataset_id for request, _future in observed}) == 2
                if screen_first:
                    assert {request.request_id for request, _future in observed} == {"screen:0", "candidate:1"}
                analyses = {binding.scenario: {binding.dataset.exchange: {
                    **reference[binding.dataset_id].metrics,
                    "liquidated":reference[binding.dataset_id].liquidated,
                }} for binding in registry.bindings}
                assert completed[0].require_full() == registry.scorer.score(plan.vector, analyses)
            for request, future in observed:
                actual = future.result()
                assert actual.metrics == pytest.approx(reference[request.dataset_id].metrics, rel=1e-7, abs=1e-9)
                assert actual.liquidated == reference[request.dataset_id].liquidated
    finally:
        manager.cleanup()
