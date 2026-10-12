from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import optimize
from optimization.native_datasets import NativeDatasetRegistry
from test_native_backend import guard_cpu, inputs, managed_arrays
from test_native_datasets import contexts


def coupled_inputs(manager, strategy, *, pinned=True, coupled=True):
    from tools.gpu_parity import build_parser, fixture_inputs

    original = inputs(manager)
    config = deepcopy(original.config)
    fixture, *_ = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config["live"]["strategy_kind"] = strategy
    for side in ("long", "short"):
        config["bot"][side]["strategy"] = deepcopy(fixture["bot"][side]["strategy"])
        config["bot"][side]["unstuck"].update(
            enabled=True, ema_gating_enabled=True, threshold=0.0, loss_allowance_pct=0.1,
        )
    config["optimize"]["bounds"] = {
        key: value for key, value in config["optimize"]["bounds"].items()
        if "initial_qty_pct" not in key
    }
    config["optimize"]["bounds"]["long_ema_span_0"] = [2.0, 20.0]
    config["optimize"]["enable_overrides"] = ["couple_unstuck_ema_spans"] if coupled else []
    config["coin_overrides"] = {"COIN01": {"bot": {"long": {"unstuck": {"close_pct": 0.05}}}}}
    if pinned:
        pin = {"ema_span_0": 7.5}
        if strategy == "trailing_martingale":
            pin = {"entry": pin}
        config["coin_overrides"]["COIN02"] = {"bot": {"long": {"strategy": {strategy: pin}}}}
    if not coupled:
        config["coin_overrides"]["COIN01"]["bot"]["long"]["unstuck"]["ema_span_0"] = 9.0
    return optimize.Evaluator(original.hlcvs_specs, original.btc_usd_specs, original.msss,
                              config, timestamps=original.timestamps)


def candidate(base, value):
    config = deepcopy(base.config)
    strategy = config["bot"]["long"]["strategy"][config["live"]["strategy_kind"]]
    if config["live"]["strategy_kind"] == "trailing_martingale":
        strategy = strategy["entry"]
    strategy["ema_span_0"] = value
    return optimize.config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape)


def registry_for(base, suite):
    evaluator = base
    if suite:
        prepared = contexts(base, base.timestamps["binance"], lazy=True)
        kind = base.config["live"]["strategy_kind"]
        prepared[1].overrides = {f"bot.long.strategy.{kind}.ema_span_1": 3.5}
        evaluator = optimize.SuiteEvaluator(base, prepared, {"default": "mean"})
    return NativeDatasetRegistry(evaluator, standalone_candle_coins={
        "binance": ("COIN00", "COIN01", "COIN02"),
    })


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_coupled_coin_spans_reuse_views_but_preserve_request_identity(monkeypatch, strategy, suite, pinned):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = coupled_inputs(manager, strategy, pinned=pinned)
        with registry_for(base, suite) as registry:
            low = registry.planner.prepare("low", candidate(base, 2.0))
            high = registry.planner.prepare("high", candidate(base, 20.0))
            assert len(registry.bindings) == (2 if suite else 1)
            assert low.effective_key != high.effective_key
            assert [row.dataset_id for row in low.requests] == [row.dataset_id for row in high.requests]
            assert all(row.parameters["long_unstuck_ema_span_0"] == 2.0 for row in low.requests)
            assert all(row.parameters["long_unstuck_ema_span_0"] == 20.0 for row in high.requests)
            if suite:
                assert low.requests[1].parameters["long_unstuck_ema_span_1"] == 3.5
                assert high.requests[1].parameters["long_unstuck_ema_span_1"] == 3.5
            registered = []
            registry.register(SimpleNamespace(register_dataset=lambda *_args: registered.append(_args)))
            for _identity, dataset in registered:
                runtime = json.loads(dataset.config_json)
                assert "optimize" not in runtime and "_optimizer_anchor" not in runtime
                inherited = runtime["coin_overrides"]["COIN01"]["bot"]["long"]["unstuck"]
                assert "ema_span_0" not in inherited and "ema_span_1" not in inherited
                if pinned:
                    pin = runtime["coin_overrides"]["COIN02"]["bot"]["long"]["unstuck"]
                    assert pin["ema_span_0"] == 7.5 and "ema_span_1" not in pin
            if pinned:
                pin = base.config["coin_overrides"]["COIN02"]["bot"]["long"]["strategy"][strategy]
                if strategy == "trailing_martingale":
                    pin = pin["entry"]
                pin["ema_span_0"] = 8.5
                with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                    registry.planner.prepare("changed-pin", candidate(base, 2.0))


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_uncoupled_coin_spans_remain_dataset_owned(monkeypatch, strategy):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = coupled_inputs(manager, strategy, coupled=False)
        with registry_for(base, False) as registry:
            registry.planner.prepare("original", candidate(base, 2.0))
            base.config["coin_overrides"]["COIN01"]["bot"]["long"]["unstuck"]["ema_span_0"] = 10.0
            with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                registry.planner.prepare("changed", candidate(base, 2.0))


def test_coupling_policy_cannot_change_a_prepared_execution_view(monkeypatch):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = coupled_inputs(manager, "trailing_martingale")
        with registry_for(base, False) as registry:
            base.config["optimize"]["enable_overrides"] = []
            registry.planner.overrides = ()
            with pytest.raises(ValueError, match="dataset-owned execution inputs"):
                registry.planner.prepare("uncoupled", candidate(base, 2.0))


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("suite", [False, True])
def test_cuda_coupled_requests_match_independent_effective_replays(monkeypatch, strategy, suite):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy

    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = coupled_inputs(manager, strategy)
        with registry_for(base, suite) as registry:
            with CudaBacktestService(batch_size=2) as service:
                registry.register(service)
                for value in (2.0, 20.0, 2.0):
                    plan = registry.planner.prepare(str(value), candidate(base, value))
                    config = optimize._canonicalize_optimizer_individual(
                        list(plan.vector), base.config, base.bounds, base.sig_digits,
                        base.key_paths, registry.planner.overrides,
                    )
                    for request in plan.requests:
                        binding = next(row for row in registry.bindings if row.dataset_id == request.dataset_id)
                        effective = config
                        if suite:
                            evaluator = registry.scorer.evaluator
                            context = next(ctx for ctx in evaluator.contexts if ctx.label == binding.scenario)
                            effective = evaluator.build_scenario_candidate_config(config, context)
                        dataset = binding.dataset
                        with dataset.attach() as (candles, btc, timestamps):
                            selected_candles = candles[:, dataset.coin_indices, :]
                            replay = MpsMulticoinProxy(config=effective, mss=json.loads(dataset.markets_json),
                                exchange=dataset.exchange, hlcvs=selected_candles, btc=btc, timestamps=timestamps,
                                needed_metrics=dataset.metrics, batch_size=1)
                            reference = replay.evaluate_results([{}])[0]
                            del replay
                        actual = service.submit(request).result(timeout=60)
                        assert actual.metrics == reference.metrics
                        assert actual.liquidated == reference.liquidated
