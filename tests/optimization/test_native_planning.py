from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from config import prepare_config
from config_utils import clean_config
from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
from optimize import Evaluator, SuiteEvaluator, config_to_individual
from shared_arrays import SharedArraySpec
from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS


def inputs(*, suite=False, fixed=False, mirror=False):
    config, candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config["optimize"]["bounds"] = {
        side: {
            "risk": {
                key: [config["bot"][side]["risk"][key]] * 2
                for key in ("max_positions", "total_wallet_exposure_limit")
            },
            "strategy": {"trailing_martingale": {"entry": {"initial_qty_fraction": [0.01, 0.05]}}},
        }
        for side in ("long", "short")
    }
    config["optimize"]["scoring"] = [
        {"metric": "adg_fraction_strategy_equity", "goal": "max"},
        {"metric": "drawdown_max_fraction_strategy_equity", "goal": "min"},
    ]
    config["optimize"]["limits"] = []
    if fixed:
        config["optimize"]["fixed_runtime_overrides"] = {
            "bot.long.strategy.trailing_martingale.entry.initial_qty_fraction": 0.025,
        }
    if mirror:
        config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    coins = deepcopy(config["backtest"]["coins"])
    config = prepare_config(clean_config(config), verbose=False)
    config["backtest"]["coins"] = coins
    spec = SharedArraySpec("candles", candles.shape, candles.dtype.str)
    base = Evaluator({"binance": spec}, {}, {"binance": markets}, config)
    contexts = [SimpleNamespace(label=label, exchanges=["binance"], config=deepcopy(config), msss={"binance": markets},
                                overrides={"bot.long.strategy.trailing_martingale.entry.initial_qty_fraction": 0.03}
                                if label == "stress" else {}) for label in ("base", "stress")]
    evaluator = SuiteEvaluator(base, contexts, {"default": "mean"}) if suite else base
    datasets = []
    for label in ("base", "stress") if suite else ("base",):
        effective = evaluator.build_scenario_candidate_config(config, contexts[label == "stress"]) if suite else config
        dataset = PreparedGpuDataset(
            config=effective, markets=markets, exchange="binance", hlcvs=spec,
            btc=SharedArraySpec("btc", (len(candles),), "<f8"),
            timestamps=SharedArraySpec("timestamps", (len(candles),), "<i8"),
            candle_coins=config["backtest"]["coins"]["binance"], metrics=DEFAULT_METRICS,
        )
        datasets.append(ScenarioBinding(label, label, dataset))
    planner = NativeCandidatePlanner(evaluator, datasets)
    vector = config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape)
    return planner, vector


def set_value(planner, vector, key, value):
    values = list(vector)
    from optimization.config_adapter import resolve_optimization_bound_path
    path = resolve_optimization_bound_path(planner.base.config, key)
    values[[tuple(target) for _name, target in planner.base.key_paths].index(path)] = value
    return values


MEMORY_PATHS = {
    "unstuck_memory_ema_span_minutes": (
        "bot.portfolio.unstuck.memory_ema_span_minutes", ("portfolio", "unstuck", "memory_ema_span_minutes"),
    ),
    "loss_gate_memory_ema_span_minutes": (
        "bot.portfolio.loss_gate.memory_ema_span_minutes", ("portfolio", "loss_gate", "memory_ema_span_minutes"),
    ),
}


def global_span_inputs(memory_key, *, suite=False, fixed=False, scenario_pin=False):
    previous, _ = inputs(suite=suite)
    config = deepcopy(previous.base.config)
    override_path, bound_path = MEMORY_PATHS[memory_key]
    from optimization.warmup import _apply_config_overrides
    _apply_config_overrides(config, {
        ".".join(("optimize", "bounds", *bound_path)): [1000.0, 50000.0],
    })
    if fixed:
        config["optimize"]["fixed_runtime_overrides"] = {override_path: 4096.5}
    base = Evaluator(previous.base.hlcvs_specs, previous.base.btc_usd_specs,
                     previous.base.msss, config)
    evaluator = base
    if suite:
        contexts = deepcopy(previous.scorer.evaluator.contexts)
        for context in contexts:
            context.config = deepcopy(config)
            if scenario_pin and context.label == "stress":
                context.overrides[override_path] = 2048.5
        evaluator = SuiteEvaluator(base, contexts, {"default": "mean"})
    bindings = []
    for binding in previous.bindings:
        original = binding.dataset
        effective = (evaluator.build_scenario_candidate_config(
            config, next(context for context in contexts if context.label == binding.scenario)
        ) if suite else config)
        dataset = PreparedGpuDataset(
            config=effective, markets=json.loads(original.markets_json), exchange=original.exchange,
            hlcvs=original.hlcvs, btc=original.btc, timestamps=original.timestamps,
            candle_coins=original.candle_coins, coin_indices=original.coin_indices,
            time_range=original.time_range, metrics=original.metrics,
        )
        assert dataset.hlcvs == original.hlcvs and dataset.btc == original.btc
        assert dataset.timestamps == original.timestamps
        bindings.append(replace(binding, dataset=dataset))
    assert [binding.dataset_id for binding in bindings] == [binding.dataset_id for binding in previous.bindings]
    planner = NativeCandidatePlanner(evaluator, bindings)
    vector = config_to_individual(config, base.bounds, optimization_shape=base.optimization_shape)
    return planner, vector


@pytest.mark.parametrize("memory_key", MEMORY_PATHS)
@pytest.mark.parametrize("suite", [False, True])
def test_global_memory_span_gene_reuses_dataset_and_changes_effective_requests(memory_key, suite):
    planner, vector = global_span_inputs(memory_key, suite=suite)
    low_vector = set_value(planner, vector, memory_key, 1000.0)
    high_vector = set_value(planner, vector, memory_key, 50000.0)
    low = planner.prepare("low", low_vector)
    high = planner.prepare("high", high_vector)
    repeated = planner.prepare("repeated", low_vector)
    assert low.effective_key == repeated.effective_key != high.effective_key
    assert [r.dataset_id for r in low.requests] == [r.dataset_id for r in high.requests]
    for plan, expected in ((low, 1000.0), (high, 50000.0)):
        assert len(plan.requests) == (2 if suite else 1)
        for request in plan.requests:
            assert request.parameters[memory_key] == expected
    other_memory = next(key for key in MEMORY_PATHS if key != memory_key)
    for low_request, high_request in zip(low.requests, high.requests, strict=True):
        assert low_request.parameters[other_memory] == high_request.parameters[other_memory]
        for side in ("long", "short"):
            hsl_key = f"{side}_hsl_memory_ema_span_minutes"
            assert low_request.parameters[hsl_key] == high_request.parameters[hsl_key]


@pytest.mark.parametrize("memory_key", MEMORY_PATHS)
def test_fixed_global_memory_span_deduplicates_shadow_gene(memory_key):
    planner, vector = global_span_inputs(memory_key, fixed=True)
    low = planner.prepare("low", set_value(planner, vector, memory_key, 1000.0))
    high = planner.prepare("high", set_value(planner, vector, memory_key, 50000.0))
    assert low.effective_key == high.effective_key
    assert low.requests[0].parameters == high.requests[0].parameters
    assert low.requests[0].parameters[memory_key] == 4096.5


@pytest.mark.parametrize("memory_key", MEMORY_PATHS)
def test_last_scenario_global_memory_span_override_is_request_owned(memory_key):
    planner, vector = global_span_inputs(memory_key, suite=True, scenario_pin=True)
    low = planner.prepare("low", set_value(planner, vector, memory_key, 1000.0))
    high = planner.prepare("high", set_value(planner, vector, memory_key, 50000.0))
    assert low.effective_key != high.effective_key
    assert [r.dataset_id for r in low.requests] == [r.dataset_id for r in high.requests]
    assert low.requests[1].parameters == high.requests[1].parameters
    assert low.requests[0].parameters[memory_key] == 1000.0
    assert high.requests[0].parameters[memory_key] == 50000.0
    assert low.requests[1].parameters[memory_key] == 2048.5


@pytest.mark.parametrize("memory_key", MEMORY_PATHS)
@pytest.mark.parametrize(
    "path,value",
    [
        ("live.startup.risk_bootstrap_max_lookback_minutes", 64800.0),
        ("bot.portfolio.loss_gate.max_drawdown_ratio", 0.5),
        ("bot.portfolio.execution.market_orders_allowed", True),
        ("backtest.starting_balance", 2000.0),
    ],
)
def test_global_memory_span_gene_does_not_relax_other_dataset_inputs(memory_key, path, value):
    planner, vector = global_span_inputs(memory_key)
    planner.base.config["optimize"]["fixed_runtime_overrides"] = {path: value}
    with pytest.raises(ValueError, match="dataset-owned execution inputs"):
        planner.prepare("changed-static", set_value(planner, vector, memory_key, 1000.0))


def test_fixed_runtime_policy_and_exact_last_scenario_override_are_materialized_once(monkeypatch):
    import backtest
    planner, vector = inputs(suite=True, fixed=True)
    def forbidden(*_args, **_kwargs):
        pytest.fail("candidate planning must not run CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(planner.base, "evaluate", forbidden)
    monkeypatch.setattr(planner.scorer.evaluator, "evaluate", forbidden)
    first = planner.prepare("first", set_value(planner, vector, "long_entry_initial_qty_pct", 0.01))
    second = planner.prepare("second", set_value(planner, vector, "long_entry_initial_qty_pct", 0.05))
    assert first.effective_key == second.effective_key
    assert [row.parameters["long_entry_initial_qty_pct"] for row in first.requests] == [0.025, 0.03]
    assert first.stage == "full"
    assert {slot.scenario for slot in first.slots} == {"base", "stress"}
    with pytest.raises(TypeError):
        first.requests[0].parameters["long_entry_initial_qty_pct"] = 1


def test_mirrored_runtime_pins_deduplicate_effectively():
    planner, vector = inputs(mirror=True)
    short_path = "bot.short.strategy.trailing_martingale.entry.initial_qty_fraction"
    assert short_path not in {key for key, _ in planner.base.key_paths}
    planner.base.config["optimize"]["fixed_runtime_overrides"] = {short_path: 0.01}
    first = planner.prepare("first", vector)
    planner.base.config["optimize"]["fixed_runtime_overrides"] = {short_path: 0.05}
    second = planner.prepare("second", vector)
    assert first.effective_key == second.effective_key
    row = first.requests[0].parameters
    assert row["long_entry_initial_qty_pct"] == row["short_entry_initial_qty_pct"]


def test_screening_identity_includes_unscreened_effective_work():
    planner, vector = inputs(suite=True)
    first = planner.prepare("first", set_value(planner, vector, "long_entry_initial_qty_pct", 0.01), scenarios=["stress"])
    second = planner.prepare("second", set_value(planner, vector, "long_entry_initial_qty_pct", 0.05), scenarios=["stress"])
    assert first.requests[0].parameters == second.requests[0].parameters
    assert first.effective_key != second.effective_key
    assert first.stage == "screening"
    full = planner.prepare("first", first.vector)
    assert full.effective_key == first.effective_key
    assert first.slots[0] == full.slots[1]


@pytest.mark.parametrize("problem", ["shape", "nan", "unknown", "empty", "execution"])
def test_invalid_candidate_or_incompatible_dataset_is_rejected_before_submission(problem):
    planner, vector = inputs(suite=True)
    scenarios = None
    if problem == "shape":
        vector.pop()
    elif problem == "nan":
        vector[0] = float("nan")
    elif problem == "unknown":
        scenarios = ["stress", "missing"]
    elif problem == "empty":
        scenarios = []
    else:
        planner.base.config["optimize"]["fixed_runtime_overrides"] = {"backtest.starting_balance": 2000}
    with pytest.raises(ValueError):
        planner.prepare("invalid", vector, scenarios=scenarios)


@pytest.mark.parametrize("change", ["disable", "enable"])
def test_candidate_side_topology_changes_require_compatible_prepared_dataset(change):
    planner, vector = inputs(suite=True)
    if change == "disable":
        planner.base.config["optimize"]["fixed_runtime_overrides"] = {
            "bot.short.risk.total_wallet_exposure_limit": 0,
        }
    else:
        bindings = []
        for binding in planner.bindings:
            dataset = binding.dataset
            config = json.loads(dataset.config_json)
            config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] = 0
            bindings.append(replace(binding, dataset=PreparedGpuDataset(
                config=config, markets=json.loads(dataset.markets_json), exchange=dataset.exchange,
                hlcvs=dataset.hlcvs, btc=dataset.btc, timestamps=dataset.timestamps,
                candle_coins=dataset.candle_coins, coin_indices=dataset.coin_indices,
                time_range=dataset.time_range, metrics=dataset.metrics,
            )))
        planner = NativeCandidatePlanner(planner.scorer.evaluator, bindings)
    with pytest.raises(ValueError, match="dataset-owned execution inputs"):
        planner.prepare("different-side-topology", vector)
