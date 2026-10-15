from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from optimization.gpu.datasets import PreparedGpuDataset
from optimization.native_planning import NativeCandidatePlanner, ScenarioBinding
from optimize import Evaluator, SuiteEvaluator, config_to_individual
from shared_arrays import SharedArraySpec
from tools.gpu_parity import build_parser, fixture_inputs, DEFAULT_METRICS


def inputs(*, suite=False, fixed=False, mirror=False):
    config, candles, markets, _btc, _timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["bounds"][f"{side}_entry_initial_qty_pct"] = [0.01, 0.05]
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
    ]
    config["optimize"]["limits"] = []
    if fixed:
        config["optimize"]["fixed_runtime_overrides"] = {
            "bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.025,
        }
    if mirror:
        config["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    spec = SharedArraySpec("candles", candles.shape, candles.dtype.str)
    base = Evaluator({"binance": spec}, {}, {"binance": markets}, config)
    contexts = [SimpleNamespace(label=label, exchanges=["binance"], config=deepcopy(config), msss={"binance": markets},
                                overrides={"bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.03}
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
    values[[name for name, _path in planner.base.key_paths].index(key)] = value
    return values


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


def test_mirrored_shadow_genes_deduplicate_effectively():
    planner, vector = inputs(mirror=True)
    first = planner.prepare("first", set_value(planner, vector, "short_entry_initial_qty_pct", 0.01))
    second = planner.prepare("second", set_value(planner, vector, "short_entry_initial_qty_pct", 0.05))
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
