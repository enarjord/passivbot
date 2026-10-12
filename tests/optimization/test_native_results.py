from types import SimpleNamespace

import pytest

from config.schema import get_template_config
from metrics_schema import build_scenario_metrics, flatten_metric_stats
from optimization.gpu.executor import BacktestResult
from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
from optimize import Evaluator, SuiteEvaluator
from suite_runner import ScenarioResult, SuiteScenario


METRICS = ("adg_strategy_eq", "drawdown_worst_usd")


def evaluator(*, suite=False, selected=None):
    config = get_template_config()
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max", "scenario": selected},
        {"metric": "drawdown_worst_usd", "goal": "min", "scenario": None, "aggregate": "max"},
    ]
    config["optimize"]["limits"] = [
        {"metric": "drawdown_worst_usd", "penalize_if": "greater_than", "value": 0.2},
    ]
    base = Evaluator(hlcvs_specs={"binance": None, "bybit": None}, btc_usd_specs={}, msss={}, config=config)
    if not suite:
        return base
    contexts = [SimpleNamespace(label=label, exchanges=["binance", "bybit"])
                for label in ("base", "stress")]
    return SuiteEvaluator(base, contexts, {"default": "mean", "drawdown_worst_usd": "max"})


def slots(*labels):
    return [ResultSlot(f"{label}:{venue}", f"data:{label}:{venue}", label, venue, METRICS)
            for label in labels for venue in ("binance", "bybit")]


def result(slot, *, gain=0.01, drawdown=0.1, liquidated=False):
    return BacktestResult(slot.request_id, slot.dataset_id,
                          {"adg_strategy_eq": gain, "drawdown_worst_usd": drawdown}, liquidated)


def prohibit_simulation(monkeypatch, value):
    import backtest
    def forbidden(*_args, **_kwargs):
        pytest.fail("result consumption must not simulate a backtest")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(value, "evaluate", forbidden)


def test_incremental_candidate_completion_reuses_canonical_single_scoring(monkeypatch):
    value = evaluator()
    prohibit_simulation(monkeypatch, value)
    scorer = CanonicalResultScorer(value)
    graph = slots("base")
    pending = CandidateEvaluation("candidate", [0.5], graph, scorer)
    second = result(graph[1], gain=0.03, drawdown=0.4, liquidated=True)
    assert pending.add_result(second) is None
    assert pending.pending_request_ids == (graph[0].request_id,)
    second.metrics["adg_strategy_eq"] = 999
    first = result(graph[0])
    complete = pending.add_result(first)
    assert not pending.pending_request_ids
    payload = complete.require_full()
    analyses = {"binance": dict(first.metrics, liquidated=False),
                "bybit": {"adg_strategy_eq": 0.03, "drawdown_worst_usd": 0.4, "liquidated": True}}
    expected_stats = build_scenario_metrics(analyses)
    expected = value.calc_fitness(flatten_metric_stats(expected_stats["stats"]), return_raw_objectives=True)
    assert payload["fitness"] == expected[0]
    assert payload["constraint_violation"] == expected[1] > 0
    assert payload["metrics"]["stats"] == expected_stats["stats"]
    assert payload["metrics"]["objectives"] == expected[2]
    assert payload["metrics"]["liquidated"] is True
    assert payload["evaluation_vector"] == [0.5]
    assert pending._rows == {}
    with pytest.raises(RuntimeError, match="already been consumed"):
        pending.add_result(first)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 1, 0, 2)])
def test_suite_completion_uses_canonical_reducers_limits_and_objective_order(monkeypatch, order):
    value = evaluator(suite=True, selected="stress")
    prohibit_simulation(monkeypatch, value)
    prohibit_simulation(monkeypatch, value.base)
    graph = slots("base", "stress")
    rows = [result(slot, gain=0.01 * (index + 1), drawdown=0.1 * (index + 1))
            for index, slot in enumerate(graph)]
    pending = CandidateEvaluation("suite", [1, 2], graph, CanonicalResultScorer(value))
    for index in order[:-1]:
        assert pending.add_result(rows[index]) is None
    actual = pending.add_result(rows[order[-1]]).require_full()
    scenarios = []
    for label in ("base", "stress"):
        analyses = {slot.exchange: dict(row.metrics, liquidated=row.liquidated)
                    for slot, row in zip(graph, rows) if slot.scenario == label}
        scenarios.append(ScenarioResult(SuiteScenario(label, None, None, None, None), analyses,
                                        build_scenario_metrics(analyses), 0.0, None))
    expected = value.score_scenario_results(scenarios)
    assert actual["fitness"] == expected["objectives"]
    assert actual["constraint_violation"] == expected["constraint_violation"] > 0
    assert actual["metrics"]["suite_metrics"] == expected["suite_metrics"]
    assert actual["metrics"]["unpenalized_objectives"] == expected["unpenalized_objectives"]


def test_partial_suite_screening_is_scored_but_cannot_be_recorded_as_full(monkeypatch):
    value = evaluator(suite=True, selected="stress")
    prohibit_simulation(monkeypatch, value)
    scorer = CanonicalResultScorer(value)
    graph = slots("stress")
    pending = CandidateEvaluation("screened", [], graph, scorer, stage="screening")
    assert pending.add_result(result(graph[0])) is None
    complete = pending.add_result(result(graph[1]))
    assert complete.stage == "screening"
    assert complete.payload["metrics"]["suite_metrics"]
    with pytest.raises(ValueError, match="cannot be recorded"):
        complete.require_full()
    with pytest.raises(ValueError, match="retain explicitly selected"):
        CandidateEvaluation("wrong", [], slots("base"), scorer, stage="screening")
    with pytest.raises(ValueError, match="complete required"):
        CandidateEvaluation("incomplete", [], graph, scorer)


@pytest.mark.parametrize("problem", ["request", "dataset", "status", "missing", "nan"])
def test_invalid_result_does_not_satisfy_a_slot(problem):
    graph = slots("base")
    pending = CandidateEvaluation("candidate", [], graph, CanonicalResultScorer(evaluator()))
    changes = {
        "request": {"request_id": "other"}, "dataset": {"dataset_id": "other"},
        "status": {"liquidated": None}, "missing": {"metrics": {"adg_strategy_eq": 0.1}},
        "nan": {"metrics": {"adg_strategy_eq": float("nan"), "drawdown_worst_usd": 0.1}},
    }
    row = result(graph[0])
    row = BacktestResult(**(vars(row) | changes[problem]))
    with pytest.raises(RuntimeError):
        pending.add_result(row)
    assert pending.pending_request_ids == tuple(slot.request_id for slot in graph)


def test_duplicate_completion_before_fan_in_is_rejected():
    graph = slots("base")
    pending = CandidateEvaluation("candidate", [], graph, CanonicalResultScorer(evaluator()))
    assert pending.add_result(result(graph[0])) is None
    with pytest.raises(RuntimeError, match="already been consumed"):
        pending.add_result(result(graph[0]))


def test_one_candidate_can_complete_while_another_is_still_waiting(monkeypatch):
    from dataclasses import replace
    value = evaluator()
    prohibit_simulation(monkeypatch, value)
    scorer = CanonicalResultScorer(value)
    slow_slots = slots("base")
    fast_slots = [replace(slot, request_id="fast:" + slot.request_id) for slot in slow_slots]
    slow = CandidateEvaluation("slow", [0.1], slow_slots, scorer)
    fast = CandidateEvaluation("fast", [0.2], fast_slots, scorer)
    assert slow.add_result(result(slow_slots[0])) is None
    assert fast.add_result(result(fast_slots[1])) is None
    complete = fast.add_result(result(fast_slots[0]))
    assert complete.candidate_id == "fast"
    assert complete.require_full()["evaluation_vector"] == [0.2]
    assert slow.pending_request_ids == (slow_slots[1].request_id,)
    assert slow.add_result(result(slow_slots[1])).candidate_id == "slow"


@pytest.mark.parametrize("suite", [False, True])
def test_nonfinite_metric_sentinel_uses_canonical_invalid_candidate_policy(suite):
    value = evaluator(suite=suite)
    graph = slots("base", "stress") if suite else slots("base")
    pending = CandidateEvaluation("invalid", [], graph, CanonicalResultScorer(value))
    for slot in graph:
        complete = pending.add_result(result(slot, gain=float("inf"), liquidated=True))
    payload = complete.require_full()
    assert payload["constraint_violation"] > 0
    assert "non-finite metric" in payload["metrics"]["error"]
    assert payload["metrics"]["liquidated"] is True


@pytest.mark.parametrize("change", ["duplicate_request", "duplicate_pair", "missing_venue", "unknown", "metric"])
def test_invalid_evaluation_plan_is_rejected_before_simulation(change):
    graph = slots("base")
    if change == "duplicate_request":
        graph[1] = ResultSlot(graph[0].request_id, graph[1].dataset_id, "base", "bybit", METRICS)
    elif change == "duplicate_pair":
        graph[1] = ResultSlot(graph[1].request_id, graph[1].dataset_id, "base", "binance", METRICS)
    elif change == "missing_venue":
        graph.pop()
    elif change == "unknown":
        graph = slots("other")
    else:
        graph[1] = ResultSlot(graph[1].request_id, graph[1].dataset_id, "base", "bybit", ("adg_strategy_eq",))
    with pytest.raises(ValueError):
        CandidateEvaluation("invalid", [], graph, CanonicalResultScorer(evaluator()))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_candidate_is_rejected_before_simulation(value):
    with pytest.raises(ValueError, match="finite before simulation"):
        CandidateEvaluation("invalid", [value], slots("base"), CanonicalResultScorer(evaluator()))


def test_cpu_result_module_imports_without_optional_gpu_runtime():
    import os
    from pathlib import Path
    import subprocess
    import sys
    source = '''
import builtins
original = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cupy'} or name == 'optimization.gpu.runtime':
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = restricted
import optimization.native_results
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    completed = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
