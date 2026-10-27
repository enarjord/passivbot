"""Core optimizer orchestration; native backtest parity is validated separately."""

from types import SimpleNamespace
import json
import math
import pytest
from config.scoring import normalize_scoring_entries, to_engine_value
from config.limits import normalize_limit_entries
from config.metrics import CORE_METRIC_DEFINITION_VERSION
from limit_utils import expand_limit_checks
from metrics_schema import (
    MetricAggregationError,
    encode_metric_values,
    build_scenario_metrics,
)
from optimize import Evaluator, SuiteEvaluator, _build_invalid_candidate_metrics
from optimization.native_results import CanonicalResultScorer
from suite_runner import ScenarioResult, SuiteScenario

CAGR = "cagr_fraction_strategy_equity"
OMEGA = "omega_ratio_strategy_equity"


def evaluator(metrics=(CAGR,), limits=()):
    result = Evaluator.__new__(Evaluator)
    result._attachments = {}
    result.scoring_specs = normalize_scoring_entries(
        [{"metric": name, "goal": "max"} for name in metrics]
    )[0]
    result.limit_checks = expand_limit_checks(
        normalize_limit_entries(list(limits)), {}, penalty_weight=1.0
    )
    result.config = {
        "optimize": {"scoring": [spec.to_config() for spec in result.scoring_specs]}
    }
    result.exchanges = ["venue"]
    return result


def source(values):
    return {
        "metric_metadata": {"definition_version": 1},
        "liquidated": False,
        **encode_metric_values(values),
    }


def scenario(label, values):
    return ScenarioResult(
        scenario=SuiteScenario(label, None, None, None, None),
        per_exchange={},
        metrics=build_scenario_metrics(
            {"venue": source(values)}, required_metrics=values
        ),
        elapsed_seconds=0.0,
        output_path=None,
    )


def suite(base, labels=("one", "two"), reducer="mean"):
    result = SuiteEvaluator.__new__(SuiteEvaluator)
    result.base = base
    result.contexts = [
        SimpleNamespace(label=label, exchanges=["venue"]) for label in labels
    ]
    result.reducer_cfg = {"default": reducer}
    result.objective_bases = [
        SimpleNamespace(scenario=None, reducer=reducer) for _ in base.scoring_specs
    ]
    result.objective_scenario = None
    return result


def test_raw_reduction_precedes_common_adapter_and_preserves_json_status():
    base = evaluator((OMEGA,))
    result = suite(base).score_scenario_results(
        [scenario("one", {OMEGA: 1.0}), scenario("two", {OMEGA: math.inf})]
    )
    assert result["objectives"] == (-1.0,)
    assert result["raw_objectives"][OMEGA] is None
    assert (
        result["raw_objectives"]["metric_status"][OMEGA]["status"]
        == "positive_infinity"
    )
    json.dumps(result, allow_nan=False)


def test_cpu_fitness_uses_only_raw_limits_and_finite_adapter():
    base = evaluator((OMEGA,))
    fitness, penalty, raw = base.calc_fitness(
        encode_metric_values({OMEGA + "_mean": math.inf}), return_raw_objectives=True
    )
    assert fitness == (-1.0,) and penalty == 0.0 and raw[OMEGA] == math.inf
    base = evaluator(
        (OMEGA,), [{"metric": OMEGA, "penalize_if": "greater_than", "value": 2.0}]
    )
    with pytest.raises(MetricAggregationError, match="infinite violation"):
        base.calc_fitness(encode_metric_values({OMEGA + "_mean": math.inf}))


def test_selected_unavailable_native_candidate_gets_explicit_infeasibility():
    base = evaluator((OMEGA,))
    scorer = CanonicalResultScorer(base)
    result = scorer.score(
        [1.0],
        {
            "base": {
                "venue": {
                    **source({}),
                    OMEGA: None,
                    "metric_status": {OMEGA: {"status": "undefined"}},
                }
            }
        },
    )
    assert result["constraint_violation"] > 0
    assert all(math.isfinite(value) for value in result["fitness"])
    assert result["metrics"]["objectives"] == {}
    assert result["metrics"]["fitness_status"] == "invalid"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", ["mean", "median", "std"])
def test_undefined_requested_suite_reduction_fails(mode):
    base = evaluator((CAGR,))
    values = [math.inf, -math.inf] if mode != "std" else [math.inf, 1.0]
    with pytest.raises(MetricAggregationError):
        suite(base, reducer=mode).score_scenario_results(
            [
                scenario(label, {CAGR: value})
                for label, value in zip(("one", "two"), values)
            ]
        )


def test_every_required_scenario_must_be_present():
    with pytest.raises(MetricAggregationError, match="scenario"):
        suite(evaluator()).score_scenario_results([scenario("one", {CAGR: 0.2})])


def test_named_scenario_selection_does_not_require_that_quantity_in_other_scenarios():
    base = evaluator((CAGR,))
    result = suite(base)
    result.objective_bases = [SimpleNamespace(scenario="one", reducer=None)]
    rows = [scenario("one", {CAGR: 0.2}), scenario("two", {OMEGA: 1.0})]
    scored = result.score_scenario_results(rows)
    assert scored["objectives"] == (to_engine_value(base.scoring_specs[0], 0.2),)


def test_invalid_transport_has_no_fabricated_raw_objective():
    fitness, penalty, metrics = _build_invalid_candidate_metrics(
        [{"metric": CAGR, "goal": "max"}], "example"
    )
    assert (
        metrics["objectives"] == {}
        and metrics["definition_version"] == CORE_METRIC_DEFINITION_VERSION
    )
    assert (
        math.isfinite(penalty)
        and penalty > 0
        and all(math.isfinite(x) for x in fitness)
    )


def test_only_the_exact_insufficient_candidate_window_error_is_recoverable():
    from optimize import _is_recoverable_backtest_candidate_error

    assert _is_recoverable_backtest_candidate_error(
        ValueError("replay has no positive evaluated interval after declared warmup")
    )
    assert not _is_recoverable_backtest_candidate_error(
        ValueError("missing required market metadata")
    )
    assert not _is_recoverable_backtest_candidate_error(
        ValueError("unknown core metric")
    )


@pytest.mark.parametrize("authored", [None, "omitted"])
def test_resume_resolves_the_same_native_campaign_half_life(authored):
    from copy import deepcopy
    from config.schema import get_expanded_config_prototype
    from config.metrics import DEFAULT_EW_HALF_LIFE_DAYS
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import (
        CONTRACT_CACHE_KEY,
        CONTRACT_KEY,
        build_evaluation_contract,
    )

    current = get_expanded_config_prototype()
    current["backtest"]["ew_half_life_days"] = DEFAULT_EW_HALF_LIFE_DAYS
    current[CONTRACT_CACHE_KEY] = build_evaluation_contract(current)
    previous = deepcopy(current)
    previous[CONTRACT_KEY] = deepcopy(current[CONTRACT_CACHE_KEY])
    if authored == "omitted":
        previous["backtest"].pop("ew_half_life_days")
    else:
        previous["backtest"]["ew_half_life_days"] = authored
    assert _resume_config_mismatches(previous, current) == []
    previous["backtest"]["ew_half_life_days"] = DEFAULT_EW_HALF_LIFE_DAYS / 2.0
    assert any(
        "ew_half_life_days" in value
        for value in _resume_config_mismatches(previous, current)
    )
