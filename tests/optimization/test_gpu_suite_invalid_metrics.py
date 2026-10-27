"""Canonical GPU fan-in preserves candidate-local status and finite engine fitness."""
import json
import math

import pytest

from metrics_schema import MetricAggregationError
from optimization.gpu.executor import BacktestResult
from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
from optimize import INVALID_BACKTEST_CANDIDATE_PENALTY
from test_core_metric_scoring import evaluator, suite, source

METRIC = "omega_ratio_strategy_equity"


def engine(value):
    return -(2 / math.pi) * math.atan(value)


def make_suite():
    return suite(evaluator((METRIC,)), labels=("synthetic",))


def evaluate(value, values):
    scorer = CanonicalResultScorer(value)
    return [scorer.score([index], {
        "synthetic": {"venue": {
            METRIC: number, "liquidated": False, "metric_metadata": {"definition_version": 1},
        }},
    }) for index, number in enumerate(values)]


def assert_invalid(row):
    assert row["constraint_violation"] == INVALID_BACKTEST_CANDIDATE_PENALTY
    assert all(math.isfinite(value) for value in row["fitness"])
    assert "unpenalized_objectives" not in row["metrics"]
    assert row["metrics"]["objectives"] == {}
    assert row["metrics"]["fitness_status"] == "invalid"
    assert row["metrics"]["suite_metrics"] == {}
    assert "MetricAggregationError" in row["metrics"]["error"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_malformed_raw_nonfinite_aggregation_does_not_abort_other_candidates(invalid):
    rows = evaluate(make_suite(), [2.0, invalid, 3.0])
    assert [row["constraint_violation"] for row in rows] == [
        0.0, INVALID_BACKTEST_CANDIDATE_PENALTY, 0.0,
    ]
    assert rows[0]["fitness"] == (engine(2.0),)
    assert rows[2]["fitness"] == (engine(3.0),)
    assert_invalid(rows[1])


@pytest.mark.parametrize("infinite", [math.inf, -math.inf])
def test_explicit_infinite_status_is_candidate_local_and_valid(infinite):
    scorer = CanonicalResultScorer(make_suite())
    rows = []
    for index, value in enumerate([2.0, infinite, 3.0]):
        slot = ResultSlot(str(index), "dataset", "synthetic", "venue", (METRIC,))
        pending = CandidateEvaluation(str(index), [index], [slot], scorer)
        encoded = source({METRIC: value})
        complete = pending.add_result(BacktestResult(str(index), "dataset",
            {METRIC: encoded[METRIC]}, False,
            {key: encoded[key] for key in ("metric_metadata", "metric_status") if key in encoded}))
        rows.append(complete.require_full())
        assert not pending.pending_request_ids
    assert rows[0]["fitness"] == (engine(2.0),)
    assert rows[1]["fitness"] == (-1.0 if infinite > 0 else 1.0,)
    assert rows[1]["constraint_violation"] == 0.0
    assert rows[1]["metrics"]["objectives"][METRIC] is None
    assert rows[2]["fitness"] == (engine(3.0),)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_raw_nonfinite_native_envelope_still_fails_as_malformed_output(invalid):
    slot = ResultSlot("request", "dataset", "synthetic", "venue", (METRIC,))
    pending = CandidateEvaluation("candidate", [], [slot], CanonicalResultScorer(make_suite()))
    with pytest.raises(RuntimeError, match="malformed metric values"):
        pending.add_result(BacktestResult("request", "dataset", {METRIC: invalid}, False,
                                        {"metric_metadata": {"definition_version": 1}}))
    assert pending.pending_request_ids == ("request",)


def test_reducer_metric_failure_is_candidate_local():
    value = make_suite()
    original = value.score_scenario_results

    def fail_one(results, *, stage="full"):
        if results[0].metrics["stats"][METRIC]["mean"] == 2.0:
            raise MetricAggregationError("synthetic reducer failure")
        return original(results, stage=stage)

    value.score_scenario_results = fail_one
    rows = evaluate(value, [1.0, 2.0, 3.0])
    assert rows[0]["fitness"] == (engine(1.0),)
    assert_invalid(rows[1])
    assert rows[2]["fitness"] == (engine(3.0),)


def test_unrelated_errors_still_propagate():
    value = make_suite()

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic scorer failure")

    value.score_scenario_results = broken
    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        evaluate(value, [1.0])
