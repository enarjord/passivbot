"""Canonical GPU suite scoring rejects bad fitness without losing other candidates."""
import math
from types import SimpleNamespace

import pytest

from metrics_schema import MetricAggregationError
from optimization.gpu.executor import BacktestResult
from optimization.native_results import CandidateEvaluation, CanonicalResultScorer, ResultSlot
from optimize import INVALID_BACKTEST_CANDIDATE_PENALTY


METRIC = "equity_choppiness_usd"


class Suite:
    """Expose CPU-owned suite scoring without a GPU backend or simulations."""

    base = SimpleNamespace(
        config={"optimize": {"scoring": [METRIC]}},
        scoring_specs=[SimpleNamespace(metric=METRIC)],
        limit_checks=[],
    )
    contexts = [SimpleNamespace(label="synthetic", exchanges=["bybit"])]
    objective_bases = [SimpleNamespace(scenario=None)]
    objective_scenario = None

    def score_scenario_results(self, results, *, stage="full"):
        """Return a finite objective for valid synthetic candidates."""
        value = results[0].metrics["stats"][METRIC]["mean"]
        return {
            "objectives": (value,),
            "unpenalized_objectives": (value,),
            "constraint_violation": 0.0,
            "suite_metrics": {"value": value},
        }


def evaluate(suite, values):
    """Exercise the real canonical aggregator for independent candidate rows."""
    scorer = CanonicalResultScorer(suite)
    return [scorer.score([index], {
        "synthetic": {"bybit": {METRIC: value, "liquidated": False}},
    }) for index, value in enumerate(values)]


def assert_invalid(row):
    assert row["constraint_violation"] == INVALID_BACKTEST_CANDIDATE_PENALTY
    assert all(math.isfinite(value) for value in row["fitness"])
    assert row["metrics"]["unpenalized_objectives"] == row["fitness"]
    assert row["metrics"]["suite_metrics"] == {}
    assert "MetricAggregationError" in row["metrics"]["error"]


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_nonfinite_aggregation_does_not_abort_other_candidates(invalid):
    """Nonfinite analyses at the canonical scoring layer receive the CPU penalty."""
    rows = evaluate(Suite(), [2.0, invalid, 3.0])
    assert [row["constraint_violation"] for row in rows] == [
        0.0, INVALID_BACKTEST_CANDIDATE_PENALTY, 0.0,
    ]
    assert rows[0]["fitness"] == (2.0,)
    assert rows[2]["fitness"] == (3.0,)
    assert_invalid(rows[1])


@pytest.mark.parametrize("invalid", [math.inf, -math.inf])
def test_native_infinity_sentinel_is_candidate_local(invalid):
    """Accepted metric sentinels fan in independently through native completions."""
    scorer = CanonicalResultScorer(Suite())
    rows = []
    for index, value in enumerate([2.0, invalid, 3.0]):
        slot = ResultSlot(str(index), "dataset", "synthetic", "bybit", (METRIC,))
        pending = CandidateEvaluation(str(index), [index], [slot], scorer)
        complete = pending.add_result(BacktestResult(str(index), "dataset", {METRIC: value}, False))
        rows.append(complete.require_full())
        assert not pending.pending_request_ids
    assert rows[0]["fitness"] == (2.0,)
    assert_invalid(rows[1])
    assert rows[2]["fitness"] == (3.0,)


def test_native_nan_envelope_still_fails_as_malformed_output():
    """NaN is a producer defect, unlike the supported infinity metric sentinels."""
    slot = ResultSlot("request", "dataset", "synthetic", "bybit", (METRIC,))
    pending = CandidateEvaluation("candidate", [], [slot], CanonicalResultScorer(Suite()))
    with pytest.raises(RuntimeError, match="malformed metric values"):
        pending.add_result(BacktestResult("request", "dataset", {METRIC: math.nan}, False))
    assert pending.pending_request_ids == ("request",)


def test_reducer_metric_failure_is_candidate_local():
    """Aggregation failures after scenario construction use the same policy."""
    class ReducerFailure(Suite):
        def score_scenario_results(self, results, *, stage="full"):
            scored = super().score_scenario_results(results, stage=stage)
            if scored["objectives"] == (2.0,):
                raise MetricAggregationError("synthetic reducer failure")
            return scored

    rows = evaluate(ReducerFailure(), [1.0, 2.0, 3.0])
    assert rows[0]["fitness"] == (1.0,)
    assert_invalid(rows[1])
    assert rows[2]["fitness"] == (3.0,)


def test_unrelated_errors_still_propagate():
    """Programming errors must not be treated as bad candidate fitness."""
    class BrokenSuite(Suite):
        def score_scenario_results(self, results, *, stage="full"):
            raise RuntimeError("synthetic scorer failure")

    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        evaluate(BrokenSuite(), [1.0])
