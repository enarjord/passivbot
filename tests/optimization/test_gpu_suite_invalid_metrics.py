"""GPU suites reject invalid candidates without losing the remaining batch."""
import math
from types import SimpleNamespace

import pytest

from metrics_schema import MetricAggregationError
from optimization.backends.gpu_backend import (
    _evaluate_gpu_suite_proxies,
    _GPU_SUITE_OBJECTIVES_KEY,
    _GPU_SUITE_UNPENALIZED_OBJECTIVES_KEY,
    _GPU_SUITE_VIOLATION_KEY,
    _GPU_SUITE_METRICS_KEY,
)
from optimize import INVALID_BACKTEST_CANDIDATE_PENALTY


class Proxy:
    """Return synthetic metrics without requiring CUDA or market data."""

    def __init__(self, values):
        self.values = values

    def evaluate(self, candidates):
        """Keep metric rows in candidate order."""
        assert len(candidates) == len(self.values)
        return [{"equity_choppiness_usd": value} for value in self.values]


class Suite:
    """Expose the same scoring configuration used by the CPU fallback."""

    base = SimpleNamespace(config={"optimize": {"scoring": ["equity_choppiness_usd"]}})

    def score_scenario_results(self, results):
        """Return a finite objective for valid synthetic candidates."""
        value = results[0].metrics["stats"]["equity_choppiness_usd"]["mean"]
        return {
            "objectives": (value,),
            "unpenalized_objectives": (value,),
            "constraint_violation": 0.0,
            "suite_metrics": {"value": value},
        }


def evaluate(suite, values):
    """Exercise the real GPU suite orchestration with a three-candidate batch."""
    return _evaluate_gpu_suite_proxies(
        suite,
        [(SimpleNamespace(label="synthetic"), (("bybit", Proxy(values)),), {})],
        [{"x": index} for index in range(len(values))],
    )


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_nonfinite_candidate_does_not_abort_batch(invalid):
    """Only the invalid middle candidate receives the CPU invalid penalty."""
    rows = evaluate(Suite(), [2.0, invalid, 3.0])
    assert [row[_GPU_SUITE_VIOLATION_KEY] for row in rows] == [
        0.0, INVALID_BACKTEST_CANDIDATE_PENALTY, 0.0
    ]
    assert rows[0][_GPU_SUITE_OBJECTIVES_KEY] == (2.0,)
    assert rows[2][_GPU_SUITE_OBJECTIVES_KEY] == (3.0,)
    assert all(math.isfinite(value) for value in rows[1][_GPU_SUITE_OBJECTIVES_KEY])
    assert rows[1][_GPU_SUITE_UNPENALIZED_OBJECTIVES_KEY] == rows[1][_GPU_SUITE_OBJECTIVES_KEY]
    assert rows[1][_GPU_SUITE_METRICS_KEY] == {}


def test_reducer_metric_failure_is_candidate_local():
    """Aggregation failures after scenario construction use the same policy."""
    class ReducerFailure(Suite):
        """Fail only one candidate in the final suite reducer."""
        def score_scenario_results(self, results):
            """Simulate an invalid derived suite metric."""
            scored = super().score_scenario_results(results)
            if scored["objectives"] == (2.0,):
                raise MetricAggregationError("synthetic reducer failure")
            return scored

    rows = evaluate(ReducerFailure(), [1.0, 2.0, 3.0])
    assert rows[1][_GPU_SUITE_VIOLATION_KEY] == INVALID_BACKTEST_CANDIDATE_PENALTY
    assert rows[2][_GPU_SUITE_OBJECTIVES_KEY] == (3.0,)


def test_unrelated_errors_still_propagate():
    """An orchestration/programming error must not be treated as bad fitness."""
    class BrokenSuite(Suite):
        """Raise an unrelated scorer failure."""
        def score_scenario_results(self, results):
            """Simulate a programming error."""
            raise RuntimeError("synthetic scorer failure")

    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        evaluate(BrokenSuite(), [1.0])
