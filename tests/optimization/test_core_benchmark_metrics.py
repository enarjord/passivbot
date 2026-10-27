"""Benchmark transport preserves the same mathematical status as runtime scoring."""
import json

import pytest

from metrics_schema import MetricAggregationError
from tools.optimizer_benchmark.worker import project_metric_values, cohort_digest

METRIC = "omega_ratio_strategy_equity"


@pytest.mark.parametrize("status", ["positive_infinity", "negative_infinity", "undefined", "invalid"])
def test_benchmark_projection_preserves_null_and_explicit_status(status):
    source = {"metric_metadata": {"definition_version": 1}, METRIC: None,
              "metric_status": {METRIC: {"status": status, "reason": "fixture"}}}
    result = project_metric_values(source, [METRIC])
    assert result[METRIC] is None
    assert result["metric_status"] == source["metric_status"]
    assert result["definition_version"] == 1
    json.dumps(result, allow_nan=False)


def test_benchmark_projection_requires_current_definition_and_factual_status():
    for source in ({METRIC: 1.0}, {"definition_version": 1, METRIC: None},
                   {"definition_version": 1, METRIC: float("inf")},
                   {"definition_version": 1, METRIC: True}):
        with pytest.raises(MetricAggregationError):
            project_metric_values(source, [METRIC])


def test_benchmark_cohort_identity_includes_metric_status():
    rows = [{"metrics": {"definition_version": 1, METRIC: None,
             "metric_status": {METRIC: {"status": status}}}, "liquidated": False}
            for status in ("positive_infinity", "undefined")]
    assert cohort_digest(rows[:1]) != cohort_digest(rows[1:])
