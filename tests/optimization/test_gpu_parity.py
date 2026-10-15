import json
import math

import pytest

from optimization.gpu.parity import MetricTolerance, compare_limits, compare_metrics


@pytest.mark.parametrize("field", ["absolute", "relative"])
@pytest.mark.parametrize("value", [False, True])
def test_numeric_tolerance_fields_reject_json_booleans(field, value):
    policy = {"absolute": 0, "relative": 0, field: value}
    with pytest.raises(TypeError, match="not boolean"):
        MetricTolerance(**policy)


def test_metric_specific_tolerances_and_cpu_reference_scale():
    report = compare_metrics(
        {"adg": 0.0, "drawdown": 0.5},
        {"adg": 1e-8, "drawdown": 0.501},
        {"adg": MetricTolerance(1e-7, 0), "drawdown": MetricTolerance(0, 1e-4)},
    )
    assert not report["passed"]
    assert report["metrics"]["adg"]["status"] == "match"
    assert report["metrics"]["drawdown"]["status"] == "mismatch"
    assert report["metrics"]["drawdown"]["allowed_error"] == 5e-5


def test_missing_values_and_unassessed_metrics_never_pass():
    report = compare_metrics(
        {"gpu_missing": 1, "unassessed": 1},
        {"cpu_missing": 1, "unassessed": 1},
        dict.fromkeys(["gpu_missing", "cpu_missing", "both_missing", "unassessed"]),
    )
    assert not report["passed"]
    assert [row["status"] for row in report["metrics"].values()] == [
        "missing_gpu", "missing_cpu", "missing_both", "unassessed"
    ]


@pytest.mark.parametrize(
    "cpu,gpu,allowed,passes",
    [
        (math.inf, math.inf, True, True),
        (-math.inf, -math.inf, True, True),
        (math.inf, math.inf, False, False),
        (math.inf, -math.inf, True, False),
        (math.nan, math.nan, True, False),
        (1.0, math.inf, True, False),
    ],
)
def test_explicit_sentinels_and_strict_json(cpu, gpu, allowed, passes):
    report = compare_metrics(
        {"metric": cpu}, {"metric": gpu},
        {"metric": MetricTolerance(0, 0, matching_infinity=allowed)},
    )
    assert report["passed"] is passes
    json.dumps(report, allow_nan=False)


def test_only_requested_metrics_are_compared():
    report = compare_metrics(
        {"metric": 1, "other": 2}, {"metric": 1, "other": 99},
        {"metric": MetricTolerance(0, 0)},
    )
    assert report["passed"]
    assert list(report["metrics"]) == ["metric"]


def test_overflow_in_derived_error_never_matches():
    report = compare_metrics(
        {"metric": 1e308}, {"metric": -1e308},
        {"metric": MetricTolerance(0, 10)},
    )
    assert not report["passed"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [-1, math.inf, math.nan])
def test_invalid_tolerances_rejected(value):
    with pytest.raises(ValueError):
        MetricTolerance(value, 0)
    with pytest.raises(ValueError):
        MetricTolerance(0, value)


def test_small_metric_error_can_still_flip_limit_feasibility():
    check = {"metric": "drawdown_worst_strategy_eq", "metric_key": "drawdown_worst_strategy_eq_mean",
             "mode": "greater_than", "bound": 0.5, "penalty_weight": 1}
    cpu = {"drawdown_worst_strategy_eq": 0.5}
    gpu = {"drawdown_worst_strategy_eq": 0.500001}
    assert compare_metrics(cpu, gpu, {"drawdown_worst_strategy_eq": MetricTolerance(1e-5, 0)})["passed"]
    report = compare_limits(cpu, gpu, [check])
    assert not report["passed"]
    assert report["cpu_feasible"] is True
    assert report["gpu_feasible"] is False
    assert report["checks"][0]["status"] == "flip"


@pytest.mark.parametrize("values", [{}, {"fills_per_day": math.inf}])
def test_missing_or_nonfinite_limit_input_never_becomes_feasible(values):
    check = {"metric": "fills_per_day", "metric_key": "fills_per_day_std",
             "mode": "greater_than", "bound": 0, "penalty_weight": 1}
    report = compare_limits(values, {"fills_per_day": 1}, [check])
    assert not report["passed"]
    assert report["cpu_feasible"] is None


def test_suite_limit_is_explicitly_unassessed_in_scalar_comparison():
    check = {"metric": "fills_per_day", "metric_key": "fills_per_day_mean",
             "mode": "greater_than", "bound": 0, "scenario": "crash", "penalty_weight": 1}
    report = compare_limits({"fills_per_day": 1}, {"fills_per_day": 1}, [check])
    assert report["checks"][0]["status"] == "suite_required"
    assert report["cpu_feasible"] is None
