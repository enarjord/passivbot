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
        {"adg_fraction": 0.0, "drawdown": 0.5},
        {"adg_fraction": 1e-8, "drawdown": 0.501},
        {"adg_fraction": MetricTolerance(1e-7, 0), "drawdown": MetricTolerance(0, 1e-4)},
    )
    assert not report["passed"]
    assert report["metrics"]["adg_fraction"]["status"] == "match"
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
    check = {"metric": "drawdown_max_fraction_strategy_equity", "metric_key": "drawdown_max_fraction_strategy_equity_mean",
             "mode": "greater_than", "bound": 0.5, "penalty_weight": 1}
    cpu = {"drawdown_max_fraction_strategy_equity": 0.5}
    gpu = {"drawdown_max_fraction_strategy_equity": 0.500001}
    assert compare_metrics(cpu, gpu, {"drawdown_max_fraction_strategy_equity": MetricTolerance(1e-5, 0)})["passed"]
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


LIMIT_METRIC = "omega_ratio_strategy_equity"


def _canonical_limit(mode, *, reducer="mean", weight=1.0):
    from limit_utils import expand_limit_checks
    entry = {"metric": LIMIT_METRIC, "penalize_if": mode, "reducer": reducer}
    entry.update({"range": [0.5, 2.0]} if mode.endswith("range") else {"value": 1.0})
    return expand_limit_checks([entry], {}, penalty_weight=weight)[0]


@pytest.mark.parametrize("reducer", ["mean", "min", "max", "median"])
@pytest.mark.parametrize("value,mode,feasible", [
    (math.inf, "less_than", True),
    (math.inf, "greater_than", False),
    (-math.inf, "less_than", False),
    (-math.inf, "greater_than", True),
])
def test_infinite_singleton_limits_use_raw_order_and_json_safe_violations(reducer, value, mode, feasible):
    from metrics_schema import decode_metric_value
    report = compare_limits({LIMIT_METRIC: value}, {LIMIT_METRIC: value},
                            [_canonical_limit(mode, reducer=reducer)])
    assert report["assessed"] and report["passed"]
    assert report["cpu_feasible"] is report["gpu_feasible"] is feasible
    assert report["checks"][0]["status"] == "match"
    for result in (report, report["checks"][0]):
        for name in ("cpu_violation", "gpu_violation"):
            assert decode_metric_value(result, name) == (0.0 if feasible else math.inf)
            if not feasible:
                assert result[name] is None
                assert result["metric_status"][name]["status"] == "positive_infinity"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [-math.inf, math.inf])
@pytest.mark.parametrize("mode,feasible", [("inside_range", True), ("outside_range", False)])
def test_infinite_limit_endpoints_obey_finite_ranges(value, mode, feasible):
    report = compare_limits({LIMIT_METRIC: value}, {LIMIT_METRIC: value}, [_canonical_limit(mode)])
    assert report["assessed"] and report["passed"]
    assert report["cpu_feasible"] is report["gpu_feasible"] is feasible
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [-math.inf, math.inf])
@pytest.mark.parametrize("reverse", [False, True])
def test_lower_and_upper_limits_accumulate_zero_and_infinity_without_losing_feasibility(value, reverse):
    from metrics_schema import decode_metric_value
    checks = [_canonical_limit("less_than"), _canonical_limit("greater_than")]
    report = compare_limits({LIMIT_METRIC: value}, {LIMIT_METRIC: value},
                            list(reversed(checks)) if reverse else checks)
    assert report["assessed"] and report["passed"]
    assert report["cpu_feasible"] is report["gpu_feasible"] is False
    assert decode_metric_value(report, "cpu_violation") == math.inf
    assert decode_metric_value(report, "gpu_violation") == math.inf
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("cpu,gpu,mode,cpu_feasible,gpu_feasible", [
    (math.inf, 0.0, "less_than", True, False),
    (math.inf, -math.inf, "greater_than", False, True),
    (1.0, -math.inf, "less_than", True, False),
])
def test_infinite_endpoints_can_flip_canonical_limit_feasibility(cpu, gpu, mode, cpu_feasible, gpu_feasible):
    report = compare_limits({LIMIT_METRIC: cpu}, {LIMIT_METRIC: gpu}, [_canonical_limit(mode)])
    assert report["assessed"] and not report["passed"]
    assert report["cpu_feasible"] is cpu_feasible
    assert report["gpu_feasible"] is gpu_feasible
    assert report["checks"][0]["status"] == "flip"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [-math.inf, math.inf])
def test_infinite_singleton_dispersion_stays_unavailable(value):
    report = compare_limits({LIMIT_METRIC: value}, {LIMIT_METRIC: 1.0},
                            [_canonical_limit("greater_than", reducer="std")])
    assert not report["assessed"] and not report["passed"]
    assert report["checks"][0]["status"] == "unavailable"
    assert report["cpu_feasible"] is report["gpu_feasible"] is None


@pytest.mark.parametrize("value", [None, math.nan, False, True])
@pytest.mark.parametrize("reducer", ["mean", "std"])
def test_unavailable_singleton_limits_never_gain_a_neutral_value(value, reducer):
    report = compare_limits({LIMIT_METRIC: value}, {LIMIT_METRIC: 1.0},
                            [_canonical_limit("greater_than", reducer=reducer)])
    assert not report["assessed"] and not report["passed"]
    assert report["checks"][0]["status"] == "unavailable"
    assert report["cpu_feasible"] is report["gpu_feasible"] is None
    json.dumps(report, allow_nan=False)


def test_zero_weight_times_infinite_violation_is_unavailable_not_feasible():
    report = compare_limits({LIMIT_METRIC: math.inf}, {LIMIT_METRIC: math.inf},
                            [_canonical_limit("greater_than", weight=0.0)])
    assert not report["assessed"] and not report["passed"]
    assert report["checks"][0]["status"] == "unavailable"
    assert report["cpu_feasible"] is report["gpu_feasible"] is None
    json.dumps(report, allow_nan=False)
