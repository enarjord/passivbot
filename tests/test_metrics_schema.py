"""Core status/reduction orchestration; Rust tests own metric arithmetic."""

import json
import math
import pytest
from metrics_schema import (
    MetricAggregationError,
    build_scenario_metrics,
    build_standalone_metrics,
    decode_metric_value,
    encode_metric_values,
    flatten_metric_stats,
    raw_reduce,
)

CAGR = "cagr_fraction_strategy_equity"
OMEGA = "omega_ratio_strategy_equity"


def source(**values):
    return {"metric_metadata": {"definition_version": 1}, **values}


@pytest.mark.parametrize("value", [0.0, 3.0, -2.0, math.inf, -math.inf])
def test_extended_real_roundtrip_is_json_safe(value):
    encoded = encode_metric_values({CAGR: value})
    json.dumps(encoded, allow_nan=False)
    assert decode_metric_value(encoded, CAGR) == value


@pytest.mark.parametrize("status", ["undefined", "invalid"])
def test_selected_unavailable_fails_and_unselected_unavailable_is_ignored(status):
    analysis = source(
        **{
            CAGR: 0.2,
            OMEGA: None,
            "metric_status": {OMEGA: {"status": status, "reason": "example"}},
        }
    )
    result = build_scenario_metrics({"venue": analysis}, required_metrics=[CAGR])
    assert set(result["stats"]) == {CAGR}
    with pytest.raises(MetricAggregationError, match="unavailable"):
        build_scenario_metrics({"venue": analysis}, required_metrics=[OMEGA])
    report = build_standalone_metrics(analysis, "venue")
    assert report["stats"][OMEGA]["mean"] is None
    assert report["stats"][OMEGA]["metric_status"]["mean"]["status"] == status
    json.dumps(report, allow_nan=False)


def test_exact_source_coverage_no_surviving_subset_mean():
    with pytest.raises(MetricAggregationError, match="missing"):
        build_scenario_metrics(
            {"one": source(**{CAGR: 0.2}), "two": source()}, required_metrics=[CAGR]
        )


def test_only_requested_reducers_and_no_numeric_support_metadata():
    analyses = {
        "a": source(**{CAGR: 1.0, "liquidated": False, "fill_count": 123}),
        "b": source(**{CAGR: 3.0, "liquidated": True}),
    }
    row = build_scenario_metrics(
        analyses, required_metrics=[CAGR], reducers={CAGR: {"max"}}
    )
    assert row["stats"] == {CAGR: {"max": 3.0}}
    assert flatten_metric_stats(row["stats"]) == {CAGR + "_max": 3.0}


@pytest.mark.parametrize(
    "mode,expected",
    [("mean", math.inf), ("min", 1.0), ("max", math.inf), ("median", math.inf)],
)
def test_raw_infinity_reduction(mode, expected):
    assert raw_reduce([1.0, math.inf], mode) == expected


@pytest.mark.parametrize(
    "mode,values",
    [
        ("mean", [-math.inf, math.inf]),
        ("median", [-math.inf, math.inf]),
        ("std", [1.0, math.inf]),
    ],
)
def test_undefined_reductions_never_get_neutral_values(mode, values):
    with pytest.raises(MetricAggregationError):
        raw_reduce(values, mode)


def test_reduce_raw_before_adapter_population_and_infinity_status_survive_flatten():
    analysis = source(**encode_metric_values({OMEGA: math.inf}))
    row = build_scenario_metrics(
        {"one": analysis, "two": source(**{OMEGA: 1.0})}, required_metrics=[OMEGA]
    )
    assert (
        decode_metric_value(flatten_metric_stats(row["stats"]), OMEGA + "_mean")
        == math.inf
    )
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize(
    "metadata", [{}, {"definition_version": 0}, {"definition_version": 2}]
)
def test_stale_or_missing_definition_rejected(metadata):
    with pytest.raises(MetricAggregationError, match="version"):
        build_scenario_metrics(
            {"one": {CAGR: 0.2, "metric_metadata": metadata}}, required_metrics=[CAGR]
        )


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_finite_status_requires_a_real_finite_value(value):
    with pytest.raises(MetricAggregationError):
        decode_metric_value({CAGR: value}, CAGR)
