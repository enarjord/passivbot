import pytest

from metrics_schema import (
    MetricAggregationError,
    build_scenario_metrics,
    flatten_metric_stats,
    merge_suite_payload,
)


def test_build_scenario_metrics_emits_median_stats():
    payload = build_scenario_metrics(
        {
            "binance": {"adg": 1.0},
            "bybit": {"adg": 3.0},
        }
    )

    assert payload["stats"]["adg"]["median"] == 2.0
    assert flatten_metric_stats(payload["stats"])["adg_median"] == 2.0


def test_build_scenario_metrics_rejects_non_finite_metric_values():
    with pytest.raises(
        MetricAggregationError,
        match="non-finite metric 'drawdown_worst'",
    ):
        build_scenario_metrics({"binance": {"drawdown_worst": float("nan")}})


def test_flatten_metric_stats_rejects_missing_stat_fields():
    with pytest.raises(MetricAggregationError, match="missing stat field"):
        flatten_metric_stats({"adg": {"mean": 1.0, "min": 0.5, "max": 1.5, "std": 0.25}})


def test_merge_suite_payload_builds_structure():
    aggregate_stats = {
        "adg": {"mean": 1.0, "min": 0.5, "max": 1.5, "std": 0.25, "median": 1.0}
    }
    aggregate_values = {"adg": 1.0}
    scenario_metrics = {
        "case_a": {"stats": {"adg": {"mean": 0.8}}},
        "case_b": {"stats": {"adg": {"mean": 1.2}}},
    }

    payload = merge_suite_payload(
        aggregate_stats,
        aggregate_values=aggregate_values,
        scenario_metrics=scenario_metrics,
    )

    assert "metrics" in payload
    adg_entry = payload["metrics"]["adg"]
    assert adg_entry["aggregated"] == 1.0
    assert adg_entry["stats"]["min"] == 0.5
    assert adg_entry["scenarios"]["case_a"] == 0.8
    assert adg_entry["scenarios"]["case_b"] == 1.2
