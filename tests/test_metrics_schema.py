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


@pytest.mark.parametrize("same", [False, True])
def test_evaluation_dates_are_preserved_without_date_statistics(same):
    first = {"n_days": 2.0, "effective_start_date": "2024-01-01T00:00:00Z",
             "effective_end_date": "2024-01-03T00:00:00Z"}
    second = dict(first) if same else dict(first, n_days=1.0, effective_start_date="2024-01-02T00:00:00Z")
    metrics = build_scenario_metrics({"binance": first, "bybit": second})
    assert set(metrics["stats"]) == {"n_days"}
    assert metrics["stats"]["n_days"]["min"] == (2.0 if same else 1.0)
    assert metrics["exchanges"]["bybit"]["effective_start_date"] == second["effective_start_date"]
    assert ("effective_start_date" in metrics) == same
    suite = merge_suite_payload(metrics["stats"], scenario_metrics={"base": metrics})
    assert suite["scenarios"]["base"]["exchanges"] == metrics["exchanges"]
    assert ("effective_start_date" in suite) == same


def test_suite_preserves_different_scenario_windows():
    scenarios = {
        label: build_scenario_metrics({"combined": {"n_days": float(day),
            "effective_start_date": f"2024-01-0{day}T00:00:00Z",
            "effective_end_date": "2024-01-04T00:00:00Z"}})
        for label, day in [("base", 1), ("stress", 2)]
    }
    suite = merge_suite_payload({}, scenario_metrics=scenarios)
    assert "effective_start_date" not in suite
    assert suite["scenarios"]["stress"]["effective_start_date"] == "2024-01-02T00:00:00Z"


def test_duration_alias_resolves_old_and_new_artifacts():
    from config.metrics import canonicalize_metric_name, resolve_metric_value
    assert canonicalize_metric_name("fills_analysis_duration_days_mean") == "n_days_mean"
    assert resolve_metric_value({"fills_analysis_duration_days_mean": 2.5}, "n_days_mean") == 2.5
    assert resolve_metric_value({"n_days_mean": 2.5}, "fills_analysis_duration_days_mean") == 2.5


def test_result_metrics_replace_previous_run_and_roundtrip(tmp_path):
    import json
    from config_utils import dump_config
    from metrics_schema import attach_result_metrics
    config = {"metrics": {"old": True}, "suite_metrics": {"old": True}}
    suite = {"scenarios": {"base": {"effective_start_date": None}}}
    dump_config(attach_result_metrics(config, suite_metrics=suite), str(tmp_path / "config.json"))
    assert json.loads((tmp_path / "config.json").read_text()) == {"suite_metrics": suite}


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_standalone_nonfinite_diagnostics_do_not_weaken_optimizer_validation(value):
    import json
    from metrics_schema import build_standalone_metrics

    analysis = {"equity_choppiness": value, "effective_start_date": "2024-01-01T00:00:00Z"}
    payload = build_standalone_metrics(analysis, "binance")
    assert payload["stats"] == {}
    assert payload["nonfinite_diagnostics"] == {"equity_choppiness": str(value)}
    json.dumps(payload, allow_nan=False)
    assert payload["exchanges"]["binance"]["effective_start_date"] == analysis["effective_start_date"]
    with pytest.raises(MetricAggregationError, match="non-finite metric"):
        build_scenario_metrics({"binance": analysis})
