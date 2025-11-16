from metrics_schema import merge_suite_payload


def test_merge_suite_payload_builds_structure():
    aggregate_stats = {"adg": {"mean": 1.0, "min": 0.5, "max": 1.5, "std": 0.25}}
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
