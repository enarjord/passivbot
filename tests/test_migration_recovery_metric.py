"""Recovery formula guesses belong only to the opt-in released-v8 converter."""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from config.json_schema import generate_schema
from config.schema import get_template_config
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import main, migrate


TARGET = "peak_recovery_max_days_strategy_equity"
ALIASES = ("strategy_eq_recovery_days_max", "peak_recovery_days_strategy_eq")


@pytest.mark.parametrize("metric", ALIASES)
@pytest.mark.parametrize("acknowledge", [False, True])
def test_best_effort_reports_recovery_formula_choice_and_preserves_policy(
    metric, acknowledge, caplog,
):
    source = released_v8_policy()
    source["optimize"]["scoring"] = [{"metric": metric, "goal": "min", "reducer": "median"}]
    source["optimize"]["limits"] = [
        {"metric": metric, "penalize_if": direction, "value": 17.5, "reducer": "max"}
        for direction in ("less_than", "greater_than")
    ]
    original = deepcopy(source)

    result = migrate(
        source, best_effort=True, acknowledge_metric_formula_change=acknowledge,
    )

    assert result.config["optimize"]["scoring"] == [
        {"metric": TARGET, "goal": "min", "reducer": "median"},
    ]
    assert result.config["optimize"]["limits"] == [
        {**limit, "metric": TARGET} for limit in original["optimize"]["limits"]
    ]
    choices = [change for change in result.changes if change["old"] == metric]
    assert len(choices) == 3
    for choice in choices:
        assert choice["new"] == TARGET and choice["best_effort"] is True
        assert "strictly-higher" in choice["reason"]
        assert "peak ties recover" in choice["reason"]
        assert "flat equity has zero recovery" in choice["reason"]
        assert "re-evaluate objectives, limits and stored fitness" in choice["reason"]
    assert metric in caplog.text and TARGET in caplog.text
    Draft202012Validator(generate_schema()).validate(result.config)
    assert source == original


@pytest.mark.parametrize("metric", ALIASES)
def test_recovery_strict_mode_requires_explicit_replacement_and_respects_it(metric):
    source = released_v8_policy()
    source["optimize"]["limits"] = [
        {"metric": metric, "penalize_if": "greater_than", "value": 17.5},
    ]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="requires --metric-replacement"):
        migrate(source, memory_span_minutes=43200, acknowledge_metric_formula_change=True)

    # A deliberately different authored formula must win over the inferred choice.
    explicit = "pain_index_fraction_strategy_equity"
    for best_effort in (False, True):
        result = migrate(
            source, memory_span_minutes=43200, best_effort=best_effort,
            acknowledge_metric_formula_change=True, metric_replacements={metric: explicit},
        )
        assert result.config["optimize"]["limits"][0]["metric"] == explicit
        choice = next(change for change in result.changes if change["old"] == metric)
        assert not choice.get("best_effort", False)
        assert "inferred" not in choice["reason"]
    assert source == original


@pytest.mark.parametrize("metric", ALIASES)
@pytest.mark.parametrize("stat", ["min", "max", "mean", "std", "median"])
def test_recovery_outer_statistic_becomes_reducer(metric, stat):
    source = released_v8_policy()
    source["optimize"]["scoring"] = [{"metric": f"{metric}_{stat}", "goal": "min"}]
    source["optimize"]["limits"] = [
        {"metric": f"{metric}_{stat}", "penalize_if": "greater_than", "value": 17.5},
    ]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [
        {"metric": TARGET, "goal": "min", "reducer": stat},
    ]
    assert result.config["optimize"]["limits"] == [
        {"metric": TARGET, "penalize_if": "greater_than", "value": 17.5, "reducer": stat},
    ]


def test_recovery_reducer_map_and_legacy_limit_map_keep_outer_statistic_and_direction():
    source = released_v8_policy()
    metric = f"{ALIASES[1]}_max"
    source["backtest"]["reducer"] = {"default": "mean", metric: "max"}
    source["optimize"]["limits"] = {f"penalize_if_lower_than_{metric}": 17.5}
    result = migrate(source, best_effort=True)
    assert result.config["backtest"]["reducer"] == {"default": "mean", TARGET: "max"}
    assert result.config["optimize"]["limits"] == [
        {"metric": TARGET, "penalize_if": "less_than", "value": 17.5, "reducer": "max"},
    ]


def test_recovery_inference_does_not_override_conflicting_authored_reducer():
    source = released_v8_policy()
    source["optimize"]["limits"] = [
        {"metric": f"{ALIASES[0]}_max", "penalize_if": "greater_than",
         "value": 17.5, "reducer": "mean"},
    ]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="outer reducer.*conflicts"):
        migrate(source, best_effort=True)
    assert source == original


def test_recovery_alias_is_not_accepted_in_current_v9_even_in_best_effort():
    source = get_template_config()
    source["optimize"]["limits"] = [
        {"metric": ALIASES[0], "penalize_if": "greater_than", "value": 17.5},
    ]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="unknown or retired"):
        migrate(source, best_effort=True)
    assert source == original


def test_recovery_aliases_coalesce_identical_objectives_with_a_reported_choice():
    source = released_v8_policy()
    source["optimize"]["scoring"] = [{"metric": metric, "goal": "min"} for metric in ALIASES]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": TARGET, "goal": "min"}]
    change = next(item for item in result.changes if item["path"] == "optimize.scoring")
    assert change["best_effort"] is True
    assert change["old_count"] == 2 and change["new_count"] == 1


@pytest.mark.parametrize("metric", [
    "peak_recovery_days_strategy_eq_long", "peak_recovery_days_strategy_eq_short",

    "strategy_eq_recovery_days_mean", "strategy_eq_recovery_days_p95",
    "hard_stop_restarts_per_year_long",
])
def test_recognized_unsupported_recovery_and_hsl_metrics_are_removed_only_in_best_effort(metric):
    source = released_v8_policy()
    source["optimize"]["limits"] = [
        {"metric": metric, "penalize_if": "greater_than", "value": 17.5},
    ]
    original = deepcopy(source)
    with pytest.raises(ValueError, match="requires --metric-replacement"):
        migrate(source, memory_span_minutes=43200, acknowledge_metric_formula_change=True)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == []
    assert any(change["path"] == "optimize.limits[0]"
               and change["old"] == original["optimize"]["limits"][0]
               and change["new"] is None and change.get("best_effort")
               for change in result.changes)
    assert source == original


def test_recovery_cli_best_effort_writes_config_and_inference_report(tmp_path):
    source = released_v8_policy()
    source["optimize"]["limits"] = [
        {"metric": ALIASES[0], "penalize_if": "greater_than", "value": 17.5},
    ]
    input_path = tmp_path / "input.v8.json"
    output_path = tmp_path / "output.v9.json"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(source))
    original = input_path.read_bytes()

    assert main([
        str(input_path), str(output_path), "--best-effort", "--report", str(report_path),
    ]) == 0

    assert input_path.read_bytes() == original
    output = json.loads(output_path.read_text())
    assert output["optimize"]["limits"][0]["metric"] == TARGET
    report = json.loads(report_path.read_text())
    choice = next(change for change in report["changes"] if change["old"] == ALIASES[0])
    assert choice["best_effort"] is True and choice["new"] == TARGET
