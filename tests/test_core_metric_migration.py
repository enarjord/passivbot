"""Released metrics cross an explicit, reported semantic definition boundary."""

from copy import deepcopy
import pytest
from config.migrations.released_v8 import get_released_v8_defaults
from tools.migrate_config_v9 import migrate
from released_v8_fixtures import RELEASED_V8_EXAMPLE_NAMES, released_v8_example_metrics


def source(metric):
    result = get_released_v8_defaults()
    result["backtest"]["scenarios"] = [{"label": "base"}]
    result["optimize"]["scoring"] = [{"metric": metric, "goal": "max"}]
    result["optimize"]["limits"] = []
    return result


def test_newly_retired_growth_requires_explicit_formula_choice_and_acknowledgment():
    raw = source("adg_strategy_eq")
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="metric-replacement"):
        migrate(raw, memory_span_minutes=123.0)
    replacement = {"adg_strategy_eq": "cagr_fraction_strategy_equity"}
    with pytest.raises(ValueError, match="acknowledge-metric-formula-change"):
        migrate(raw, memory_span_minutes=123.0, metric_replacements=replacement)
    result = migrate(
        raw,
        memory_span_minutes=123.0,
        metric_replacements=replacement,
        acknowledge_metric_formula_change=True,
    )
    assert result.config["optimize"]["scoring"] == [
        {"metric": "cagr_fraction_strategy_equity", "goal": "max"}
    ]
    assert any(
        row["old"] == "adg_strategy_eq"
        and row["new"] == "cagr_fraction_strategy_equity"
        and "definition version 1" in row["reason"]
        for row in result.changes
    )
    assert raw == before


def test_retained_name_with_revised_fee_definition_still_requires_acknowledgment():
    raw = source("loss_profit_ratio")
    with pytest.raises(ValueError, match="acknowledge-metric-formula-change"):
        migrate(raw, memory_span_minutes=123.0)
    result = migrate(
        raw, memory_span_minutes=123.0, acknowledge_metric_formula_change=True
    )
    assert any(
        row["old"] == row["new"] == "loss_profit_ratio"
        and "definition version 1" in row["reason"]
        for row in result.changes
    )


def test_best_effort_removes_retired_median_growth_without_inventing_an_objective():
    raw = source("mdg_strategy_eq")
    before = deepcopy(raw)
    result = migrate(raw, best_effort=True)
    assert result.config["optimize"]["scoring"] == []
    assert any(row["path"] == "optimize.scoring[0]"
               and row["old"] == before["optimize"]["scoring"][0]
               and row["new"] is None and row.get("best_effort")
               for row in result.changes)
    assert raw == before


@pytest.mark.parametrize("best_effort", [False, True])
def test_explicit_replacement_cannot_resurrect_a_retired_selector(best_effort):
    with pytest.raises(ValueError, match="unknown or retired"):
        migrate(
            source("adg_strategy_eq"),
            memory_span_minutes=123.0,
            metric_replacements={"adg_strategy_eq": "adg_fraction_usd"},
            acknowledge_metric_formula_change=True, best_effort=best_effort,
        )


def test_released_outer_stat_moves_to_existing_reducer_field_without_runtime_alias():
    raw = source("sharpe_ratio_strategy_eq_max")
    result = migrate(
        raw, memory_span_minutes=123.0, acknowledge_metric_formula_change=True
    )
    assert result.config["optimize"]["scoring"] == [
        {"metric": "sharpe_ratio_strategy_equity", "goal": "max", "reducer": "max"}
    ]


def test_released_outer_stat_conflict_is_not_silently_collapsed():
    raw = source("sharpe_ratio_strategy_eq_max")
    raw["optimize"]["scoring"][0]["reducer"] = "mean"
    with pytest.raises(ValueError, match="conflicts"):
        migrate(raw, memory_span_minutes=123.0, acknowledge_metric_formula_change=True)


def test_released_limit_map_keeps_direction_and_outer_reducer():
    raw = source("loss_profit_ratio")
    raw["optimize"]["limits"] = {
        "penalize_if_greater_than_sharpe_ratio_strategy_eq_max": 2.0
    }
    result = migrate(
        raw, memory_span_minutes=123.0, acknowledge_metric_formula_change=True
    )
    entry = result.config["optimize"]["limits"][0]
    assert entry["metric"] == "sharpe_ratio_strategy_equity"
    assert (
        entry["reducer"] == "max"
        and entry["penalize_if"] == "greater_than"
        and entry["value"] == 2.0
    )


@pytest.mark.parametrize("name", RELEASED_V8_EXAMPLE_NAMES)
def test_public_released_examples_report_best_effort_formulas_and_removals(name):
    from config.metrics import KNOWN_METRICS
    raw = released_v8_example_metrics(name)
    before = deepcopy(raw)
    result = migrate(raw, best_effort=True)
    assert all(entry["metric"] in KNOWN_METRICS
               for entry in result.config["optimize"]["scoring"] + result.config["optimize"]["limits"])
    assert set(result.config["backtest"]["reducer"]) <= {"default", *KNOWN_METRICS}
    assert any(row.get("best_effort") for row in result.changes)
    assert migrate(result.config).changes == []
    assert raw == before
