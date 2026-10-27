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


def test_best_effort_does_not_invent_a_replacement_for_retired_median_growth():
    with pytest.raises(ValueError, match="metric-replacement"):
        migrate(source("mdg_strategy_eq"), best_effort=True)


def test_explicit_replacement_cannot_resurrect_a_retired_selector():
    with pytest.raises(ValueError, match="unknown or retired"):
        migrate(
            source("adg_strategy_eq"),
            memory_span_minutes=123.0,
            metric_replacements={"adg_strategy_eq": "adg_fraction_usd"},
            acknowledge_metric_formula_change=True,
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
def test_public_released_examples_do_not_guess_new_metric_formulas(name):
    raw = released_v8_example_metrics(name)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="metric-replacement"):
        migrate(raw, best_effort=True)
    assert raw == before
