"""Known released formulas have observable best-effort choices, never runtime aliases."""

from copy import deepcopy
import logging

import pytest

from config.migrations.final_v9 import replace_retired_metrics
from config.migrations.released_v8_metrics import (
    RELEASED_V8_EW_REPLACEMENTS,
    RELEASED_V8_ORDINARY_FALLBACKS,
    RELEASED_V8_STAT_SUFFIXES,
    released_v8_ordinary_fallback,
)
from released_v8_fixtures import RELEASED_V8_EXAMPLE_NAMES, released_v8_example_metrics
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("old,choice", RELEASED_V8_ORDINARY_FALLBACKS.items())
@pytest.mark.parametrize("stat", [None, *RELEASED_V8_STAT_SUFFIXES])
def test_only_known_released_ordinary_families_keep_one_outer_statistic_without_ew(old, choice, stat, caplog):
    source_name = old if stat is None else f"{old}_{stat}"
    target = choice[0] if stat is None else f"{choice[0]}_{stat}"
    source = {"optimize": {"scoring": [{"metric": source_name, "goal": "max"}]}}
    changes = []
    with caplog.at_level(logging.WARNING):
        replace_retired_metrics(source, replacements={}, acknowledge=False, changes=changes, best_effort=True)
    assert source["optimize"]["scoring"] == [{"metric": target, "goal": "max"}]
    assert "ew_half_life_days" not in source["backtest"]
    assert len(changes) == 1
    assert changes[0]["old"] == source_name and changes[0]["new"] == target
    assert changes[0]["best_effort"] is True
    assert "full-window ordinary" in changes[0]["reason"] and choice[1] in changes[0]["reason"]
    assert "not equivalent" in changes[0]["reason"] and "shorter suffix" in changes[0]["reason"]
    assert source_name in caplog.text and target in caplog.text
    assert "adopt exponential" not in caplog.text


def test_choice_maps_are_frozen_and_ew_owner_remains_unchanged():
    assert len(RELEASED_V8_ORDINARY_FALLBACKS) == 4
    assert len(RELEASED_V8_EW_REPLACEMENTS) == 20
    assert not RELEASED_V8_ORDINARY_FALLBACKS.keys() & RELEASED_V8_EW_REPLACEMENTS.keys()
    with pytest.raises(TypeError):
        RELEASED_V8_ORDINARY_FALLBACKS["another_w"] = ("another", "invented")


@pytest.mark.parametrize("metric", [
    "another_w", "volume_pct_per_day_avg_w_sum", "adg_pnl_w_max_mean",
    "mdg_strategy_eq_w_p95", "mdg_pnl", "strategy_eq_underwater_pct_median",
])
def test_no_suffix_stripping_or_unpublished_family_inference(metric):
    assert released_v8_ordinary_fallback(metric) is None
    if "_w" in metric or metric.startswith("strategy_eq_underwater_pct_"):
        source = {"optimize": {"scoring": [{"metric": metric, "goal": "max"}]}}
        before = deepcopy(source)
        with pytest.raises(ValueError, match="no supported|no inferred"):
            replace_retired_metrics(source, replacements={}, acknowledge=False, changes=[], best_effort=True)
        assert source == before


@pytest.mark.parametrize("old", RELEASED_V8_ORDINARY_FALLBACKS)
def test_strict_conversion_still_requires_an_explicit_acknowledged_formula(old):
    source = {"optimize": {"scoring": [{"metric": old, "goal": "max"}]}}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="requires --metric-replacement"):
        replace_retired_metrics(source, replacements={}, acknowledge=True, changes=[])
    assert source == before
    with pytest.raises(ValueError, match="acknowledge-metric-formula-change"):
        replace_retired_metrics(source, replacements={old: RELEASED_V8_ORDINARY_FALLBACKS[old][0]},
                                acknowledge=False, changes=[])


@pytest.mark.parametrize("best_effort", [False, True])
def test_explicit_choice_has_priority_over_the_ordinary_fallback(best_effort):
    source = {"optimize": {"scoring": [{"metric": "adg_pnl_w", "goal": "max"}]}}
    changes = []
    replace_retired_metrics(source, replacements={"adg_pnl_w": "adg_fraction_strategy_equity_ew"},
                            acknowledge=True, changes=changes, best_effort=best_effort)
    assert source["optimize"]["scoring"][0]["metric"] == "adg_fraction_strategy_equity_ew"
    assert not any("full-window ordinary" in item["reason"] for item in changes)


@pytest.mark.parametrize("old,choice", RELEASED_V8_ORDINARY_FALLBACKS.items())
def test_all_metric_reference_contexts_report_their_own_fallback_path(old, choice):
    source = {"backtest": {"reducer": {old: "min"},
                           "scenarios": [{"overrides": {"optimize.scoring": [{"metric": old, "goal": "max"}]}}]},
              "optimize": {"scoring": [old], "limits": [{"metric": old, "penalize_if": "greater_than", "value": .125}],
                           "fixed_runtime_overrides": {"optimize.limits": {f"upper_bound_{old}": .25}}}}
    changes = []
    replace_retired_metrics(source, replacements={}, acknowledge=False, changes=changes, best_effort=True)
    expected = {f"backtest.reducer.{old}",
                "backtest.scenarios[0].overrides.optimize.scoring[0].metric", "optimize.scoring[0]",
                "optimize.limits[0].metric", f"optimize.fixed_runtime_overrides.optimize.limits.upper_bound_{old}"}
    assert {item["path"] for item in changes} == expected
    assert all(item["old"] == old and item["new"] == choice[0] and item["best_effort"] for item in changes)
    assert source["optimize"]["limits"][0]["value"] == .125
    assert source["optimize"]["fixed_runtime_overrides"]["optimize.limits"] == {f"upper_bound_{choice[0]}": .25}


@pytest.mark.parametrize("old,ordinary,target", [
    ("mdg_strategy_eq_w", "mdg_strategy_eq", "mdg_fraction_strategy_equity"),
    ("adg_pnl_w", "adg_pnl", "daily_realized_pnl_mean_fraction"),
])
def test_formula_choice_reports_the_actual_duplicate_objective_consolidation(old, ordinary, target, caplog):
    source = {"optimize": {"scoring": [{"metric": ordinary, "goal": "max"}, {"metric": old, "goal": "max"}]}}
    before = deepcopy(source["optimize"]["scoring"])
    changes = []
    with caplog.at_level(logging.WARNING):
        replace_retired_metrics(source, replacements={}, acknowledge=False, changes=changes, best_effort=True)
    row = next(item for item in changes if item["path"] == "optimize.scoring")
    assert row["old"] == before
    assert row["new"] == source["optimize"]["scoring"] == [{"metric": target, "goal": "max"}]
    assert (row["old_count"], row["new_count"], row["best_effort"]) == (2, 1, True)
    assert "2 -> 1" in row["reason"] and "2 -> 1" in caplog.text


def test_conflicting_objective_goals_are_never_coalesced():
    source = {"optimize": {"scoring": [{"metric": "adg_pnl", "goal": "min"},
                                        {"metric": "adg_pnl_w", "goal": "max"}]}}
    with pytest.raises(ValueError, match="conflicts with another objective"):
        replace_retired_metrics(source, replacements={}, acknowledge=False, changes=[], best_effort=True)


@pytest.mark.parametrize("old", RELEASED_V8_ORDINARY_FALLBACKS)
def test_best_effort_does_not_migrate_earlier_development_v9_metric_names(old):
    from config.schema import get_expanded_config_prototype

    source = get_expanded_config_prototype()
    source["optimize"]["scoring"] = [{"metric": old, "goal": "max"}]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="retired"):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("name", RELEASED_V8_EXAMPLE_NAMES)
def test_every_frozen_public_example_metric_selection_migrates_with_reported_choices(name):
    source = released_v8_example_metrics(name)
    original = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert source == original
    inferred = [item for item in result.changes if item.get("best_effort")]
    assert inferred and all(item.get("path") and item.get("reason") for item in inferred)
    for index, objective in enumerate(original["optimize"]["scoring"]):
        old = objective["metric"] if isinstance(objective, dict) else objective
        if old in RELEASED_V8_ORDINARY_FALLBACKS:
            target = RELEASED_V8_ORDINARY_FALLBACKS[old][0]
            assert any(item["path"] == f"optimize.scoring[{index}].metric" and item["old"] == old
                       and item["new"] == target and "full-window ordinary" in item["reason"] for item in inferred)
    if name in {"hsl_npos1", "xmr_long_short"}:
        assert any(item["path"] == "optimize.scoring" and item["new_count"] < item["old_count"] for item in inferred)
    assert migrate(result.config, best_effort=True).config == result.config
