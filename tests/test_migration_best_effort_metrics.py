"""Best-effort formula choices are observable; malformed policy is never erased."""

from copy import deepcopy
import json
import math

import pytest

from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import main, migrate


CAGR = "cagr_fraction_strategy_equity"
CAGR_EW = "cagr_fraction_strategy_equity_ew"
SHORT_SHARE = "short_net_profit_contribution_fraction"


def retired_source(metric, *, goal="max", limit=None):
    source = released_v8_policy()
    source["optimize"]["scoring"] = [{"metric": metric, "goal": goal}]
    if limit is not None:
        source["optimize"]["limits"] = [{"metric": metric, **limit}]
    return source


def removed(result, path, original):
    matches = [row for row in result.changes if row["path"] == path
               and row["new"] is None and row["old"] == original]
    assert len(matches) == 1
    assert matches[0].get("best_effort") is True
    assert matches[0]["reason"]
    return matches[0]


@pytest.mark.parametrize("old,new", [
    ("adg_strategy_eq", CAGR), ("adg_strategy_pnl_rebased", CAGR),
    ("adg_strategy_eq_w", CAGR_EW), ("adg_strategy_pnl_rebased_w", CAGR_EW),
    ("usd_adg_strategy_eq_w", CAGR_EW),
    ("usd_strategy_eq_recovery_days_max", "peak_recovery_max_days_strategy_equity"),
    ("btc_peak_recovery_days_strategy_eq", "peak_recovery_max_days_strategy_equity"),
    ("volume_pct_per_day_avg", "gross_turnover_ratio_per_day"),
    ("volume_pct_per_day_avg_w", "gross_turnover_ratio_per_day"),
    ("strategy_eq_underwater_pct_mean", "pain_index_fraction_strategy_equity"),
])
@pytest.mark.parametrize("acknowledge", [False, True])
def test_close_formula_choices_preserve_authored_objective_policy_and_report(old, new, acknowledge, caplog):
    source = retired_source(old)
    source["optimize"]["scoring"][0].update(goal="min", reducer="median", scenario=None)
    before = deepcopy(source)
    result = migrate(source, best_effort=True, acknowledge_metric_formula_change=acknowledge)
    assert result.config["optimize"]["scoring"] == [{
        "metric": new, "goal": "min", "reducer": "median", "scenario": None,
    }]
    choice = next(row for row in result.changes if row["path"] == "optimize.scoring[0].metric")
    assert choice["old"] == old and choice["new"] == new and choice.get("best_effort")
    assert "re-evaluate" in choice["reason"]
    assert old in caplog.text and new in caplog.text
    assert source == before


@pytest.mark.parametrize("metric", [
    "mdg_strategy_eq", "mdg_strategy_eq_w", "adg_pnl", "adg_pnl_w", "adg_btc",
    "adg_per_exposure_long", "hard_stop_restarts_per_year_long",
    "drawdown_worst_mean_1pct_strategy_eq", "fills_gap_p95_hours",
    "position_held_hours_mean", "peak_recovery_hours_strategy_eq",
])
def test_recognized_unsupported_entries_are_removed_whole_and_never_supply_defaults(metric, caplog):
    source = retired_source(metric, limit={"penalize_if": "greater_than", "value": 2.5, "reducer": "max"})
    source["backtest"]["reducer"] = {"default": "median", metric: "max"}
    before = deepcopy(source)
    result = migrate(source, best_effort=True, acknowledge_metric_formula_change=True)
    assert result.config["optimize"]["scoring"] == []
    assert result.config["optimize"]["limits"] == []
    assert result.config["backtest"]["reducer"] == {"default": "median"}
    removed(result, "optimize.scoring[0]", before["optimize"]["scoring"][0])
    removed(result, "optimize.limits[0]", before["optimize"]["limits"][0])
    assert any(row["path"] == f"backtest.reducer.{metric}" and row["new"] is None
               and row.get("best_effort") for row in result.changes)
    assert metric in caplog.text
    assert migrate(result.config).config["optimize"]["scoring"] == []
    assert source == before


def test_mixed_keep_infer_remove_uses_original_paths_and_coalesces_identical_choices():
    source = released_v8_policy()
    source["optimize"]["scoring"] = [
        {"metric": "mdg_strategy_eq", "goal": "max"},
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "adg_strategy_pnl_rebased", "goal": "max"},
        {"metric": "sharpe_ratio_strategy_eq", "goal": "max"},
    ]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [
        {"metric": CAGR, "goal": "max"}, {"metric": "sharpe_ratio_strategy_equity", "goal": "max"},
    ]
    removed(result, "optimize.scoring[0]", before["optimize"]["scoring"][0])
    assert any(row["path"] == "optimize.scoring[2].metric" and row["old"] == "adg_strategy_pnl_rebased"
               for row in result.changes)
    assert any(row["path"] == "optimize.scoring" and row.get("best_effort") for row in result.changes)
    assert source == before


@pytest.mark.parametrize("second", [
    {"goal": "min"}, {"goal": "max", "reducer": "max"}, {"goal": "max", "scenario": "base"},
])
def test_inferred_duplicate_objectives_do_not_resolve_conflicting_policy(second):
    source = released_v8_policy()
    source["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {"metric": "adg_strategy_pnl_rebased", **second},
    ]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflict"):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("bound", [-0.001, 0.0, 0.002])
@pytest.mark.parametrize("direction", ["less_than", "greater_than"])
def test_inferred_daily_growth_limits_annualize_the_nominal_bound(bound, direction):
    source = retired_source("adg_strategy_eq", limit={"penalize_if": direction, "value": bound, "reducer": "mean"})
    result = migrate(source, best_effort=True)
    limit = result.config["optimize"]["limits"][0]
    assert limit["metric"] == CAGR and limit["penalize_if"] == direction and limit["reducer"] == "mean"
    assert limit["value"] == pytest.approx(math.expm1(365.25 * math.log1p(bound)), rel=1e-12, abs=1e-15)
    assert any(row.get("best_effort") and row["path"].startswith("optimize.limits[0]")
               and "365.25" in row["reason"] for row in result.changes)


def test_growth_range_and_holding_hour_ranges_keep_units_and_order():
    source = released_v8_policy()
    source["optimize"]["limits"] = [
        {"metric": "adg_strategy_eq", "penalize_if": "outside_range", "range": [-0.001, 0.002], "reducer": "median"},
        {"metric": "position_held_hours_max", "penalize_if": "inside_range", "range": [24.0, 72.0], "reducer": "max"},
    ]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    growth, duration = result.config["optimize"]["limits"]
    assert growth["metric"] == CAGR and growth["penalize_if"] == "outside_range" and growth["reducer"] == "median"
    assert growth["range"] == pytest.approx([math.expm1(365.25 * math.log1p(x)) for x in [-0.001, 0.002]])
    assert duration == {"metric": "position_held_max_days", "penalize_if": "inside_range", "range": [1.0, 3.0], "reducer": "max"}
    assert source == before


@pytest.mark.parametrize("reducer,bound", [("std", .001), ("mean", -1.5), ("mean", 10000.0)])
def test_unconvertible_nominal_growth_limits_drop_with_full_policy_warning(reducer, bound):
    source = retired_source("adg_strategy_eq", limit={"penalize_if": "less_than", "value": bound, "reducer": reducer})
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == []
    removed(result, "optimize.limits[0]", before["optimize"]["limits"][0])
    assert result.config["optimize"]["scoring"] == [{"metric": CAGR, "goal": "max"}]
    assert source == before


@pytest.mark.parametrize("old", ["long_short_profit_ratio", "pnl_ratio_long_short"])
@pytest.mark.parametrize("reducer,new_reducer,goal", [
    ("min", "max", "min"), ("max", "min", "min"),
    ("mean", "mean", "min"), ("median", "median", "min"), ("std", "std", "max"),
])
def test_long_share_complement_changes_objective_goal_and_reducer_except_std(old, reducer, new_reducer, goal):
    source = retired_source(old)
    source["optimize"]["scoring"][0]["reducer"] = reducer
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": goal, "reducer": new_reducer}]


@pytest.mark.parametrize("direction,opposite", [
    ("greater_than", "less_than"), ("greater_than_or_equal", "less_than_or_equal"),
    ("less_than", "greater_than"), ("less_than_or_equal", "greater_than_or_equal"),
    ("equal_to", "equal_to"), ("not_equal", "not_equal"),
])
def test_long_share_complement_preserves_the_penalized_set(direction, opposite):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{"metric": "long_short_profit_ratio", "penalize_if": direction, "value": 1.25, "reducer": "min"}]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == [{"metric": SHORT_SHARE, "penalize_if": opposite, "value": -.25, "reducer": "max"}]


@pytest.mark.parametrize("direction", ["inside_range", "outside_range"])
def test_long_share_complements_range_endpoints_but_preserves_membership_mode(direction):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{"metric": "long_short_profit_ratio", "penalize_if": direction, "range": [.1, .6], "value": 1.25, "reducer": "median"}]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == [{"metric": SHORT_SHARE, "penalize_if": direction, "range": pytest.approx([.4, .9]), "reducer": "median"}]
    removed(result, "optimize.limits[0].value", 1.25)


def test_long_share_std_is_identity_for_goal_direction_and_bound():
    source = retired_source("long_short_profit_ratio", goal="min", limit={"penalize_if": "less_than", "value": .3, "reducer": "std"})
    source["optimize"]["scoring"][0]["reducer"] = "std"
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": "min", "reducer": "std"}]
    assert result.config["optimize"]["limits"] == [{"metric": SHORT_SHARE, "penalize_if": "less_than", "value": .3, "reducer": "std"}]


@pytest.mark.parametrize("old,target,value", [
    ("adg_strategy_eq", CAGR, .002),
    ("position_held_hours_max", "position_held_max_days", 48.0),
    ("long_short_profit_ratio", SHORT_SHARE, .2),
    ("mdg_strategy_eq", CAGR, .004),
])
def test_explicit_replacements_keep_authored_goals_directions_and_values(old, target, value):
    source = retired_source(old, limit={"penalize_if": "greater_than", "value": value, "reducer": "max"})
    before = deepcopy(source)
    result = migrate(source, best_effort=True, acknowledge_metric_formula_change=True, metric_replacements={old: target})
    assert result.config["optimize"]["scoring"] == [{"metric": target, "goal": "max"}]
    assert result.config["optimize"]["limits"] == [{"metric": target, "penalize_if": "greater_than", "value": value, "reducer": "max"}]
    assert source == before


@pytest.mark.parametrize("metric", ["adgg_strategy_eq", "adg_strategy_eq_w_typo", "unknown_w", "strategy_eq_underwater_pct_typo", "strategy_eq_recovery_days_max_w", "strategy_eq_recovery_day_max"])
def test_unknown_selectors_and_weighted_typos_are_not_removal_authority(metric):
    source = retired_source(metric)
    before = deepcopy(source)
    with pytest.raises(ValueError):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("patch", [
    {"goal": "sideways"}, {"reducer": "bogus"}, {"scenario": 3},
    {"reducer": "mean", "stat": "max"}, {"scenario": "base", "reducer": "mean"}, {"unexpected": 1},
])
def test_removal_does_not_hide_malformed_objective_policy(patch):
    source = retired_source("mdg_strategy_eq")
    source["optimize"]["scoring"][0].update(patch)
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("patch", [
    {"penalize_if": "bogus"}, {"value": "not-a-number"}, {"value": float("nan")},
    {"value": float("inf")}, {"reducer": "bogus"},
    {"scenario": 3}, {"reducer": "min", "field": "max"}, {"scenario": "base", "reducer": "mean"},
    {"unexpected": 1}, {"penalize_if": "outside_range", "range": [1.0]},
])
def test_removal_does_not_hide_malformed_limit_policy(patch):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{"metric": "hard_stop_restarts_per_year_long", "penalize_if": "greater_than", "value": 2.5, **patch}]
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


def test_legacy_limit_dictionary_and_reducer_rules_infer_remove_and_preserve_defaults():
    source = released_v8_policy()
    source["optimize"]["limits"] = {
        "penalize_if_lower_than_adg_strategy_eq": .001,
        "penalize_if_greater_than_position_held_hours_max": 48.0,
        "penalize_if_greater_than_hard_stop_restarts_per_year_long": 3.0,
    }
    source["backtest"]["reducer"] = {"default": "mean", "adg_strategy_eq": "median", "hard_stop_restarts_per_year_long": "max"}
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["backtest"]["reducer"] == {"default": "mean", CAGR: "median"}
    limits = result.config["optimize"]["limits"]
    assert len(limits) == 2
    assert limits[0]["metric"] == CAGR and limits[0]["penalize_if"] == "less_than"
    assert limits[0]["value"] == pytest.approx(math.expm1(365.25 * math.log1p(.001)))
    assert limits[1]["metric"] == "position_held_max_days" and limits[1]["value"] == 2.0
    assert source == before


def test_inferred_reducer_collision_does_not_choose_between_conflicting_rules():
    source = released_v8_policy()
    source["backtest"]["reducer"] = {"default": "mean", "adg_strategy_eq": "min", "adg_strategy_pnl_rebased": "max"}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflict"):
        migrate(source, best_effort=True)
    assert source == before


def test_cli_warns_and_reports_complete_removals_and_never_writes_on_unknown_typo(tmp_path):
    source = retired_source("adg_strategy_eq", limit={"penalize_if": "greater_than", "value": .002})
    unsupported = {"metric": "hard_stop_restarts_per_year_long", "penalize_if": "greater_than", "value": 3.0}
    source["optimize"]["limits"].append(unsupported)
    src, dst, report = [tmp_path / name for name in ("source.json", "converted.json", "report.json")]
    src.write_text(json.dumps(source));original = src.read_bytes()
    assert main([str(src), str(dst), "--best-effort", "--report", str(report)]) == 0
    assert src.read_bytes() == original
    converted, changes = json.loads(dst.read_text()), json.loads(report.read_text())["changes"]
    assert len(converted["optimize"]["limits"]) == 1
    assert any(row["path"] == "optimize.limits[1]" and row["old"] == unsupported and row["new"] is None and row.get("best_effort") for row in changes)
    source["optimize"]["scoring"][0]["metric"] = "adgg_strategy_eq"
    bad_src, bad_dst, bad_report = [tmp_path / name for name in ("bad.json", "bad-output.json", "bad-report.json")]
    bad_src.write_text(json.dumps(source))
    with pytest.raises(SystemExit) as error:
        main([str(bad_src), str(bad_dst), "--best-effort", "--report", str(bad_report)])
    assert error.value.code != 0
    assert not bad_dst.exists() and not bad_report.exists()


@pytest.mark.parametrize("surface", ["reducer", "legacy_limit", "unknown_reducer", "conflicting_reducer_maps"])
def test_removing_stale_rules_never_hides_malformed_maps(surface):
    source = released_v8_policy()
    retired = "hard_stop_restarts_per_year_long"
    if surface == "reducer":
        source["backtest"]["reducer"] = {"default": "mean", retired: "bogus"}
    elif surface == "legacy_limit":
        source["optimize"]["limits"] = {"penalize_if_greater_than_" + retired: "not-a-number"}
    elif surface == "unknown_reducer":
        source["backtest"]["reducer"] = {"default": "mean", "unknown_w": "max"}
    else:
        source["backtest"]["reducer"] = {"default": "mean", retired: "min"}
        source["backtest"]["aggregate"] = {"default": "mean", retired: "max"}
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("alias", ["aggregate", "stat", "scenario_stat"])
def test_original_reducer_alias_and_metric_identity_control_complement_policy(alias):
    source = retired_source("long_short_profit_ratio")
    source["backtest"].pop("reducer")
    source["backtest"][alias] = {"default": "mean", "pnl_ratio_long_short": "min"}
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": "min", "reducer": "max"}]
    assert result.config["backtest"]["reducer"] == {"default": "mean", SHORT_SHARE: "max"}
    assert source == before


def test_nonempty_legacy_suite_uses_its_original_metric_reducer_even_when_enabled_is_false():
    source = retired_source("long_short_profit_ratio")
    source["backtest"].pop("scenarios")
    source["backtest"]["suite"] = {
        "enabled": False, "scenarios": [{"label": "base"}],
        "aggregate": {"default": "mean", "pnl_ratio_long_short": "std"},
    }
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": "max", "reducer": "std"}]
    assert source == before


@pytest.mark.parametrize("scenario,expected", [
    ("inherit", {"metric": SHORT_SHARE, "goal": "min"}),
    (None, {"metric": SHORT_SHARE, "goal": "max", "scenario": None, "reducer": "std"}),
    ("base", {"metric": SHORT_SHARE, "goal": "min", "scenario": "base"}),
])
def test_objective_scenario_inheritance_and_explicit_null_select_the_original_basis(scenario, expected):
    source = retired_source("long_short_profit_ratio")
    source["optimize"]["objective_scenario"] = "base"
    source["backtest"]["reducer"] = {"default": "std"}
    if scenario != "inherit":
        source["optimize"]["scoring"][0]["scenario"] = scenario
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [expected]
    assert result.config["optimize"]["objective_scenario"] == "base"
    assert source == before


def test_long_share_with_unresolved_automatic_limit_direction_is_removed():
    source = released_v8_policy()
    source["optimize"]["limits"] = {"long_short_profit_ratio": .3}
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == []
    assert any(row["path"] == "optimize.limits.long_short_profit_ratio" and row["new"] is None
               and row.get("best_effort") for row in result.changes)
    assert source == before


def test_nominal_growth_total_loss_boundary_has_the_continuous_annual_value():
    source = retired_source("adg_strategy_eq", limit={"penalize_if": "less_than", "value": -1.0})
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"][0]["value"] == -1.0


@pytest.mark.parametrize("metric,value,expected,direction", [
    ("position_held_hours_max", 24.0, 1.0, "greater_than"),
    ("long_short_profit_ratio", .8, .2, "less_than"),
    ("adg_strategy_eq", .001, math.expm1(365.25 * math.log1p(.001)), "greater_than"),
])
def test_scalar_limit_transform_uses_active_value_even_when_inactive_range_is_present(metric, value, expected, direction):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": metric, "penalize_if": "greater_than", "value": value,
        "range": [0.0, 48.0], "reducer": "mean",
    }]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    limit = result.config["optimize"]["limits"][0]
    assert limit["value"] == pytest.approx(expected)
    assert limit["penalize_if"] == direction
    assert "range" not in limit
    removed(result, "optimize.limits[0].range", [0.0, 48.0])
    assert source == before


@pytest.mark.parametrize("metric", ["position_held_hours_max", "long_short_profit_ratio", "adg_strategy_eq"])
def test_active_nonfinite_scalar_cannot_hide_behind_an_inactive_finite_range(metric):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": metric, "penalize_if": "greater_than", "value": float("inf"),
        "range": [0.0, 1.0], "reducer": "mean",
    }]
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("explicit_empty,goal,reducer", [(True, "max", "std"), (False, "min", "mean")])
def test_active_suite_empty_reducer_map_and_omitted_map_keep_distinct_inheritance(explicit_empty, goal, reducer):
    source = retired_source("long_short_profit_ratio")
    source["backtest"]["reducer"] = {"default": "std"}
    source["backtest"].pop("scenarios")
    source["backtest"]["suite"] = {"enabled": True, "scenarios": [{"label": "base"}]}
    if explicit_empty:
        source["backtest"]["suite"]["aggregate"] = {}
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": goal, "reducer": reducer}]
    assert source == before


def test_inferred_outer_dispersion_string_needs_a_declared_objective_goal():
    source = released_v8_policy()
    source["optimize"]["scoring"] = ["adg_strategy_eq_std"]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == []
    removed(result, "optimize.scoring[0]", "adg_strategy_eq_std")
    assert source == before
    source["optimize"]["scoring"] = [{"metric": "adg_strategy_eq_std", "goal": "min"}]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["scoring"] == [{"metric": CAGR, "goal": "min", "reducer": "std"}]


def test_holding_hour_outer_standard_deviation_scales_to_days():
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": "position_held_hours_max_std", "penalize_if": "greater_than", "value": 48.0,
    }]
    result = migrate(source, best_effort=True)
    assert result.config["optimize"]["limits"] == [{
        "metric": "position_held_max_days", "penalize_if": "greater_than", "value": 2.0, "reducer": "std",
    }]


@pytest.mark.parametrize("inactive", [[0.0], [0.0, float("inf")]])
def test_malformed_inactive_bounds_are_not_erased_during_inference(inactive):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": "adg_strategy_eq", "penalize_if": "greater_than", "value": .001,
        "range": inactive, "reducer": "mean",
    }]
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("metric", ["adg_strategy_eq", "position_held_hours_max", "long_short_profit_ratio"])
@pytest.mark.parametrize("inactive", [["typo"], [0.0, float("inf")], [0.0, 1.0, 2.0]])
def test_inactive_list_value_is_validated_before_removal(metric, inactive):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": metric, "penalize_if": "outside_range", "range": [.001, .002],
        "value": inactive, "reducer": "mean",
    }]
    before = deepcopy(source)
    with pytest.raises((ValueError, TypeError)):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("metric,bounds,target,expected", [
    ("adg_strategy_eq", [.001, .002], CAGR,
     [math.expm1(365.25 * math.log1p(x)) for x in [.001, .002]]),
    ("position_held_hours_max", [24.0, 48.0], "position_held_max_days", [1.0, 2.0]),
    ("long_short_profit_ratio", [.1, .6], SHORT_SHARE, [.4, .9]),
])
@pytest.mark.parametrize("legacy_value_form", [False, True])
def test_range_constraints_keep_the_active_range_and_report_raw_value_removal(
    metric, bounds, target, expected, legacy_value_form,
):
    source = released_v8_policy()
    old_value = bounds if legacy_value_form else ["0", "48"]
    entry = {"metric": metric, "penalize_if": "outside_range", "value": old_value,
             "reducer": "mean"}
    if not legacy_value_form:
        entry["range"] = bounds
    source["optimize"]["limits"] = [entry]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    limit = result.config["optimize"]["limits"][0]
    assert limit["metric"] == target and limit["penalize_if"] == "outside_range"
    assert limit["range"] == pytest.approx(expected)
    assert "value" not in limit
    removed(result, "optimize.limits[0].value", old_value)
    assert source == before


@pytest.mark.parametrize("metric", ["long_short_profit_ratio", "hard_stop_restarts_per_year_long", "loss_profit_ratio"])
@pytest.mark.parametrize("bounds", [[0.0, float("nan")], [float("nan"), 0.0], ["0", "NaN"], {"low": "0", "high": "NaN"}])
@pytest.mark.parametrize("inactive", [False, True])
def test_raw_nonfinite_endpoints_fail_before_range_sorting_or_removal(metric, bounds, inactive):
    source = released_v8_policy()
    entry = {"metric": metric, "penalize_if": "outside_range", "range": bounds,
             "reducer": "mean"}
    if inactive:
        entry.update(range=[.1, .9], value=bounds)
    source["optimize"]["limits"] = [entry]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="finite"):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("metric", ["long_short_profit_ratio", "hard_stop_restarts_per_year_long", "loss_profit_ratio"])
def test_nonfinite_scalar_string_cannot_be_removed_or_hidden(metric):
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": metric, "penalize_if": "greater_than", "value": "NaN", "range": [0, 1],
    }]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="finite"):
        migrate(source, best_effort=True)
    assert source == before


def test_legacy_limit_dictionary_rejects_nonfinite_numeric_strings_before_conversion():
    source = released_v8_policy()
    source["optimize"]["limits"] = {"penalize_if_greater_than_loss_profit_ratio": "NaN"}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="finite"):
        migrate(source, best_effort=True)
    assert source == before


@pytest.mark.parametrize("pad_selector", [False, True])
@pytest.mark.parametrize("pad_rule", [False, True])
@pytest.mark.parametrize("growth", [False, True])
def test_original_reducer_identity_strips_metric_and_rule_whitespace(pad_selector, pad_rule, growth):
    metric = "adg_strategy_pnl_rebased" if growth else "long_short_profit_ratio"
    rule = "adg_strategy_eq" if growth else "pnl_ratio_long_short"
    if pad_selector:
        metric = " \t" + metric + "  "
    if pad_rule:
        rule = " \t" + rule + "  "
    source = retired_source(metric, limit={"penalize_if": "greater_than", "value": .001 if growth else .8})
    source["backtest"]["reducer"] = {"default": "mean", rule: "std"}
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    if growth:
        assert result.config["optimize"]["limits"] == []
        assert result.config["backtest"]["reducer"][CAGR] == "std"
    else:
        assert result.config["optimize"]["scoring"] == [{"metric": SHORT_SHARE, "goal": "max", "reducer": "std"}]
        assert result.config["optimize"]["limits"] == [{
            "metric": SHORT_SHARE, "penalize_if": "greater_than", "value": .8, "reducer": "std",
        }]
    assert source == before


def test_inactive_numeric_string_range_is_reported_in_its_authored_form():
    source = released_v8_policy()
    source["optimize"]["limits"] = [{
        "metric": "adg_strategy_eq", "penalize_if": "greater_than", "value": .001,
        "range": ["0", "48"], "reducer": "mean",
    }]
    before = deepcopy(source)
    result = migrate(source, best_effort=True)
    limit = result.config["optimize"]["limits"][0]
    assert limit["value"] == pytest.approx(math.expm1(365.25 * math.log1p(.001)))
    assert limit["metric"] == CAGR and "range" not in limit
    removed(result, "optimize.limits[0].range", ["0", "48"])
    assert source == before


@pytest.mark.parametrize("best_effort", [False, True])
def test_restored_fill_gap_metric_preserves_objective_limit_and_reducer(best_effort):
    metric = "fills_gap_time_weighted_mean_hours"
    source = retired_source(metric, goal="min", limit={
        "penalize_if": "greater_than", "value": 12.5, "reducer": "max",
    })
    source["backtest"]["reducer"] = {"default": "mean", metric: "median"}
    before = deepcopy(source)
    result = migrate(source, best_effort=best_effort, memory_span_minutes=1440.,
                     acknowledge_metric_formula_change=not best_effort)
    assert result.config["optimize"]["scoring"] == [{"metric": metric, "goal": "min"}]
    assert result.config["optimize"]["limits"] == source["optimize"]["limits"]
    assert result.config["backtest"]["reducer"][metric] == "median"
    assert source == before
