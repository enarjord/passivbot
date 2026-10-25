"""Current metric quantities are exact; only released v8 has a conversion boundary."""

import pytest

from analysis_visibility import resolve_visible_metric_names, validate_visible_metrics_config
from config.ew import EW_METRICS
from config.metric_definitions import METRIC_DEFINITIONS
from config.metrics import (
    CURRENCY_METRICS, KNOWN_METRICS, canonicalize_metric_name,
    metric_with_curve, resolve_metric_value, split_metric_curve, split_metric_stat_suffix,
)
from config.migrations.final_v9 import replace_retired_metrics
from config.migrations.released_v8_metrics import RELEASED_V8_METRIC_NAMES, released_v8_metric_name
from config.schema import get_template_config
from config.scoring import normalize_scoring_entries
from metrics_schema import build_scenario_metrics, flatten_metric_stats


@pytest.mark.parametrize("base,curve,expected", [
    ("adg_fraction_ew", "btc", "adg_fraction_btc_ew"),
    ("gain_multiplier_per_twel_ratio_long", "usd", "gain_multiplier_per_twel_ratio_usd_long"),
    ("sharpe_ratio_ew", "strategy_equity", "sharpe_ratio_strategy_equity_ew"),
])
def test_curve_side_weighting_order_and_outer_statistics(base, curve, expected):
    assert metric_with_curve(base, curve) == expected
    assert canonicalize_metric_name(expected) == expected
    assert split_metric_stat_suffix(expected) == (expected, None)
    assert split_metric_stat_suffix(expected + "_mean") == (expected, "mean")
    if curve in {"usd", "btc"}:
        assert split_metric_curve(expected) == (base, curve)


def test_quantity_metadata_matches_all_exact_selectors_and_currency_variants():
    assert METRIC_DEFINITIONS.keys() == KNOWN_METRICS
    for base in CURRENCY_METRICS:
        for curve in ("usd", "btc"):
            name = metric_with_curve(base, curve)
            assert METRIC_DEFINITIONS[name].curve == curve
            assert canonicalize_metric_name(base) == metric_with_curve(base, "usd")
    assert all(name.endswith("_ew") for name in EW_METRICS)


@pytest.mark.parametrize("old", [
    "adg", "adg_ew_usd", "adg_strategy_eq", "gain_per_exposure_long_btc",
    "fills_analysis_duration_days", "n_days", "long_short_profit_ratio",
    "pnl_ratio_long_short", "peak_recovery_days_strategy_eq", "strategy_eq_recovery_days_max",
    "drawdown_worst_hsl", "pain_index_strategy_eq", "ulcer_index_strategy_eq",
    "total_wallet_exposure_rms_long", "hard_stop_time_in_red_pct",
])
def test_removed_development_names_are_not_runtime_aliases(old):
    assert canonicalize_metric_name(old) == old
    specs, _ = normalize_scoring_entries([{"metric": old, "goal": "max"}])
    assert specs[0].metric == old
    assert resolve_metric_value({"adg_fraction_usd": .0125}, old) is None
    config = get_template_config()
    config["optimize"].update(scoring=[], limits=[])
    config["backtest"]["visible_metrics"] = [old]
    # Existing family prefixes match current names without an alias table.
    if any(name.startswith(old + "_") for name in KNOWN_METRICS):
        validate_visible_metrics_config(config)
    else:
        with pytest.raises(ValueError, match="unknown backtest.visible_metrics"):
            validate_visible_metrics_config(config)


def test_shared_usd_loss_amount_is_not_an_account_curve_selector():
    name = "hsl_panic_fill_loss_max_usd"
    assert split_metric_curve(name) == (name, None)
    assert canonicalize_metric_name(name) == name
    assert METRIC_DEFINITIONS[name].unit == "usd"
    assert METRIC_DEFINITIONS[name].curve is None


def test_structured_outer_statistics_preserve_scale_and_intrinsic_statistic():
    payload = build_scenario_metrics({
        "a": {"drawdown_daily_worst_1pct_mean_fraction_usd": .0125},
        "b": {"drawdown_daily_worst_1pct_mean_fraction_usd": .0375},
    })
    flat = flatten_metric_stats(payload["stats"])
    assert flat["drawdown_daily_worst_1pct_mean_fraction_usd_mean"] == pytest.approx(.025)
    assert split_metric_stat_suffix(next(key for key in flat if key.endswith("_mean"))) == (
        "drawdown_daily_worst_1pct_mean_fraction_usd", "mean",
    )


def test_released_v8_vocabulary_is_explicit_and_emits_only_current_names():
    assert len(RELEASED_V8_METRIC_NAMES) == 245
    for old, new in RELEASED_V8_METRIC_NAMES.items():
        assert new in KNOWN_METRICS
        assert released_v8_metric_name(old) == new
    # EW, Pain and RMS selectors did not exist in the released v8 vocabulary.
    for dev_only in ("adg_strategy_eq_ew", "pain_index_strategy_eq", "total_wallet_exposure_rms_long"):
        with pytest.raises(ValueError, match="not a supported released-v8"):
            released_v8_metric_name(dev_only)


@pytest.mark.parametrize("first,second,new", [
    ("n_days", "fills_analysis_duration_days", "analysis_duration_days"),
    ("pnl_ratio_long_short", "long_short_profit_ratio", "long_pnl_to_total_pnl_ratio"),
    ("peak_recovery_days_strategy_eq", "strategy_eq_recovery_days_max", "recovery_max_days_strategy_equity"),
])
def test_released_alias_collisions_consolidate_equal_values_and_reject_conflicts(first, second, new):
    source = {"optimize": {"reducer": {first: "max", second: "max"}}}
    changes = []
    replace_retired_metrics(source, replacements={}, acknowledge=False, changes=changes)
    assert source["optimize"]["reducer"] == {new: "max"}
    assert all(change["new"] == new for change in changes)
    conflicting = {"optimize": {"reducer": {first: "max", second: "mean"}}}
    with pytest.raises(ValueError, match="conflicts with selected metric"):
        replace_retired_metrics(conflicting, replacements={}, acknowledge=False, changes=[])


def test_best_effort_retired_formula_choices_are_reported_as_new_formulas():
    source = {"optimize": {"scoring": ["adg_w_btc", "strategy_eq_underwater_pct_mean"]}}
    changes = []
    replace_retired_metrics(source, replacements={}, acknowledge=False, changes=changes, best_effort=True)
    assert source["optimize"]["scoring"] == [
        "adg_fraction_btc_ew", "pain_index_fraction_strategy_equity",
    ]
    assert source["backtest"]["ew_half_life_days"] == 180.0
    assert all(change.get("reason") for change in changes)


def test_canonical_recovery_max_preserves_actual_gpu_admission():
    pytest.importorskip("torch")
    from optimization.gpu.metrics import validate_gpu_metric_names

    specs, _ = normalize_scoring_entries(["recovery_max_days_strategy_equity"])
    assert validate_gpu_metric_names([specs[0].metric], native_replay=True, device="mps") == frozenset({
        "recovery_max_days_strategy_equity",
    })


def test_generic_math_selectors_keep_passthrough_without_owned_aliases():
    specs, _ = normalize_scoring_entries([{"metric": "m0", "goal": "min"}, {"metric": "metric_a", "goal": "max"}])
    assert [(spec.metric, spec.goal) for spec in specs] == [("m0", "min"), ("metric_a", "max")]
    assert canonicalize_metric_name("metric_a_mean") == "metric_a_mean"
    assert split_metric_stat_suffix("metric_a_mean") == ("metric_a", "mean")


def test_visibility_families_expand_only_current_producer_names():
    config = get_template_config()
    config["optimize"].update(scoring=[], limits=[])
    config["backtest"]["visible_metrics"] = ["hsl"]
    validate_visible_metrics_config(config)
    keys = ["hsl_trigger_count", "hsl_red_time_fraction", "fills_count"]
    assert resolve_visible_metric_names(config, keys) == keys[:2]


@pytest.mark.parametrize("prefix,renamed", [
    ("fills", {"analysis_duration_days", "entry_to_close_fill_count_ratio", "fills_entry_count"}),
    (" fills ", {"analysis_duration_days", "entry_to_close_fill_count_ratio", "fills_entry_count"}),
    ("drawdown", {"hsl_signal_ema_max_fraction", "drawdown_max_fraction_strategy_equity", "drawdown_max_fraction_usd"}),
    ("position_held", {"position_held_mean_hours", "position_held_max_days"}),
    ("equity_balance_diff", {"equity_balance_gap_negative_max_fraction_usd", "equity_balance_gap_positive_mean_fraction_btc"}),
])
def test_full_released_v8_migration_preserves_visibility_family_membership(prefix, renamed):
    from copy import deepcopy
    from config import prepare_config
    from config.migrations.released_v8 import get_released_v8_defaults
    from config.migrations.released_v8_metrics import released_v8_visibility_family
    from tools.migrate_config_v9 import migrate

    source = get_released_v8_defaults()
    source["backtest"]["visible_metrics"] = [prefix]
    source["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    prepared = prepare_config(result.config, verbose=False)
    selected = prepared["backtest"]["visible_metrics"]
    expected = {released_v8_metric_name(name) for name in released_v8_visibility_family(prefix)}
    assert set(selected) == expected
    assert renamed <= expected
    assert len(selected) == len(expected)
    assert source == before
    assert any(change["old"] == prefix and change["new"] == selected
               and "visibility family" in change["reason"] for change in result.changes)


def test_released_visibility_family_does_not_hide_retired_formula_choices():
    source = {"backtest": {"visible_metrics": ["sharpe"]}}
    with pytest.raises(ValueError, match="retired metric"):
        replace_retired_metrics(source, replacements={}, acknowledge=False, changes=[])
    assert source == {"backtest": {"visible_metrics": ["sharpe"]}}


@pytest.mark.parametrize("metric,bare", [
    ("adg_fraction_per_twel_ratio_usd_long", "adg_fraction_per_twel_ratio_long"),
    ("mdg_fraction_per_twel_ratio_btc_short", "mdg_fraction_per_twel_ratio_short"),
    ("gain_multiplier_per_twel_ratio_btc_long", "gain_multiplier_per_twel_ratio_long"),
    ("adg_fraction_usd", "adg_fraction"),
    ("custom_metric_usd", "custom_metric"),
    ("custom_metric_btc", "custom_metric"),
])
@pytest.mark.parametrize("suite", [False, True])
def test_scoring_preserves_curve_free_fallback_after_side_reordering(metric, bare, suite):
    result = _score_with_bare_metric(metric, bare, suite)
    assert result == pytest.approx((-.125,))


@pytest.mark.parametrize("metric,bare", [
    ("adg_fraction_usd_ew", "adg_fraction_ew"),
    ("hsl_panic_fill_loss_max_usd", "hsl_panic_fill_loss_max"),
])
@pytest.mark.parametrize("suite", [False, True])
def test_scoring_never_falls_back_from_ew_or_shared_cash_to_another_quantity(metric, bare, suite):
    with pytest.raises(ValueError, match="missing optimizer scoring metric|required EW metric"):
        _score_with_bare_metric(metric, bare, suite)


def _score_with_bare_metric(metric, bare, suite):
    from types import SimpleNamespace
    from optimize import Evaluator, SuiteEvaluator
    from suite_runner import ScenarioResult, SuiteScenario

    base = Evaluator.__new__(Evaluator)
    base._attachments = {}
    base.limit_checks = []
    base.scoring_specs, _ = normalize_scoring_entries([{"metric": metric, "goal": "max"}])
    if not suite:
        return base.calc_fitness({bare + "_mean": .125})[0]
    evaluator = SuiteEvaluator.__new__(SuiteEvaluator)
    evaluator.base = base
    evaluator.contexts = []
    evaluator.reducer_cfg = {"default": "mean"}
    evaluator.objective_bases = [SimpleNamespace(scenario=None, reducer="mean")]
    row = ScenarioResult(
        scenario=SuiteScenario("one", None, None, None, None), per_exchange={},
        metrics={"stats": {bare: dict(mean=.125, min=.125, max=.125, std=0, median=.125)}},
        elapsed_seconds=0, output_path=None,
    )
    return evaluator.score_scenario_results([row])["objectives"]
