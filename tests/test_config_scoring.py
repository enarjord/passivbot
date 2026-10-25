import pytest

from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import (
    ScenarioSelection,
    default_objective_goal,
    normalize_scoring_entries,
    resolve_objective_basis,
)


def test_default_objective_goal_recognizes_new_ratio_metrics():
    assert default_objective_goal("adg_to_paper_loss_max_ratio") == "max"
    assert default_objective_goal("adg_to_paper_loss_mean_ratio") == "max"
    assert default_objective_goal("adg_to_absolute_net_exposure_max_ratio") == "max"
    assert default_objective_goal("adg_to_absolute_net_exposure_mean_ratio") == "max"
    assert default_objective_goal("adg_to_paper_loss_max_ratio_usd") == "max"
    assert default_objective_goal("adg_to_absolute_net_exposure_max_ratio_btc") == "max"


def test_normalize_scoring_entries_accepts_new_ratio_metrics():
    specs, changed = normalize_scoring_entries(
        ["adg_to_paper_loss_max_ratio", "adg_to_paper_loss_mean_ratio", "adg_to_absolute_net_exposure_max_ratio", "adg_to_absolute_net_exposure_mean_ratio"]
    )

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("adg_to_paper_loss_max_ratio_usd", "max"),
        ("adg_to_paper_loss_mean_ratio_usd", "max"),
        ("adg_to_absolute_net_exposure_max_ratio_usd", "max"),
        ("adg_to_absolute_net_exposure_mean_ratio_usd", "max"),
    ]


def test_default_objective_goal_recognizes_fill_activity_metrics():
    assert default_objective_goal("entry_interval_p95_hours") == "min"
    assert default_objective_goal("entry_interval_p99_hours") == "min"
    assert default_objective_goal("fills_gap_p95_hours") == "min"
    assert default_objective_goal("fills_gap_p99_hours") == "min"
    assert default_objective_goal("fills_gap_max_days") == "min"
    assert default_objective_goal("fills_gap_time_weighted_mean_hours") == "min"
    assert default_objective_goal("fills_per_day") == "max"
    assert default_objective_goal("fills_entry_per_day") == "max"
    assert default_objective_goal("fills_active_day_fraction") == "max"
    assert default_objective_goal("fills_top_symbol_fraction") == "min"
    assert default_objective_goal("backtest_completion_fraction") == "max"


def test_default_objective_goal_recognizes_gain_quality_metrics():
    assert default_objective_goal("adg_rolling_hmean_fraction_strategy_equity") == "max"
    assert default_objective_goal("adg_time_integrated_fraction_strategy_equity") == "max"
    assert default_objective_goal("positive_log_return_participation_score_strategy_equity") == "max"


def test_default_objective_goal_recognizes_strategy_eq_recovery_metrics():
    assert default_objective_goal("drawdown_max_fraction_strategy_equity_long") == "min"
    assert default_objective_goal("drawdown_max_fraction_strategy_equity_short") == "min"
    assert default_objective_goal("hsl_signal_ema_max_fraction") == "min"
    assert default_objective_goal("hsl_signal_ema_max_fraction_long") == "min"
    assert default_objective_goal("hsl_signal_ema_max_fraction_short") == "min"
    assert default_objective_goal("drawdown_daily_worst_1pct_mean_fraction_strategy_equity_long") == "min"
    assert default_objective_goal("drawdown_daily_worst_1pct_mean_fraction_strategy_equity_short") == "min"
    assert default_objective_goal("recovery_mean_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_median_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_p95_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_p99_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_worst_5pct_mean_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_worst_1pct_mean_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_max_days_strategy_equity") == "min"
    assert default_objective_goal("pain_index_fraction_strategy_equity") == "min"
    assert default_objective_goal("ulcer_index_fraction_strategy_equity") == "min"
    assert default_objective_goal("recovery_max_days_strategy_equity") == "min"
    assert default_objective_goal("recovery_max_hours_strategy_equity_long") == "min"
    assert default_objective_goal("recovery_max_hours_strategy_equity_short") == "min"
    assert default_objective_goal("recovery_max_days_strategy_equity_long") == "min"
    assert default_objective_goal("recovery_max_days_strategy_equity_short") == "min"


def test_drawdown_indices_are_shared_and_default_to_minimize():
    specs, changed = normalize_scoring_entries(
        ["pain_index_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity"]
    )

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("pain_index_fraction_strategy_equity", "min"),
        ("ulcer_index_fraction_strategy_equity", "min"),
    ]
    metrics = {
        "pain_index_fraction_strategy_equity": 0.12,
        "pain_index_fraction_strategy_equity_mean": 0.13,
    }
    assert resolve_metric_value(metrics, "pain_index_fraction_strategy_equity") == 0.12
    assert resolve_metric_value(metrics, "pain_index_fraction_strategy_equity_mean") == 0.13


def test_hard_stop_panic_close_drawdown_metrics_are_shared():
    metric = "hsl_panic_episode_loss_mean_fraction"

    assert canonicalize_metric_name(metric) == metric
    assert canonicalize_metric_name(f"usd_{metric}") == f"usd_{metric}"
    assert resolve_metric_value({metric: 0.125}, metric) == 0.125


def test_recovery_max_uses_one_canonical_name():
    specs, changed = normalize_scoring_entries(["recovery_max_days_strategy_equity"])

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("recovery_max_days_strategy_equity", "min")
    ]


@pytest.mark.parametrize("suffix", ["", "_mean", "_min", "_max", "_std", "_median"])
def test_profit_ratio_current_name_preserves_outer_statistic(suffix):
    current = f"long_pnl_to_total_pnl_ratio{suffix}"
    assert canonicalize_metric_name(current) == current
    assert resolve_metric_value({current: 0.3}, current) == 0.3
    assert resolve_metric_value({f"pnl_ratio_long_short{suffix}": 99.0}, current) is None


def test_strategy_equity_recovery_lookup_preserves_current_statistic():
    metrics = {
        "recovery_max_days_strategy_equity": 12.5,
        "recovery_max_days_strategy_equity_mean": 9.0,
    }

    assert resolve_metric_value(metrics, "recovery_max_days_strategy_equity") == 12.5
    assert resolve_metric_value(metrics, "recovery_max_days_strategy_equity_mean") == 9.0


def test_scoring_basis_preserves_inherit_named_and_explicit_reducer_scenarios():
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "adg_fraction_strategy_equity", "goal": "max"},
            {
                "metric": "pain_index_fraction_strategy_equity",
                "goal": "min",
                "scenario": " stress ",
            },
            {
                "metric": "recovery_max_days_strategy_equity",
                "goal": "min",
                "scenario": None,
                "aggregate": " MAX ",
            },
        ]
    )

    assert specs[0].scenario is ScenarioSelection.INHERIT
    assert specs[0].to_config() == {
        "metric": "adg_fraction_strategy_equity",
        "goal": "max",
    }
    assert specs[1].to_config() == {
        "metric": "pain_index_fraction_strategy_equity",
        "goal": "min",
        "scenario": "stress",
    }
    assert specs[2].to_config() == {
        "metric": "recovery_max_days_strategy_equity",
        "goal": "min",
        "scenario": None,
        "reducer": "max",
    }


def test_scoring_basis_resolves_omitted_and_null_scenario_in_both_default_directions():
    reducer_cfg = {
        "default": "mean",
        "recovery_max_days_strategy_equity": "max",
    }
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "adg_fraction_strategy_equity", "goal": "max"},
            {
                "metric": "pain_index_fraction_strategy_equity",
                "goal": "min",
                "scenario": None,
            },
            {
                "metric": "recovery_max_days_strategy_equity",
                "goal": "min",
                "scenario": None,
            },
            {
                "metric": "position_held_max_days",
                "goal": "min",
                "scenario": "stress",
            },
        ]
    )

    assert resolve_objective_basis(
        specs[0],
        default_scenario="base",
        reducer_cfg=reducer_cfg,
    ).scenario == "base"
    underwater_basis = resolve_objective_basis(
        specs[1],
        default_scenario="base",
        reducer_cfg=reducer_cfg,
    )
    assert underwater_basis.scenario is None
    assert underwater_basis.reducer == "mean"
    recovery_basis = resolve_objective_basis(
        specs[2],
        default_scenario="base",
        reducer_cfg=reducer_cfg,
    )
    assert recovery_basis.scenario is None
    assert recovery_basis.reducer == "max"

    inherited_reducer = resolve_objective_basis(
        specs[0],
        default_scenario=None,
        reducer_cfg=reducer_cfg,
    )
    assert inherited_reducer.scenario is None
    assert inherited_reducer.reducer == "mean"
    named_override = resolve_objective_basis(
        specs[3],
        default_scenario=None,
        reducer_cfg=reducer_cfg,
    )
    assert named_override.scenario == "stress"
    assert named_override.reducer is None


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            {
                "metric": "adg_fraction_strategy_equity",
                "goal": "max",
                "aggregate": "mean",
                "stat": "max",
            },
            "conflicting reducer aliases",
        ),
        (
            {
                "metric": "adg_fraction_strategy_equity",
                "goal": "max",
                "scenario": "base",
                "aggregate": "mean",
            },
            "cannot set both",
        ),
        (
            {"metric": "adg_fraction_strategy_equity", "goal": "max", "scenario": ""},
            "non-empty scenario",
        ),
        (
            {"metric": "adg_fraction_strategy_equity", "goal": "max", "aggregate": "p95"},
            "must be one of",
        ),
    ],
)
def test_scoring_basis_rejects_ambiguous_or_unknown_fields(entry, match):
    with pytest.raises(ValueError, match=match):
        normalize_scoring_entries([entry])


def test_scoring_reducer_override_requires_effective_suite_scenario():
    specs, _ = normalize_scoring_entries(
        [{"metric": "adg_fraction_strategy_equity", "goal": "max", "aggregate": "max"}]
    )

    with pytest.raises(ValueError, match="set scenario to null"):
        resolve_objective_basis(
            specs[0],
            default_scenario="base",
            reducer_cfg={"default": "mean"},
        )


def test_canonical_duration_loads_with_default_max_objective():
    metric = "analysis_duration_days"
    from config import prepare_config
    from config_utils import get_template_config

    config = get_template_config()
    config["optimize"]["scoring"] = [{"metric": metric, "goal": "max"}]
    prepared = prepare_config(config, verbose=False)
    assert prepared["optimize"]["scoring"] == [{"metric": "analysis_duration_days", "goal": "max"}]
    assert default_objective_goal(metric) == "max"


def test_position_held_time_weighted_metric_defaults_to_minimize():
    specs, _ = normalize_scoring_entries(["position_held_time_weighted_mean_hours"])
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("position_held_time_weighted_mean_hours", "min")
    ]


@pytest.mark.parametrize("old", ["strategy_eq_underwater_pct_mean", "strategy_eq_underwater_pct_median"])
@pytest.mark.parametrize("object_form", [False, True])
def test_removed_underwater_metrics_are_rejected_without_aliases(old, object_form):
    entry = {"metric": old, "goal": "min"} if object_form else old
    with pytest.raises(ValueError, match="daily-worst underwater metric.*removed"):
        normalize_scoring_entries([entry])


@pytest.mark.parametrize('old', ['strategy_eq_underwater_pct_mean', 'strategy_eq_underwater_pct_median'])
@pytest.mark.parametrize('surface', ['limit', 'visible', 'reducer'])
def test_removed_underwater_metrics_are_rejected_in_limits_visibility_and_reducers(old, surface):
    from analysis_visibility import validate_visible_metrics_config
    from config.limits import normalize_limit_entries
    from config.reducers import reducer_mapping_from_aliases
    with pytest.raises(ValueError, match='daily-worst underwater metric.*removed'):
        if surface == 'limit':
            normalize_limit_entries([{'metric': old, 'penalize_if': 'greater_than', 'value': .1}])
        elif surface == 'visible':
            validate_visible_metrics_config({'backtest': {'visible_metrics': [old]}})
        else:
            reducer_mapping_from_aliases({'reducer': {old: 'mean'}}, path='backtest')
