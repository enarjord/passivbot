import pytest

from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import (
    ScenarioSelection,
    default_objective_goal,
    normalize_scoring_entries,
    resolve_objective_basis,
)


def test_default_objective_goal_recognizes_new_ratio_metrics():
    assert default_objective_goal("paper_loss_ratio") == "max"
    assert default_objective_goal("paper_loss_mean_ratio") == "max"
    assert default_objective_goal("exposure_ratio") == "max"
    assert default_objective_goal("exposure_mean_ratio") == "max"
    assert default_objective_goal("paper_loss_ratio_usd") == "max"
    assert default_objective_goal("exposure_ratio_btc") == "max"


def test_normalize_scoring_entries_accepts_new_ratio_metrics():
    specs, changed = normalize_scoring_entries(
        ["paper_loss_ratio", "paper_loss_ratio_w", "exposure_ratio", "exposure_mean_ratio_w"]
    )

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("paper_loss_ratio_usd", "max"),
        ("paper_loss_ratio_w_usd", "max"),
        ("exposure_ratio_usd", "max"),
        ("exposure_mean_ratio_w_usd", "max"),
    ]


def test_default_objective_goal_recognizes_fill_activity_metrics():
    assert default_objective_goal("entry_interval_hours_p95") == "min"
    assert default_objective_goal("entry_interval_hours_p99") == "min"
    assert default_objective_goal("fills_gap_p95_hours") == "min"
    assert default_objective_goal("fills_gap_p99_hours") == "min"
    assert default_objective_goal("fills_gap_longest_days") == "min"
    assert default_objective_goal("fills_per_day") == "max"
    assert default_objective_goal("fills_per_day_entry") == "max"
    assert default_objective_goal("fills_active_days_ratio") == "max"
    assert default_objective_goal("fills_top_symbol_share") == "min"
    assert default_objective_goal("backtest_completion_ratio") == "max"


def test_default_objective_goal_recognizes_strategy_eq_recovery_metrics():
    assert default_objective_goal("strategy_eq_recovery_days_mean") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_median") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_p95") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_p99") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_mean_worst_5pct") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_mean_worst_1pct") == "min"
    assert default_objective_goal("strategy_eq_recovery_days_max") == "min"
    assert default_objective_goal("strategy_eq_underwater_pct_mean") == "min"
    assert default_objective_goal("strategy_eq_underwater_pct_median") == "min"
    assert default_objective_goal("peak_recovery_days_strategy_eq") == "min"


def test_strategy_eq_underwater_metrics_are_shared_despite_stat_like_suffixes():
    specs, changed = normalize_scoring_entries(
        ["strategy_eq_underwater_pct_mean", "strategy_eq_underwater_pct_median"]
    )

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("strategy_eq_underwater_pct_mean", "min"),
        ("strategy_eq_underwater_pct_median", "min"),
    ]
    metrics = {
        "strategy_eq_underwater_pct_mean": 0.12,
        "strategy_eq_underwater_pct_mean_mean": 0.13,
    }
    assert resolve_metric_value(metrics, "strategy_eq_underwater_pct_mean") == 0.12
    assert resolve_metric_value(metrics, "strategy_eq_underwater_pct_mean_mean") == 0.13


def test_hard_stop_panic_close_drawdown_metrics_are_shared():
    metric = "hard_stop_panic_close_loss_drawdown_pct_mean"

    assert canonicalize_metric_name(metric) == metric
    assert canonicalize_metric_name(f"usd_{metric}") == metric
    assert resolve_metric_value({metric: 0.125}, metric) == 0.125


def test_peak_recovery_days_strategy_eq_normalizes_to_recovery_max_alias():
    specs, changed = normalize_scoring_entries(["peak_recovery_days_strategy_eq"])

    assert changed
    assert [(spec.metric, spec.goal) for spec in specs] == [
        ("strategy_eq_recovery_days_max", "min")
    ]


def test_strategy_eq_recovery_max_resolves_legacy_peak_metric_value():
    metrics = {
        "peak_recovery_days_strategy_eq": 12.5,
        "peak_recovery_days_strategy_eq_mean": 9.0,
    }

    assert resolve_metric_value(metrics, "strategy_eq_recovery_days_max") == 12.5
    assert resolve_metric_value(metrics, "strategy_eq_recovery_days_max_mean") == 9.0


def test_scoring_basis_preserves_inherit_named_and_explicit_reducer_scenarios():
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "adg_strategy_eq", "goal": "max"},
            {
                "metric": "strategy_eq_underwater_pct_mean",
                "goal": "min",
                "scenario": " stress ",
            },
            {
                "metric": "strategy_eq_recovery_days_max",
                "goal": "min",
                "scenario": None,
                "aggregate": " MAX ",
            },
        ]
    )

    assert specs[0].scenario is ScenarioSelection.INHERIT
    assert specs[0].to_config() == {
        "metric": "adg_strategy_eq",
        "goal": "max",
    }
    assert specs[1].to_config() == {
        "metric": "strategy_eq_underwater_pct_mean",
        "goal": "min",
        "scenario": "stress",
    }
    assert specs[2].to_config() == {
        "metric": "strategy_eq_recovery_days_max",
        "goal": "min",
        "scenario": None,
        "reducer": "max",
    }


def test_scoring_basis_resolves_omitted_and_null_scenario_in_both_default_directions():
    reducer_cfg = {
        "default": "mean",
        "strategy_eq_recovery_days_max": "max",
    }
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "adg_strategy_eq", "goal": "max"},
            {
                "metric": "strategy_eq_underwater_pct_mean",
                "goal": "min",
                "scenario": None,
            },
            {
                "metric": "strategy_eq_recovery_days_max",
                "goal": "min",
                "scenario": None,
            },
            {
                "metric": "position_held_days_max",
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
                "metric": "adg_strategy_eq",
                "goal": "max",
                "aggregate": "mean",
                "stat": "max",
            },
            "conflicting reducer aliases",
        ),
        (
            {
                "metric": "adg_strategy_eq",
                "goal": "max",
                "scenario": "base",
                "aggregate": "mean",
            },
            "cannot set both",
        ),
        (
            {"metric": "adg_strategy_eq", "goal": "max", "scenario": ""},
            "non-empty scenario",
        ),
        (
            {"metric": "adg_strategy_eq", "goal": "max", "aggregate": "p95"},
            "must be one of",
        ),
    ],
)
def test_scoring_basis_rejects_ambiguous_or_unknown_fields(entry, match):
    with pytest.raises(ValueError, match=match):
        normalize_scoring_entries([entry])


def test_scoring_reducer_override_requires_effective_suite_scenario():
    specs, _ = normalize_scoring_entries(
        [{"metric": "adg_strategy_eq", "goal": "max", "aggregate": "max"}]
    )

    with pytest.raises(ValueError, match="set scenario to null"):
        resolve_objective_basis(
            specs[0],
            default_scenario="base",
            reducer_cfg={"default": "mean"},
        )
