import pytest

from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import (
    ScenarioSelection,
    default_objective_goal,
    normalize_scoring_entries,
    resolve_objective_basis,
)












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






@pytest.mark.parametrize("suffix", ["", "_mean", "_min", "_max", "_std", "_median"])
def test_profit_ratio_current_name_preserves_outer_statistic(suffix):
    current = f"short_net_profit_contribution_fraction{suffix}"
    assert canonicalize_metric_name(current) == current
    assert resolve_metric_value({current: 0.3}, current) == 0.3
    assert resolve_metric_value({f"pnl_ratio_long_short{suffix}": 99.0}, current) is None


def test_strategy_equity_recovery_lookup_preserves_current_statistic():
    metrics = {
        "peak_recovery_max_days_strategy_equity": 12.5,
        "peak_recovery_max_days_strategy_equity_mean": 9.0,
    }

    assert resolve_metric_value(metrics, "peak_recovery_max_days_strategy_equity") == 12.5
    assert resolve_metric_value(metrics, "peak_recovery_max_days_strategy_equity_mean") == 9.0


def test_scoring_basis_preserves_inherit_named_and_explicit_reducer_scenarios():
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
            {
                "metric": "pain_index_fraction_strategy_equity",
                "goal": "min",
                "scenario": " stress ",
            },
            {
                "metric": "peak_recovery_max_days_strategy_equity",
                "goal": "min",
                "scenario": None,
                "aggregate": " MAX ",
            },
        ]
    )

    assert specs[0].scenario is ScenarioSelection.INHERIT
    assert specs[0].to_config() == {
        "metric": "cagr_fraction_strategy_equity",
        "goal": "max",
    }
    assert specs[1].to_config() == {
        "metric": "pain_index_fraction_strategy_equity",
        "goal": "min",
        "scenario": "stress",
    }
    assert specs[2].to_config() == {
        "metric": "peak_recovery_max_days_strategy_equity",
        "goal": "min",
        "scenario": None,
        "reducer": "max",
    }


def test_scoring_basis_resolves_omitted_and_null_scenario_in_both_default_directions():
    reducer_cfg = {
        "default": "mean",
        "peak_recovery_max_days_strategy_equity": "max",
    }
    specs, _ = normalize_scoring_entries(
        [
            {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
            {
                "metric": "pain_index_fraction_strategy_equity",
                "goal": "min",
                "scenario": None,
            },
            {
                "metric": "peak_recovery_max_days_strategy_equity",
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
                "metric": "cagr_fraction_strategy_equity",
                "goal": "max",
                "aggregate": "mean",
                "stat": "max",
            },
            "conflicting reducer aliases",
        ),
        (
            {
                "metric": "cagr_fraction_strategy_equity",
                "goal": "max",
                "scenario": "base",
                "aggregate": "mean",
            },
            "cannot set both",
        ),
        (
            {"metric": "cagr_fraction_strategy_equity", "goal": "max", "scenario": ""},
            "non-empty scenario",
        ),
        (
            {"metric": "cagr_fraction_strategy_equity", "goal": "max", "aggregate": "p95"},
            "must be one of",
        ),
    ],
)
def test_scoring_basis_rejects_ambiguous_or_unknown_fields(entry, match):
    with pytest.raises(ValueError, match=match):
        normalize_scoring_entries([entry])


def test_scoring_reducer_override_requires_effective_suite_scenario():
    specs, _ = normalize_scoring_entries(
        [{"metric": "cagr_fraction_strategy_equity", "goal": "max", "aggregate": "max"}]
    )

    with pytest.raises(ValueError, match="set scenario to null"):
        resolve_objective_basis(
            specs[0],
            default_scenario="base",
            reducer_cfg={"default": "mean"},
        )




@pytest.mark.parametrize("metric", ["position_held_time_weighted_mean_hours", "fills_gap_time_weighted_mean_hours"])
def test_time_weighted_duration_metric_defaults_to_minimize(metric):
    specs, _ = normalize_scoring_entries([metric])
    assert [(spec.metric, spec.goal) for spec in specs] == [
        (metric, "min")
    ]


@pytest.mark.parametrize("old", ["strategy_eq_underwater_pct_mean", "strategy_eq_underwater_pct_median"])
@pytest.mark.parametrize("object_form", [False, True])
def test_removed_underwater_metrics_are_rejected_without_aliases(old, object_form):
    entry = {"metric": old, "goal": "min"} if object_form else old
    with pytest.raises(ValueError, match="unknown or retired"):
        normalize_scoring_entries([entry])


@pytest.mark.parametrize('old', ['strategy_eq_underwater_pct_mean', 'strategy_eq_underwater_pct_median'])
@pytest.mark.parametrize('surface', ['limit', 'reducer'])
def test_removed_underwater_metrics_are_rejected_in_limits_and_reducers(old, surface):
    from config.limits import normalize_limit_entries
    from config.reducers import reducer_mapping_from_aliases
    with pytest.raises(ValueError, match='unknown or retired'):
        if surface == 'limit':
            normalize_limit_entries([{'metric': old, 'penalize_if': 'greater_than', 'value': .1}])
        else:
            reducer_mapping_from_aliases({'reducer': {old: 'mean'}}, path='backtest')


def test_recommended_roles_come_from_native_catalog_without_limiting_explicit_goals():
    from config.metrics import CORE_METRIC_CATALOG
    for name, field in CORE_METRIC_CATALOG.items():
        expected={"maximize":"max","minimize":"min","limit":None}[field["role"]]
        assert default_objective_goal(name)==expected
        for goal in ("min","max"):
            assert normalize_scoring_entries([{"metric":name,"goal":goal}])[0][0].goal==goal
