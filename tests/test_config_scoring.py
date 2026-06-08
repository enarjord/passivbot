from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import default_objective_goal, normalize_scoring_entries


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
    assert default_objective_goal("peak_recovery_days_strategy_eq") == "min"


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
