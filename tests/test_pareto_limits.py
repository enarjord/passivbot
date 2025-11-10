import operator

from pareto_store import LimitSpec, _evaluate_limits


def test_limit_metric_mean_default():
    specs = [LimitSpec(metric="peak_recovery_hours_pnl", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_hours_pnl_mean": 750.0}
    aggregated = {}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_metric_aggregated_preferred():
    specs = [LimitSpec(metric="peak_recovery_hours_pnl", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_hours_pnl_mean": 900.0}
    aggregated = {"peak_recovery_hours_pnl": 700.0}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_metric_max():
    specs = [LimitSpec(metric="peak_recovery_hours_pnl", field="max", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_hours_pnl_max": 750.0}
    aggregated = {}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)
    stats_flat["peak_recovery_hours_pnl_max"] = 850.0
    assert not _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_w_metric():
    specs = [LimitSpec(metric="w_4", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_hours_pnl_mean": 700.0}
    aggregated = {}
    objectives = {"w_4": 750.0}
    metric_map = {"w_4": "peak_recovery_hours_pnl"}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)
    objectives["w_4"] = 900.0
    assert not _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)
