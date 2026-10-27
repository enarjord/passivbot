import operator

import pytest

from pareto_store import LimitSpec, _evaluate_limits


def test_limit_metric_mean_default():
    specs = [LimitSpec(metric="peak_recovery_max_days_strategy_equity", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_max_days_strategy_equity_mean": 750.0}
    aggregated = {}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_metric_aggregated_preferred():
    specs = [LimitSpec(metric="peak_recovery_max_days_strategy_equity", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_max_days_strategy_equity_mean": 900.0}
    aggregated = {"peak_recovery_max_days_strategy_equity": 700.0}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_metric_max():
    specs = [LimitSpec(metric="peak_recovery_max_days_strategy_equity", field="max", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_max_days_strategy_equity_max": 750.0}
    aggregated = {}
    objectives = {}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)
    stats_flat["peak_recovery_max_days_strategy_equity_max"] = 850.0
    assert not _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_limit_prefers_named_raw_objective():
    specs = [LimitSpec(metric="peak_recovery_max_days_strategy_equity", field="auto", op=operator.lt, value=800)]
    stats_flat = {"peak_recovery_max_days_strategy_equity_mean": 700.0}
    aggregated = {}
    objectives = {"peak_recovery_max_days_strategy_equity": 750.0}
    metric_map = {}
    assert _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)
    objectives["peak_recovery_max_days_strategy_equity"] = 900.0
    assert not _evaluate_limits(specs, stats_flat, aggregated, objectives, metric_map)


def test_metric_name_ending_with_suffix_is_not_misparsed():
    specs = [
        LimitSpec(metric="position_held_max_days", field="auto", op=operator.lt, value=800),
        LimitSpec(metric="position_held_time_weighted_mean_hours", field="auto", op=operator.lt, value=600),
    ]
    aggregated = {
        "position_held_max_days": 710.0,
        "position_held_time_weighted_mean_hours": 590.0,
    }
    stats_flat = {
        "position_held_max_days_mean": 900.0,
        "position_held_time_weighted_mean_hours_mean": 650.0,
    }
    assert _evaluate_limits(specs, stats_flat, aggregated, {}, {})


def test_missing_limit_metric_raises_instead_of_passing():
    specs = [LimitSpec(metric="missing_metric", field="auto", op=operator.lt, value=800)]
    with pytest.raises(ValueError, match="Limit metric 'missing_metric' could not be resolved"):
        _evaluate_limits(
            specs,
            {"peak_recovery_max_days_strategy_equity_mean": 750.0},
            {},
            {},
            {},
        )


def test_pareto_store_cli_retains_saved_infinite_objective(
    tmp_path, monkeypatch, capsys
):
    import json
    import sys
    import matplotlib.pyplot as plt
    from pareto_store import ParetoStore, main

    metric = "omega_ratio_strategy_equity"
    store = ParetoStore(directory=str(tmp_path), flush_interval=10000, max_size=10)
    entry = {
        "optimize": {"scoring": [metric]},
        "metrics": {
            "definition_version": 1,
            "objectives": {
                metric: None,
                "metric_status": {metric: {"status": "positive_infinity"}},
            },
            "constraint_violation": 0.0,
        },
    }
    assert store.add_entry(entry)
    store.flush_now()
    monkeypatch.setattr(
        sys, "argv", ["pareto_store.py", str(tmp_path / "pareto"), "--json"]
    )
    main()
    output = capsys.readouterr().out
    assert "Error loading" not in output
    summary = json.loads(
        output[output.index("{\n") :],
        parse_constant=lambda value: pytest.fail(f"raw nonfinite literal {value}"),
    )
    assert summary["n_members"] == 1
    for section in ("ideal", "closest"):
        assert summary[section][metric] is None
        assert (
            summary[section]["metric_status"][metric]["status"] == "positive_infinity"
        )
    plt.close("all")
