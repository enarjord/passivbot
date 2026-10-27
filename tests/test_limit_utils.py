import pytest

from limit_utils import expand_limit_checks, compute_limit_violation
from config.limits import resolve_aggregate_mode, resolve_limit_stat
from config.metrics import resolve_metric_value


def _single_violation(entry, value, weights=None, penalty_weight=1000.0):
    checks = expand_limit_checks([entry], weights or {}, penalty_weight=penalty_weight)
    assert len(checks) == 1
    return compute_limit_violation(checks[0], value)


def test_greater_than_limit_penalizes_excess():
    entry = {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.5}
    violation = _single_violation(entry, value=0.6, penalty_weight=10)
    assert violation == (0.6 - 0.5) * 10


def test_less_than_limit_penalizes_deficit_with_custom_stat():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than", "value": 0.001, "reducer": "min"}
    violation = _single_violation(entry, value=0.0002, penalty_weight=5)
    assert violation == (0.001 - 0.0002) * 5


def test_greater_than_or_equal_limit_penalizes_boundary_hit():
    entry = {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than_or_equal", "value": 0.5}
    assert _single_violation(entry, value=0.49, penalty_weight=10) == 0.0
    assert _single_violation(entry, value=0.5, penalty_weight=10) > 0.0


def test_less_than_or_equal_limit_penalizes_boundary_hit():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than_or_equal", "value": 0.001, "reducer": "min"}
    assert _single_violation(entry, value=0.0011, penalty_weight=5) == 0.0
    assert _single_violation(entry, value=0.001, penalty_weight=5) > 0.0


def test_equal_to_limit_penalizes_exact_match_only():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "equal_to", "value": 0.001, "reducer": "mean"}
    assert _single_violation(entry, value=0.001, penalty_weight=5) > 0.0
    assert _single_violation(entry, value=0.0011, penalty_weight=5) == 0.0


def test_not_equal_limit_penalizes_non_matching_values_only():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "not_equal", "value": 0.001, "reducer": "mean"}
    assert _single_violation(entry, value=0.001, penalty_weight=5) == 0.0
    assert _single_violation(entry, value=0.0011, penalty_weight=5) > 0.0


def test_outside_range_penalty_hits_when_value_leaves_band():
    entry = {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]}
    violation_low = _single_violation(entry, value=0.1, penalty_weight=7)
    violation_high = _single_violation(entry, value=0.9, penalty_weight=7)
    assert violation_low == (0.2 - 0.1) * 7
    assert violation_high == (0.9 - 0.5) * 7


def test_inside_range_penalty_hits_inside_band_only():
    entry = {"metric": "omega_ratio_strategy_equity", "penalize_if": "inside_range", "range": [1.5, 2.0]}
    violation_inside = _single_violation(entry, value=1.7, penalty_weight=11)
    violation_outside = _single_violation(entry, value=2.5, penalty_weight=11)
    assert violation_inside == min(1.7 - 1.5, 2.0 - 1.7) * 11
    assert violation_outside == 0.0


def test_auto_penalize_if_respects_scoring_weight_sign():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "auto", "value": 0.0005}
    weights = {"cagr_fraction_strategy_equity": -1.0}
    violation = _single_violation(entry, value=0.0001, weights=weights, penalty_weight=3)
    assert violation == (0.0005 - 0.0001) * 3


def test_auto_limit_without_scoring_weight_requires_explicit_direction():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "auto", "value": 0.0005}
    with pytest.raises(ValueError, match="requires an explicit penalty direction"):
        expand_limit_checks([entry], {}, penalty_weight=1000.0)


def test_unknown_auto_limit_metric_raises_before_weight_lookup():
    entry = {"metric": "adgg", "penalize_if": "auto", "value": 0.0005}

    with pytest.raises(ValueError, match="unknown optimizer limit metric 'adgg'"):
        expand_limit_checks([entry], {}, penalty_weight=1000.0)


def test_disabled_unknown_auto_limit_is_skipped():
    entry = {
        "metric": "adgg",
        "penalize_if": "auto",
        "value": 0.0005,
        "enabled": False,
    }
    assert expand_limit_checks([entry], {}, penalty_weight=1000.0) == []


def test_disabled_limit_entry_is_skipped():
    entry = {
        "metric": "drawdown_max_fraction_strategy_equity",
        "penalize_if": "greater_than",
        "value": 0.5,
        "enabled": False,
    }
    checks = expand_limit_checks([entry], {}, penalty_weight=1000.0)
    assert checks == []


def test_unknown_limit_metric_raises_with_suggestion():
    entry = {
        "metric": "drawdown_max_fraction_strategy_equiy",
        "penalize_if": "greater_than_or_equal",
        "value": 0.5,
    }

    with pytest.raises(ValueError) as exc_info:
        expand_limit_checks([entry], {}, penalty_weight=1000.0)

    message = str(exc_info.value)
    assert "unknown optimizer limit metric 'drawdown_max_fraction_strategy_equiy'" in message
    assert "Did you mean:" in message
    assert "drawdown_max_fraction_strategy_equity" in message


def test_exact_core_limit_metric_uses_separate_reducer():
    entry = {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.8}
    checks = expand_limit_checks([entry], {}, penalty_weight=1000.0)

    assert checks[0]["metric"] == "drawdown_max_fraction_strategy_equity"
    assert checks[0]["metric_key"] == "drawdown_max_fraction_strategy_equity_mean"


def test_omitted_reducer_uses_default_instead_of_operator_direction():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than_or_equal", "value": 0.001}
    checks = expand_limit_checks(
        [entry],
        {},
        penalty_weight=1000.0,
        reducer_cfg={"default": "mean"},
    )
    assert checks[0]["reducer"] == "mean"
    assert checks[0]["metric_key"] == "cagr_fraction_strategy_equity_mean"


def test_omitted_reducer_uses_metric_specific_override():
    entry = {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.8}
    checks = expand_limit_checks(
        [entry],
        {},
        penalty_weight=1000.0,
        reducer_cfg={"default": "mean", "drawdown_max_fraction_strategy_equity": "max"},
    )
    assert checks[0]["reducer"] == "max"
    assert checks[0]["metric_key"] == "drawdown_max_fraction_strategy_equity_max"


def test_metric_value_resolution_reads_exact_stored_reducer_fields():
    metrics = {
        "cagr_fraction_strategy_equity_mean": 0.001,
        "drawdown_max_fraction_strategy_equity_max": 0.42,
    }

    assert resolve_metric_value(metrics, "cagr_fraction_strategy_equity_mean") == 0.001
    assert resolve_metric_value(metrics, "drawdown_max_fraction_strategy_equity_max") == 0.42


def test_explicit_reducer_overrides_reducer_config():
    entry = {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than_or_equal", "value": 0.001, "reducer": "min"}
    checks = expand_limit_checks(
        [entry],
        {},
        penalty_weight=1000.0,
        reducer_cfg={"default": "mean", "cagr_fraction_strategy_equity": "max"},
    )
    assert checks[0]["reducer"] == "min"
    assert checks[0]["metric_key"] == "cagr_fraction_strategy_equity_min"


def test_legacy_reducer_helpers_accept_legacy_keyword_arguments():
    assert resolve_aggregate_mode("cagr_fraction_strategy_equity", aggregate_cfg={"default": "max"}) == "max"
    assert (
        resolve_limit_stat(
            {"metric": "cagr_fraction_strategy_equity", "reducer": "min"},
            aggregate_cfg={"default": "max"},
        )
        == "min"
    )


def test_named_scenario_limit_uses_scenario_mean():
    entry = {
        "metric": "drawdown_max_fraction_strategy_equity",
        "penalize_if": "greater_than",
        "scenario": "base",
        "value": 0.5,
    }
    checks = expand_limit_checks(
        [entry],
        {},
        penalty_weight=1000.0,
        reducer_cfg={"default": "max"},
    )

    assert checks[0]["scenario"] == "base"
    assert checks[0]["reducer"] == "mean"
    assert checks[0]["metric_key"] == "drawdown_max_fraction_strategy_equity_mean"


def test_explicit_null_scenario_uses_suite_stat():
    entry = {
        "metric": "drawdown_max_fraction_strategy_equity",
        "penalize_if": "greater_than",
        "scenario": None,
        "reducer": "max",
        "value": 0.7,
    }
    checks = expand_limit_checks(
        [entry],
        {},
        penalty_weight=1000.0,
        reducer_cfg={"default": "mean"},
    )

    assert checks[0]["scenario"] is None
    assert checks[0]["reducer"] == "max"
    assert checks[0]["metric_key"] == "drawdown_max_fraction_strategy_equity_max"
