from copy import deepcopy

import pytest

from backtest import expand_analysis
from config_utils import get_template_config
from metrics_schema import build_scenario_metrics, flatten_metric_stats
from optimize import Evaluator


def _make_config(limits, scoring=None):
    cfg = deepcopy(get_template_config())
    cfg["optimize"]["scoring"] = scoring or ["adg"]
    cfg["optimize"]["limits"] = limits
    return cfg


def test_evaluator_applies_limit_penalties():
    limits = [
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4, "stat": "max"},
        {"metric": "adg", "penalize_if": "less_than", "value": 0.001, "stat": "min"},
        {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]},
        {"metric": "omega_ratio", "penalize_if": "inside_range", "range": [1.0, 1.5]},
    ]
    cfg = _make_config(limits)
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "drawdown_worst_max": 0.45,
        "adg_mean": 0.001,
        "adg_min": 0.0008,
        "loss_profit_ratio_mean": 0.55,
        "omega_ratio_mean": 1.2,
    }
    expected_modifier = (
        (0.45 - 0.4) * 1e6 + (0.001 - 0.0008) * 1e6 + (0.55 - 0.5) * 1e6 + min(0.2, 0.3) * 1e6
    )

    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == expected_modifier
    assert pytest.approx(penalty) == expected_modifier


def test_evaluator_returns_weighted_metric_when_within_limits():
    limits = [
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4, "stat": "max"},
        {"metric": "adg", "penalize_if": "less_than", "value": 0.001},
        {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]},
        {"metric": "omega_ratio", "penalize_if": "inside_range", "range": [1.0, 1.5]},
    ]
    cfg = _make_config(limits)
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "drawdown_worst_max": 0.2,
        "adg_mean": 0.002,
        "loss_profit_ratio_mean": 0.3,
        "omega_ratio_mean": 1.8,
    }

    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == -0.002  # adg weight is -1, so maximize adg
    assert pytest.approx(penalty) == 0.0


def test_limit_penalty_applies_only_to_matching_objective():
    limits = [
        {
            "metric": "loss_profit_ratio",
            "penalize_if": "greater_than",
            "value": 0.5,
            "stat": "mean",
        },
    ]
    cfg = _make_config(limits, scoring=["adg", "loss_profit_ratio"])
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "adg_mean": 0.0015,
        "loss_profit_ratio_mean": 0.6,
    }
    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == -0.0015  # unaffected objective
    assert pytest.approx(scores[1]) == (0.6 - 0.5) * 1e6
    assert pytest.approx(penalty) == (0.6 - 0.5) * 1e6


def test_limit_penalty_remains_global_for_non_scoring_metric():
    limits = [
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4, "stat": "max"},
    ]
    cfg = _make_config(limits, scoring=["adg", "loss_profit_ratio"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "adg_mean": 0.001,
        "loss_profit_ratio_mean": 0.3,
        "drawdown_worst_max": 0.6,
    }
    expected_penalty = (0.6 - 0.4) * 1e6
    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == expected_penalty
    assert pytest.approx(scores[1]) == expected_penalty
    assert pytest.approx(penalty) == expected_penalty


def test_loss_profit_ratio_scoring_survives_finite_zero_profit_analysis():
    cfg = _make_config([], scoring=[{"goal": "min", "metric": "loss_profit_ratio"}])
    analysis = expand_analysis(
        {
            "loss_profit_ratio": 1000.0,
            "loss_profit_ratio_long": 1000.0,
            "loss_profit_ratio_short": 1.0,
        },
        {},
        None,
        [],
        cfg,
    )
    flat_stats = flatten_metric_stats(build_scenario_metrics({"combined": analysis})["stats"])

    assert flat_stats["loss_profit_ratio_mean"] == 1000.0
    scores, penalty = Evaluator({}, {}, {}, cfg).calc_fitness(flat_stats)
    assert scores == (1000.0,)
    assert penalty == 0.0


def test_missing_limit_metric_raises_instead_of_passing():
    limits = [
        {"metric": "fills_gap_p99_hours", "penalize_if": "greater_than", "value": 72.0},
    ]
    cfg = _make_config(limits, scoring=["adg"])
    evaluator = Evaluator({}, {}, {}, cfg)

    with pytest.raises(
        ValueError,
        match="missing optimizer limit metric 'fills_gap_p99_hours_mean'",
    ):
        evaluator.calc_fitness({"adg_mean": 0.001})


def test_missing_scoring_metric_raises_instead_of_zeroing():
    cfg = _make_config([], scoring=["fills_gap_p99_hours"])
    evaluator = Evaluator({}, {}, {}, cfg)

    with pytest.raises(
        ValueError,
        match="missing optimizer scoring metric 'fills_gap_p99_hours'",
    ):
        evaluator.calc_fitness({"adg_mean": 0.001})


def test_fill_gap_limit_uses_aggregate_default_stat():
    limits = [
        {
            "metric": "fills_gap_p99_hours",
            "penalize_if": "greater_than_or_equal",
            "value": 72.0,
        },
    ]
    cfg = _make_config(limits, scoring=["adg"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "adg_mean": 0.001,
        "fills_gap_p99_hours_mean": 73.0,
    }

    scores, penalty = evaluator.calc_fitness(stats)

    assert pytest.approx(scores[0]) == (73.0 - 72.0) * 1e6
    assert pytest.approx(penalty) == (73.0 - 72.0) * 1e6


def test_fill_gap_limit_honors_explicit_max_stat():
    limits = [
        {
            "metric": "fills_gap_p99_hours",
            "penalize_if": "greater_than_or_equal",
            "value": 72.0,
            "stat": "max",
        },
    ]
    cfg = _make_config(limits, scoring=["adg"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "adg_mean": 0.001,
        "fills_gap_p99_hours_mean": 10.0,
        "fills_gap_p99_hours_max": 73.0,
    }

    scores, penalty = evaluator.calc_fitness(stats)

    assert pytest.approx(scores[0]) == (73.0 - 72.0) * 1e6
    assert pytest.approx(penalty) == (73.0 - 72.0) * 1e6


def test_limit_can_use_median_stat_emitted_by_metric_schema():
    limits = [
        {
            "metric": "fills_gap_p99_hours",
            "penalize_if": "greater_than",
            "value": 4.0,
            "stat": "median",
        },
    ]
    cfg = _make_config(limits, scoring=["adg"])
    flat_stats = flatten_metric_stats(
        build_scenario_metrics(
            {
                "binance": {"adg": 0.001, "fills_gap_p99_hours": 1.0},
                "bybit": {"adg": 0.001, "fills_gap_p99_hours": 5.0},
                "kucoin": {"adg": 0.001, "fills_gap_p99_hours": 100.0},
            }
        )["stats"]
    )

    assert flat_stats["fills_gap_p99_hours_median"] == 5.0
    scores, penalty = Evaluator({}, {}, {}, cfg).calc_fitness(flat_stats)

    assert pytest.approx(scores[0]) == (5.0 - 4.0) * 1e6
    assert pytest.approx(penalty) == (5.0 - 4.0) * 1e6
