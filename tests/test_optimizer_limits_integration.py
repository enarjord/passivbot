from copy import deepcopy

import pytest

from config_utils import get_template_config
from optimize import Evaluator


def _make_config(limits, scoring=None):
    cfg = deepcopy(get_template_config())
    cfg["optimize"]["scoring"] = scoring or ["adg"]
    cfg["optimize"]["limits"] = limits
    return cfg


def test_evaluator_applies_limit_penalties():
    limits = [
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4},
        {"metric": "adg", "penalize_if": "less_than", "value": 0.001, "stat": "min"},
        {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]},
        {"metric": "omega_ratio", "penalize_if": "inside_range", "range": [1.0, 1.5]},
    ]
    cfg = _make_config(limits)
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "drawdown_worst_max": 0.45,
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
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4},
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
        {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.4},
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
