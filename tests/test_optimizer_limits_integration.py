from copy import deepcopy

import pytest
import math
from config.scoring import default_objective_goal

from config_utils import get_template_config
from config import prepare_config
from metrics_schema import build_scenario_metrics, flatten_metric_stats
from optimize import Evaluator


def _make_config(limits, scoring=None):
    cfg = deepcopy(get_template_config())
    cfg["optimize"]["scoring"] = [
        {"metric": item, "goal": default_objective_goal(item) or "min"} if isinstance(item, str) else item
        for item in (scoring or ["cagr_fraction_strategy_equity"])
    ]
    cfg["optimize"]["limits"] = limits
    return prepare_config(cfg, verbose=False)


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3, 0.5, 1.5, -0.5])
def test_profit_ratio_alias_matches_canonical_limit_fitness(ratio):
    results = []
    for name in ("short_net_profit_contribution_fraction", "short_net_profit_contribution_fraction"):
        cfg = _make_config(
            [{"metric": name, "penalize_if": "less_than_or_equal", "value": 0.2}]
        )
        evaluator = Evaluator({}, {}, {}, cfg)
        results.append(
            evaluator.calc_fitness({"cagr_fraction_strategy_equity_mean": 0.001, "short_net_profit_contribution_fraction_mean": ratio})
        )
    assert results[0] == results[1]
    assert (results[0][1] > 0.0) == (ratio <= 0.2)


def test_evaluator_applies_limit_penalties():
    limits = [
        {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.4, "reducer": "max"},
        {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than", "value": 0.001, "reducer": "min"},
        {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]},
        {"metric": "omega_ratio_strategy_equity", "penalize_if": "inside_range", "range": [1.0, 1.5]},
    ]
    cfg = _make_config(limits)
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "drawdown_max_fraction_strategy_equity_max": 0.45,
        "cagr_fraction_strategy_equity_mean": 0.001,
        "cagr_fraction_strategy_equity_min": 0.0008,
        "loss_profit_ratio_mean": 0.55,
        "omega_ratio_strategy_equity_mean": 1.2,
    }
    expected_modifier = (
        (0.45 - 0.4) * 1e6 + (0.001 - 0.0008) * 1e6 + (0.55 - 0.5) * 1e6 + min(0.2, 0.3) * 1e6
    )

    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == expected_modifier
    assert pytest.approx(penalty) == expected_modifier


def test_evaluator_returns_weighted_metric_when_within_limits():
    limits = [
        {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.4, "reducer": "max"},
        {"metric": "cagr_fraction_strategy_equity", "penalize_if": "less_than", "value": 0.001},
        {"metric": "loss_profit_ratio", "penalize_if": "outside_range", "range": [0.2, 0.5]},
        {"metric": "omega_ratio_strategy_equity", "penalize_if": "inside_range", "range": [1.0, 1.5]},
    ]
    cfg = _make_config(limits)
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "drawdown_max_fraction_strategy_equity_max": 0.2,
        "cagr_fraction_strategy_equity_mean": 0.002,
        "loss_profit_ratio_mean": 0.3,
        "omega_ratio_strategy_equity_mean": 1.8,
    }

    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == -(2 / math.pi) * math.atan(.002)  # adg_fraction weight is -1, so maximize adg_fraction
    assert pytest.approx(penalty) == 0.0


def test_evaluator_can_score_one_metric_source_and_limit_another():
    limits = [
        {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.4, "reducer": "max"},
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    evaluator = Evaluator({}, {}, {}, cfg)

    objective_stats = {"cagr_fraction_strategy_equity_mean": 0.002}
    suite_limit_stats = {"drawdown_max_fraction_strategy_equity_max": 0.45}
    scores, penalty = evaluator.calc_fitness(
        objective_stats,
        limit_metrics=suite_limit_stats,
    )

    expected_penalty = (0.45 - 0.4) * 1e6
    assert scores == pytest.approx((expected_penalty,))
    assert penalty == pytest.approx(expected_penalty)


def test_limit_penalty_applies_only_to_matching_objective():
    limits = [
        {
            "metric": "loss_profit_ratio",
            "penalize_if": "greater_than",
            "value": 0.5,
            "reducer": "mean",
        },
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity", "loss_profit_ratio"])
    evaluator = Evaluator({}, {}, {}, cfg)

    stats = {
        "cagr_fraction_strategy_equity_mean": 0.0015,
        "loss_profit_ratio_mean": 0.6,
    }
    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == -(2 / math.pi) * math.atan(.0015)  # unaffected objective
    assert pytest.approx(scores[1]) == (0.6 - 0.5) * 1e6
    assert pytest.approx(penalty) == (0.6 - 0.5) * 1e6


def test_limit_penalty_remains_global_for_non_scoring_metric():
    limits = [
        {"metric": "drawdown_max_fraction_strategy_equity", "penalize_if": "greater_than", "value": 0.4, "reducer": "max"},
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity", "loss_profit_ratio"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "cagr_fraction_strategy_equity_mean": 0.001,
        "loss_profit_ratio_mean": 0.3,
        "drawdown_max_fraction_strategy_equity_max": 0.6,
    }
    expected_penalty = (0.6 - 0.4) * 1e6
    scores, penalty = evaluator.calc_fitness(stats)
    assert pytest.approx(scores[0]) == expected_penalty
    assert pytest.approx(scores[1]) == expected_penalty
    assert pytest.approx(penalty) == expected_penalty


def test_loss_profit_ratio_scoring_preserves_true_infinite_zero_profit_status():
    cfg = _make_config([], scoring=[{"goal": "min", "metric": "loss_profit_ratio"}])
    analysis = {"loss_profit_ratio": None,
                "metric_status": {"loss_profit_ratio": {"status": "positive_infinity"}},
                "metric_metadata": {"definition_version": 1}}
    flat = flatten_metric_stats(build_scenario_metrics({"combined": analysis}, required_metrics=["loss_profit_ratio"])["stats"])
    assert flat["loss_profit_ratio_mean"] is None
    scores, penalty = Evaluator({}, {}, {}, cfg).calc_fitness(flat)
    assert scores == (1.,) and penalty == 0.


def test_missing_limit_metric_raises_instead_of_passing():
    limits = [
        {"metric": "fills_gap_max_days", "penalize_if": "greater_than", "value": 72.0},
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    evaluator = Evaluator({}, {}, {}, cfg)

    with pytest.raises(
        ValueError,
        match="missing optimizer limit metric 'fills_gap_max_days_mean'",
    ):
        evaluator.calc_fitness({"cagr_fraction_strategy_equity_mean": 0.001})


def test_missing_scoring_metric_raises_instead_of_zeroing():
    cfg = _make_config([], scoring=["fills_gap_max_days"])
    evaluator = Evaluator({}, {}, {}, cfg)

    with pytest.raises(
        ValueError,
        match="missing optimizer scoring metric 'fills_gap_max_days'",
    ):
        evaluator.calc_fitness({"cagr_fraction_strategy_equity_mean": 0.001})


def test_fill_gap_limit_uses_aggregate_default_stat():
    limits = [
        {
            "metric": "fills_gap_max_days",
            "penalize_if": "greater_than_or_equal",
            "value": 72.0,
        },
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "cagr_fraction_strategy_equity_mean": 0.001,
        "fills_gap_max_days_mean": 73.0,
    }

    scores, penalty = evaluator.calc_fitness(stats)

    assert pytest.approx(scores[0]) == (73.0 - 72.0) * 1e6
    assert pytest.approx(penalty) == (73.0 - 72.0) * 1e6


def test_fill_gap_limit_honors_explicit_max_stat():
    limits = [
        {
            "metric": "fills_gap_max_days",
            "penalize_if": "greater_than_or_equal",
            "value": 72.0,
            "reducer": "max",
        },
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    evaluator = Evaluator({}, {}, {}, cfg)
    stats = {
        "cagr_fraction_strategy_equity_mean": 0.001,
        "fills_gap_max_days_mean": 10.0,
        "fills_gap_max_days_max": 73.0,
    }

    scores, penalty = evaluator.calc_fitness(stats)

    assert pytest.approx(scores[0]) == (73.0 - 72.0) * 1e6
    assert pytest.approx(penalty) == (73.0 - 72.0) * 1e6


def test_limit_can_use_median_stat_emitted_by_metric_schema():
    limits = [
        {
            "metric": "fills_gap_max_days",
            "penalize_if": "greater_than",
            "value": 4.0,
            "reducer": "median",
        },
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    flat_stats = flatten_metric_stats(
        build_scenario_metrics(
            {
                "binance": {"cagr_fraction_strategy_equity": 0.001, "fills_gap_max_days": 1.0, "metric_metadata": {"definition_version": 1}},
                "bybit": {"cagr_fraction_strategy_equity": 0.001, "fills_gap_max_days": 5.0, "metric_metadata": {"definition_version": 1}},
                "kucoin": {"cagr_fraction_strategy_equity": 0.001, "fills_gap_max_days": 100.0, "metric_metadata": {"definition_version": 1}},
            }, reducers={"cagr_fraction_strategy_equity": {"mean"}, "fills_gap_max_days": {"median"}}
        )["stats"]
    )

    assert flat_stats["fills_gap_max_days_median"] == 5.0
    scores, penalty = Evaluator({}, {}, {}, cfg).calc_fitness(flat_stats)

    assert pytest.approx(scores[0]) == (5.0 - 4.0) * 1e6
    assert pytest.approx(penalty) == (5.0 - 4.0) * 1e6


def test_evaluator_combines_scenario_specific_and_suite_stat_limit_penalties():
    limits = [
        {
            "metric": "drawdown_max_fraction_strategy_equity",
            "penalize_if": "greater_than",
            "scenario": "base",
            "value": 0.5,
        },
        {
            "metric": "drawdown_max_fraction_strategy_equity",
            "penalize_if": "greater_than",
            "reducer": "max",
            "value": 0.7,
        },
    ]
    cfg = _make_config(limits, scoring=["cagr_fraction_strategy_equity"])
    evaluator = Evaluator({}, {}, {}, cfg)
    evaluator.build_limit_checks({"default": "mean"})

    scores, penalty = evaluator.calc_fitness(
        {"cagr_fraction_strategy_equity_mean": 0.002},
        limit_metrics={"drawdown_max_fraction_strategy_equity_max": 0.8},
        scenario_limit_metrics={
            "base": {"drawdown_max_fraction_strategy_equity_mean": 0.6},
        },
    )

    expected_penalty = ((0.6 - 0.5) + (0.8 - 0.7)) * 1e6
    assert scores == pytest.approx((expected_penalty,))
    assert penalty == pytest.approx(expected_penalty)


def test_scenario_specific_limit_requires_suite_metrics():
    limits = [
        {
            "metric": "drawdown_max_fraction_strategy_equity",
            "penalize_if": "greater_than",
            "scenario": "base",
            "value": 0.5,
        },
    ]
    evaluator = Evaluator({}, {}, {}, _make_config(limits))

    with pytest.raises(ValueError, match="requires suite mode"):
        evaluator.calc_fitness(
            {
                "cagr_fraction_strategy_equity_mean": 0.001,
                "drawdown_max_fraction_strategy_equity_mean": 0.6,
            }
        )


def test_iterative_backtester_rejects_scenario_specific_optimize_limits():
    from tools.iterative_backtester import build_limit_checks

    with pytest.raises(ValueError, match="require suite optimizer evaluation"):
        build_limit_checks(
            [
                {
                    "metric": "drawdown_max_fraction_strategy_equity",
                    "penalize_if": "greater_than",
                    "scenario": "base",
                    "value": 0.5,
                }
            ],
            {},
        )
