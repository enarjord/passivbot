"""Exponential objectives remain opt-in and cannot dilute selected-source fitness."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from config.ew import (
    EW_METRICS,
    is_ew_metric,
    needs_legacy_weighted_analysis,
    validate_ew_config,
    validate_ew_half_life_days,
)
from config.metrics import canonicalize_metric_name, resolve_metric_value
from config.scoring import default_objective_goal, extract_objective_specs
from config.schema import get_template_config
from metrics_schema import (
    MetricAggregationError,
    build_scenario_metrics,
    build_standalone_metrics,
    merge_suite_payload,
    validate_required_ew_metrics,
)


@pytest.mark.parametrize("value", [True, False, "180", -1, 0, 0.99, float("nan"), float("inf")])
def test_half_life_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="ew_half_life_days"):
        validate_ew_half_life_days(value)


@pytest.mark.parametrize("value", [None, 1, 1.5, 180])
def test_half_life_accepts_days_and_preserves_disabled(value):
    assert validate_ew_half_life_days(value) == value


def test_default_scoring_and_weighted_semantics_unchanged():
    config = get_template_config()
    assert config["backtest"]["ew_half_life_days"] is None
    assert not any(is_ew_metric(spec.metric) for spec in extract_objective_specs(config))
    assert needs_legacy_weighted_analysis(spec.metric for spec in extract_objective_specs(config))
    assert resolve_metric_value({"adg_strategy_eq_w": 0.1}, "adg_strategy_eq_ew") is None
    assert len(EW_METRICS) == 12
    for metric in EW_METRICS:
        assert canonicalize_metric_name(metric) == metric
        assert default_objective_goal(metric) == "max"


@pytest.mark.parametrize("kind", ["scoring", "limits"])
def test_ew_objectives_and_limits_require_opt_in(kind):
    config = get_template_config()
    config["optimize"][kind] = (
        ["adg_strategy_eq_ew"] if kind == "scoring" else
        [{"metric": "adg_ew_btc", "penalize_if": "less_than", "value": 0}]
    )
    with pytest.raises(ValueError, match="require backtest.ew_half_life_days"):
        validate_ew_config(config)
    config["backtest"]["ew_half_life_days"] = 180.5
    validate_ew_config(config)
    assert config["backtest"]["ew_half_life_days"] == 180.5


@pytest.mark.parametrize("overrides", [
    {"backtest.ew_half_life_days": 30}, {"backtest": {"ew_half_life_days": 180}},
])
def test_scenarios_cannot_override_half_life_even_to_global_value(overrides):
    config = get_template_config()
    config["backtest"].update(ew_half_life_days=180, scenarios=[{"label": "base", "overrides": overrides}])
    with pytest.raises(ValueError, match="global half-life"):
        validate_ew_config(config)


@pytest.mark.parametrize("metrics,expected", [
    (["adg_strategy_eq_ew", "drawdown_worst_usd"], False),
    (["adg_strategy_eq_w"], True), (["adg_w_per_exposure_long_usd_mean"], True),
    (["adg_strategy_pnl_rebased_w"], True), (["sharpe_ratio_ew_btc"], False),
])
def test_legacy_analysis_selection(metrics, expected):
    assert needs_legacy_weighted_analysis(metrics) is expected


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf")])
def test_required_ew_metric_must_be_finite_in_every_exchange(bad):
    analyses = {"first": {"adg_ew_usd": 0.1}, "second": {"adg_ew_usd": bad}}
    with pytest.raises(MetricAggregationError, match="second"):
        validate_required_ew_metrics(analyses, ["adg_ew_usd_mean"])


def test_missing_currency_or_exchange_cannot_be_replaced():
    with pytest.raises(MetricAggregationError, match="adg_ew_btc"):
        validate_required_ew_metrics({"base": {"adg_ew_usd": .1, "adg_ew": .1}}, ["adg_ew_btc"])
    analyses = {"first": {"adg_ew_usd": .1}, "second": {"adg_usd": .2}}
    with pytest.raises(MetricAggregationError, match="second"):
        validate_required_ew_metrics(analyses, ["adg_ew_usd"])
    assert "adg_ew_usd" not in build_scenario_metrics(analyses)["stats"]


def test_metadata_is_preserved_outside_numeric_scoring():
    metadata = {"usd": {"half_life_days": 180.0, "definition_version": 1, "observation_count": 10}}
    analysis = {"adg_ew_usd": .1, "ew_metadata": metadata}
    scenario = build_scenario_metrics({"combined": analysis})
    assert scenario["ew_metadata"] == {"combined": metadata}
    assert set(scenario["stats"]) == {"adg_ew_usd"}
    standalone = build_standalone_metrics(analysis, "combined")
    assert standalone["ew_metadata"] == scenario["ew_metadata"]
    suite = merge_suite_payload({}, scenario_metrics={"base": scenario})
    assert suite["ew_metadata"] == {"base": scenario["ew_metadata"]}


def test_expand_analysis_extracts_rust_metadata_and_denomination():
    from backtest import expand_analysis
    metadata = {"usd": {"definition_version": 1}, "strategy_eq": {"definition_version": 1}}
    result = expand_analysis(
        {"adg_ew": .1, "adg_strategy_eq_ew": .2, "_ew_metadata": metadata}, {},
        fills=None, equities_array=None, config={"bot": {}},
    )
    assert result["adg_ew_usd"] == .1
    assert result["adg_strategy_eq_ew"] == .2
    assert result["ew_metadata"] == metadata
    assert not any(key.startswith("_ew_metadata") for key in result)
    assert "adg_ew_btc" not in result


def test_resume_half_life_invalidates_saved_fitness_and_absent_matches_null():
    from optimize import _resume_config_mismatches
    from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
    config = get_template_config()
    config["optimize"]["bounds"] = {}
    old = {**deepcopy(config), CONTRACT_KEY: build_evaluation_contract(config)}
    old["backtest"].pop("ew_half_life_days")
    assert _resume_config_mismatches(old, config) == []
    config["backtest"]["ew_half_life_days"] = 180.0
    assert any("ew_half_life_days" in mismatch for mismatch in _resume_config_mismatches(old, config))


def test_cpu_suite_rejects_missing_selected_scenario_before_reduction():
    from optimize import SuiteEvaluator
    from config.scoring import ObjectiveSpec, ObjectiveBasis
    evaluator = object.__new__(SuiteEvaluator)
    evaluator.base = SimpleNamespace(scoring_specs=[ObjectiveSpec("adg_strategy_eq_ew", "max")], limit_checks=[])
    evaluator.objective_bases = [ObjectiveBasis(scenario=None, reducer="mean")]
    evaluator.reducer_cfg = {"default": "mean"}
    results = [
        SimpleNamespace(scenario=SimpleNamespace(label="first"), metrics={"stats": {"adg_strategy_eq_ew": {"mean": .1}}}),
        SimpleNamespace(scenario=SimpleNamespace(label="second"), metrics={"stats": {"adg_strategy_eq": {"mean": .2}}}),
    ]
    with pytest.raises(MetricAggregationError, match="second"):
        evaluator.score_scenario_results(results)


def test_named_scenario_requires_only_selected_ew_sources():
    from optimize import SuiteEvaluator
    from config.scoring import ObjectiveSpec, ObjectiveBasis
    evaluator = object.__new__(SuiteEvaluator)
    evaluator.base = SimpleNamespace(scoring_specs=[ObjectiveSpec("adg_strategy_eq_ew", "max")], limit_checks=[])
    evaluator.objective_bases = [ObjectiveBasis(scenario="base", reducer=None)]
    assert evaluator.required_metrics_for_scenario("base") == ["adg_strategy_eq_ew"]
    assert evaluator.required_metrics_for_scenario("stress") == []


@pytest.mark.parametrize("half_life", [None, 180.5])
def test_canonical_config_roundtrip_retains_ew_setting(tmp_path, half_life):
    import json
    from config import prepare_config
    from config_utils import dump_config, sanitize_prepared_config_for_dump
    source = get_template_config()
    source["backtest"]["ew_half_life_days"] = half_life
    if half_life is not None:
        source["optimize"]["scoring"] = ["adg_strategy_eq_ew", "sharpe_ratio_ew_btc"]
    prepared = prepare_config(source, verbose=False)
    saved = sanitize_prepared_config_for_dump(prepared)
    dump_config(saved, str(tmp_path / "config.json"))
    restored = prepare_config(json.loads((tmp_path / "config.json").read_text()), verbose=False)
    assert restored["backtest"]["ew_half_life_days"] == half_life
    assert restored["optimize"]["scoring"] == prepared["optimize"]["scoring"]


def test_cpu_fitness_has_no_ew_currency_fallback():
    from optimize import Evaluator
    from config.scoring import ObjectiveSpec
    evaluator = object.__new__(Evaluator)
    evaluator.scoring_specs = [ObjectiveSpec("adg_ew_btc", "max")]
    evaluator.limit_checks = []
    evaluator._attachments = {}
    with pytest.raises(ValueError, match="missing optimizer scoring metric"):
        evaluator.calc_fitness({"adg_ew_mean": .1, "adg_ew_usd_mean": .1})


def test_scenario_apply_rejects_half_life_override():
    from suite_runner import apply_scenario_overrides
    with pytest.raises(ValueError, match="global half-life"):
        apply_scenario_overrides(get_template_config(), {"backtest.ew_half_life_days": 180})


def test_half_life_reaches_rust_payload_and_cached_policy_cannot_change():
    from backtest import prep_backtest_args
    from config import prepare_config
    raw = get_template_config()
    raw["backtest"]["ew_half_life_days"] = 90.5
    config = prepare_config(raw, verbose=False)
    config["backtest"]["coins"] = {"binance": ["BTC"]}
    market = {"BTC": dict(maker=.0002, taker=.00055, qty_step=.001, price_step=.1,
                          min_qty=.001, min_cost=10.0, c_mult=1.0)}
    *_, params = prep_backtest_args(config, market, "binance")
    assert params["ew_half_life_days"] == 90.5
    config["backtest"]["ew_half_life_days"] = 180
    with pytest.raises(ValueError, match="cached EW half-life"):
        prep_backtest_args(config, market, "binance", backtest_params=params)


def test_cpu_suite_rejects_absent_aggregate_source_scenario():
    from optimize import SuiteEvaluator
    from config.scoring import ObjectiveSpec, ObjectiveBasis
    evaluator = object.__new__(SuiteEvaluator)
    evaluator.base = SimpleNamespace(scoring_specs=[ObjectiveSpec("adg_strategy_eq_ew", "max")], limit_checks=[])
    evaluator.objective_bases = [ObjectiveBasis(scenario=None, reducer="mean")]
    evaluator.contexts = [SimpleNamespace(label="base"), SimpleNamespace(label="stress")]
    results = [SimpleNamespace(scenario=SimpleNamespace(label="base"), metrics={"stats": {"adg_strategy_eq_ew": {"mean": .1}}})]
    with pytest.raises(MetricAggregationError, match="stress.*absent"):
        evaluator.score_scenario_results(results)


@pytest.mark.parametrize("initial", [None, 90.0])
def test_cli_half_life_numeric_null_and_omission(initial):
    import argparse
    from config_utils import add_arguments_recursively, update_config_with_args
    config = {"backtest": {"ew_half_life_days": initial}}
    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, config)
    update_config_with_args(config, parser.parse_args([]))
    assert config["backtest"]["ew_half_life_days"] == initial
    update_config_with_args(config, parser.parse_args(["--backtest.ew_half_life_days", "180.5"]))
    assert config["backtest"]["ew_half_life_days"] == 180.5
    update_config_with_args(config, parser.parse_args(["--backtest.ew_half_life_days", "null"]))
    assert config["backtest"]["ew_half_life_days"] is None


def test_disabled_ew_limit_does_not_require_half_life():
    from config import prepare_config
    from config.ew import configured_ew_metrics
    config = get_template_config()
    config["optimize"]["limits"] = [
        {"metric": "adg_strategy_eq_ew", "enabled": False,
         "penalize_if": "less_than", "value": 0}
    ]
    assert configured_ew_metrics(config) == set()
    validate_ew_config(config)
    prepared = prepare_config(config, verbose=False)
    assert prepared["backtest"]["ew_half_life_days"] is None
    assert prepared["optimize"]["limits"][0]["enabled"] is False
    assert prepared["optimize"]["scoring"] == config["optimize"]["scoring"]


@pytest.mark.parametrize("metric,feature", [
    ("adg_strategy_eq_w", "weighted_raw_equity"),
    ("adg_w_usd", "weighted_account_equity"),
])
@pytest.mark.parametrize("enabled", [True, False])
def test_native_active_legacy_limits_control_analysis_capture(metric, feature, enabled):
    from optimization.native_results import CanonicalResultScorer
    from optimization.gpu.service import mps_requested_metric_features
    from optimize import Evaluator
    config = get_template_config()
    config["backtest"]["ew_half_life_days"] = 180.5
    config["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq_ew", "goal": "max"}
    ]
    config["optimize"]["limits"] = [
        {"metric": metric, "enabled": enabled,
         "penalize_if": "less_than", "value": 0}
    ]
    evaluator = Evaluator(hlcvs_specs={"binance": None}, btc_usd_specs={},
                          msss={}, config=config)
    required = CanonicalResultScorer(evaluator).required_metrics("base")
    assert "adg_strategy_eq_ew" in required
    assert (metric in required) is enabled
    features = mps_requested_metric_features(required, strategy_kind="trailing_martingale")
    assert "exponential_equity" in features
    assert (feature in features) is enabled
    other = {"weighted_raw_equity", "weighted_account_equity"} - {feature}
    assert not (features & other)
