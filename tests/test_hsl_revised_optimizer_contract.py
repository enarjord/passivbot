"""Scope-aware revised optimizer inputs; offline, with the real native simulator."""
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from config.hsl_revised import validate_optimizer_metrics
from optimize import Evaluator, SuiteEvaluator, _canonicalize_optimizer_individual
from optimization.shape import build_optimization_shape
from test_hsl_revised_backtest_config import inputs, payload, run


def fixed_side_bounds(cfg):
    cfg["optimize"]["bounds"] = {
        side: {"risk": {"n_positions": [1, 1], "total_wallet_exposure_limit": [1, 1]}}
        for side in ("long", "short")
    }


def threshold_vector(shape, threshold):
    return [threshold if path[-1] == "red_threshold" else bound.low
            for (_, path), bound in zip(shape.key_paths, shape.bounds)]


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_candidate_threshold_reaches_native_revised_signal(mode):
    cfg, markets, _ = inputs(mode)
    fixed_side_bounds(cfg)
    target = cfg["optimize"]["bounds"] if mode == "unified" else cfg["optimize"]["bounds"]["long"]
    target["hsl"] = {"red_threshold": [.01, .9]}
    cfg["optimize"]["scoring"] = ["hard_stop_triggers"]
    cfg["optimize"]["limits"] = []
    shape = build_optimization_shape(cfg)
    assert sum(bound.low != bound.high for bound in shape.bounds) == 1
    counts = []
    original = deepcopy(cfg)
    for threshold in [.01, .9]:
        candidate = _canonicalize_optimizer_individual(
            threshold_vector(shape, threshold), cfg, shape.bounds, shape.sig_digits, shape.key_paths, [])
        args = payload(mode, cfg=candidate, mss=markets)
        hsl = args[-1]["equity_hard_stop_loss"]
        policy = hsl["portfolio"] if mode == "unified" else hsl["sides"][0]
        assert policy["red_threshold"] == threshold
        result = run(args)
        counts.append(result[2]["hard_stop_triggers"])
    assert counts[0] > 0 and counts[1] == 0
    assert cfg == original


@pytest.mark.parametrize("metric", [
    "hard_stop_triggers_long", "hard_stop_restarts_short_mean",
    "drawdown_worst_ema_strategy_eq_long", "drawdown_worst_ema_hsl_short_max",
    "drawdown_worst_mean_1pct_ema_hsl_long", "usd_drawdown_worst_ema_hsl_short",
])
def test_unified_rejects_inactive_side_signal_objectives(metric):
    cfg, _, _ = inputs("unified")
    with pytest.raises(ValueError, match="no side controller"):
        validate_optimizer_metrics(cfg, [metric])
    for mode in ["coin", "pside"]:
        cfg["live"]["hsl_signal_mode"] = mode
        validate_optimizer_metrics(cfg, [metric])
    cfg["live"].update(hsl_engine="legacy", hsl_signal_mode="unified")
    validate_optimizer_metrics(cfg, [metric])


def test_unified_keeps_general_side_performance_and_portfolio_signal_metrics():
    cfg, _, _ = inputs("unified")
    validate_optimizer_metrics(cfg, ["drawdown_worst_strategy_eq_long",
        "peak_recovery_hours_strategy_eq_short", "hard_stop_triggers",
        "drawdown_worst_ema_strategy_eq"])


@pytest.mark.parametrize("tier", ["yellow", "orange"])
def test_revised_runtime_boundary_rejects_removed_metrics(tier):
    cfg, _, _ = inputs()
    with pytest.raises(ValueError, match="removed for revised"):
        validate_optimizer_metrics(cfg, [f"hard_stop_time_in_{tier}_pct"])


def test_cpu_candidate_rejects_inactive_objective_before_dataset_attachment():
    cfg, _, _ = inputs("unified")
    fixed_side_bounds(cfg)
    cfg["optimize"]["bounds"]["hsl"] = {"red_threshold": [.01, .9]}
    cfg["optimize"]["scoring"] = ["drawdown_worst_ema_hsl_long"]
    cfg["optimize"]["limits"] = []
    evaluator = Evaluator({"binance": None}, {}, {}, cfg)
    with pytest.raises(ValueError, match="no side controller"):
        evaluator.evaluate(threshold_vector(evaluator.optimization_shape, .1), [])
    assert evaluator.shared_hlcvs_np == {}


@pytest.mark.parametrize("section", ["scoring", "limits"])
@pytest.mark.parametrize("scenario", ["side", "portfolio", None])
def test_suite_metric_validation_uses_only_its_selected_scenarios(section, scenario):
    cfg, _, _ = inputs("pside")
    cfg["optimize"]["scoring"] = ["drawdown_worst_strategy_eq"]
    cfg["optimize"]["limits"] = []
    cfg["optimize"][section] = [dict(metric="drawdown_worst_ema_hsl_long", scenario=scenario,
        **({"goal": "min"} if section == "scoring" else {"penalize_if": "greater_than", "value": .5}))]
    portfolio, _, _ = inputs("unified")
    cfg["bot"]["hsl"] = deepcopy(portfolio["bot"]["hsl"])
    fixed_side_bounds(cfg)
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    contexts = [
        SimpleNamespace(label="side", config=cfg, overrides={}),
        SimpleNamespace(label="portfolio", config=portfolio, overrides={
            "live.hsl_signal_mode": "unified", "bot.hsl": portfolio["bot"]["hsl"]}),
    ]
    evaluator = SuiteEvaluator(Evaluator({}, {}, {}, cfg), contexts, {"default": "mean"})
    evaluator.build_scenario_candidate_config(cfg, contexts[0])
    if scenario == "side":
        result = evaluator.build_scenario_candidate_config(cfg, contexts[1])
        assert result["live"]["hsl_signal_mode"] == "unified"
    else:
        with pytest.raises(ValueError, match="no side controller"):
            evaluator.build_scenario_candidate_config(cfg, contexts[1])


def test_gpu_rejects_revised_before_loading_gpu_runtime(monkeypatch):
    from optimization.backends.gpu_backend import run_backend
    cfg, _, _ = inputs()
    monkeypatch.setitem(sys.modules, "optimization.gpu.service", None)
    with pytest.raises(ValueError, match="GPU optimization does not implement revised HSL"):
        run_backend(config=cfg, evaluator=None, evaluator_for_pool=None, recorder=None,
            overrides_list=None, duplicate_counter=None, starting_configs_path=None,
            constraint_fitness_cls=None, ignore_sigint_in_worker=None,
            get_starting_configs=None, configs_to_individuals=None)


@pytest.mark.parametrize("through_cli", [False, True])
def test_gpu_preparation_rejects_revised_before_torch_probe(monkeypatch, through_cli):
    from optimize import _run_gpu_preparation_preflight
    from optimization.backends.gpu_backend import validate_gpu_preparation_scope
    cfg, _, _ = inputs()
    fixed_side_bounds(cfg)
    cfg["optimize"]["backend"] = "gpu"
    monkeypatch.setitem(sys.modules, "torch", None)
    preflight = _run_gpu_preparation_preflight if through_cli else validate_gpu_preparation_scope
    with pytest.raises(ValueError, match="GPU optimization does not implement revised HSL"):
        preflight(cfg, {"enabled": False})
