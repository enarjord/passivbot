"""Scope-aware hsl optimizer inputs; offline, with the real native simulator."""

from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest

from config.hsl import validate_optimizer_metrics
from optimize import Evaluator, SuiteEvaluator, _canonicalize_optimizer_individual
from optimization.shape import build_optimization_shape
from test_hsl_backtest_config import inputs, payload, run


def fixed_side_bounds(cfg):
    cfg["optimize"]["bounds"] = {
        side: {"risk": {"max_positions": [1, 1], "total_wallet_exposure_limit_ratio": [1, 1]}}
        for side in ("long", "short")
    }
    if cfg["bot"]["portfolio"]["hsl"]["scope"] == "portfolio":
        cfg["optimize"]["bounds"]["portfolio"] = {"hsl": {}}


def threshold_vector(shape, threshold):
    return [
        threshold if path[-1] == "drawdown_threshold_fraction" else bound.low
        for (_, path), bound in zip(shape.key_paths, shape.bounds)
    ]


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_candidate_threshold_reaches_native_hsl_signal(mode):
    cfg, markets, _ = inputs(mode)
    fixed_side_bounds(cfg)
    target = (
        cfg["optimize"]["bounds"]["portfolio"]
        if mode == "portfolio"
        else cfg["optimize"]["bounds"]["long"]
    )
    target["hsl"] = {"drawdown_threshold_fraction": [0.01, 0.9]}
    cfg["optimize"]["scoring"] = ["hsl_triggers_per_year"]
    cfg["optimize"]["limits"] = []
    shape = build_optimization_shape(cfg)
    assert sum(bound.low != bound.high for bound in shape.bounds) == 1
    counts = []
    original = deepcopy(cfg)
    for threshold in [0.01, 0.9]:
        candidate = _canonicalize_optimizer_individual(
            threshold_vector(shape, threshold),
            cfg,
            shape.bounds,
            shape.sig_digits,
            shape.key_paths,
            [],
        )
        args = payload(mode, cfg=candidate, mss=markets)
        hsl = args[-1]["equity_hard_stop_loss"]
        policy = hsl["portfolio"] if mode == "portfolio" else hsl["sides"][0]
        assert policy["drawdown_threshold_fraction"] == threshold
        result = run(args)
        counts.append(result[2]["metric_metadata"]["hsl_trigger_count"])
    assert counts[0] > 0 and counts[1] == 0
    assert cfg == original


@pytest.mark.parametrize(
    "metric",
    [
        "hsl_trigger_count_long",
        "hsl_restart_count_short_mean",
        "hsl_signal_ema_max_fraction_long",
        "hsl_signal_ema_max_fraction_short_max",
        "hsl_signal_ema_worst_1pct_mean_fraction_long",
        "hsl_signal_ema_max_fraction_short",
    ],
)
def test_retired_side_signal_objectives_are_rejected(metric):
    from config.metrics import reject_retired_metric

    for mode in ("coin", "position_side", "portfolio"):
        cfg, _, _ = inputs(mode)
        with pytest.raises(ValueError, match="unknown or retired core metric"):
            reject_retired_metric(metric, path="optimize.scoring")


def test_unified_keeps_core_performance_utilization_and_hsl_metrics():
    cfg, _, _ = inputs("portfolio")
    validate_optimizer_metrics(cfg, [
        "drawdown_max_fraction_strategy_equity", "peak_recovery_max_days_strategy_equity",
        "position_exposure_utilization_rms_ratio_long", "position_exposure_utilization_rms_ratio_short",
        "hsl_triggers_per_year", "hsl_red_time_fraction", "hsl_panic_episode_loss_max_fraction",
    ])


@pytest.mark.parametrize("tier", ["yellow", "orange"])
def test_hsl_runtime_boundary_rejects_removed_metrics(tier):
    cfg, _, _ = inputs()
    with pytest.raises(ValueError, match="unknown or retired core metric"):
        validate_optimizer_metrics(cfg, [f"hard_stop_time_in_{tier}_pct"])


def test_cpu_candidate_rejects_inactive_objective_before_dataset_attachment():
    cfg, _, _ = inputs("portfolio")
    fixed_side_bounds(cfg)
    cfg["optimize"]["bounds"].setdefault("portfolio", {})["hsl"] = {"drawdown_threshold_fraction": [0.01, 0.9]}
    cfg["bot"]["portfolio"]["hsl"]["enabled"] = False
    cfg["optimize"]["scoring"] = ["hsl_triggers_per_year"]
    cfg["optimize"]["limits"] = []
    evaluator = Evaluator({"binance": None}, {}, {}, cfg)
    with pytest.raises(ValueError, match="no enabled controller"):
        evaluator.evaluate(threshold_vector(evaluator.optimization_shape, 0.1), [])
    assert evaluator.shared_hlcvs_np == {}


@pytest.mark.parametrize("section", ["scoring", "limits"])
@pytest.mark.parametrize("scenario", ["side", "portfolio", None])
def test_suite_metric_validation_uses_only_its_selected_scenarios(section, scenario):
    cfg, _, _ = inputs("position_side")
    cfg["optimize"]["scoring"] = ["drawdown_max_fraction_strategy_equity"]
    cfg["optimize"]["limits"] = []
    cfg["optimize"][section] = [
        dict(
            metric="hsl_triggers_per_year",
            scenario=scenario,
            **(
                {"goal": "min"}
                if section == "scoring"
                else {"penalize_if": "greater_than", "value": 0.5}
            ),
        )
    ]
    portfolio, _, _ = inputs("portfolio")
    portfolio["bot"]["portfolio"]["hsl"]["enabled"] = False
    fixed_side_bounds(cfg)
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    contexts = [
        SimpleNamespace(label="side", config=cfg, overrides={}, msss={}),
        SimpleNamespace(
            label="portfolio",
            config=portfolio,
            msss={},
            overrides={
                "bot.portfolio.hsl": portfolio["bot"]["portfolio"]["hsl"],
            },
        ),
    ]
    evaluator = SuiteEvaluator(
        Evaluator({}, {}, {}, cfg), contexts, {"default": "mean"}
    )
    evaluator.build_scenario_candidate_config(cfg, contexts[0])
    if scenario == "side":
        result = evaluator.build_scenario_candidate_config(cfg, contexts[1])
        assert result["bot"]["portfolio"]["hsl"]["scope"] == "portfolio"
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            evaluator.build_scenario_candidate_config(cfg, contexts[1])


@pytest.mark.parametrize("through_cli", [False, True])
def test_gpu_preparation_rejects_coarse_hsl_candles_before_torch_probe(
    monkeypatch, through_cli
):
    from optimize import _run_gpu_preparation_preflight
    from config.gpu import validate_gpu_backtest_config

    cfg, _, _ = inputs()
    fixed_side_bounds(cfg)
    cfg["optimize"]["backend"] = "gpu"
    cfg["backtest"]["candle_interval_minutes"] = 1 if through_cli else 5
    if through_cli:
        cfg["optimize"]["fixed_runtime_overrides"] = {
            "backtest.candle_interval_minutes": 5,
        }
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "optimization.gpu.native", None)
    preflight = (
        _run_gpu_preparation_preflight
        if through_cli
        else lambda cfg, _suite: validate_gpu_backtest_config(cfg)
    )
    with pytest.raises(ValueError, match="GPU HSL requires 1m candles"):
        preflight(cfg, {"enabled": False})


@pytest.mark.parametrize("mode", ["coin", "position_side"])
@pytest.mark.parametrize(
    "metric",
    [
        "hsl_trigger_count_short",
        "hsl_signal_ema_max_fraction_short_max",
        "hsl_restart_count_short_mean",
    ],
)
def test_retired_side_signal_metrics_are_rejected(mode, metric):
    from config.metrics import reject_retired_metric

    cfg, _, _ = inputs(mode)
    for enabled in (False, True):
        cfg["bot"]["short"]["hsl"]["enabled"] = enabled
        with pytest.raises(ValueError, match="unknown or retired core metric"):
            reject_retired_metric(metric, path="optimize.scoring")


@pytest.mark.parametrize(
    "base,override,accepted",
    [
        (False, True, True),
        (True, False, False),
        (False, False, False),
        (True, True, True),
    ],
)
def test_coin_metrics_follow_effective_dataset_policy(base, override, accepted):
    cfg, markets, _ = inputs("coin")
    cfg["bot"]["long"]["hsl"]["enabled"] = False
    cfg["bot"]["short"]["hsl"]["enabled"] = base
    cfg["coin_overrides"] = {
        "AAA": {"bot": {"short": {"hsl": {"enabled": override}}}},
        "OUTSIDE": {"bot": {"short": {"hsl": {"enabled": True}}}},
    }
    if accepted:
        validate_optimizer_metrics(
            cfg, ["hsl_triggers_per_year"], markets_by_exchange={"binance": markets}
        )
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            validate_optimizer_metrics(
                cfg,
                ["hsl_triggers_per_year"],
                markets_by_exchange={"binance": markets},
            )


def test_coin_policy_metric_check_uses_same_alias_resolution_as_payload(monkeypatch):
    import backtest

    cfg, markets, _ = inputs("coin")
    cfg["bot"]["long"]["hsl"]["enabled"] = False
    cfg["coin_overrides"] = {
        "AAA/USDT:USDT": {"bot": {"short": {"hsl": {"enabled": True}}}}
    }
    markets["AAA"]["symbol"] = "AAA/USDT:USDT"
    monkeypatch.setattr(
        backtest, "coin_to_symbol", lambda value, *a, **k: "AAA/USDT:USDT"
    )
    validate_optimizer_metrics(
        cfg, ["hsl_triggers_per_year"], markets_by_exchange={"binance": markets}
    )


def test_retired_side_objective_rejected_before_cpu_dataset_attachment():
    cfg, _, _ = inputs("position_side")
    fixed_side_bounds(cfg)
    cfg["optimize"]["scoring"] = ["hsl_trigger_count_short"]
    cfg["optimize"]["limits"] = []
    evaluator = None
    with pytest.raises(ValueError, match="unknown or retired core metric"):
        evaluator = Evaluator({"binance": None}, {}, {}, cfg)
        evaluator.evaluate([bound.low for bound in evaluator.bounds], [])
    if evaluator is not None:
        assert evaluator.shared_hlcvs_np == {}


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize(
    "metric",
    [
        "hsl_triggers_per_year",
        "hsl_red_time_fraction",
        "hsl_panic_episode_loss_max_fraction",
        "hsl_panic_episode_loss_mean_fraction",
    ],
)
def test_disabled_portfolio_signal_metrics_are_rejected(mode, metric):
    cfg, markets, _ = inputs(mode)
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["enabled"] = False
    if mode == "portfolio":
        cfg["bot"]["portfolio"]["hsl"]["enabled"] = False
    with pytest.raises(ValueError, match="no enabled controller"):
        validate_optimizer_metrics(
            cfg, [metric], markets_by_exchange={"binance": markets}
        )
    validate_optimizer_metrics(cfg, ["drawdown_max_fraction_strategy_equity", "cagr_fraction_strategy_equity"])
    policy = cfg["bot"]["portfolio"]["hsl"] if mode == "portfolio" else cfg["bot"]["long"]["hsl"]
    policy["enabled"] = True
    validate_optimizer_metrics(cfg, [metric], markets_by_exchange={"binance": markets})


@pytest.mark.parametrize("override_enabled", [False, True])
def test_portfolio_metric_uses_effective_combined_coin_policies(override_enabled):
    cfg, markets, _ = inputs("coin")
    cfg["bot"]["long"]["hsl"]["enabled"] = False
    cfg["bot"]["long"]["hsl"]["enabled"] = not override_enabled
    cfg["backtest"]["coins"] = {"combined": ["AAA"]}
    cfg["coin_overrides"] = {
        "AAA": {"bot": {"long": {"hsl": {"enabled": override_enabled}}}},
        "OUTSIDE": {"bot": {"short": {"hsl": {"enabled": True}}}},
    }
    if override_enabled:
        validate_optimizer_metrics(
            cfg, ["hsl_triggers_per_year"], markets_by_exchange={"combined": markets}
        )
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            validate_optimizer_metrics(
                cfg, ["hsl_triggers_per_year"], markets_by_exchange={"combined": markets}
            )


@pytest.mark.parametrize("section", ["scoring", "limits"])
@pytest.mark.parametrize("scenario", ["active", "inactive", None])
def test_suite_portfolio_metrics_only_require_the_consumed_scenarios(section, scenario):
    cfg, _, _ = inputs("portfolio")
    fixed_side_bounds(cfg)
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    cfg["optimize"]["scoring"] = ["cagr_fraction_strategy_equity"]
    cfg["optimize"]["limits"] = []
    cfg["optimize"][section] = [
        dict(
            metric="hsl_triggers_per_year",
            scenario=scenario,
            **(
                {"goal": "min"}
                if section == "scoring"
                else {"penalize_if": "greater_than", "value": 5}
            ),
        )
    ]
    active = SimpleNamespace(label="active", config=cfg, overrides={}, msss={})
    inactive = SimpleNamespace(
        label="inactive", config=cfg, overrides={"bot.portfolio.hsl.enabled": False}, msss={}
    )
    suite = SuiteEvaluator(
        Evaluator({}, {}, {}, cfg), [active, inactive], {"default": "mean"}
    )
    suite.build_scenario_candidate_config(cfg, active)
    if scenario == "active":
        suite.build_scenario_candidate_config(cfg, inactive)
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            suite.build_scenario_candidate_config(cfg, inactive)


@pytest.mark.parametrize(
    "base_enabled,override_enabled", [(False, True), (True, False)]
)
def test_combined_suite_validates_actual_source_market_policy(
    base_enabled, override_enabled
):
    cfg, markets, _ = inputs("coin")
    cfg["bot"]["long"]["hsl"]["enabled"] = False
    fixed_side_bounds(cfg)
    cfg["bot"]["short"]["hsl"].update(enabled=base_enabled)
    cfg["backtest"]["coins"] = {"combined": ["AAA"]}
    cfg["coin_overrides"] = {
        "AAA": {
            "bot": {
                "short": {
                    "hsl": {
                        "enabled": override_enabled,
                    }
                }
            }
        }
    }
    cfg["optimize"]["scoring"] = [{"metric": "hsl_triggers_per_year", "goal": "min"}]
    cfg["optimize"]["limits"] = []
    ctx = SimpleNamespace(
        label="combined", config=cfg, overrides={}, msss={"combined": markets}
    )
    suite = SuiteEvaluator(Evaluator({}, {}, {}, cfg), [ctx], {"default": "mean"})
    if override_enabled:
        candidate = suite.build_scenario_candidate_config(cfg, ctx)
        # The native transport and the validator must resolve the same policy.
        from backtest import prep_backtest_args

        native = prep_backtest_args(candidate, markets, "combined")[-1][
            "equity_hard_stop_loss"
        ]
        assert native["coins"]["AAA"][1]["enabled"] is True
    else:
        with pytest.raises(ValueError, match="no enabled controller"):
            suite.build_scenario_candidate_config(cfg, ctx)
