from copy import deepcopy

import pytest

from config import load_prepared_config
from config_utils import format_config, get_template_config
from suite_runner import extract_suite_config


def test_retired_optimize_suite_is_rejected():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["scenarios"] = [{"label": "s1", "start_date": "2022-01-01"}]
    base["backtest"]["reducer"] = {"default": "mean"}
    base["optimize"]["suite"] = {"enabled": True, "aggregate": {"default": "median"}}
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(deepcopy(base), verbose=False)


def test_suite_reducer_default_preserved():
    """Test that the reducer default is preserved in the flattened structure."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["reducer"]["default"] = "median"
    formatted = format_config(deepcopy(base), verbose=False)
    assert formatted["backtest"]["reducer"]["default"] == "median"


@pytest.mark.parametrize("alias", ["reducer", "aggregate", "stat", "scenario_stat"])
def test_only_canonical_reducer_is_accepted_across_suite_scoring_and_limits(alias):
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"].pop("reducer", None)
    base["backtest"][alias] = {"default": "max"}
    base["optimize"]["scoring"] = [
        {
            "metric": "strategy_eq_recovery_days_max",
            "goal": "min",
            "scenario": None,
            alias: "max",
        }
    ]
    base["optimize"]["limits"] = [
        {
            "metric": "strategy_eq_recovery_days_max",
            "penalize_if": "greater_than",
            "scenario": None,
            alias: "max",
            "value": 100,
        }
    ]

    if alias != "reducer":
        with pytest.raises(ValueError, match="unknown configuration field"):
            format_config(deepcopy(base), verbose=False)
        return

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["backtest"]["reducer"] == {"default": "max"}
    assert formatted["optimize"]["scoring"][0]["reducer"] == "max"
    assert formatted["optimize"]["limits"][0]["reducer"] == "max"
    for legacy_alias in {"aggregate", "stat", "scenario_stat"}:
        assert legacy_alias not in formatted["backtest"]
        assert legacy_alias not in formatted["optimize"]["scoring"][0]
        assert legacy_alias not in formatted["optimize"]["limits"][0]


@pytest.mark.parametrize("scope", ["backtest", "scoring", "limits"])
def test_same_valued_retired_reducer_aliases_are_rejected(scope):
    base = get_template_config()
    base["optimize"]["scoring"] = [
        {
            "metric": "adg_strategy_eq",
            "goal": "max",
            "scenario": None,
            "reducer": "mean",
        }
    ]
    base["optimize"]["limits"] = [
        {
            "metric": "strategy_eq_recovery_days_max",
            "penalize_if": "greater_than",
            "reducer": "max",
            "value": 100,
        }
    ]

    if scope == "backtest":
        base["backtest"]["aggregate"] = {"default": "mean"}
    elif scope == "scoring":
        base["optimize"]["scoring"][0]["stat"] = "mean"
    else:
        base["optimize"]["limits"][0]["scenario_stat"] = "max"
    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(deepcopy(base), verbose=False)


@pytest.mark.parametrize("scope", ["backtest", "scoring", "limits"])
def test_conflicting_reducer_aliases_are_rejected_by_authored_boundary(scope):
    base = get_template_config()
    if scope == "backtest":
        base["backtest"]["aggregate"] = {"default": "max"}
    elif scope == "scoring":
        base["optimize"]["scoring"] = [
            {
                "metric": "adg_strategy_eq",
                "goal": "max",
                "reducer": "mean",
                "stat": "max",
            }
        ]
    else:
        base["optimize"]["limits"] = [
            {
                "metric": "strategy_eq_recovery_days_max",
                "penalize_if": "greater_than",
                "reducer": "mean",
                "scenario_stat": "max",
                "value": 100,
            }
        ]

    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(deepcopy(base), verbose=False)


def test_optimizer_objective_scenario_is_preserved():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["optimize"]["objective_scenario"] = " base "

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["objective_scenario"] == "base"


def test_optimizer_scoring_basis_round_trip_preserves_omitted_named_and_null_scenarios():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["optimize"]["objective_scenario"] = "base"
    base["optimize"]["scoring"] = [
        {"metric": "adg_strategy_eq", "goal": "max"},
        {
            "metric": "strategy_eq_underwater_pct_mean",
            "goal": "min",
            "scenario": None,
        },
        {
            "metric": "strategy_eq_recovery_days_max",
            "goal": "min",
            "scenario": "stress",
        },
        {
            "metric": "position_held_days_max",
            "goal": "min",
            "scenario": None,
            "reducer": "max",
        },
    ]

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["scoring"][-1]["reducer"] == "max"
    assert "aggregate" not in formatted["optimize"]["scoring"][-1]
    assert formatted["optimize"]["objective_scenario"] == "base"
    objectives = formatted["optimize"]["scoring"]
    assert "scenario" not in objectives[0]
    assert objectives[1]["scenario"] is None
    assert objectives[2]["scenario"] == "stress"
    assert objectives[3]["scenario"] is None


def test_optimizer_limit_basis_round_trip_preserves_named_and_null_scenarios():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["optimize"]["limits"] = [
        {
            "metric": "drawdown_worst_strategy_eq",
            "penalize_if": "greater_than",
            "scenario": "base",
            "value": 0.5,
        },
        {
            "metric": "drawdown_worst_strategy_eq",
            "penalize_if": "greater_than",
            "scenario": None,
            "reducer": "max",
            "value": 0.7,
        },
    ]

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["limits"][-1]["reducer"] == "max"
    assert "stat" not in formatted["optimize"]["limits"][-1]
    assert formatted["optimize"]["limits"][0]["scenario"] == "base"
    assert formatted["optimize"]["limits"][1]["scenario"] is None


def test_optimizer_preserves_explicit_hsl_reducer_config():
    """Optimizer must not inherit template metric-specific reducer overrides."""
    cfg = load_prepared_config("configs/examples/hsl_npos1.json", verbose=False)
    suite_cfg = extract_suite_config(cfg, suite_override=None)

    assert suite_cfg["reducer"] == {"default": "mean"}


def test_retired_suite_wrapper_is_rejected():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["suite"] = {
        "enabled": True,
        "include_base_scenario": True,
        "base_label": "combined",
        "aggregate": {"default": "median"},
        "scenarios": [
            {"label": "binance_only", "exchanges": ["binance"]},
            {"label": "bybit_only", "exchanges": ["bybit"]},
        ],
    }
    base["backtest"].pop("scenarios", None)
    base["backtest"].pop("reducer", None)

    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(deepcopy(base), verbose=False)


def test_flat_suite_scenarios_and_reducer_are_preserved():
    base = get_template_config()
    base["backtest"].update(suite_enabled=True, reducer={"default": "median"}, scenarios=[
        {"label": "combined"},
        {"label": "binance_only", "exchanges": ["binance"]},
        {"label": "bybit_only", "exchanges": ["bybit"]},
    ])
    formatted = format_config(deepcopy(base), verbose=False)

    assert "suite" not in formatted["backtest"]
    assert len(formatted["backtest"]["scenarios"]) == 3
    assert formatted["backtest"]["scenarios"][0]["label"] == "combined"
    assert formatted["backtest"]["scenarios"][1]["label"] == "binance_only"
    assert formatted["backtest"]["scenarios"][2]["label"] == "bybit_only"
    assert formatted["backtest"]["reducer"]["default"] == "median"
    assert "aggregate" not in formatted["backtest"]
    suite = extract_suite_config(formatted, None)
    assert suite["enabled"] is True
    assert suite["scenarios"] == base["backtest"]["scenarios"]


def test_retired_combine_ohlcvs_is_rejected():
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["combine_ohlcvs"] = True

    with pytest.raises(ValueError, match="unknown configuration field"):
        format_config(deepcopy(base), verbose=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_canonical_volume_normalization_is_preserved(enabled):
    base = get_template_config()
    base["backtest"]["volume_normalization"] = enabled
    formatted = format_config(deepcopy(base), verbose=False)
    assert formatted["backtest"]["volume_normalization"] is enabled
