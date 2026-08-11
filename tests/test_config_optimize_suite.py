from copy import deepcopy

import pytest

from config import load_prepared_config
from config_utils import format_config, get_template_config
from suite_runner import extract_suite_config


def test_optimize_suite_is_ignored_and_removed(caplog):
    """Test that optimize.suite is removed and warning is logged."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # Use new flattened structure for backtest scenarios
    base["backtest"]["scenarios"] = [{"label": "s1", "start_date": "2022-01-01"}]
    base["backtest"]["reducer"] = {"default": "mean"}
    # Add optimize.suite which should be removed with a warning
    base["optimize"]["suite"] = {"enabled": True, "aggregate": {"default": "median"}}

    formatted = format_config(deepcopy(base), verbose=False)

    # optimize.suite should be removed
    assert "suite" not in formatted.get("optimize", {})
    # backtest scenarios should remain
    assert formatted["backtest"]["scenarios"] == [{"label": "s1", "start_date": "2022-01-01"}]
    # Warning should have been logged about optimize.suite
    assert any("optimize.suite" in rec.message for rec in caplog.records)


def test_suite_reducer_default_preserved():
    """Test that the reducer default is preserved in the flattened structure."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["reducer"]["default"] = "median"
    formatted = format_config(deepcopy(base), verbose=False)
    assert formatted["backtest"]["reducer"]["default"] == "median"


@pytest.mark.parametrize("alias", ["reducer", "aggregate", "stat", "scenario_stat"])
def test_reducer_aliases_normalize_across_suite_scoring_and_limits(alias):
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

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["backtest"]["reducer"] == {"default": "max"}
    assert formatted["optimize"]["scoring"][0]["reducer"] == "max"
    assert formatted["optimize"]["limits"][0]["reducer"] == "max"
    for legacy_alias in {"aggregate", "stat", "scenario_stat"}:
        assert legacy_alias not in formatted["backtest"]
        assert legacy_alias not in formatted["optimize"]["scoring"][0]
        assert legacy_alias not in formatted["optimize"]["limits"][0]


def test_same_valued_reducer_aliases_collapse_to_canonical_output():
    base = get_template_config()
    base["backtest"]["aggregate"] = {"default": "mean"}
    base["optimize"]["scoring"] = [
        {
            "metric": "adg_strategy_eq",
            "goal": "max",
            "scenario": None,
            "reducer": "mean",
            "stat": "mean",
        }
    ]
    base["optimize"]["limits"] = [
        {
            "metric": "strategy_eq_recovery_days_max",
            "penalize_if": "greater_than",
            "reducer": "max",
            "scenario_stat": "max",
            "value": 100,
        }
    ]

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["backtest"]["reducer"] == {"default": "mean"}
    assert formatted["optimize"]["scoring"][0]["reducer"] == "mean"
    assert formatted["optimize"]["limits"][0]["reducer"] == "max"
    assert "stat" not in formatted["optimize"]["scoring"][0]
    assert "scenario_stat" not in formatted["optimize"]["limits"][0]


@pytest.mark.parametrize("scope", ["backtest", "scoring", "limits"])
def test_conflicting_reducer_aliases_fail_loudly(scope):
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

    with pytest.raises(ValueError, match="conflicting reducer aliases"):
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
            "aggregate": "max",
        },
    ]

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["scoring"][-1]["reducer"] == "max"
    assert "aggregate" not in formatted["optimize"]["scoring"][-1]


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
            "stat": "max",
            "value": 0.7,
        },
    ]

    formatted = format_config(deepcopy(base), verbose=False)

    assert formatted["optimize"]["limits"][-1]["reducer"] == "max"
    assert "stat" not in formatted["optimize"]["limits"][-1]


def test_optimizer_preserves_explicit_hsl_reducer_config():
    """Optimizer must not inherit template metric-specific reducer overrides."""
    cfg = load_prepared_config("configs/examples/hsl_npos1.json", verbose=False)
    suite_cfg = extract_suite_config(cfg, suite_override=None)

    assert suite_cfg["reducer"] == {"default": "mean"}


def test_legacy_suite_migration():
    """Test that legacy backtest.suite structure is migrated to new format."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # Inject legacy suite structure
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
    # Remove new-style keys to simulate old config
    base["backtest"].pop("scenarios", None)
    base["backtest"].pop("reducer", None)

    formatted = format_config(deepcopy(base), verbose=True)

    # suite wrapper should be removed
    assert "suite" not in formatted["backtest"]
    # scenarios should be at top level with base scenario prepended
    assert len(formatted["backtest"]["scenarios"]) == 3
    assert formatted["backtest"]["scenarios"][0]["label"] == "combined"  # base scenario
    assert formatted["backtest"]["scenarios"][1]["label"] == "binance_only"
    assert formatted["backtest"]["scenarios"][2]["label"] == "bybit_only"
    # reducer should be canonical at top level
    assert formatted["backtest"]["reducer"]["default"] == "median"
    assert "aggregate" not in formatted["backtest"]


def test_legacy_combine_ohlcvs_removed():
    """Test that legacy combine_ohlcvs is removed during migration."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["combine_ohlcvs"] = True

    formatted = format_config(deepcopy(base), verbose=True)

    # combine_ohlcvs should be removed
    assert "combine_ohlcvs" not in formatted["backtest"]
    # volume_normalization should exist
    assert "volume_normalization" in formatted["backtest"]
