import asyncio
from copy import deepcopy

import pytest

from suite_runner import (
    SuiteScenario,
    ScenarioResult,
    aggregate_metrics,
    apply_scenario,
    build_scenarios,
    extract_suite_config,
    resolve_coin_sources,
)


def test_extract_suite_config_merges_override():
    base = {"backtest": {"suite": {"enabled": False, "scenarios": ["base"]}}}
    override = {"enabled": True, "scenarios": ["override"]}
    merged = extract_suite_config(base, override)
    assert merged["enabled"] is True
    assert merged["scenarios"] == ["override"]


def test_build_scenarios_include_base():
    suite_cfg = {
        "scenarios": [{"label": "A"}],
        "include_base_scenario": True,
        "base_label": "base",
    }
    scenarios, aggregate_cfg, include_base, base_label = build_scenarios(suite_cfg)
    assert len(scenarios) == 1
    assert scenarios[0].label == "A"
    assert include_base is True
    assert base_label == "base"


def test_build_scenarios_handles_exchanges_and_coin_sources():
    suite_cfg = {
        "scenarios": [
            {
                "label": "X",
                "exchanges": ["binance", "bybit"],
                "coin_sources": {"BTC": "binance"},
            }
        ],
    }
    scenarios, *_ = build_scenarios(suite_cfg)
    scenario = scenarios[0]
    assert scenario.exchanges == ["binance", "bybit"]
    assert scenario.coin_sources == {"BTC": "binance"}


def test_apply_scenario_filters_unavailable_coins():
    base_config = {
        "backtest": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "coins": {},
            "cache_dir": {},
        },
        "live": {
            "approved_coins": {"long": [], "short": []},
            "ignored_coins": {"long": [], "short": []},
        },
    }
    scenario = SuiteScenario(
        label="test",
        start_date=None,
        end_date=None,
        coins=["BTC", "UNKNOWN"],
        ignored_coins=["ETH"],
    )
    cfg, coins = apply_scenario(
        base_config,
        scenario,
        master_coins=["BTC"],
        master_ignored=["ETH"],
        available_exchanges=["binance"],
        available_coins={"BTC"},
    )
    assert coins == ["BTC"]
    assert cfg["live"]["approved_coins"]["long"] == ["BTC"]


def test_resolve_coin_sources_merges_overrides():
    base = {"BTC": "binance"}
    overrides = {"ETH": "bybit"}
    resolved = resolve_coin_sources(base, overrides)
    assert resolved == {"BTC": "binance", "ETH": "bybit"}


def test_aggregate_metrics_computes_stats():
    scenario_results = [
        ScenarioResult(
            scenario=SuiteScenario("a", None, None, None, None),
            per_exchange={},
            combined_metrics={"metric": 1.0},
            elapsed_seconds=0.0,
            output_path=None,
        ),
        ScenarioResult(
            scenario=SuiteScenario("b", None, None, None, None),
            per_exchange={},
            combined_metrics={"metric": 3.0},
            elapsed_seconds=0.0,
            output_path=None,
        ),
    ]
    summary = aggregate_metrics(scenario_results, {"default": "mean"})
    assert summary["aggregated"]["metric"] == pytest.approx(2.0)
    assert summary["stats"]["metric"]["max"] == pytest.approx(3.0)
