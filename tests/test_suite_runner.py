import asyncio
from copy import deepcopy

import pytest

from suite_runner import (
    SuiteScenario,
    ScenarioResult,
    aggregate_metrics,
    apply_scenario,
    build_scenarios,
    collect_suite_coin_sources,
    extract_suite_config,
    filter_coins_by_exchange_assignment,
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
        base_coin_sources={"BTC": "binance"},
    )
    assert coins == ["BTC"]
    assert cfg["live"]["approved_coins"]["long"] == ["BTC"]
    assert cfg["backtest"]["coin_sources"] == {"BTC": "binance"}


def test_resolve_coin_sources_merges_overrides():
    base = {"BTC": "binance"}
    overrides = {"ETH": "bybit"}
    resolved = resolve_coin_sources(base, overrides)
    assert resolved == {"BTC": "binance", "ETH": "bybit"}


def test_collect_suite_coin_sources_detects_conflicts():
    config = {"backtest": {"coin_sources": {"BTC": "binance"}}}
    scenarios = [
        SuiteScenario("one", None, None, None, None, coin_sources={"ETH": "bybit"}),
        SuiteScenario("two", None, None, None, None, coin_sources={"ETH": "bybit"}),
    ]
    merged = collect_suite_coin_sources(config, scenarios)
    assert merged == {"BTC": "binance", "ETH": "bybit"}

    conflicting = [
        SuiteScenario("one", None, None, None, None, coin_sources={"ADA": "binance"}),
        SuiteScenario("two", None, None, None, None, coin_sources={"ADA": "bybit"}),
    ]
    with pytest.raises(ValueError):
        collect_suite_coin_sources(config, conflicting)


def test_filter_coins_by_exchange_assignment_filters_correctly():
    coins = ["BTC", "ETH", "ADA"]
    allowed = ["binanceusdm"]
    coin_map = {"BTC": "binanceusdm", "ETH": "bybit", "ADA": "binanceusdm"}
    selected, skipped = filter_coins_by_exchange_assignment(
        coins, allowed, coin_map, default_exchange="combined"
    )
    assert selected == ["BTC", "ADA"]
    assert skipped == ["ETH"]


def test_aggregate_metrics_computes_stats():
    scenario_results = [
        ScenarioResult(
            scenario=SuiteScenario("a", None, None, None, None),
            per_exchange={},
            metrics={"stats": {"metric": {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}}},
            elapsed_seconds=0.0,
            output_path=None,
        ),
        ScenarioResult(
            scenario=SuiteScenario("b", None, None, None, None),
            per_exchange={},
            metrics={"stats": {"metric": {"mean": 3.0, "min": 3.0, "max": 3.0, "std": 0.0}}},
            elapsed_seconds=0.0,
            output_path=None,
        ),
    ]
    summary = aggregate_metrics(scenario_results, {"default": "mean"})
    assert summary["aggregated"]["metric"] == pytest.approx(2.0)
    assert summary["stats"]["metric"]["max"] == pytest.approx(3.0)
