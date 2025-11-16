import asyncio
from copy import deepcopy

import numpy as np
import pytest

from suite_runner import (
    SuiteScenario,
    ScenarioResult,
    ExchangeDataset,
    aggregate_metrics,
    apply_scenario,
    build_scenarios,
    collect_suite_coin_sources,
    extract_suite_config,
    filter_coins_by_exchange_assignment,
    resolve_coin_sources,
    _prepare_dataset_subset,
    summarize_scenario_metrics,
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
            "exchanges": ["binance"],
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


def test_summarize_scenario_metrics_prefers_mean():
    metrics = {
        "stats": {
            "adg_btc": {"mean": 0.1, "min": 0.05, "max": 0.2, "std": 0.01},
            "mdg_btc": 0.2,
            "drawdown": {"max": -0.1},
        }
    }
    simplified = summarize_scenario_metrics(metrics)
    assert simplified["adg_btc"] == 0.1
    assert simplified["mdg_btc"] == 0.2
    assert simplified["drawdown"] == -0.1


def test_apply_scenario_records_transform_log():
    base_config = {
        "backtest": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "coins": {},
            "cache_dir": {},
            "exchanges": ["binance"],
        },
        "live": {
            "approved_coins": {"long": [], "short": []},
            "ignored_coins": {"long": [], "short": []},
        },
    }
    scenario = SuiteScenario(
        label="scenario_a",
        start_date="2021-02-01",
        end_date=None,
        coins=["BTC"],
        ignored_coins=["ETH"],
    )
    cfg, _ = apply_scenario(
        base_config,
        scenario,
        master_coins=["BTC"],
        master_ignored=["ETH"],
        available_exchanges=["binance"],
        available_coins={"BTC", "ETH"},
        base_coin_sources={"BTC": "binance"},
    )
    entry = cfg["_transform_log"][-1]
    assert entry["step"] == "apply_scenario"
    assert entry["details"]["scenario"] == "scenario_a"
    assert any(change["path"] == "backtest.start_date" for change in entry["details"]["changes"])


def test_apply_scenario_overrides_update_config():
    base_config = {
        "backtest": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "coins": {},
            "cache_dir": {},
            "exchanges": ["binance"],
        },
        "live": {
            "approved_coins": {"long": [], "short": []},
            "ignored_coins": {"long": [], "short": []},
        },
        "bot": {"long": {"n_positions": 8}, "short": {"n_positions": 8}},
    }
    scenario = SuiteScenario(
        label="override",
        start_date=None,
        end_date=None,
        coins=["BTC"],
        ignored_coins=[],
        overrides={"bot.long.n_positions": 3},
    )
    cfg, _ = apply_scenario(
        base_config,
        scenario,
        master_coins=["BTC"],
        master_ignored=[],
        available_exchanges=["binance"],
        available_coins={"BTC"},
        base_coin_sources={"BTC": "binance"},
    )
    assert cfg["bot"]["long"]["n_positions"] == 3
    entry = cfg["_transform_log"][-1]
    paths = [change["path"] for change in entry["details"]["changes"]]
    assert "bot.long.n_positions" in paths


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


def test_prepare_dataset_subset_clips_dates(monkeypatch):
    coins = ["BTC", "ETH"]
    timestamps = np.array([0, 60_000, 120_000, 180_000], dtype=np.int64)
    hlcvs = np.random.random((len(timestamps), len(coins), 4))
    btc_prices = np.linspace(10000, 10030, len(timestamps))
    dataset = ExchangeDataset(
        exchange="combined",
        coins=coins,
        coin_index={coin: idx for idx, coin in enumerate(coins)},
        coin_exchange={coin: "combined" for coin in coins},
        available_exchanges=["combined"],
        hlcvs=hlcvs,
        mss={coin: {} for coin in coins},
        btc_usd_prices=btc_prices,
        timestamps=timestamps,
        cache_dir="/tmp",
    )
    scenario_config = {
        "backtest": {"start_date": "1970-01-01T00:01:00", "end_date": "1970-01-01T00:03:00"},
        "live": {"warmup_ratio": 0.0},
        "bot": {"long": {}, "short": {}},
        "optimize": {"bounds": {}},
    }
    monkeypatch.setattr(
        "suite_runner.compute_backtest_warmup_minutes",
        lambda cfg: 1,  # minute
    )
    monkeypatch.setattr(
        "suite_runner.compute_per_coin_warmup_minutes",
        lambda cfg: {"__default__": 0},
    )
    subset_hlcvs, subset_btc, subset_ts, subset_mss = _prepare_dataset_subset(
        dataset, scenario_config, ["BTC"], "scenario_clip"
    )
    # Expect 4 timesteps: warmup extends to t=0 and end is inclusive
    assert subset_hlcvs.shape[0] == 4
    assert subset_ts[0] == 0
    assert subset_ts[-1] == 180_000
    meta = subset_mss["__meta__"]
    assert meta["requested_start_ts"] == 60_000
    assert meta["requested_end_ts"] == 180_000
    assert "warmup_minutes_requested" in meta
