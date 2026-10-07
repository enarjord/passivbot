from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from config_utils import get_template_config
import suite_runner as suite


@pytest.fixture
def prepared():
    config = get_template_config()
    config["backtest"].update(
        start_date="2026-01-01", end_date="2026-01-02", coins={}, cache_dir={}
    )
    config["live"].update(warmup_ratio=0, max_warmup_minutes=0)
    coins = ["BTC", "XMR"]
    dataset = suite.ExchangeDataset(
        exchange="combined",
        coins=coins,
        coin_index={coin: i for i, coin in enumerate(coins)},
        coin_exchange={coin: "binance" for coin in coins},
        available_exchanges=["binance", "bybit"],
        hlcvs=np.ones((3, 2, 4)),
        btc_usd_prices=np.ones(3),
        timestamps=1767225600000 + np.arange(3, dtype=np.int64) * 60_000,
        mss={coin: {"qty_step": 1, "exchange": "binance"} for coin in coins},
        cache_dir="unused",
    )
    scenario = suite.SuiteScenario(
        "base", None, None, None, None, exchanges=["binance", "bybit"]
    )
    return config, dataset, scenario


def _pipeline(paths, *, fail=False):
    def build(*args):
        return SimpleNamespace()

    def execute(*args):
        return [], [], {"value": 1}

    def post(*args, **kwargs):
        if fail:
            raise OSError("output unavailable")
        path = kwargs["output_directory"]
        paths.append(path)
        (path / "config.json").write_text("{}")

    return build, execute, post, lambda payload: None


def test_combined_scenario_writes_directly_to_scenario_directory(prepared, tmp_path):
    config, dataset, scenario = prepared
    paths = []
    suite._run_combined_dataset(
        dataset, scenario, config, ["XMR"], tmp_path, *_pipeline(paths)
    )
    assert paths == [tmp_path]
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


@pytest.mark.parametrize("count", [1, 2])
def test_exchange_subdirectories_exist_only_for_multiple_results(
    prepared, tmp_path, count
):
    config, dataset, scenario = prepared
    datasets = {}
    for exchange in ["binance", "bybit"][:count]:
        item = deepcopy(dataset)
        item.exchange = exchange
        datasets[exchange] = item
    paths = []
    suite._run_multi_dataset(
        datasets,
        scenario,
        config,
        ["XMR"],
        tmp_path,
        *_pipeline(paths),
        ["binance", "bybit"],
    )
    assert paths == (
        [tmp_path] if count == 1 else [tmp_path / "binance", tmp_path / "bybit"]
    )
    assert all((path / "config.json").is_file() for path in paths)


def test_output_failure_does_not_publish_successful_scenario(prepared, tmp_path):
    config, dataset, scenario = prepared
    with pytest.raises(OSError, match="output unavailable"):
        suite._run_combined_dataset(
            dataset, scenario, config, ["XMR"], tmp_path, *_pipeline([], fail=True)
        )


def test_suite_setup_uses_union_of_actual_scenario_coins(prepared, monkeypatch):
    config, dataset, _ = prepared
    scenarios = [
        suite.SuiteScenario(coin, None, None, [coin], None) for coin in ["BTC", "XMR"]
    ]

    def apply(base, scenario, **kwargs):
        return deepcopy(config), scenario.coins

    monkeypatch.setattr(suite, "apply_scenario", apply)
    inputs, configs, coins, sources = suite._suite_session_inputs(
        scenarios,
        config,
        {"combined": dataset},
        ["BTC", "XMR"],
        [],
        ["binance", "bybit"],
        ["BTC", "XMR"],
        {},
        ["BTC", "XMR"],
        [],
    )
    assert coins == ["BTC", "XMR"]
    assert sources == ["combined"]
    assert [item["data"]["combined"]["coins"] for item in inputs] == [["BTC"], ["XMR"]]


def test_scenario_labels_cannot_collide_on_case_insensitive_filesystems():
    with pytest.raises(ValueError, match="filesystem"):
        suite.build_scenarios({"scenarios": [{"label": "Base"}, {"label": "base"}]})
