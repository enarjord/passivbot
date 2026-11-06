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


@pytest.mark.asyncio
async def test_partial_combined_setup_fingerprints_executed_inputs(prepared, monkeypatch):
    import backtest
    from tools import iterative_backtester

    config, combined, scenario = prepared
    combined.timestamps += 12 * 60 * 60_000
    combined.coin_exchange = {coin: "bybit" for coin in combined.coins}
    for coin in combined.coins:
        combined.mss[coin]["exchange"] = "bybit"
    binance = deepcopy(combined)
    binance.exchange = "binance"
    binance.available_exchanges = ["binance"]
    binance.coin_exchange = {coin: "binance" for coin in binance.coins}
    binance.hlcvs *= 2
    for coin in binance.coins:
        binance.mss[coin]["exchange"] = "binance"
    datasets = {"combined": combined, "binance": binance}
    available = suite._compute_dataset_available_exchanges(datasets)

    monkeypatch.setattr(
        suite, "apply_scenario", lambda *args, **kwargs: (deepcopy(config), ["BTC"])
    )
    inputs, _, _, sources = suite._suite_session_inputs(
        [scenario], config, datasets, combined.coins, [], available,
        set(combined.coins), {}, combined.coins, [],
    )
    executed = {}

    def build(hlcvs, mss, cfg, exchange, btc_prices, timestamps):
        executed[exchange] = suite.materialized_dataset_identity(
            cfg["backtest"]["coins"][exchange], hlcvs, btc_prices, timestamps, mss
        )
        return SimpleNamespace()

    monkeypatch.setattr(backtest, "build_backtest_payload", build)
    monkeypatch.setattr(backtest, "execute_backtest", lambda *args: ([], [], {}))
    monkeypatch.setattr(iterative_backtester, "combine_analyses", lambda analyses: {})
    result = await suite.run_backtest_scenario(
        scenario, config, datasets, combined.coins, [], available,
        set(combined.coins), None, True,
    )

    assert sources == ["combined"]
    assert list(result.per_exchange) == ["combined"]
    assert executed == inputs[0]["data"]
    assert suite._compute_effective_coin_exchange(
        scenario, ["BTC"], datasets, available
    ) == {"BTC": "bybit"}


@pytest.mark.asyncio
async def test_scenario_only_exchange_preserves_combined_pool(prepared, monkeypatch, tmp_path):
    import backtest
    import utils
    from tools import iterative_backtester

    config, template, _ = prepared
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    config["live"]["approved_coins"] = {"long": ["ETH"], "short": ["ETH"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    scenario_configs = [
        {"label": "base_pool", "coins": ["ETH"]},
        {"label": "bitget_only", "coins": ["BTC"], "exchanges": ["bitget"],
         "coin_sources": {"BTC": "bitget"}},
    ]
    combined = deepcopy(template)
    combined.coins = ["ETH"]
    combined.coin_index = {"ETH": 0}
    combined.coin_exchange = {"ETH": "binance"}
    combined.hlcvs = np.ones((3, 1, 4))
    combined.mss = {"ETH": {"exchange": "binance", "qty_step": 1}}
    bitget = deepcopy(combined)
    bitget.exchange = "bitget"
    bitget.available_exchanges = ["bitget"]
    bitget.coins = ["BTC"]
    bitget.coin_index = {"BTC": 0}
    bitget.coin_exchange = {"BTC": "bitget"}
    bitget.hlcvs *= 2
    bitget.mss = {"BTC": {"exchange": "bitget", "qty_step": 1}}
    datasets = {"combined": combined, "bitget": bitget}
    monkeypatch.chdir(tmp_path)
    markets = {
        f"{coin}/USDT:USDT": {
            "id": f"{coin}USDT", "symbol": f"{coin}/USDT:USDT", "base": coin,
            "quote": "USDT", "settle": "USDT", "swap": True, "linear": True,
            "active": True,
        }
        for coin in ("BTC", "ETH")
    }
    for exchange in ("binance", "bybit", "bitget"):
        assert utils.create_coin_symbol_map_cache(exchange, markets, verbose=False)

    async def noop(*args, **kwargs):
        return None

    async def prepare(base_config, exchanges, **kwargs):
        assert base_config["backtest"]["exchanges"] == ["binance", "bybit"]
        assert base_config["backtest"]["coin_sources"] == {"BTC": "bitget"}
        assert set(exchanges) == {"binance", "bybit", "bitget"}
        assert kwargs["needed_individual_exchanges"] == {"bitget"}
        return datasets

    for name in ("load_markets", "format_approved_ignored_coins",
                 "reject_cross_exchange_market_identifier_collisions"):
        monkeypatch.setattr(suite, name, noop)
    monkeypatch.setattr(suite, "prepare_master_datasets", prepare)
    executed = []

    def build(hlcvs, mss, cfg, exchange, btc_prices, timestamps):
        executed.append((exchange, suite.materialized_dataset_identity(
            cfg["backtest"]["coins"][exchange], hlcvs, btc_prices, timestamps, mss
        )))
        return SimpleNamespace()

    def post(*args, **kwargs):
        (kwargs["output_directory"] / "config.json").write_text("{}")

    monkeypatch.setattr(backtest, "build_backtest_payload", build)
    monkeypatch.setattr(backtest, "execute_backtest", lambda *args: ([], [], {}))
    monkeypatch.setattr(backtest, "post_process", post)
    monkeypatch.setattr(backtest.BacktestPlotContext, "from_payload", lambda payload: None)
    monkeypatch.setattr(iterative_backtester, "combine_analyses", lambda analyses: {})
    summary = await suite.run_backtest_suite_async(
        config, {"scenarios": scenario_configs}, disable_plotting=True,
        suite_output_root=tmp_path / "output",
    )

    assert [result.scenario.label for result in summary.scenarios] == ["base_pool", "bitget_only"]
    assert [list(result.per_exchange) for result in summary.scenarios] == [["combined"], ["bitget"]]
    session = json.loads((summary.output_dir / "session.json").read_text())
    saved = session["setup"]["scenarios"]
    assert [item["data"] for item in saved] == [{exchange: identity} for exchange, identity in executed]
    available = suite._compute_dataset_available_exchanges(datasets)
    assignments = [
        suite._compute_effective_coin_exchange(result.scenario, result.scenario.coins, datasets, available)
        for result in summary.scenarios
    ]
    assert assignments == [{"ETH": "binance"}, {"BTC": "bitget"}]


def test_scenario_labels_cannot_collide_on_case_insensitive_filesystems():
    with pytest.raises(ValueError, match="filesystem"):
        suite.build_scenarios({"scenarios": [{"label": "Base"}, {"label": "base"}]})
