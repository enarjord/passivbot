import numpy as np
import pytest

import optimize_suite
from config.schema import get_template_config
from suite_runner import ExchangeDataset


class _NoSharedArrayManager:
    def create_from(self, _array):
        raise AssertionError("test dataset should use lazy master specs")


def _make_lazy_dataset(
    *,
    exchange="combined",
    coins=("HYPE",),
    coin_exchange=None,
    available_exchanges=None,
):
    timestamps = np.arange(1441, dtype=np.int64) * 60_000 + 1704067200000
    coins = list(coins)
    coin_exchange = coin_exchange or {coin: exchange for coin in coins}
    return ExchangeDataset(
        exchange=exchange,
        coins=coins,
        coin_index={coin: idx for idx, coin in enumerate(coins)},
        coin_exchange=coin_exchange,
        available_exchanges=available_exchanges or [exchange],
        hlcvs=np.ones((len(timestamps), len(coins), 4), dtype=np.float64),
        mss={
            **{
                coin: {
                    "exchange": coin_exchange.get(coin, exchange),
                    "first_valid_index": 0,
                    "last_valid_index": len(timestamps) - 1,
                }
                for coin in coins
            },
            "__meta__": {"data_interval_minutes": 1},
        },
        btc_usd_prices=np.ones(len(timestamps), dtype=np.float64),
        timestamps=timestamps,
        cache_dir="",
        hlcvs_spec=object(),
        btc_spec=object(),
    )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_keeps_directional_scenarios_with_default_short_disabled(
    monkeypatch,
):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance", "bybit"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "base"},
        {"label": "long_only", "overrides": {"bot.short.total_wallet_exposure_limit": 0}},
        {"label": "short_only", "overrides": {"bot.long.total_wallet_exposure_limit": 0}},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    # Schema defaults keep shorts disabled. Optimizer candidates may enable
    # shorts later, so context preparation must not dedupe base vs long_only.
    config["bot"]["short"]["total_wallet_exposure_limit"] = 0.0

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {
            "combined": _make_lazy_dataset(
                coins=("HYPE",),
                coin_exchange={"HYPE": "binance"},
                available_exchanges=["binance", "bybit"],
            )
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _aggregate_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert [ctx.label for ctx in contexts] == ["base", "long_only", "short_only"]


@pytest.mark.asyncio
async def test_prepare_suite_contexts_master_universe_keeps_base_and_scenario_coins(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "explicit", "coins": ["DOGE"]},
        {"label": "default"},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    captured = {}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    async def fake_prepare_master_datasets(base_config, exchanges, *_args, **_kwargs):
        captured["approved"] = list(base_config["live"]["approved_coins"]["long"])
        captured["exchanges"] = list(exchanges)
        return {
            "combined": _make_lazy_dataset(
                coins=("DOGE", "HYPE"),
                coin_exchange={"DOGE": "binance", "HYPE": "binance"},
                available_exchanges=["binance"],
            )
        }

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _aggregate_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert captured["approved"] == ["DOGE", "HYPE"]
    assert captured["exchanges"] == ["binance"]
    assert [ctx.label for ctx in contexts] == ["explicit", "default"]


@pytest.mark.asyncio
async def test_prepare_suite_contexts_expands_scenario_required_exchanges(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "bybit_only", "exchanges": ["bybit"], "coins": ["HYPE"]},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    loaded_exchanges = []
    captured = {}

    async def fake_load_markets(exchange, verbose=False):
        loaded_exchanges.append(exchange)
        return {}

    async def fake_format_approved_ignored_coins(_config, exchanges, verbose=False):
        captured["formatted_exchanges"] = list(exchanges)
        return None

    async def fake_prepare_master_datasets(_base_config, exchanges, *_args, **kwargs):
        captured["dataset_exchanges"] = list(exchanges)
        captured["needed"] = sorted(kwargs["needed_individual_exchanges"])
        return {"bybit": _make_lazy_dataset(exchange="bybit", coins=("HYPE",))}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    contexts, _aggregate_cfg = await optimize_suite.prepare_suite_contexts(
        config,
        suite_cfg,
        shared_array_manager=_NoSharedArrayManager(),
    )

    assert loaded_exchanges == ["binance", "bybit"]
    assert captured["formatted_exchanges"] == ["binance", "bybit"]
    assert captured["dataset_exchanges"] == ["binance", "bybit"]
    assert captured["needed"] == ["bybit"]
    assert contexts[0].exchanges == ["bybit"]


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_unavailable_scenario_exchange(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [
        {"label": "bybit_only", "exchanges": ["bybit"], "coins": ["HYPE"]},
    ]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {"binance": _make_lazy_dataset(exchange="binance", coins=("HYPE",))}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="requests unavailable exchange"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_scenario_with_no_usable_coins(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [{"label": "missing_coin", "coins": ["MISSING"]}]
    config["live"]["approved_coins"] = {"long": ["HYPE"], "short": ["HYPE"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    async def fake_prepare_master_datasets(*_args, **_kwargs):
        return {"combined": _make_lazy_dataset(coins=("HYPE",), available_exchanges=["binance"])}

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )
    monkeypatch.setattr(optimize_suite, "prepare_master_datasets", fake_prepare_master_datasets)

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="missing_coin could not be prepared"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )


@pytest.mark.asyncio
async def test_prepare_suite_contexts_rejects_asymmetric_side_coin_lists(monkeypatch):
    config = get_template_config()
    config["backtest"]["start_date"] = "2024-01-01"
    config["backtest"]["end_date"] = "2024-01-02"
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["suite_enabled"] = True
    config["backtest"]["scenarios"] = [{"label": "base"}]
    config["live"]["approved_coins"] = {"long": ["BTC"], "short": ["ETH"]}
    config["live"]["ignored_coins"] = {"long": [], "short": []}

    async def fake_load_markets(_exchange, verbose=False):
        return {}

    async def fake_format_approved_ignored_coins(_config, _exchanges, verbose=False):
        return None

    monkeypatch.setattr(optimize_suite, "load_markets", fake_load_markets)
    monkeypatch.setattr(
        optimize_suite,
        "format_approved_ignored_coins",
        fake_format_approved_ignored_coins,
    )

    suite_cfg = optimize_suite.extract_suite_config(config, suite_override=None)
    with pytest.raises(ValueError, match="asymmetric live.approved_coins"):
        await optimize_suite.prepare_suite_contexts(
            config,
            suite_cfg,
            shared_array_manager=_NoSharedArrayManager(),
        )
