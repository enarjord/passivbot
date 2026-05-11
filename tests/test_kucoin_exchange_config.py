import asyncio
import logging
import types

import pytest

from exchanges.kucoin import AsyncKucoinBrokerFutures, KucoinBot
from market_snapshot import MarketSnapshotProvider


class DummyTask:
    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        return self._coro.__await__()


class DummyCCA:
    def __init__(self):
        self.position_mode_calls = []
        self.margin_calls = []
        self.leverage_calls = []

    async def set_position_mode(self, hedged):
        self.position_mode_calls.append(hedged)
        return {"code": "200000", "hedged": hedged}

    async def set_margin_mode(self, **params):
        self.margin_calls.append(params)
        return {"symbol": params["symbol"], "marginMode": params["marginMode"]}

    async def set_leverage(self, **params):
        self.leverage_calls.append(params)
        return {"symbol": params["symbol"], "leverage": params["leverage"]}


def make_bot():
    bot = KucoinBot.__new__(KucoinBot)
    bot.cca = DummyCCA()
    bot.hedge_mode = True
    bot.max_leverage = {}
    return bot


def test_broker_futures_sign_adds_partner_and_broker_name_headers():
    client = AsyncKucoinBrokerFutures(
        {
            "apiKey": "api_key",
            "secret": "api_secret",
            "password": "api_passphrase",
            "options": {
                "partner": {
                    "future": {
                        "id": "passivbotFutures",
                        "secret": "broker_secret",
                        "name": "passivbotFutures",
                    }
                }
            },
        }
    )

    signed = client.sign("orders", "futuresPrivate", "GET", {})
    headers = signed["headers"]

    assert headers["KC-API-PARTNER"] == "passivbotFutures"
    assert headers["KC-API-PARTNER-VERIFY"] == "true"
    assert headers["KC-API-PARTNER-SIGN"]
    assert headers["KC-BROKER-NAME"] == "passivbotFutures"


def test_broker_futures_sign_fails_loudly_without_broker_name():
    client = AsyncKucoinBrokerFutures(
        {
            "apiKey": "api_key",
            "secret": "api_secret",
            "password": "api_passphrase",
            "options": {
                "partner": {
                    "future": {
                        "id": "passivbotFutures",
                        "secret": "broker_secret",
                    }
                }
            },
        }
    )

    with pytest.raises(ValueError, match="broker-name"):
        client.sign("orders", "futuresPrivate", "GET", {})


def test_create_ccxt_sessions_requires_complete_futures_broker_config():
    bot = KucoinBot.__new__(KucoinBot)
    bot.user_info = {"key": "api_key", "secret": "api_secret", "passphrase": "api_passphrase"}
    bot.broker_code = {"futures": {"partner": "passivbotFutures", "broker-key": "broker_secret"}}

    with pytest.raises(ValueError, match="broker-name"):
        bot.create_ccxt_sessions()


def test_kucoin_ticker_normalizer_labels_last_price_fallback(caplog):
    caplog.set_level(logging.WARNING)
    bot = KucoinBot.__new__(KucoinBot)
    bot.markets_dict = {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}

    out = bot._normalize_tickers(
        {
            "BTC/USDT:USDT": {
                "bid": None,
                "ask": None,
                "last": 76256.3,
                "info": {"lastTradePrice": "76255.0", "markPrice": "76254.0"},
            },
            "ETH/USDT:USDT": {
                "bid": 3000.0,
                "ask": 3001.0,
                "last": 3000.5,
            },
            "DOGE/USDT:USDT": {"last": 0.1},
        }
    )

    assert out["BTC/USDT:USDT"]["bid"] == pytest.approx(76256.3)
    assert out["BTC/USDT:USDT"]["ask"] == pytest.approx(76256.3)
    assert out["BTC/USDT:USDT"]["last"] == pytest.approx(76256.3)
    assert out["BTC/USDT:USDT"]["source"] == "kucoin_last_fallback"
    assert out["ETH/USDT:USDT"]["source"] == "kucoin_ccxt_ticker"
    assert "DOGE/USDT:USDT" not in out
    assert "kucoin ticker bid/ask missing" in caplog.text


@pytest.mark.asyncio
async def test_kucoin_last_price_fallback_is_valid_market_snapshot():
    bot = KucoinBot.__new__(KucoinBot)
    bot.markets_dict = {"BTC/USDT:USDT": {}}

    async def fetch_tickers():
        return bot._normalize_tickers(
            {
                "BTC/USDT:USDT": {
                    "bid": None,
                    "ask": None,
                    "last": None,
                    "close": None,
                    "info": {"lastTradePrice": "76255.0"},
                }
            }
        )

    provider = MarketSnapshotProvider(
        exchange_name="kucoin",
        fetch_tickers=fetch_tickers,
        ticker_strategy="bulk",
    )

    snapshots = await provider.get_snapshots(["BTC/USDT:USDT"], max_age_ms=10_000)

    snap = snapshots["BTC/USDT:USDT"]
    assert snap.bid == pytest.approx(76255.0)
    assert snap.ask == pytest.approx(76255.0)
    assert snap.last == pytest.approx(76255.0)
    assert snap.source == "kucoin_last_fallback"


@pytest.mark.asyncio
async def test_update_exchange_config_sets_position_mode_when_supported(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    await bot.update_exchange_config()
    assert bot.cca.position_mode_calls == [True]
    assert "set_position_mode hedged=True" in caplog.text


@pytest.mark.asyncio
async def test_update_exchange_config_handles_missing_position_mode(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    bot.cca = types.SimpleNamespace()
    await bot.update_exchange_config()
    assert "set_position_mode not supported" in caplog.text


@pytest.mark.asyncio
async def test_update_exchange_config_by_symbols_sets_margin_and_leverage(monkeypatch):
    bot = make_bot()
    leverage_cfg = {
        "BTC/USDT:USDT": 5,
        "ETH/USDT:USDT": 3,
    }
    bot.max_leverage = {
        "BTC/USDT:USDT": 10,
        "ETH/USDT:USDT": 2,
    }

    def config_get(path, *, symbol=None):
        if path == ["live", "leverage"]:
            return leverage_cfg[symbol]
        raise KeyError(path)

    bot.config_get = config_get

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    symbols = list(leverage_cfg.keys())
    await bot.update_exchange_config_by_symbols(symbols)

    margin_symbols = [call["symbol"] for call in bot.cca.margin_calls]
    assert margin_symbols == symbols
    assert all(call["marginMode"] == "cross" for call in bot.cca.margin_calls)

    leverage_map = {call["symbol"]: call["leverage"] for call in bot.cca.leverage_calls}
    # leverage is clamped by max_leverage
    assert leverage_map["BTC/USDT:USDT"] == 5  # min(10, 5)
    assert leverage_map["ETH/USDT:USDT"] == 2  # min(2, 3)


@pytest.mark.asyncio
async def test_update_exchange_config_by_symbols_treats_missing_max_leverage_as_configured(
    monkeypatch,
):
    bot = make_bot()
    bot.max_leverage = {"BTC/USDT:USDT": None}
    bot.config_get = lambda path, *, symbol=None: 5

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert bot.cca.leverage_calls[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.cca.leverage_calls[0]["leverage"] == 5
