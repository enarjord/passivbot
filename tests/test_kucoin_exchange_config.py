import asyncio
import logging
import math
import socket
import types

import pytest
import ccxt.pro as ccxt_pro

from exchanges.kucoin import (
    AsyncKucoinBrokerFutures,
    KucoinBot,
    ProKucoinBrokerFutures,
)
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
        return {"code": "200000", "hedged": hedged, "apiKey": "SECRET"}

    async def set_margin_mode(self, **params):
        self.margin_calls.append(params)
        return {"symbol": params["symbol"], "marginMode": params["marginMode"]}

    async def set_leverage(self, **params):
        self.leverage_calls.append(params)
        return {"symbol": params["symbol"], "leverage": params["leverage"]}


def test_kucoin_futures_expired_ws_token_invalidates_exact_negotiated_url(monkeypatch):
    client = ProKucoinBrokerFutures({})
    private_url = object()
    private_futures_url = object()
    client.options["urls"] = {
        "private": private_url,
        "privateFutures": private_futures_url,
    }
    delegated = []

    def handle_error_message(_self, ws_client, message):
        delegated.append((ws_client, message))
        return False

    monkeypatch.setattr(
        ccxt_pro.kucoinfutures,
        "handle_error_message",
        handle_error_message,
    )
    ws_client = types.SimpleNamespace(
        url="wss://push-private.kucoin.example/endpoint?token=redacted&connectId=privateFutures"
    )
    message = {"type": "error", "data": "token is expired"}

    assert client.handle_error_message(ws_client, message) is False

    assert client.options["urls"]["privateFutures"] is None
    assert client.options["urls"]["private"] is private_url
    assert delegated == [(ws_client, message)]


def test_kucoin_ws_non_expiry_error_preserves_negotiated_url(monkeypatch):
    client = ProKucoinBrokerFutures({})
    private_futures_url = object()
    client.options["urls"] = {"privateFutures": private_futures_url}

    monkeypatch.setattr(
        ccxt_pro.kucoinfutures,
        "handle_error_message",
        lambda _self, _ws_client, _message: False,
    )
    ws_client = types.SimpleNamespace(
        url="wss://push-private.kucoin.example/endpoint?token=redacted&connectId=privateFutures"
    )

    assert (
        client.handle_error_message(
            ws_client,
            {"type": "error", "data": "type is not supported"},
        )
        is False
    )

    assert client.options["urls"]["privateFutures"] is private_futures_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class", [AsyncKucoinBrokerFutures, ProKucoinBrokerFutures]
)
async def test_kucoin_clients_use_ipv4_transport(client_class):
    exchange = client_class({})
    try:
        exchange.open()
        assert exchange.tcp_connector._family == socket.AF_INET
    finally:
        await exchange.close()


def make_bot():
    bot = KucoinBot.__new__(KucoinBot)
    bot.cca = DummyCCA()
    bot.hedge_mode = True
    bot.max_leverage = {}
    bot.positions = {
        "BTC/USDT:USDT": {
            "long": {"size": 0.0},
            "short": {"size": 0.0},
        }
    }
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


@pytest.mark.parametrize("timeout", [None, 65_000])
@pytest.mark.parametrize("native_keys", [False, True])
def test_kucoin_session_setup_preserves_common_timeout_and_broker_headers(timeout, native_keys):
    bot = KucoinBot.__new__(KucoinBot)
    bot.exchange = "kucoin"
    bot.user_info = (
        {"apiKey": "api_key", "secret": "api_secret", "password": "api_passphrase"}
        if native_keys else
        {"key": "api_key", "secret": "api_secret", "passphrase": "api_passphrase"}
    )
    if timeout is not None:
        bot.user_info["timeout"] = timeout
    bot.broker_code = {"futures": {
        "partner": "passivbotFutures", "broker-key": "broker_secret",
        "broker-name": "passivbotFutures",
    }}
    bot.ws_enabled = True
    bot._build_ccxt_options = lambda: {}
    bot._apply_endpoint_override = lambda client: None

    bot.create_ccxt_sessions()

    for client in (bot.cca, bot.ccp):
        assert client.timeout == (30_000 if timeout is None else timeout)
        assert client.enableRateLimit is True
        assert (client.apiKey, client.secret, client.password) == (
            "api_key", "api_secret", "api_passphrase"
        )
        assert client.options["defaultType"] == "swap"
        headers = client.sign("orders", "futuresPrivate", "GET", {})["headers"]
        assert headers["KC-API-PARTNER"] == "passivbotFutures"
        assert headers["KC-API-PARTNER-VERIFY"] == "true"
        assert headers["KC-API-PARTNER-SIGN"]
        assert headers["KC-BROKER-NAME"] == "passivbotFutures"


def test_kucoin_ticker_normalizer_rejects_missing_book(caplog):
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

    assert "BTC/USDT:USDT" not in out
    assert out["ETH/USDT:USDT"]["source"] == "kucoin_futures_all_tickers"
    assert "DOGE/USDT:USDT" not in out
    assert "using last as bid=ask" not in caplog.text


@pytest.mark.asyncio
async def test_kucoin_native_book_is_valid_market_snapshot():
    bot = KucoinBot.__new__(KucoinBot)
    bot.markets_dict = {"BTC/USDT:USDT": {}}

    async def fetch_tickers():
        return bot._normalize_tickers(
            {
                "BTC/USDT:USDT": {
                    "bid": 76254.0,
                    "ask": 76256.0,
                    "last": 76255.0,
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
    assert snap.bid == pytest.approx(76254.0)
    assert snap.ask == pytest.approx(76256.0)
    assert snap.last == pytest.approx(76255.0)
    assert snap.source == "kucoin_futures_all_tickers"


@pytest.mark.asyncio
async def test_update_exchange_config_sets_position_mode_when_supported(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    await bot.update_exchange_config()
    assert bot.cca.position_mode_calls == [True]
    assert "set_position_mode hedged=True result=ok" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_update_exchange_config_handles_missing_position_mode(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    bot.cca = types.SimpleNamespace()
    with pytest.raises(NotImplementedError, match="set_position_mode not supported"):
        await bot.update_exchange_config()
    assert "set_position_mode hedged=True not applied" in caplog.text


def test_determine_pos_side_prefers_explicit_payload_over_existing_long_position():
    bot = make_bot()
    bot.positions["BTC/USDT:USDT"]["long"]["size"] = 1.0

    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "short",
        "info": {},
    }

    assert bot.determine_pos_side(order) == "short"


def test_determine_pos_side_prefers_info_position_side_when_both_psides_open():
    bot = make_bot()
    bot.positions["BTC/USDT:USDT"]["long"]["size"] = 1.0
    bot.positions["BTC/USDT:USDT"]["short"]["size"] = -1.0

    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "info": {"positionSide": "SHORT"},
    }

    assert bot.determine_pos_side(order) == "short"


def test_determine_pos_side_rejects_hedge_order_without_durable_payload():
    bot = make_bot()
    bot.positions["BTC/USDT:USDT"]["long"]["size"] = 1.0
    bot.positions["BTC/USDT:USDT"]["short"]["size"] = -1.0

    with pytest.raises(ValueError, match="missing durable long/short attribution"):
        bot.determine_pos_side({"symbol": "BTC/USDT:USDT", "side": "buy", "info": {}})


def test_determine_pos_side_uses_exchange_hedge_mode_when_strategy_hedge_is_disabled():
    bot = make_bot()
    bot._config_hedge_mode = False

    assert (
        bot.determine_pos_side(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "info": {"positionSide": "LONG"},
            }
        )
        == "long"
    )


def test_determine_pos_side_one_way_uses_side_and_reduce_only_not_position_state():
    bot = make_bot()
    bot._config_hedge_mode = False
    bot.hedge_mode = False
    bot.positions["BTC/USDT:USDT"]["long"]["size"] = 1.0

    assert (
        bot.determine_pos_side(
            {
                "symbol": "BTC/USDT:USDT",
                "side": "sell",
                "reduceOnly": False,
                "info": {"reduceOnly": False},
            }
        )
        == "short"
    )


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
        if path == ["live", "account", "leverage"]:
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


@pytest.mark.asyncio
async def test_update_exchange_config_by_symbols_bounds_failure_logs(caplog):
    class FailingCCA:
        async def set_margin_mode(self, **_params):
            raise RuntimeError("SECRET_MARGIN")

        async def set_leverage(self, **_params):
            raise ValueError("SECRET_LEVERAGE")

    bot = make_bot()
    bot.cca = FailingCCA()
    bot.max_leverage = {"BTC/USDT:USDT": 10}
    bot.config_get = lambda path, *, symbol=None: 5

    with caplog.at_level(logging.WARNING):
        await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert "error_type=RuntimeError" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_max_leverage", ["bad", 0, -1, math.inf, math.nan])
async def test_update_exchange_config_by_symbols_rejects_invalid_max_leverage(
    raw_max_leverage,
    monkeypatch,
):
    bot = make_bot()
    bot.max_leverage = {"BTC/USDT:USDT": raw_max_leverage}
    bot.config_get = lambda path, *, symbol=None: 5

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    with pytest.raises(ValueError, match="invalid max leverage from exchange metadata"):
        await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert bot.cca.leverage_calls == []


@pytest.mark.parametrize("field,value", [
    ("bid", None), ("ask", None), ("last", None), ("last", float("nan")),
    ("bid", float("inf")), ("ask", 0.0), ("last", True), ("bid", 12.0),
])
def test_kucoin_ticker_rejects_missing_nonfinite_or_crossed_quotes(field, value):
    bot = KucoinBot.__new__(KucoinBot)
    bot.markets_dict = {"BTC/USDT:USDT": {}}
    row = {"bid": 9.0, "ask": 11.0, "last": 10.0}
    row[field] = value
    assert bot._normalize_tickers({"BTC/USDT:USDT": row}) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("targeted", [True, False])
async def test_kucoin_fetches_one_authoritative_bulk_book_request(monkeypatch, targeted):
    client = AsyncKucoinBrokerFutures({"options": {"defaultType": "swap"}})
    client.set_markets([{
        "id": market_id, "symbol": symbol, "base": base, "quote": "USDT",
        "settle": "USDT", "type": "swap", "spot": False, "swap": True,
        "contract": True, "linear": True, "inverse": False,
    } for market_id, symbol, base in (("XBTUSDTM", "BTC/USDT:USDT", "BTC"),
                                    ("ETHUSDTM", "ETH/USDT:USDT", "ETH"))])
    requests = []

    async def request(path, api="public", method="GET", params=None, **kwargs):
        requests.append((path, api, method, params))
        return {"code": "200000", "data": [
            {"symbol": market_id, "price": "100", "bestBidPrice": "99",
             "bestAskPrice": "101", "ts": 1800000000000000000}
            for market_id in ("XBTUSDTM", "ETHUSDTM")
        ]}

    monkeypatch.setattr(client, "request", request)
    bot = KucoinBot.__new__(KucoinBot)
    bot.cca = client
    bot.markets_dict = client.markets
    try:
        result = (await bot.fetch_tickers_for_symbols(["BTC/USDT:USDT"])
                  if targeted else await bot.fetch_tickers())
        assert requests == [("allTickers", "futuresPublic", "GET", {})]
        assert set(result) == ({"BTC/USDT:USDT"} if targeted
                               else {"BTC/USDT:USDT", "ETH/USDT:USDT"})
        assert all(row["bid"] == 99.0 and row["ask"] == 101.0
                   and row["last"] == 100.0 for row in result.values())
        assert all(row["source"] == "kucoin_futures_all_tickers" for row in result.values())
    finally:
        await client.close()


@pytest.mark.parametrize("field", ["markPrice", "indexPrice"])
def test_kucoin_reference_price_does_not_replace_last_trade(field):
    bot = KucoinBot.__new__(KucoinBot)
    bot.markets_dict = {"BTC/USDT:USDT": {}}
    row = {"bid": 9.0, "ask": 11.0, field: 10.0, "info": {field: "10"}}
    assert bot._normalize_tickers({"BTC/USDT:USDT": row}) == {}
