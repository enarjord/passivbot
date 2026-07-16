from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from ccxt_contracts import build_contract_bot, get_bot_class
from exchanges.weex import AsyncWeex, WeexBot
from fill_events_manager import WeexFetcher, _build_fetcher_for_bot


SYMBOL = "BTC/USDT:USDT"


def _market() -> dict:
    return {
        "id": "BTCUSDT",
        "symbol": SYMBOL,
        "base": "BTC",
        "quote": "USDT",
        "settle": "USDT",
        "type": "swap",
        "spot": False,
        "swap": True,
        "future": False,
        "contract": True,
        "linear": True,
        "inverse": False,
        "active": True,
        "precision": {"amount": 0.0001, "price": 0.1},
        "limits": {
            "amount": {"min": 0.0001, "max": 1200.0},
            "price": {"min": None, "max": None},
            "cost": {"min": None, "max": None},
            "leverage": {"min": 1.0, "max": 400.0},
        },
        "contractSize": 0.0001,
        "info": {"maxLeverage": "400", "contractVal": "0.0001"},
    }


def _ccxt_exchange() -> ccxt.weex:
    exchange = ccxt.weex(
        {"apiKey": "key", "secret": "secret", "password": "passphrase"}
    )
    market = _market()
    exchange.markets = {SYMBOL: market}
    exchange.markets_by_id = {"BTCUSDT": [market]}
    exchange.symbols = [SYMBOL]
    return exchange


def _bot(*, time_in_force: str = "gtc") -> WeexBot:
    bot = build_contract_bot("weex")
    bot.config["live"]["time_in_force"] = time_in_force
    return bot


def test_ccxt_contract_registry_uses_weex_adapter():
    assert get_bot_class("weex") is WeexBot


def test_weex_client_accepts_only_documented_success_envelope():
    exchange = AsyncWeex()
    assert (
        exchange.handle_errors(
            200,
            "OK",
            "https://api-contract.weex.com/capi/v3/account/leverage",
            "POST",
            {},
            '{"code":"200","msg":"success"}',
            {"code": "200", "msg": "success"},
            {},
            "{}",
        )
        is None
    )

    with pytest.raises(ccxt.ExchangeError):
        exchange.handle_errors(
            400,
            "Bad Request",
            "https://api-contract.weex.com/capi/v3/account/leverage",
            "POST",
            {},
            '{"code":"-3313","msg":"bad leverage"}',
            {"code": "-3313", "msg": "bad leverage"},
            {},
            "{}",
        )


def test_weex_order_params_match_v3_hedge_contract():
    bot = _bot(time_in_force="post_only")
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": "long",
            "reduce_only": True,
            "custom_id": "close_grid_long_01",
        }
    )

    assert params == {
        "positionSide": "LONG",
        "clientOrderId": "close_grid_long_01",
        "timeInForce": "POST_ONLY",
    }
    assert "reduceOnly" not in params
    assert "postOnly" not in params


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_side", ""),
        ("position_side", "both"),
        ("custom_id", ""),
        ("custom_id", "x" * 37),
        ("custom_id", "invalid id"),
    ],
)
def test_weex_order_params_reject_ambiguous_side_or_invalid_client_id(field, value):
    bot = _bot()
    order = {
        "symbol": SYMBOL,
        "type": "limit",
        "position_side": "long",
        "reduce_only": False,
        "custom_id": "entry_grid_long_01",
    }
    order[field] = value

    with pytest.raises(ValueError):
        bot._build_order_params(order)


@pytest.mark.parametrize(
    "side,position_side,reduce_only",
    [
        ("buy", "long", False),
        ("sell", "short", False),
        ("sell", "long", True),
        ("buy", "short", True),
    ],
)
def test_weex_order_params_survive_ccxt_request_construction(
    side: str, position_side: str, reduce_only: bool
):
    bot = _bot(time_in_force="post_only")
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": position_side,
            "reduce_only": reduce_only,
            "custom_id": f"pb_{position_side}_{int(reduce_only)}",
        }
    )
    exchange = _ccxt_exchange()

    request = exchange.create_contract_order_request(
        SYMBOL, "limit", side, 0.0001, 60_000.0, params
    )

    assert request["symbol"] == "BTCUSDT"
    assert request["side"] == side.upper()
    assert request["positionSide"] == position_side.upper()
    assert "positionId" not in request
    assert request["timeInForce"] == "POST_ONLY"
    assert request["newClientOrderId"] == f"pb_{position_side}_{int(reduce_only)}"
    assert "reduceOnly" not in request
    assert "postOnly" not in request


def test_weex_signed_order_contains_required_auth_and_no_reduce_only():
    bot = _bot(time_in_force="gtc")
    exchange = _ccxt_exchange()
    params = bot._build_order_params(
        {
            "symbol": SYMBOL,
            "type": "limit",
            "position_side": "short",
            "reduce_only": True,
            "custom_id": "close_grid_short_01",
        }
    )
    request = exchange.create_contract_order_request(
        SYMBOL, "limit", "buy", 0.0001, 60_000.0, params
    )

    signed = exchange.sign(
        "capi/v3/order", "contractPrivate", "POST", request
    )
    body = json.loads(signed["body"])

    assert body["positionSide"] == "SHORT"
    assert body["timeInForce"] == "GTC"
    assert "reduceOnly" not in body
    assert signed["headers"]["ACCESS-KEY"] == "key"
    assert signed["headers"]["ACCESS-PASSPHRASE"] == "passphrase"
    assert signed["headers"]["ACCESS-SIGN"]


def test_weex_book_ticker_normalization_is_fresh_and_strict():
    bot = _bot()
    bot.markets_dict = {SYMBOL: _market()}
    bot.symbol_ids_inv = {"BTCUSDT": SYMBOL}

    tickers = bot._normalize_tickers(
        [
            {
                "symbol": "BTCUSDT",
                "bidPrice": "60000.1",
                "askPrice": "60000.2",
                "time": 1234,
            },
            {
                "symbol": "BADUSDT",
                "bidPrice": "1",
                "askPrice": "2",
                "time": 1234,
            },
        ]
    )

    assert tickers == {
        SYMBOL: {
            "bid": 60000.1,
            "ask": 60000.2,
            "last": pytest.approx(60000.15),
            "timestamp": 1234,
        }
    }
    assert bot._normalize_tickers(
        [{"symbol": "BTCUSDT", "bidPrice": "2", "askPrice": "1"}]
    ) == {}
    assert bot._normalize_tickers(
        [{"symbol": "BTCUSDT", "bidPrice": "1", "askPrice": "2", "time": 0}]
    ) == {}


def test_weex_market_sizing_uses_base_quantity_not_contract_val():
    bot = _bot()
    bot.markets_dict = {SYMBOL: _market()}
    bot.eligible_symbols = {SYMBOL}

    bot.set_market_specific_settings()

    assert bot.min_qtys[SYMBOL] == pytest.approx(0.0001)
    assert bot.qty_steps[SYMBOL] == pytest.approx(0.0001)
    assert bot.c_mults[SYMBOL] == 1.0
    assert bot.max_leverage[SYMBOL] == 400


@pytest.mark.asyncio
async def test_weex_symbol_config_sets_combined_margin_and_leverage():
    class _Api:
        def __init__(self):
            self.calls = []

        async def fetch_position_mode(self, symbol):
            self.calls.append(("fetch_position_mode", symbol))
            return {"hedged": True}

        async def fetch_margin_mode(self, symbol):
            self.calls.append(("fetch_margin_mode", symbol))
            return {"marginMode": "isolated"}

        async def set_position_mode(self, hedged, symbol=None, params=None):
            self.calls.append(("set_position_mode", hedged, symbol, params))
            return {"code": "200", "msg": "success"}

        async def set_leverage(self, leverage, symbol=None, params=None):
            self.calls.append(("set_leverage", leverage, symbol, params))
            return {"code": "200", "msg": "success"}

    bot = _bot()
    bot.cca = _Api()
    bot.max_leverage = {SYMBOL: 400}
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 10

    await bot.update_exchange_config_by_symbols([SYMBOL])

    assert ("set_position_mode", False, SYMBOL, {"marginMode": "cross"}) in bot.cca.calls
    assert ("set_leverage", 10, SYMBOL, {"marginMode": "cross"}) in bot.cca.calls


@pytest.mark.asyncio
async def test_weex_symbol_config_skips_unchanged_mode_but_sets_leverage():
    class _Api:
        def __init__(self):
            self.calls = []

        async def fetch_position_mode(self, symbol):
            return {"hedged": False}

        async def fetch_margin_mode(self, symbol):
            return {"marginMode": "cross"}

        async def set_position_mode(self, *args, **kwargs):
            raise AssertionError("unchanged WEEX mode must not be rewritten")

        async def set_leverage(self, leverage, symbol=None, params=None):
            self.calls.append((leverage, symbol, params))
            return {"code": "200", "msg": "success"}

    bot = _bot()
    bot.cca = _Api()
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 7

    await bot.update_exchange_config_by_symbols([SYMBOL])

    assert bot.cca.calls == [(7, SYMBOL, {"marginMode": "cross"})]


class _FakeWeexApi:
    def __init__(self, batches):
        self.batches = list(batches)
        self.trade_calls = []
        self.order_calls = []

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        self.trade_calls.append(
            {"symbol": symbol, "since": since, "limit": limit, "params": dict(params or {})}
        )
        return self.batches.pop(0) if self.batches else []

    async def fetch_order(self, order_id, symbol):
        self.order_calls.append((order_id, symbol))
        return {"clientOrderId": f"entry_grid_long_{order_id}", "info": {}}


def _trade(trade_id: str, order_id: str, timestamp: int, *, position_side="LONG"):
    return {
        "id": trade_id,
        "order": order_id,
        "timestamp": timestamp,
        "symbol": SYMBOL,
        "side": "buy",
        "amount": 0.0001,
        "price": 60_000.0,
        "fee": {"currency": "USDT", "cost": 0.0012},
        "info": {
            "id": trade_id,
            "orderId": order_id,
            "time": timestamp,
            "side": "BUY",
            "positionSide": position_side,
            "realizedPnl": "1.25",
        },
    }


@pytest.mark.asyncio
async def test_weex_fetcher_windows_paginates_normalizes_and_enriches():
    day = 24 * 60 * 60 * 1000
    api = _FakeWeexApi(
        [
            [_trade("t1", "o1", day), _trade("t2", "o2", day + 1)],
            [_trade("t3", "o3", day + 2)],
            [_trade("t4", "o4", 8 * day)],
        ]
    )
    fetcher = WeexFetcher(api, trade_limit=2, now_func=lambda: 9 * day)
    detail_cache = {}

    events = await fetcher.fetch(day, 9 * day, detail_cache)

    assert [event["id"] for event in events] == ["t1", "t2", "t3", "t4"]
    assert all(event["position_side"] == "long" for event in events)
    assert all(event["pnl"] == pytest.approx(1.25) for event in events)
    assert all(event["c_mult"] == 1.0 for event in events)
    assert all(event["client_order_id"].startswith("entry_grid_long_") for event in events)
    assert len(api.order_calls) == 4
    assert all(call["params"]["type"] == "swap" for call in api.trade_calls)
    assert all(
        call["params"]["until"] - call["since"] < WeexFetcher.WINDOW_MS
        for call in api.trade_calls
    )
    assert set(detail_cache) == {"t1", "t2", "t3", "t4"}


def test_weex_fetcher_rejects_ambiguous_position_side():
    with pytest.raises(ValueError, match="positionSide"):
        WeexFetcher._normalize_trade(
            _trade("t1", "o1", 1, position_side="")
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("amount", 0.0, "non-positive"),
        ("price", 0.0, "non-positive"),
        ("realizedPnl", None, "realizedPnl"),
    ],
)
def test_weex_fetcher_rejects_invalid_accounting_values(field, value, match):
    trade = _trade("t1", "o1", 1)
    if field == "realizedPnl":
        trade["info"].pop(field)
    else:
        trade[field] = value
        trade["info"]["qty" if field == "amount" else field] = value

    with pytest.raises(ValueError, match=match):
        WeexFetcher._normalize_trade(trade)


def test_build_fetcher_for_weex():
    bot = SimpleNamespace(
        exchange="weex",
        cca="api",
        user="weex_user",
        markets_dict={},
        coin_to_symbol=lambda value, verbose=False: value,
    )

    assert isinstance(_build_fetcher_for_bot(bot, symbols=["BTC"]), WeexFetcher)


def test_setup_bot_weex_uses_weex_adapter():
    from passivbot import setup_bot

    config = {"live": {"user": "weex_01"}}
    with patch("passivbot.load_user_info", return_value={"exchange": "weex"}):
        with patch("exchanges.weex.WeexBot") as mock_cls:
            mock_bot = MagicMock()
            mock_cls.return_value = mock_bot
            assert setup_bot(config) is mock_bot
            mock_cls.assert_called_once_with(config)
