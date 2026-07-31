import asyncio
import hashlib
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from ccxt.base.errors import (
    AuthenticationError,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    OrderNotFound,
    RateLimitExceeded,
)

from ccxt_contracts import build_contract_bot, get_bot_class
from custom_endpoint_overrides import (
    CustomEndpointConfigError,
    ResolvedEndpointOverride,
)
from exchanges.bitunix import BitunixBot, BitunixClient, BitunixOrderStream
from fill_events_manager import (
    BitunixFetcher,
    _build_fetcher_for_bot,
    signed_fee_paid_from_payload,
)
from candlestick_manager import CandlestickManager


MARKET_ROW = {
    "symbol": "BTCUSDT",
    "base": "BTC",
    "quote": "USDT",
    "minTradeVolume": "0.0001",
    "basePrecision": 4,
    "quotePrecision": 1,
    "minLeverage": 1,
    "maxLeverage": 125,
    "defaultLeverage": 20,
    "defaultMarginMode": "CROSS",
    "symbolStatus": "OPEN",
    "isApiSupported": True,
}


def _market():
    return {
        "id": "BTCUSDT",
        "symbol": "BTC/USDT:USDT",
        "active": True,
        "maker": BitunixClient.DEFAULT_MAKER_FEE,
        "taker": BitunixClient.DEFAULT_TAKER_FEE,
        "info": MARKET_ROW,
    }


def _prepared_client() -> BitunixClient:
    client = BitunixClient({"apiKey": "key", "secret": "secret"})
    client.markets = {"BTC/USDT:USDT": _market()}
    client.markets_by_id = {"BTCUSDT": client.markets["BTC/USDT:USDT"]}
    client.symbols = ["BTC/USDT:USDT"]
    return client


def _order_row(**overrides):
    row = {
        "orderId": "order-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "tradeQty": "0",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "price": "50000",
        "side": "BUY",
        "orderType": "LIMIT",
        "effect": "POST_ONLY",
        "clientId": "clock_entry_long_1",
        "reduceOnly": False,
        "status": "NEW",
        "ctime": 1_700_000_000_000,
        "mtime": 1_700_000_000_001,
    }
    row.update(overrides)
    return row


def _trade_row(**overrides):
    row = {
        "tradeId": "trade-1",
        "orderId": "order-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "price": "50000",
        "side": "BUY",
        "orderType": "MARKET",
        "effect": "",
        "clientId": "clock_entry_long_1",
        "reduceOnly": False,
        "fee": "0.01",
        "realizedPNL": "0",
        "ctime": 1_700_000_000_000,
        "roleType": "TAKER",
    }
    row.update(overrides)
    return row


def _position_row(**overrides):
    row = {
        "positionId": "position-1",
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "entryValue": "50",
        "side": "LONG",
        "positionMode": "HEDGE",
        "marginMode": "CROSS",
        "leverage": 3,
        "unrealizedPNL": "0",
        "realizedPNL": "0",
        "avgOpenPrice": "50000",
        "ctime": 1_700_000_000_000,
        "mtime": 1_700_000_000_001,
    }
    row.update(overrides)
    return row


def test_ccxt_contract_registry_uses_bitunix_adapter():
    assert get_bot_class("bitunix") is BitunixBot


def test_candlestick_manager_uses_bitunix_kline_page_limit(tmp_path):
    client = _prepared_client()
    manager = CandlestickManager(
        exchange=client,
        exchange_name="bitunix",
        cache_dir=str(tmp_path / "caches"),
    )
    assert manager._ccxt_limit_default == 200


def test_rest_signature_matches_documented_double_sha256(monkeypatch):
    client = BitunixClient({"apiKey": "api", "secret": "secret"})
    monkeypatch.setattr("exchanges.bitunix.uuid.uuid4", lambda: SimpleNamespace(hex="nonce"))
    monkeypatch.setattr(client, "milliseconds", lambda: 1_700_000_000_000)

    headers = client._signed_headers({"symbol": "BTCUSDT", "limit": 10}, "")

    first = hashlib.sha256(
        b"nonce1700000000000apilimit10symbolBTCUSDT"
    ).hexdigest()
    expected = hashlib.sha256(f"{first}secret".encode()).hexdigest()
    assert headers == {
        "api-key": "api",
        "nonce": "nonce",
        "timestamp": "1700000000000",
        "sign": expected,
        "Content-Type": "application/json",
    }


def test_private_websocket_signature_uses_seconds(monkeypatch):
    client = BitunixClient({"apiKey": "api", "secret": "secret"})
    stream = BitunixOrderStream(client)
    monkeypatch.setattr("exchanges.bitunix.uuid.uuid4", lambda: SimpleNamespace(hex="nonce"))
    monkeypatch.setattr("exchanges.bitunix.time.time", lambda: 1_700_000_000.9)

    payload = stream._login_payload()

    first = hashlib.sha256(b"nonce1700000000api").hexdigest()
    expected = hashlib.sha256(f"{first}secret".encode()).hexdigest()
    assert payload["args"][0] == {
        "apiKey": "api",
        "timestamp": 1_700_000_000,
        "nonce": "nonce",
        "sign": expected,
    }


@pytest.mark.parametrize(
    "code,error_type",
    [
        (10004, AuthenticationError),
        (10001, NetworkError),
        (10006, RateLimitExceeded),
        (20003, InsufficientFunds),
        (30042, InvalidOrder),
    ],
)
def test_api_errors_map_to_ccxt_exception_contract(code, error_type):
    with pytest.raises(error_type):
        BitunixClient._raise_api_error(code, "test")


def test_observed_code_one_network_envelope_is_retryable():
    with pytest.raises(NetworkError, match="Network Error"):
        BitunixClient._raise_api_error(1, "Network Error")


@pytest.mark.asyncio
async def test_load_markets_maps_base_quantity_and_tick_sizes():
    client = BitunixClient()
    client._request = AsyncMock(return_value=[MARKET_ROW])

    markets = await client.load_markets(True)

    market = markets["BTC/USDT:USDT"]
    assert market["id"] == "BTCUSDT"
    assert market["swap"] is True
    assert market["linear"] is True
    assert market["active"] is True
    assert market["contractSize"] == 1.0
    assert market["maker"] == pytest.approx(0.0002)
    assert market["taker"] == pytest.approx(0.0006)
    assert market["precision"] == {"amount": 0.0001, "price": 0.1}
    assert market["limits"]["amount"]["min"] == pytest.approx(0.0001)
    assert market["limits"]["leverage"]["max"] == pytest.approx(125)


@pytest.mark.asyncio
async def test_load_markets_retries_observed_network_error(monkeypatch):
    client = BitunixClient()
    client._request = AsyncMock(
        side_effect=[NetworkError("transient"), [MARKET_ROW]]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("exchanges.bitunix.asyncio.sleep", sleep)

    markets = await client.load_markets(True)

    assert "BTC/USDT:USDT" in markets
    assert client._request.await_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_list", [False, True])
async def test_balance_reconstructs_wallet_and_accepts_both_account_shapes(as_list):
    client = _prepared_client()
    raw = {
        "marginCoin": "USDT",
        "available": "1000",
        "frozen": "4",
        "margin": "10",
        "positionMode": "HEDGE",
        "crossUnrealizedPNL": "2",
        "isolationUnrealizedPNL": "-1",
    }
    client._request = AsyncMock(return_value=[raw] if as_list else raw)

    balance = await client.fetch_balance()

    assert balance["free"]["USDT"] == 1000.0
    assert balance["used"]["USDT"] == 14.0
    assert balance["total"]["USDT"] == 1013.0
    assert client._request.await_args.kwargs["params"] == {"marginCoin": "USDT"}


def test_hedge_order_normalization_preserves_position_and_action_side():
    client = _prepared_client()

    entry = client._normalize_order(_order_row())
    close = client._normalize_order(
        _order_row(
            side="SELL", reduceOnly=True, clientId="clock_close_long_1"
        )
    )
    short_close = client._normalize_order(
        _order_row(
            side="BUY",
            reduceOnly=True,
            clientId="clock_close_short_1",
        )
    )

    assert (entry["side"], entry["info"]["positionSide"], entry["reduceOnly"]) == (
        "buy",
        "LONG",
        False,
    )
    assert (close["side"], close["info"]["positionSide"], close["reduceOnly"]) == (
        "sell",
        "LONG",
        True,
    )
    assert (
        short_close["side"],
        short_close["info"]["positionSide"],
        short_close["reduceOnly"],
    ) == ("buy", "SHORT", True)


def test_order_status_accepts_live_trailing_enum_padding():
    client = _prepared_client()
    assert client._normalize_order(_order_row(status="NEW_"))["status"] == "open"


@pytest.mark.parametrize("price", [None, "", "0", "-1", "nan"])
def test_limit_order_requires_positive_finite_price(price):
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.price"):
        client._normalize_order(_order_row(price=price))


@pytest.mark.parametrize("qty", [None, "", "0", "-1", "nan"])
def test_order_requires_positive_finite_quantity(qty):
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.qty"):
        client._normalize_order(_order_row(qty=qty))


def test_terminal_market_order_allows_missing_request_price():
    client = _prepared_client()

    order = client._normalize_order(
        _order_row(orderType="MARKET", price=None, status="FILLED")
    )

    assert order["type"] == "market"
    assert order["price"] == 0.0


def test_open_market_order_requires_positive_price_for_reconciliation():
    client = _prepared_client()

    with pytest.raises(ValueError, match="order.price"):
        client._normalize_order(
            _order_row(orderType="MARKET", price=None, status="NEW")
        )


@pytest.mark.asyncio
async def test_open_orders_paginate_to_stable_reported_total():
    client = _prepared_client()
    client.milliseconds = lambda: 1_700_001_000_000
    client._request = AsyncMock(
        side_effect=[
            {
                "orderList": [
                    _order_row(
                        orderId="order-2",
                        ctime=1_700_000_000_002,
                        mtime=1_700_000_000_002,
                    )
                ],
                "total": 2,
            },
            {"orderList": [_order_row()], "total": 2},
        ]
    )

    orders = await client.fetch_open_orders(limit=1)

    assert [order["id"] for order in orders] == ["order-1", "order-2"]
    assert [
        call.kwargs["params"]["skip"]
        for call in client._request.await_args_list
    ] == [0, 1]
    assert {
        call.kwargs["params"]["endTime"]
        for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"orderList": [_order_row()]},
        {"orderList": [_order_row()], "total": None},
        {"orderList": [_order_row()], "total": 0},
        {"orderList": [], "total": -1},
    ],
)
async def test_open_orders_reject_missing_or_inconsistent_total(payload):
    client = _prepared_client()
    client._request = AsyncMock(return_value=payload)

    with pytest.raises(ValueError, match="open-orders total"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_open_orders_reject_total_changes_between_pages():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"orderList": [_order_row(orderId="order-2")], "total": 2},
            {"orderList": [_order_row()], "total": 3},
        ]
    )

    with pytest.raises(ValueError, match="open-orders total changed"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_open_orders_reject_duplicate_or_incomplete_pagination():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"orderList": [_order_row()], "total": 2},
            {"orderList": [_order_row()], "total": 2},
        ]
    )

    with pytest.raises(ValueError, match="duplicate or incomplete"):
        await client.fetch_open_orders(limit=1)


@pytest.mark.asyncio
async def test_create_close_order_uses_hedge_side_and_position_id():
    client = _prepared_client()
    client._position_ids[("BTC/USDT:USDT", "long")] = "position-1"
    client._request = AsyncMock(
        return_value={"orderId": "order-1", "clientId": "clock_close_long_1"}
    )

    result = await client.create_order(
        "BTC/USDT:USDT",
        "limit",
        "sell",
        0.001,
        50_000,
        {
            "positionSide": "LONG",
            "clientOrderId": "clock_close_long_1",
            "reduceOnly": True,
            "effect": "POST_ONLY",
        },
    )

    body = client._request.await_args.kwargs["body"]
    assert body == {
        "symbol": "BTCUSDT",
        "qty": "0.001",
        "side": "BUY",
        "tradeSide": "CLOSE",
        "orderType": "LIMIT",
        "reduceOnly": True,
        "price": "50000",
        "effect": "POST_ONLY",
        "clientId": "clock_close_long_1",
        "positionId": "position-1",
    }
    assert result["side"] == "sell"
    assert result["reduceOnly"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side,position_side,reduce_only",
    [
        ("buy", "LONG", False),
        ("sell", "LONG", True),
    ],
)
async def test_create_market_order_returns_identifier_acknowledgement(
    side, position_side, reduce_only
):
    client = _prepared_client()
    if reduce_only:
        client._position_ids[("BTC/USDT:USDT", "long")] = "position-1"
    client._request = AsyncMock(
        return_value={"orderId": "order-1", "clientId": "clock_market_long_1"}
    )

    result = await client.create_order(
        "BTC/USDT:USDT",
        "market",
        side,
        0.001,
        None,
        {
            "positionSide": position_side,
            "clientOrderId": "clock_market_long_1",
            "reduceOnly": reduce_only,
        },
    )

    assert result["id"] == "order-1"
    assert result["clientOrderId"] == "clock_market_long_1"
    assert result["type"] == "market"
    assert result["side"] == side
    assert result["price"] is None
    assert result["status"] is None
    assert result["info"]["positionSide"] == position_side
    assert result["reduceOnly"] is reduce_only
    assert build_contract_bot("bitunix").did_create_order(result) is True
    body = client._request.await_args.kwargs["body"]
    assert body["orderType"] == "MARKET"
    assert "price" not in body
    assert "effect" not in body
    if reduce_only:
        assert body["positionId"] == "position-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_side,expected",
    [("LONG", "long"), ("BUY", "long"), ("SHORT", "short"), ("SELL", "short")],
)
async def test_positions_accept_documented_and_live_hedge_side_aliases(
    raw_side, expected
):
    client = _prepared_client()
    client._request = AsyncMock(return_value=[_position_row(side=raw_side)])

    positions = await client.fetch_positions(["BTC/USDT:USDT"])

    assert positions[0]["side"] == expected
    assert client._position_ids[("BTC/USDT:USDT", expected)] == "position-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_price", [None, "", "0", "-1", "nan"])
async def test_nonzero_position_requires_positive_finite_entry_price(entry_price):
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[_position_row(avgOpenPrice=entry_price)]
    )

    with pytest.raises(ValueError, match="entry price"):
        await client.fetch_positions(["BTC/USDT:USDT"])


@pytest.mark.asyncio
async def test_create_close_order_rejects_missing_live_position():
    client = _prepared_client()
    client.fetch_positions = AsyncMock(return_value=[])
    with pytest.raises(InvalidOrder, match="no live positionId"):
        await client.create_order(
            "BTC/USDT:USDT",
            "market",
            "sell",
            0.001,
            None,
            {
                "positionSide": "LONG",
                "clientOrderId": "clock_close_long_1",
                "reduceOnly": True,
            },
        )


@pytest.mark.asyncio
async def test_fill_history_paginates_and_normalizes_hedge_actions():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {
                "tradeList": [
                    _trade_row(
                        tradeId="trade-2",
                        side="SELL",
                        reduceOnly=True,
                        realizedPNL="0.5",
                        ctime=1_700_000_000_002,
                    )
                ],
                "total": 2,
            },
            {
                "tradeList": [_trade_row()],
                "total": 2,
            },
        ]
    )

    trades = await client.fetch_my_trades(
        since=1_699_999_000_000,
        limit=1,
        params={"until": 1_700_001_000_000},
    )

    assert [trade["id"] for trade in trades] == ["trade-1", "trade-2"]
    assert trades[0]["side"] == "buy"
    assert trades[0]["info"]["positionSide"] == "LONG"
    assert trades[1]["side"] == "sell"
    assert trades[1]["info"]["positionSide"] == "LONG"
    assert client._request.await_args_list[1].kwargs["params"]["skip"] == 1
    assert {
        call.kwargs["params"]["endTime"]
        for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
async def test_fill_history_freezes_end_time_when_until_is_omitted(monkeypatch):
    client = _prepared_client()
    monkeypatch.setattr(client, "milliseconds", lambda: 1_700_001_000_000)
    client._request = AsyncMock(
        side_effect=[
            {"tradeList": [_trade_row(tradeId="trade-2")], "total": 2},
            {"tradeList": [_trade_row()], "total": 2},
        ]
    )

    await client.fetch_my_trades(limit=1)

    assert [
        call.kwargs["params"]["skip"] for call in client._request.await_args_list
    ] == [0, 1]
    assert {
        call.kwargs["params"]["endTime"]
        for call in client._request.await_args_list
    } == {1_700_001_000_000}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"tradeList": [_trade_row()]},
        {"tradeList": [_trade_row()], "total": None},
        {"tradeList": [_trade_row()], "total": 0},
        {"tradeList": [_trade_row()], "total": -1},
    ],
)
async def test_fill_history_rejects_missing_or_inconsistent_total(payload):
    client = _prepared_client()
    client._request = AsyncMock(return_value=payload)

    with pytest.raises(ValueError, match="fill-history total"):
        await client.fetch_my_trades(limit=1)


@pytest.mark.asyncio
async def test_fill_history_rejects_total_changes_between_pages():
    client = _prepared_client()
    client._request = AsyncMock(
        side_effect=[
            {"tradeList": [_trade_row(tradeId="trade-2")], "total": 2},
            {"tradeList": [_trade_row()], "total": 3},
        ]
    )

    with pytest.raises(ValueError, match="fill-history total changed"):
        await client.fetch_my_trades(limit=1)


def test_trade_normalization_preserves_fee_rebate_sign():
    client = _prepared_client()

    trade = client._normalize_trade(_trade_row(fee="-0.01"))
    event = BitunixFetcher._normalize_trade(trade)

    assert trade["fee"]["cost"] == -0.01
    assert trade["fees"][0]["cost"] == -0.01
    assert signed_fee_paid_from_payload(event) == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_ohlcv_is_clamped_reversed_and_deduplicated():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 120_000,
                "open": "2",
                "high": "3",
                "low": "1",
                "close": "2.5",
                "quoteVol": "4",
                "baseVol": "10",
            },
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": "3",
                "baseVol": "4.5",
            },
        ]
    )

    candles = await client.fetch_ohlcv(
        "BTC/USDT:USDT",
        "1m",
        since=60_000,
        limit=1000,
        params={"until": 180_000},
    )

    assert candles == [
        [60_000, 1.0, 2.0, 0.5, 1.5, 3.0],
        [120_000, 2.0, 3.0, 1.0, 2.5, 4.0],
    ]
    query = client._request.await_args.kwargs["params"]
    assert query["limit"] == 200
    assert query["startTime"] == 60_000
    assert query["endTime"] == 180_000


@pytest.mark.asyncio
async def test_ohlcv_derives_forward_page_end_from_since_when_venue_ignores_start():
    client = _prepared_client()
    client._request = AsyncMock(return_value=[])

    await client.fetch_ohlcv(
        "BTC/USDT:USDT", "1m", since=60_000, limit=200
    )

    query = client._request.await_args.kwargs["params"]
    assert query["startTime"] == 60_000
    assert query["endTime"] == 60_000 + 200 * 60_000


@pytest.mark.asyncio
@pytest.mark.parametrize("volume", [None, "", "nan", "-1"])
async def test_ohlcv_rejects_missing_or_invalid_base_volume(volume):
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": volume,
            }
        ]
    )

    with pytest.raises(ValueError, match="kline.quoteVol"):
        await client.fetch_ohlcv("BTC/USDT:USDT", "1m", since=60_000)


@pytest.mark.asyncio
async def test_ohlcv_accepts_explicit_zero_base_volume():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "time": 60_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "quoteVol": "0",
            }
        ]
    )

    candles = await client.fetch_ohlcv(
        "BTC/USDT:USDT", "1m", since=60_000
    )

    assert candles[0][5] == 0.0


@pytest.mark.asyncio
async def test_order_stream_surfaces_unenriched_row_when_detail_is_not_found():
    client = _prepared_client()
    client.fetch_order = AsyncMock(side_effect=OrderNotFound("not found"))
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": {
                            "orderId": "order-1",
                            "symbol": "BTCUSDT",
                            "status": "FILLED",
                        },
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows == [
        {
            "orderId": "order-1",
            "symbol": "BTC/USDT:USDT",
            "status": "FILLED",
        }
    ]


@pytest.mark.asyncio
async def test_order_stream_sends_documented_json_ping_when_idle(monkeypatch):
    client = _prepared_client()
    stream = BitunixOrderStream(client)
    stream._last_ping_monotonic = (
        time.monotonic() - stream.PING_INTERVAL_SECONDS - 1.0
    )
    monkeypatch.setattr("exchanges.bitunix.time.time", lambda: 1_700_000_000.9)
    order_message = SimpleNamespace(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "ch": "order",
                "data": {
                    "orderId": "order-1",
                    "symbol": "BTCUSDT",
                    "status": "FILLED",
                },
            }
        ),
    )
    ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(return_value=order_message),
        send_json=AsyncMock(),
    )
    stream._ws = ws
    client.fetch_order = AsyncMock(
        return_value={"id": "order-1", "symbol": "BTC/USDT:USDT"}
    )

    rows = await stream.watch_orders()

    ws.send_json.assert_awaited_once_with(
        {"op": "ping", "ping": 1_700_000_000}
    )
    assert rows == [{"id": "order-1", "symbol": "BTC/USDT:USDT"}]


@pytest.mark.asyncio
async def test_order_stream_isolates_malformed_rest_detail_to_its_row():
    client = _prepared_client()
    valid_order = client._normalize_order(_order_row(orderId="order-2"))
    client.fetch_order = AsyncMock(
        side_effect=[ValueError("malformed order detail"), valid_order]
    )
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": [
                            {
                                "orderId": "order-1",
                                "symbol": "BTCUSDT",
                                "status": "FILLED",
                            },
                            {
                                "orderId": "order-2",
                                "symbol": "BTCUSDT",
                                "status": "FILLED",
                            },
                        ],
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows[0] == {
        "orderId": "order-1",
        "symbol": "BTC/USDT:USDT",
        "status": "FILLED",
    }
    assert rows[1] == valid_order


@pytest.mark.asyncio
async def test_order_stream_propagates_rest_transport_failure():
    client = _prepared_client()
    client.fetch_order = AsyncMock(side_effect=NetworkError("unavailable"))
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": {
                            "orderId": "order-1",
                            "symbol": "BTCUSDT",
                            "status": "FILLED",
                        },
                    }
                ),
            )
        ),
    )

    with pytest.raises(NetworkError, match="unavailable"):
        await stream.watch_orders()


@pytest.mark.asyncio
async def test_order_stream_forwards_malformed_rows_for_account_refresh():
    client = _prepared_client()
    client.fetch_order = AsyncMock()
    stream = BitunixOrderStream(client)
    stream._ws = SimpleNamespace(
        closed=False,
        receive=AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(
                    {
                        "ch": "order",
                        "data": [
                            {"symbol": "BTCUSDT", "status": "FILLED"},
                            "malformed",
                        ],
                    }
                ),
            )
        ),
    )

    rows = await stream.watch_orders()

    assert rows == [
        {
            "symbol": "BTC/USDT:USDT",
            "info": {
                "raw": {"symbol": "BTCUSDT", "status": "FILLED"}
            },
        },
        {"info": {"raw": "malformed"}},
    ]
    client.fetch_order.assert_not_awaited()

    bot = build_contract_bot("bitunix")
    bot.ccp = SimpleNamespace(has={"watchOrders": True})
    bot.stop_websocket = False
    bot._do_watch_orders = AsyncMock(
        side_effect=[rows, asyncio.CancelledError]
    )
    bot._mark_account_critical_state_dirty = MagicMock()
    bot.handle_order_update = MagicMock()

    await bot.watch_orders()

    bot._mark_account_critical_state_dirty.assert_called_once_with(
        reason="order_ws_semantics_unavailable",
        symbols={"BTC/USDT:USDT"},
        source="bitunix_order_ws",
        level=logging.DEBUG,
    )
    bot.handle_order_update.assert_not_called()


@pytest.mark.asyncio
async def test_unenriched_order_row_requests_account_state_refresh():
    bot = build_contract_bot("bitunix")
    bot.ccp = SimpleNamespace(has={"watchOrders": True})
    bot.stop_websocket = False
    bot._do_watch_orders = AsyncMock(
        side_effect=[
            [
                {
                    "orderId": "order-1",
                    "symbol": "BTC/USDT:USDT",
                    "status": "FILLED",
                }
            ],
            asyncio.CancelledError,
        ]
    )
    bot._mark_account_critical_state_dirty = MagicMock()
    bot.handle_order_update = MagicMock()

    await bot.watch_orders()

    bot._mark_account_critical_state_dirty.assert_called_once_with(
        reason="order_ws_semantics_unavailable",
        symbols={"BTC/USDT:USDT"},
        source="bitunix_order_ws",
        level=logging.DEBUG,
    )
    bot.handle_order_update.assert_not_called()


@pytest.mark.asyncio
async def test_ticker_depth_fallback_is_capped_before_any_rest_fanout():
    client = _prepared_client()
    eth_market = {
        **_market(),
        "id": "ETHUSDT",
        "symbol": "ETH/USDT:USDT",
    }
    client.markets[eth_market["symbol"]] = eth_market
    client.markets_by_id[eth_market["id"]] = eth_market
    client.symbols.append(eth_market["symbol"])
    client.TICKER_WAIT_SECONDS = 0.0
    client.MAX_DEPTH_FALLBACK_SYMBOLS = 1
    client._ensure_ticker_tasks = lambda: None
    client._fetch_depth_ticker = AsyncMock()

    with pytest.raises(NetworkError, match="fallback limited"):
        await client.fetch_tickers(client.symbols)

    client._fetch_depth_ticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_rest_only_tickers_skip_public_websocket_and_use_depth():
    client = _prepared_client()
    client.ws_enabled = False
    client._ensure_ticker_tasks = MagicMock(
        side_effect=AssertionError("public websocket must stay disabled")
    )
    expected = {
        "symbol": "BTC/USDT:USDT",
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 1_700_000_000_000,
    }
    client._fetch_depth_ticker = AsyncMock(return_value=expected)

    result = await client.fetch_tickers(["BTC/USDT:USDT"])

    assert result == {"BTC/USDT:USDT": expected}
    client._ensure_ticker_tasks.assert_not_called()
    client._fetch_depth_ticker.assert_awaited_once_with("BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_rest_only_tickers_reject_unbounded_bulk_request():
    client = _prepared_client()
    client.ws_enabled = False
    client._fetch_depth_ticker = AsyncMock()

    with pytest.raises(NetworkError, match="requires explicit symbols"):
        await client.fetch_tickers()

    client._fetch_depth_ticker.assert_not_awaited()


@pytest.mark.asyncio
async def test_depth_ticker_uses_midpoint_as_synthetic_last():
    client = _prepared_client()
    client._request = AsyncMock(
        return_value={"bids": [["99", "1"]], "asks": [["101", "1"]]}
    )
    client.milliseconds = lambda: 1_700_000_000_000

    ticker = await client._fetch_depth_ticker("BTC/USDT:USDT")

    assert ticker["last"] == 100.0
    assert ticker["source"] == "bitunix_depth_mid"


@pytest.mark.asyncio
async def test_future_exchange_timestamp_does_not_extend_cache_freshness():
    client = _prepared_client()
    symbol = "BTC/USDT:USDT"
    client.TICKER_WAIT_SECONDS = 0.0
    client._ensure_ticker_tasks = lambda: None
    client.milliseconds = lambda: 100_000
    client._ticker_cache[symbol] = {
        "symbol": symbol,
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 10**15,
    }
    client._ticker_received_monotonic[symbol] = 60.0
    fallback = {
        "symbol": symbol,
        "bid": 98.0,
        "ask": 102.0,
        "last": 100.0,
        "timestamp": 100_000,
        "source": "bitunix_depth_mid",
    }
    client._fetch_depth_ticker = AsyncMock(return_value=fallback)

    with patch("exchanges.bitunix.time.monotonic", return_value=100.0):
        tickers = await client.fetch_tickers([symbol])

    assert tickers == {symbol: fallback}
    client._fetch_depth_ticker.assert_awaited_once_with(symbol)


@pytest.mark.asyncio
async def test_bot_ticker_normalization_preserves_source_and_timestamp():
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    bot.markets_dict = {symbol: _market()}
    bot.cca = SimpleNamespace(
        fetch_tickers=AsyncMock(
            return_value={
                symbol: {
                    "bid": 99.0,
                    "ask": 101.0,
                    "last": 100.0,
                    "timestamp": 1_700_000_000_000,
                    "source": "bitunix_depth_mid",
                }
            }
        )
    )

    tickers = await bot.fetch_tickers_for_symbols([symbol])

    assert tickers[symbol] == {
        "bid": 99.0,
        "ask": 101.0,
        "last": 100.0,
        "timestamp": 1_700_000_000_000,
        "source": "bitunix_depth_mid",
    }


def test_ticker_tasks_restart_when_active_market_ids_change():
    client = _prepared_client()
    existing_task = MagicMock()
    existing_task.done.return_value = False
    client._ticker_tasks = [existing_task]
    client._ticker_subscription_ids = ("BTCUSDT",)
    created_tasks = []

    def fake_create_task(coroutine, *, name):
        coroutine.close()
        task = MagicMock()
        task.done.return_value = False
        created_tasks.append((task, name))
        return task

    with patch(
        "exchanges.bitunix.asyncio.create_task", side_effect=fake_create_task
    ) as create_task:
        client._ensure_ticker_tasks()
        create_task.assert_not_called()
        existing_task.cancel.assert_not_called()

        eth_market = {
            **_market(),
            "id": "ETHUSDT",
            "symbol": "ETH/USDT:USDT",
        }
        client.markets[eth_market["symbol"]] = eth_market
        client._ensure_ticker_tasks()

    existing_task.cancel.assert_called_once_with()
    assert client._ticker_subscription_ids == ("BTCUSDT", "ETHUSDT")
    assert len(created_tasks) == 1
    assert client._ticker_tasks == [created_tasks[0][0]]


class _ResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _RecordingSession:
    def __init__(self, response):
        self.response = response
        self.call = None

    def request(self, *args, **kwargs):
        self.call = (args, kwargs)
        return _ResponseContext(self.response)


@pytest.mark.asyncio
async def test_native_request_uses_configured_rest_base_and_extra_headers():
    client = BitunixClient(
        {
            "apiKey": "key",
            "secret": "secret",
            "restUrl": "https://proxy.example/base/",
            "headers": {"X-Proxy": "enabled"},
        }
    )
    response = SimpleNamespace(
        status=200,
        text=AsyncMock(return_value='{"code": 0, "data": []}'),
    )
    session = _RecordingSession(response)
    client._get_session = AsyncMock(return_value=session)
    client._throttle = AsyncMock()

    await client._request("GET", "/private", private=True)

    args, kwargs = session.call
    assert args[:2] == ("GET", "https://proxy.example/base/private")
    assert kwargs["headers"]["X-Proxy"] == "enabled"
    assert kwargs["headers"]["api-key"] == "key"


@pytest.mark.parametrize(
    "header_name", ["api-key", "Api-Key", "NONCE", "Timestamp", "sIgN"]
)
def test_native_session_rejects_case_insensitive_auth_header_collision(
    header_name,
):
    with pytest.raises(ValueError, match="reserved authentication header"):
        BitunixClient(
            {
                "apiKey": "key",
                "secret": "secret",
                "headers": {header_name: "must-not-win"},
            }
        )


def test_native_session_applies_bitunix_endpoint_override():
    bot = build_contract_bot("bitunix")
    bot.user_info = {
        "exchange": "bitunix",
        "key": "key",
        "secret": "secret",
        "headers": {"Existing": "header"},
    }
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_url_overrides={"api": "https://proxy.example/bitunix"},
        rest_extra_headers={"X-Proxy": "enabled"},
        disable_ws=True,
    )
    bot.ws_enabled = False

    bot.create_ccxt_sessions()

    assert bot.cca.rest_url == "https://proxy.example/bitunix"
    assert bot.cca.headers == {
        "Existing": "header",
        "X-Proxy": "enabled",
    }
    assert bot.cca.ws_enabled is False
    assert bot.ccp is None
    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_native_session_applies_bitunix_domain_rewrite():
    bot = build_contract_bot("bitunix")
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_domain_rewrites={
            "https://fapi.bitunix.com": "https://proxy.example"
        },
    )
    bot.ws_enabled = False

    bot.create_ccxt_sessions()

    assert bot.cca.rest_url == "https://proxy.example"


def test_native_session_rejects_unsupported_endpoint_url_keys():
    bot = build_contract_bot("bitunix")
    bot.endpoint_override = ResolvedEndpointOverride(
        exchange_id="bitunix",
        rest_url_overrides={"private": "https://proxy.example"},
    )
    bot.ws_enabled = False

    with pytest.raises(CustomEndpointConfigError, match="unsupported REST URL override"):
        bot.create_ccxt_sessions()


def test_market_settings_upgrade_legacy_cache_with_conservative_fees():
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    market = {
        **_market(),
        "precision": {"amount": 0.0001, "price": 0.1},
        "limits": {
            "amount": {"min": 0.0001, "max": None},
            "price": {"min": 0.1, "max": None},
            "cost": {"min": None, "max": None},
        },
        "contractSize": 1.0,
    }
    market.pop("maker")
    market.pop("taker")
    bot.markets_dict = {symbol: market}
    bot.eligible_symbols = {symbol}

    bot.set_market_specific_settings()

    assert bot.markets_dict[symbol]["maker"] == pytest.approx(0.0002)
    assert bot.markets_dict[symbol]["taker"] == pytest.approx(0.0006)


def test_bot_order_params_require_durable_hedge_semantics():
    bot = build_contract_bot("bitunix")

    params = bot._build_order_params(
        {
            "position_side": "short",
            "custom_id": "clock_close_short_1",
            "reduce_only": True,
            "type": "limit",
        }
    )

    assert params == {
        "positionSide": "SHORT",
        "clientOrderId": "clock_close_short_1",
        "reduceOnly": True,
        "effect": "GTC",
    }
    with pytest.raises(ValueError, match="invalid position_side"):
        bot._build_order_params(
            {
                "position_side": "",
                "custom_id": "clock_entry_long_1",
                "type": "limit",
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current_mode,current_leverage,expect_margin,expect_leverage",
    [
        ("CROSS", 3, False, False),
        ("CROSS", 5, False, True),
        # A margin-mode change may reset venue leverage, so set it again even
        # when the pre-mutation leverage happened to match.
        ("ISOLATION", 3, True, True),
    ],
)
async def test_symbol_config_mutates_only_needed_state_and_reapplies_leverage(
    current_mode,
    current_leverage,
    expect_margin,
    expect_leverage,
):
    bot = build_contract_bot("bitunix")
    symbol = "BTC/USDT:USDT"
    bot.max_leverage = {symbol: 125}
    bot.markets_dict = {symbol: _market()}
    bot.cca = SimpleNamespace(
        fetch_leverage_margin_mode=AsyncMock(
            return_value={
                "marginMode": current_mode,
                "leverage": current_leverage,
            }
        ),
        set_margin_mode=AsyncMock(return_value={"code": 0}),
        set_leverage=AsyncMock(return_value={"code": 0}),
    )

    await bot.update_exchange_config_by_symbols([symbol])

    assert bot.cca.set_margin_mode.await_count == int(expect_margin)
    assert bot.cca.set_leverage.await_count == int(expect_leverage)
    if expect_leverage:
        bot.cca.set_leverage.assert_awaited_once_with(3, symbol)


@pytest.mark.asyncio
async def test_bitunix_fetcher_normalizes_accounting_fields():
    client = _prepared_client()
    client.fetch_my_trades = AsyncMock(
        return_value=[client._normalize_trade(_trade_row())]
    )
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(
        1_699_999_000_000, 1_700_001_000_000, cache
    )

    assert len(events) == 1
    assert events[0]["id"] == "trade-1"
    assert events[0]["position_side"] == "long"
    assert events[0]["side"] == "buy"
    assert events[0]["pnl"] == 0.0
    assert events[0]["c_mult"] == 1.0
    assert signed_fee_paid_from_payload(events[0]) == pytest.approx(-0.01)
    assert cache["trade-1"][0] == "clock_entry_long_1"


@pytest.mark.asyncio
async def test_bitunix_fetcher_enriches_missing_fill_client_id_from_order_detail():
    client = _prepared_client()
    trade = client._normalize_trade(_trade_row(clientId=""))
    client.fetch_my_trades = AsyncMock(return_value=[trade])
    client.fetch_order = AsyncMock(
        return_value={
            "clientOrderId": "clock_entry_long_1",
            "info": {"clientId": "clock_entry_long_1"},
        }
    )
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(None, None, cache)

    assert events[0]["client_order_id"] == "clock_entry_long_1"
    assert events[0]["pb_order_type"]
    assert cache["trade-1"][0] == "clock_entry_long_1"


@pytest.mark.asyncio
async def test_bitunix_fetcher_retains_fill_when_order_detail_expired():
    client = _prepared_client()
    trade = client._normalize_trade(_trade_row(clientId=""))
    client.fetch_my_trades = AsyncMock(return_value=[trade])
    client.fetch_order = AsyncMock(side_effect=OrderNotFound("expired"))
    fetcher = BitunixFetcher(client)
    cache = {}

    events = await fetcher.fetch(None, None, cache)

    assert [event["id"] for event in events] == ["trade-1"]
    assert events[0]["client_order_id"] == ""
    assert events[0]["pb_order_type"] == "unknown"
    assert cache["trade-1"] == ("", "unknown")


def test_build_fetcher_for_bitunix():
    bot = SimpleNamespace(
        exchange="bitunix",
        cca=object(),
        user="bitunix_user",
        config={"live": {}},
    )
    assert isinstance(_build_fetcher_for_bot(bot, []), BitunixFetcher)


def test_setup_bot_bitunix_uses_native_adapter():
    from passivbot import setup_bot

    config = {"live": {"user": "bitunix_01"}}
    with patch("passivbot.load_user_info", return_value={"exchange": "bitunix"}):
        with patch("exchanges.bitunix.BitunixBot") as mock_cls:
            result = setup_bot(config)
    assert result is mock_cls.return_value
    mock_cls.assert_called_once_with(config)
