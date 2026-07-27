import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from ccxt.base.errors import (
    AuthenticationError,
    InsufficientFunds,
    InvalidOrder,
    NetworkError,
    RateLimitExceeded,
)

from ccxt_contracts import build_contract_bot, get_bot_class
from exchanges.bitunix import BitunixBot, BitunixClient, BitunixOrderStream
from fill_events_manager import BitunixFetcher, _build_fetcher_for_bot
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
    assert market["precision"] == {"amount": 0.0001, "price": 0.1}
    assert market["limits"]["amount"]["min"] == pytest.approx(0.0001)
    assert market["limits"]["leverage"]["max"] == pytest.approx(125)


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
    "raw_side,expected",
    [("LONG", "long"), ("BUY", "long"), ("SHORT", "short"), ("SELL", "short")],
)
async def test_positions_accept_documented_and_live_hedge_side_aliases(
    raw_side, expected
):
    client = _prepared_client()
    client._request = AsyncMock(
        return_value=[
            {
                "positionId": "position-1",
                "symbol": "BTCUSDT",
                "qty": "0.001",
                "entryValue": "50",
                "side": raw_side,
                "positionMode": "HEDGE",
                "marginMode": "CROSS",
                "leverage": 3,
                "unrealizedPNL": "0",
                "realizedPNL": "0",
                "avgOpenPrice": "50000",
                "ctime": 1_700_000_000_000,
                "mtime": 1_700_000_000_001,
            }
        ]
    )

    positions = await client.fetch_positions(["BTC/USDT:USDT"])

    assert positions[0]["side"] == expected
    assert client._position_ids[("BTC/USDT:USDT", expected)] == "position-1"


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
