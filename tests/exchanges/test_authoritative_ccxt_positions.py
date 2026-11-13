from types import SimpleNamespace

import pytest

from exchanges.bybit import BybitBot
from exchanges.okx import OKXBot
from live.state_refresh import AuthoritativeSurfaceUnavailable


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [BybitBot, OKXBot])
@pytest.mark.parametrize("method", ["fetch_positions", "capture_positions_snapshot"])
@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("contracts", None, "invalid_contracts"),
        ("contracts", float("nan"), "invalid_contracts"),
        ("side", None, "invalid_position_side"),
        ("side", "sell", "invalid_position_side"),
        ("entryPrice", None, "invalid_entry_price"),
        ("entryPrice", 0, "invalid_entry_price"),
    ],
)
async def test_exchange_rejects_incomplete_held_position_snapshot(adapter, method, field, value, reason):
    bot = adapter.__new__(adapter)
    bot.exchange = "test"
    bot._live_margin_modes = {}
    valid = {
        "symbol": "BTC/USDT:USDT",
        "contracts": "2",
        "entryPrice": "100",
        "side": "short",
        "marginMode": "isolated",
    }
    malformed = {**valid, field: value}
    if value is None:
        del malformed[field]

    async def fetch_positions(**kwargs):
        return [valid, malformed]

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match=reason):
        await getattr(bot, method)()
    assert bot._live_margin_modes == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [BybitBot, OKXBot])
@pytest.mark.parametrize(
    ("payload", "reason"),
    [(None, "invalid_payload"), ({}, "invalid_payload"), ([None], "invalid_row")],
)
async def test_exchange_rejects_malformed_positions_payload(adapter, payload, reason):
    bot = adapter.__new__(adapter)
    bot.exchange = "test"

    async def fetch_positions(**kwargs):
        return payload

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match=reason):
        await bot.capture_positions_snapshot()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [BybitBot, OKXBot])
async def test_exchange_retains_authoritative_position_fields_and_skips_explicit_zero(adapter):
    bot = adapter.__new__(adapter)
    bot.exchange = "test"
    bot._live_margin_modes = {}
    info = {"positionIdx": "2", "updatedTime": "1001"}
    held = {
        "symbol": "BTC/USDT:USDT",
        "contracts": "2",
        "entryPrice": "100",
        "side": "SHORT",
        "marginMode": "isolated",
        "lastUpdateTimestamp": 1001,
        "info": info,
    }

    async def fetch_positions(**kwargs):
        return [{"contracts": "0"}, held]

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    raw, positions = await bot.capture_positions_snapshot()
    assert raw == [{"contracts": "0"}, held]
    assert positions == [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "short",
            "size": 2,
            "price": 100,
            "margin_mode": "isolated",
            "lastUpdateTimestamp": 1001,
            "info": info,
        }
    ]
    assert bot._live_margin_modes == {"BTC/USDT:USDT": "isolated"}


@pytest.mark.asyncio
@pytest.mark.parametrize("position_idx", [1, 3, None])
async def test_bybit_rejects_held_position_with_conflicting_or_invalid_native_index(position_idx):
    bot = BybitBot.__new__(BybitBot)

    async def fetch_positions(**kwargs):
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "contracts": 2,
                "entryPrice": 100,
                "side": "short",
                "info": {"positionIdx": position_idx},
            }
        ]

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="invalid_position_side"):
        await bot.capture_positions_snapshot()


@pytest.mark.asyncio
async def test_bybit_validates_duplicate_position_on_later_page_before_deduplication():
    bot = BybitBot.__new__(BybitBot)
    bot._live_margin_modes = {}
    calls = []
    held = {
        "symbol": "BTC/USDT:USDT",
        "contracts": 2,
        "entryPrice": 100,
        "side": "short",
        "marginMode": "isolated",
        "info": {"positionIdx": 2, "nextPageCursor": "next"},
    }

    async def fetch_positions(*, params):
        calls.append(params)
        if params.get("cursor") == "next":
            return [{**held, "entryPrice": None}]
        return [held] * 200

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="invalid_entry_price"):
        await bot.capture_positions_snapshot()
    assert calls == [
        {"limit": 200, "paginate": False},
        {"cursor": "next", "limit": 200, "paginate": False},
    ]
    assert bot._live_margin_modes == {}



@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", [1, 200])
async def test_bybit_follows_page_cursor_when_every_position_on_page_is_flat(page_size):
    bot = BybitBot.__new__(BybitBot)
    calls = []
    flat = {"contracts": 0, "info": {"nextPageCursor": "next"}}
    held = {
        "symbol": "BTC/USDT:USDT",
        "contracts": 2,
        "entryPrice": 100,
        "side": "short",
    }

    async def fetch_positions(*, params):
        calls.append(params)
        if params.get("cursor") == "next":
            return [held]
        return [flat] + [{"contracts": 0}] * (page_size - 1)

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    raw, normalized = await bot.capture_positions_snapshot()
    assert calls == [
        {"limit": 200, "paginate": False},
        {"cursor": "next", "limit": 200, "paginate": False},
    ]
    assert raw[-1] == held
    assert [position["symbol"] for position in normalized] == [held["symbol"]]


@pytest.mark.asyncio
async def test_bybit_rejects_repeated_position_cursor():
    bot = BybitBot.__new__(BybitBot)
    calls = []
    held = {
        "symbol": "BTC/USDT:USDT",
        "contracts": 2,
        "entryPrice": 100,
        "side": "short",
        "info": {"nextPageCursor": "next"},
    }

    async def fetch_positions(*, params):
        calls.append(params)
        return [held]

    bot.cca = SimpleNamespace(fetch_positions=fetch_positions)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match="repeated_pagination_cursor"):
        await bot.capture_positions_snapshot()
    assert len(calls) == 2
