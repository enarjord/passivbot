from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import time
from typing import Any, AsyncIterator, Iterable, Mapping

from outcome.adapters import hyperliquid, polymarket
from outcome.models import (
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeBookSnapshot,
    OutcomePriceGridChange,
    OutcomeVenue,
)


HYPERLIQUID_PUBLIC_WS_URL = "wss://api.hyperliquid.xyz/ws"
POLYMARKET_PUBLIC_MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


async def _polymarket_ping_loop(websocket: Any) -> None:
    """Polymarket requires the literal text heartbeat rather than websocket ping frames."""

    while True:
        await asyncio.sleep(10)
        await websocket.send_str("PING")


def _market_by_symbol(
    markets: Iterable[NormalizedOutcomeMarket],
) -> dict[str, NormalizedOutcomeMarket]:
    result: dict[str, NormalizedOutcomeMarket] = {}
    for market in markets:
        for asset in (market.yes_asset, market.no_asset):
            if asset.market_data_symbol in result:
                raise ValueError(f"duplicate outcome market-data symbol {asset.market_data_symbol!r}")
            result[asset.market_data_symbol] = market
    return result


def decode_hyperliquid_ws_message(
    message: Mapping[str, Any],
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    received_time_ms: int,
    collector_sequence_start: int | None = None,
) -> list[NormalizedOutcomeTrade]:
    if message.get("channel") != "trades":
        return []
    data = message.get("data")
    if not isinstance(data, list):
        raise ValueError("Hyperliquid trades websocket data must be an array")
    by_symbol = _market_by_symbol(markets)
    trades = []
    for index, payload in enumerate(data):
        if not isinstance(payload, Mapping):
            raise ValueError("Hyperliquid websocket trade must be an object")
        coin = str(payload.get("coin", ""))
        market = by_symbol.get(coin)
        if market is None:
            raise ValueError(f"unsubscribed Hyperliquid outcome symbol {coin!r}")
        trades.append(
            hyperliquid.normalize_trade(
                payload,
                market,
                received_time_ms=received_time_ms,
                collector_sequence=(
                    collector_sequence_start + index
                    if collector_sequence_start is not None
                    else None
                ),
            )
        )
    return trades


def decode_hyperliquid_ws_book_message(
    message: Mapping[str, Any],
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    received_time_ms: int,
) -> list[OutcomeBookSnapshot]:
    if message.get("channel") != "l2Book":
        return []
    data = message.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Hyperliquid l2Book websocket data must be an object")
    by_symbol = _market_by_symbol(markets)
    coin = str(data.get("coin", ""))
    market = by_symbol.get(coin)
    if market is None:
        raise ValueError(f"unsubscribed Hyperliquid outcome symbol {coin!r}")
    asset = market.asset_for_id(coin)
    return [
        hyperliquid.normalize_l2_book(
            data,
            market,
            outcome=asset.side,
            received_time_ms=received_time_ms,
        )
    ]


def decode_polymarket_ws_message(
    message: Mapping[str, Any] | list[Any],
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    received_time_ms: int,
    collector_sequence_start: int | None = None,
) -> list[NormalizedOutcomeTrade]:
    entries = message if isinstance(message, list) else [message]
    by_symbol = _market_by_symbol(markets)
    trades = []
    for payload in entries:
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket websocket event must be an object")
        if payload.get("event_type") != "last_trade_price":
            continue
        asset_id = str(payload.get("asset_id", ""))
        market = by_symbol.get(asset_id)
        if market is None:
            raise ValueError(f"unsubscribed Polymarket outcome asset {asset_id!r}")
        trades.append(
            polymarket.normalize_market_ws_trade(
                payload,
                market,
                received_time_ms=received_time_ms,
                collector_sequence=(
                    collector_sequence_start + len(trades)
                    if collector_sequence_start is not None
                    else None
                ),
            )
        )
    return trades


def decode_polymarket_ws_book_message(
    message: Mapping[str, Any] | list[Any],
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    received_time_ms: int,
) -> list[OutcomeBookSnapshot]:
    entries = message if isinstance(message, list) else [message]
    by_symbol = _market_by_symbol(markets)
    books = []
    for payload in entries:
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket websocket event must be an object")
        if payload.get("event_type") != "book":
            continue
        asset_id = str(payload.get("asset_id", ""))
        market = by_symbol.get(asset_id)
        if market is None:
            raise ValueError(f"unsubscribed Polymarket outcome asset {asset_id!r}")
        books.append(
            polymarket.normalize_market_ws_book(
                payload,
                market,
                received_time_ms=received_time_ms,
            )
        )
    return books


def decode_polymarket_ws_price_grid_change(
    message: Mapping[str, Any] | list[Any],
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    received_time_ms: int,
) -> list[OutcomePriceGridChange]:
    entries = message if isinstance(message, list) else [message]
    by_symbol = _market_by_symbol(markets)
    changes = []
    for payload in entries:
        if not isinstance(payload, Mapping):
            raise ValueError("Polymarket websocket event must be an object")
        if payload.get("event_type") != "tick_size_change":
            continue
        asset_id = str(payload.get("asset_id", ""))
        market = by_symbol.get(asset_id)
        if market is None:
            raise ValueError(f"unsubscribed Polymarket outcome asset {asset_id!r}")
        changes.append(
            polymarket.normalize_market_ws_price_grid_change(
                payload,
                market,
                received_time_ms=received_time_ms,
            )
        )
    return changes


async def stream_hyperliquid_public_trades(
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    ws_url: str = HYPERLIQUID_PUBLIC_WS_URL,
) -> AsyncIterator[NormalizedOutcomeTrade]:
    """Yield one connected public session; caller owns reconnect and coverage boundaries."""

    import aiohttp

    market_list = list(markets)
    if not market_list or any(market.venue is not OutcomeVenue.HYPERLIQUID for market in market_list):
        raise ValueError("Hyperliquid stream requires at least one Hyperliquid outcome market")
    symbols = [
        asset.market_data_symbol
        for market in market_list
        for asset in (market.yes_asset, market.no_asset)
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
    collector_sequence = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(ws_url, heartbeat=30) as websocket:
            for symbol in symbols:
                await websocket.send_json(
                    {
                        "method": "subscribe",
                        "subscription": {"type": "trades", "coin": symbol},
                    }
                )
            async for event in websocket:
                if event.type == aiohttp.WSMsgType.TEXT:
                    received_time_ms = int(time.time() * 1_000)
                    payload = json.loads(event.data)
                    for trade in decode_hyperliquid_ws_message(
                        payload,
                        market_list,
                        received_time_ms=received_time_ms,
                        collector_sequence_start=collector_sequence,
                    ):
                        yield trade
                        collector_sequence += 1
                elif event.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break


async def stream_hyperliquid_public_books(
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    ws_url: str = HYPERLIQUID_PUBLIC_WS_URL,
) -> AsyncIterator[OutcomeBookSnapshot]:
    """Yield full public L2 snapshots; books are archival/live inputs, never candle inputs."""

    import aiohttp
    market_list = list(markets)
    if not market_list or any(
        market.venue is not OutcomeVenue.HYPERLIQUID for market in market_list
    ):
        raise ValueError("Hyperliquid stream requires at least one Hyperliquid outcome market")
    symbols = [
        asset.market_data_symbol
        for market in market_list
        for asset in (market.yes_asset, market.no_asset)
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(ws_url, heartbeat=30) as websocket:
            for symbol in symbols:
                await websocket.send_json(
                    {
                        "method": "subscribe",
                        "subscription": {"type": "l2Book", "coin": symbol},
                    }
                )
            async for event in websocket:
                if event.type == aiohttp.WSMsgType.TEXT:
                    received_time_ms = int(time.time() * 1_000)
                    payload = json.loads(event.data)
                    for book in decode_hyperliquid_ws_book_message(
                        payload,
                        market_list,
                        received_time_ms=received_time_ms,
                    ):
                        yield book
                elif event.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break


async def stream_polymarket_public_trades(
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    ws_url: str = POLYMARKET_PUBLIC_MARKET_WS_URL,
) -> AsyncIterator[NormalizedOutcomeTrade]:
    """Yield one connected public session; caller owns reconnect and coverage boundaries."""

    import aiohttp

    market_list = list(markets)
    if not market_list or any(market.venue is not OutcomeVenue.POLYMARKET for market in market_list):
        raise ValueError("Polymarket stream requires at least one Polymarket outcome market")
    asset_ids = [
        asset.market_data_symbol
        for market in market_list
        for asset in (market.yes_asset, market.no_asset)
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
    collector_sequence = 0
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(ws_url, heartbeat=10) as websocket:
            await websocket.send_json({"assets_ids": asset_ids, "type": "market"})
            ping_task = asyncio.create_task(_polymarket_ping_loop(websocket))
            try:
                async for event in websocket:
                    if event.type == aiohttp.WSMsgType.TEXT:
                        received_time_ms = int(time.time() * 1_000)
                        if event.data == "PONG":
                            continue
                        payload = json.loads(event.data)
                        for trade in decode_polymarket_ws_message(
                            payload,
                            market_list,
                            received_time_ms=received_time_ms,
                            collector_sequence_start=collector_sequence,
                        ):
                            yield trade
                            collector_sequence += 1
                    elif event.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
            finally:
                ping_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ping_task


async def stream_polymarket_public_books(
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    ws_url: str = POLYMARKET_PUBLIC_MARKET_WS_URL,
) -> AsyncIterator[OutcomeBookSnapshot]:
    """Yield full public book snapshots; order-book events never contribute to candles."""

    import aiohttp
    market_list = list(markets)
    if not market_list or any(
        market.venue is not OutcomeVenue.POLYMARKET for market in market_list
    ):
        raise ValueError("Polymarket stream requires at least one Polymarket outcome market")
    asset_ids = [
        asset.market_data_symbol
        for market in market_list
        for asset in (market.yes_asset, market.no_asset)
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(ws_url, heartbeat=10) as websocket:
            await websocket.send_json({"assets_ids": asset_ids, "type": "market"})
            ping_task = asyncio.create_task(_polymarket_ping_loop(websocket))
            try:
                async for event in websocket:
                    if event.type == aiohttp.WSMsgType.TEXT:
                        received_time_ms = int(time.time() * 1_000)
                        if event.data == "PONG":
                            continue
                        payload = json.loads(event.data)
                        for book in decode_polymarket_ws_book_message(
                            payload,
                            market_list,
                            received_time_ms=received_time_ms,
                        ):
                            yield book
                    elif event.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
            finally:
                ping_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ping_task


async def stream_polymarket_public_price_grid_changes(
    markets: Iterable[NormalizedOutcomeMarket],
    *,
    ws_url: str = POLYMARKET_PUBLIC_MARKET_WS_URL,
) -> AsyncIterator[OutcomePriceGridChange]:
    """Yield public constraint changes; these are replay inputs, never candle inputs."""

    import aiohttp

    market_list = list(markets)
    if not market_list or any(
        market.venue is not OutcomeVenue.POLYMARKET for market in market_list
    ):
        raise ValueError("Polymarket stream requires at least one Polymarket outcome market")
    asset_ids = [
        asset.market_data_symbol
        for market in market_list
        for asset in (market.yes_asset, market.no_asset)
    ]
    timeout = aiohttp.ClientTimeout(total=None, connect=20, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.ws_connect(ws_url, heartbeat=10) as websocket:
            await websocket.send_json({"assets_ids": asset_ids, "type": "market"})
            ping_task = asyncio.create_task(_polymarket_ping_loop(websocket))
            try:
                async for event in websocket:
                    if event.type == aiohttp.WSMsgType.TEXT:
                        received_time_ms = int(time.time() * 1_000)
                        if event.data == "PONG":
                            continue
                        payload = json.loads(event.data)
                        for change in decode_polymarket_ws_price_grid_change(
                            payload,
                            market_list,
                            received_time_ms=received_time_ms,
                        ):
                            yield change
                    elif event.type in {
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
            finally:
                ping_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ping_task
