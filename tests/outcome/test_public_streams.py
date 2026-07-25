from __future__ import annotations

import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.models import OutcomeSide
from outcome.public_streams import (
    _HyperliquidSubscriptionGate,
    decode_hyperliquid_ws_book_message,
    decode_hyperliquid_ws_message,
    decode_polymarket_ws_book_message,
    decode_polymarket_ws_message,
    decode_polymarket_ws_price_grid_change,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_decode_hyperliquid_trade_channel_and_ignore_control_messages():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    assert (
        decode_hyperliquid_ws_message(
            {"channel": "subscriptionResponse", "data": {}},
            [market],
            received_time_ms=2_000,
        )
        == []
    )
    trades = decode_hyperliquid_ws_message(
        {
            "channel": "trades",
            "data": [
                {
                    "coin": "#9130",
                    "side": "A",
                    "px": "0.42",
                    "sz": "3",
                    "hash": "0xabc",
                    "time": 1_900,
                    "tid": 123,
                    "users": ["0xbuyer", "0xseller"],
                }
            ],
        },
        [market],
        received_time_ms=2_000,
    )
    assert len(trades) == 1
    assert trades[0].outcome is OutcomeSide.YES


def test_hyperliquid_trade_gate_waits_for_all_subscription_acknowledgements():
    gate = _HyperliquidSubscriptionGate("trades", ("#9130", "#9131"))
    trade_message = {"channel": "trades", "data": []}

    assert gate.allows(trade_message) is False
    assert (
        gate.allows(
            {
                "channel": "subscriptionResponse",
                "data": {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": "#9130"},
                },
            }
        )
        is False
    )
    assert gate.allows(trade_message) is False
    assert (
        gate.allows(
            {
                "channel": "subscriptionResponse",
                "data": {
                    "method": "subscribe",
                    "subscription": {"type": "trades", "coin": "#9131"},
                },
            }
        )
        is False
    )
    assert gate.allows(trade_message) is True


def test_websocket_decoder_assigns_monotonic_collector_sequence_to_batch():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    trades = decode_hyperliquid_ws_message(
        {
            "channel": "trades",
            "data": [
                {
                    "coin": "#9130",
                    "side": "A",
                    "px": "0.42",
                    "sz": "3",
                    "hash": "0xabc",
                    "time": 1_900,
                    "tid": 123,
                    "users": ["0xbuyer", "0xseller"],
                },
                {
                    "coin": "#9130",
                    "side": "A",
                    "px": "0.43",
                    "sz": "2",
                    "hash": "0xdef",
                    "time": 1_900,
                    "tid": 456,
                    "users": ["0xbuyer", "0xseller"],
                },
            ],
        },
        [market],
        received_time_ms=2_000,
        collector_sequence_start=50,
    )

    assert [trade.collector_sequence for trade in trades] == [50, 51]


def test_decode_polymarket_market_channel_filters_quote_events():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    messages = [
        {
            "event_type": "best_bid_ask",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "best_bid": "0.4",
            "best_ask": "0.5",
            "timestamp": "1",
        },
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.no_asset.asset_id,
            "price": "0.6",
            "size": "2",
            "side": "SELL",
            "timestamp": "1750428146322",
            "transaction_hash": "0xtransaction",
        },
    ]
    trades = decode_polymarket_ws_message(messages, [market], received_time_ms=1750428146400)

    assert len(trades) == 1
    assert trades[0].outcome is OutcomeSide.NO
    assert trades[0].canonical_yes_price == pytest.approx(0.4)


def test_decode_hyperliquid_book_is_archival_market_data_not_a_trade():
    market = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    books = decode_hyperliquid_ws_book_message(
        {
            "channel": "l2Book",
            "data": {
                "coin": "#9130",
                "time": 1_900,
                "levels": [
                    [{"px": "0.4", "sz": "10", "n": 2}],
                    [{"px": "0.42", "sz": "12", "n": 1}],
                ],
            },
        },
        [market],
        received_time_ms=2_000,
    )

    assert len(books) == 1
    assert books[0].outcome is OutcomeSide.YES
    assert books[0].bids[0].native_price == pytest.approx(0.4)
    assert books[0].bids[0].order_count == 2
    assert (
        decode_hyperliquid_ws_message(
            {
                "channel": "l2Book",
                "data": books[0].raw_payload,
            },
            [market],
            received_time_ms=2_000,
        )
        == []
    )


def test_decode_polymarket_book_sorts_levels_without_inventing_order_counts():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    books = decode_polymarket_ws_book_message(
        {
            "event_type": "book",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "timestamp": "1750428146322",
            "bids": [
                {"price": "0.39", "size": "10"},
                {"price": "0.4", "size": "5"},
            ],
            "asks": [
                {"price": "0.43", "size": "8"},
                {"price": "0.42", "size": "7"},
            ],
        },
        [market],
        received_time_ms=1750428146400,
    )

    assert [level.native_price for level in books[0].bids] == [0.4, 0.39]
    assert [level.native_price for level in books[0].asks] == [0.42, 0.43]
    assert books[0].bids[0].order_count is None


def test_decode_polymarket_tick_change_is_not_a_trade_or_book():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    event = {
        "event_type": "tick_size_change",
        "asset_id": market.yes_asset.asset_id,
        "market": market.market_id,
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
        "timestamp": "1750428146322",
    }

    assert decode_polymarket_ws_message(
        event,
        [market],
        received_time_ms=1750428146400,
    ) == []
    assert decode_polymarket_ws_book_message(
        event,
        [market],
        received_time_ms=1750428146400,
    ) == []
    changes = decode_polymarket_ws_price_grid_change(
        event,
        [market],
        received_time_ms=1750428146400,
    )

    assert len(changes) == 1
    assert changes[0].old_grid.fixed_step == pytest.approx(0.01)
    assert changes[0].new_grid.fixed_step == pytest.approx(0.001)
