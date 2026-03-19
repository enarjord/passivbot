from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from exchanges.fake import FakeCCXTClient
from fill_events_manager import FakeFetcher, _build_fetcher_for_bot


def _scenario() -> dict:
    return {
        "name": "unit",
        "start_time": "2026-01-01T00:00:00Z",
        "tick_interval_seconds": 60,
        "boot_index": 0,
        "account": {"balance": 1000.0},
        "symbols": {
            "BTC/USDT:USDT": {
                "qty_step": 0.001,
                "price_step": 0.1,
                "min_qty": 0.001,
                "min_cost": 5.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.00055,
            }
        },
        "timeline": [
            {"t": 0, "prices": {"BTC/USDT:USDT": 100.0}},
            {"t": 1, "prices": {"BTC/USDT:USDT": 98.0}},
            {"t": 2, "prices": {"BTC/USDT:USDT": 105.0}},
        ],
    }


@pytest.mark.asyncio
async def test_fake_ccxt_client_limit_and_market_fills():
    client = FakeCCXTClient(_scenario(), quote="USDT")

    order = await client.create_order(
        "BTC/USDT:USDT",
        "limit",
        "buy",
        1.0,
        price=99.0,
        params={"positionSide": "LONG", "clientOrderId": "entry_long"},
    )
    assert order["status"] == "open"
    assert len(await client.fetch_open_orders()) == 1

    assert client.advance_time() is True
    assert len(await client.fetch_open_orders()) == 0
    positions = await client.fetch_positions()
    assert positions == [
        {
            "symbol": "BTC/USDT:USDT",
            "contracts": 1.0,
            "entryPrice": 99.0,
            "side": "long",
            "info": {"positionSide": "LONG"},
        }
    ]
    assert len(client.fills) == 1
    assert client.fills[0]["clientOrderId"] == "entry_long"

    assert client.advance_time() is True
    close_order = await client.create_order(
        "BTC/USDT:USDT",
        "market",
        "sell",
        1.0,
        price=None,
        params={"positionSide": "LONG", "reduceOnly": True, "clientOrderId": "close_long"},
    )
    assert close_order["status"] == "closed"
    positions = await client.fetch_positions()
    assert positions == []
    assert len(client.fills) == 2
    assert client.balance_total > 1000.0


@pytest.mark.asyncio
async def test_fake_fetcher_reads_fill_ledger():
    client = FakeCCXTClient(_scenario(), quote="USDT")
    await client.create_order(
        "BTC/USDT:USDT",
        "market",
        "buy",
        1.0,
        params={"positionSide": "LONG", "clientOrderId": "entry_long"},
    )

    fetcher = FakeFetcher(api=client)
    events = await fetcher.fetch(None, None, detail_cache={})
    assert len(events) == 1
    assert events[0]["symbol"] == "BTC/USDT:USDT"
    assert events[0]["position_side"] == "long"
    assert events[0]["client_order_id"] == "entry_long"


def test_build_fetcher_fake():
    bot = SimpleNamespace()
    bot.exchange = "fake"
    bot.cca = "dummy"
    bot.user = "fake_user"
    bot.markets_dict = {}
    bot.coin_to_symbol = lambda value, verbose=False: value
    fetcher = _build_fetcher_for_bot(bot, symbols=["BTC"])
    assert isinstance(fetcher, FakeFetcher)


def test_setup_bot_fake_uses_fake_bot():
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}
    mock_user_info = {"exchange": "fake", "quote": "USDT", "fake_scenario_path": "scenario.hjson"}

    with patch("passivbot.load_user_info", return_value=mock_user_info):
        with patch("exchanges.fake.FakeBot") as mock_fake_bot:
            mock_bot = MagicMock()
            mock_fake_bot.return_value = mock_bot
            result = setup_bot(config)
            mock_fake_bot.assert_called_once_with(config)
            assert result == mock_bot


def test_fake_ccxt_client_builds_replay_timeline_from_file(tmp_path):
    ohlcv_path = tmp_path / "btc.npy"
    np.save(
        ohlcv_path,
        np.array(
            [
                [1_704_067_200_000, 100.0, 101.0, 99.0, 100.5, 10.0],
                [1_704_067_260_000, 100.5, 102.0, 100.0, 101.5, 11.0],
                [1_704_067_320_000, 101.5, 103.0, 101.0, 102.5, 12.0],
            ],
            dtype=float,
        ),
    )
    scenario = {
        "name": "replay",
        "boot_index": 1,
        "account": {"balance": 1000.0},
        "symbols": {
            "BTC/USDT:USDT": {
                "qty_step": 0.001,
                "price_step": 0.1,
                "min_qty": 0.001,
                "min_cost": 5.0,
            }
        },
        "replay": {
            "symbols": {
                "BTC/USDT:USDT": {
                    "file": str(ohlcv_path),
                }
            }
        },
    }

    client = FakeCCXTClient(scenario, quote="USDT")
    assert client.now_ms == 1_704_067_260_000
    assert client.get_current_step()["prices"]["BTC/USDT:USDT"] == 101.5


@pytest.mark.asyncio
async def test_fake_ccxt_client_supports_boot_fill_history_and_1h_aggregation():
    scenario = _scenario()
    scenario["account"] = {
        "balance": 200.0,
        "positions": [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "qty": 5.0,
                "price": 100.0,
            }
        ],
        "fills": [
            {
                "id": "10",
                "order": "10",
                "timestamp": "2026-01-01T00:00:00Z",
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "buy",
                "amount": 5.0,
                "price": 80.0,
                "pnl": 0.0,
                "clientOrderId": "entry_boot",
            },
            {
                "id": "11",
                "order": "11",
                "timestamp": "2026-01-01T00:01:00Z",
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "sell",
                "amount": 5.0,
                "price": 100.0,
                "pnl": 100.0,
                "reduceOnly": True,
                "clientOrderId": "close_boot",
            },
            {
                "id": "12",
                "order": "12",
                "timestamp": "2026-01-01T00:02:00Z",
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "buy",
                "amount": 5.0,
                "price": 100.0,
                "pnl": 0.0,
                "clientOrderId": "reentry_boot",
            },
        ],
    }
    scenario["boot_index"] = 2
    client = FakeCCXTClient(scenario, quote="USDT")

    trades = await client.fetch_my_trades("BTC/USDT:USDT")
    assert [trade["clientOrderId"] for trade in trades] == [
        "entry_boot",
        "close_boot",
        "reentry_boot",
    ]
    assert client.realized_pnl == pytest.approx(100.0)
    assert client.balance_total == pytest.approx(200.0)

    candles_1h = await client.fetch_ohlcv(
        "BTC/USDT:USDT",
        timeframe="1h",
        since=1_767_225_600_000,
    )
    assert candles_1h == [
        [1_767_225_600_000.0, 100.0, 105.0, 98.0, 105.0, 0.0],
    ]
