import sys
import types

import pytest

# Stub passivbot_rust before importing passivbot to avoid native dependency during unit test.
sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

from passivbot import Passivbot


@pytest.mark.asyncio
async def test_update_positions_applies_balance_and_positions(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.quote = "USDT"
    calls = {"balance": 0}

    async def fake_fetch_positions():
        return (
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "position_side": "long",
                    "size": 1.0,
                    "price": 100.0,
                }
            ],
            200.0,
        )

    async def fake_log_position_changes(old, new):
        return

    async def fake_handle_balance_update(upd, source="REST"):
        calls["balance"] += 1
        assert upd["USDT"]["total"] == pytest.approx(200.0)

    bot.fetch_positions = fake_fetch_positions
    bot.log_position_changes = fake_log_position_changes
    bot.handle_balance_update = fake_handle_balance_update

    ok = await bot.update_positions()
    assert ok is True
    assert bot.fetched_positions[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(1.0)
    assert calls["balance"] == 1
    # balance cached and marked applied
    assert getattr(bot, "_last_balance_from_positions") == pytest.approx(200.0)
    assert getattr(bot, "_balance_from_positions_applied") is True

    # update_balance should skip reapplying cached balance
    async def fail_fetch_balance():
        raise AssertionError("fetch_balance should not be called when cached balance exists")

    bot.fetch_balance = fail_fetch_balance
    ok = await bot.update_balance()
    assert ok is True
    assert calls["balance"] == 1


@pytest.mark.asyncio
async def test_update_balance_uses_fetch_balance_when_no_cache():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    calls = {"balance": 0}

    async def fake_handle_balance_update(upd, source="REST"):
        calls["balance"] += 1
        assert upd["USDT"]["total"] == pytest.approx(123.0)

    async def fake_fetch_balance():
        return 123.0

    bot.handle_balance_update = fake_handle_balance_update
    bot.fetch_balance = fake_fetch_balance

    ok = await bot.update_balance()
    assert ok is True
    assert calls["balance"] == 1
