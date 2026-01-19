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
async def test_update_positions_only_updates_positions(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.quote = "USDT"

    async def fake_fetch_positions():
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 1.0,
                "price": 100.0,
            }
        ]

    async def fake_log_position_changes(old, new):
        return

    async def fail_handle_balance_update(source="REST"):
        raise AssertionError("handle_balance_update should not be called by update_positions")

    bot.fetch_positions = fake_fetch_positions
    bot.log_position_changes = fake_log_position_changes
    bot.handle_balance_update = fail_handle_balance_update

    ok = await bot.update_positions()
    assert ok is True
    assert bot.fetched_positions[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_update_balance_uses_fetch_balance_when_no_cache():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    calls = {"balance": 0}

    async def fake_handle_balance_update(source="REST"):
        calls["balance"] += 1
        assert bot.balance == pytest.approx(123.0)

    async def fake_fetch_balance():
        return 123.0

    bot.fetch_balance = fake_fetch_balance

    ok = await bot.update_balance()
    assert ok is True
    # handle_balance_update should not be called automatically
    assert calls["balance"] == 0
    # manual invocation uses latest balance/positions
    bot.handle_balance_update = fake_handle_balance_update
    await bot.handle_balance_update()
    assert calls["balance"] == 1


@pytest.mark.asyncio
async def test_update_balance_failure_keeps_previous():
    """Test that when fetch_balance raises an exception, previous balance is preserved.

    Per CLAUDE.md: exchange fetch methods should raise exceptions on failure,
    not return False. The exception propagates to the caller who handles it.
    """
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        raise Exception("network error")  # simulate failed fetch

    bot.fetch_balance = fake_fetch_balance

    with pytest.raises(Exception, match="network error"):
        await bot.update_balance()
    # Balance should remain unchanged because assignment was never reached
    assert bot.balance == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)
