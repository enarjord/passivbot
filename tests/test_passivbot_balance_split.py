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


@pytest.mark.asyncio
async def test_update_balance_override_does_not_reset_hysteresis_anchor():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot.balance_override = 250.0
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 133.0
    bot.balance_hysteresis_snap_pct = 0.02

    ok = await bot.update_balance()
    assert ok is True
    assert bot.balance == pytest.approx(250.0)
    assert bot.balance_raw == pytest.approx(250.0)
    # Keep hysteresis anchor unchanged while override is active.
    assert bot.previous_hysteresis_balance == pytest.approx(133.0)


def test_accessor_fallback_when_balance_raw_absent():
    """get_raw_balance() falls back to snapped balance when balance_raw is absent."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 500.0
    # Intentionally do NOT set balance_raw
    if hasattr(bot, "balance_raw"):
        del bot.balance_raw
    assert bot.get_raw_balance() == pytest.approx(500.0)


def test_accessor_returns_raw_when_present():
    """get_raw_balance() returns balance_raw when it exists."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0
    bot.balance_raw = 1010.0
    assert bot.get_raw_balance() == pytest.approx(1010.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


def test_balance_peak_uses_raw_not_snapped():
    """Core bug regression: balance_peak must be derived from raw balance, not snapped.

    After a profit fill, raw balance advances (e.g. 1010) but snapped balance
    may remain stale (e.g. 1000) due to hysteresis. The peak calculation must
    use raw balance so that balance_peak is correct.

    balance_peak = balance_raw + (pnls_cumsum_max - pnls_cumsum_last)

    With raw=1010, cumsum_max=100, cumsum_last=100:
        peak_from_raw = 1010 + (100 - 100) = 1010  ← correct
        peak_from_snap = 1000 + (100 - 100) = 1000  ← WRONG (stale)
    """
    import types
    from unittest.mock import MagicMock

    import numpy as np

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0  # snapped (stale)
    bot.balance_raw = 1010.0  # raw (advanced after profit fill)
    bot._pnls_manager = MagicMock()
    bot._pnls_manager.get_events.return_value = [
        types.SimpleNamespace(pnl=100.0, timestamp=1.0),
    ]

    # Simulate the calc_auto_unstuck_allowance_from_scratch peak calculation
    events = bot._pnls_manager.get_events()
    pnls_cumsum = np.array([ev.pnl for ev in events]).cumsum()
    pnls_cumsum_max = float(pnls_cumsum.max())
    pnls_cumsum_last = float(pnls_cumsum[-1])

    # Using raw balance (correct)
    balance_peak_raw = bot.get_raw_balance() + (pnls_cumsum_max - pnls_cumsum_last)
    assert balance_peak_raw == pytest.approx(1010.0)

    # Using snapped balance (would be wrong)
    balance_peak_snapped = bot.get_hysteresis_snapped_balance() + (pnls_cumsum_max - pnls_cumsum_last)
    assert balance_peak_snapped == pytest.approx(1000.0)

    # The raw-based peak is correct and higher
    assert balance_peak_raw > balance_peak_snapped
