import json
import logging
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


@pytest.mark.asyncio
async def test_update_balance_nan_keeps_previous_and_logs_warning(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        return float("nan")

    bot.fetch_balance = fake_fetch_balance

    with caplog.at_level(logging.WARNING):
        ok = await bot.update_balance()
    assert ok is False
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)
    assert "non-finite balance fetch result; keeping previous balance" in caplog.text


@pytest.mark.asyncio
async def test_update_balance_inf_keeps_previous_and_logs_warning(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        return float("inf")

    bot.fetch_balance = fake_fetch_balance

    with caplog.at_level(logging.WARNING):
        ok = await bot.update_balance()
    assert ok is False
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)
    assert "non-finite balance fetch result; keeping previous balance" in caplog.text


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


def test_effective_min_cost_filter_uses_snapped_balance():
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 10.0
    bot.effective_min_cost = {"BTC/USDT:USDT": 40.0}
    bot.live_value = lambda key: key == "filter_by_min_effective_cost"
    bot.get_wallet_exposure_limit = lambda pside, symbol=None: 1.0
    bot.bp = lambda pside, key, symbol=None: 0.5 if key == "entry_initial_qty_pct" else 0.0

    # Passes only when snapped balance is used:
    # 100 * 1.0 * 0.5 = 50 >= 40; raw path would fail (10 * 1.0 * 0.5 = 5).
    assert bot.effective_min_cost_is_low_enough("long", "BTC/USDT:USDT") is True


def test_unstuck_allowance_routes_raw_balance_to_rust(monkeypatch):
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 200.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=10.0), types.SimpleNamespace(pnl=-4.0)]
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.2 if pside == "long" else 0.0
        if key == "total_wallet_exposure_limit":
            return 0.5
        return 0.0

    bot.bot_value = bot_value

    calls = []

    def fake_calc_auto_unstuck_allowance(balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last):
        calls.append((balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last))
        return 123.45

    monkeypatch.setattr(pb_mod.pbr, "calc_auto_unstuck_allowance", fake_calc_auto_unstuck_allowance)

    out = bot._calc_unstuck_allowances(allow_new_unstuck=True)

    assert out["long"] == pytest.approx(123.45)
    assert out["short"] == pytest.approx(0.0)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(200.0)  # raw balance


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_routes_split_balances(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        positions = {}
        balance = 120.0  # snapped
        balance_raw = 175.0  # raw
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(bot, snapshot, return_snapshot=False)

    assert captured["input"]["balance"] == pytest.approx(120.0)
    assert captured["input"]["balance_raw"] == pytest.approx(175.0)


def test_unstuck_logging_peak_stays_stable_when_profit_updates_both_balance_and_pnl():
    """Regression for peak drift: peak must not decay when profits increase pnl_last and balance_raw."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0  # stale snapped value to simulate hysteresis lag

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.0
        if key == "unstuck_loss_allowance_pct":
            return 0.1
        return 0.0

    bot.bot_value = bot_value

    # State A: peak = 100 + (100 - 0) = 200
    bot.balance_raw = 100.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=100.0), types.SimpleNamespace(pnl=-100.0)]
    )
    info_a = bot._calc_unstuck_allowance_for_logging("long")

    # State B (after net +50 realized since trough): peak should still be 200
    # If snapped balance were incorrectly used here, peak would drift down to 150.
    bot.balance_raw = 150.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=100.0), types.SimpleNamespace(pnl=-50.0)]
    )
    info_b = bot._calc_unstuck_allowance_for_logging("long")

    assert info_a["status"] == "ok"
    assert info_b["status"] == "ok"
    assert info_a["peak"] == pytest.approx(200.0)
    assert info_b["peak"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_update_balance_hysteresis_divergence():
    """Calls update_balance() twice with values within snap threshold, verifies balance_raw != balance."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        # First call: 100.0, second call: 100.5 (within 2% snap threshold)
        return 100.0 if call_count[0] == 1 else 100.5

    bot.fetch_balance = fake_fetch_balance

    ok1 = await bot.update_balance()
    assert ok1 is True
    # After first call, both should be the same
    assert bot.balance_raw == pytest.approx(100.0)
    assert bot.balance == pytest.approx(100.0)

    ok2 = await bot.update_balance()
    assert ok2 is True
    # Raw should update to 100.5, but snapped stays at 100.0 (within 2% threshold)
    assert bot.balance_raw == pytest.approx(100.5)
    assert bot.balance == pytest.approx(100.0)
    # Verify divergence
    assert bot.balance_raw != bot.balance


@pytest.mark.asyncio
async def test_update_balance_divergence_routes_to_orchestrator_input():
    """After creating divergence, builds orchestrator input dict and asserts both values pass through."""
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0 if call_count[0] == 1 else 1005.0

    bot.fetch_balance = fake_fetch_balance

    await bot.update_balance()
    await bot.update_balance()

    # Verify divergence exists
    assert bot.balance_raw == pytest.approx(1005.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now verify get_raw_balance / get_hysteresis_snapped_balance route correctly
    assert bot.get_raw_balance() == pytest.approx(1005.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_ws_balance_update_sets_balance_raw_correctly():
    """Simulate a WS-triggered balance update and verify balance_raw is set."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    # First establish a baseline via REST update_balance
    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0

    bot.fetch_balance = fake_fetch_balance
    await bot.update_balance()

    assert bot.balance_raw == pytest.approx(1000.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now simulate a WS balance update: exchange pushes a new raw balance
    # directly onto the bot attributes (as exchange-specific subclasses do).
    bot.balance_raw = 1002.0
    # Snapped balance stays at 1000.0 since WS doesn't re-run hysteresis

    # Verify handle_balance_update sees the divergence and schedules execution
    async def fake_calc_upnl_sum():
        return 50.0

    bot.calc_upnl_sum = fake_calc_upnl_sum
    bot.execution_scheduled = False

    await bot.handle_balance_update(source="WS")

    # After WS update, the raw accessor should return the new value
    assert bot.get_raw_balance() == pytest.approx(1002.0)
    # Snapped balance hasn't changed (no hysteresis re-snap via WS path)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)
    # Execution should have been scheduled due to balance_raw change
    assert bot.execution_scheduled is True
