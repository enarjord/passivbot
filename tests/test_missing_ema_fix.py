"""Tests for MissingEma fix: EMA paths and error handling."""

import math
import time
import pytest
import numpy as np

from candlestick_manager import (
    CandlestickManager,
    CANDLE_DTYPE,
    ONE_MIN_MS,
    _floor_minute,
)

ONE_HOUR_MS = 3_600_000


def _make_cm(tmp_path):
    return CandlestickManager(
        exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches")
    )


# ---------------------------------------------------------------------------
# get_latest_ema_close with leading gaps → finite result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_ema_close_with_leading_gaps(tmp_path, monkeypatch):
    """When leading candles are missing, EMA should still return a finite value."""
    cm = _make_cm(tmp_path)
    fixed_now_ms = 1_725_590_400_000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    span = 10
    # Create only the last 3 candles out of the 10 expected
    base = fixed_now_ms - span * ONE_MIN_MS
    arr = []
    for i in range(span - 3, span):
        ts = base + i * ONE_MIN_MS
        arr.append((ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1.0))
    arr = np.array(arr, dtype=CANDLE_DTYPE)
    symbol = "SPARSE/USDT"
    cm._cache[symbol] = arr

    ema = await cm.get_latest_ema_close(symbol, float(span))
    assert math.isfinite(ema), f"EMA should be finite, got {ema}"
    assert ema > 0


@pytest.mark.asyncio
async def test_get_latest_ema_close_no_candles_returns_nan(tmp_path, monkeypatch):
    """When there are zero candles, EMA should return NaN (not crash)."""
    cm = _make_cm(tmp_path)
    fixed_now_ms = 1_725_590_400_000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    symbol = "EMPTY/USDT"
    cm._cache[symbol] = np.empty((0,), dtype=CANDLE_DTYPE)

    ema = await cm.get_latest_ema_close(symbol, 5.0)
    assert math.isnan(ema)


# ---------------------------------------------------------------------------
# MissingEma graceful handling in passivbot.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_ema_raises_from_snapshot(monkeypatch):
    """MissingEma in calc_ideal_orders_orchestrator_from_snapshot re-raises."""
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    class FakeBot:
        positions = {}
        balance = 1000.0
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

        def get_raw_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

        def get_hysteresis_snapped_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
    }

    def fake_compute(json_str):
        raise Exception("MissingEma { symbol_idx: 0 }")

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    with pytest.raises(Exception, match="MissingEma"):
        await method(bot, snapshot, return_snapshot=False)


@pytest.mark.asyncio
async def test_missing_ema_raises_from_snapshot_with_return(monkeypatch):
    """MissingEma with return_snapshot=True also re-raises."""
    try:
        import passivbot as pb_mod
    except ImportError:
        pytest.skip("passivbot module not importable in test environment")

    class FakeBot:
        positions = {}
        balance = 1000.0
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

        def get_raw_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

        def get_hysteresis_snapped_balance(self):
            return float(getattr(self, "balance", 0.0) or 0.0)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
    }

    def fake_compute(json_str):
        raise Exception("MissingEma { symbol_idx: 0 }")

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    with pytest.raises(Exception, match="MissingEma"):
        await method(bot, snapshot, return_snapshot=True)
