"""Tests for live warmup window calculation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from passivbot import compute_live_warmup_windows


def test_compute_live_warmup_windows_forager_toggle():
    symbols_by_side = {"long": {"BTC"}, "short": {"ETH"}}
    values = {
        ("long", "ema_span_0", "BTC"): 1000.0,
        ("long", "ema_span_1", "BTC"): 1500.0,
        ("long", "forager_volume_ema_span_1m", "BTC"): 2000.0,
        ("long", "forager_volatility_ema_span_1m", "BTC"): 1800.0,
        ("long", "entry_volatility_ema_span_1h", "BTC"): 48.0,
        ("short", "ema_span_0", "ETH"): 900.0,
        ("short", "ema_span_1", "ETH"): 800.0,
        ("short", "forager_volume_ema_span_1m", "ETH"): 4000.0,
        ("short", "forager_volatility_ema_span_1m", "ETH"): 3000.0,
        ("short", "entry_volatility_ema_span_1h", "ETH"): 24.0,
    }

    def bp(pside, key, sym):
        return values.get((pside, key, sym), 0.0)

    wins, h1, skip = compute_live_warmup_windows(
        symbols_by_side,
        bp,
        forager_enabled={"long": True, "short": False},
        warmup_ratio=0.5,
    )

    assert wins["BTC"] == 3000
    assert h1["BTC"] == 72
    assert skip["BTC"] is False

    assert wins["ETH"] == 1350
    assert h1["ETH"] == 36
    assert skip["ETH"] is True


def test_compute_live_warmup_windows_override():
    symbols_by_side = {"long": {"BTC", "ETH"}}

    def bp(pside, key, sym):
        return 0.0

    wins, h1, skip = compute_live_warmup_windows(
        symbols_by_side,
        bp,
        window_candles=120,
    )

    assert wins["BTC"] == 120
    assert h1["BTC"] == 2
    assert skip["BTC"] is True
    assert wins["ETH"] == 120


def test_compute_live_warmup_windows_uses_v8_strategy_and_forager_groups():
    symbols_by_side = {"long": {"HYPE"}}

    def bp(pside, key, sym):
        return 0.0

    def strategy(pside, key, sym):
        values = {
            "ema_span_0": 110.0,
            "ema_span_1": 260.0,
            "volatility_ema_span_1m": 86.0,
            "volatility_ema_span_1h": 1785.0,
        }
        return values.get(key, 0.0)

    def forager(pside, key, sym):
        values = {
            "forager_volume_ema_span_1m": 760.0,
            "forager_volatility_ema_span_1m": 110.0,
        }
        return values.get(key, 0.0)

    wins, h1, skip = compute_live_warmup_windows(
        symbols_by_side,
        bp,
        forager_enabled={"long": True},
        strategy_lookup=strategy,
        forager_lookup=forager,
        warmup_ratio=0.3,
    )

    assert wins["HYPE"] == 988
    assert h1["HYPE"] == 2321
    assert skip["HYPE"] is True
