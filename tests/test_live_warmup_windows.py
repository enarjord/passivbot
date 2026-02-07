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
        ("long", "filter_volume_ema_span", "BTC"): 2000.0,
        ("long", "filter_volatility_ema_span", "BTC"): 1800.0,
        ("long", "entry_volatility_ema_span_hours", "BTC"): 48.0,
        ("short", "ema_span_0", "ETH"): 900.0,
        ("short", "ema_span_1", "ETH"): 800.0,
        ("short", "filter_volume_ema_span", "ETH"): 4000.0,
        ("short", "filter_volatility_ema_span", "ETH"): 3000.0,
        ("short", "entry_volatility_ema_span_hours", "ETH"): 24.0,
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
