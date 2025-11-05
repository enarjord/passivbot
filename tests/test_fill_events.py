import json
import sys
from pathlib import Path
from types import MethodType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from passivbot import Passivbot


async def _noop_init(*_args, **_kwargs):
    return None


def _make_bot(tmp_path: Path, initial_events):
    bot = Passivbot.__new__(Passivbot)
    bot.stop_signal_received = False
    bot.quote = "USDT"
    bot.fill_events = [dict(evt) for evt in initial_events]
    bot._fill_event_fingerprints = {}
    bot.fill_events_loaded = True
    bot.fill_events_cache_path = tmp_path / "fill_events.json"
    bot.live_value = MethodType(lambda self, key: 30 if key == "pnls_max_lookback_days" else 0, bot)
    max_ts = max((evt["timestamp"] for evt in initial_events), default=0)
    exchange_time = max_ts + 1_000 if max_ts else 1_800_000_000_000
    bot.get_exchange_time = MethodType(lambda self: exchange_time, bot)
    bot.init_fill_events = MethodType(_noop_init, bot)
    for evt in bot.fill_events:
        fp = bot._fingerprint_event(evt)
        bot._fill_event_fingerprints.setdefault(evt["id"], set()).add(fp)
    if bot.fill_events:
        with open(bot.fill_events_cache_path, "w") as fh:
            json.dump(bot.fill_events, fh)
    return bot


@pytest.mark.asyncio
async def test_update_fill_events_replaces_cached_event(tmp_path):
    cached_evt = {
        "id": "id-1",
        "timestamp": 1_762_094_827_318,
        "symbol": "ZEC/USDT:USDT",
        "side": "buy",
        "qty": 1.595,
        "price": 378.93,
        "pnl": 0.0,
        "position_side": "long",
        "pb_order_type": "entry_grid_normal_long",
    }
    bot = _make_bot(tmp_path, [cached_evt])

    fresh_evt = dict(cached_evt)
    fresh_evt["qty"] = 0.145

    async def fetch_fill_events(self, start_time=None, end_time=None, limit=None):
        return [fresh_evt]

    bot.fetch_fill_events = MethodType(fetch_fill_events, bot)

    await bot.update_fill_events()

    assert len(bot.fill_events) == 1
    updated = bot.fill_events[0]
    assert updated["id"] == cached_evt["id"]
    assert updated["qty"] == pytest.approx(fresh_evt["qty"])
    assert updated["price"] == pytest.approx(fresh_evt["price"])


@pytest.mark.asyncio
async def test_update_fill_events_merges_multiple_updates(tmp_path):
    cached_evt = {
        "id": "id-2",
        "timestamp": 1_762_097_960_854,
        "symbol": "TAO/USDT:USDT",
        "side": "buy",
        "qty": 0.07,
        "price": 493.0,
        "pnl": 0.0,
        "position_side": "long",
        "pb_order_type": "entry_initial_normal_long",
    }
    bot = _make_bot(tmp_path, [cached_evt])

    fresh_events = [
        dict(cached_evt, qty=0.02, price=492.5),
        dict(cached_evt, qty=0.03, price=493.5),
    ]

    async def fetch_fill_events(self, start_time=None, end_time=None, limit=None):
        return fresh_events

    bot.fetch_fill_events = MethodType(fetch_fill_events, bot)

    await bot.update_fill_events()

    assert len(bot.fill_events) == 1
    updated = bot.fill_events[0]
    assert updated["id"] == cached_evt["id"]
    assert updated["qty"] == pytest.approx(0.05)
    expected_price = (0.02 * 492.5 + 0.03 * 493.5) / 0.05
    assert updated["price"] == pytest.approx(expected_price)
