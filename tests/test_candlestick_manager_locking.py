from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List

import numpy as np
import pytest

from candlestick_manager import CANDLE_DTYPE, CandlestickManager


class FakeExchange:
    id = "fake"

    def __init__(self):
        self._sleep = 0.02

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params=None,
    ) -> List[List[float]]:
        await asyncio.sleep(self._sleep)
        step = 60_000 if timeframe == "1m" else 60_000
        start = since or int(time.time() * 1000) - step * 5
        lim = limit or 5
        out = []
        for i in range(lim):
            ts = start + i * step
            out.append([ts, 1.0, 1.0, 1.0, 1.0, 1.0])
        return out


def _fake_paginated(
    self, symbol, since_ms, end_exclusive_ms, *, timeframe=None, tf=None, on_batch=None
):
    step = 60_000
    start = since_ms
    end = end_exclusive_ms
    rows = []
    ts = start
    while ts < end:
        rows.append((ts, 1.0, 1.0, 1.0, 1.0, 1.0))
        ts += step
    arr = np.array(rows, dtype=CANDLE_DTYPE)
    if on_batch:
        on_batch(arr)
    return arr


@pytest.mark.asyncio
async def test_concurrent_refresh_no_deadlock(tmp_path):
    cache_dir = tmp_path / "caches"
    symbols = ["BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC", "DOGE/USDC:USDC"]

    cm1 = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(cache_dir),
        default_window_candles=5,
    )
    cm2 = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(cache_dir),
        default_window_candles=5,
    )

    # Monkeypatch fetch to avoid network requests and keep deterministic timing
    async def fake_fetch(
        self, symbol, since_ms, end_exclusive_ms, *, timeframe=None, tf=None, on_batch=None
    ):
        await asyncio.sleep(0.01)
        return _fake_paginated(
            self, symbol, since_ms, end_exclusive_ms, timeframe=timeframe, tf=tf, on_batch=on_batch
        )

    cm1._fetch_ohlcv_paginated = fake_fetch.__get__(cm1, CandlestickManager)
    cm2._fetch_ohlcv_paginated = fake_fetch.__get__(cm2, CandlestickManager)

    async def run_manager(cm: CandlestickManager):
        for sym in symbols:
            await cm.refresh(sym)

    await asyncio.wait_for(asyncio.gather(run_manager(cm1), run_manager(cm2)), timeout=5.0)

    for sym in symbols:
        assert sym in cm1._cache
        assert sym in cm2._cache

    assert cm1._held_fetch_locks == {}
    assert cm2._held_fetch_locks == {}


@pytest.mark.asyncio
async def test_fetch_lock_watchdog_logs_stale_local_holder_without_release(tmp_path, caplog):
    cache_dir = tmp_path / "caches"
    cm = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(cache_dir),
        default_window_candles=5,
    )
    cm._lock_hold_timeout_seconds = 0.01
    cm.debug_level = 1
    key = ("BTC/USDC:USDC", "1m")
    lock_path = cm._fetch_lock_path(*key)

    with caplog.at_level(logging.WARNING, logger="passivbot.candlestick_manager"):
        async with cm._acquire_fetch_lock(*key):
            with open(lock_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            assert payload["pid"]
            assert payload["symbol"] == key[0]
            assert payload["timeframe"] == key[1]

            await asyncio.sleep(0.03)
            assert key in cm._held_fetch_locks

    assert key not in cm._held_fetch_locks
    warning = next(
        record for record in caplog.records if "fetch_lock_hold_timeout" in record.message
    )
    assert "called_by=" in warning.message
    assert "exchange=fake" in warning.message
    assert "symbol=BTC" in warning.message
    assert "timeframe=1m" in warning.message
    assert "owner=pid=" in warning.message
    assert "task=" in warning.message
    assert "attempt=1" in warning.message
    assert "held_s=" in warning.message
    assert "lock_path=" not in warning.message
    assert warning.message.count("exchange=fake") == 1
    assert warning.message.count("symbol=BTC") == 1
    assert warning.message.count("timeframe=1m") == 1
    assert "held_seconds=" not in warning.message
    assert "action=holder_still_active" not in warning.message
    rendered = f"2026-07-15 12:34:56,789 WARNING passivbot.candlestick_manager {warning.message}"
    assert len(rendered) <= 240


def test_read_lockfile_owner_defaults_to_full_context_for_stale_waiting(tmp_path):
    cm = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        default_window_candles=5,
    )
    path = tmp_path / "fetch.lock"
    path.write_text(
        json.dumps(
            {
                "pid": 1234,
                "exchange": "hyperliquid",
                "symbol": "BTC/USDC:USDC",
                "timeframe": "1m",
                "task": "candle-refresh",
                "attempt": 2,
                "acquired_at": time.time() - 1.0,
            }
        ),
        encoding="utf-8",
    )

    owner = cm._read_lockfile_owner(str(path))
    compact_owner = cm._read_lockfile_owner(str(path), compact=True)

    assert "pid=1234" in owner
    assert "exchange=hyperliquid" in owner
    assert "symbol=BTC/USDC:USDC" in owner
    assert "timeframe=1m" in owner
    assert "task=candle-refresh" in owner
    assert "attempt=2" in owner
    assert "held_s=" in owner
    assert "exchange=" not in compact_owner
    assert "symbol=" not in compact_owner
    assert "timeframe=" not in compact_owner


@pytest.mark.asyncio
async def test_fetch_lock_wait_aborts_when_shutdown_requested(tmp_path):
    cache_dir = tmp_path / "caches"
    key = ("BTC/USDC:USDC", "1m")
    holder = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(cache_dir),
        default_window_candles=5,
    )
    checks = 0

    def stop_requested():
        nonlocal checks
        checks += 1
        return checks >= 2

    waiter = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(cache_dir),
        default_window_candles=5,
        stop_requested_callback=stop_requested,
    )

    async with holder._acquire_fetch_lock(*key):
        with pytest.raises(asyncio.CancelledError):
            async with waiter._acquire_fetch_lock(*key):
                pass

    assert checks >= 2
    assert waiter._held_fetch_locks == {}
