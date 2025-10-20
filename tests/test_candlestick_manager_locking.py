from __future__ import annotations

import asyncio
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
