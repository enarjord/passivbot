import asyncio
import os
import time
import math
import json
import zlib
import pytest
import numpy as np
from pathlib import Path

from candlestick_manager import (
    CandlestickManager,
    CANDLE_DTYPE,
    ONE_MIN_MS,
    _floor_minute,
)


@pytest.mark.parametrize("debug", [False])
def test_standardize_gaps_inserts_zero_candles(tmp_path, debug):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    # create two candles with a one-minute gap between them
    base = int(time.time() * 1000)
    t0 = _floor_minute(base) - 3 * ONE_MIN_MS
    t1 = t0 + ONE_MIN_MS
    t2 = t0 + 2 * ONE_MIN_MS  # gap: missing t0+1*ONE_MIN_MS
    a = np.array(
        [
            (t0, 100.0, 105.0, 99.0, 102.0, 1.0),
            (t2, 103.0, 104.0, 100.0, 101.0, 0.5),
        ],
        dtype=CANDLE_DTYPE,
    )
    res = cm.standardize_gaps(a, start_ts=t0, end_ts=t2, strict=False)
    # expect three candles: t0, t1 (synthesized), t2
    assert res.shape[0] == 3
    assert int(res[0]["ts"]) == t0
    assert int(res[1]["ts"]) == t1
    assert int(res[2]["ts"]) == t2
    # synthesized middle candle should have bv == 0 and c equal to previous close (102.0)
    assert float(res[1]["bv"]) == 0.0
    assert math.isclose(float(res[1]["c"]), 102.0, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_get_candles_range_and_inclusive(tmp_path):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    base = _floor_minute(int(time.time() * 1000)) - 10 * ONE_MIN_MS
    # create 6 candles
    arr = []
    for i in range(6):
        ts = base + i * ONE_MIN_MS
        arr.append((ts, 1.0 + i, 1.0 + i, 1.0 + i, 1.0 + i, float(i)))
    arr = np.array(arr, dtype=CANDLE_DTYPE)
    symbol = "FOO/USDT"
    cm._cache[symbol] = arr
    start = base + ONE_MIN_MS
    end = base + 3 * ONE_MIN_MS
    res = await cm.get_candles(symbol, start_ts=start, end_ts=end, max_age_ms=0)
    # should return minutes: start, start+1, end -> 3 entries
    assert res.shape[0] == 3
    assert list(res["ts"]) == [start, start + ONE_MIN_MS, end]


@pytest.mark.asyncio
async def test_get_latest_ema_close_correctness(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    # create 5 candles closes: 10,11,12,13,14
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    base = fixed_now_ms - 5 * ONE_MIN_MS
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    arr = []
    for i, c in enumerate(closes):
        ts = base + i * ONE_MIN_MS
        arr.append((ts, c, c, c, c, 1.0))
    arr = np.array(arr, dtype=CANDLE_DTYPE)
    symbol = "BAR/USDT"
    cm._cache[symbol] = arr
    span = 5
    ema = await cm.get_latest_ema_close(symbol, span)
    # compute expected EMA manually
    alpha = 2.0 / (span + 1.0)
    expected = closes[0]
    for v in closes[1:]:
        expected = alpha * v + (1 - alpha) * expected
    assert pytest.approx(expected, rel=1e-9) == ema


@pytest.mark.asyncio
async def test_get_candles_negative_max_age_raises(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "Z/USDT"
    with pytest.raises(ValueError):
        await cm.get_candles(symbol, max_age_ms=-1)


@pytest.mark.asyncio
async def test_warmup_since_calls_refresh(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    called = []

    async def fake_refresh(symbol, through_ts=None):
        called.append((symbol, through_ts))

    monkeypatch.setattr(cm, "refresh", fake_refresh)
    symbols = ["A/USDT", "B/USDT"]
    await cm.warmup_since(symbols, since_ts=0)
    assert len(called) == len(symbols)
    assert {c[0] for c in called} == set(symbols)


def test_save_shard_writes_index_and_shard(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "SAVE/USDT"
    ts = _floor_minute(int(time.time() * 1000))
    arr = np.array([(ts, 1.0, 2.0, 0.5, 1.5, 0.3)], dtype=CANDLE_DTYPE)
    # date_key as YYYY-MM-DD
    date_key = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000.0))
    cm._save_shard(symbol, date_key, arr)
    shard_path = cm._shard_path(symbol, date_key)
    assert os.path.exists(shard_path)
    idx = cm._index[f"{symbol}::1m"]
    assert date_key in idx["shards"]
    info = idx["shards"][date_key]
    assert "crc32" in info
    assert info["min_ts"] == int(arr[0]["ts"]) and info["max_ts"] == int(arr[0]["ts"])

    # Also verify 1h persistence path when timeframe provided
    cm._save_shard(symbol, date_key, arr, timeframe="1h")
    shard_path_1h = cm._shard_path(symbol, date_key, timeframe="1h")
    assert os.path.exists(shard_path_1h)
    idx_1h = cm._index[f"{symbol}::1h"]
    assert date_key in idx_1h["shards"]


@pytest.mark.asyncio
async def test_zero_candles_not_persisted(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "NOP/USDT"
    # empty cache
    res = await cm.get_candles(symbol, start_ts=0, end_ts=ONE_MIN_MS * 2, max_age_ms=0)
    # no shard files should be created for symbol
    symbol_dir = Path(cm._symbol_dir(symbol, timeframe="1m"))
    assert not symbol_dir.exists() or not any(symbol_dir.rglob("*.npy"))


@pytest.mark.asyncio
async def test_tf_persistence_via_get_candles(tmp_path, monkeypatch):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TFP/USDT"
    base = _floor_minute(int(time.time() * 1000)) - 6 * ONE_MIN_MS * 60
    # Monkeypatch fetcher to simulate 1h candles
    period = 60 * ONE_MIN_MS

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        s = int(since_ms)
        e = int(end_exclusive_ms)
        ts = list(range(s, e, period))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            arr["o"] = 1.0
            arr["h"] = 2.0
            arr["l"] = 0.5
            arr["c"] = 1.5
            arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)
    start_ts = base
    end_ts = base + 5 * period
    out = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, timeframe="1h", strict=True)
    assert out.size > 0
    # Verify 1h shard saved
    date_key = time.strftime("%Y-%m-%d", time.gmtime(start_ts / 1000.0))
    shard_path = cm._shard_path(symbol, date_key, timeframe="1h")
    assert os.path.exists(shard_path)
    # Index for 1h present
    assert f"{symbol}::1h" in cm._index


@pytest.mark.asyncio
async def test_tf_loads_from_disk_without_network(tmp_path, monkeypatch):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "REUSE/USDT"
    # Prepare 1h on-disk by first networked call
    base = _floor_minute(int(time.time() * 1000)) - 6 * ONE_MIN_MS * 60
    period = 60 * ONE_MIN_MS

    net_calls = {"n": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        net_calls["n"] += 1
        s = int(since_ms)
        e = int(end_exclusive_ms)
        ts = list(range(s, e, period))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            arr["o"] = 1.0
            arr["h"] = 2.0
            arr["l"] = 0.5
            arr["c"] = 1.5
            arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)
    start_ts = base
    end_ts = base + 5 * period
    out1 = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, timeframe="1h", strict=True)
    assert out1.size > 0
    first_calls = net_calls["n"]
    assert first_calls >= 1

    # Clear TF LRU cache to force disk path on second call
    cm._tf_range_cache.clear()

    out2 = await cm.get_candles(
        symbol, start_ts=start_ts, end_ts=end_ts, timeframe="1h", strict=True, max_age_ms=600_000
    )
    assert out2.size == out1.size
    # Should not perform any new network calls; served from disk
    assert net_calls["n"] == first_calls


# EOF
@pytest.mark.asyncio
async def test_concurrent_requests_share_fetch(tmp_path, monkeypatch):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "LOCK/USDT"
    start_ts = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    end_ts = start_ts + 4 * ONE_MIN_MS

    calls = {"count": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        calls["count"] += 1
        await asyncio.sleep(0.05)
        ts = list(range(int(since_ms), int(end_exclusive_ms), ONE_MIN_MS))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            arr["o"] = 1.0
            arr["h"] = 2.0
            arr["l"] = 0.5
            arr["c"] = 1.5
            arr["bv"] = 1.0
        if on_batch is not None:
            on_batch(arr)
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    async def one_call():
        return await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=0)

    out1, out2 = await asyncio.gather(one_call(), one_call())
    assert out1.size > 0 and out2.size > 0
    assert calls["count"] == 1


# EOF
@pytest.mark.asyncio
async def test_tf_range_cache_reuse_within_ttl(monkeypatch, tmp_path):
    # Fixed now for deterministic bucket alignment
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    # Dummy exchange id present to enable tf fetch path
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))

    tf = "1h"
    period = 60 * ONE_MIN_MS
    span = 5
    symbol = "BTC/USDT:USDT"

    calls = {"fetch": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["fetch"] += 1
        # Generate hourly candles aligned to tf
        s = int(since_ms)
        e = int(end_exclusive_ms)
        ts = list(range(s, e, period))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            # trivial ohlcv
            arr["o"] = 1.0
            arr["h"] = 2.0
            arr["l"] = 0.5
            arr["c"] = 1.5
            arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    # First: compute series -> should fetch once
    ser = await cm.get_ema_volume_series(symbol, span=span, timeframe=tf, max_age_ms=60_000)
    assert ser.size > 0
    assert calls["fetch"] == 1

    # Second: compute different metric latest, same tf and span -> reuse tf range cache, no extra fetch
    val = await cm.get_latest_ema_log_range(symbol, span=span, timeframe=tf, max_age_ms=60_000)
    assert isinstance(val, float)
    assert calls["fetch"] == 1


@pytest.mark.asyncio
async def test_get_candles_1m_avoids_refetch_after_sharding(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "AVAX/USDT:USDT"

    calls = {"fetch": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["fetch"] += 1
        s = int(since_ms)
        e = int(end_exclusive_ms)
        ts = list(range(s, e, ONE_MIN_MS))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            arr["o"] = 10.0
            arr["h"] = 11.0
            arr["l"] = 9.0
            arr["c"] = 10.5
            arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    # Range covering two hours ending one minute before now
    end_final = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    start_ts = end_final - ONE_MIN_MS * 120

    # First call fetches and writes shards
    arr1 = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_final, strict=True)
    assert arr1.size > 0
    assert calls["fetch"] == 1

    # Drop memory to force disk load path
    cm._cache.pop(symbol, None)

    # Second call for same range should load from shards, not fetch again
    arr2 = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_final, strict=True)
    assert arr2.size > 0
    assert calls["fetch"] == 1


@pytest.mark.asyncio
async def test_get_current_close_uses_ttl(monkeypatch):
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    call_counter = {"ticker": 0}

    class _Ex:
        id = "okx"

        async def fetch_ticker(self, symbol):
            call_counter["ticker"] += 1
            return {"last": 123.45}

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx")
    symbol = "BTC/USDT:USDT"

    # First call fetches ticker
    p1 = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert p1 == pytest.approx(123.45)
    assert call_counter["ticker"] == 1


@pytest.mark.asyncio
async def test_get_current_close_primes_ttl_for_candles(monkeypatch, tmp_path):
    # Fixed now for deterministic minute boundaries
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"

    # Track whether network pagination is called by get_candles
    calls = {"paginated": 0}

    async def fake_paginated(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["paginated"] += 1
        return np.empty((0,), dtype=CANDLE_DTYPE)

    # Return a single current-minute candle via low-level OHLCV fetch used by get_current_close
    async def fake_once(symbol_, since_ms, limit, end_exclusive_ms=None, timeframe=None):
        ts = int((fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS)
        return [[ts, 1.0, 1.0, 1.0, 1.23, 1.0]]

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_paginated)
    monkeypatch.setattr(cm, "_ccxt_fetch_ohlcv_once", fake_once)

    # 1) Call get_current_close: this should fetch/merge current-minute candle and update last_refresh_ms
    p = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert p == pytest.approx(1.23)

    # 2) Call get_candles ending at latest finalized minute with TTL: should NOT call _fetch_ohlcv_paginated
    end_finalized = (fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS - ONE_MIN_MS
    start_ts = end_finalized - ONE_MIN_MS * 10
    out = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_finalized, max_age_ms=60_000)
    assert isinstance(out, np.ndarray)
    assert calls["paginated"] == 0

    # No additional network calls expected here; TTL should prevent refresh


@pytest.mark.asyncio
async def test_get_current_close_tail_fetch_merges_and_primes(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    # Use small overlap to keep payloads small
    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"), overlap_candles=5
    )
    symbol = "ETH/USDT:USDT"

    # Count paginated fetches triggered by get_candles
    calls = {"paginated": 0}

    async def fake_paginated(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["paginated"] += 1
        return np.empty((0,), dtype=CANDLE_DTYPE)

    # Capture args and return 5 recent 1m candles including current minute
    called = {}

    async def fake_once(symbol_, since_ms, limit, end_exclusive_ms=None, timeframe=None):
        called["since_ms"] = int(since_ms)
        called["limit"] = int(limit)
        end_current = (fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS
        # build a sequence from since_ms to end_current inclusive, step 1m
        ts = list(range(int(since_ms), int(end_current) + ONE_MIN_MS, ONE_MIN_MS))
        arr = []
        for i, t in enumerate(ts):
            # close sequence i + 1.0 to verify values present
            c = float(i + 1)
            arr.append([t, c, c, c, c, 1.0])
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_paginated)
    monkeypatch.setattr(cm, "_ccxt_fetch_ohlcv_once", fake_once)

    # Trigger tail fetch via get_current_close
    p = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert isinstance(p, float)
    # Ensure we asked for overlap_candles candles
    assert called.get("limit") == 5
    # Ensure since_ms aligns to end_current - 4 minutes
    end_current = (fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS
    assert called.get("since_ms") == end_current - ONE_MIN_MS * 4

    # Cache should now contain at least those 5 candles including current minute
    arr = cm._cache.get(symbol)
    assert arr is not None and arr.size >= 5
    arr = np.sort(arr, order="ts")
    assert int(arr[-1]["ts"]) == end_current

    # A subsequent get_candles within TTL should not paginate
    end_finalized = end_current - ONE_MIN_MS
    start_ts = end_finalized - ONE_MIN_MS * 2
    out = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_finalized, max_age_ms=60_000)
    assert out.size > 0
    assert calls["paginated"] == 0
