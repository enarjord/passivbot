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
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
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
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
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
    idx = cm._index[symbol]
    assert date_key in idx["shards"]
    info = idx["shards"][date_key]
    assert "crc32" in info
    assert info["min_ts"] == int(arr[0]["ts"]) and info["max_ts"] == int(arr[0]["ts"])


@pytest.mark.asyncio
async def test_zero_candles_not_persisted(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "NOP/USDT"
    # empty cache
    res = await cm.get_candles(symbol, start_ts=0, end_ts=ONE_MIN_MS * 2, max_age_ms=0)
    # no shard files should be created for symbol
    symbol_dir = Path(cm._symbol_dir(symbol))
    assert not symbol_dir.exists() or not any(symbol_dir.rglob("*.npy"))


# EOF
