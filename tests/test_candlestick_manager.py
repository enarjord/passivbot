import asyncio
import logging
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
    GAP_REASON_NO_ARCHIVE,
    ONE_MIN_MS,
    _GAP_MAX_RETRIES,
    _GATEIO_RECENT_1M_LIMIT_CANDLES,
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


def test_kucoin_synthetic_batch_summary_is_info_not_warning(tmp_path, caplog):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches"))
    cm.start_synth_candle_batch()
    cm._synth_candle_batch["ILLQ/USDT:USDT"] = {
        "count": 5000,
        "min_ts": 1725590400000,
        "max_ts": 1725890340000,
    }

    caplog.set_level(logging.INFO, logger=cm.log.name)
    cm.flush_synth_candle_batch()

    records = [rec for rec in caplog.records if "synthesized 5000 zero-candles" in rec.getMessage()]
    assert records
    assert records[0].levelno == logging.INFO
    assert "expected on sparse KuCoin no-trade minutes" in records[0].getMessage()


def test_non_kucoin_large_synthetic_batch_summary_stays_warning(tmp_path, caplog):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    cm.start_synth_candle_batch()
    cm._synth_candle_batch["ILLQ/USDT:USDT"] = {
        "count": 5000,
        "min_ts": 1725590400000,
        "max_ts": 1725890340000,
    }

    caplog.set_level(logging.INFO, logger=cm.log.name)
    cm.flush_synth_candle_batch()

    records = [rec for rec in caplog.records if "synthesized 5000 zero-candles" in rec.getMessage()]
    assert records
    assert records[0].levelno == logging.WARNING


def test_candle_manager_hides_high_volume_cache_debug_below_trace(tmp_path, caplog):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    cm.debug_level = 2
    caplog.set_level(logging.DEBUG, logger=cm.log.name)

    cm._log("debug", "index_cached", symbol="BTC/USDT:USDT", timeframe="1m", mtime=1.0)
    cm._log("debug", "ccxt_fetch_ohlcv", symbol="BTC/USDT:USDT", tf="1m")
    cm._log("debug", "saved_range", symbol="BTC/USDT:USDT", rows=10)

    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("event=index_cached" in msg for msg in messages)
    assert not any("event=ccxt_fetch_ohlcv" in msg for msg in messages)
    assert any("event=saved_range" in msg for msg in messages)


def test_candle_manager_emits_high_volume_cache_debug_at_trace(tmp_path, caplog):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    cm.debug_level = 3
    caplog.set_level(int(getattr(logging, "TRACE", 5)), logger=cm.log.name)

    cm._log("debug", "index_cached", symbol="BTC/USDT:USDT", timeframe="1m", mtime=1.0)

    records = [rec for rec in caplog.records if "event=index_cached" in rec.getMessage()]
    assert records
    assert records[0].levelno == int(getattr(logging, "TRACE", 5))


@pytest.mark.asyncio
async def test_get_candles_aborts_when_stop_requested(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
        stop_requested_callback=lambda: True,
    )

    with pytest.raises(asyncio.CancelledError):
        await cm.get_candles("FOO/USDT")


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


def test_persist_batch_observer_receives_saved_batch(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "OBS/USDT"
    ts = _floor_minute(int(time.time() * 1000))
    arr = np.array([(ts, 1.0, 2.0, 0.5, 1.5, 0.3)], dtype=CANDLE_DTYPE)
    observed = []

    def observer(observed_symbol, timeframe, batch):
        observed.append((observed_symbol, timeframe, batch.copy()))

    cm.set_persist_batch_observer(observer)
    cm._persist_batch(symbol, arr, timeframe="1m")

    assert len(observed) == 1
    observed_symbol, timeframe, batch = observed[0]
    assert observed_symbol == symbol
    assert timeframe == "1m"
    assert np.array_equal(batch, arr)


def test_rebuild_index_for_range_updates_and_prunes(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "REBUILD/USDT"
    base = 1725590400000  # 2024-09-06 00:00:00 UTC
    date_key0 = cm._date_key(base)
    date_key1 = cm._date_key(base + 24 * 60 * 60 * 1000)
    day0_start, day0_end = cm._date_range_of_key(date_key0)
    day1_start, day1_end = cm._date_range_of_key(date_key1)

    # Create a real shard for day0 with minimal data
    arr = np.array(
        [
            (day0_start, 1.0, 2.0, 0.5, 1.5, 0.1),
            (day0_start + ONE_MIN_MS, 1.1, 2.1, 0.6, 1.6, 0.2),
        ],
        dtype=CANDLE_DTYPE,
    )
    shard_path0 = cm._shard_path(symbol, date_key0)
    os.makedirs(os.path.dirname(shard_path0), exist_ok=True)
    np.save(shard_path0, arr)

    # Write a corrupted index: wrong metadata + a missing shard entry + future last_refresh
    idx_path = cm._index_path(symbol, timeframe="1m")
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    future_refresh = int(time.time() * 1000) + 10 * ONE_MIN_MS
    bad_idx = {
        "shards": {
            date_key0: {
                "path": shard_path0,
                "min_ts": 0,
                "max_ts": 0,
                "count": 0,
                "crc32": 0,
            },
            date_key1: {
                "path": cm._shard_path(symbol, date_key1),
                "min_ts": 0,
                "max_ts": 0,
                "count": 0,
                "crc32": 0,
            },
        },
        "meta": {"last_refresh_ms": future_refresh},
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(bad_idx, f)

    res = cm.rebuild_index_for_range(symbol, day0_start, day1_end, timeframe="1m", log_level="debug")
    idx = cm._ensure_symbol_index(symbol, tf="1m")

    assert date_key0 in idx["shards"]
    assert date_key1 not in idx["shards"]
    info = idx["shards"][date_key0]
    assert info["count"] == int(arr.shape[0])
    assert info["min_ts"] == int(arr[0]["ts"])
    assert info["max_ts"] == int(arr[-1]["ts"])
    assert idx["meta"]["last_refresh_ms"] == 0
    assert res["updated"] >= 1


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


def test_merge_overwrite_prefers_new_on_conflict(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    ts = _floor_minute(int(time.time() * 1000))
    existing = np.array([(ts, 1.0, 1.0, 1.0, 1.0, 1.0)], dtype=CANDLE_DTYPE)
    new = np.array([(ts, 2.0, 2.0, 2.0, 2.0, 2.0)], dtype=CANDLE_DTYPE)
    merged = cm._merge_overwrite(existing, new)
    assert merged.size == 1
    assert float(merged[0]["c"]) == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_get_latest_ema_metrics_calls_get_candles_once_and_caches(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"

    # Provide a deterministic window of 10 candles.
    base = _floor_minute(fixed_now_ms) - ONE_MIN_MS * 20
    ts = [base + i * ONE_MIN_MS for i in range(10)]
    arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
    arr["ts"] = np.asarray(ts, dtype=np.int64)
    arr["h"] = np.linspace(100.0, 109.0, len(ts)).astype(np.float32)
    arr["l"] = (arr["h"] - 1.0).astype(np.float32)
    arr["c"] = (arr["h"] - 0.5).astype(np.float32)
    arr["bv"] = np.linspace(1.0, 2.0, len(ts)).astype(np.float32)

    # Force the manager to use this exact range.
    end_ts = ts[-1]
    start_ts = end_ts - 9 * ONE_MIN_MS

    async def fake_latest_finalized_range(span, period_ms=ONE_MIN_MS):
        return (start_ts, end_ts)

    monkeypatch.setattr(cm, "_latest_finalized_range", fake_latest_finalized_range)

    calls = {"n": 0}

    async def fake_get_candles(
        symbol_,
        *,
        start_ts=None,
        end_ts=None,
        max_age_ms=None,
        strict=False,
        timeframe=None,
        tf=None,
        fill_leading_gaps=False,
        max_lookback_candles=None,
    ):
        calls["n"] += 1
        return arr

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    spans = {"qv": 5.0, "log_range": 3.0}
    out1 = await cm.get_latest_ema_metrics(symbol, spans, max_age_ms=60_000, timeframe=None)
    assert calls["n"] == 1
    assert set(out1.keys()) == set(spans.keys())

    qv_series = (
        np.asarray(arr[-5:]["bv"], dtype=np.float64)
        * (
            np.asarray(arr[-5:]["h"], dtype=np.float64)
            + np.asarray(arr[-5:]["l"], dtype=np.float64)
            + np.asarray(arr[-5:]["c"], dtype=np.float64)
        )
        / 3.0
    )
    lr_series = np.log(
        np.maximum(np.asarray(arr[-3:]["h"], dtype=np.float64), 1e-12)
        / np.maximum(np.asarray(arr[-3:]["l"], dtype=np.float64), 1e-12)
    )
    assert out1["qv"] == pytest.approx(float(cm._ema(qv_series, 5.0)))
    assert out1["log_range"] == pytest.approx(float(cm._ema(lr_series, 3.0)))

    # Second call should hit EMA cache (no new get_candles call).
    out2 = await cm.get_latest_ema_metrics(symbol, spans, max_age_ms=60_000, timeframe=None)
    assert calls["n"] == 1
    assert out2["qv"] == pytest.approx(out1["qv"])
    assert out2["log_range"] == pytest.approx(out1["log_range"])


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
async def test_gateio_old_1m_window_is_marked_without_remote_fetch(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "gateio"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="gateio", cache_dir=str(tmp_path / "caches")
    )
    symbol = "ADA/USDT:USDT"
    calls = {"fetch": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        calls["fetch"] += 1
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    start_ts = end_finalized - ONE_MIN_MS * 20_000
    end_ts = end_finalized - ONE_MIN_MS * 15_000

    out = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=True)

    assert out.size == 0
    assert calls["fetch"] == 0
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert len(gaps) == 1
    assert gaps[0]["start_ts"] == start_ts
    assert gaps[0]["end_ts"] == end_ts
    assert gaps[0]["retry_count"] == _GAP_MAX_RETRIES
    assert gaps[0]["reason"] == GAP_REASON_NO_ARCHIVE


@pytest.mark.asyncio
async def test_gateio_partial_1m_window_clips_fetch_to_recent_limit(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "gateio"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="gateio", cache_dir=str(tmp_path / "caches")
    )
    symbol = "SOL/USDT:USDT"
    calls = []

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        calls.append((int(since_ms), int(end_exclusive_ms)))
        ts = list(range(int(since_ms), int(end_exclusive_ms), ONE_MIN_MS))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        if ts:
            arr["ts"] = np.asarray(ts, dtype=np.int64)
            arr["o"] = 1.0
            arr["h"] = 1.0
            arr["l"] = 1.0
            arr["c"] = 1.0
            arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    earliest = end_finalized - ONE_MIN_MS * (_GATEIO_RECENT_1M_LIMIT_CANDLES - 1)
    start_ts = earliest - 20 * ONE_MIN_MS
    end_ts = earliest + 20 * ONE_MIN_MS

    await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=True)

    assert calls
    assert calls[0][0] >= earliest
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert len(gaps) == 1
    assert gaps[0]["start_ts"] == start_ts
    assert gaps[0]["end_ts"] == earliest - ONE_MIN_MS
    assert gaps[0]["retry_count"] == _GAP_MAX_RETRIES
    assert gaps[0]["reason"] == GAP_REASON_NO_ARCHIVE


@pytest.mark.asyncio
async def test_get_current_close_uses_latest_completed_candle_not_ticker(monkeypatch):
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

        async def fetch_ticker(self, symbol):
            raise AssertionError("CandlestickManager must not fetch tickers")

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx")
    symbol = "BTC/USDT:USDT"
    last_final = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(last_final, 123.45, 123.45, 123.45, 123.45, 1.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._set_last_refresh_meta(symbol, fixed_now_ms, last_final_ts=last_final)

    p1 = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert p1 == pytest.approx(123.45)


@pytest.mark.asyncio
async def test_get_current_close_never_persists_current_in_progress_candle(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"
    end_current = (fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS
    end_finalized = end_current - ONE_MIN_MS

    calls = {"paginated": 0}

    async def fake_paginated(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["paginated"] += 1
        arr = np.array(
            [
                (end_finalized, 1.0, 1.0, 1.0, 1.23, 1.0),
                (end_current, 2.0, 2.0, 2.0, 2.34, 1.0),
            ],
            dtype=CANDLE_DTYPE,
        )
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_paginated)

    p = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert p == pytest.approx(1.23)
    assert calls["paginated"] == 1

    cached = np.sort(cm._cache[symbol], order="ts")
    assert int(cached[-1]["ts"]) == end_finalized
    disk = cm._load_from_disk(symbol, end_finalized, end_current, timeframe="1m")
    assert disk.size
    assert int(np.sort(disk, order="ts")[-1]["ts"]) == end_finalized


@pytest.mark.asyncio
async def test_get_candles_ttl_skips_single_trailing_present_gap(monkeypatch, tmp_path):
    fixed_now_ms = 1725590520000  # 2024-09-06 00:02:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "bybit"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="bybit", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"
    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    cached_last = end_finalized - ONE_MIN_MS
    start_ts = cached_last - 3 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(start_ts + i * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0 + i, 1.0) for i in range(4)],
        dtype=CANDLE_DTYPE,
    )
    cm._set_last_refresh_meta(symbol, fixed_now_ms, last_final_ts=cached_last)
    calls = {"paginated": 0}

    async def fake_paginated(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        calls["paginated"] += 1
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_paginated)

    out = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_finalized,
        max_age_ms=365 * 24 * 3600 * 1000,
        strict=False,
    )

    assert calls["paginated"] == 0
    assert int(out[-1]["ts"]) == end_finalized


@pytest.mark.asyncio
async def test_get_last_prices_uses_completed_candles_not_bulk_tickers(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "bybit"

        async def fetch_tickers(self, symbols=None):
            raise AssertionError("CandlestickManager must not fetch tickers")

    cm = CandlestickManager(exchange=_Ex(), exchange_name="bybit", cache_dir=str(tmp_path / "caches"))
    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    for symbol, price in (("ADA/USDT:USDT", 0.75), ("SOL/USDT:USDT", 180.0)):
        cm._cache[symbol] = np.array(
            [(end_finalized, price, price, price, price, 1.0)],
            dtype=CANDLE_DTYPE,
        )
        cm._set_last_refresh_meta(symbol, fixed_now_ms, last_final_ts=end_finalized)

    prices = await cm.get_last_prices(["ADA/USDT:USDT", "SOL/USDT:USDT"], max_age_ms=10_000)

    assert prices == {
        "ADA/USDT:USDT": pytest.approx(0.75),
        "SOL/USDT:USDT": pytest.approx(180.0),
    }


@pytest.mark.asyncio
async def test_remote_ohlcv_fetch_spacing_paces_concurrent_calls(tmp_path):
    class _Ex:
        id = "okx"

        def __init__(self):
            self.call_times = []

        async def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
            self.call_times.append(time.monotonic())
            return [[int(since or 0), 1.0, 1.0, 1.0, 1.0, 1.0]]

    ex = _Ex()
    cm = CandlestickManager(
        exchange=ex,
        exchange_name="okx",
        cache_dir=str(tmp_path / "caches"),
        remote_fetch_min_interval_ms=40,
    )

    await asyncio.gather(
        cm._ccxt_fetch_ohlcv_once("BTC/USDT:USDT", 0, 1, timeframe="1m"),
        cm._ccxt_fetch_ohlcv_once("ETH/USDT:USDT", 0, 1, timeframe="1m"),
    )

    assert len(ex.call_times) == 2
    assert ex.call_times[1] - ex.call_times[0] >= 0.025


@pytest.mark.asyncio
async def test_get_current_close_does_not_tail_fetch_current_minute(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    # Use small overlap to keep payloads small
    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"), overlap_candles=5
    )
    symbol = "ETH/USDT:USDT"
    end_current = (fixed_now_ms // ONE_MIN_MS) * ONE_MIN_MS
    end_finalized = end_current - ONE_MIN_MS

    async def fake_paginated(symbol_, since_ms, end_exclusive_ms, *, timeframe=None):
        assert int(end_exclusive_ms) <= end_current
        return np.array(
            [(end_finalized, 5.0, 5.0, 5.0, 5.0, 1.0)],
            dtype=CANDLE_DTYPE,
        )

    async def fake_once(symbol_, since_ms, limit, end_exclusive_ms=None, timeframe=None):
        raise AssertionError("CandlestickManager must not tail-fetch current OHLCV")

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_paginated)
    monkeypatch.setattr(cm, "_ccxt_fetch_ohlcv_once", fake_once)

    p = await cm.get_current_close(symbol, max_age_ms=60_000)
    assert p == pytest.approx(5.0)

    arr = cm._cache.get(symbol)
    assert arr is not None and arr.size
    arr = np.sort(arr, order="ts")
    assert int(arr[-1]["ts"]) == end_finalized


@pytest.mark.asyncio
async def test_get_candles_materializes_runtime_synthetic_after_long_no_fill_gap(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "KBONK/USDC:USDC"
    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    old_ts = end_finalized - 6 * 60 * ONE_MIN_MS
    old_close = 0.1234

    seed = np.array([(old_ts, old_close, old_close, old_close, old_close, 1.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, seed, timeframe="1m", merge_cache=True)
    cm._cache.pop(symbol, None)  # simulate process restart (seed only on disk)

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    start_ts = end_finalized - 5 * ONE_MIN_MS
    out = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_finalized,
        max_age_ms=30_000,
        strict=False,
    )

    assert out.size == 6
    assert list(out["ts"]) == [start_ts + i * ONE_MIN_MS for i in range(6)]
    assert np.allclose(np.asarray(out["c"], dtype=np.float64), old_close)
    assert np.allclose(np.asarray(out["bv"], dtype=np.float64), 0.0)

    # Runtime synthetic candles must remain memory-only: shard still has only the original seed.
    day_key = cm._date_key(old_ts)
    shard = cm._load_shard(cm._shard_path(symbol, day_key, timeframe="1m"))
    assert shard.size == 1


def test_real_batch_overrides_runtime_synthetic_and_invalidates_ema_cache(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "ILLQ/USDT:USDT"
    base_ts = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    base_close = 11.0

    seed = np.array(
        [(base_ts, base_close, base_close, base_close, base_close, 1.0)], dtype=CANDLE_DTYPE
    )
    cm._cache[symbol] = seed
    synthesized = cm._materialize_runtime_synthetic_gap(symbol, base_ts + 2 * ONE_MIN_MS)
    assert synthesized == 2

    cm._ema_cache[symbol] = {("close", 5.0, str(ONE_MIN_MS)): (base_close, base_ts, base_ts)}

    real_ts = base_ts + ONE_MIN_MS
    assert real_ts in cm._synthetic_timestamps.get(symbol, set())
    real = np.array([(real_ts, 12.0, 12.5, 11.5, 12.2, 9.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, real, timeframe="1m", merge_cache=True, last_refresh_ms=base_ts + 3_000)

    arr = np.sort(cm._cache[symbol], order="ts")
    i = int(np.where(arr["ts"] == real_ts)[0][0])
    assert float(arr[i]["c"]) == pytest.approx(12.2)
    assert float(arr[i]["bv"]) == pytest.approx(9.0)

    assert symbol not in cm._ema_cache
    assert real_ts not in cm._synthetic_timestamps.get(symbol, set())


def test_materialize_runtime_synthetic_gap_caps_at_max_synth(tmp_path):
    """Gap larger than 24*60 minutes should only synthesize the most recent 1440 candles."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "DEAD/USDT:USDT"
    # Place a seed candle 3 days (4320 minutes) before through_ts.
    through_ts = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    gap_minutes = 3 * 24 * 60  # 4320 minutes
    seed_ts = through_ts - gap_minutes * ONE_MIN_MS
    seed_close = 42.0

    seed = np.array(
        [(seed_ts, seed_close, seed_close, seed_close, seed_close, 1.0)], dtype=CANDLE_DTYPE
    )
    cm._cache[symbol] = seed

    synthesized = cm._materialize_runtime_synthetic_gap(symbol, through_ts)

    # max_synth caps at min(max_memory_candles_per_symbol, 24*60) = 1440
    max_synth = min(cm.max_memory_candles_per_symbol, 24 * 60)
    assert synthesized == max_synth

    arr = np.sort(cm._cache[symbol], order="ts")
    synth_only = arr[arr["ts"] > seed_ts]
    assert synth_only.shape[0] == max_synth
    # First synthetic candle should start at through_ts - (max_synth - 1) * ONE_MIN_MS
    expected_first = through_ts - (max_synth - 1) * ONE_MIN_MS
    assert int(synth_only[0]["ts"]) == expected_first
    assert int(synth_only[-1]["ts"]) == through_ts
    # All synthetic candles carry the seed close and zero volume
    assert np.allclose(np.asarray(synth_only["c"], dtype=np.float64), seed_close)
    assert np.allclose(np.asarray(synth_only["bv"], dtype=np.float64), 0.0)


def test_completed_candle_health_excludes_current_minute_and_reports_gaps(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "HEALTH/USDT:USDT"
    now_ms = 1725590400000
    current_minute = _floor_minute(now_ms)
    last_final = current_minute - ONE_MIN_MS
    start = last_final - 2 * ONE_MIN_MS
    candles = np.array(
        [
            (start, 10.0, 10.0, 10.0, 10.0, 1.0),
            (last_final, 12.0, 12.0, 12.0, 12.0, 1.0),
            (current_minute, 99.0, 99.0, 99.0, 99.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, candles, timeframe="1m", merge_cache=True, last_refresh_ms=now_ms)

    report = cm.get_completed_candle_health(symbol, {"1m": 3}, now_ms=now_ms)

    one_m = report["timeframes"]["1m"]
    assert report["ok"] is False
    assert one_m["current_in_progress_excluded"] is True
    assert one_m["end_ts"] == last_final
    assert one_m["missing_candles"] == 1
    assert one_m["missing_spans"] == [(start + ONE_MIN_MS, start + ONE_MIN_MS)]
    assert one_m["last_cached_ts"] == last_final
    assert one_m["loaded_rows"] == 2


def test_completed_candle_health_reports_synthetic_and_hour_boundary(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "SYNTH/USDT:USDT"
    now_ms = 1725590400000
    last_hour = (now_ms // (60 * ONE_MIN_MS)) * (60 * ONE_MIN_MS) - 60 * ONE_MIN_MS
    current_hour = last_hour + 60 * ONE_MIN_MS
    hour_candles = np.array(
        [
            (last_hour, 100.0, 101.0, 99.0, 100.5, 50.0),
            (current_hour, 200.0, 201.0, 199.0, 200.5, 50.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, hour_candles, timeframe="1h", merge_cache=False)

    last_minute = _floor_minute(now_ms) - ONE_MIN_MS
    seed = np.array(
        [(last_minute - ONE_MIN_MS, 11.0, 11.0, 11.0, 11.0, 1.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = seed
    assert cm._materialize_runtime_synthetic_gap(symbol, last_minute) == 1
    cm._synthetic_timestamps[symbol] = {last_minute}

    report = cm.get_completed_candle_health(symbol, {"1m": 2, "1h": 1}, now_ms=now_ms)

    one_m = report["timeframes"]["1m"]
    one_h = report["timeframes"]["1h"]
    assert one_m["coverage_ok"] is True
    assert one_m["runtime_synthetic_count"] == 1
    assert one_h["coverage_ok"] is True
    assert one_h["end_ts"] == last_hour
    assert one_h["loaded_rows"] == 1
    assert one_h["last_cached_ts"] == last_hour


def test_completed_candle_health_non_required_window_does_not_fail_overall(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    report = cm.get_completed_candle_health(
        "DIAG/USDT:USDT",
        {"15m": {"candles": 1, "required": False}},
        now_ms=1725590400000,
    )

    assert report["ok"] is True
    assert report["timeframes"]["15m"]["required"] is False
    assert report["timeframes"]["15m"]["coverage_ok"] is False
    assert report["timeframes"]["15m"]["missing_candles"] == 1


@pytest.mark.asyncio
async def test_refresh_bounds_disk_load_range(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="okx",
        cache_dir=str(tmp_path / "caches"),
        default_window_candles=100,
        overlap_candles=30,
    )
    symbol = "BTC/USDT:USDT"
    calls = []

    def fake_load_from_disk(symbol_, start_ts, end_ts, *, timeframe=None, tf=None):
        calls.append((int(start_ts), int(end_ts), (timeframe or tf)))
        return None

    async def fake_fetch(
        symbol_, since_ms, end_exclusive_ms, *, timeframe=None, tf=None, on_batch=None
    ):
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_load_from_disk", fake_load_from_disk)
    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    await cm.refresh(symbol)

    end_exclusive = _floor_minute(fixed_now_ms)
    lookback_candles = max(100, 30) + 10
    disk_since = max(0, end_exclusive - lookback_candles * ONE_MIN_MS)

    assert calls
    refresh_window_calls = [(start, end, tf) for (start, end, tf) in calls if end == end_exclusive]
    assert refresh_window_calls
    assert all(start >= disk_since for start, _, _ in refresh_window_calls)


# ----- Enhanced Gap Metadata Tests -----


def test_enhanced_gap_metadata_new_format(tmp_path):
    """Test that gaps are stored in enhanced format with retry counts."""
    from candlestick_manager import GapEntry, GAP_REASON_FETCH_FAILED, _GAP_MAX_RETRIES

    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add a gap
    cm._add_known_gap(symbol, 1000000, 2000000, reason=GAP_REASON_FETCH_FAILED)

    # Verify enhanced format
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert len(gaps) == 1
    assert gaps[0]["start_ts"] == 1000000
    assert gaps[0]["end_ts"] == 2000000
    assert gaps[0]["retry_count"] == 1
    assert gaps[0]["reason"] == GAP_REASON_FETCH_FAILED
    assert "added_at" in gaps[0]

    # Backward compatibility: simple tuple format still works
    simple_gaps = cm._get_known_gaps(symbol)
    assert len(simple_gaps) == 1
    assert simple_gaps[0] == (1000000, 2000000)


def test_gap_retry_count_increments(tmp_path):
    """Test that retry count increments when gap is re-added."""
    from candlestick_manager import _GAP_MAX_RETRIES

    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add gap 3 times (overlapping)
    for i in range(3):
        cm._add_known_gap(symbol, 1000000, 2000000, increment_retry=True)
        gaps = cm._get_known_gaps_enhanced(symbol)
        assert gaps[0]["retry_count"] == i + 1

    # After max retries, gap should be considered persistent
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert gaps[0]["retry_count"] >= _GAP_MAX_RETRIES
    assert not cm._should_retry_gap(gaps[0])


def test_gap_retry_without_increment(tmp_path):
    """Test that retry count doesn't increment when increment_retry=False."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    cm._add_known_gap(symbol, 1000000, 2000000, increment_retry=False)
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert gaps[0]["retry_count"] == 0


def test_clear_known_gaps_all(tmp_path):
    """Test clearing all gaps for a symbol."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add multiple gaps
    cm._add_known_gap(symbol, 1000000, 2000000)
    cm._add_known_gap(symbol, 5000000, 6000000)
    assert len(cm._get_known_gaps(symbol)) == 2

    # Clear all
    cleared = cm.clear_known_gaps(symbol)
    assert cleared == 2
    assert len(cm._get_known_gaps(symbol)) == 0


def test_clear_known_gaps_by_date_range(tmp_path):
    """Test clearing gaps within a specific date range."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add gaps at different ranges
    cm._add_known_gap(symbol, 1000000, 2000000)  # Gap 1
    cm._add_known_gap(symbol, 5000000, 6000000)  # Gap 2
    cm._add_known_gap(symbol, 9000000, 10000000)  # Gap 3

    # Clear only gaps in middle range
    cleared = cm.clear_known_gaps(symbol, date_range=(4000000, 7000000))
    assert cleared == 1

    # Verify remaining gaps
    gaps = cm._get_known_gaps(symbol)
    assert len(gaps) == 2
    assert (1000000, 2000000) in gaps
    assert (9000000, 10000000) in gaps


def test_gap_summary(tmp_path):
    """Test gap summary generation."""
    from candlestick_manager import GAP_REASON_FETCH_FAILED, GAP_REASON_AUTO, _GAP_MAX_RETRIES

    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add gaps with different retry counts
    cm._add_known_gap(symbol, 1000000, 1060000, reason=GAP_REASON_FETCH_FAILED)  # 1 retry
    cm._add_known_gap(symbol, 5000000, 5120000, reason=GAP_REASON_AUTO)  # 1 retry

    # Make second gap persistent
    for _ in range(_GAP_MAX_RETRIES - 1):
        cm._add_known_gap(symbol, 5000000, 5120000)

    summary = cm.get_gap_summary(symbol)
    assert summary["total_gaps"] == 2
    assert summary["persistent_gaps"] == 1
    assert summary["retryable_gaps"] == 1
    assert GAP_REASON_FETCH_FAILED in summary["by_reason"]
    assert len(summary["gaps"]) == 2


def test_legacy_gap_format_upgrade(tmp_path):
    """Test that old gap format is auto-upgraded to enhanced format."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Manually inject old format into index
    idx = cm._ensure_symbol_index(symbol)
    idx["meta"]["known_gaps"] = [[1000000, 2000000], [5000000, 6000000]]
    cm._index[symbol] = idx
    cm._save_index(symbol)

    # Read should auto-upgrade
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert len(gaps) == 2
    for gap in gaps:
        assert "start_ts" in gap
        assert "end_ts" in gap
        assert "retry_count" in gap
        assert "reason" in gap
        assert "added_at" in gap


@pytest.mark.asyncio
async def test_force_refetch_gaps_clears_gaps(tmp_path):
    """Test that force_refetch_gaps clears gaps in the requested range."""
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TEST/USDT"

    # Add a gap
    cm._add_known_gap(symbol, 1000000, 2000000)
    assert len(cm._get_known_gaps(symbol)) == 1

    # Pre-populate cache to avoid network fetch
    base = 1000000
    arr = np.array(
        [(base + i * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0) for i in range(20)], dtype=CANDLE_DTYPE
    )
    cm._cache[symbol] = arr

    # Call get_candles with force_refetch_gaps
    await cm.get_candles(
        symbol,
        start_ts=1000000,
        end_ts=2000000,
        force_refetch_gaps=True,
    )

    # Gap should be cleared
    assert len(cm._get_known_gaps(symbol)) == 0
