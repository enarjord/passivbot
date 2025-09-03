import asyncio
import gzip
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pytest

import candlestick_manager as cm


ONE_MIN = cm.ONE_MIN_MS


def ts_ms(y, m, d, hh=0, mm=0, ss=0):
    import datetime as dt

    return int(dt.datetime(y, m, d, hh, mm, ss, tzinfo=dt.timezone.utc).timestamp() * 1000)


class DummyExchange:
    """
    Minimal async CCXT-like stub for fetch_ohlcv with deterministic data and call counts.
    """

    def __init__(self, config=None):
        self.options = {}
        self._store: Dict[str, List[List[float]]] = {}
        self._call_count = 0
        self._delay = 0.0  # seconds artificial delay to simulate network

    def set_series(self, symbol: str, series: List[List[float]]):
        self._store[symbol] = series

    def set_delay(self, seconds: float):
        self._delay = seconds

    async def fetch_ohlcv(self, symbol: str, timeframe="1m", since=None, limit=1000):
        assert timeframe == "1m"
        self._call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        series = self._store.get(symbol, [])
        if since is None:
            start_idx = 0
        else:
            # first index with ts >= since
            start_idx = 0
            for i, row in enumerate(series):
                if int(row[0]) >= int(since):
                    start_idx = i
                    break
            else:
                return []
        end_idx = min(len(series), start_idx + int(limit))
        return [list(map(float, row)) for row in series[start_idx:end_idx]]

    async def load_markets(self, reload=False, params=None):
        return {}

    async def close(self):
        return


class CCXTNamespace:
    """
    Namespace that mimics ccxt.async_support module attribute access by exchange name.
    """

    def __init__(self, exchange_cls):
        self._cls = exchange_cls

    def __getattr__(self, name):
        # Return exchange constructor when accessed by attribute name
        return self._cls


@pytest.fixture
def fixed_now_ms():
    # 2024-01-10 00:10:30 UTC (moved later to ensure all test request periods are in the past)
    return ts_ms(2024, 1, 10, 0, 10, 30)


@pytest.fixture
def patch_now(monkeypatch, fixed_now_ms):
    monkeypatch.setattr(cm, "_utc_now_ms", lambda: fixed_now_ms)


@pytest.fixture
def shared_file_lock(monkeypatch):
    """
    Ensure cross-instance locking works even if 'filelock' isn't available,
    by monkeypatching _acquire_lock to return a process-wide async lock per-path.
    """
    locks: Dict[str, asyncio.Lock] = {}

    class SharedAsyncLock:
        def __init__(self, path: str):
            self._path = path
            if path not in locks:
                locks[path] = asyncio.Lock()
            self._lock = locks[path]

        async def __aenter__(self):
            await locks[self._path].acquire()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            locks[self._path].release()
            return False

    def acquire_lock(self, path):
        return SharedAsyncLock(str(path))

    monkeypatch.setattr(cm.CandlestickManager, "_acquire_lock", acquire_lock, raising=False)


@pytest.fixture
def dummy_ccxt(monkeypatch):
    """
    Patch candlestick_manager.ccxt to our dummy namespace/exchange.
    Returns an instance of DummyExchange so tests can seed data and assert call counts.
    """
    ex = DummyExchange()
    ns = CCXTNamespace(lambda config=None: ex)
    monkeypatch.setattr(cm, "ccxt", ns, raising=False)
    return ex


def make_series(start_ms: int, num: int, base_price=100.0, dv=0.5, base_vol=1.0):
    """
    Create 'num' minutes of deterministic OHLCV, base volume constant.
    Returns list of CCXT rows: [ts, o, h, l, c, volume_base]
    """
    out = []
    price = base_price
    for i in range(num):
        ts = start_ms + i * ONE_MIN
        o = price
        c = price + ((-1) ** i) * dv
        h = max(o, c) + dv
        l = min(o, c) - dv
        out.append([ts, o, h, l, c, base_vol])
        price = c
    return out


@pytest.mark.asyncio
async def test_basic_fetch_and_cache(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "BTC/USDT:USDT"

    # Build 90 minutes from 2024-01-01 00:00
    start = ts_ms(2024, 1, 1, 0, 0, 0)
    series = make_series(start, 90, base_price=100.0, dv=1.0, base_vol=2.0)
    dummy_ccxt.set_series(symbol, series)

    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        # Request a subrange [00:10, 01:00)
        req_start = start + 10 * ONE_MIN
        req_end = start + 60 * ONE_MIN
        arr = await mgr.get(req_start, req_end)

        assert arr.shape == (50, 6)
        # Check timestamps are minute-aligned and within range
        assert int(arr[0, 0]) == req_start
        assert int(arr[-1, 0]) == req_end - ONE_MIN

        # Verify quote volume = base_vol * close
        # series base_vol = 2.0
        for row in arr:
            ts = int(row[0])
            src = next(r for r in series if int(r[0]) == ts)
            assert np.isclose(row[5], src[5] * src[4])

        # Second call should hit cache and do no additional fetches
        calls_before = dummy_ccxt._call_count
        arr2 = await mgr.get(req_start, req_end)
        assert np.array_equal(arr, arr2)
        assert dummy_ccxt._call_count == calls_before


@pytest.mark.asyncio
async def test_excludes_current_incomplete_minute(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "ETH/USDT:USDT"

    # now_floor = 2024-01-02 00:10:00
    now_floor = cm._floor_minute(cm._utc_now_ms())
    # Provide 3 candles ending exactly at now_floor (which must be excluded)
    series = make_series(now_floor - 3 * ONE_MIN, 3, base_price=2000.0, dv=2.0, base_vol=1.0)
    # Also append an extra candle at now_floor to ensure it's excluded by manager
    series.append([now_floor, 2006.0, 2008.0, 2004.0, 2006.5, 1.0])
    dummy_ccxt.set_series(symbol, series)

    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        arr = await mgr.get(now_floor - 3 * ONE_MIN, None)
        # Expect 2 candles: at -3m and -2m and -1m? Wait, end exclusive is now_floor,
        # and our data includes [-3m, -2m, -1m], so count=3
        assert arr.shape[0] == 3
        assert int(arr[-1, 0]) == now_floor - ONE_MIN
        assert (arr[:, 0] < now_floor).all()


@pytest.mark.asyncio
async def test_multi_day_fetch_and_cache(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "SOL/USDT:USDT"

    day1 = ts_ms(2024, 1, 1, 0, 0, 0)
    day2 = ts_ms(2024, 1, 2, 0, 0, 0)

    # 1.5 days of data starting day1 00:00
    series = make_series(day1, 36 * 60, base_price=100.0, dv=0.2, base_vol=3.0)
    dummy_ccxt.set_series(symbol, series)

    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        req_start = day1 + 23 * 60 * ONE_MIN  # 23:00 day1
        req_end = day2 + 3 * 60 * ONE_MIN  # 03:00 day2
        arr = await mgr.get(req_start, req_end)
        expected = (60) + (
            3 * 60
        )  # 23:00..23:59 (60), 00:00..02:59 (180) -> 240? Wait: 23:00..02:59 = 300?
        # re-evaluate: from 23:00 to next day 03:00 exclusive is 4 hours = 240 minutes
        assert arr.shape[0] == 240
        assert int(arr[0, 0]) == req_start
        assert int(arr[-1, 0]) == req_end - ONE_MIN

        # Call again to ensure cache hit across both days
        calls_before = dummy_ccxt._call_count
        arr2 = await mgr.get(req_start, req_end)
        assert np.array_equal(arr, arr2)
        assert dummy_ccxt._call_count == calls_before


@pytest.mark.asyncio
async def test_concurrent_instances_safe(tmp_path, dummy_ccxt, patch_now, shared_file_lock):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "XRP/USDT:USDT"

    day = ts_ms(2024, 1, 3, 0, 0, 0)
    series = make_series(day, 60, base_price=0.5, dv=0.01, base_vol=5.0)
    dummy_ccxt.set_series(symbol, series)
    dummy_ccxt.set_delay(0.05)

    async def run_one():
        async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
            return await mgr.get(day, day + 60 * ONE_MIN)

    # Kick off two concurrent fetches
    arr1, arr2 = await asyncio.gather(run_one(), run_one())
    assert np.array_equal(arr1, arr2)
    assert arr1.shape[0] == 60

    # Inspect saved file to ensure no duplicates
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr_check:
        day_path = mgr_check.layout.day_path(exchange, symbol, day)
        assert day_path.exists()
        with gzip.open(day_path, "rb") as f:
            buf = io.BytesIO(f.read())
        saved = np.load(buf, allow_pickle=False)
        # Timestamps unique and sorted
        assert saved.ndim == 2 and saved.shape[1] == 6
        ts = saved[:, 0].astype(np.int64)
        assert np.all(np.diff(ts) > 0)
        assert len(np.unique(ts)) == len(ts)


@pytest.mark.asyncio
async def test_cache_integrity_fixes_and_backfill(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "ADA/USDT:USDT"

    day = ts_ms(2024, 1, 4, 0, 0, 0)
    # True complete series for an hour
    series = make_series(day, 60, base_price=0.3, dv=0.005, base_vol=2.5)
    dummy_ccxt.set_series(symbol, series)

    # Pre-write a broken cached file:
    # - include misaligned ts (+1000 ms)
    # - duplicate a proper minute
    # - remove some minutes to create gaps that must be backfilled
    broken = []
    for i in range(10):
        broken.append(
            [day + i * ONE_MIN + 1000, 0.3, 0.31, 0.29, 0.305, 1.0]
        )  # misaligned -> must be dropped
    # proper aligned duplicates for minute 10
    r10 = series[10]
    broken.append(r10)
    broken.append(r10)
    # sparse aligned points: 20, 40
    broken.append(series[20])
    broken.append(series[40])

    # Save broken file to cache
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr_prep:
        path = mgr_prep.layout.day_path(exchange, symbol, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as gz:
            np.save(gz, np.array(broken, dtype=np.float64), allow_pickle=False)

    # Now fetch the hour; manager should drop misaligned, dedup duplicates, and backfill missing via exchange
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        arr = await mgr.get(day, day + 60 * ONE_MIN)
        assert arr.shape[0] == 60
        # Ensure fully continuous timestamps
        assert np.all(np.diff(arr[:, 0]) == ONE_MIN)


@pytest.mark.asyncio
async def test_metadata_earliest_discovery_and_reuse(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "DOGE/USDT:USDT"

    # Build series starting at a specific earliest timestamp
    earliest = ts_ms(2023, 12, 31, 0, 0, 0)
    series = make_series(earliest, 30, base_price=0.1, dv=0.001, base_vol=1.5)
    dummy_ccxt.set_series(symbol, series)

    calls = {"discover": 0}

    async def fake_discover(self):
        calls["discover"] += 1
        return earliest

    # First call: no metadata file, should call discover and then save metadata
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        # Patch discover method on instance
        mgr._discover_earliest_ts = fake_discover.__get__(mgr, cm.CandlestickManager)
        arr = await mgr.get(None, earliest + 10 * ONE_MIN)
        assert arr.shape[0] == 10
        # Metadata saved
        meta_path = mgr.layout.metadata_path(exchange, symbol)
        assert meta_path.exists()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert int(meta["earliest_ts"]) == earliest
        assert calls["discover"] == 1

    # Second call: should reuse metadata without calling discover
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr2:
        mgr2._discover_earliest_ts = fake_discover.__get__(mgr2, cm.CandlestickManager)
        _ = await mgr2.get(None, earliest + 5 * ONE_MIN)
        # discover should not be called again
        assert calls["discover"] == 1


@pytest.mark.asyncio
async def test_start_is_clamped_to_earliest_when_provided(tmp_path, dummy_ccxt, patch_now):
    data_root = tmp_path / "candles"
    exchange = "binanceusdm"
    symbol = "ATOM/USDT:USDT"

    earliest = ts_ms(2023, 12, 30, 0, 0, 0)
    series = make_series(earliest, 15, base_price=10.0, dv=0.1, base_vol=1.2)
    dummy_ccxt.set_series(symbol, series)

    bad_start = earliest - 2 * 24 * 60 * ONE_MIN  # 2 days before earliest
    async with cm.CandlestickManager(exchange, symbol, data_root=data_root) as mgr:
        arr = await mgr.get(bad_start, earliest + 5 * ONE_MIN)
        # Should clamp to earliest and return 5 minutes
        assert arr.shape[0] == 5
        assert int(arr[0, 0]) == earliest

        # Metadata persisted with earliest
        meta_path = mgr.layout.metadata_path(exchange, symbol)
        assert meta_path.exists()
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert int(meta["earliest_ts"]) == earliest
