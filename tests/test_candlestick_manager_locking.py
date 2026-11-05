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


@pytest.mark.asyncio
async def test_reentrant_fetch_lock_does_not_suppress_body_exception_when_record_missing(
    tmp_path,
):
    cm = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        default_window_candles=5,
    )
    key = ("BTC/USDC:USDC", "1m")

    async with cm._acquire_fetch_lock(*key):
        outer_record = cm._held_fetch_locks[key]
        try:
            with pytest.raises(RuntimeError, match="body failure"):
                async with cm._acquire_fetch_lock(*key):
                    cm._held_fetch_locks.pop(key)
                    raise RuntimeError("body failure")
        finally:
            cm._held_fetch_locks[key] = outer_record

    assert cm._held_fetch_locks == {}


@pytest.mark.asyncio
async def test_fetch_lock_is_reentrant_only_for_owning_task(tmp_path):
    cm = CandlestickManager(
        exchange=FakeExchange(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        default_window_candles=5,
    )
    key = ("BTC/USDC:USDC", "1m")
    contender_entered = asyncio.Event()

    async def contend_for_lock():
        async with cm._acquire_fetch_lock(*key):
            contender_entered.set()

    async with cm._acquire_fetch_lock(*key):
        owner_record = cm._held_fetch_locks[key]
        assert owner_record.owner_task is asyncio.current_task()

        # Nested acquisition by the owning task remains reentrant.
        async with cm._acquire_fetch_lock(*key):
            assert cm._held_fetch_locks[key].count == 2
        assert cm._held_fetch_locks[key].count == 1

        # A separate task using the same manager must wait.
        contender = asyncio.create_task(contend_for_lock())
        await asyncio.sleep(cm._lock_backoff_initial * 1.5)
        assert not contender_entered.is_set()

    await asyncio.wait_for(contender, timeout=1.0)
    assert contender_entered.is_set()
    assert cm._held_fetch_locks == {}


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


@pytest.mark.asyncio
async def test_stalled_refresh_preserves_pages_releases_lock_and_ws_can_extend(tmp_path):
    cm = CandlestickManager(
        exchange=FakeExchange(), exchange_name="kucoin", cache_dir=str(tmp_path),
        default_window_candles=5, archive_enabled=False,
    )
    minute = 60_000
    cm._now_ms_callback = lambda: 1000 * minute + 1_000
    cm._ccxt_page_overlap_candles = 1
    calls = []

    async def repeated_tail(_symbol, since, _limit, **_kwargs):
        calls.append(since)
        assert len(calls) <= 2
        return [[i * minute, 100, 103, 99, 100 if len(calls) == 1 else 102, 1] for i in (995, 996, 997)]

    cm._ccxt_fetch_ohlcv_once = repeated_tail
    symbol = "SPARSE/USDT:USDT"
    await cm.refresh(symbol)
    assert len(calls) == 2
    assert cm._held_fetch_locks == {}
    assert cm.get_last_final_ts(symbol) == 997 * minute
    assert 998 * minute not in set(cm._cache[symbol]["ts"])
    assert cm._cache[symbol][-1]["c"] == 102

    # A successor-proven WS row can extend the preserved REST basis after
    # pagination yields; the missing open tail was never fabricated.
    assert await cm.ingest_live_ws_ohlcv(
        symbol, [[998 * minute, 100, 102, 99, 101, 1]], now_ms=999 * minute - 1_000,
    ) == 0
    assert await cm.ingest_live_ws_ohlcv(
        symbol, [[999 * minute, 101, 101, 101, 101, 0]], now_ms=999 * minute + 1_000,
    ) == 1
    assert cm.get_last_final_ts(symbol) == 998 * minute
    assert cm._held_fetch_locks == {}


@pytest.mark.asyncio
async def test_live_fetch_budget_cancels_fetch_releases_lock_and_allows_retry(tmp_path):
    exchange = FakeExchange()
    exchange._sleep = 10.0
    cm = CandlestickManager(
        exchange=exchange, exchange_name="fake", cache_dir=str(tmp_path),
        default_window_candles=5, archive_enabled=False,
    )
    with cm.remote_fetch_budget(0.02):
        with pytest.raises(TimeoutError, match="budget expired"):
            await cm.refresh("BTC/USDT:USDT")
    assert cm._held_fetch_locks == {}

    # The deadline is caller-owned; a subsequent pass can acquire and fetch.
    exchange._sleep = 0.0
    await asyncio.wait_for(cm.refresh("BTC/USDT:USDT"), timeout=1.0)
    assert cm._held_fetch_locks == {}
    assert cm.get_last_final_ts("BTC/USDT:USDT") is not None


@pytest.mark.asyncio
async def test_live_fetch_budget_bounds_contended_lock_wait(tmp_path):
    holder = CandlestickManager(
        exchange=FakeExchange(), cache_dir=str(tmp_path), archive_enabled=False,
    )
    waiter = CandlestickManager(
        exchange=FakeExchange(), cache_dir=str(tmp_path), archive_enabled=False,
    )
    key = ("BTC/USDT:USDT", "1m")
    async with holder._acquire_fetch_lock(*key):
        started = time.monotonic()
        with waiter.remote_fetch_budget(0.02):
            with pytest.raises(TimeoutError):
                async with waiter._acquire_fetch_lock(*key):
                    pytest.fail("contender acquired the holder's lock")
        assert time.monotonic() - started < 0.25
        assert key in holder._held_fetch_locks
    assert waiter._held_fetch_locks == {}


@pytest.mark.asyncio
async def test_live_budget_shared_by_children_preserves_cache_reads_after_expiry(tmp_path):
    cm = CandlestickManager(
        exchange=FakeExchange(), exchange_name="fake", cache_dir=str(tmp_path),
        default_window_candles=5, archive_enabled=False,
    )
    minute = 60_000
    cm._now_ms_callback = lambda: 1000 * minute + 1000
    symbol = "CACHED/USDT:USDT"
    rows = np.array([(i * minute, 1, 1, 1, 1, 1) for i in range(995, 1000)], dtype=CANDLE_DTYPE)
    cm._cache[symbol] = rows.copy()

    async def after_expiry():
        await asyncio.sleep(0.03)
        with pytest.raises(TimeoutError):
            await cm._ccxt_fetch_ohlcv_once("MISSING/USDT:USDT", 995 * minute, 5)
        # A later/narrower scope must not renew the inherited exhausted budget.
        with cm.remote_fetch_budget(10.0):
            with pytest.raises(TimeoutError):
                cm._remote_fetch_time_remaining()
        return await cm.get_candles(
            symbol, start_ts=995 * minute, end_ts=999 * minute,
            allow_remote_fetch=False, standardize=False,
        )

    with cm.remote_fetch_budget(0.01):
        task = asyncio.create_task(after_expiry())
        result = await task
    np.testing.assert_array_equal(result, rows)
    assert cm._remote_fetch_time_remaining() is None


@pytest.mark.asyncio
async def test_cancel_at_lock_acquisition_releases_for_next_owner(tmp_path, monkeypatch):
    import portalocker

    cm = CandlestickManager(
        exchange=FakeExchange(), cache_dir=str(tmp_path), archive_enabled=False,
    )
    successor = CandlestickManager(
        exchange=FakeExchange(), cache_dir=str(tmp_path), archive_enabled=False,
    )
    original_acquire = portalocker.Lock.acquire
    loop = asyncio.get_running_loop()
    owner = None
    cancel_once = True

    def acquire_then_cancel(lock, *args, **kwargs):
        nonlocal cancel_once
        result = original_acquire(lock, *args, **kwargs)
        if cancel_once:
            cancel_once = False
            loop.call_soon_threadsafe(owner.cancel)
        return result

    monkeypatch.setattr(portalocker.Lock, "acquire", acquire_then_cancel)
    key = ("BTC/USDT:USDT", "1m")

    async def hold():
        async with cm._acquire_fetch_lock(*key):
            await asyncio.sleep(10)

    owner = asyncio.create_task(hold())
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert cm._held_fetch_locks == {}
    with successor.remote_fetch_budget(0.1):
        async with successor._acquire_fetch_lock(*key):
            assert key in successor._held_fetch_locks
    assert successor._held_fetch_locks == {}


@pytest.mark.asyncio
async def test_live_timeout_preserves_successful_pages_for_restart(tmp_path):
    minute = 60_000

    class SlowSecondPage(FakeExchange):
        def __init__(self):
            self.calls = 0

        async def fetch_ohlcv(self, symbol, **kwargs):
            self.calls += 1
            if self.calls > 1:
                await asyncio.sleep(10)
            return [[i * minute, 1, 1, 1, 1, 1] for i in range(995, 998)]

    cm = CandlestickManager(
        exchange=SlowSecondPage(), exchange_name="fake", cache_dir=str(tmp_path),
        default_window_candles=5, archive_enabled=False,
    )
    cm._now_ms_callback = lambda: 1000 * minute + 1000
    symbol = "PARTIAL/USDT:USDT"
    with cm.remote_fetch_budget(0.02):
        with pytest.raises(TimeoutError):
            await cm.refresh(symbol)
    assert cm._held_fetch_locks == {}
    assert cm.get_last_final_ts(symbol) == 997 * minute

    restarted = CandlestickManager(
        exchange=None, exchange_name="fake", cache_dir=str(tmp_path),
        default_window_candles=5, archive_enabled=False,
    )
    restarted._now_ms_callback = cm._now_ms_callback
    rows = await restarted.get_candles(
        symbol, start_ts=995 * minute, end_ts=999 * minute,
        allow_remote_fetch=False, standardize=False,
    )
    assert rows["ts"].tolist() == [i * minute for i in range(995, 998)]


@pytest.mark.asyncio
async def test_budget_timeout_emits_terminal_remote_fetch_diagnostic_without_retry(tmp_path):
    exchange = FakeExchange()
    exchange._sleep = 10.0
    events = []
    cm = CandlestickManager(
        exchange=exchange, exchange_name="fake", cache_dir=str(tmp_path),
        archive_enabled=False, remote_fetch_callback=events.append,
    )
    with cm.remote_fetch_budget(0.02):
        with pytest.raises(TimeoutError, match="budget expired"):
            await cm._ccxt_fetch_ohlcv_once("BTC/USDT:USDT", 995 * 60_000, 5)
    requests = [event for event in events if event.get("kind") == "ccxt_fetch_ohlcv"]
    assert [event["stage"] for event in requests] == ["start", "error"]
    assert requests[1]["attempt"] == 1
    assert requests[1]["error_type"] == "TimeoutError"
    assert requests[1]["elapsed_ms"] >= 0
    assert requests[0]["symbol"] == requests[1]["symbol"] == "BTC/USDT:USDT"
    assert requests[0]["tf"] == requests[1]["tf"] == "1m"
    assert requests[0]["since_ts"] == requests[1]["since_ts"]


@pytest.mark.asyncio
async def test_live_budget_bounds_spacing_lock_wait_without_cancelling_holder(tmp_path):
    cm = CandlestickManager(
        exchange=FakeExchange(), cache_dir=str(tmp_path), archive_enabled=False,
        remote_fetch_min_interval_ms=1.0,
    )
    acquired, release = asyncio.Event(), asyncio.Event()

    async def hold_spacing_lock():
        async with cm._remote_fetch_spacing_lock:
            acquired.set()
            await release.wait()

    holder = asyncio.create_task(hold_spacing_lock())
    await acquired.wait()
    started = time.monotonic()
    try:
        with cm.remote_fetch_budget(0.02):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    cm._ccxt_fetch_ohlcv_once("BTC/USDT:USDT", 60_000, 2), timeout=0.3
                )
        assert time.monotonic() - started < 0.12
        assert cm._remote_fetch_spacing_lock.locked()
        assert not holder.done()
    finally:
        release.set()
        await holder
    assert not cm._remote_fetch_spacing_lock.locked()


@pytest.mark.asyncio
async def test_archive_http_body_obeys_budget_and_emits_terminal_event(tmp_path):
    closed = []
    events = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            closed.append(True)

        def raise_for_status(self):
            pass

        async def read(self):
            await asyncio.sleep(0.15)
            return b"late"

    class Session:
        def get(self, _url):
            return Response()

    cm = CandlestickManager(
        exchange=None, cache_dir=str(tmp_path), remote_fetch_callback=events.append,
    )

    async def get_session():
        return Session()

    cm._get_http_session = get_session
    started = time.monotonic()
    with cm.remote_fetch_budget(0.02):
        with pytest.raises(TimeoutError):
            await cm._archive_fetch_bytes("https://archive.invalid/day.zip")
    assert time.monotonic() - started < 0.12
    assert closed == [True]
    assert [event["stage"] for event in events] == ["start", "error"]
    assert events[-1]["error_type"] == "TimeoutError"
    assert events[-1]["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_archive_prefetch_budget_keeps_completed_days_and_stops_new_batches(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    cm._archive_supported = lambda: True
    day_ms, minute = 86_400_000, 60_000
    first = 1_577_836_800_000  # 2020-01-01 UTC
    rows = np.array([
        (first + i * minute, 1, 1, 1, 1, 1) for i in range(1440)
    ], dtype=CANDLE_DTYPE)
    started_days, cancelled_days, events = [], [], []
    cm._remote_fetch_callback = events.append

    async def fetch_day(_symbol, day):
        started_days.append(day)
        if day == "2020-01-01":
            return rows
        try:
            await asyncio.sleep(0.15)
        finally:
            cancelled_days.append(day)

    cm._archive_fetch_day = fetch_day
    symbol = "BTC/USDT:USDT"
    started = time.monotonic()
    with cm.remote_fetch_budget(0.03):
        with pytest.raises(TimeoutError):
            await cm._prefetch_archives_for_range(
                symbol, first, first + 3 * day_ms - minute, parallel_days=2,
            )
        # The expired remote budget does not prevent a factual local read.
        cached = await cm.get_candles(
            symbol, start_ts=first, end_ts=first + day_ms - minute,
            allow_remote_fetch=False, standardize=False,
        )
    assert time.monotonic() - started < 0.2
    assert started_days == ["2020-01-01", "2020-01-02"]
    assert cancelled_days == ["2020-01-02"]
    np.testing.assert_array_equal(cached, rows)
    persisted_index = json.loads(open(cm._index_path(symbol)).read())
    assert "2020-01-01" in persisted_index["shards"]
    assert "2020-01-03" not in persisted_index["shards"]
    assert events[-1]["stage"] == "error"
    assert events[-1]["kind"] == "archive_prefetch"
    restarted = CandlestickManager(exchange=None, exchange_name="test", cache_dir=str(tmp_path))
    from_disk = await restarted.get_candles(
        symbol, start_ts=first, end_ts=first + day_ms - minute,
        allow_remote_fetch=False, standardize=False,
    )
    np.testing.assert_array_equal(from_disk, rows)


@pytest.mark.asyncio
async def test_cancelled_archive_prefetch_drains_all_current_children(tmp_path):
    cm = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    cm._archive_supported = lambda: True
    day_ms, minute = 86_400_000, 60_000
    first = 1_577_836_800_000
    entered, finished = [], []
    both_started = asyncio.Event()

    async def fetch_day(_symbol, day):
        entered.append(day)
        if len(entered) == 2:
            both_started.set()
        try:
            await asyncio.sleep(10)
        finally:
            finished.append(day)

    cm._archive_fetch_day = fetch_day
    task = asyncio.create_task(cm._prefetch_archives_for_range(
        "BTC/USDT:USDT", first, first + 3 * day_ms - minute, parallel_days=2,
    ))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(finished) == sorted(entered) == ["2020-01-01", "2020-01-02"]
