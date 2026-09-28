"""Offline timing attribution and routing, independent of wall-clock speed."""
import asyncio
from contextlib import asynccontextmanager

import pytest

from live import ema_timing


@pytest.mark.asyncio
async def test_nested_timings_counts_bounds_and_task_isolation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ema_timing, "perf_counter", lambda: clock[0])
    timings = ema_timing.EmaBundleTimings()

    async def load(symbol):
        with ema_timing.measure("candles"):
            with ema_timing.measure("disk_load"):
                clock[0] += 0.002
        await asyncio.sleep(0)
        return symbol

    assert await asyncio.gather(*(
        timings.run_symbol(str(i), load) for i in range(12)
    )) == [str(i) for i in range(12)]
    summary = timings.summary()
    assert summary["symbol_count"] == 12
    assert summary["symbols_omitted"] == 4
    assert len(summary["slowest_symbols"]) == 8
    assert summary["stage_totals"]["disk_load_ms"] == pytest.approx(24.0)
    assert summary["stage_totals"]["disk_load_calls"] == 12
    assert summary["stage_totals"]["candles_ms"] == pytest.approx(24.0)
    assert all(row["disk_load_calls"] == 1 for row in summary["slowest_symbols"])
    assert ema_timing._current.get() is None


@pytest.mark.asyncio
async def test_context_entry_excludes_body_and_preserves_exception(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ema_timing, "perf_counter", lambda: clock[0])
    released = []

    @ema_timing.timed_async_entry("fetch_lock_wait")
    @asynccontextmanager
    async def lock():
        clock[0] += 0.003
        try:
            yield 7
        finally:
            released.append(True)
            clock[0] += 0.005

    async def load(symbol):
        async with lock() as value:
            assert value == 7
            clock[0] += 0.010
            raise RuntimeError("unchanged")

    timings = ema_timing.EmaBundleTimings()
    with pytest.raises(RuntimeError, match="unchanged"):
        await timings.run_symbol("BTC/USDT:USDT", load)
    assert released == [True]
    assert timings.summary()["stage_totals"]["fetch_lock_wait_ms"] == pytest.approx(3.0)
    assert timings.summary()["symbol_elapsed_ms"] == pytest.approx(18.0)
    assert ema_timing._current.get() is None


@pytest.mark.asyncio
async def test_cancellation_resets_context_and_records_requested_actual_sleep(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ema_timing, "perf_counter", lambda: clock[0])
    timings = ema_timing.EmaBundleTimings()

    async def load(symbol):
        with ema_timing.measure("remote_spacing_sleep", requested_ms=200):
            clock[0] += 0.05
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await timings.run_symbol("BTC/USDT:USDT", load)
    totals = timings.summary()["stage_totals"]
    assert totals["remote_spacing_sleep_requested_ms"] == 200
    assert totals["remote_spacing_sleep_ms"] == pytest.approx(50)
    assert ema_timing._current.get() is None


@pytest.mark.asyncio
async def test_real_candle_stages_and_pacing_are_observed(tmp_path, monkeypatch):
    from candlestick_manager import CandlestickManager

    class Exchange:
        id = "okx"

        async def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
            return [[int(since or 0), 1, 1, 1, 1, 1]]

    cm = CandlestickManager(
        exchange=Exchange(), exchange_name="okx", cache_dir=str(tmp_path),
        remote_fetch_min_interval_ms=40,
    )
    clock_ms = [1000]
    monkeypatch.setattr("candlestick_manager._utc_now_ms", lambda: clock_ms[0])

    async def sleep(seconds, *, stage):
        assert stage == "remote_fetch_spacing"
        clock_ms[0] += round(seconds * 1000)

    monkeypatch.setattr(cm, "_sleep_interruptible", sleep)
    timings = ema_timing.EmaBundleTimings()

    async def load(symbol):
        async with cm._acquire_fetch_lock(symbol, "1m"):
            cm._load_from_disk(symbol, 0, 60_000)
            for _ in range(2):
                await cm._ccxt_fetch_ohlcv_once(symbol, 0, 1, timeframe="1m")

    await timings.run_symbol("BTC/USDT:USDT", load)
    totals = timings.summary()["stage_totals"]
    assert totals["fetch_lock_wait_calls"] == 1
    assert totals["disk_load_calls"] == 1
    assert totals["remote_fetch_calls"] == 2
    assert totals["remote_spacing_sleep_calls"] == 1
    assert totals["remote_spacing_sleep_requested_ms"] > 0
    assert totals["remote_spacing_sleep_ms"] > 0
