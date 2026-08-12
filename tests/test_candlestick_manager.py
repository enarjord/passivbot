import asyncio
import builtins
import logging
import os
import sys
import time
import math
import json
import types
import zlib
from collections import OrderedDict
import pytest
import numpy as np
from pathlib import Path

import candlestick_manager
from candlestick_manager import (
    CandlestickManager,
    CANDLE_DTYPE,
    GAP_REASON_AUTO,
    GAP_REASON_FETCH_FAILED,
    GAP_REASON_NO_ARCHIVE,
    GAP_REASON_NO_TRADES,
    ONE_MIN_MS,
    OhlcvFetchError,
    OhlcvTerminalEmptyPage,
    _GAP_MAX_RETRIES,
    _GAP_PERSISTENT_RETRY_MS,
    _GATEIO_RECENT_1M_LIMIT_CANDLES,
    _floor_minute,
    fetch_candles_with_resolution_ladder,
    sanitize_remote_fetch_diagnostic,
    synthesize_1m_from_higher_tf,
)
from logging_setup import DEFAULT_DATEFMT, DEFAULT_FORMAT_WITH_PREFIX


def _resolution_candles(*minutes, close_offset=0.0):
    return np.array(
        [
            (
                minute * ONE_MIN_MS,
                100.0 + close_offset + minute,
                101.0 + close_offset + minute,
                99.0 + close_offset + minute,
                100.0 + close_offset + minute,
                1.0,
            )
            for minute in minutes
        ],
        dtype=CANDLE_DTYPE,
    )


@pytest.mark.asyncio
async def test_resolution_ladder_stops_when_exact_1m_reaches_start():
    calls = []

    async def fetch(*, timeframe, start_ts, end_ts):
        calls.append((timeframe, start_ts, end_ts))
        assert timeframe == "1m"
        return _resolution_candles(0, 1, 2)

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=2 * ONE_MIN_MS,
    )

    assert [call[0] for call in calls] == ["1m"]
    assert result.source_counts == {"1m": 3}
    assert result.failures == {}
    assert result.candles["ts"].tolist() == [0, ONE_MIN_MS, 2 * ONE_MIN_MS]


@pytest.mark.asyncio
async def test_resolution_ladder_fills_only_prefix_before_exact_1m():
    calls = []

    async def fetch(*, timeframe, start_ts, end_ts):
        calls.append((timeframe, start_ts, end_ts))
        if timeframe == "1m":
            return _resolution_candles(10, 11, 13)
        if timeframe == "5m":
            return _resolution_candles(0, 5, close_offset=1_000.0)
        raise AssertionError(f"unexpected timeframe {timeframe}")

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=13 * ONE_MIN_MS,
    )

    assert [call[0] for call in calls] == ["1m", "5m"]
    assert calls[1][2] == 9 * ONE_MIN_MS
    assert result.source_counts == {"5m": 10, "1m": 3}
    assert 12 * ONE_MIN_MS not in set(result.candles["ts"])
    exact_row = result.candles[result.candles["ts"] == 10 * ONE_MIN_MS][0]
    assert float(exact_row["c"]) == pytest.approx(110.0)


@pytest.mark.asyncio
async def test_resolution_ladder_rejects_coarse_bucket_crossing_1m_boundary():
    async def fetch(*, timeframe, start_ts, end_ts):
        if timeframe == "1m":
            return _resolution_candles(12, 13)
        if timeframe == "5m":
            return _resolution_candles(5, 10, close_offset=500.0)
        return _resolution_candles()

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=5 * ONE_MIN_MS,
        end_ts=13 * ONE_MIN_MS,
    )

    timestamps = set(result.candles["ts"])
    expected_timestamps = {minute * ONE_MIN_MS for minute in range(5, 10)} | {
        12 * ONE_MIN_MS,
        13 * ONE_MIN_MS,
    }
    assert timestamps == expected_timestamps
    assert result.source_counts == {"5m": 5, "1m": 2}


@pytest.mark.asyncio
async def test_manager_resolution_ladder_reuses_canonical_candle_reads(tmp_path):
    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: 13 * ONE_MIN_MS
    calls = []

    async def fake_get_candles(
        symbol, *, start_ts, end_ts, strict, timeframe=None, **_kwargs
    ):
        calls.append((symbol, timeframe or "1m", start_ts, end_ts, strict))
        if timeframe is None:
            return _resolution_candles(10, 11)
        if timeframe == "5m":
            return _resolution_candles(0, 5, close_offset=500.0)
        return _resolution_candles()

    manager.get_candles = fake_get_candles

    result = await manager.get_candles_with_resolution_ladder(
        "TEST/USDT", start_ts=0, end_ts=11 * ONE_MIN_MS, strict=False
    )

    assert [call[1] for call in calls] == ["1m", "5m"]
    assert result.source_counts == {"5m": 10, "1m": 2}
    assert result.candles.size == 12


@pytest.mark.asyncio
async def test_resolution_ladder_uses_finer_sources_before_one_hour():
    calls = []

    async def fetch(*, timeframe, start_ts, end_ts):
        calls.append(timeframe)
        if timeframe == "1m":
            return _resolution_candles(*range(60, 75))
        if timeframe == "5m":
            return _resolution_candles(50, close_offset=500.0)
        if timeframe == "15m":
            return _resolution_candles(30, close_offset=1_500.0)
        if timeframe == "1h":
            return _resolution_candles(0, close_offset=6_000.0)
        raise AssertionError(timeframe)

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=74 * ONE_MIN_MS,
    )

    assert calls == ["1m", "5m", "15m", "1h"]
    assert result.source_counts == {"1h": 40, "15m": 15, "5m": 5, "1m": 15}
    assert result.candles.size == 75


@pytest.mark.asyncio
async def test_resolution_ladder_skips_unsupported_tier_and_records_failures():
    calls = []

    async def fetch(*, timeframe, start_ts, end_ts):
        calls.append(timeframe)
        if timeframe in {"1m", "15m"}:
            raise RuntimeError(f"{timeframe} unavailable")
        if timeframe == "1h":
            return _resolution_candles(0)
        raise AssertionError(timeframe)

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=59 * ONE_MIN_MS,
        supported_timeframes={"1m", "15m", "1h"},
    )

    assert calls == ["1m", "15m", "1h"]
    assert set(result.failures) == {"1m", "15m"}
    assert result.source_counts == {"1h": 60}
    assert result.candles.size == 60


def test_synthesize_1m_from_one_hour_candle():
    result = synthesize_1m_from_higher_tf(_resolution_candles(0), 60)

    assert result.size == 60
    assert int(result[0]["ts"]) == 0
    assert int(result[-1]["ts"]) == 59 * ONE_MIN_MS
    with pytest.raises(ValueError, match="must be > 1"):
        synthesize_1m_from_higher_tf(_resolution_candles(0), 1)


def test_normalize_ccxt_ohlcv_filters_nonfinite_and_nonpositive_rows(tmp_path):
    class _Ex:
        id = "binance"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    base = _floor_minute(int(time.time() * 1000)) - 10 * ONE_MIN_MS
    rows = [
        [base, 100.0, 101.0, 99.0, 100.5, 7.0],
        [base + ONE_MIN_MS, 100.0, float("nan"), 99.0, 100.5, 7.0],
        [base + 2 * ONE_MIN_MS, 100.0, 101.0, 0.0, 100.5, 7.0],
        [base + 3 * ONE_MIN_MS, 100.0, 101.0, 99.0, 100.5, -1.0],
    ]

    arr = cm._normalize_ccxt_ohlcv(rows)

    assert arr.size == 1
    assert int(arr[0]["ts"]) == base
    assert float(arr[0]["c"]) == pytest.approx(100.5)


def test_ema_series_skips_leading_nonfinite_without_poisoning_window(tmp_path):
    class _Ex:
        id = "binance"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    values = np.asarray([float("nan"), 1.0, 3.0], dtype=np.float64)

    out = cm._ema_series(values, span=3.0)

    assert math.isnan(float(out[0]))
    assert float(out[-1]) == pytest.approx(2.0)
    assert math.isnan(float(cm._ema_series(np.asarray([float("nan")]), span=3.0)[-1]))


@pytest.mark.parametrize("span", [1.0, 2.5, 10.0, 100.0])
def test_final_ema_matches_full_series_without_allocation(tmp_path, span):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="test",
        cache_dir=str(tmp_path / "caches"),
    )
    values = np.asarray(
        [float("nan"), 1.0, 2.0, float("nan"), -3.0, 8.0, 13.0],
        dtype=np.float64,
    )

    assert cm._ema(values, span) == pytest.approx(
        float(cm._ema_series(values, span)[-1]),
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.parametrize("span", [1.0, 2.5, 10.0, 100.0, 2_000.0])
def test_rust_ema_last_matches_python_reference(span):
    import passivbot_rust as pbr

    values = np.asarray(
        [float("nan"), 1.0, 2.0, float("nan"), -3.0, 8.0, 13.0],
        dtype=np.float64,
    )
    alpha = 2.0 / (span + 1.0)
    expected = 1.0
    for value in [2.0, -3.0, 8.0, 13.0]:
        expected = alpha * value + (1.0 - alpha) * expected
    assert pbr.ema_last(values, span) == pytest.approx(
        expected,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.asyncio
async def test_latest_ema_log_range_ignores_leading_nonfinite_sample(tmp_path):
    class _Ex:
        id = "binance"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    base = _floor_minute(int(time.time() * 1000)) - 10 * ONE_MIN_MS
    arr = np.array(
        [
            (base, 100.0, float("nan"), 99.0, 100.0, 1.0),
            (base + ONE_MIN_MS, 100.0, 102.0, 100.0, 101.0, 1.0),
            (base + 2 * ONE_MIN_MS, 101.0, 104.0, 101.0, 103.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )

    async def latest_range(_span, *, period_ms=ONE_MIN_MS):
        return base, base + 2 * ONE_MIN_MS

    async def get_candles(*_args, **_kwargs):
        return arr

    cm._latest_finalized_range = latest_range
    cm.get_candles = get_candles

    val = await cm.get_latest_ema_log_range("BAD/USDT:USDT", span=3.0)

    assert math.isfinite(val)
    assert val > 0.0


@pytest.mark.asyncio
async def test_latest_cached_ema_metrics_carries_values_without_tail_zeroing(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "STALE/USDT:USDT"
    now_ms = 10 * ONE_MIN_MS
    latest_expected = 9 * ONE_MIN_MS
    last_cached = 7 * ONE_MIN_MS
    cm._now_ms = lambda: now_ms
    candles = np.array(
        [
            (5 * ONE_MIN_MS, 100.0, 102.0, 99.0, 101.0, 2.0),
            (6 * ONE_MIN_MS, 101.0, 103.0, 100.0, 102.0, 3.0),
            (last_cached, 102.0, 105.0, 101.0, 104.0, 4.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, candles, timeframe="1m", merge_cache=True, last_refresh_ms=now_ms)

    out = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"qv": 2.0, "log_range": 2.0},
        max_staleness_ms=latest_expected - last_cached,
        window_candles=3,
    )

    assert out["qv"] > 0.0
    assert out["log_range"] > 0.0
    too_stale = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"qv": 2.0, "log_range": 2.0},
        max_staleness_ms=ONE_MIN_MS,
        window_candles=3,
    )
    assert too_stale == {}


@pytest.mark.asyncio
async def test_latest_cached_ema_metric_spans_loads_one_window_for_all_spans(
    tmp_path,
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "BATCHED/USDT:USDT"
    now_ms = 10 * ONE_MIN_MS
    cm._now_ms = lambda: now_ms
    candles = np.array(
        [
            (
                minute * ONE_MIN_MS,
                100.0 + minute,
                102.0 + minute,
                99.0 + minute,
                101.0 + minute,
                2.0 + minute,
            )
            for minute in range(5, 10)
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(
        symbol,
        candles,
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    original_get_candles = cm.get_candles
    calls = 0

    async def counted_get_candles(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_get_candles(*args, **kwargs)

    cm.get_candles = counted_get_candles
    out = await cm.get_latest_cached_ema_metric_spans(
        symbol,
        {"qv": [2.0, 4.0], "log_range": [2.0, 4.0]},
        max_staleness_ms=0,
    )

    assert calls == 1
    assert set(out) == {"qv", "log_range"}
    assert set(out["qv"]) == {2.0, 4.0}
    assert set(out["log_range"]) == {2.0, 4.0}
    assert all(
        math.isfinite(value)
        for metric_values in out.values()
        for value in metric_values.values()
    )


@pytest.mark.asyncio
async def test_latest_cached_ema_metric_spans_preserves_complete_shorter_window(
    tmp_path,
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MIXED/USDT:USDT"
    now_ms = 10 * ONE_MIN_MS
    cm._now_ms = lambda: now_ms
    candles = np.array(
        [
            (
                minute * ONE_MIN_MS,
                100.0 + minute,
                102.0 + minute,
                99.0 + minute,
                101.0 + minute,
                2.0 + minute,
            )
            for minute in range(5, 10)
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(
        symbol,
        candles,
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    original_get_candles = cm.get_candles
    calls = 0

    async def counted_get_candles(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_get_candles(*args, **kwargs)

    cm.get_candles = counted_get_candles
    out = await cm.get_latest_cached_ema_metric_spans(
        symbol,
        {"qv": [3.0, 8.0], "log_range": [3.0, 8.0]},
        max_staleness_ms=0,
        window_candles=8,
    )

    assert calls == 1
    assert set(out["qv"]) == {3.0}
    assert set(out["log_range"]) == {3.0}
    assert all(
        math.isfinite(value)
        for metric_values in out.values()
        for value in metric_values.values()
    )


@pytest.mark.asyncio
async def test_latest_cached_h1_ema_metrics_use_h1_index(tmp_path):
    class _Ex:
        id = "weex"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="weex", cache_dir=str(tmp_path / "caches")
    )
    symbol = "STALE-H1/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    now_ms = 10 * hour_ms
    latest_expected = 9 * hour_ms
    last_cached = 8 * hour_ms
    cm._now_ms = lambda: now_ms
    candles = np.array(
        [
            (5 * hour_ms, 100.0, 102.0, 99.0, 101.0, 2.0),
            (6 * hour_ms, 101.0, 103.0, 100.0, 102.0, 3.0),
            (7 * hour_ms, 102.0, 105.0, 101.0, 104.0, 4.0),
            (last_cached, 104.0, 106.0, 103.0, 105.0, 5.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, candles, timeframe="1h")

    assert cm.get_last_final_ts(symbol, timeframe="1h") == last_cached
    cm._ensure_symbol_index(symbol, timeframe="1h")["meta"]["last_final_ts"] = 0
    assert cm.get_last_final_ts(symbol, timeframe="1h") == last_cached
    assert cm.get_last_final_ts(symbol) == 0
    out = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"log_range": 4.0},
        max_staleness_ms=latest_expected - last_cached,
        window_candles=4,
        timeframe="1h",
    )

    assert out["log_range"] > 0.0
    assert symbol not in cm._ema_cache
    too_stale = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"log_range": 4.0},
        max_staleness_ms=30 * ONE_MIN_MS,
        window_candles=4,
        timeframe="1h",
    )
    assert too_stale == {}


@pytest.mark.asyncio
async def test_latest_cached_h1_ema_rejects_internal_gap_on_non_weex(tmp_path):
    class _Ex:
        id = "bybit"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="bybit", cache_dir=str(tmp_path / "caches")
    )
    symbol = "GAPPED-H1/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    now_ms = 10 * hour_ms
    cm._now_ms = lambda: now_ms
    candles = np.array(
        [
            (5 * hour_ms, 100.0, 102.0, 99.0, 101.0, 2.0),
            (6 * hour_ms, 101.0, 103.0, 100.0, 102.0, 3.0),
            (8 * hour_ms, 104.0, 106.0, 103.0, 105.0, 5.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, candles, timeframe="1h")

    out = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"log_range": 4.0},
        max_staleness_ms=60 * ONE_MIN_MS,
        window_candles=4,
        timeframe="1h",
    )

    assert out == {}
    assert symbol not in cm._ema_cache


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


def test_standardize_gaps_complete_range_returns_equal_independent_copy(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    candles = np.array(
        [
            (minute * ONE_MIN_MS, 100.0, 101.0, 99.0, 100.5, 2.0)
            for minute in range(5)
        ],
        dtype=CANDLE_DTYPE,
    )

    out = cm.standardize_gaps(
        candles,
        start_ts=0,
        end_ts=4 * ONE_MIN_MS,
        assume_sorted=True,
    )

    assert np.array_equal(out, candles)
    assert not np.shares_memory(out, candles)


def test_standardize_gaps_does_not_fill_open_tail_when_disabled(tmp_path):
    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    t0 = _floor_minute(int(time.time() * 1000)) - 3 * ONE_MIN_MS
    t1 = t0 + ONE_MIN_MS
    t2 = t0 + 2 * ONE_MIN_MS
    a = np.array([(t0, 100.0, 100.0, 100.0, 100.0, 1.0)], dtype=CANDLE_DTYPE)

    res = cm.standardize_gaps(
        a, start_ts=t0, end_ts=t2, strict=False, fill_trailing_gaps=False, symbol="TAIL"
    )

    assert list(res["ts"]) == [t0]
    assert not cm._synthetic_timestamps.get("TAIL")

    bounded = np.array(
        [
            (t0, 100.0, 100.0, 100.0, 100.0, 1.0),
            (t2, 102.0, 102.0, 102.0, 102.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    res = cm.standardize_gaps(
        bounded, start_ts=t0, end_ts=t2, strict=False, fill_trailing_gaps=False, symbol="TAIL"
    )

    assert list(res["ts"]) == [t0, t1, t2]
    assert t1 in cm._synthetic_timestamps.get("TAIL", set())


def test_standardize_gaps_strict_does_not_allocate_expected_minutes(
    monkeypatch, tmp_path
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    t0 = 10 * ONE_MIN_MS
    t2 = t0 + 2 * ONE_MIN_MS
    candles = np.array(
        [
            (t0, 1.0, 1.0, 1.0, 1.0, 1.0),
            (t2, 2.0, 2.0, 2.0, 2.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )

    def fail_arange(*_args, **_kwargs):
        raise AssertionError("strict reads must not materialize the minute range")

    monkeypatch.setattr(np, "arange", fail_arange)

    result = cm.standardize_gaps(
        candles,
        start_ts=t0,
        end_ts=t2,
        strict=True,
        excluded_synthetic_ranges=[(t0 + ONE_MIN_MS, t0 + ONE_MIN_MS)],
    )

    assert list(result["ts"]) == [t0, t2]


def test_archive_day_conversion_does_not_fill_edge_gaps(tmp_path):
    import pandas as pd

    class _Ex:
        id = "binance"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    day_key = "2026-04-01"
    start_ts, _end_ts = cm._date_range_of_key(day_key)
    first_real_ts = start_ts + 2 * ONE_MIN_MS

    out = cm._ohlcv_df_to_day_arr(
        pd.DataFrame(
            {
                "timestamp": [first_real_ts],
                "open": [101.0],
                "high": [103.0],
                "low": [99.0],
                "close": [102.0],
                "volume": [7.0],
            }
        ),
        day_key,
    )

    assert out.size == 1
    assert int(out[0]["ts"]) == first_real_ts
    assert float(out[0]["c"]) == pytest.approx(102.0)


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
    cm._log("debug", "get_candles_present_inner", symbol="BTC/USDT:USDT", need_fetch=True)
    cm._log("debug", "legacy_index_built", symbol="BTC/USDT:USDT", legacy_days=1000)
    cm._log("debug", "saved_range", symbol="BTC/USDT:USDT", rows=10)

    messages = [rec.getMessage() for rec in caplog.records]
    assert not any("event=index_cached" in msg for msg in messages)
    assert not any("event=ccxt_fetch_ohlcv" in msg for msg in messages)
    assert not any("event=get_candles_present_inner" in msg for msg in messages)
    assert not any("event=legacy_index_built" in msg for msg in messages)
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


def test_remote_fetch_callback_is_sanitized_and_exception_is_isolated(tmp_path):
    calls = []

    def callback(payload):
        calls.append(payload)
        raise RuntimeError("callback failed")

    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
        remote_fetch_callback=callback,
    )

    url = "https://api.example.invalid/ohlcv?apiKey=SECRET"
    cm._emit_remote_fetch(
        {
            "kind": "ccxt_fetch_ohlcv",
            "stage": "error",
            "url": url,
            "params": {"until": 123, "apiKey": "SECRET"},
            "error_type": "TokenError",
            "error": f"GET {url}",
            "error_repr": f"AuthError({url!r})",
        }
    )

    assert len(calls) == 1
    payload = calls[0]
    assert payload["param_keys"] == ["apiKey", "until"]
    assert len(payload["url_hash"]) == 64
    assert payload["error_type"] == "Error"
    assert "url" not in payload
    assert "params" not in payload
    assert "error" not in payload
    assert "error_repr" not in payload
    assert "SECRET" not in str(payload)
    assert sanitize_remote_fetch_diagnostic(payload) == payload


def test_cache_migration_diagnostics_redact_hostile_exception_text(tmp_path, monkeypatch, caplog):
    secret = "cache-secret-should-not-reach-logs"

    def fail_cleanup(_cache_base):
        raise RuntimeError(secret)

    def fail_migration(_cache_base):
        raise RuntimeError(secret)

    monkeypatch.setattr(candlestick_manager, "_quarantine_root_level_timeframe_debris", fail_cleanup)
    monkeypatch.setattr(candlestick_manager, "standardize_cache_directories", fail_migration)
    caplog.set_level(logging.ERROR)

    CandlestickManager(exchange=None, exchange_name="cache-test", cache_dir=str(tmp_path / "caches"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("Root-level OHLCV cache cleanup failed" in message for message in messages)
    assert any("Cache migration failed" in message for message in messages)
    assert all(secret not in message for message in messages)
    assert all("error_type=RuntimeError" in message for message in messages)
    assert all(record.exc_info is None for record in caplog.records)


def test_gateio_cache_quarantine_failure_keeps_bounded_context(tmp_path, monkeypatch, caplog):
    secret = "gateio-cache-secret"
    cache_base = tmp_path / "ohlcv"
    shard_dir = cache_base / "gateio" / "1m" / "BTC_USDT"
    shard_dir.mkdir(parents=True)
    (shard_dir / "2026-02-06.npy").write_bytes(b"cache")

    def fail_rename(_source, _target):
        raise RuntimeError(secret)

    monkeypatch.setattr(candlestick_manager.os, "rename", fail_rename)
    caplog.set_level(logging.ERROR)

    candlestick_manager._quarantine_gateio_cache_if_stale(str(cache_base), "2026-02-07")

    record = next(record for record in caplog.records if "Failed to move GateIO cache" in record.getMessage())
    message = record.getMessage()
    assert secret not in message
    assert "error_type=RuntimeError" in message
    assert f"cache_base={cache_base / 'gateio'}" in message
    assert "backup=" in message
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_cache_diagnostics_redact_hostile_exception_text_and_keep_context(
    tmp_path, monkeypatch, caplog
):
    secret = "cache-diagnostic-secret"
    symbol = "BTC/USDT:USDT"
    cm = CandlestickManager(exchange=None, exchange_name="cache-test", cache_dir=str(tmp_path / "caches"))
    cm.debug_level = 3
    caplog.set_level(int(getattr(logging, "TRACE", 5)), logger=cm.log.name)

    class FailingLock:
        def release(self):
            raise RuntimeError(secret)

    await cm._release_lock(FailingLock(), str(tmp_path / "fetch.lock"), symbol, "1m")

    index_path = Path(cm._index_path(symbol, tf="1m"))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{}", encoding="utf-8")
    original_open = builtins.open

    def fail_index_open(path, *args, **kwargs):
        if str(path) == str(index_path):
            raise RuntimeError(secret)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_index_open)
    cm._ensure_symbol_index(symbol, tf="1m")
    monkeypatch.setattr(builtins, "open", original_open)

    shard_path = tmp_path / "broken.npy"
    shard_path.write_bytes(b"broken")
    monkeypatch.setattr(candlestick_manager.np, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)))
    cm._load_shard(str(shard_path))

    def fail_disk_load(*_args, **_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(cm, "_load_from_disk", fail_disk_load)
    cm.get_completed_candle_health(symbol, {"1m": 1}, now_ms=2 * ONE_MIN_MS)

    events = []
    monkeypatch.setattr(cm, "_log", lambda level, event, **fields: events.append((level, event, fields)))
    monkeypatch.setattr(cm, "_ensure_symbol_index", lambda *_args, **_kwargs: {"meta": {}, "shards": {}})
    monkeypatch.setattr(cm, "_get_inception_ts", lambda _symbol: None)
    monkeypatch.setattr(
        cm,
        "_prune_pre_inception_gaps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    cm._set_authoritative_start_ts(symbol, ONE_MIN_MS, source="test", save=False)

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_pyarrow.array = lambda values: values
    fake_pyarrow.table = lambda _columns: (_ for _ in ()).throw(RuntimeError(secret))
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)
    cm._save_tradfi_cache(np.empty((0,), dtype=CANDLE_DTYPE), tmp_path / "tradfi.parquet")

    messages = [record.getMessage() for record in caplog.records]
    assert all(secret not in message for message in messages)
    assert all(record.exc_info is None for record in caplog.records)
    assert any("event=fetch_lock_release_error" in message and "symbol=BTC" in message for message in messages)
    assert any("event=index_load_failed" in message and "timeframe=1m" in message for message in messages)
    assert any("Failed loading shard" in message and "error_type=RuntimeError" in message for message in messages)
    assert any("event=candle_health_disk_load_failed" in message for message in messages)
    by_event = {event: fields for _level, event, fields in events}
    assert by_event["prune_pre_inception_gaps_failed"] == {
        "symbol": symbol,
        "error_type": "RuntimeError",
    }
    assert by_event["tradfi_cache_save_error"] == {
        "path": str(tmp_path / "tradfi.parquet"),
        "error_type": "RuntimeError",
    }
    assert secret not in str(events)


@pytest.mark.asyncio
async def test_paginated_cache_callback_failure_is_redacted_and_stops_pagination(
    tmp_path, caplog
):
    secret = "callback-secret-should-not-reach-logs"
    symbol = "BTC/USDT:USDT"
    start_ts = 10 * ONE_MIN_MS
    end_exclusive_ts = start_ts + 2 * ONE_MIN_MS

    class _Exchange:
        id = "cache-test"

    cm = CandlestickManager(
        exchange=_Exchange(),
        exchange_name="cache-test",
        cache_dir=str(tmp_path / "caches"),
    )
    calls = 0

    async def fake_once(
        _symbol,
        since_ms,
        _limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        nonlocal calls
        calls += 1
        return [[since_ms, 1.0, 1.0, 1.0, 1.0, 1.0]]

    def fail_on_batch(_arr):
        raise RuntimeError(secret)

    cm._ccxt_fetch_ohlcv_once = fake_once
    caplog.set_level(logging.ERROR, logger=cm.log.name)

    result = await cm._fetch_ohlcv_paginated(
        symbol,
        start_ts,
        end_exclusive_ts,
        on_batch=fail_on_batch,
    )

    record = next(
        record
        for record in caplog.records
        if "on_batch callback failed" in record.getMessage()
    )
    assert calls == 1
    assert result.size == 1
    assert secret not in record.getMessage()
    assert record.error_type == "RuntimeError"
    assert not hasattr(record, "error")
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_ccxt_fetch_warning_uses_bounded_signature_and_sanitizes_callback_payload(
    tmp_path, monkeypatch
):
    url = "https://api.example.invalid/ohlcv?apiKey=SECRET&signature=abc"

    class UrlBearingError(RuntimeError):
        pass

    class _Ex:
        id = "binance"

        async def fetch_ohlcv(self, *_args, **_kwargs):
            raise UrlBearingError(url)

    callback_payloads = []
    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="binance",
        cache_dir=str(tmp_path / "caches"),
        debug=1,
        remote_fetch_callback=callback_payloads.append,
    )

    async def no_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(cm, "_sleep_interruptible", no_sleep)
    monkeypatch.setattr("candlestick_manager.time.monotonic", lambda: 50.0)
    warning_records = []
    rendered_warnings = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            record.log_prefix = "kucoin"
            warning_records.append(record)
            rendered_warnings.append(self.format(record))

    handler = CaptureHandler(level=logging.WARNING)
    formatter = logging.Formatter(DEFAULT_FORMAT_WITH_PREFIX, datefmt=DEFAULT_DATEFMT)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    capture_log = logging.Logger(
        "test.candlestick_manager.ccxt_fetch_warning", level=logging.WARNING
    )
    capture_log.propagate = False
    capture_log.addHandler(handler)
    original_global_disable = logging.root.manager.disable
    original_log = cm.log
    logging.disable(logging.NOTSET)
    cm.log = capture_log
    try:
        with pytest.raises(OhlcvFetchError):
            await cm._ccxt_fetch_ohlcv_once(
                "BTC/USDT:USDT",
                since_ms=1_723_456_000_000,
                limit=100,
                end_exclusive_ms=1_723_456_060_000,
                timeframe="1H",
            )
    finally:
        cm.log = original_log
        logging.disable(original_global_disable)
        capture_log.removeHandler(handler)
        handler.close()

    warning_records = [
        record
        for record in warning_records
        if "event=ccxt_fetch_ohlcv_failed" in record.getMessage()
    ]
    assert len(warning_records) == 2
    retry_warning = warning_records[0].getMessage()
    exhausted_warning = warning_records[1].getMessage()
    for field in (
        "exchange=binance",
        "symbol=BTC",
        "tf=1h",
        "attempt=1",
        "max_attempts=5",
        "elapsed_ms=",
        "error_type=RuntimeError",
        "action=retry",
    ):
        assert field in retry_warning
    for field in (
        "exchange=binance",
        "symbol=BTC",
        "tf=1h",
        "attempt=5",
        "max_attempts=5",
        "elapsed_ms=",
        "error_type=RuntimeError",
        "action=exhausted",
    ):
        assert field in exhausted_warning
    for raw_value in (
        url,
        repr(UrlBearingError(url)),
        "params=",
        "error=",
        "error_repr=",
    ):
        assert all(raw_value not in warning for warning in (retry_warning, exhausted_warning))
    assert len(retry_warning) <= 240
    assert len(exhausted_warning) <= 240
    assert len(rendered_warnings) == 2
    assert all(len(warning) <= 240 for warning in rendered_warnings)
    assert all("called_by=" not in warning for warning in rendered_warnings)
    assert all("[kucoin]" in warning for warning in rendered_warnings)

    error_payloads = [payload for payload in callback_payloads if payload.get("stage") == "error"]
    assert error_payloads
    assert error_payloads[0]["param_keys"] == ["until"]
    assert error_payloads[0]["error_type"] == "RuntimeError"
    assert "params" not in error_payloads[0]
    assert "error" not in error_payloads[0]
    assert "error_repr" not in error_payloads[0]
    assert url not in str(error_payloads[0])


@pytest.mark.asyncio
async def test_ccxt_fetch_debug_log_keeps_param_keys_without_values(tmp_path, caplog):
    class _Ex:
        id = "bybit"

        async def fetch_ohlcv(self, *_args, **_kwargs):
            return []

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="bybit",
        cache_dir=str(tmp_path / "caches"),
        debug=3,
    )
    caplog.set_level(int(getattr(logging, "TRACE", 5)), logger=cm.log.name)

    await cm._ccxt_fetch_ohlcv_once(
        "BTC/USDT:USDT",
        since_ms=1_723_456_000_000,
        limit=100,
        end_exclusive_ms=1_723_456_060_000,
        timeframe="1m",
    )

    fetch_lines = [
        record.getMessage()
        for record in caplog.records
        if "event=ccxt_fetch_ohlcv " in record.getMessage()
    ]
    assert len(fetch_lines) == 1
    assert "param_keys=['category']" in fetch_lines[0]
    assert "params=" not in fetch_lines[0]
    assert "linear" not in fetch_lines[0]


@pytest.mark.asyncio
async def test_archive_fetch_diagnostics_keep_only_url_hash_and_error_type(tmp_path, monkeypatch):
    url = "https://data.example.invalid/archive.zip?apiKey=SECRET&signature=abc"

    class UrlBearingError(RuntimeError):
        pass

    class _Response:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            raise UrlBearingError(url)

    class _Session:
        def get(self, _url):
            return _Response()

    cm = CandlestickManager(exchange=None, exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    callback_payloads = []
    logs = []
    cm._remote_fetch_callback = callback_payloads.append

    async def fake_get_session():
        return _Session()

    monkeypatch.setattr(cm, "_get_http_session", fake_get_session)
    monkeypatch.setattr(cm, "_log", lambda level, event, **fields: logs.append((level, event, fields)))

    with pytest.raises(UrlBearingError):
        await cm._archive_fetch_bytes(url)

    by_event = {event: fields for _level, event, fields in logs}
    assert by_event["archive_http_get"]["url_hash"]
    assert by_event["archive_http_error"] == {
        "url_hash": by_event["archive_http_get"]["url_hash"],
        "error_type": "RuntimeError",
    }
    assert all("url" not in payload for payload in callback_payloads)
    assert all("error" not in payload and "error_repr" not in payload for payload in callback_payloads)
    assert all("SECRET" not in str(payload) for payload in callback_payloads)


@pytest.mark.asyncio
async def test_archive_day_warning_keeps_only_bounded_error_type(tmp_path, monkeypatch):
    url = "https://data.example.invalid/archive.zip?apiKey=SECRET&signature=abc"

    class UrlBearingError(RuntimeError):
        pass

    cm = CandlestickManager(exchange=None, exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    logs = []
    monkeypatch.setattr(cm, "_archive_supported", lambda: True)
    monkeypatch.setattr(cm, "_get_authoritative_start_ts", lambda _symbol: None)
    monkeypatch.setattr(cm, "_date_keys_between", lambda _start, _end: {"1970-01-01": (0, 86_340_000)})
    monkeypatch.setattr(cm, "_iter_shard_paths", lambda _symbol, tf: {})
    monkeypatch.setattr(cm, "_get_legacy_shard_paths", lambda _symbol, _tf: {})
    monkeypatch.setattr(cm, "_ensure_symbol_index", lambda _symbol, tf: {"shards": {}})
    monkeypatch.setattr(cm, "_log", lambda level, event, **fields: logs.append((level, event, fields)))

    async def fail_archive_day(_symbol, _day_key):
        raise UrlBearingError(url)

    monkeypatch.setattr(cm, "_archive_fetch_day", fail_archive_day)

    await cm._prefetch_archives_for_range("BTC/USDT:USDT", 0, 86_340_000, parallel_days=1)

    warnings = [fields for level, event, fields in logs if level == "warning" and event == "archive_day_failed"]
    assert warnings == [{"symbol": "BTC/USDT:USDT", "day": "1970-01-01", "error_type": "RuntimeError"}]


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
async def test_get_candles_cache_only_does_not_remote_fetch(tmp_path, monkeypatch):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    symbol = "CACHE/USDT"
    latest_final = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    stale_ts = latest_final - 5 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(stale_ts, 100.0, 101.0, 99.0, 100.5, 1.0)],
        dtype=CANDLE_DTYPE,
    )

    async def fail_refresh(*args, **kwargs):
        raise AssertionError("refresh must not be called for cache-only get_candles")

    async def fail_fetch(*args, **kwargs):
        raise AssertionError("remote fetch must not be called for cache-only get_candles")

    monkeypatch.setattr(cm, "refresh", fail_refresh)
    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fail_fetch)
    monkeypatch.setattr(cm, "_prefetch_archives_for_range", fail_refresh)

    res = await cm.get_candles(
        symbol,
        start_ts=stale_ts,
        end_ts=latest_final,
        max_age_ms=0,
        allow_remote_fetch=False,
    )

    assert list(res["ts"]) == [stale_ts]


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
async def test_get_latest_ema_metric_spans_batches_and_matches_individual_helpers(
    tmp_path, monkeypatch
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    symbol = "BATCH/USDT"
    end_ts = fixed_now_ms - ONE_MIN_MS
    rows = []
    for index in range(12):
        ts = end_ts - (11 - index) * ONE_MIN_MS
        close = 100.0 + index
        rows.append(
            (
                ts,
                close,
                close + 1.0,
                close - 1.0,
                close,
                1.0 + index,
            )
        )
    cm._cache[symbol] = np.array(rows, dtype=CANDLE_DTYPE)

    original_get_candles = cm.get_candles
    calls = {"count": 0}

    async def counted_get_candles(*args, **kwargs):
        calls["count"] += 1
        return await original_get_candles(*args, **kwargs)

    monkeypatch.setattr(cm, "get_candles", counted_get_candles)
    spans = {
        "close": [3.0, 5.0, 8.5],
        "qv": [4.0, 7.0],
        "log_range": [6.0],
    }
    batched = await cm.get_latest_ema_metric_spans(symbol, spans)
    assert calls["count"] == 1

    cm._ema_cache.clear()
    expected = {
        "close": {
            span: await cm.get_latest_ema_close(symbol, span)
            for span in spans["close"]
        },
        "qv": {
            span: await cm.get_latest_ema_quote_volume(symbol, span)
            for span in spans["qv"]
        },
        "log_range": {
            span: await cm.get_latest_ema_log_range(symbol, span)
            for span in spans["log_range"]
        },
    }
    for metric_key, values in expected.items():
        for span, value in values.items():
            assert batched[metric_key][span] == pytest.approx(value)


@pytest.mark.asyncio
async def test_latest_ema_helpers_reject_short_tail_without_caching(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    symbol = "SHORTTAIL/USDT"
    span = 3.0
    end_ts = fixed_now_ms - ONE_MIN_MS
    start_ts = end_ts - 2 * ONE_MIN_MS
    stale = np.array(
        [
            (start_ts, 10.0, 11.0, 9.0, 10.0, 1.0),
            (start_ts + ONE_MIN_MS, 11.0, 12.0, 10.0, 11.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )

    async def fake_get_candles(*_args, **_kwargs):
        return stale

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    close = await cm.get_latest_ema_close(symbol, span, max_age_ms=60_000)
    quote_volume = await cm.get_latest_ema_quote_volume(symbol, span, max_age_ms=60_000)
    log_range = await cm.get_latest_ema_log_range(symbol, span, max_age_ms=60_000)
    metrics = await cm.get_latest_ema_metrics(
        symbol,
        {"close": span, "qv": span, "log_range": span},
        max_age_ms=60_000,
    )

    assert math.isnan(close)
    assert math.isnan(quote_volume)
    assert math.isnan(log_range)
    assert all(math.isnan(metrics[key]) for key in ("close", "qv", "log_range"))
    assert cm._ema_cache.get(symbol, {}) == {}


@pytest.mark.asyncio
async def test_latest_ema_helpers_reject_internal_candle_gap(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange=None, exchange_name="weex", cache_dir=str(tmp_path / "caches"))
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    symbol = "GAPPED/USDT"
    span = 3.0
    end_ts = fixed_now_ms - ONE_MIN_MS
    start_ts = end_ts - 2 * ONE_MIN_MS
    gapped = np.array(
        [
            (start_ts, 10.0, 11.0, 9.0, 10.0, 1.0),
            (end_ts, 12.0, 13.0, 11.0, 12.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )

    async def fake_get_candles(*_args, **_kwargs):
        return gapped

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    close = await cm.get_latest_ema_close(symbol, span)
    quote_volume = await cm.get_latest_ema_quote_volume(symbol, span)
    log_range = await cm.get_latest_ema_log_range(symbol, span)
    metrics = await cm.get_latest_ema_metrics(
        symbol, {"close": span, "qv": span, "log_range": span}
    )

    assert math.isnan(close)
    assert math.isnan(quote_volume)
    assert math.isnan(log_range)
    assert all(math.isnan(metrics[key]) for key in ("close", "qv", "log_range"))


@pytest.mark.asyncio
async def test_latest_ema_helpers_reject_unverified_hyperliquid_gap(
    tmp_path, monkeypatch
):
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "GAPPED/USDC:USDC"
    span = 3.0
    end_ts = fixed_now_ms - ONE_MIN_MS
    start_ts = end_ts - 2 * ONE_MIN_MS
    missing_ts = start_ts + ONE_MIN_MS
    gapped = np.array(
        [
            (start_ts, 10.0, 11.0, 9.0, 10.0, 1.0),
            (end_ts, 12.0, 13.0, 11.0, 12.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        missing_ts,
        missing_ts,
        reason=GAP_REASON_FETCH_FAILED,
    )

    async def fake_get_candles(*_args, **_kwargs):
        return gapped

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    close = await cm.get_latest_ema_close(symbol, span)
    quote_volume = await cm.get_latest_ema_quote_volume(symbol, span)
    log_range = await cm.get_latest_ema_log_range(symbol, span)
    metrics = await cm.get_latest_ema_metrics(
        symbol, {"close": span, "qv": span, "log_range": span}
    )
    with pytest.raises(RuntimeError, match="unverified internal candle gap"):
        await cm.get_projected_open_tail_ema_metrics(
            symbol,
            {"close": [span]},
            latest_expected_ts=end_ts,
            last_cached_ts=end_ts,
            max_tail_gap_ms=5 * ONE_MIN_MS,
        )

    assert math.isnan(close)
    assert math.isnan(quote_volume)
    assert math.isnan(log_range)
    assert all(math.isnan(metrics[key]) for key in ("close", "qv", "log_range"))
    assert cm._ema_cache.get(symbol, {}) == {}


@pytest.mark.asyncio
async def test_latest_ema_accepts_complete_rows_despite_stale_gap_metadata(
    tmp_path, monkeypatch
):
    fixed_now_ms = 1725590400000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "RECOVERED/USDC:USDC"
    span = 3.0
    end_ts = fixed_now_ms - ONE_MIN_MS
    start_ts = end_ts - 2 * ONE_MIN_MS
    recovered_ts = start_ts + ONE_MIN_MS
    complete = np.array(
        [
            (start_ts, 10.0, 11.0, 9.0, 10.0, 1.0),
            (recovered_ts, 11.0, 12.0, 10.0, 11.0, 1.0),
            (end_ts, 12.0, 13.0, 11.0, 12.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        recovered_ts,
        recovered_ts,
        reason=GAP_REASON_FETCH_FAILED,
    )

    async def fake_get_candles(*_args, **_kwargs):
        return complete

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    result = await cm.get_latest_ema_close(symbol, span)

    assert math.isfinite(result)
    assert ("close", span, str(ONE_MIN_MS)) in cm._ema_cache[symbol]


@pytest.mark.asyncio
async def test_fake_live_ema_helpers_accept_authoritative_sparse_timeline(
    tmp_path, monkeypatch
):
    fixed_now_ms = 1725590400000
    cm = CandlestickManager(
        exchange=None,
        exchange_name="fake",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "SPARSE/USDT:USDT"
    span = 3.0
    end_ts = fixed_now_ms - ONE_MIN_MS
    start_ts = end_ts - 2 * ONE_MIN_MS
    missing_ts = start_ts + ONE_MIN_MS
    sparse = np.array(
        [
            (start_ts, 10.0, 11.0, 9.0, 10.0, 1.0),
            (end_ts, 12.0, 13.0, 11.0, 12.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._now_ms = lambda: fixed_now_ms
    cm._add_known_gap(
        symbol,
        missing_ts,
        missing_ts,
        reason=GAP_REASON_FETCH_FAILED,
    )

    async def fake_get_candles(*_args, **_kwargs):
        return sparse

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    assert math.isfinite(await cm.get_latest_ema_close(symbol, span))
    projected = await cm.get_projected_open_tail_ema_metrics(
        symbol,
        {"close": [span]},
        latest_expected_ts=end_ts,
        last_cached_ts=end_ts,
        max_tail_gap_ms=5 * ONE_MIN_MS,
    )
    assert math.isfinite(projected["close"][span])


@pytest.mark.asyncio
async def test_stock_perp_latest_emas_do_not_bypass_open_tail_policy(
    tmp_path, monkeypatch
):
    fixed_now_ms = 1725811200000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    cm = CandlestickManager(exchange=None, exchange_name="hyperliquid", cache_dir=str(tmp_path / "caches"))
    symbol = "xyz:DELL/USDC:USDC"
    last_final = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    seed_ts = last_final - 60 * ONE_MIN_MS
    seed_close = 123.45
    cm._cache[symbol] = np.array(
        [(seed_ts, seed_close, seed_close, seed_close, seed_close, 10.0)],
        dtype=CANDLE_DTYPE,
    )

    close = await cm.get_latest_ema_close(
        symbol, 5.0, allow_remote_fetch=False
    )
    log_range = await cm.get_latest_ema_log_range(
        symbol, 5.0, allow_remote_fetch=False
    )

    assert math.isnan(close)
    assert math.isnan(log_range)
    assert last_final not in cm._synthetic_timestamps.get(symbol, set())


@pytest.mark.asyncio
async def test_crypto_latest_emas_do_not_fill_open_tail_by_default(tmp_path, monkeypatch):
    fixed_now_ms = 1725811200000
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    cm = CandlestickManager(exchange=None, exchange_name="binance", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"
    last_final = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    seed_ts = last_final - 60 * ONE_MIN_MS
    seed_close = 123.45
    cm._cache[symbol] = np.array(
        [(seed_ts, seed_close, seed_close, seed_close, seed_close, 10.0)],
        dtype=CANDLE_DTYPE,
    )

    close = await cm.get_latest_ema_close(symbol, 5.0, allow_remote_fetch=False)

    assert math.isnan(close)
    assert not cm._synthetic_timestamps.get(symbol)


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


def test_disk_load_observer_receives_summary_and_is_best_effort(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "LOAD/USDT"
    ts0 = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    ts1 = ts0 + ONE_MIN_MS
    arr = np.array(
        [
            (ts0, 1.0, 2.0, 0.5, 1.5, 0.3),
            (ts1, 1.5, 2.5, 1.0, 2.0, 0.4),
        ],
        dtype=CANDLE_DTYPE,
    )
    observed = []

    def observer(payload):
        observed.append(dict(payload))

    cm._persist_batch(symbol, arr, timeframe="1m")
    cm.set_disk_load_observer(observer)
    loaded = cm._load_from_disk(symbol, ts0, ts1, timeframe="1m")

    assert loaded is not None
    assert loaded.shape[0] == 2
    assert len(observed) == 1
    payload = observed[0]
    assert payload["symbol"] == symbol
    assert payload["timeframe"] == "1m"
    assert payload["start_ts"] == ts0
    assert payload["end_ts"] == ts1
    assert payload["loaded_rows"] == 2
    assert payload["loaded_start_ts"] == ts0
    assert payload["loaded_end_ts"] == ts1
    assert payload["days"] == 1
    assert payload["source_days"] == {"primary": 1, "legacy": 0, "merged": 0}
    assert payload["elapsed_ms"] >= 0

    def failing_observer(_payload):
        raise RuntimeError("observer failed")

    cm.set_disk_load_observer(failing_observer)
    loaded_again = cm._load_from_disk(symbol, ts0, ts1, timeframe="1m")
    assert loaded_again is not None
    assert loaded_again.shape[0] == 2


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


def test_merge_overwrite_prefers_lower_valued_new_row_on_conflict(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    ts = _floor_minute(int(time.time() * 1000))
    existing = np.array([(ts, 300.0, 300.0, 300.0, 300.0, 0.0)], dtype=CANDLE_DTYPE)
    new = np.array([(ts, 200.0, 201.0, 199.0, 200.0, 5.0)], dtype=CANDLE_DTYPE)

    merged = cm._merge_overwrite(existing, new)

    assert merged.size == 1
    assert float(merged[0]["c"]) == pytest.approx(200.0)
    assert float(merged[0]["bv"]) == pytest.approx(5.0)


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
        fill_trailing_gaps=None,
        max_lookback_candles=None,
        allow_remote_fetch=True,
        allow_provisional_internal_gaps=False,
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
async def test_get_latest_ema_close_1h_excludes_current_hour_at_boundary(monkeypatch, tmp_path):
    fixed_now_ms = 1725580800000  # 2024-09-06 00:00:00 UTC, exact hour boundary
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "BTC/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    expected_end = fixed_now_ms - hour_ms
    expected_start = fixed_now_ms - 2 * hour_ms
    seen = {}

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
        fill_trailing_gaps=None,
        max_lookback_candles=None,
        allow_remote_fetch=True,
        allow_provisional_internal_gaps=False,
    ):
        seen.update(
            {
                "symbol": symbol_,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "timeframe": timeframe,
            }
        )
        arr = np.zeros(2, dtype=CANDLE_DTYPE)
        arr["ts"] = np.asarray([expected_start, expected_end], dtype=np.int64)
        arr["c"] = np.asarray([100.0, 102.0], dtype=np.float32)
        arr["h"] = arr["c"]
        arr["l"] = arr["c"]
        arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "get_candles", fake_get_candles)

    value = await cm.get_latest_ema_close(symbol, span=2.0, timeframe="1h", max_age_ms=60_000)

    assert np.isfinite(value)
    assert seen == {
        "symbol": symbol,
        "start_ts": expected_start,
        "end_ts": expected_end,
        "timeframe": "1h",
    }
    assert seen["end_ts"] < fixed_now_ms


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


@pytest.mark.asyncio
async def test_tf_force_refresh_bypasses_partial_range_cache(monkeypatch, tmp_path):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches")
    )
    timeframe = "1h"
    period_ms = 60 * ONE_MIN_MS
    end_ts = (fixed_now_ms // period_ms) * period_ms - period_ms
    start_ts = end_ts - 4 * period_ms
    symbol = "FORCE/USDT:USDT"

    partial = np.zeros(1, dtype=CANDLE_DTYPE)
    partial["ts"] = np.asarray([start_ts], dtype=np.int64)
    partial["o"] = 1.0
    partial["h"] = 2.0
    partial["l"] = 0.5
    partial["c"] = 1.5
    partial["bv"] = 1.0
    cache_key = (timeframe, start_ts, end_ts)
    cm._tf_range_cache[symbol] = OrderedDict(
        [(cache_key, (partial, fixed_now_ms))]
    )

    calls = {"fetch": 0}

    async def fake_fetch(
        symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None
    ):
        calls["fetch"] += 1
        ts = list(range(int(since_ms), int(end_exclusive_ms), period_ms))
        arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
        arr["ts"] = np.asarray(ts, dtype=np.int64)
        arr["o"] = 1.0
        arr["h"] = 2.0
        arr["l"] = 0.5
        arr["c"] = 1.5
        arr["bv"] = 1.0
        return arr

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    out = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        max_age_ms=0,
        timeframe=timeframe,
        max_lookback_candles=5,
    )

    assert calls["fetch"] == 1
    assert out.size == 5
    assert int(out["ts"][-1]) == end_ts


@pytest.mark.asyncio
async def test_tf_force_refresh_retains_disk_coverage_and_invalidates_tf_ema(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches")
    )
    symbol = "PARTIAL/USDT:USDT"
    timeframe = "1h"
    period_ms = 60 * ONE_MIN_MS
    end_ts = (fixed_now_ms // period_ms) * period_ms - period_ms
    start_ts = end_ts - 4 * period_ms

    full = np.zeros(5, dtype=CANDLE_DTYPE)
    full["ts"] = np.arange(start_ts, end_ts + period_ms, period_ms, dtype=np.int64)
    full["o"] = 1.0
    full["h"] = 2.0
    full["l"] = 0.5
    full["c"] = 1.5
    full["bv"] = 1.0
    cm._persist_batch(symbol, full, timeframe=timeframe)

    h1_ema_key = ("log_range", 10.0, str(period_ms))
    m1_ema_key = ("close", 10.0, str(ONE_MIN_MS))
    cm._ema_cache[symbol] = {
        h1_ema_key: (0.5, end_ts, fixed_now_ms),
        m1_ema_key: (1.5, end_ts, fixed_now_ms),
    }
    shorter_start_ts = end_ts - 2 * period_ms
    shorter_cache_key = (timeframe, shorter_start_ts, end_ts)
    stale_shorter = full[-3:].copy()
    stale_shorter["c"] = 1.5
    cm._tf_range_cache[symbol] = OrderedDict(
        [(shorter_cache_key, (stale_shorter, fixed_now_ms))]
    )
    calls = {"fetch": 0}

    async def fake_fetch(
        symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None
    ):
        calls["fetch"] += 1
        partial = np.zeros(1, dtype=CANDLE_DTYPE)
        partial["ts"] = np.asarray([end_ts], dtype=np.int64)
        partial["o"] = 8.0
        partial["h"] = 10.0
        partial["l"] = 7.0
        partial["c"] = 9.0
        partial["bv"] = 2.0
        if on_batch is not None:
            on_batch(partial)
        return partial

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    refreshed = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        max_age_ms=0,
        timeframe=timeframe,
        max_lookback_candles=5,
    )

    assert calls["fetch"] == 1
    assert refreshed.size == 5
    assert float(refreshed["c"][-1]) == pytest.approx(9.0)
    assert h1_ema_key not in cm._ema_cache[symbol]
    assert m1_ema_key in cm._ema_cache[symbol]
    assert shorter_cache_key not in cm._tf_range_cache[symbol]

    cached = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        max_age_ms=600_000,
        timeframe=timeframe,
        max_lookback_candles=5,
    )

    assert calls["fetch"] == 1
    assert cached.size == 5
    assert float(cached["c"][-1]) == pytest.approx(9.0)

    shorter_cached = await cm.get_candles(
        symbol,
        start_ts=shorter_start_ts,
        end_ts=end_ts,
        max_age_ms=600_000,
        timeframe=timeframe,
        max_lookback_candles=3,
    )

    assert calls["fetch"] == 1
    assert shorter_cached.size == 3
    assert float(shorter_cached["c"][-1]) == pytest.approx(9.0)


@pytest.mark.asyncio
async def test_tf_force_refresh_keeps_partial_range_out_of_ema_cache(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches")
    )
    symbol = "PARTIAL-EMA/USDT:USDT"
    timeframe = "1h"
    period_ms = 60 * ONE_MIN_MS
    end_ts = (fixed_now_ms // period_ms) * period_ms - period_ms
    start_ts = end_ts - 4 * period_ms

    disk_tail = np.zeros(1, dtype=CANDLE_DTYPE)
    disk_tail["ts"] = np.asarray([end_ts - period_ms], dtype=np.int64)
    disk_tail["o"] = 1.0
    disk_tail["h"] = 2.0
    disk_tail["l"] = 0.5
    disk_tail["c"] = 1.5
    disk_tail["bv"] = 1.0
    cm._persist_batch(symbol, disk_tail, timeframe=timeframe)
    strict_flags = []

    async def fake_fetch(
        symbol_,
        since_ms,
        end_exclusive_ms,
        *,
        timeframe=None,
        on_batch=None,
        raise_on_partial_empty_page=False,
    ):
        strict_flags.append(bool(raise_on_partial_empty_page))
        remote_tail = np.zeros(1, dtype=CANDLE_DTYPE)
        remote_tail["ts"] = np.asarray([end_ts], dtype=np.int64)
        remote_tail["o"] = 2.0
        remote_tail["h"] = 3.0
        remote_tail["l"] = 1.0
        remote_tail["c"] = 2.5
        remote_tail["bv"] = 1.0
        if on_batch is not None:
            on_batch(remote_tail)
        return remote_tail

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    refreshed = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        max_age_ms=0,
        timeframe=timeframe,
        max_lookback_candles=5,
    )

    cache_key = (timeframe, start_ts, end_ts)
    assert strict_flags == [True]
    assert refreshed.size == 2
    assert cache_key not in cm._tf_range_cache[symbol]

    ema = await cm.get_latest_ema_log_range(
        symbol,
        span=5.0,
        max_age_ms=600_000,
        timeframe=timeframe,
        allow_remote_fetch=False,
    )

    assert np.isnan(ema)
    assert ("log_range", 5.0, str(period_ms)) not in cm._ema_cache[symbol]


@pytest.mark.asyncio
async def test_tf_force_refresh_empty_result_does_not_fall_back_to_disk(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches")
    )
    symbol = "EMPTY-REMOTE/USDT:USDT"
    timeframe = "1h"
    period_ms = 60 * ONE_MIN_MS
    end_ts = (fixed_now_ms // period_ms) * period_ms - period_ms
    start_ts = end_ts - 4 * period_ms

    disk_tail = np.zeros(1, dtype=CANDLE_DTYPE)
    disk_tail["ts"] = np.asarray([end_ts], dtype=np.int64)
    disk_tail["o"] = 1.0
    disk_tail["h"] = 2.0
    disk_tail["l"] = 0.5
    disk_tail["c"] = 1.5
    disk_tail["bv"] = 1.0
    cm._persist_batch(symbol, disk_tail, timeframe=timeframe)

    async def fake_fetch(
        symbol_,
        since_ms,
        end_exclusive_ms,
        *,
        timeframe=None,
        on_batch=None,
        raise_on_partial_empty_page=False,
    ):
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    refreshed = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        max_age_ms=0,
        timeframe=timeframe,
        max_lookback_candles=5,
    )

    assert refreshed.size == 0
    assert (timeframe, start_ts, end_ts) not in cm._tf_range_cache[symbol]


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
async def test_get_candles_ttl_does_not_synthesize_single_trailing_present_gap(
    monkeypatch, tmp_path
):
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
    assert int(out[-1]["ts"]) == cached_last
    assert not cm._synthetic_timestamps.get(symbol)


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
async def test_get_last_prices_bounds_failed_completed_close_diagnostic(tmp_path):
    class _Ex:
        id = "bybit"

    secret = "https://private.example.test/?api_key=SECRET_LAST_PRICE"
    cm = CandlestickManager(exchange=_Ex(), exchange_name="bybit", cache_dir=str(tmp_path / "caches"))
    diagnostics = []

    async def fake_completed_close(symbol, **kwargs):
        if symbol == "BROKEN/USDT:USDT":
            raise RuntimeError(secret)
        return 123.45

    cm.get_latest_completed_close = fake_completed_close
    cm._log = lambda level, event, **data: diagnostics.append((level, event, data))

    prices = await cm.get_last_prices(["BROKEN/USDT:USDT", "OK/USDT:USDT"])

    assert prices == {"BROKEN/USDT:USDT": 0.0, "OK/USDT:USDT": 123.45}
    assert diagnostics == [
        (
            "debug",
            "get_last_prices_completed_close_failed",
            {"symbol": "BROKEN/USDT:USDT", "error_type": "RuntimeError"},
        )
    ]
    assert secret not in repr(diagnostics)
    assert "Traceback" not in repr(diagnostics)


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
async def test_get_candles_does_not_synthesize_open_ended_tail_gap(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "okx"

    cm = CandlestickManager(exchange=_Ex(), exchange_name="okx", cache_dir=str(tmp_path / "caches"))
    symbol = "KBONK/USDC:USDC"
    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    start_ts = end_finalized - 5 * ONE_MIN_MS
    old_ts = start_ts
    old_close = 0.1234

    seed = np.array([(old_ts, old_close, old_close, old_close, old_close, 1.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, seed, timeframe="1m", merge_cache=True)
    cm._cache.pop(symbol, None)  # simulate process restart (seed only on disk)

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    out = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_finalized,
        max_age_ms=30_000,
        strict=False,
    )

    assert out.size == 1
    assert list(out["ts"]) == [start_ts]
    assert np.allclose(np.asarray(out["c"], dtype=np.float64), old_close)
    assert np.allclose(np.asarray(out["bv"], dtype=np.float64), 1.0)
    assert not cm._synthetic_timestamps.get(symbol)

    # Open-ended missing tail is not synthesized; shard still has only the original seed.
    day_key = cm._date_key(old_ts)
    shard = cm._load_shard(cm._shard_path(symbol, day_key, timeframe="1m"))
    assert shard.size == 1


@pytest.mark.asyncio
async def test_large_present_touching_warmup_does_not_synthesize_open_tail(
    monkeypatch, tmp_path
):
    fixed_now_ms = 1725590400000  # 2024-09-06 00:00:00 UTC
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    symbol = "STALE/USDT:USDT"
    end_finalized = _floor_minute(fixed_now_ms) - ONE_MIN_MS
    start_ts = end_finalized - 3 * 24 * 60 * ONE_MIN_MS
    close = 7.0
    seed = np.array([(start_ts, close, close, close, close, 1.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, seed, timeframe="1m", merge_cache=True)
    cm._cache.pop(symbol, None)

    calls = {"fetches": 0}

    async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
        calls["fetches"] += 1
        return np.empty((0,), dtype=CANDLE_DTYPE)

    monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

    out = await cm.get_candles(
        symbol,
        start_ts=start_ts,
        end_ts=end_finalized,
        max_age_ms=30_000,
        strict=False,
    )

    assert calls["fetches"] >= 1
    assert out.size == 1
    assert int(out[0]["ts"]) == start_ts
    assert float(out[0]["c"]) == pytest.approx(close)
    assert not cm._synthetic_timestamps.get(symbol)


def test_real_batch_overrides_bounded_runtime_synthetic_and_invalidates_ema_cache(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "ILLQ/USDT:USDT"
    base_ts = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    base_close = 11.0

    seed = np.array(
        [
            (base_ts, base_close, base_close, base_close, base_close, 1.0),
            (base_ts + 2 * ONE_MIN_MS, 13.0, 13.0, 13.0, 13.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    standardized = cm.standardize_gaps(
        seed,
        start_ts=base_ts,
        end_ts=base_ts + 2 * ONE_MIN_MS,
        strict=False,
        fill_trailing_gaps=False,
        symbol=symbol,
    )
    cm._cache[symbol] = standardized

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


def test_materialize_runtime_synthetic_gap_skips_open_tail(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "DEAD/USDT:USDT"
    through_ts = _floor_minute(int(time.time() * 1000)) - 5 * ONE_MIN_MS
    gap_minutes = 3 * 24 * 60  # 4320 minutes
    seed_ts = through_ts - gap_minutes * ONE_MIN_MS
    seed_close = 42.0

    seed = np.array(
        [(seed_ts, seed_close, seed_close, seed_close, seed_close, 1.0)], dtype=CANDLE_DTYPE
    )
    cm._cache[symbol] = seed

    synthesized = cm._materialize_runtime_synthetic_gap(symbol, through_ts)

    assert synthesized == 0

    arr = np.sort(cm._cache[symbol], order="ts")
    assert arr.shape[0] == 1
    assert int(arr[0]["ts"]) == seed_ts


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
    assert one_m["open_tail_gap"] is False
    assert one_m["last_cached_ts"] == last_final
    assert one_m["loaded_rows"] == 2


def test_completed_candle_health_reports_open_tail_gap(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TAILHEALTH/USDT:USDT"
    now_ms = 1725590400000
    current_minute = _floor_minute(now_ms)
    last_final = current_minute - ONE_MIN_MS
    start = last_final - 2 * ONE_MIN_MS
    candles = np.array([(start, 10.0, 10.0, 10.0, 10.0, 1.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, candles, timeframe="1m", merge_cache=True, last_refresh_ms=now_ms)

    report = cm.get_completed_candle_health(symbol, {"1m": 3}, now_ms=now_ms)

    one_m = report["timeframes"]["1m"]
    assert report["ok"] is False
    assert one_m["coverage_ok"] is False
    assert one_m["missing_spans"] == [(start + ONE_MIN_MS, last_final)]
    assert one_m["missing_candles"] == 2
    assert one_m["open_tail_gap"] is True
    assert one_m["tail_gap_candles"] == 2
    assert one_m["tail_gap_age_ms"] == 2 * ONE_MIN_MS


def test_completed_candle_health_reports_prior_cached_ts_for_one_candle_tail_gap(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name="ex", cache_dir=str(tmp_path / "caches"))
    symbol = "TAILPRIOR/USDT:USDT"
    now_ms = 10 * ONE_MIN_MS
    latest_expected = 9 * ONE_MIN_MS
    prior = 6 * ONE_MIN_MS
    candles = np.array([(prior, 10.0, 10.0, 10.0, 10.0, 1.0)], dtype=CANDLE_DTYPE)
    cm._persist_batch(symbol, candles, timeframe="1m", merge_cache=True, last_refresh_ms=now_ms)

    report = cm.get_completed_candle_health(symbol, {"1m": 1}, now_ms=now_ms)

    one_m = report["timeframes"]["1m"]
    assert report["ok"] is False
    assert one_m["coverage_ok"] is False
    assert one_m["missing_spans"] == [(latest_expected, latest_expected)]
    assert one_m["open_tail_gap"] is True
    assert one_m["last_cached_ts"] == prior
    assert one_m["tail_gap_candles"] == 3
    assert one_m["tail_gap_age_ms"] == latest_expected - prior


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
    missing_minute = last_minute - ONE_MIN_MS
    first_minute = last_minute - 2 * ONE_MIN_MS
    seed = np.array(
        [
            (first_minute, 11.0, 11.0, 11.0, 11.0, 1.0),
            (last_minute, 12.0, 12.0, 12.0, 12.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = cm.standardize_gaps(
        seed,
        start_ts=first_minute,
        end_ts=last_minute,
        strict=False,
        fill_trailing_gaps=False,
        symbol=symbol,
    )
    cm._synthetic_timestamps[symbol] = {missing_minute}

    report = cm.get_completed_candle_health(symbol, {"1m": 3, "1h": 1}, now_ms=now_ms)

    one_m = report["timeframes"]["1m"]
    one_h = report["timeframes"]["1h"]
    assert one_m["coverage_ok"] is True
    assert one_m["runtime_synthetic_count"] == 1
    assert missing_minute in cm._synthetic_timestamps.get(symbol, set())
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


def test_completed_candle_health_distinguishes_verified_sparse_continuity(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "SPARSE/USDT:USDT"
    now_ms = 6 * ONE_MIN_MS
    cm._persist_batch(
        symbol,
        np.array(
            [
                (2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0),
                (5 * ONE_MIN_MS, 11.0, 11.0, 11.0, 11.0, 2.0),
            ],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    cm._record_verified_gap(
        symbol,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
        reason=GAP_REASON_NO_TRADES,
    )

    report = cm.get_completed_candle_health(
        symbol, {"1m": 4}, now_ms=now_ms
    )["timeframes"]["1m"]

    assert report["coverage_ok"] is False
    assert report["missing_candles"] == 2
    assert report["verified_no_trade_missing_candles"] == 2
    assert report["deferred_missing_candles"] == 0
    assert report["refreshable_missing_candles"] == 0
    assert report["refresh_needed"] is False


def test_completed_candle_health_keeps_adjacent_unverified_minute_refreshable(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "PARTIALPROOF/USDT:USDT"
    now_ms = 7 * ONE_MIN_MS
    cm._persist_batch(
        symbol,
        np.array(
            [
                (2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0),
                (6 * ONE_MIN_MS, 11.0, 11.0, 11.0, 11.0, 2.0),
            ],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    cm._record_verified_gap(
        symbol,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
        reason=GAP_REASON_NO_TRADES,
    )
    cm._add_known_gap(
        symbol,
        5 * ONE_MIN_MS,
        5 * ONE_MIN_MS,
        reason=GAP_REASON_AUTO,
        increment_retry=False,
    )

    gaps = cm._get_known_gaps_enhanced(symbol)
    assert [
        (gap["start_ts"], gap["end_ts"], gap["reason"]) for gap in gaps
    ] == [
        (3 * ONE_MIN_MS, 4 * ONE_MIN_MS, GAP_REASON_NO_TRADES),
        (5 * ONE_MIN_MS, 5 * ONE_MIN_MS, GAP_REASON_AUTO),
    ]
    report = cm.get_completed_candle_health(
        symbol, {"1m": 5}, now_ms=now_ms
    )["timeframes"]["1m"]

    assert report["missing_candles"] == 3
    assert report["verified_no_trade_missing_candles"] == 2
    assert report["refreshable_missing_candles"] == 1
    assert report["refresh_needed"] is True


def test_completed_candle_health_keeps_fresh_same_reason_suffix_refreshable(
    tmp_path, monkeypatch
):
    now = {"ms": 7 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "FRESHEPOCH/USDT:USDT"
    cm._persist_batch(
        symbol,
        np.array(
            [
                (2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0),
                (6 * ONE_MIN_MS, 11.0, 11.0, 11.0, 11.0, 2.0),
            ],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now["ms"],
    )
    cm._add_known_gap(
        symbol,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
        reason=GAP_REASON_AUTO,
        retry_count=_GAP_MAX_RETRIES,
    )
    now["ms"] += 1
    cm._add_known_gap(
        symbol,
        3 * ONE_MIN_MS,
        5 * ONE_MIN_MS,
        reason=GAP_REASON_AUTO,
    )

    gaps = cm._get_known_gaps_enhanced(symbol)
    assert [
        (gap["start_ts"], gap["end_ts"], gap["retry_count"])
        for gap in gaps
    ] == [
        (3 * ONE_MIN_MS, 4 * ONE_MIN_MS, _GAP_MAX_RETRIES),
        (5 * ONE_MIN_MS, 5 * ONE_MIN_MS, 1),
    ]
    report = cm.get_completed_candle_health(
        symbol, {"1m": 5}, now_ms=now["ms"]
    )["timeframes"]["1m"]

    assert report["missing_candles"] == 3
    assert report["deferred_missing_candles"] == 2
    assert report["refreshable_missing_candles"] == 1
    assert report["refresh_needed"] is True


def test_gap_normalization_merges_only_matching_retry_epochs(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "RETRYEPISODES/USDT:USDT"
    base_gap = {
        "retry_count": _GAP_MAX_RETRIES,
        "reason": GAP_REASON_AUTO,
        "added_at": 1,
        "last_retry_at": 1,
        "last_contextual_retry_at": 0,
    }
    cm._save_known_gaps_enhanced(
        symbol,
        [
            {
                **base_gap,
                "start_ts": ONE_MIN_MS,
                "end_ts": ONE_MIN_MS,
            },
            {
                **base_gap,
                "start_ts": 2 * ONE_MIN_MS,
                "end_ts": 2 * ONE_MIN_MS,
            },
            {
                **base_gap,
                "start_ts": 3 * ONE_MIN_MS,
                "end_ts": 3 * ONE_MIN_MS,
                "last_retry_at": 2,
            },
        ],
    )

    gaps = cm._get_known_gaps_enhanced(symbol)
    assert [(gap["start_ts"], gap["end_ts"]) for gap in gaps] == [
        (ONE_MIN_MS, 2 * ONE_MIN_MS),
        (3 * ONE_MIN_MS, 3 * ONE_MIN_MS),
    ]
    assert [gap["last_retry_at"] for gap in gaps] == [1, 2]


def test_gap_retry_update_is_scoped_to_attempted_overlap(tmp_path, monkeypatch):
    now = {"ms": 10 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "PARTIALEPOCH/USDT:USDT"
    cm._add_known_gap(
        symbol,
        ONE_MIN_MS,
        3 * ONE_MIN_MS,
        reason=GAP_REASON_AUTO,
    )
    now["ms"] += 1
    cm._add_known_gap(
        symbol,
        2 * ONE_MIN_MS,
        2 * ONE_MIN_MS,
        reason=GAP_REASON_AUTO,
    )

    gaps = cm._get_known_gaps_enhanced(symbol)
    assert [
        (gap["start_ts"], gap["end_ts"], gap["retry_count"])
        for gap in gaps
    ] == [
        (ONE_MIN_MS, ONE_MIN_MS, 1),
        (2 * ONE_MIN_MS, 2 * ONE_MIN_MS, 2),
        (3 * ONE_MIN_MS, 3 * ONE_MIN_MS, 1),
    ]


def test_known_gap_normalization_scales_to_large_sparse_history(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MANYGAPS/USDT:USDT"
    gap_count = 5_000
    gaps = [
        {
            "start_ts": index * 2 * ONE_MIN_MS,
            "end_ts": index * 2 * ONE_MIN_MS,
            "retry_count": 1,
            "reason": GAP_REASON_AUTO,
            "added_at": 1,
            "last_retry_at": 1,
        }
        for index in range(gap_count)
    ]

    started = time.perf_counter()
    cm._save_known_gaps_enhanced(symbol, gaps, defer_index=True)
    elapsed = time.perf_counter() - started

    assert len(cm._get_known_gaps_enhanced(symbol)) == gap_count
    assert elapsed < 2.0


def test_completed_candle_health_scales_to_many_sparse_deferred_gaps(
    tmp_path, monkeypatch
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MANYHEALTHGAPS/USDT:USDT"
    gap_count = 5_000
    end_ts = gap_count * 2 * ONE_MIN_MS
    candles = np.array(
        [
            (ts, 1.0, 1.0, 1.0, 1.0, 1.0)
            for ts in range(0, end_ts + 1, 2 * ONE_MIN_MS)
        ],
        dtype=CANDLE_DTYPE,
    )
    gaps = [
        {
            "start_ts": index * 2 * ONE_MIN_MS + ONE_MIN_MS,
            "end_ts": index * 2 * ONE_MIN_MS + ONE_MIN_MS,
            "retry_count": 1,
            "reason": GAP_REASON_AUTO,
            "added_at": 1,
            "last_retry_at": 1,
        }
        for index in range(gap_count)
    ]
    monkeypatch.setattr(cm, "_load_from_disk", lambda *_args, **_kwargs: candles)
    monkeypatch.setattr(cm, "_get_known_gaps_enhanced", lambda _symbol: gaps)
    monkeypatch.setattr(cm, "_should_retry_gap", lambda *_args, **_kwargs: False)

    started = time.perf_counter()
    report = cm.get_completed_candle_health(
        symbol,
        {"1m": gap_count * 2 + 1},
        now_ms=end_ts + ONE_MIN_MS,
    )["timeframes"]["1m"]
    elapsed = time.perf_counter() - started

    assert report["missing_candles"] == gap_count
    assert report["deferred_missing_candles"] == gap_count
    assert report["refreshable_missing_candles"] == 0
    assert report["refresh_needed"] is False
    assert elapsed < 2.0


def test_completed_candle_health_scales_to_many_verified_sparse_gaps(
    tmp_path, monkeypatch
):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="ex",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MANYVERIFIEDGAPS/USDT:USDT"
    gap_count = 5_000
    end_ts = gap_count * 2 * ONE_MIN_MS
    candles = np.array(
        [
            (ts, 1.0, 1.0, 1.0, 1.0, 1.0)
            for ts in range(0, end_ts + 1, 2 * ONE_MIN_MS)
        ],
        dtype=CANDLE_DTYPE,
    )
    gaps = [
        {
            "start_ts": index * 2 * ONE_MIN_MS + ONE_MIN_MS,
            "end_ts": index * 2 * ONE_MIN_MS + ONE_MIN_MS,
            "retry_count": 0,
            "reason": GAP_REASON_NO_TRADES,
            "added_at": 1,
            "last_retry_at": 1,
        }
        for index in range(gap_count)
    ]
    monkeypatch.setattr(cm, "_load_from_disk", lambda *_args, **_kwargs: candles)
    monkeypatch.setattr(cm, "_get_known_gaps_enhanced", lambda _symbol: gaps)

    started = time.perf_counter()
    report = cm.get_completed_candle_health(
        symbol,
        {"1m": gap_count * 2 + 1},
        now_ms=end_ts + ONE_MIN_MS,
    )["timeframes"]["1m"]
    elapsed = time.perf_counter() - started

    assert report["missing_candles"] == gap_count
    assert report["verified_no_trade_missing_candles"] == gap_count
    assert report["refreshable_missing_candles"] == 0
    assert report["refresh_needed"] is False
    assert elapsed < 2.0


def test_completed_candle_health_defers_unknown_gap_until_retry_is_due(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "DEFERRED/USDT:USDT"
    now_ms = 6 * ONE_MIN_MS
    cm._persist_batch(
        symbol,
        np.array(
            [
                (2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0),
                (5 * ONE_MIN_MS, 11.0, 11.0, 11.0, 11.0, 2.0),
            ],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    cm._add_known_gap(
        symbol,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=_GAP_MAX_RETRIES,
    )
    gaps = cm._get_known_gaps_enhanced(symbol)
    gaps[0]["last_contextual_retry_at"] = now_ms
    cm._save_known_gaps_enhanced(symbol, gaps)

    report = cm.get_completed_candle_health(
        symbol, {"1m": 4}, now_ms=now_ms
    )["timeframes"]["1m"]

    assert report["coverage_ok"] is False
    assert report["verified_no_trade_missing_candles"] == 0
    assert report["deferred_missing_candles"] == 2
    assert report["refreshable_missing_candles"] == 0
    assert report["refresh_needed"] is False


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


def test_persisted_rows_trim_known_gaps_and_preserve_metadata(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "TEST/USDC:USDC"
    base = 1_000_000
    cm._add_known_gap(
        symbol,
        base,
        base + 4 * ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=2,
    )
    original = cm._get_known_gaps_enhanced(symbol)[0]
    batch = np.array(
        [
            (
                base + 2 * ONE_MIN_MS,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
            )
        ],
        dtype=CANDLE_DTYPE,
    )

    cm._persist_batch(symbol, batch, timeframe="1m")

    gaps = cm._get_known_gaps_enhanced(symbol)
    assert [(g["start_ts"], g["end_ts"]) for g in gaps] == [
        (base, base + ONE_MIN_MS),
        (base + 3 * ONE_MIN_MS, base + 4 * ONE_MIN_MS),
    ]
    assert all(g["retry_count"] == 2 for g in gaps)
    assert all(g["reason"] == GAP_REASON_FETCH_FAILED for g in gaps)
    assert all(g["added_at"] == original["added_at"] for g in gaps)
    assert all(g["last_retry_at"] == original["last_retry_at"] for g in gaps)

    cm._persist_batch(
        symbol,
        np.array(
            [
                (base, 1.0, 1.0, 1.0, 1.0, 0.0),
                (base + ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 0.0),
                (base + 3 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 0.0),
                (base + 4 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 0.0),
            ],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
    )
    assert cm._get_known_gaps_enhanced(symbol) == []


def test_adding_known_gap_invalidates_1m_ema_and_projection_caches(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "TEST/USDC:USDC"
    m1_key = ("close", 5.0, str(ONE_MIN_MS))
    h1_key = ("close", 5.0, str(60 * ONE_MIN_MS))
    cm._ema_cache[symbol] = {
        m1_key: (1.0, 1_000_000, 1_000_000),
        h1_key: (2.0, 1_000_000, 1_000_000),
    }
    cm._projected_open_tail_ema_cache[symbol] = {("cached",): {"close": 1.0}}

    cm._add_known_gap(
        symbol,
        1_000_000,
        1_000_000,
        reason=GAP_REASON_FETCH_FAILED,
    )

    assert m1_key not in cm._ema_cache[symbol]
    assert h1_key in cm._ema_cache[symbol]
    assert symbol not in cm._projected_open_tail_ema_cache


def test_persisted_rows_leave_unrelated_known_gap_unchanged(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "TEST/USDC:USDC"
    cm._add_known_gap(symbol, 1_000_000, 1_120_000)
    original = cm._get_known_gaps_enhanced(symbol)

    cm._persist_batch(
        symbol,
        np.array(
            [(2_000_000, 1.0, 1.0, 1.0, 1.0, 0.0)],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
    )

    assert cm._get_known_gaps_enhanced(symbol) == original


def test_hyperliquid_recent_gap_retries_are_time_spaced(monkeypatch, tmp_path):
    now = {"ms": 10_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MU/USDC:USDC"
    start = now["ms"] - 5 * ONE_MIN_MS
    end = now["ms"] - ONE_MIN_MS

    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert gap["retry_count"] == 1
    assert not cm._should_retry_gap(gap, now_ms=now["ms"])

    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    assert cm._get_known_gaps_enhanced(symbol)[0]["retry_count"] == 1

    now["ms"] += 5 * ONE_MIN_MS
    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert cm._should_retry_gap(gap, now_ms=now["ms"])
    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    assert cm._get_known_gaps_enhanced(symbol)[0]["retry_count"] == 2

    now["ms"] += 5 * ONE_MIN_MS
    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert gap["retry_count"] == _GAP_MAX_RETRIES
    assert not cm._should_retry_gap(gap, now_ms=now["ms"])

    now["ms"] += 15 * ONE_MIN_MS
    assert cm._should_retry_gap(gap, now_ms=now["ms"])
    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert gap["retry_count"] == _GAP_MAX_RETRIES
    assert not cm._should_retry_gap(gap, now_ms=now["ms"])

    now["ms"] += 5 * ONE_MIN_MS
    assert not cm._should_retry_gap(gap, now_ms=now["ms"])
    now["ms"] += 10 * ONE_MIN_MS
    assert cm._should_retry_gap(gap, now_ms=now["ms"])


def test_hyperliquid_gap_retry_metadata_uses_manager_clock(
    monkeypatch, tmp_path
):
    wall_now_ms = 10_000 * ONE_MIN_MS
    replay_now = {"ms": 100 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: wall_now_ms / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: replay_now["ms"]
    symbol = "MU/USDC:USDC"
    gap_ts = replay_now["ms"] - ONE_MIN_MS

    cm._add_known_gap(
        symbol,
        gap_ts,
        gap_ts,
        reason=GAP_REASON_FETCH_FAILED,
    )

    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert gap["added_at"] == replay_now["ms"]
    assert gap["last_retry_at"] == replay_now["ms"]
    assert not cm._should_retry_gap(gap)

    replay_now["ms"] += 5 * ONE_MIN_MS
    assert cm._should_retry_gap(gap)


def test_hyperliquid_accelerated_retry_excludes_large_recent_ending_gap(
    monkeypatch, tmp_path
):
    now = {"ms": 20_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "NEW/USDC:USDC"
    cm._add_known_gap(
        symbol,
        now["ms"] - 24 * 60 * ONE_MIN_MS,
        now["ms"] - ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=_GAP_MAX_RETRIES,
    )
    gap = cm._get_known_gaps_enhanced(symbol)[0]

    now["ms"] += 15 * ONE_MIN_MS

    assert not cm._is_recent_hyperliquid_gap(gap, now_ms=now["ms"])
    assert not cm._should_retry_gap(gap, now_ms=now["ms"])


def test_deferred_unknown_gap_exclusions_remain_compact(monkeypatch, tmp_path):
    now = {"ms": 10_000_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "NEW/USDC:USDC"
    start = now["ms"] - 5 * 365 * 24 * 60 * ONE_MIN_MS
    end = now["ms"] - ONE_MIN_MS
    cm._add_known_gap(
        symbol,
        start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=_GAP_MAX_RETRIES,
    )

    assert cm._unverified_gap_ranges(symbol, start, end) == [
        (start, end)
    ]


@pytest.mark.asyncio
async def test_hyperliquid_known_tail_gap_cooldown_precedes_present_fetch(
    monkeypatch, tmp_path
):
    now = {"ms": 30_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    start = _floor_minute(now["ms"]) - 3 * ONE_MIN_MS
    gap_start = start + ONE_MIN_MS
    end = _floor_minute(now["ms"]) - ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(start, 1.0, 1.0, 1.0, 1.0, 1.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        gap_start,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def fetch_paginated(*args, **kwargs):
        calls.append((args, kwargs))
        return np.empty((0,), dtype=CANDLE_DTYPE)

    cm._fetch_ohlcv_paginated = fetch_paginated

    await cm.get_candles(symbol, start_ts=start, end_ts=end)

    assert calls == []


@pytest.mark.asyncio
async def test_hyperliquid_deferred_tail_gap_still_fetches_finalized_suffix(
    monkeypatch, tmp_path
):
    now = {"ms": 30_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    end = _floor_minute(now["ms"]) - ONE_MIN_MS
    start = end - 3 * ONE_MIN_MS
    gap_start = start + ONE_MIN_MS
    gap_end = end - ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(start, 1.0, 1.0, 1.0, 1.0, 1.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        gap_start,
        gap_end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.array(
            [(end, 2.0, 2.0, 2.0, 2.0, 2.0)],
            dtype=CANDLE_DTYPE,
        )

    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        fill_trailing_gaps=False,
    )

    assert calls == [(end, end + ONE_MIN_MS)]
    assert int(result[-1]["ts"]) == end
    assert float(result[-1]["c"]) == pytest.approx(2.0)
    assert gap_start not in set(result["ts"].astype(np.int64))
    assert gap_end not in set(result["ts"].astype(np.int64))
    assert not (
        {gap_start, gap_end}
        & set(cm._synthetic_timestamps.get(symbol, set()))
    )


@pytest.mark.asyncio
async def test_deferred_tail_does_not_block_unrelated_internal_gap_repair(
    monkeypatch, tmp_path
):
    now = {"ms": 40_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    end = _floor_minute(now["ms"]) - ONE_MIN_MS
    start = end - 4 * ONE_MIN_MS
    internal_gap = start + ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 1.0, 1.0, 1.0, 1.0, 1.0),
            (start + 2 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
            (start + 3 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        end,
        end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.array(
            [(internal_gap, 2.0, 2.0, 2.0, 2.0, 2.0)],
            dtype=CANDLE_DTYPE,
        )

    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        fill_trailing_gaps=False,
    )

    assert calls == [(internal_gap, internal_gap + ONE_MIN_MS)]
    assert internal_gap in set(result["ts"].astype(np.int64))
    assert end not in set(result["ts"].astype(np.int64))


@pytest.mark.asyncio
async def test_extended_missing_tail_never_refetches_deferred_prefix(
    monkeypatch, tmp_path
):
    now = {"ms": 50_000_000}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    end = _floor_minute(now["ms"]) - ONE_MIN_MS
    start = end - 3 * ONE_MIN_MS
    gap_start = start + ONE_MIN_MS
    gap_end = end - ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [(start, 1.0, 1.0, 1.0, 1.0, 1.0)],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        gap_start,
        gap_end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.empty((0,), dtype=CANDLE_DTYPE)

    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        fill_trailing_gaps=False,
    )

    assert calls
    assert all(since == end for since, _end_exclusive in calls)
    assert gap_start not in set(result["ts"].astype(np.int64))
    assert gap_end not in set(result["ts"].astype(np.int64))
    assert end not in set(result["ts"].astype(np.int64))


@pytest.mark.asyncio
async def test_partial_gap_recovery_defers_and_preserves_unresolved_minutes(
    monkeypatch, tmp_path
):
    now = {"ms": 60 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    end = _floor_minute(now["ms"]) - ONE_MIN_MS
    start = end - 4 * ONE_MIN_MS
    gap_start = start + ONE_MIN_MS
    gap_end = end - ONE_MIN_MS
    recovered = start + 2 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 1.0, 1.0, 1.0, 1.0, 1.0),
            (end, 2.0, 2.0, 2.0, 2.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        gap_start,
        gap_end,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gaps = cm._get_known_gaps_enhanced(symbol)
    gaps[0]["last_retry_at"] = now["ms"] - 5 * ONE_MIN_MS
    cm._save_known_gaps_enhanced(symbol, gaps)
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.array(
            [(recovered, 1.5, 1.5, 1.5, 1.5, 1.0)],
            dtype=CANDLE_DTYPE,
        )

    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(symbol, start_ts=start, end_ts=end)

    assert calls == [(gap_start, gap_end + ONE_MIN_MS)]
    assert list(result["ts"]) == [start, recovered, end]
    unresolved = cm._get_known_gaps_enhanced(symbol)
    assert [(gap["start_ts"], gap["end_ts"]) for gap in unresolved] == [
        (gap_start, gap_start),
        (gap_end, gap_end),
    ]
    assert all(gap["last_retry_at"] == now["ms"] for gap in unresolved)

    calls.clear()
    repeated = await cm.get_candles(symbol, start_ts=start, end_ts=end)
    assert calls == []
    assert list(repeated["ts"]) == [start, recovered, end]


@pytest.mark.asyncio
async def test_due_unknown_gap_stays_unavailable_without_remote_fetch(
    monkeypatch, tmp_path
):
    now = {"ms": 70 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    start = now["ms"] - 3 * ONE_MIN_MS
    missing = start + ONE_MIN_MS
    end = start + 2 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 1.0, 1.0, 1.0, 1.0, 1.0),
            (end, 2.0, 2.0, 2.0, 2.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        missing,
        missing,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gaps = cm._get_known_gaps_enhanced(symbol)
    gaps[0]["last_retry_at"] = now["ms"] - 5 * ONE_MIN_MS
    cm._save_known_gaps_enhanced(symbol, gaps)
    assert cm._should_retry_gap(
        cm._get_known_gaps_enhanced(symbol)[0], now_ms=now["ms"]
    )

    result = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        allow_remote_fetch=False,
    )

    assert list(result["ts"]) == [start, end]
    assert missing not in set(cm._synthetic_timestamps.get(symbol, set()))


@pytest.mark.asyncio
async def test_live_ema_provisionally_fills_bounded_unknown_gap_and_recomputes(
    monkeypatch, tmp_path
):
    now = 11 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="kucoinfutures",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    start = 8 * ONE_MIN_MS
    missing = 9 * ONE_MIN_MS
    end = 10 * ONE_MIN_MS
    authoritative = np.array(
        [
            (start, 100.0, 100.0, 100.0, 100.0, 1.0),
            (end, 120.0, 120.0, 120.0, 120.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = authoritative.copy()
    cm._add_known_gap(
        symbol,
        missing,
        missing,
        reason=GAP_REASON_FETCH_FAILED,
    )

    ordinary = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        allow_remote_fetch=False,
    )
    assert list(ordinary["ts"]) == [start, end]

    strict_candidate = await cm.get_latest_ema_close(
        symbol,
        3.0,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=False,
    )
    assert math.isnan(strict_candidate)

    provisional = await cm.get_latest_ema_close(
        symbol,
        3.0,
        allow_remote_fetch=False,
    )
    expected_provisional = cm._ema(
        np.asarray([100.0, 100.0, 120.0], dtype=np.float64),
        3.0,
    )
    assert provisional == pytest.approx(expected_provisional)
    assert np.array_equal(cm._cache[symbol], authoritative)
    assert missing in cm._synthetic_timestamps[symbol]
    assert ("close", 3.0, str(ONE_MIN_MS)) in cm._ema_cache[symbol]
    strict_after_provisional = await cm.get_latest_ema_close(
        symbol,
        3.0,
        max_age_ms=ONE_MIN_MS,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=False,
    )
    assert math.isnan(strict_after_provisional)

    cm._persist_batch(
        symbol,
        np.array(
            [(missing, 110.0, 110.0, 110.0, 110.0, 2.0)],
            dtype=CANDLE_DTYPE,
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now,
    )

    assert missing not in cm._synthetic_timestamps.get(symbol, set())
    assert ("close", 3.0, str(ONE_MIN_MS)) not in cm._ema_cache.get(symbol, {})
    authoritative_ema = await cm.get_latest_ema_close(
        symbol,
        3.0,
        allow_remote_fetch=False,
    )
    expected_authoritative = cm._ema(
        np.asarray([100.0, 110.0, 120.0], dtype=np.float64),
        3.0,
    )
    assert authoritative_ema == pytest.approx(expected_authoritative)
    assert authoritative_ema != pytest.approx(provisional)


@pytest.mark.asyncio
async def test_refreshed_forager_metrics_bridge_bounded_internal_gap(
    monkeypatch, tmp_path
):
    now = 11 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="kucoinfutures",
        cache_dir=str(tmp_path / "caches"),
        provisional_internal_gap_tolerance_minutes=10,
    )
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    start = 8 * ONE_MIN_MS
    missing = 9 * ONE_MIN_MS
    end = 10 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 100.0, 101.0, 99.0, 100.0, 1.0),
            (end, 120.0, 121.0, 119.0, 120.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        missing,
        missing,
        reason=GAP_REASON_FETCH_FAILED,
    )
    spans = {"qv": [3.0], "log_range": [3.0]}

    strict = await cm.get_latest_ema_metric_spans(
        symbol,
        spans,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=False,
    )
    assert math.isnan(strict["qv"][3.0])
    assert math.isnan(strict["log_range"][3.0])

    refreshed = await cm.get_latest_ema_metric_spans(
        symbol,
        spans,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=True,
    )
    expected_qv = cm._ema(np.asarray([100.0, 0.0, 120.0]), 3.0)
    expected_log_range = cm._ema(
        np.log(np.asarray([101.0 / 99.0, 1.0, 121.0 / 119.0])),
        3.0,
    )
    assert refreshed["qv"][3.0] == pytest.approx(expected_qv)
    assert refreshed["log_range"][3.0] == pytest.approx(expected_log_range)
    qv_context = cm.get_ema_provisional_internal_gap_context(
        symbol, "qv", 3.0, timeframe="1m"
    )
    log_range_context = cm.get_ema_provisional_internal_gap_context(
        symbol, "log_range", 3.0, timeframe="1m"
    )
    assert qv_context == log_range_context
    assert qv_context["gap_count"] == 1
    assert qv_context["gap_candles"] == 1
    assert qv_context["max_gap_candles"] == 1
    assert qv_context["oldest_gap_age_ms"] == 2 * ONE_MIN_MS
    assert np.array_equal(
        cm._cache[symbol]["ts"],
        np.asarray([start, end], dtype=np.int64),
    )


def test_synthetic_timestamp_retention_uses_replay_clock(monkeypatch, tmp_path):
    wall_now = 10_000 * ONE_MIN_MS
    replay_now = 100 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: wall_now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="fake",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: replay_now
    symbol = "REPLAY/USDT:USDT"
    synthetic_ts = replay_now - ONE_MIN_MS
    cm._ema_cache[symbol] = {
        ("close", 3.0, str(ONE_MIN_MS)): (100.0, synthetic_ts, replay_now)
    }

    cm._track_synthetic_timestamps(symbol, [synthetic_ts])

    assert cm._synthetic_timestamps[symbol] == {synthetic_ts}
    cm._check_synthetic_replacement(
        symbol,
        np.array(
            [(synthetic_ts, 101.0, 101.0, 101.0, 101.0, 1.0)],
            dtype=CANDLE_DTYPE,
        ),
    )
    assert cm._synthetic_timestamps.get(symbol, set()) == set()
    assert cm._ema_cache.get(symbol, {}) == {}


@pytest.mark.asyncio
async def test_live_ema_refuses_provisional_internal_gap_beyond_tolerance(
    monkeypatch, tmp_path
):
    now = 15 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="testex",
        cache_dir=str(tmp_path / "caches"),
        provisional_internal_gap_tolerance_minutes=2,
    )
    cm._now_ms_callback = lambda: now
    symbol = "WIDEGAP/USDT:USDT"
    start = 10 * ONE_MIN_MS
    end = 14 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 100.0, 100.0, 100.0, 100.0, 1.0),
            (end, 110.0, 110.0, 110.0, 110.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        start + ONE_MIN_MS,
        end - ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
    )

    assert math.isnan(
        await cm.get_latest_ema_close(
            symbol,
            5.0,
            allow_remote_fetch=False,
        )
    )
    assert cm._synthetic_timestamps.get(symbol, set()) == set()


@pytest.mark.asyncio
async def test_provisional_gap_tolerance_uses_full_recorded_gap_not_clipped_overlap(
    monkeypatch, tmp_path
):
    now = 205 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="testex",
        cache_dir=str(tmp_path / "caches"),
        provisional_internal_gap_tolerance_minutes=10,
    )
    cm._now_ms_callback = lambda: now
    symbol = "CLIPPEDGAP/USDT:USDT"
    requested_start = 200 * ONE_MIN_MS
    requested_end = 204 * ONE_MIN_MS
    previous_real = 188 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (previous_real, 100.0, 100.0, 100.0, 100.0, 1.0),
            (201 * ONE_MIN_MS, 101.0, 101.0, 101.0, 101.0, 1.0),
            (202 * ONE_MIN_MS, 102.0, 102.0, 102.0, 102.0, 1.0),
            (203 * ONE_MIN_MS, 103.0, 103.0, 103.0, 103.0, 1.0),
            (requested_end, 104.0, 104.0, 104.0, 104.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        189 * ONE_MIN_MS,
        requested_start,
        reason=GAP_REASON_FETCH_FAILED,
    )

    result = await cm.get_candles(
        symbol,
        start_ts=requested_start,
        end_ts=requested_end,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=True,
    )

    assert list(result["ts"]) == [
        201 * ONE_MIN_MS,
        202 * ONE_MIN_MS,
        203 * ONE_MIN_MS,
        requested_end,
    ]
    assert requested_start not in cm._synthetic_timestamps.get(symbol, set())


@pytest.mark.asyncio
async def test_provisional_gap_tolerance_measures_remaining_uncovered_span(
    monkeypatch, tmp_path
):
    now = 205 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    cm = CandlestickManager(
        exchange=None,
        exchange_name="testex",
        cache_dir=str(tmp_path / "caches"),
        provisional_internal_gap_tolerance_minutes=10,
    )
    cm._now_ms_callback = lambda: now
    symbol = "RECOVEREDGAP/USDT:USDT"
    requested_start = 200 * ONE_MIN_MS
    requested_end = 204 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (
                ts,
                float(ts // ONE_MIN_MS),
                float(ts // ONE_MIN_MS),
                float(ts // ONE_MIN_MS),
                float(ts // ONE_MIN_MS),
                1.0,
            )
            for ts in [
                *range(189 * ONE_MIN_MS, requested_start, ONE_MIN_MS),
                201 * ONE_MIN_MS,
                202 * ONE_MIN_MS,
                203 * ONE_MIN_MS,
                requested_end,
            ]
        ],
        dtype=CANDLE_DTYPE,
    )
    # Metadata still has the original 12-minute outage, but authoritative
    # recovery has reduced the contiguous uncovered portion to one minute.
    cm._add_known_gap(
        symbol,
        189 * ONE_MIN_MS,
        requested_start,
        reason=GAP_REASON_FETCH_FAILED,
    )

    result = await cm.get_candles(
        symbol,
        start_ts=requested_start,
        end_ts=requested_end,
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=True,
    )

    assert list(result["ts"]) == [
        requested_start,
        201 * ONE_MIN_MS,
        202 * ONE_MIN_MS,
        203 * ONE_MIN_MS,
        requested_end,
    ]
    assert requested_start in cm._synthetic_timestamps[symbol]


@pytest.mark.asyncio
async def test_historical_fetch_splits_around_deferred_gap(
    monkeypatch, tmp_path
):
    now = {"ms": 1000 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    end = now["ms"] - 10 * ONE_MIN_MS
    start = end - 4 * ONE_MIN_MS
    deferred = start + ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 1.0, 1.0, 1.0, 1.0, 1.0),
            (end, 2.0, 2.0, 2.0, 2.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        deferred,
        deferred,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def prefetch_archives(*_args, **_kwargs):
        return None

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        rows = [
            (ts, 1.5, 1.5, 1.5, 1.5, 1.0)
            for ts in range(since, end_exclusive, ONE_MIN_MS)
        ]
        return np.array(rows, dtype=CANDLE_DTYPE)

    cm._prefetch_archives_for_range = prefetch_archives
    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(symbol, start_ts=start, end_ts=end)

    assert calls
    assert all(
        not (since <= deferred < end_exclusive)
        for since, end_exclusive in calls
    )
    assert deferred not in set(result["ts"].astype(np.int64))


@pytest.mark.asyncio
async def test_historical_partial_terminal_empty_flushes_deferred_index(
    monkeypatch, tmp_path
):
    now = {"ms": 1000 * ONE_MIN_MS}
    monkeypatch.setattr("time.time", lambda: now["ms"] / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._now_ms_callback = lambda: now["ms"]
    symbol = "MU/USDC:USDC"
    start = now["ms"] - 8 * ONE_MIN_MS
    end = now["ms"] - 4 * ONE_MIN_MS
    flush_calls = []

    async def prefetch_archives(*_args, **_kwargs):
        return None

    def flush_deferred_index(_symbol, *, timeframe=None, tf=None):
        flush_calls.append((_symbol, timeframe if timeframe is not None else tf))

    async def fetch_paginated(
        _symbol, since, end_exclusive, *, on_batch=None, **_kwargs
    ):
        partial = np.array(
            [(since, 1.0, 1.0, 1.0, 1.0, 1.0)],
            dtype=CANDLE_DTYPE,
        )
        assert on_batch is not None
        on_batch(partial)
        raise OhlcvTerminalEmptyPage(
            "terminal empty",
            partial_rows=partial,
            terminal_start_ts=since + ONE_MIN_MS,
            requested_end_ts=end_exclusive,
            pages=1,
        )

    cm._prefetch_archives_for_range = prefetch_archives
    cm.flush_deferred_index = flush_deferred_index
    cm._fetch_ohlcv_paginated = fetch_paginated

    with pytest.raises(OhlcvTerminalEmptyPage):
        await cm.get_candles(
            symbol,
            start_ts=start,
            end_ts=end,
            max_age_ms=0,
        )

    assert flush_calls == [(symbol, "1m")]


def test_nonexpiring_gap_does_not_defer_fetch_beyond_recorded_end(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="binance",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "TEST/USDT:USDT"
    start = 10 * ONE_MIN_MS
    cm._record_verified_gap(symbol, start, start)

    assert cm._known_gap_retry_deferred_at(symbol, start, start)
    assert not cm._known_gap_retry_deferred_at(
        symbol, start, start + ONE_MIN_MS
    )
    assert (
        cm._fetch_start_after_deferred_gap_prefix(
            symbol, start, start + ONE_MIN_MS
        )
        == start + ONE_MIN_MS
    )


@pytest.mark.asyncio
async def test_1m_force_refresh_raises_on_partial_terminal_empty_page(
    monkeypatch, tmp_path
):
    fixed_now_ms = 20 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        default_window_candles=4,
        overlap_candles=0,
    )
    cm._now_ms_callback = lambda: fixed_now_ms
    cm._ccxt_limit_default = 2
    cm._ccxt_page_overlap_candles = 0
    calls = []

    async def fetch_once(
        _symbol,
        since_ms,
        _limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        calls.append(since_ms)
        if len(calls) == 1:
            return [
                [since_ms, 1.0, 1.0, 1.0, 1.0, 1.0],
                [since_ms + ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        return []

    cm._ccxt_fetch_ohlcv_once = fetch_once
    start = fixed_now_ms - 4 * ONE_MIN_MS
    end = fixed_now_ms - ONE_MIN_MS

    with pytest.raises(OhlcvTerminalEmptyPage) as exc_info:
        await cm.get_candles(
            "MU/USDC:USDC",
            start_ts=start,
            end_ts=end,
            max_age_ms=0,
            max_lookback_candles=4,
        )

    assert calls == [start, start + 2 * ONE_MIN_MS]
    assert exc_info.value.pages == 1


@pytest.mark.asyncio
async def test_refresh_overlap_splits_around_deferred_internal_gap(
    monkeypatch, tmp_path
):
    fixed_now_ms = 100 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        overlap_candles=3,
    )
    cm._now_ms_callback = lambda: fixed_now_ms
    symbol = "MU/USDC:USDC"
    deferred = 95 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (94 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
            (96 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
            (97 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        deferred,
        deferred,
        reason=GAP_REASON_FETCH_FAILED,
    )
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.array(
            [
                (ts, 2.0, 2.0, 2.0, 2.0, 1.0)
                for ts in range(since, end_exclusive, ONE_MIN_MS)
            ],
            dtype=CANDLE_DTYPE,
        )

    cm._fetch_ohlcv_paginated = fetch_paginated

    await cm.refresh(symbol, through_ts=fixed_now_ms - ONE_MIN_MS)

    assert calls == [
        (94 * ONE_MIN_MS, 95 * ONE_MIN_MS),
        (96 * ONE_MIN_MS, 100 * ONE_MIN_MS),
    ]
    assert all(
        not (since <= deferred < end_exclusive)
        for since, end_exclusive in calls
    )


@pytest.mark.asyncio
async def test_refresh_stamps_due_gap_remainder_before_targeted_repair(
    monkeypatch, tmp_path
):
    fixed_now_ms = 100 * ONE_MIN_MS
    monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

    class _Ex:
        id = "hyperliquid"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="hyperliquid",
        cache_dir=str(tmp_path / "caches"),
        overlap_candles=3,
    )
    cm._now_ms_callback = lambda: fixed_now_ms
    symbol = "MU/USDC:USDC"
    start = 94 * ONE_MIN_MS
    missing = 95 * ONE_MIN_MS
    end = 99 * ONE_MIN_MS
    cm._cache[symbol] = np.array(
        [
            (start, 1.0, 1.0, 1.0, 1.0, 1.0),
            (96 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
            (97 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        missing,
        missing,
        reason=GAP_REASON_FETCH_FAILED,
    )
    gaps = cm._get_known_gaps_enhanced(symbol)
    gaps[0]["last_retry_at"] = fixed_now_ms - 5 * ONE_MIN_MS
    cm._save_known_gaps_enhanced(symbol, gaps)
    calls = []

    async def fetch_paginated(_symbol, since, end_exclusive, **_kwargs):
        calls.append((since, end_exclusive))
        return np.array(
            [
                (ts, 2.0, 2.0, 2.0, 2.0, 1.0)
                for ts in range(since, end_exclusive, ONE_MIN_MS)
                if ts != missing
            ],
            dtype=CANDLE_DTYPE,
        )

    cm._fetch_ohlcv_paginated = fetch_paginated

    result = await cm.get_candles(
        symbol,
        start_ts=start,
        end_ts=end,
        max_age_ms=0,
        fill_trailing_gaps=False,
    )

    assert calls == [(start, end + ONE_MIN_MS)]
    assert missing not in set(result["ts"].astype(np.int64))
    unresolved = cm._get_known_gaps_enhanced(symbol)
    assert [(gap["start_ts"], gap["end_ts"]) for gap in unresolved] == [
        (missing, missing)
    ]
    assert unresolved[0]["last_retry_at"] == fixed_now_ms


@pytest.mark.asyncio
async def test_overlap_pagination_stops_after_requested_end(tmp_path):
    class _Ex:
        id = "bitget"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="bitget",
        cache_dir=str(tmp_path / "caches"),
    )
    cm._ccxt_limit_default = 2
    cm._ccxt_page_overlap_candles = 1
    start = 10 * ONE_MIN_MS
    end_exclusive = start + 4 * ONE_MIN_MS
    calls = []

    async def fetch_once(
        _symbol,
        since_ms,
        _limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        calls.append(since_ms)
        if len(calls) == 1:
            return [
                [start, 1.0, 1.0, 1.0, 1.0, 1.0],
                [start + ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        if len(calls) == 2:
            return [
                [start + ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
                [start + 2 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
                [start + 3 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        return []

    cm._ccxt_fetch_ohlcv_once = fetch_once

    result = await cm._fetch_ohlcv_paginated(
        "BTC/USDT:USDT",
        start,
        end_exclusive,
        raise_on_partial_empty_page=True,
    )

    assert calls == [start - ONE_MIN_MS, start]
    assert set(result["ts"].astype(np.int64)) == {
        start,
        start + ONE_MIN_MS,
        start + 2 * ONE_MIN_MS,
        start + 3 * ONE_MIN_MS,
    }


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


def test_kucoin_between_page_holes_recorded_as_expiring_auto_gaps(tmp_path):
    """Intra-payload holes are exchange-verified no-trade minutes (the exchange
    returned the surrounding candles in one response) and stay permanent.
    Between-page holes are indistinguishable from a pagination stall or outage,
    so they must be recorded with the expiring auto_detected classification and
    remain retryable instead of being permanently masked as no_trades."""

    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    assert cm._record_payload_gaps_as_known

    base = _floor_minute(int(time.time() * 1000)) - 100 * ONE_MIN_MS

    def t(i):
        return base + i * ONE_MIN_MS

    def row(i):
        return [t(i), 100.0, 101.0, 99.0, 100.5, 5.0]

    pages = [
        [row(0), row(1), row(3)],  # intra-payload hole at minute 2
        [row(6), row(7)],  # between-page hole covering minutes 4-5
    ]

    async def fake_once(symbol, since_ms, limit, end_exclusive_ms=None, timeframe=None, *, tf=None):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(cm._fetch_ohlcv_paginated("ETH/USDT:USDT", t(0), t(8)))
    assert arr.shape[0] == 5

    gaps = cm._get_known_gaps_enhanced("ETH/USDT:USDT")
    by_range = {(int(g["start_ts"]), int(g["end_ts"])): g for g in gaps}
    assert set(by_range) == {(t(2), t(2)), (t(4), t(5))}

    intra = by_range[(t(2), t(2))]
    assert intra["reason"] == "no_trades"
    assert int(intra["retry_count"]) >= _GAP_MAX_RETRIES
    assert not cm._should_retry_gap(intra)

    between = by_range[(t(4), t(5))]
    assert between["reason"] == "auto_detected"
    assert int(between["retry_count"]) < _GAP_MAX_RETRIES
    assert cm._should_retry_gap(between)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_count", [1, _GAP_MAX_RETRIES])
async def test_kucoin_contextual_retry_proves_sparse_gap_immediately(
    tmp_path, monkeypatch, retry_count
):
    """A retry must include real rows on both sides of a sparse KuCoin gap.

    Querying only the absent timestamps returns an empty payload and leaves the
    gap as fetch_failed forever. One successful payload containing both bounds
    proves the omitted interval as no-trade continuity.
    """

    now = 25 * ONE_MIN_MS + 1_000
    monkeypatch.setattr("time.time", lambda: now / 1000.0)
    calls = []

    class _Ex:
        id = "kucoinfutures"

        async def fetch_ohlcv(
            self, symbol, timeframe=None, since=None, limit=None, params=None
        ):
            calls.append((symbol, timeframe, since, limit, params))
            return [
                [0, 100.0, 101.0, 99.0, 100.0, 5.0],
                [23 * ONE_MIN_MS, 110.0, 111.0, 109.0, 110.0, 7.0],
            ]

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "SPARSE/USDT:USDT"
    cm._cache[symbol] = np.array(
        [
            (0, 100.0, 101.0, 99.0, 100.0, 5.0),
            (23 * ONE_MIN_MS, 110.0, 111.0, 109.0, 110.0, 7.0),
            (24 * ONE_MIN_MS, 110.0, 112.0, 109.0, 111.0, 8.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        ONE_MIN_MS,
        22 * ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=retry_count,
    )

    result = await cm.get_candles(
        symbol,
        start_ts=0,
        end_ts=24 * ONE_MIN_MS,
        max_age_ms=None,
    )

    assert calls
    assert calls[0][2] == 0
    assert list(result["ts"]) == list(range(0, 25 * ONE_MIN_MS, ONE_MIN_MS))
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == GAP_REASON_NO_TRADES
    assert gaps[0]["start_ts"] == ONE_MIN_MS
    assert gaps[0]["end_ts"] == 22 * ONE_MIN_MS
    assert not cm._should_retry_gap(gaps[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "returned_minutes",
    [
        (),
        (0,),
        (23,),
        (0, 10),
        (0, 10, 23),
    ],
    ids=(
        "empty",
        "left-only",
        "right-only",
        "left-plus-interior",
        "both-bounds-plus-interior",
    ),
)
async def test_kucoin_contextual_retry_requires_complete_single_payload_proof_and_cools_down(
    tmp_path, monkeypatch, returned_minutes
):
    """Incomplete contextual proof must remain unavailable and respect cooldown."""

    clock = {"now": 25 * ONE_MIN_MS + 1_000}
    monkeypatch.setattr("time.time", lambda: clock["now"] / 1000.0)
    calls = []

    class _Ex:
        id = "kucoinfutures"

        async def fetch_ohlcv(
            self, symbol, timeframe=None, since=None, limit=None, params=None
        ):
            calls.append((symbol, timeframe, since, limit, params))
            return [
                [
                    minute * ONE_MIN_MS,
                    100.0 + minute,
                    101.0 + minute,
                    99.0 + minute,
                    100.0 + minute,
                    5.0,
                ]
                for minute in returned_minutes
            ]

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "SPARSE/USDT:USDT"
    cm._cache[symbol] = np.array(
        [
            (0, 100.0, 101.0, 99.0, 100.0, 5.0),
            (23 * ONE_MIN_MS, 110.0, 111.0, 109.0, 110.0, 7.0),
            (24 * ONE_MIN_MS, 110.0, 112.0, 109.0, 111.0, 8.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._add_known_gap(
        symbol,
        ONE_MIN_MS,
        22 * ONE_MIN_MS,
        reason=GAP_REASON_FETCH_FAILED,
        retry_count=_GAP_MAX_RETRIES,
    )
    first = await cm.get_candles(
        symbol,
        start_ts=0,
        end_ts=24 * ONE_MIN_MS,
        max_age_ms=None,
    )

    assert len(calls) == 1
    expected_real_timestamps = {
        0,
        23 * ONE_MIN_MS,
        24 * ONE_MIN_MS,
    } | {
        int(minute) * ONE_MIN_MS
        for minute in returned_minutes
        if 1 <= int(minute) <= 22
    }
    assert set(int(ts) for ts in first["ts"]) == expected_real_timestamps
    unresolved = cm._get_known_gaps_enhanced(symbol)
    assert unresolved
    assert all(gap["reason"] == GAP_REASON_FETCH_FAILED for gap in unresolved)
    assert all(
        int(gap["retry_count"]) >= _GAP_MAX_RETRIES
        for gap in unresolved
    )
    assert all(
        int(gap["last_contextual_retry_at"]) == clock["now"]
        for gap in unresolved
    )

    await cm.get_candles(
        symbol,
        start_ts=0,
        end_ts=24 * ONE_MIN_MS,
        max_age_ms=None,
    )
    assert len(calls) == 1

    due_again = cm._get_known_gaps_enhanced(symbol)
    for gap in due_again:
        gap["last_contextual_retry_at"] = (
            clock["now"] - _GAP_PERSISTENT_RETRY_MS - 1
        )
    cm._save_known_gaps_enhanced(symbol, due_again)
    await cm.get_candles(
        symbol,
        start_ts=0,
        end_ts=24 * ONE_MIN_MS,
        max_age_ms=None,
    )
    assert len(calls) == 1 + len(unresolved)


@pytest.mark.asyncio
async def test_kucoin_contextual_page_overlaps_exclusive_since_to_include_left_bound(
    tmp_path,
):
    calls = []
    left = 10 * ONE_MIN_MS
    right = 13 * ONE_MIN_MS

    class _Ex:
        id = "kucoinfutures"

        async def fetch_ohlcv(
            self, symbol, timeframe=None, since=None, limit=None, params=None
        ):
            calls.append((symbol, timeframe, since, limit, params))
            return [
                [left, 100.0, 101.0, 99.0, 100.0, 5.0],
                [right, 110.0, 111.0, 109.0, 110.0, 7.0],
            ]

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    rows, proof = await cm._fetch_kucoin_contextual_gap_page(
        "SPARSE/USDT:USDT",
        left_boundary_ts=left,
        right_boundary_ts=right,
        gap_start_ts=11 * ONE_MIN_MS,
        gap_end_ts=12 * ONE_MIN_MS,
    )

    assert calls[0][2] == left - ONE_MIN_MS
    assert list(rows["ts"]) == [left, right]
    assert proof is True


def test_kucoin_h1_payload_synthesizes_only_internally_bounded_no_trade_buckets(
    tmp_path,
):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 10 * hour_ms

    def row(offset, close):
        ts = base + offset * hour_ms
        return [ts, close, close + 1.0, close - 1.0, close, 5.0]

    pages = [
        [
            row(1, 101.0),
            row(2, 102.0),
            row(4, 104.0),
        ],
        [
            row(7, 107.0),
            row(8, 108.0),
        ],
    ]
    persisted = []

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            base,
            base + 9 * hour_ms,
            timeframe="1h",
            on_batch=lambda batch: persisted.append(batch.copy()),
        )
    )

    assert list(arr["ts"]) == [
        base + hour_ms,
        base + 2 * hour_ms,
        base + 3 * hour_ms,
        base + 4 * hour_ms,
        base + 7 * hour_ms,
        base + 8 * hour_ms,
    ]
    synthetic = arr[arr["ts"] == base + 3 * hour_ms]
    assert synthetic.shape[0] == 1
    assert float(synthetic[0]["o"]) == pytest.approx(102.0)
    assert float(synthetic[0]["h"]) == pytest.approx(102.0)
    assert float(synthetic[0]["l"]) == pytest.approx(102.0)
    assert float(synthetic[0]["c"]) == pytest.approx(102.0)
    assert float(synthetic[0]["bv"]) == pytest.approx(0.0)
    assert base not in set(arr["ts"])
    assert base + 5 * hour_ms not in set(arr["ts"])
    assert base + 6 * hour_ms not in set(arr["ts"])
    assert len(persisted) == 2
    assert base + 3 * hour_ms in set(persisted[0]["ts"])
    assert base + 5 * hour_ms not in set(persisted[1]["ts"])


@pytest.mark.parametrize(
    "invalid_values",
    [
        (100.0, float("nan"), 99.0, 100.0, 5.0),
        (100.0, 101.0, 0.0, 100.0, 5.0),
        (100.0, 101.0, 99.0, 100.0, -1.0),
    ],
)
def test_kucoin_h1_rejected_payload_bucket_remains_unavailable(
    tmp_path,
    invalid_values,
):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 3 * hour_ms

    raw = [
        [base, 100.0, 101.0, 99.0, 100.0, 5.0],
        [base + hour_ms, *invalid_values],
        [base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0],
    ]
    pages = [raw]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            base,
            base + 3 * hour_ms,
            timeframe="1h",
        )
    )

    assert list(arr["ts"]) == [base, base + 2 * hour_ms]


def test_kucoin_h1_rejected_real_bucket_evicts_persisted_sparse_placeholder(
    tmp_path,
):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    symbol = "MORPHO/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 3 * hour_ms
    cached = np.array(
        [
            (base, 100.0, 101.0, 99.0, 100.0, 5.0),
            (base + hour_ms, 100.0, 100.0, 100.0, 100.0, 0.0),
            (base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, cached, timeframe="1h")
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [base + hour_ms, 100.0, float("nan"), 99.0, 100.0, 5.0],
            [base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    asyncio.run(
        cm._fetch_ohlcv_paginated(
            symbol,
            base,
            base + 3 * hour_ms,
            timeframe="1h",
        )
    )

    reloaded = cm._load_from_disk(
        symbol,
        base,
        base + 2 * hour_ms,
        timeframe="1h",
    )
    assert list(reloaded["ts"]) == [base, base + 2 * hour_ms]


def test_kucoin_h1_unidentifiable_rejection_evicts_bounded_cached_placeholders(
    tmp_path,
):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    symbol = "MORPHO/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 3 * hour_ms
    cached = np.array(
        [
            (base, 100.0, 101.0, 99.0, 100.0, 5.0),
            (base + hour_ms, 100.0, 100.0, 100.0, 100.0, 0.0),
            (base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, cached, timeframe="1h")
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [None, 101.0, 102.0, 100.0, 101.0, 5.0],
            [base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    asyncio.run(
        cm._fetch_ohlcv_paginated(
            symbol,
            base,
            base + 3 * hour_ms,
            timeframe="1h",
        )
    )

    reloaded = cm._load_from_disk(
        symbol,
        base,
        base + 2 * hour_ms,
        timeframe="1h",
    )
    assert list(reloaded["ts"]) == [base, base + 2 * hour_ms]


@pytest.mark.parametrize("include_one_accepted_row", [False, True])
def test_kucoin_h1_unidentifiable_rejection_without_two_bounds_evicts_requested_placeholders(
    tmp_path,
    include_one_accepted_row,
):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    symbol = "MORPHO/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 3 * hour_ms
    cached = np.array(
        [
            (base, 100.0, 101.0, 99.0, 100.0, 5.0),
            (base + hour_ms, 100.0, 100.0, 100.0, 100.0, 0.0),
            (base + 2 * hour_ms, 100.0, 100.0, 100.0, 100.0, 0.0),
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._persist_batch(symbol, cached, timeframe="1h")
    raw_page = [[None, 101.0, 102.0, 100.0, 101.0, 5.0]]
    if include_one_accepted_row:
        raw_page.insert(0, [base, 100.0, 101.0, 99.0, 100.0, 5.0])
    pages = [raw_page]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    asyncio.run(
        cm._fetch_ohlcv_paginated(
            symbol,
            base,
            base + 3 * hour_ms,
            timeframe="1h",
        )
    )

    reloaded = cm._load_from_disk(
        symbol,
        base,
        base + 2 * hour_ms,
        timeframe="1h",
    )
    assert list(reloaded["ts"]) == [base]
    assert cm.get_last_final_ts(symbol, timeframe="1h") == base


def test_native_sparse_placeholder_eviction_recomputes_empty_index_bounds(tmp_path):
    cm = CandlestickManager(
        exchange=None,
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
    )
    symbol = "MORPHO/USDT:USDT"
    hour_ms = 60 * ONE_MIN_MS
    ts = (int(time.time() * 1000) // hour_ms) * hour_ms - hour_ms
    cm._persist_batch(
        symbol,
        np.array([(ts, 100.0, 100.0, 100.0, 100.0, 0.0)], dtype=CANDLE_DTYPE),
        timeframe="1h",
    )
    assert cm.get_last_final_ts(symbol, timeframe="1h") == ts

    removed = cm._evict_rejected_native_sparse_synthetics(
        symbol,
        timeframe="1h",
        rejected_timestamps={ts},
    )

    assert removed == {ts}
    assert cm.get_last_final_ts(symbol, timeframe="1h") == 0
    idx = cm._ensure_symbol_index(symbol, timeframe="1h")
    assert idx["meta"]["observed_start_ts"] is None
    assert idx["meta"]["inception_ts"] is None


def test_kucoin_h1_unidentifiable_rejected_row_disables_page_synthesis(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 3 * hour_ms
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [None, 101.0, 102.0, 100.0, 101.0, 5.0],
            [base + 2 * hour_ms, 102.0, 103.0, 101.0, 102.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            base,
            base + 3 * hour_ms,
            timeframe="1h",
        )
    )

    assert list(arr["ts"]) == [base, base + 2 * hour_ms]


def test_kucoin_h1_rejected_bucket_is_continuity_barrier(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 4 * hour_ms
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [base + hour_ms, 100.0, float("nan"), 99.0, 100.0, 5.0],
            [base + 3 * hour_ms, 300.0, 301.0, 299.0, 300.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            base,
            base + 4 * hour_ms,
            timeframe="1h",
        )
    )

    assert list(arr["ts"]) == [base, base + 3 * hour_ms]


def test_kucoin_h1_sparse_expansion_is_bounded_to_requested_range(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
        gap_tolerance_ohlcvs_minutes=2_000 * 60,
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 2_000 * hour_ms
    request_start = base + 1_000 * hour_ms
    request_end = request_start + 3 * hour_ms
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [base + 2_000 * hour_ms, 200.0, 201.0, 199.0, 200.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            request_start,
            request_end,
            timeframe="1h",
        )
    )

    in_range = arr[
        (arr["ts"] >= request_start)
        & (arr["ts"] < request_end)
    ]
    assert list(in_range["ts"]) == [
        request_start,
        request_start + hour_ms,
        request_start + 2 * hour_ms,
    ]
    assert arr.shape[0] == 4


def test_kucoin_h1_sparse_synthesis_respects_internal_gap_tolerance(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(),
        exchange_name="kucoin",
        cache_dir=str(tmp_path / "caches"),
        gap_tolerance_ohlcvs_minutes=120,
    )
    hour_ms = 60 * ONE_MIN_MS
    base = (int(time.time() * 1000) // hour_ms) * hour_ms - 5 * hour_ms
    pages = [
        [
            [base, 100.0, 101.0, 99.0, 100.0, 5.0],
            [base + 4 * hour_ms, 104.0, 105.0, 103.0, 104.0, 5.0],
        ]
    ]

    async def fake_once(
        symbol,
        since_ms,
        limit,
        end_exclusive_ms=None,
        timeframe=None,
        *,
        tf=None,
    ):
        return pages.pop(0) if pages else []

    cm._ccxt_fetch_ohlcv_once = fake_once
    arr = asyncio.run(
        cm._fetch_ohlcv_paginated(
            "MORPHO/USDT:USDT",
            base,
            base + 5 * hour_ms,
            timeframe="1h",
        )
    )

    assert list(arr["ts"]) == [base, base + 4 * hour_ms]


def test_rejected_duplicate_timestamp_is_accepted_when_any_row_is_valid(tmp_path):
    class _Ex:
        id = "kucoinfutures"

    cm = CandlestickManager(
        exchange=_Ex(), exchange_name="kucoin", cache_dir=str(tmp_path / "caches")
    )
    base = _floor_minute(int(time.time() * 1000)) - ONE_MIN_MS
    raw = [
        [base, 100.0, float("nan"), 99.0, 100.0, 5.0],
        [base, 100.0, 101.0, 99.0, 100.0, 5.0],
    ]

    normalized = cm._normalize_ccxt_ohlcv(raw)

    assert list(normalized["ts"]) == [base]
    assert cm._rejected_ccxt_ohlcv_timestamps(raw, normalized) == set()
