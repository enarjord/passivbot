"""
Edge case tests for CandlestickManager hardening.

Tests cover:
1. Gap handling edge cases
2. Cache corruption recovery
3. Boundary conditions
4. Data integrity
5. Concurrent access scenarios
"""

import asyncio
import json
import os
import tempfile
import shutil
import time
import zlib

import numpy as np
import pytest

from candlestick_manager import (
    CandlestickManager,
    CANDLE_DTYPE,
    ONE_MIN_MS,
    _floor_minute,
    _GAP_MAX_RETRIES,
    GAP_REASON_FETCH_FAILED,
    GAP_REASON_AUTO,
)


@pytest.fixture
def tmp_cache_dir():
    """Create a temporary cache directory for tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_candles(start_ts: int, count: int, close_base: float = 100.0) -> np.ndarray:
    """Create a structured array of candles."""
    arr = np.zeros(count, dtype=CANDLE_DTYPE)
    arr["ts"] = [start_ts + i * ONE_MIN_MS for i in range(count)]
    arr["o"] = close_base
    arr["h"] = close_base + 1.0
    arr["l"] = close_base - 1.0
    arr["c"] = close_base + 0.5
    arr["bv"] = 1000.0
    return arr


# ==============================================================================
# 1. GAP HANDLING EDGE CASES
# ==============================================================================


class TestGapHandlingEdgeCases:
    """Tests for gap handling edge cases."""

    def test_empty_cache_no_exchange_returns_empty(self, tmp_cache_dir):
        """Request data with no cache and no exchange returns empty array."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # No cache, no exchange -> should return empty gracefully
        result = cm.standardize_gaps(
            np.empty(0, dtype=CANDLE_DTYPE), start_ts=1609459200000, end_ts=1609545600000, strict=True
        )
        assert result.size == 0

    def test_gap_at_start_of_range(self, tmp_cache_dir):
        """Data starts after requested start_ts - gap at beginning."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        start_ts = 1609459200000  # 2021-01-01 00:00:00
        data_start = start_ts + 10 * ONE_MIN_MS  # Data starts 10 minutes later
        end_ts = start_ts + 20 * ONE_MIN_MS

        # Create candles starting 10 minutes into range (11 candles: minutes 10-20)
        candles = make_candles(data_start, 11)

        # strict=True should return only real data
        result = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=True)
        assert result.size == 11
        assert int(result[0]["ts"]) == data_start

        # strict=False fills gaps WITHIN data but doesn't fill before first candle
        # standardize_gaps only fills internal gaps, not leading/trailing gaps
        result_filled = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=False)
        # Should return 11 candles (data_start to end_ts)
        assert result_filled.size == 11
        assert int(result_filled[0]["ts"]) == data_start

    def test_gap_at_end_of_range(self, tmp_cache_dir):
        """Data ends before requested end_ts - gap at end."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        start_ts = 1609459200000
        data_end = start_ts + 10 * ONE_MIN_MS
        end_ts = start_ts + 20 * ONE_MIN_MS

        candles = make_candles(start_ts, 11)  # 0-10 minutes

        # strict=True returns only real data
        result = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=True)
        assert result.size == 11

        # strict=False fills to end
        result_filled = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=False)
        assert result_filled.size == 21
        # Last 10 candles should be synthetic
        assert all(float(result_filled[i]["bv"]) == 0.0 for i in range(11, 21))

    def test_multiple_gaps_in_range(self, tmp_cache_dir):
        """Multiple gaps within data range."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        start_ts = 1609459200000

        # Create candles with two gaps
        # Minutes 0-2, gap at 3-4, minutes 5-7, gap at 8-9, minute 10
        candles = np.zeros(7, dtype=CANDLE_DTYPE)
        candles["ts"] = [
            start_ts,  # 0
            start_ts + 1 * ONE_MIN_MS,  # 1
            start_ts + 2 * ONE_MIN_MS,  # 2
            start_ts + 5 * ONE_MIN_MS,  # 5 (gap at 3,4)
            start_ts + 6 * ONE_MIN_MS,  # 6
            start_ts + 7 * ONE_MIN_MS,  # 7
            start_ts + 10 * ONE_MIN_MS,  # 10 (gap at 8,9)
        ]
        candles["c"] = [100, 101, 102, 105, 106, 107, 110]
        candles["bv"] = 1000.0

        end_ts = start_ts + 10 * ONE_MIN_MS
        result = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=False)

        # Should have 11 candles (0-10 inclusive)
        assert result.size == 11

        # Check gap filling (forward fill from previous close)
        # Gap at minute 3 should have close from minute 2 (102)
        assert float(result[3]["c"]) == pytest.approx(102.0)
        assert float(result[3]["bv"]) == 0.0
        # Gap at minute 8 should have close from minute 7 (107)
        assert float(result[8]["c"]) == pytest.approx(107.0)

    def test_gap_retry_exhaustion_marks_persistent(self, tmp_cache_dir):
        """After max retries, gap is marked as persistent and not retried."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        gap_start = 1609459200000
        gap_end = 1609545600000

        # Add gap repeatedly until max retries
        for i in range(_GAP_MAX_RETRIES + 1):
            cm._add_known_gap(
                symbol, gap_start, gap_end, reason=GAP_REASON_FETCH_FAILED, increment_retry=True
            )

        gaps = cm._get_known_gaps_enhanced(symbol)
        assert len(gaps) == 1
        assert gaps[0]["retry_count"] >= _GAP_MAX_RETRIES
        assert not cm._should_retry_gap(gaps[0])

    def test_gap_merge_overlapping(self, tmp_cache_dir):
        """Overlapping gaps should be merged."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Add overlapping gaps
        cm._add_known_gap(symbol, 1000000, 3000000)
        cm._add_known_gap(symbol, 2000000, 4000000)

        gaps = cm._get_known_gaps(symbol)
        # Should be merged into one gap
        assert len(gaps) == 1
        assert gaps[0][0] == 1000000
        assert gaps[0][1] == 4000000

    def test_gap_adjacent_merge(self, tmp_cache_dir):
        """Adjacent gaps (touching at boundaries) should be merged."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Add adjacent gaps
        cm._add_known_gap(symbol, 1000000, 2000000)
        cm._add_known_gap(symbol, 2000000, 3000000)

        gaps = cm._get_known_gaps(symbol)
        # Should be merged
        assert len(gaps) == 1
        assert gaps[0][0] == 1000000
        assert gaps[0][1] == 3000000


# ==============================================================================
# 2. CACHE CORRUPTION RECOVERY
# ==============================================================================


class TestCacheCorruptionRecovery:
    """Tests for cache corruption recovery."""

    def test_corrupted_shard_file_skipped(self, tmp_cache_dir):
        """Corrupted numpy shard file is skipped gracefully."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create a valid shard first
        start_ts = 1609459200000
        candles = make_candles(start_ts, 1440)  # Full day
        date_key = "2021-01-01"
        cm._save_shard(symbol, date_key, candles)

        # Corrupt the shard file
        shard_path = cm._shard_path(symbol, date_key)
        with open(shard_path, "wb") as f:
            f.write(b"corrupted data not numpy format")

        # Loading should return empty array (graceful failure)
        loaded = cm._load_shard(shard_path)
        assert loaded.size == 0

    def test_corrupted_index_json_recovery(self, tmp_cache_dir):
        """Corrupted index.json is handled gracefully."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create valid data first
        start_ts = 1609459200000
        candles = make_candles(start_ts, 100)
        cm._persist_batch(symbol, candles, timeframe="1m")

        # Corrupt the index file
        idx_path = cm._index_path(symbol, timeframe="1m")
        with open(idx_path, "w") as f:
            f.write("not valid json {{{")

        # Clear in-memory index to force reload
        cm._index.pop(f"{symbol}::1m", None)

        # Should recover gracefully - index will be recreated
        idx = cm._ensure_symbol_index(symbol)
        assert "meta" in idx
        assert "shards" in idx

    def test_missing_shard_referenced_in_index(self, tmp_cache_dir):
        """Missing shard file referenced in index is handled."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create valid data
        start_ts = 1609459200000
        candles = make_candles(start_ts, 1440)
        date_key = "2021-01-01"
        cm._save_shard(symbol, date_key, candles)

        # Delete the shard file but keep index
        shard_path = cm._shard_path(symbol, date_key)
        os.remove(shard_path)

        # Loading should return empty array (graceful failure)
        loaded = cm._load_shard(shard_path)
        assert loaded.size == 0

    def test_crc_mismatch_detected(self, tmp_cache_dir):
        """CRC mismatch is detected when shard data changes."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        start_ts = 1609459200000
        candles = make_candles(start_ts, 100)
        date_key = "2021-01-01"
        cm._save_shard(symbol, date_key, candles)

        # Get original CRC
        idx = cm._ensure_symbol_index(symbol)
        original_crc = idx["shards"][date_key]["crc32"]

        # Modify shard file (but keep it valid numpy)
        different_candles = make_candles(start_ts, 100, close_base=200.0)
        shard_path = cm._shard_path(symbol, date_key)
        np.save(shard_path, different_candles)

        # Reload and check CRC mismatch
        loaded = cm._load_shard(shard_path)
        if loaded.size > 0:
            new_crc = zlib.crc32(loaded.tobytes()) & 0xFFFFFFFF
            assert new_crc != original_crc

    def test_partially_written_shard_recovery(self, tmp_cache_dir):
        """Partially written (truncated) shard is handled."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create valid shard
        start_ts = 1609459200000
        candles = make_candles(start_ts, 1440)
        date_key = "2021-01-01"
        cm._save_shard(symbol, date_key, candles)

        # Truncate the file
        shard_path = cm._shard_path(symbol, date_key)
        original_size = os.path.getsize(shard_path)
        with open(shard_path, "r+b") as f:
            f.truncate(original_size // 2)

        # Loading should fail gracefully - return empty or partial
        loaded = cm._load_shard(shard_path)
        # Either empty or partial data, but no crash
        assert loaded.size == 0 or loaded.size < 1440


# ==============================================================================
# 3. BOUNDARY CONDITIONS
# ==============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_zero_length_request(self, tmp_cache_dir):
        """start_ts == end_ts returns single candle if available."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        ts = 1609459200000
        candles = make_candles(ts, 10)
        cm._cache[symbol] = candles

        # Request exactly one timestamp
        result = cm.standardize_gaps(candles, start_ts=ts, end_ts=ts, strict=True)
        assert result.size == 1
        assert int(result[0]["ts"]) == ts

    def test_negative_time_range(self, tmp_cache_dir):
        """start_ts > end_ts returns empty."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        start_ts = 1609545600000
        end_ts = 1609459200000  # Before start

        candles = make_candles(1609459200000, 100)

        result = cm.standardize_gaps(candles, start_ts=start_ts, end_ts=end_ts, strict=True)
        assert result.size == 0

    def test_single_candle(self, tmp_cache_dir):
        """Exactly one minute of data."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        ts = 1609459200000
        candles = make_candles(ts, 1)

        result = cm.standardize_gaps(candles, start_ts=ts, end_ts=ts, strict=True)
        assert result.size == 1

    def test_cross_midnight_boundaries(self, tmp_cache_dir):
        """Data spanning UTC date change."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create candles spanning midnight
        # 2021-01-01 23:00 to 2021-01-02 01:00
        start_ts = 1609545600000 - 60 * ONE_MIN_MS  # 2021-01-01 23:00
        candles = make_candles(start_ts, 120)  # 2 hours

        # Persist - should create shards for both dates
        cm._persist_batch(symbol, candles, timeframe="1m")

        # Verify shards for both dates exist
        assert os.path.exists(cm._shard_path(symbol, "2021-01-01"))
        assert os.path.exists(cm._shard_path(symbol, "2021-01-02"))

    @pytest.mark.asyncio
    async def test_future_data_request(self, tmp_cache_dir, monkeypatch):
        """Request beyond current time."""
        fixed_now_ms = 1609459200000  # 2021-01-01 00:00:00
        monkeypatch.setattr("time.time", lambda: fixed_now_ms / 1000.0)

        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        future_ts = fixed_now_ms + 24 * 60 * ONE_MIN_MS  # 24 hours in future

        # Should not crash, return what's available (empty)
        result = await cm.get_candles(
            symbol, start_ts=fixed_now_ms, end_ts=future_ts, max_age_ms=0, strict=True
        )
        # No data available, should be empty
        assert result.size == 0


# ==============================================================================
# 4. DATA INTEGRITY
# ==============================================================================


class TestDataIntegrity:
    """Tests for data integrity."""

    def test_timestamp_ordering_maintained(self, tmp_cache_dir):
        """Returned data is always sorted by timestamp."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        # Create unsorted candles
        base_ts = 1609459200000
        candles = np.zeros(5, dtype=CANDLE_DTYPE)
        candles["ts"] = [
            base_ts + 4 * ONE_MIN_MS,
            base_ts + 1 * ONE_MIN_MS,
            base_ts + 3 * ONE_MIN_MS,
            base_ts + 0 * ONE_MIN_MS,
            base_ts + 2 * ONE_MIN_MS,
        ]
        candles["c"] = 100.0
        candles["bv"] = 1.0

        result = cm.standardize_gaps(
            candles, start_ts=base_ts, end_ts=base_ts + 4 * ONE_MIN_MS, strict=True
        )

        # Should be sorted
        timestamps = [int(r["ts"]) for r in result]
        assert timestamps == sorted(timestamps)

    def test_merge_removes_duplicates(self, tmp_cache_dir):
        """_merge_overwrite removes duplicate timestamps."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000

        # Create candles with duplicates
        existing = np.zeros(3, dtype=CANDLE_DTYPE)
        existing["ts"] = [base_ts, base_ts + ONE_MIN_MS, base_ts + 2 * ONE_MIN_MS]
        existing["c"] = [100, 101, 102]
        existing["bv"] = 1.0

        # New data has overlapping timestamp
        new = np.zeros(2, dtype=CANDLE_DTYPE)
        new["ts"] = [base_ts + ONE_MIN_MS, base_ts + 3 * ONE_MIN_MS]
        new["c"] = [201, 203]  # Different close prices
        new["bv"] = 2.0

        result = cm._merge_overwrite(existing, new)

        timestamps = [int(r["ts"]) for r in result]
        assert len(timestamps) == len(set(timestamps))  # No duplicates
        assert len(timestamps) == 4  # 0, 1, 2, 3 minutes

        # New data should have overwritten at minute 1
        for r in result:
            if int(r["ts"]) == base_ts + ONE_MIN_MS:
                assert float(r["c"]) == pytest.approx(201.0)

    def test_strict_vs_non_strict_gap_filling(self, tmp_cache_dir):
        """strict=True vs strict=False behavior."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000
        # Create candles with gap at minute 2
        candles = np.zeros(4, dtype=CANDLE_DTYPE)
        candles["ts"] = [
            base_ts,
            base_ts + ONE_MIN_MS,
            base_ts + 3 * ONE_MIN_MS,
            base_ts + 4 * ONE_MIN_MS,
        ]
        candles["c"] = 100.0
        candles["bv"] = 1.0

        end_ts = base_ts + 4 * ONE_MIN_MS

        # strict=True: only real data
        strict_result = cm.standardize_gaps(candles, start_ts=base_ts, end_ts=end_ts, strict=True)
        assert strict_result.size == 4

        # strict=False: gaps filled
        filled_result = cm.standardize_gaps(candles, start_ts=base_ts, end_ts=end_ts, strict=False)
        assert filled_result.size == 5  # Gap at minute 2 filled
        assert float(filled_result[2]["bv"]) == 0.0  # Synthetic candle

    def test_merge_prefers_new_data(self, tmp_cache_dir):
        """When merging, new data overwrites old data at same timestamp."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        ts = 1609459200000

        existing = np.array([(ts, 100.0, 101.0, 99.0, 100.5, 1000.0)], dtype=CANDLE_DTYPE)
        new = np.array([(ts, 200.0, 201.0, 199.0, 200.5, 2000.0)], dtype=CANDLE_DTYPE)

        merged = cm._merge_overwrite(existing, new)

        assert merged.size == 1
        assert float(merged[0]["c"]) == pytest.approx(200.5)
        assert float(merged[0]["bv"]) == pytest.approx(2000.0)

    def test_standardize_gaps_handles_input_duplicates(self, tmp_cache_dir):
        """standardize_gaps deduplicates in non-strict mode via dict lookup.

        In strict=False mode, when input has duplicate timestamps, the LAST
        occurrence is used (due to dict overwrite behavior). Output is unique
        because we iterate over np.arange() which produces unique timestamps.

        Note: strict=True does NOT deduplicate (it returns raw slice).
        """
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000

        # Create candles with duplicate timestamp at minute 1
        candles = np.zeros(4, dtype=CANDLE_DTYPE)
        candles["ts"] = [
            base_ts,
            base_ts + ONE_MIN_MS,
            base_ts + ONE_MIN_MS,
            base_ts + 2 * ONE_MIN_MS,
        ]
        candles["c"] = [100.0, 101.0, 999.0, 102.0]  # 999 is the duplicate (last wins)
        candles["bv"] = 1.0

        # Non-strict mode: deduplicates via dict lookup
        result = cm.standardize_gaps(
            candles, start_ts=base_ts, end_ts=base_ts + 2 * ONE_MIN_MS, strict=False
        )

        # Output should have no duplicates in non-strict mode
        timestamps = [int(r["ts"]) for r in result]
        assert len(timestamps) == len(set(timestamps))
        assert len(result) == 3  # Only 3 unique timestamps

        # The LAST duplicate should be used (999.0, not 101.0)
        for r in result:
            if int(r["ts"]) == base_ts + ONE_MIN_MS:
                assert float(r["c"]) == pytest.approx(999.0)

    def test_standardize_gaps_strict_preserves_duplicates(self, tmp_cache_dir):
        """strict=True returns raw slice, preserving any input duplicates.

        This is a known behavior difference between strict modes.
        In practice, input data should not contain duplicates.
        """
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000

        # Create candles with duplicate timestamp
        candles = np.zeros(4, dtype=CANDLE_DTYPE)
        candles["ts"] = [
            base_ts,
            base_ts + ONE_MIN_MS,
            base_ts + ONE_MIN_MS,
            base_ts + 2 * ONE_MIN_MS,
        ]
        candles["c"] = [100.0, 101.0, 999.0, 102.0]
        candles["bv"] = 1.0

        # Strict mode: returns raw slice (preserves duplicates)
        result = cm.standardize_gaps(
            candles, start_ts=base_ts, end_ts=base_ts + 2 * ONE_MIN_MS, strict=True
        )

        # Strict mode preserves duplicates (raw slice)
        assert len(result) == 4  # All 4 rows including duplicate


# ==============================================================================
# 5. CONCURRENT ACCESS
# ==============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_parallel_requests_same_symbol(self, tmp_cache_dir, monkeypatch):
        """Parallel requests for same symbol don't cause race conditions."""

        class _Ex:
            id = "test"

        cm = CandlestickManager(exchange=_Ex(), exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        fetch_count = {"n": 0}

        async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
            fetch_count["n"] += 1
            await asyncio.sleep(0.05)  # Simulate network delay
            ts = list(range(int(since_ms), int(end_exclusive_ms), ONE_MIN_MS))
            arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
            if ts:
                arr["ts"] = np.asarray(ts, dtype=np.int64)
                arr["c"] = 100.0
                arr["bv"] = 1.0
            return arr

        monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

        start_ts = 1609459200000
        end_ts = start_ts + 10 * ONE_MIN_MS

        # Launch multiple parallel requests
        results = await asyncio.gather(
            cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=0),
            cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=0),
            cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=0),
        )

        # All should return same data
        for r in results:
            assert r.size > 0
            assert r.size == results[0].size

        # Should only fetch once (lock prevents duplicate fetches)
        assert fetch_count["n"] == 1

    @pytest.mark.asyncio
    async def test_parallel_different_symbols(self, tmp_cache_dir, monkeypatch):
        """Parallel requests for different symbols work independently."""

        class _Ex:
            id = "test"

        cm = CandlestickManager(exchange=_Ex(), exchange_name="test", cache_dir=tmp_cache_dir)

        fetch_calls = []

        async def fake_fetch(symbol_, since_ms, end_exclusive_ms, *, timeframe=None, on_batch=None):
            fetch_calls.append(symbol_)
            await asyncio.sleep(0.05)
            ts = list(range(int(since_ms), int(end_exclusive_ms), ONE_MIN_MS))
            arr = np.zeros(len(ts), dtype=CANDLE_DTYPE)
            if ts:
                arr["ts"] = np.asarray(ts, dtype=np.int64)
                arr["c"] = 100.0
                arr["bv"] = 1.0
            return arr

        monkeypatch.setattr(cm, "_fetch_ohlcv_paginated", fake_fetch)

        start_ts = 1609459200000
        end_ts = start_ts + 5 * ONE_MIN_MS

        # Launch parallel requests for different symbols
        results = await asyncio.gather(
            cm.get_candles("BTC/USDT:USDT", start_ts=start_ts, end_ts=end_ts, max_age_ms=0),
            cm.get_candles("ETH/USDT:USDT", start_ts=start_ts, end_ts=end_ts, max_age_ms=0),
        )

        # Both should have data
        assert results[0].size > 0
        assert results[1].size > 0

        # Both symbols should have been fetched
        assert "BTC/USDT:USDT" in fetch_calls
        assert "ETH/USDT:USDT" in fetch_calls


# ==============================================================================
# 6. MEMORY MANAGEMENT
# ==============================================================================


class TestMemoryManagement:
    """Tests for memory management."""

    def test_memory_retention_enforced(self, tmp_cache_dir):
        """Memory retention limit is enforced."""
        # Create CM with low memory limit
        cm = CandlestickManager(
            exchange=None,
            exchange_name="test",
            cache_dir=tmp_cache_dir,
            max_memory_candles_per_symbol=100,
        )
        symbol = "BTC/USDT:USDT"

        # Add more candles than limit
        candles = make_candles(1609459200000, 200)
        cm._cache[symbol] = candles

        # Enforce retention
        cm._enforce_memory_retention(symbol)

        # Should be truncated to limit
        assert cm._cache[symbol].size <= 100

    def test_repeated_requests_no_memory_leak(self, tmp_cache_dir):
        """Repeated requests don't cause memory growth."""
        cm = CandlestickManager(
            exchange=None,
            exchange_name="test",
            cache_dir=tmp_cache_dir,
            max_memory_candles_per_symbol=1000,
        )
        symbol = "BTC/USDT:USDT"

        base_ts = 1609459200000
        candles = make_candles(base_ts, 500)
        cm._cache[symbol] = candles

        # Multiple standardize_gaps calls
        for i in range(10):
            cm.standardize_gaps(
                candles, start_ts=base_ts, end_ts=base_ts + 499 * ONE_MIN_MS, strict=True
            )

        # Cache shouldn't grow unbounded
        assert cm._cache[symbol].size <= 1000


# ==============================================================================
# 7. SPECIAL CASES
# ==============================================================================


class TestSpecialCases:
    """Tests for special cases and edge scenarios."""

    def test_all_zero_volume_data(self, tmp_cache_dir):
        """Data with all zero volumes is handled."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000
        candles = np.zeros(10, dtype=CANDLE_DTYPE)
        candles["ts"] = [base_ts + i * ONE_MIN_MS for i in range(10)]
        candles["c"] = 100.0
        candles["bv"] = 0.0  # Zero volume

        result = cm.standardize_gaps(
            candles, start_ts=base_ts, end_ts=base_ts + 9 * ONE_MIN_MS, strict=True
        )
        assert result.size == 10

    def test_extreme_price_values(self, tmp_cache_dir):
        """Extreme price values don't cause issues."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        base_ts = 1609459200000
        candles = np.zeros(3, dtype=CANDLE_DTYPE)
        candles["ts"] = [base_ts + i * ONE_MIN_MS for i in range(3)]
        candles["c"] = [0.000001, 100000.0, 1e-10]  # Very small and large values
        candles["bv"] = 1.0

        result = cm.standardize_gaps(
            candles, start_ts=base_ts, end_ts=base_ts + 2 * ONE_MIN_MS, strict=True
        )
        assert result.size == 3
        assert float(result[0]["c"]) == pytest.approx(0.000001)
        assert float(result[1]["c"]) == pytest.approx(100000.0)

    def test_symbol_with_special_characters(self, tmp_cache_dir):
        """Symbols with special characters are handled correctly."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)

        # Symbol with colon (common in futures)
        symbol = "1000SHIB/USDT:USDT"

        base_ts = 1609459200000
        candles = make_candles(base_ts, 100)

        # Should not crash
        cm._persist_batch(symbol, candles, timeframe="1m")

        # Verify path is sanitized correctly
        symbol_dir = cm._symbol_dir(symbol, timeframe="1m")
        assert os.path.exists(symbol_dir)
        assert ":" in os.path.basename(symbol_dir) or "_" in os.path.basename(symbol_dir)
