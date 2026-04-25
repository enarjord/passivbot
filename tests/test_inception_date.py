"""Tests for inception date tracking in CandlestickManager."""

import numpy as np
import pytest
import os
import json
import tempfile
import shutil

from candlestick_manager import CandlestickManager, CANDLE_DTYPE, ONE_MIN_MS


@pytest.fixture
def tmp_cache_dir():
    """Create a temporary cache directory for tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_candles(timestamps: list[int]) -> np.ndarray:
    """Create a structured array of candles from timestamps."""
    n = len(timestamps)
    arr = np.zeros(n, dtype=CANDLE_DTYPE)
    arr["ts"] = timestamps
    arr["o"] = 100.0
    arr["h"] = 101.0
    arr["l"] = 99.0
    arr["c"] = 100.5
    arr["bv"] = 1000.0  # base volume
    return arr


def make_full_day(day_start_ts: int) -> np.ndarray:
    """Create a full UTC day of 1m candles starting at day_start_ts."""
    return make_candles([day_start_ts + i * ONE_MIN_MS for i in range(1440)])


class TestInceptionDateTracking:
    """Tests for inception date tracking."""

    def test_inception_ts_initially_none(self, tmp_cache_dir):
        """Inception timestamp should be None initially."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        inception = cm._get_inception_ts(symbol)
        assert inception is None

    def test_inception_ts_set_on_persist(self, tmp_cache_dir):
        """Inception timestamp should be set when data is persisted."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # Create candles starting at a known timestamp
        start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
        timestamps = [start_ts + i * ONE_MIN_MS for i in range(10)]
        candles = make_candles(timestamps)

        # Persist the batch
        cm._persist_batch(symbol, candles, timeframe="1m")

        # Check inception was set
        inception = cm._get_inception_ts(symbol)
        assert inception == start_ts

    def test_inception_ts_updates_to_earlier(self, tmp_cache_dir):
        """Inception timestamp should update if earlier data is found."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # First batch starting at 2021-02-01
        later_ts = 1612137600000  # 2021-02-01 00:00:00 UTC
        later_candles = make_candles([later_ts + i * ONE_MIN_MS for i in range(5)])
        cm._persist_batch(symbol, later_candles, timeframe="1m")

        assert cm._get_inception_ts(symbol) == later_ts

        # Second batch starting at 2021-01-01 (earlier)
        earlier_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
        earlier_candles = make_candles([earlier_ts + i * ONE_MIN_MS for i in range(5)])
        cm._persist_batch(symbol, earlier_candles, timeframe="1m")

        # Inception should now be the earlier date
        assert cm._get_inception_ts(symbol) == earlier_ts

    def test_inception_ts_not_update_to_later(self, tmp_cache_dir):
        """Inception timestamp should NOT update if later data is found."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        # First batch starting at 2021-01-01
        earlier_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
        earlier_candles = make_candles([earlier_ts + i * ONE_MIN_MS for i in range(5)])
        cm._persist_batch(symbol, earlier_candles, timeframe="1m")

        assert cm._get_inception_ts(symbol) == earlier_ts

        # Second batch starting at 2021-02-01 (later)
        later_ts = 1612137600000  # 2021-02-01 00:00:00 UTC
        later_candles = make_candles([later_ts + i * ONE_MIN_MS for i in range(5)])
        cm._persist_batch(symbol, later_candles, timeframe="1m")

        # Inception should still be the earlier date
        assert cm._get_inception_ts(symbol) == earlier_ts

    def test_inception_ts_persisted_to_index(self, tmp_cache_dir):
        """Inception timestamp should be persisted to index.json."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
        candles = make_candles([start_ts + i * ONE_MIN_MS for i in range(10)])
        cm._persist_batch(symbol, candles, timeframe="1m")

        # Check it was written to disk
        idx_path = cm._index_path(symbol, tf="1m")
        with open(idx_path, "r") as f:
            idx = json.load(f)

        assert idx["meta"]["inception_ts"] == start_ts

    def test_inception_ts_loaded_on_restart(self, tmp_cache_dir):
        """Inception timestamp should be loaded from index.json on restart."""
        symbol = "BTC/USDT:USDT"
        start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC

        # First CM instance persists data
        cm1 = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        candles = make_candles([start_ts + i * ONE_MIN_MS for i in range(10)])
        cm1._persist_batch(symbol, candles, timeframe="1m")
        assert cm1._get_inception_ts(symbol) == start_ts

        # Second CM instance should load it from disk
        cm2 = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        assert cm2._get_inception_ts(symbol) == start_ts

    def test_add_known_gap_with_explicit_retry_count(self, tmp_cache_dir):
        """Test that _add_known_gap can set explicit retry_count."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        start_ts = 1609459200000
        end_ts = 1609545600000

        # Add gap with explicit retry_count (used for pre-inception)
        cm._add_known_gap(symbol, start_ts, end_ts, reason="pre_inception", retry_count=3)

        gaps = cm._get_known_gaps_enhanced(symbol)
        assert len(gaps) == 1
        assert gaps[0]["retry_count"] == 3
        assert gaps[0]["reason"] == "pre_inception"

    def test_pre_inception_gap_is_persistent(self, tmp_cache_dir):
        """Gaps marked with max retry count should be persistent."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"

        start_ts = 1609459200000
        end_ts = 1609545600000

        # Add gap with max retry_count (pre-inception pattern)
        cm._add_known_gap(symbol, start_ts, end_ts, reason="pre_inception", retry_count=3)

        gaps = cm._get_known_gaps_enhanced(symbol)
        assert len(gaps) == 1

        # This gap should not be retried
        assert not cm._should_retry_gap(gaps[0])

    def test_legacy_inception_with_matching_pre_inception_gap_preserves_authoritative_bound(
        self, tmp_cache_dir
    ):
        """Legacy caches that had already learned a pre-inception boundary must preserve it."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"
        legacy_ts = 1609459200000

        idx_path = cm._index_path(symbol, tf="1m")
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "shards": {},
                    "meta": {
                        "inception_ts": legacy_ts,
                        "known_gaps": [
                            {
                                "start_ts": legacy_ts - 10 * ONE_MIN_MS,
                                "end_ts": legacy_ts - ONE_MIN_MS,
                                "retry_count": 3,
                                "reason": "pre_inception",
                                "added_at": legacy_ts,
                            }
                        ],
                    },
                },
                f,
            )

        cm2 = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        idx = cm2._ensure_symbol_index(symbol, tf="1m")

        assert cm2._get_inception_ts(symbol) == legacy_ts
        assert idx["meta"]["observed_start_ts"] == legacy_ts
        assert idx["meta"]["authoritative_start_ts"] == legacy_ts
        assert idx["meta"]["authoritative_start_source"] == "legacy_pre_inception_gap"
        assert len(idx["meta"]["known_gaps"]) == 1

    def test_stale_legacy_pre_inception_gap_is_dropped_without_authoritative_bound(
        self, tmp_cache_dir
    ):
        """Non-matching legacy pre_inception gaps should not survive migration."""
        cm = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        symbol = "BTC/USDT:USDT"
        legacy_ts = 1609459200000

        idx_path = cm._index_path(symbol, tf="1m")
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "shards": {},
                    "meta": {
                        "inception_ts": legacy_ts,
                        "known_gaps": [
                            {
                                "start_ts": legacy_ts - 20 * ONE_MIN_MS,
                                "end_ts": legacy_ts - 11 * ONE_MIN_MS,
                                "retry_count": 3,
                                "reason": "pre_inception",
                                "added_at": legacy_ts,
                            }
                        ],
                    },
                },
                f,
            )

        cm2 = CandlestickManager(exchange=None, exchange_name="test", cache_dir=tmp_cache_dir)
        idx = cm2._ensure_symbol_index(symbol, tf="1m")

        assert cm2._get_inception_ts(symbol) == legacy_ts
        assert idx["meta"]["observed_start_ts"] == legacy_ts
        assert idx["meta"]["authoritative_start_ts"] is None
        assert idx["meta"]["known_gaps"] == []

    @pytest.mark.asyncio
    async def test_prefetch_backfills_earlier_days_when_only_observed_start_is_known(
        self, tmp_cache_dir
    ):
        """Earlier backfills must not be blocked just because later shards already exist."""
        cm = CandlestickManager(exchange=None, exchange_name="binance", cache_dir=tmp_cache_dir)
        symbol = "ETH/USDT:USDT"
        day1 = 1672531200000  # 2023-01-01
        day2 = day1 + 24 * 60 * 60 * 1000
        day3 = day2 + 24 * 60 * 60 * 1000
        day2_end = day2 + 1439 * ONE_MIN_MS

        cm._persist_batch(symbol, make_full_day(day3), timeframe="1m")
        assert cm._get_inception_ts(symbol) == day3
        assert cm._get_authoritative_start_ts(symbol) is None

        calls = []

        async def fake_archive_fetch_day(_symbol: str, day_key: str):
            calls.append(day_key)
            start_ts, _ = cm._date_range_of_key(day_key)
            return make_full_day(start_ts)

        cm._archive_supported = lambda: True
        cm._archive_fetch_day = fake_archive_fetch_day

        await cm._prefetch_archives_for_range(symbol, day1, day2_end, parallel_days=2)

        shard_days = sorted(cm._ensure_symbol_index(symbol, tf="1m")["shards"].keys())
        assert calls == ["2023-01-01", "2023-01-02"]
        assert shard_days[:3] == ["2023-01-01", "2023-01-02", "2023-01-03"]
        assert cm._get_inception_ts(symbol) == day1
        assert cm._get_authoritative_start_ts(symbol) is None

    @pytest.mark.asyncio
    async def test_prefetch_uses_exchange_specific_cache_as_authoritative_lower_bound(
        self, tmp_cache_dir
    ):
        """Exchange-specific first-timestamp cache may clip earlier ranges authoritatively."""
        cm = CandlestickManager(exchange=None, exchange_name="binance", cache_dir=tmp_cache_dir)
        symbol = "ETH/USDT:USDT"
        authoritative_ts = 1672617600000  # 2023-01-02
        day1 = 1672531200000  # 2023-01-01
        day2 = authoritative_ts
        day2_end = day2 + 1439 * ONE_MIN_MS

        cache_path = os.path.join(tmp_cache_dir, "first_ohlcv_timestamps_unified_exchange_specific.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"ETH": {"binanceusdm": authoritative_ts}}, f)

        calls = []

        async def fake_archive_fetch_day(_symbol: str, day_key: str):
            calls.append(day_key)
            start_ts, _ = cm._date_range_of_key(day_key)
            return make_full_day(start_ts)

        cm._archive_supported = lambda: True
        cm._archive_fetch_day = fake_archive_fetch_day

        await cm._prefetch_archives_for_range(symbol, day1, day2_end, parallel_days=2)

        idx = cm._ensure_symbol_index(symbol, tf="1m")
        gaps = cm._get_known_gaps_enhanced(symbol)
        assert calls == ["2023-01-02"]
        assert idx["meta"]["authoritative_start_ts"] == authoritative_ts
        assert idx["meta"]["authoritative_start_source"] == "exchange_specific_cache"
        assert any(
            g["start_ts"] == day1 and g["end_ts"] == day2 - ONE_MIN_MS and g["reason"] == "pre_inception"
            for g in gaps
        )
