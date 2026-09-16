"""Coverage queries must preserve independent retry records without stranding gaps."""

import numpy as np
import pytest

from candlestick_manager import (
    CANDLE_DTYPE,
    CandlestickManager,
    GAP_REASON_FETCH_FAILED,
    GAP_REASON_NO_TRADES,
    ONE_MIN_MS,
    _GAP_MAX_RETRIES,
    _GAP_PERSISTENT_RETRY_MS,
)


def seed_fragmented_gap(cm, symbol, end_minute, now):
    rows = np.array(
        [
            (m * ONE_MIN_MS, 100, 101, 99, 100, 5)
            for m in range(end_minute + 1)
            if m not in (11, 12)
        ],
        dtype=CANDLE_DTYPE,
    )
    cm._cache[symbol] = rows
    gaps = [
        dict(
            start_ts=m * ONE_MIN_MS,
            end_ts=m * ONE_MIN_MS,
            reason=GAP_REASON_FETCH_FAILED,
            retry_count=_GAP_MAX_RETRIES,
            added_at=now - m * 1000,
            last_retry_at=now - m * 1000,
            last_contextual_retry_at=0,
        )
        for m in (11, 12)
    ]
    cm._save_known_gaps_enhanced(symbol, gaps)
    return rows, gaps


@pytest.mark.asyncio
@pytest.mark.parametrize("end_minute", [24, 4000])
async def test_kucoin_contextual_proof_spans_adjacent_retry_records(tmp_path, end_minute):
    now = (end_minute + 1) * ONE_MIN_MS + 1000
    calls = []

    class Exchange:
        id = "kucoinfutures"

        async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            calls.append((since, limit))
            return [
                list(row) for row in rows
                if since < row[0] <= since + limit * ONE_MIN_MS
            ]

    cm = CandlestickManager(exchange=Exchange(), cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    rows, _ = seed_fragmented_gap(cm, symbol, end_minute, now)

    result = await cm.get_candles(symbol, start_ts=0, end_ts=end_minute * ONE_MIN_MS)

    assert calls == [(9 * ONE_MIN_MS, 5)]
    assert list(result["ts"]) == list(range(0, (end_minute + 1) * ONE_MIN_MS, ONE_MIN_MS))
    gaps = cm._get_known_gaps_enhanced(symbol)
    assert all(gap["reason"] == GAP_REASON_NO_TRADES for gap in gaps)
    assert not np.isin([11 * ONE_MIN_MS, 12 * ONE_MIN_MS], cm._cache[symbol]["ts"]).any()


@pytest.mark.asyncio
@pytest.mark.parametrize("historical", [False, True])
async def test_adjacent_deferred_records_do_not_refetch_complete_history(tmp_path, historical):
    end_minute = 4000
    now = (end_minute + (20 if historical else 1)) * ONE_MIN_MS + 1000
    calls = []

    class Exchange:
        id = "dummy"

        async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            calls.append((since, limit))
            return []

    cm = CandlestickManager(exchange=Exchange(), cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    rows, gaps = seed_fragmented_gap(cm, symbol, end_minute, now)

    result = await cm.get_candles(symbol, start_ts=0, end_ts=end_minute * ONE_MIN_MS)

    assert not calls
    assert list(result["ts"]) == list(rows["ts"])
    assert cm._get_known_gaps_enhanced(symbol) == gaps


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_part", ["first", "last", "verified", "unknown", "terminal"])
async def test_contextual_proof_preserves_fragment_evidence_and_cooldowns(tmp_path, blocked_part):
    now = 25 * ONE_MIN_MS + 1000
    calls = []

    class Exchange:
        id = "kucoinfutures"

        async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            calls.append((since, limit))
            return [list(row) for row in rows if since < row[0] <= since + limit * ONE_MIN_MS]

    cm = CandlestickManager(exchange=Exchange(), cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    rows, gaps = seed_fragmented_gap(cm, symbol, 24, now)
    if blocked_part in ("first", "last"):
        gaps[0 if blocked_part == "first" else 1]["last_contextual_retry_at"] = now
    elif blocked_part == "verified":
        gaps[0]["reason"] = GAP_REASON_NO_TRADES
    elif blocked_part == "unknown":
        gaps.pop(0)
    else:
        gaps[0]["reason"] = "no_archive"
    cm._save_known_gaps_enhanced(symbol, gaps)

    result = await cm.get_candles(symbol, start_ts=0, end_ts=24 * ONE_MIN_MS)

    if blocked_part == "verified":
        assert calls == [(9 * ONE_MIN_MS, 5)]
        assert result.size == 25
    elif blocked_part in ("first", "last", "terminal"):
        assert not calls
        assert 12 * ONE_MIN_MS not in result["ts"]
        assert cm._get_known_gaps_enhanced(symbol) == gaps
    else:
        # Unknown timestamps follow ordinary repair; no contextual request may
        # certify the neighbouring deferred fragment without complete evidence.
        assert (9 * ONE_MIN_MS, 5) not in calls
        assert any(
            g["reason"] == GAP_REASON_FETCH_FAILED
            for g in cm._get_known_gaps_enhanced(symbol)
        )


@pytest.mark.asyncio
async def test_failed_contextual_proof_retries_after_bounded_delay(tmp_path):
    clock = {"now": 25 * ONE_MIN_MS + 1000}
    calls = []

    class Exchange:
        id = "kucoinfutures"

        async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            calls.append((since, limit))
            if len(calls) == 1:
                return []
            return [list(row) for row in rows if since < row[0] <= since + limit * ONE_MIN_MS]

    cm = CandlestickManager(exchange=Exchange(), cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms_callback = lambda: clock["now"]
    symbol = "SPARSE/USDT:USDT"
    rows, _ = seed_fragmented_gap(cm, symbol, 24, clock["now"])
    await cm.get_candles(symbol, start_ts=0, end_ts=24 * ONE_MIN_MS)
    assert len(calls) == 1
    gap = cm._get_known_gaps_enhanced(symbol)[0]
    assert not cm._kucoin_contextual_retry_due(gap, now_ms=clock["now"] + 5 * ONE_MIN_MS - 1)
    assert cm._kucoin_contextual_retry_due(gap, now_ms=clock["now"] + 5 * ONE_MIN_MS)
    # The ordinary missing-range cooldown remains independent and much longer.
    assert not cm._should_retry_gap(gap, now_ms=clock["now"] + 5 * ONE_MIN_MS)
    assert cm._should_retry_gap(gap, now_ms=clock["now"] + _GAP_PERSISTENT_RETRY_MS)
    clock["now"] += 5 * ONE_MIN_MS
    rows = np.concatenate(
        [
            rows,
            np.array(
                [(m * ONE_MIN_MS, 100, 101, 99, 100, 5) for m in range(25, 30)],
                dtype=CANDLE_DTYPE,
            ),
        ]
    )
    cm._cache[symbol] = rows
    recovered = await cm.get_candles(symbol, start_ts=0, end_ts=29 * ONE_MIN_MS)
    assert calls == [(9 * ONE_MIN_MS, 5), (9 * ONE_MIN_MS, 5)]
    assert recovered.size == 30
    assert all(g["reason"] == GAP_REASON_NO_TRADES for g in cm._get_known_gaps_enhanced(symbol))


def test_many_deferred_fragments_use_one_metadata_snapshot(tmp_path, monkeypatch):
    cm = CandlestickManager(cache_dir=str(tmp_path), archive_enabled=False)
    now = 2000 * ONE_MIN_MS
    gaps = [
        dict(start_ts=m * ONE_MIN_MS, end_ts=m * ONE_MIN_MS,
             reason=GAP_REASON_FETCH_FAILED, retry_count=_GAP_MAX_RETRIES,
             added_at=now, last_retry_at=now)
        for m in range(1000)
    ]
    reads = []

    def load_gaps(symbol):
        reads.append(symbol)
        return gaps

    monkeypatch.setattr(cm, "_get_known_gaps_enhanced", load_gaps)
    assert cm._known_gap_retry_deferred_at(
        "SPARSE/USDT:USDT", 0, 999 * ONE_MIN_MS, now_ms=now
    )
    assert len(reads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("gap_minutes", [197, 198])
async def test_contextual_proof_requires_boundaries_within_one_page(tmp_path, gap_minutes):
    end_minute = gap_minutes + 14
    now = (end_minute + 1) * ONE_MIN_MS + 1000
    calls = []
    rows = np.array(
        [(m * ONE_MIN_MS, 100, 101, 99, 100, 5)
         for m in range(end_minute + 1)
         if not 11 <= m <= 10 + gap_minutes], dtype=CANDLE_DTYPE
    )

    class Exchange:
        id = "kucoinfutures"

        async def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None, params=None):
            calls.append((since, limit))
            return [list(row) for row in rows if since < row[0] <= since + limit * ONE_MIN_MS]

    cm = CandlestickManager(exchange=Exchange(), cache_dir=str(tmp_path), archive_enabled=False)
    cm._now_ms_callback = lambda: now
    symbol = "SPARSE/USDT:USDT"
    cm._cache[symbol] = rows
    cm._save_known_gaps_enhanced(symbol, [
        dict(start_ts=m * ONE_MIN_MS, end_ts=m * ONE_MIN_MS,
             reason=GAP_REASON_FETCH_FAILED, retry_count=_GAP_MAX_RETRIES,
             added_at=now, last_retry_at=now, last_contextual_retry_at=0)
        for m in range(11, 11 + gap_minutes)
    ])
    result = await cm.get_candles(symbol, start_ts=0, end_ts=end_minute * ONE_MIN_MS)
    if gap_minutes == 197:
        assert calls == [(9 * ONE_MIN_MS, 200)]
        assert result.size == end_minute + 1
    else:
        assert calls == []
        assert list(result["ts"]) == list(rows["ts"])
        assert all(g["reason"] == GAP_REASON_FETCH_FAILED
                   for g in cm._get_known_gaps_enhanced(symbol))
