"""Recent gap decisions limit repeated index and historical metadata work."""
import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest

from candlestick_manager import (
    CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS, _GAP_MAX_RETRIES,
    _GAP_PERSISTENT_RETRY_MS, _KnownGapIndex,
)

NOW = 1_800_000_000_000
SYMBOL = 'TEST/USDT:USDT'


def gaps():
    records = [dict(start_ts=i * 2 * ONE_MIN_MS, end_ts=i * 2 * ONE_MIN_MS,
                    reason='no_trades', retry_count=_GAP_MAX_RETRIES,
                    added_at=1, last_retry_at=1) for i in range(5000)]
    records.append(dict(start_ts=NOW - 5 * ONE_MIN_MS, end_ts=NOW - 3 * ONE_MIN_MS,
                        reason='fetch_failed', retry_count=_GAP_MAX_RETRIES,
                        added_at=NOW, last_retry_at=NOW))
    return records


def manager(path):
    cm = CandlestickManager(exchange=None, exchange_name='fixture', cache_dir=str(path))
    cm._now_ms_callback = lambda: NOW
    cm._save_known_gaps_enhanced(SYMBOL, gaps())
    return cm


@pytest.mark.asyncio
async def test_batched_ema_checks_one_fresh_index_for_all_spans(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path))
    other = CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path))
    cm._now_ms_callback = other._now_ms_callback = lambda: NOW
    end = NOW - ONE_MIN_MS
    hole = end - 3 * ONE_MIN_MS
    rows = np.array(
        [(end - i * ONE_MIN_MS, 10, 11, 9, 10, 1) for i in range(9, -1, -1) if i != 3],
        dtype=CANDLE_DTYPE,
    )
    proof = dict(start_ts=hole, end_ts=hole, reason="no_trades", retry_count=0)
    cm._persist_batch(SYMBOL, rows, timeframe="1m", merge_cache=True, last_refresh_ms=NOW)
    day_key = cm._date_key(end)
    path = Path(cm._shard_path(SYMBOL, day_key, tf="1m"))
    events = []
    original = cm._ensure_symbol_index

    def validated(*args, **kwargs):
        index = original(*args, **kwargs)
        events.append(("index", day_key in index["shards"]))
        return index

    async def loaded(*args, **kwargs):
        events.append("load")
        await asyncio.sleep(0)
        if len(events) == 1:
            # Change the gap proof while this candle read suspends. The
            # post-load validation must see the published index.
            other._save_known_gaps_enhanced(SYMBOL, [proof])
        else:
            other._save_known_gaps_enhanced(SYMBOL, [])
            # No index write follows this deletion. The next access must prune.
            path.unlink()
        events.append("loaded")
        return rows

    monkeypatch.setattr(cm, "_ensure_symbol_index", validated)
    monkeypatch.setattr(cm, "get_candles", loaded)
    spans = {"close": [1.5, 5.0], "qv": [6.5]}

    try:
        accepted = await cm.get_latest_ema_metric_spans(SYMBOL, spans)
        assert events == ["load", "loaded", ("index", True)]
        assert all(math.isfinite(value) for metrics in accepted.values() for value in metrics.values())

        rejected = await cm.get_latest_ema_metric_spans(SYMBOL, spans)
        assert events == ["load", "loaded", ("index", True), "load", "loaded", ("index", False)]
        assert rejected["close"][1.5] == accepted["close"][1.5]
        assert math.isnan(rejected["close"][5.0])
        assert math.isnan(rejected["qv"][6.5])
    finally:
        other.close()
        cm.close()


def test_candle_gap_queries_share_index_and_preserve_full_proof_bounds(tmp_path, monkeypatch):
    cm = CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path))
    hole_start, query_start, hole_end = 4 * ONE_MIN_MS, 6 * ONE_MIN_MS, 7 * ONE_MIN_MS
    cm._cache[SYMBOL] = np.zeros(2, dtype=CANDLE_DTYPE)
    cm._cache[SYMBOL]["ts"] = [hole_start - ONE_MIN_MS, hole_end + ONE_MIN_MS]
    # Give the fragments distinct retry metadata so they remain separate.
    # The physical-hole query also needs proof before the requested window.
    cm._save_known_gaps_enhanced(SYMBOL, [
        dict(start_ts=hole_start, end_ts=query_start - ONE_MIN_MS, reason="no_trades", added_at=1),
        dict(start_ts=query_start, end_ts=hole_end, reason="no_trades", added_at=2),
    ])
    calls = []
    original = cm._ensure_symbol_index

    def validated(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(cm, "_ensure_symbol_index", validated)
    try:
        assert cm._unverified_candle_gap_ranges(SYMBOL, query_start, hole_end) == []
        assert calls == [(SYMBOL,)]
    finally:
        cm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("exchange", "timeframe", "step"), [
    ("fixture", "1h", 60 * ONE_MIN_MS), ("fake", "1m", ONE_MIN_MS),
])
async def test_batched_ema_coverage_skips_unneeded_index(tmp_path, monkeypatch, exchange, timeframe, step):
    cm = CandlestickManager(exchange_name=exchange, cache_dir=str(tmp_path))
    cm._now_ms_callback = lambda: NOW
    end = NOW // step * step - step
    rows = np.array([(end - i * step, 10, 11, 9, 10, 1) for i in range(7, -1, -1)], dtype=CANDLE_DTYPE)

    async def loaded(*args, **kwargs):
        return rows

    def unexpected_index(*args, **kwargs):
        pytest.fail("EMA coverage does not use the one-minute gap index here")

    monkeypatch.setattr(cm, "get_candles", loaded)
    monkeypatch.setattr(cm, "_ensure_symbol_index", unexpected_index)
    try:
        outcomes = await cm.get_latest_ema_metric_spans(SYMBOL, {"close": [1.5, 3.5]}, timeframe=timeframe)
        assert outcomes["close"] == pytest.approx({1.5: 10.0, 3.5: 10.0})
    finally:
        cm.close()


@pytest.mark.parametrize('operation', ['prefix', 'ranges', 'trim_disjoint'])
def test_recent_queries_normalize_only_overlapping_records(tmp_path, monkeypatch, operation):
    cm = manager(tmp_path)
    start, end = NOW - 4 * ONE_MIN_MS, NOW - ONE_MIN_MS
    normalized_sizes = []
    original = cm._normalize_gap_entries

    def observed(records):
        normalized_sizes.append(len(records))
        return original(records)

    before = open(cm._index_path(SYMBOL), 'rb').read()
    reference = _KnownGapIndex(cm._get_known_gaps_enhanced(SYMBOL))
    monkeypatch.setattr(cm, '_normalize_gap_entries', observed)
    try:
        if operation == 'prefix':
            expected = cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end, now_ms=NOW, gap_index=reference)
            assert cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end, now_ms=NOW) == expected == NOW - 2 * ONE_MIN_MS
        elif operation == 'ranges':
            assert cm._fetch_ranges_excluding_deferred_gaps(SYMBOL, start, end, now_ms=NOW) == [(NOW - 2 * ONE_MIN_MS, end)]
        else:
            rows = np.zeros(1, dtype=CANDLE_DTYPE)
            rows['ts'] = NOW
            assert cm._trim_known_gaps_covered_by_rows(SYMBOL, rows) is False
        assert normalized_sizes and max(normalized_sizes) <= 1
        assert open(cm._index_path(SYMBOL), 'rb').read() == before
    finally:
        cm.close()


def test_gap_query_mutation_alias_and_external_replacement_stay_visible(tmp_path):
    cm = manager(tmp_path)
    start, end = NOW - 4 * ONE_MIN_MS, NOW - ONE_MIN_MS
    try:
        assert cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end) == NOW - 2 * ONE_MIN_MS
        records = cm._ensure_symbol_index(SYMBOL)['meta']['known_gaps']
        records[-1]['retry_count'] = 0
        assert cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end) == start
        # Unowned aliases cannot use identity as a proof of unchanged content.
        plain = [dict(row) for row in records]
        cm._ensure_symbol_index(SYMBOL)['meta']['known_gaps'] = plain
        plain[-1]['retry_count'] = _GAP_MAX_RETRIES
        assert cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end) == NOW - 2 * ONE_MIN_MS
        other = CandlestickManager(exchange=None, exchange_name='fixture', cache_dir=str(tmp_path))
        other._now_ms_callback = lambda: NOW
        try:
            other._save_known_gaps_enhanced(SYMBOL, [dict(gaps()[-1], reason='no_trades')])
            assert cm._fetch_ranges_excluding_deferred_gaps(SYMBOL, start, end) == [(NOW - 2 * ONE_MIN_MS, end)]
            other._save_known_gaps_enhanced(SYMBOL, [])
            assert cm._fetch_start_after_deferred_gap_prefix(SYMBOL, start, end) == start
        finally:
            other.close()
    finally:
        cm.close()


def test_trim_rewrite_and_health_match_full_scan_with_proof_epochs(tmp_path, monkeypatch):
    optimized, reference = manager(tmp_path / 'optimized'), manager(tmp_path / 'reference')
    monkeypatch.setattr(reference, '_gap_records_in_range', lambda index, *args: index['meta']['known_gaps'])
    rows = np.zeros(2, dtype=CANDLE_DTYPE)
    rows['ts'] = [NOW - 4 * ONE_MIN_MS, NOW - 3 * ONE_MIN_MS]
    rows['o'] = rows['h'] = rows['l'] = rows['c'] = 10
    rows['bv'] = 2
    try:
        for cm in (optimized, reference):
            assert cm._trim_known_gaps_covered_by_rows(SYMBOL, rows)
        a = json.load(open(optimized._index_path(SYMBOL)))
        b = json.load(open(reference._index_path(SYMBOL)))
        assert a == b
        assert optimized.get_completed_candle_health(SYMBOL, {'1m': 6}, now_ms=NOW) == reference.get_completed_candle_health(SYMBOL, {'1m': 6}, now_ms=NOW)
        # An untouched unknown prefix keeps its original retry epoch/reason;
        # real rows remove only timestamps actually supplied by the source.
        retained = optimized._get_known_gaps_enhanced(SYMBOL)[-1]
        assert retained['start_ts'] == retained['end_ts'] == NOW - 5 * ONE_MIN_MS
        assert retained['reason'] == 'fetch_failed' and retained['last_retry_at'] == NOW
    finally:
        optimized.close()
        reference.close()


@pytest.mark.parametrize("clock_offset_ms,persistent", [(0, 2), (1, 1)])
def test_candle_health_counts_gaps_without_detail_retry_pass(
    tmp_path, monkeypatch, clock_offset_ms, persistent
):
    cm = CandlestickManager(exchange=None, exchange_name="kucoin", cache_dir=str(tmp_path))
    cm._now_ms_callback = lambda: NOW + clock_offset_ms
    old = NOW - 20 * ONE_MIN_MS
    retry_boundary = NOW - _GAP_PERSISTENT_RETRY_MS
    records = [
        dict(start_ts=old + i * 2 * ONE_MIN_MS, end_ts=old + i * 2 * ONE_MIN_MS,
             reason=reason, retry_count=count, added_at=retry, last_retry_at=retry)
        for i, (reason, count, retry) in enumerate([
            ("no_trades", _GAP_MAX_RETRIES, retry_boundary),
            ("fetch_failed", _GAP_MAX_RETRIES, retry_boundary + 1),
            ("fetch_failed", _GAP_MAX_RETRIES, retry_boundary),
            ("auto_detected", _GAP_MAX_RETRIES - 1, NOW),
        ])
    ]
    rows = np.zeros(3, dtype=CANDLE_DTYPE)
    rows["ts"] = [NOW - i * ONE_MIN_MS for i in (3, 2, 1)]
    rows["o"] = rows["h"] = rows["l"] = rows["c"] = 10
    cm._cache[SYMBOL] = rows
    try:
        cm._save_known_gaps_enhanced(SYMBOL, records)
        details = cm.get_gap_summary(SYMBOL)
        assert [gap["persistent"] for gap in details["gaps"]] == [
            True, clock_offset_ms == 0, False, False,
        ]
        retry_checks = []
        original = cm._should_retry_gap

        def counted(gap, **kwargs):
            retry_checks.append(gap["start_ts"])
            return original(gap, **kwargs)

        monkeypatch.setattr(cm, "_should_retry_gap", counted)
        report = cm.get_completed_candle_health(SYMBOL, {"1m": 3}, now_ms=NOW)
        health = report["timeframes"]["1m"]
        assert health["coverage_ok"]
        assert health["known_gaps_total"] == details["total_gaps"] == 4
        assert health["known_gaps_minutes"] == details["total_minutes"] == 4
        assert health["known_gaps_persistent"] == details["persistent_gaps"] == persistent
        assert health["known_gaps_retryable"] == details["retryable_gaps"] == 4 - persistent
        assert health["known_gaps_by_reason"] == details["by_reason"]
        assert retry_checks == [gap["start_ts"] for gap in records[:3]]
    finally:
        cm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provisional", [False, True])
async def test_candle_gap_masks_skip_unused_pass_and_recheck_external_proof(
    tmp_path, monkeypatch, provisional
):
    cm = CandlestickManager(
        exchange=None, exchange_name="fixture", cache_dir=str(tmp_path),
        provisional_internal_gap_tolerance_minutes=2,
    )
    other = CandlestickManager(
        exchange=None, exchange_name="fixture", cache_dir=str(tmp_path)
    )
    cm._now_ms_callback = other._now_ms_callback = lambda: NOW
    start = NOW - 16 * ONE_MIN_MS
    absent = {3, 6, 7, 8, 11, 15}
    source = np.array(
        [
            (start + i * ONE_MIN_MS, 100 + i, 100 + i, 100 + i, 100 + i, 1.0)
            for i in range(16) if i not in absent
        ],
        dtype=CANDLE_DTYPE,
    )
    gaps = [
        dict(
            start_ts=start + lo * ONE_MIN_MS, end_ts=start + hi * ONE_MIN_MS,
            reason=reason, retry_count=_GAP_MAX_RETRIES, added_at=NOW, last_retry_at=NOW,
        )
        for lo, hi, reason in (
            (3, 3, "fetch_failed"), (6, 8, "fetch_failed"), (11, 11, "no_trades"),
        )
    ]
    calls = []
    summarize = cm._unverified_candle_gap_ranges

    def observe(*args):
        calls.append(args)
        return summarize(*args)

    monkeypatch.setattr(cm, "_unverified_candle_gap_ranges", observe)

    async def read():
        return await cm.get_candles(
            SYMBOL, start_ts=start, end_ts=NOW, allow_remote_fetch=False,
            allow_provisional_internal_gaps=provisional, fill_trailing_gaps=False,
        )

    def assert_rows(actual, synthesized):
        expected = []
        previous_close = None
        for i in range(16):
            if i in synthesized:
                expected.append((start + i * ONE_MIN_MS, *([previous_close] * 4), 0.0))
            elif i not in absent:
                previous_close = 100 + i
                expected.append((start + i * ONE_MIN_MS, *([previous_close] * 4), 1.0))
        np.testing.assert_array_equal(actual, np.array(expected, dtype=CANDLE_DTYPE))

    try:
        cm._persist_batch(
            SYMBOL, source, timeframe="1m", merge_cache=True, last_refresh_ms=NOW - 1
        )
        cm._save_known_gaps_enhanced(SYMBOL, gaps)
        synthesized = {3, 11} if provisional else {11}
        assert_rows(await read(), synthesized)

        # Another writer can certify the whole outage. This read must observe
        # the replaced index, even when the discarded mask is never built.
        other._save_known_gaps_enhanced(
            SYMBOL, [gaps[0], dict(gaps[1], reason="no_trades"), gaps[2]]
        )
        synthesized.update({6, 7, 8})
        assert_rows(await read(), synthesized)
        assert calls == ([] if provisional else [(SYMBOL, start, NOW - ONE_MIN_MS)] * 2)
        assert cm.get_last_refresh_ms(SYMBOL) == NOW - 1
        assert cm._synthetic_timestamps[SYMBOL] == {
            start + i * ONE_MIN_MS for i in synthesized
        }
        np.testing.assert_array_equal(cm._cache[SYMBOL], source)
    finally:
        other.close()
        cm.close()


@pytest.mark.asyncio
async def test_real_historical_read_uses_bounded_gap_cohort_without_network(tmp_path, monkeypatch):
    cm = manager(tmp_path)
    class NoRequests:
        id = 'fixture'
        async def fetch_ohlcv(self, *args, **kwargs):
            pytest.fail('deferred gap must not become a remote request')
    cm.exchange = NoRequests()
    rows = np.zeros(3, dtype=CANDLE_DTYPE)
    rows['ts'] = [NOW - 6 * ONE_MIN_MS, NOW - 2 * ONE_MIN_MS, NOW - ONE_MIN_MS]
    rows['o'] = rows['h'] = rows['l'] = rows['c'] = 10
    rows['bv'] = 2
    cm._cache[SYMBOL] = rows
    counts = []
    original = cm._normalize_gap_entries
    def observed(records):
        counts.append(len(records))
        return original(records)
    monkeypatch.setattr(cm, '_normalize_gap_entries', observed)
    try:
        actual = await cm.get_candles(SYMBOL, start_ts=int(rows['ts'][0]),
            end_ts=NOW - 2 * ONE_MIN_MS, allow_remote_fetch=True, standardize=False)
        np.testing.assert_array_equal(actual, rows[:2])
        assert counts and max(counts) <= 1
    finally:
        cm.close()
