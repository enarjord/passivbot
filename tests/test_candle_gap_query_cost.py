"""Recent gap decisions must not normalize unrelated historical metadata."""
import json

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
    monkeypatch.setattr(reference, '_get_known_gaps_in_range', lambda symbol, *args: reference._get_known_gaps_enhanced(symbol))
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
