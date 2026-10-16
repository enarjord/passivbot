"""Differential tests against the frozen pre-contiguous-index gap algorithm.

All candles, clocks and metadata are synthetic. No exchange calls or live files.
"""
import copy
import random
from typing import List, Tuple

import numpy as np
import pytest

import candlestick_manager as module
from candlestick_manager import (CandlestickManager, CANDLE_DTYPE, ONE_MIN_MS,
    GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED, _sorted_candle_copy)
from test_gap_metadata_performance_parity import manager, install, SYMBOL, NOW


class StridedSearchReference(CandlestickManager):
    """Freeze the exact immediately preceding algorithm, including full gap widths."""

    def _unverified_uncovered_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Tuple[int, int]]:
        """Return full still-missing spans from unknown gaps overlapping a range.

        Known-gap metadata may temporarily cover minutes which a later refresh
        has already populated. Width-sensitive provisional policy must measure
        the remaining contiguous outage, not either the caller-clipped overlap
        or stale outer metadata bounds.
        """
        gaps = [
            gap
            for gap in self._get_known_gaps_in_range(symbol, start_ts, end_ts)
            if str(gap.get("reason", GAP_REASON_AUTO))
            in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
            and int(gap["start_ts"]) <= int(end_ts)
            and int(gap["end_ts"]) >= int(start_ts)
        ]
        if not gaps:
            return []
        cached = self._cache.get(symbol)
        if cached is None or cached.size == 0:
            cached = np.empty((0,), dtype=CANDLE_DTYPE)
        else:
            cached = _sorted_candle_copy(cached)
            provisional_ts = self._synthetic_timestamps.get(symbol, set())
            if provisional_ts:
                cached = cached[
                    ~np.isin(
                        cached["ts"].astype(np.int64),
                        np.asarray(tuple(provisional_ts), dtype=np.int64),
                    )
                ]

        missing: List[Tuple[int, int]] = []
        for gap in gaps:
            gap_start = int(gap["start_ts"])
            gap_end = int(gap["end_ts"])
            gap_rows = self._slice_ts_range(
                cached,
                gap_start,
                gap_end,
                assume_sorted=True,
            )
            missing.extend(self._missing_spans(gap_rows, gap_start, gap_end))

        merged: List[Tuple[int, int]] = []
        for gap_start, gap_end in sorted(missing):
            if merged and int(gap_start) <= int(merged[-1][1]) + ONE_MIN_MS:
                merged[-1] = (
                    int(merged[-1][0]),
                    max(int(merged[-1][1]), int(gap_end)),
                )
            else:
                merged.append((int(gap_start), int(gap_end)))
        return [
            (gap_start, gap_end)
            for gap_start, gap_end in merged
            if gap_start <= int(end_ts) and gap_end >= int(start_ts)
        ]



def previous(manager):
    """Share current facts and replace only the frozen gap-check method."""
    result = copy.copy(manager)
    result.__class__ = StridedSearchReference
    return result


def candles(timestamps):
    """Build nonconstant OHLCV rows, preserving duplicates and supplied order."""
    rows = np.zeros(len(timestamps), dtype=CANDLE_DTYPE)
    rows['ts'] = timestamps
    rows['c'] = 100 + np.sin(np.arange(len(rows)) / 7)
    rows['o'], rows['h'], rows['l'], rows['bv'] = rows['c'] + .1, rows['c'] + 2, rows['c'] - 1, 5
    return rows


def assert_equal(manager, start, end):
    """Require exact ordered integer spans, not approximate numeric agreement."""
    actual = manager._unverified_uncovered_gap_ranges(SYMBOL, start, end)
    assert actual == previous(manager)._unverified_uncovered_gap_ranges(SYMBOL, start, end)
    return actual


@pytest.mark.parametrize('seed', range(20))
def test_randomized_strided_reference_parity(manager, seed):
    """2,000 independent states cover overlap, duplicates, off-grid and malformed gaps."""
    rng = random.Random(seed)
    minute = ONE_MIN_MS
    for _ in range(100):
        base = rng.choice([0, NOW, 2**62])
        timestamps = [base + rng.randrange(-100, 101) * minute + rng.choice([0, 0, 1, 23000])
                      for _ in range(rng.randrange(0, 160))]
        rng.shuffle(timestamps)
        manager._cache[SYMBOL] = candles(timestamps)
        manager._synthetic_timestamps[SYMBOL] = set(rng.sample(timestamps, min(8, len(timestamps))))
        gaps = []
        for __ in range(rng.randrange(0, 70)):
            a, b = [base + rng.randrange(-120, 121) * minute + rng.choice([0, 17]) for ___ in range(2)]
            reason = rng.choice(['auto_detected', 'fetch_failed', 'no_trades', 'unknown'])
            gaps.append({'start_ts': a, 'end_ts': b, 'reason': reason, 'added_at': NOW})
        gaps.extend([None, {}, ['bad', 8], [base - minute, base + minute]])
        manager._ensure_symbol_index(SYMBOL)['meta']['known_gaps'] = gaps
        start = base + rng.randrange(-150, 151) * minute
        end = base + rng.randrange(-150, 151) * minute
        assert_equal(manager, start, end)


@pytest.mark.parametrize('coverage', ['empty', 'missing', 'real', 'synthetic', 'duplicates', 'offgrid'])
def test_full_gap_bounds_are_not_clipped_to_query(manager, coverage):
    """A narrow query still measures the complete outage for width-sensitive policy."""
    minute = ONE_MIN_MS
    install(manager, [{'start_ts': 0, 'end_ts': 20 * minute, 'reason': 'auto_detected'}])
    ts = list(range(0, 21 * minute, minute))
    if coverage == 'empty':
        ts = []
    elif coverage == 'missing':
        ts = [value for value in ts if not 3 * minute <= value <= 17 * minute]
    elif coverage == 'duplicates':
        ts = ts[::-1] + ts[5:8]
    elif coverage == 'offgrid':
        ts = [value + 7 for value in ts]
    manager._cache[SYMBOL] = candles(ts)
    if coverage == 'synthetic':
        manager._synthetic_timestamps[SYMBOL] = set(ts[3:18])
    result = assert_equal(manager, 9 * minute, 11 * minute)
    if coverage in {'missing', 'synthetic'}:
        assert result == [(3 * minute, 17 * minute)]
    if coverage == 'empty':
        assert result == [(0, 20 * minute)]


def test_mutations_between_calls_never_reuse_a_timestamp_index(manager):
    """Corrections, RAM replacement/eviction and synthetic changes are visible next read."""
    minute = ONE_MIN_MS
    install(manager, [{'start_ts': minute, 'end_ts': 4 * minute, 'reason': 'auto_detected'}])
    manager._cache[SYMBOL] = candles([minute, 2 * minute, 3 * minute, 4 * minute])
    assert assert_equal(manager, minute, 4 * minute) == []
    manager._synthetic_timestamps[SYMBOL] = {2 * minute}
    assert assert_equal(manager, minute, 4 * minute) == [(2 * minute, 2 * minute)]
    manager._synthetic_timestamps[SYMBOL].clear()
    manager._cache[SYMBOL]['ts'][1] = 4 * minute
    assert assert_equal(manager, minute, 4 * minute) == [(2 * minute, 2 * minute)]
    manager._cache[SYMBOL] = candles([minute, 2 * minute, 4 * minute])
    assert assert_equal(manager, minute, 4 * minute) == [(3 * minute, 3 * minute)]
    manager._cache.pop(SYMBOL)
    assert assert_equal(manager, minute, 4 * minute) == [(minute, 4 * minute)]
    manager._ensure_symbol_index(SYMBOL)['meta']['known_gaps'][0]['reason'] = 'no_trades'
    assert assert_equal(manager, minute, 4 * minute) == []


def test_one_contiguous_index_per_read_cohort(manager, monkeypatch):
    """Every binary search shares one contiguous array; no history copy per gap."""
    minute = ONE_MIN_MS
    manager._cache[SYMBOL] = candles(np.arange(5000) * minute)
    install(manager, [{'start_ts': i * minute, 'end_ts': i * minute, 'reason': 'auto_detected'}
                      for i in range(1000)])
    original_contiguous = np.ascontiguousarray
    original_search = np.searchsorted
    built, searched = [], []

    def contiguous(array, *args, **kwargs):
        """Count explicit timestamp materialization, independent of timing."""
        result = original_contiguous(array, *args, **kwargs)
        built.append(result)
        return result

    def search(array, *args, **kwargs):
        """Require contiguous storage for all binary searches in this cohort."""
        assert array.flags.c_contiguous
        searched.append(array)
        return original_search(array, *args, **kwargs)

    monkeypatch.setattr(module.np, 'ascontiguousarray', contiguous)
    monkeypatch.setattr(module.np, 'searchsorted', search)
    assert manager._unverified_uncovered_gap_ranges(SYMBOL, 0, 5000 * minute) == []
    assert len(built) == 1
    assert len(searched) == 2000
    assert all(array is built[0] for array in searched)


def test_no_unknown_gaps_do_not_build_timestamp_index(manager, monkeypatch):
    """The existing cheap early exit remains cheap for resolved metadata."""
    install(manager, [{'start_ts': 0, 'end_ts': ONE_MIN_MS, 'reason': 'no_trades'}])
    monkeypatch.setattr(module.np, 'ascontiguousarray',
                        lambda *a, **kw: pytest.fail('no index is needed'))
    assert manager._unverified_uncovered_gap_ranges(SYMBOL, 0, ONE_MIN_MS) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('coverage', ['covered', 'missing', 'synthetic', 'wide_missing'])
@pytest.mark.parametrize('provisional', [False, True])
@pytest.mark.parametrize('span', [13.5, 77.0])
async def test_candle_and_ema_bytes_match_previous_algorithm(manager, tmp_path, coverage, provisional, span):
    """Cold/warm candle arrays and all three EMA series remain bit-for-bit equal."""
    before = StridedSearchReference(exchange=None, exchange_name='hyperliquid',
                                    cache_dir=str(tmp_path / 'prior'))
    before._now_ms_callback = lambda: NOW
    minute = ONE_MIN_MS
    end = NOW // minute * minute - minute
    rows = candles(np.arange(end - 799 * minute, end + 1, minute))
    first, last = (600, 601) if coverage != 'wide_missing' else (400, 700)
    gaps = [{'start_ts': int(rows['ts'][i]), 'end_ts': int(rows['ts'][i]),
             'reason': 'auto_detected', 'added_at': NOW} for i in range(0, 800, 3)]
    gaps.append({'start_ts': int(rows['ts'][first]), 'end_ts': int(rows['ts'][last]),
                 'reason': 'fetch_failed', 'added_at': NOW})
    missing_ts = set(int(ts) for ts in rows['ts'][first:last + 1])
    if coverage in {'missing', 'wide_missing'}:
        rows = np.delete(rows, np.arange(first, last + 1))
    try:
        for cm in (manager, before):
            install(cm, copy.deepcopy(gaps))
            cm._cache[SYMBOL] = rows.copy()
            if coverage == 'synthetic':
                cm._synthetic_timestamps[SYMBOL] = set(missing_ts)
        for _ in range(3):
            kwargs = dict(start_ts=end - 699 * minute, end_ts=end,
                          allow_remote_fetch=False, allow_provisional_internal_gaps=provisional)
            actual = await manager.get_candles(SYMBOL, **kwargs)
            expected = await before.get_candles(SYMBOL, **kwargs)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            if coverage == 'covered':
                assert actual.size > 0
            for name in ('get_ema_close_series', 'get_ema_volume_series', 'get_ema_log_range_series'):
                actual = await getattr(manager, name)(SYMBOL, span)
                expected = await getattr(before, name)(SYMBOL, span)
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes()
                if coverage == 'covered':
                    assert actual.size > 0
    finally:
        before._cleanup_on_exit()
