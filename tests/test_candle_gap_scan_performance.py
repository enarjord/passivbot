"""Read-path scaling and retry equivalence, with no network or timing assertions."""
import random
from contextlib import asynccontextmanager

import numpy as np
import pytest

from candlestick_manager import CandlestickManager, CANDLE_DTYPE, ONE_MIN_MS, _KnownGapIndex


def reference_prefix(cm, gaps, start, end, now):
    for gap in sorted(gaps, key=lambda g: g['start_ts']):
        if start > end:
            return None
        if gap['end_ts'] < start:
            continue
        if gap['start_ts'] > start:
            break
        if not cm._should_retry_gap(gap, now_ms=now):
            start = gap['end_ts'] + ONE_MIN_MS
    return None if start > end else start


def test_gap_index_preserves_overlap_and_independent_retry_clocks(tmp_path):
    cm = CandlestickManager(exchange=None, exchange_name='kucoin', cache_dir=str(tmp_path))
    rng = random.Random(314)
    now = 1000 * ONE_MIN_MS
    for _ in range(100):
        gaps = []
        for _ in range(40):
            start = rng.randrange(800, 1001) * ONE_MIN_MS
            gaps.append(dict(start_ts=start, end_ts=start+rng.randrange(20)*ONE_MIN_MS,
                             reason=rng.choice(['auto_detected', 'no_trades', 'fetch_failed']),
                             retry_count=rng.randrange(5), last_retry_at=now-rng.randrange(10**9)))
        index = _KnownGapIndex(gaps)
        for _ in range(30):
            start = rng.randrange(790, 1020) * ONE_MIN_MS
            end = start + rng.randrange(20) * ONE_MIN_MS
            actual = cm._fetch_start_after_deferred_gap_prefix('TEST', start, end, now_ms=now, gap_index=index)
            assert actual == reference_prefix(cm, gaps, start, end, now)


@pytest.mark.asyncio
@pytest.mark.parametrize('standardize', [False, True])
async def test_present_and_historical_sparse_scan_normalizes_metadata_once(tmp_path, monkeypatch, standardize):
    class Exchange:
        id = 'kucoinfutures'
        async def fetch_ohlcv(self, *args, **kwargs):
            pytest.fail('verified no-trade gaps must not cause network calls')
    cm = CandlestickManager(exchange=Exchange(), exchange_name='kucoin', cache_dir=str(tmp_path))
    # The synthetic multi-day window exercises gap scanning, not archive I/O.
    # KuCoin supports archives independently of Exchange.fetch_ohlcv.
    monkeypatch.setattr(cm, '_archive_supported', lambda: False)
    n = 3000
    now = (2*n+1)*ONE_MIN_MS+1000
    monkeypatch.setattr(cm, '_now_ms', lambda: now)
    symbol = 'SPARSE/USDT:USDT'
    arr = np.array([(i*ONE_MIN_MS, 100., 100., 100., 100., 1.) for i in range(0, 2*n+1, 2)], dtype=CANDLE_DTYPE)
    cm._cache[symbol] = arr
    gaps = [dict(start_ts=i*ONE_MIN_MS, end_ts=i*ONE_MIN_MS, reason='no_trades', retry_count=3,
                 added_at=now, last_retry_at=now) for i in range(1, 2*n, 2)]
    cm._save_known_gaps_enhanced(symbol, gaps)
    monkeypatch.setattr(cm, '_load_from_disk', lambda *a, **kw: arr)
    original = cm._get_known_gaps_enhanced
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(cm, '_get_known_gaps_enhanced', counted)
    result = await cm.get_candles(symbol, start_ts=0, end_ts=2*n*ONE_MIN_MS, max_age_ms=None,
                                  standardize=standardize)
    assert len(calls) < 12  # independent of thousands of missing spans
    if standardize:
        assert result['ts'].tolist() == list(range(0, (2*n+1)*ONE_MIN_MS, ONE_MIN_MS))
        assert np.all(result['c'] == 100.)
    else:
        np.testing.assert_array_equal(result, arr)
        assert not np.shares_memory(result, arr)


@pytest.mark.asyncio
async def test_gap_scan_reloads_metadata_after_lock_wait(tmp_path, monkeypatch):
    class Exchange:
        id = 'kucoinfutures'
        async def fetch_ohlcv(self, *args, **kwargs):
            pytest.fail('newly verified gap must defer the fetch after waiting for the lock')
    cm = CandlestickManager(exchange=Exchange(), exchange_name='kucoin', cache_dir=str(tmp_path))
    symbol = 'SPARSE/USDT:USDT'
    now = 4*ONE_MIN_MS+1000
    monkeypatch.setattr(cm, '_now_ms', lambda: now)
    arr = np.array([(i*ONE_MIN_MS, 100., 100., 100., 100., 1.) for i in (0, 2, 3)], dtype=CANDLE_DTYPE)
    cm._cache[symbol] = arr
    cm._add_known_gap(symbol, ONE_MIN_MS, ONE_MIN_MS, reason='fetch_failed', retry_count=1)
    monkeypatch.setattr(cm, '_load_from_disk', lambda *a, **kw: arr)
    waits = []
    @asynccontextmanager
    async def lock(*args):
        waits.append(1)
        # Use a second manager to exercise shared-file change detection too.
        other = CandlestickManager(exchange=None, exchange_name='kucoin', cache_dir=str(tmp_path))
        other._record_verified_gap(symbol, ONE_MIN_MS, ONE_MIN_MS)
        yield
    monkeypatch.setattr(cm, '_acquire_fetch_lock', lock)
    result = await cm.get_candles(symbol, start_ts=0, end_ts=3*ONE_MIN_MS, max_age_ms=None)
    assert waits
    assert result['ts'].tolist() == [i*ONE_MIN_MS for i in range(4)]
    assert cm._get_known_gaps_enhanced(symbol)[0]['reason'] == 'no_trades'
