"""Bounded synthetic recent-gap query cost; no network or live throughput claim.

PYTHONPATH=src python tests/benchmarks/candle_gap_queries.py
"""
from pathlib import Path
import statistics
import tempfile
import time

from candlestick_manager import CandlestickManager, ONE_MIN_MS, _GAP_MAX_RETRIES, _KnownGapIndex


with tempfile.TemporaryDirectory() as directory:
    cm = CandlestickManager(exchange=None, exchange_name='fixture', cache_dir=directory)
    now = 1_800_000_000_000
    cm._now_ms_callback = lambda: now
    symbol = 'TEST/USDT:USDT'
    records = [dict(start_ts=i * 2 * ONE_MIN_MS, end_ts=i * 2 * ONE_MIN_MS,
                    reason='no_trades', retry_count=_GAP_MAX_RETRIES,
                    added_at=1, last_retry_at=1) for i in range(5000)]
    records.append(dict(start_ts=now - 5 * ONE_MIN_MS, end_ts=now - 3 * ONE_MIN_MS,
                        reason='fetch_failed', retry_count=_GAP_MAX_RETRIES,
                        added_at=now, last_retry_at=now))
    cm._save_known_gaps_enhanced(symbol, records)
    start, end = now - 4 * ONE_MIN_MS, now - ONE_MIN_MS

    def full_scan():
        return cm._fetch_start_after_deferred_gap_prefix(symbol, start, end,
            now_ms=now, gap_index=_KnownGapIndex(cm._get_known_gaps_enhanced(symbol)))

    def range_view():
        return cm._fetch_start_after_deferred_gap_prefix(symbol, start, end, now_ms=now)

    try:
        assert full_scan() == range_view() == now - 2 * ONE_MIN_MS
        before = Path(cm._index_path(symbol)).read_bytes()
        samples = {'full_scan': [], 'range_view': []}
        for _ in range(7):
            for name, query in [('full_scan', full_scan), ('range_view', range_view)]:
                began = time.process_time()
                for _ in range(100):
                    assert query() == now - 2 * ONE_MIN_MS
                samples[name].append(time.process_time() - began)
        assert Path(cm._index_path(symbol)).read_bytes() == before
        for name, values in samples.items():
            print(name, 'median_cpu_s', statistics.median(values), 'samples', values)
        print('ratio', statistics.median(samples['range_view']) / statistics.median(samples['full_scan']))
    finally:
        cm.close()
