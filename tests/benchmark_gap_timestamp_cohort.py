"""Offline paired CPU benchmark for synchronous gap timestamp searches.

Run with the project environment:
    PYTHONPATH=src python tests/benchmark_gap_timestamp_cohort.py
This measures synthetic gap-check CPU, not whole-bot load or HSL evaluation.
"""
import argparse
import json
import os
import statistics
import tempfile
import time

import numpy as np
from candlestick_manager import CandlestickManager, ONE_MIN_MS
from test_gap_timestamp_cohort_parity import candles, previous, SYMBOL, NOW


def main():
    """Compare the production method to its frozen prior implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=43200)
    parser.add_argument('--gaps', type=int, default=16884)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.gaps, args.repeats) <= 0:
        parser.error('all counts must be positive')
    with tempfile.TemporaryDirectory(prefix='gap-cohort-benchmark-') as directory:
        original_cwd = os.getcwd()
        os.chdir(directory)
        cm = CandlestickManager(exchange=None, exchange_name='synthetic', cache_dir=directory)
        cm._now_ms_callback = lambda: NOW
        try:
            rows = candles(np.arange(args.rows) * ONE_MIN_MS)
            cm._cache[SYMBOL] = rows
            indexes = np.linspace(0, args.rows - 1, args.gaps, dtype=np.int64)
            gaps = [{'start_ts': int(rows['ts'][i]), 'end_ts': int(rows['ts'][i]),
                     'reason': 'auto_detected', 'added_at': NOW} for i in indexes]
            cm._ensure_symbol_index(SYMBOL)['meta']['known_gaps'] = gaps
            cm._save_index(SYMBOL)
            cm._index.clear()
            old = previous(cm)
            start, end = int(rows['ts'][0]), int(rows['ts'][-1])
            assert cm._unverified_uncovered_gap_ranges(SYMBOL, start, end) == []
            assert old._unverified_uncovered_gap_ranges(SYMBOL, start, end) == []
            samples = {'before': [], 'after': []}
            for trial in range(args.repeats):
                order = [('before', old), ('after', cm)]
                if trial % 2:
                    order.reverse()
                for label, manager in order:
                    began = time.process_time()
                    result = manager._unverified_uncovered_gap_ranges(SYMBOL, start, end)
                    elapsed = (time.process_time() - began) * 1000
                    assert result == []
                    samples[label].append(round(elapsed, 3))
            before, after = (statistics.median(samples[key]) for key in ('before', 'after'))
            print(json.dumps({'rows': args.rows, 'gaps': args.gaps,
                              'cpu_ms': samples, 'median_before_ms': before,
                              'median_after_ms': after, 'speedup': round(before / after, 2),
                              'identical_results': True}, indent=2))
        finally:
            cm._cleanup_on_exit()
            os.chdir(original_cwd)


if __name__ == '__main__':
    main()
