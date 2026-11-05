"""Compare complete current payload reads with NumPy; synthetic files, no network.

PYTHONPATH=src python tests/benchmarks/candle_shard_headers.py
This measures repeated parsing/I/O only, not full live-bot throughput.
"""
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np

from candle_shard_io import load_candle_array
from candlestick_manager import CANDLE_DTYPE


def numpy_load(path):
    with path.open('rb') as handle:
        return np.load(handle, allow_pickle=False)


def current_load(path):
    with path.open('rb') as handle:
        return load_candle_array(handle, CANDLE_DTYPE)


with tempfile.TemporaryDirectory() as directory:
    paths = []
    for index in range(24):
        path = Path(directory) / f'{index}.npy'
        value = np.zeros(1440, dtype=CANDLE_DTYPE)
        value['ts'] = (np.arange(1440) + index * 1440) * 60_000
        for field in ('o', 'h', 'l', 'c'):
            value[field] = 100 + index
        value['bv'] = index + 1
        np.save(path, value, allow_pickle=False)
        np.testing.assert_array_equal(numpy_load(path), current_load(path))
        paths.append(path)
    samples = {'numpy': [], 'current': []}
    for repeat in range(7):
        for name, loader in [('numpy', numpy_load), ('current', current_load)]:
            start = time.process_time()
            for _ in range(20):
                for path in paths:
                    loader(path)
            samples[name].append(time.process_time() - start)
    for name, timings in samples.items():
        print(name, 'median_cpu_s', statistics.median(timings), 'samples', timings)
    print('ratio', statistics.median(samples['current']) / statistics.median(samples['numpy']))
