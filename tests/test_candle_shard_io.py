"""NPY metadata reuse must never reuse candle data or waive source validation."""

import os

import numpy as np
import pytest

from candle_shard_io import _parsed_header, load_candle_array
from candlestick_manager import CANDLE_DTYPE, CandlestickManager


def rows(count=8):
    value = np.zeros(count, dtype=CANDLE_DTYPE)
    value['ts'] = np.arange(count) * 60_000
    for field in ('o', 'h', 'l', 'c'):
        value[field] = np.arange(count) + 100
    value['bv'] = 2
    return value


def write(path, value, version=(1, 0)):
    with path.open('wb') as handle:
        np.lib.format.write_array(handle, value, version=version, allow_pickle=False)


def read(path):
    with path.open('rb') as handle:
        return load_candle_array(handle, CANDLE_DTYPE)


@pytest.mark.parametrize('version', [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize('count', [0, 1, 1440])
def test_arrays_match_numpy_detached_and_writeable(tmp_path, version, count):
    path = tmp_path / 'candles.npy'
    expected = rows(count)
    write(path, expected, version)
    first = read(path)
    np.testing.assert_array_equal(first, np.load(path, allow_pickle=False))
    assert first.flags.writeable
    if count:
        first['c'][0] = -1
        np.testing.assert_array_equal(read(path), expected)


def test_exact_header_reused_but_same_size_and_mtime_corrections_read(tmp_path):
    path = tmp_path / 'candles.npy'
    expected = rows()
    _parsed_header.cache_clear()
    write(path, expected)
    read(path)
    assert _parsed_header.cache_info().misses == 1
    stat = path.stat()
    expected['c'][-1] = 456
    write(path, expected)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    np.testing.assert_array_equal(read(path), expected)
    assert _parsed_header.cache_info().hits == 1
    replacement = tmp_path / 'new.npy'
    expected['c'][0] = 789
    write(replacement, expected)
    os.replace(replacement, path)
    np.testing.assert_array_equal(read(path), expected)


@pytest.mark.parametrize('change', ['shape', 'dtype', 'bad_header', 'truncated_payload', 'missing'])
def test_warm_header_does_not_mask_source_change_or_failure(tmp_path, change):
    path = tmp_path / 'candles.npy'
    write(path, rows())
    read(path)
    if change == 'shape':
        write(path, rows(9))
    elif change == 'dtype':
        write(path, np.arange(12, dtype=np.float64).reshape(2, 6))
    elif change == 'bad_header':
        payload = path.read_bytes().replace(b"'shape':", b"'shope':", 1)
        path.write_bytes(payload)
    elif change == 'truncated_payload':
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.unlink()
    try:
        expected = np.load(path, allow_pickle=False)
    except Exception as exc:
        with pytest.raises(type(exc)):
            read(path)
    else:
        np.testing.assert_array_equal(read(path), expected)


@pytest.mark.parametrize('value', [np.arange(12.).reshape(2, 6), np.array(3.),
                                  np.asfortranarray(np.ones((3, 4)))])
def test_noncanonical_arrays_retain_numpy_fallback(tmp_path, value):
    path = tmp_path / 'other.npy'
    write(path, value)
    np.testing.assert_array_equal(read(path), np.load(path, allow_pickle=False))


def test_objects_and_oversized_headers_keep_numpy_safety_limits(tmp_path):
    path = tmp_path / 'unsafe.npy'
    np.save(path, np.array([{'never': 'unpickle'}], dtype=object))
    with pytest.raises(ValueError):
        read(path)
    dtype = np.dtype([(f'field_{i}', np.float64) for i in range(1000)])
    write(path, np.zeros(1, dtype=dtype))
    with pytest.raises(ValueError):
        read(path)


def test_header_cache_is_bounded(tmp_path):
    _parsed_header.cache_clear()
    path = tmp_path / 'candles.npy'
    for count in range(140):
        write(path, rows(count))
        read(path)
    assert _parsed_header.cache_info().currsize == 128


def test_manager_strict_failure_and_legacy_conversion(tmp_path):
    manager = CandlestickManager(exchange=None, exchange_name='synthetic', cache_dir=str(tmp_path))
    path = tmp_path / 'legacy.npy'
    try:
        write(path, np.array([[60_000, 1., 2., .5, 1.5, 3.]]))
        expected = np.array([(60_000, 1., 2., .5, 1.5, 3.)], dtype=CANDLE_DTYPE)
        np.testing.assert_array_equal(manager._load_shard(str(path), strict=True), expected)
        write(path, rows())
        manager._load_shard(str(path), strict=True)
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(ValueError):
            manager._load_shard(str(path), strict=True)
        assert manager._load_shard(str(path)).size == 0
    finally:
        manager._cleanup_on_exit()
