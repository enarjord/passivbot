"""NPY header reuse and independently verified decoded candle reads."""

import os
import sys

import numpy as np
import pytest

import candlestick_manager as candle_manager
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


@pytest.fixture
def manager(tmp_path):
    value = CandlestickManager(exchange=None, exchange_name='synthetic', cache_dir=str(tmp_path))
    yield value
    value._cleanup_on_exit()


@pytest.mark.skipif(sys.platform.startswith('win'), reason='reuse requires a change-time clock')
def test_overlapping_daily_selections_decode_each_shard_once(manager, monkeypatch):
    symbol = 'TEST/USDT:USDT'
    daily_rows = []
    one_day_ms = 24 * 60 * 60_000
    for day in range(3):
        daily = rows(4)
        daily['ts'] = 1_700_006_400_000 + day * one_day_ms + np.arange(4) * 3_600_000
        daily_rows.append(daily)
    manager._save_range_incremental(symbol, np.concatenate(daily_rows), timeframe='1h')
    decoded = []

    def record_decode(handle, dtype):
        decoded.append(handle.name)
        return load_candle_array(handle, dtype)

    monkeypatch.setattr(candle_manager, 'load_candle_array', record_decode)
    for start, end in ((0, 1), (1, 2), (0, 2)):
        actual = manager._load_from_disk(
            symbol, int(daily_rows[start]['ts'][0]), int(daily_rows[end]['ts'][-1]), timeframe='1h'
        )
        np.testing.assert_array_equal(actual, np.concatenate(daily_rows[start : end + 1]))
        actual['c'][0] = -1
    assert len(decoded) == len(set(decoded)) == 3
    assert all(not entry[1].flags.writeable for entry in manager._decoded_shards.values())
    strict = manager._load_from_disk(
        symbol, int(daily_rows[1]['ts'][0]), int(daily_rows[2]['ts'][-1]), timeframe='1h', strict=True
    )
    np.testing.assert_array_equal(strict, np.concatenate(daily_rows[1:]))
    assert len(decoded) == 5


@pytest.mark.skipif(sys.platform.startswith('win'), reason='reuse requires a change-time clock')
@pytest.mark.parametrize('change', ['overwrite', 'replace', 'corrupt', 'delete'])
def test_decoded_shard_invalidates_after_source_changes(manager, tmp_path, change):
    path = tmp_path / 'daily.npy'
    expected = rows()
    write(path, expected)
    manager._load_shard(str(path), reuse_decoded=True)
    old = path.stat()
    expected['c'][0] += 1
    if change == 'overwrite':
        write(path, expected)
    elif change == 'replace':
        replacement = tmp_path / 'replacement.npy'
        write(replacement, expected)
        os.replace(replacement, path)
    elif change == 'corrupt':
        original = path.read_bytes()
        path.write_bytes(original.replace(b"'shape':", b"'shope':", 1))
    else:
        path.unlink()
    if path.exists():
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        assert path.stat().st_size == old.st_size
    if change in {'corrupt', 'delete'}:
        assert manager._load_shard(str(path), reuse_decoded=True).size == 0
        assert not manager._decoded_shards
        write(path, expected)
    np.testing.assert_array_equal(manager._load_shard(str(path), reuse_decoded=True), expected)


@pytest.mark.skipif(sys.platform.startswith('win'), reason='reuse requires a change-time clock')
def test_decoded_shard_is_not_retained_if_replaced_during_read(manager, monkeypatch, tmp_path):
    path = tmp_path / 'daily.npy'
    replacement = tmp_path / 'replacement.npy'
    before = rows()
    after = before.copy()
    after['c'][-1] += 1
    write(path, before)
    write(replacement, after)

    def replace_after_decode(handle, dtype):
        result = load_candle_array(handle, dtype)
        os.replace(replacement, path)
        return result

    monkeypatch.setattr(candle_manager, 'load_candle_array', replace_after_decode)
    np.testing.assert_array_equal(manager._load_shard(str(path), reuse_decoded=True), before)
    assert not manager._decoded_shards
    monkeypatch.setattr(candle_manager, 'load_candle_array', load_candle_array)
    np.testing.assert_array_equal(manager._load_shard(str(path), reuse_decoded=True), after)


@pytest.mark.skipif(sys.platform.startswith('win'), reason='reuse requires a change-time clock')
def test_decoded_shard_metadata_failure_still_decodes_open_file(manager, monkeypatch, tmp_path):
    path = tmp_path / 'daily.npy'
    expected = rows()
    write(path, expected)
    manager._load_shard(str(path), reuse_decoded=True)
    fstat = os.fstat

    def fail_identity_once(fd):
        monkeypatch.setattr(os, 'fstat', fstat)
        raise OSError('synthetic descriptor identity failure')

    monkeypatch.setattr(os, 'fstat', fail_identity_once)
    np.testing.assert_array_equal(manager._load_shard(str(path), reuse_decoded=True), expected)
    assert not manager._decoded_shards


@pytest.mark.skipif(sys.platform.startswith('win'), reason='reuse requires a change-time clock')
def test_decoded_shard_lru_enforces_both_limits(manager, monkeypatch, tmp_path):
    one_shard = rows()
    monkeypatch.setattr(candle_manager, '_MAX_DECODED_SHARDS', 2)
    monkeypatch.setattr(candle_manager, '_MAX_DECODED_SHARD_BYTES', 3 * one_shard.nbytes)
    paths = [tmp_path / f'{ix}.npy' for ix in range(4)]
    for path in paths[:3]:
        write(path, one_shard)
    for path in (paths[0], paths[1], paths[0], paths[2]):
        manager._load_shard(str(path), reuse_decoded=True)
    assert list(manager._decoded_shards) == [str(paths[0]), str(paths[2])]
    write(paths[3], rows(3 * len(one_shard) - 1))
    manager._load_shard(str(paths[3]), reuse_decoded=True)
    assert list(manager._decoded_shards) == [str(paths[3])]
    assert manager._decoded_shard_bytes == rows(3 * len(one_shard) - 1).nbytes
