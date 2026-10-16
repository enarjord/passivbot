"""Synthetic regression coverage for disk-row reuse and live native HSL parity."""
from pathlib import Path
import builtins
import json
import os
import sys
from types import SimpleNamespace

import candlestick_manager as candle_module

import numpy as np
import pytest

from candlestick_manager import CandlestickManager, CANDLE_DTYPE

SYMBOL = 'TEST/USDT:USDT'
NOW = 1_800_000_012_345


def candles(count=32, *, step=60_000):
    """Construct unique sorted synthetic source rows."""
    end = (NOW // step) * step - step
    rows = np.zeros(count, dtype=CANDLE_DTYPE)
    rows['ts'] = np.arange(end - (count - 1) * step, end + 1, step)
    for field in ('o', 'h', 'l', 'c'):
        rows[field] = 100.0
    rows['bv'] = 1.0
    return rows


@pytest.fixture
def manager(tmp_path):
    """Own every migration, shard and index write below a pytest temp directory."""
    value = CandlestickManager(exchange=None, exchange_name='synthetic',
                               cache_dir=str(tmp_path / 'caches'),
                               progress_log_interval_seconds=0)
    value._now_ms_callback = lambda: NOW
    yield value
    value._cleanup_on_exit()


def persist(manager, rows, timeframe='1m'):
    """Persist synthetic data through the normal shard writer."""
    manager._save_range_incremental(SYMBOL, rows, timeframe=timeframe)


def load(manager, rows, timeframe='1m', **kwargs):
    """Read the entire synthetic range through the production loader."""
    return manager._load_from_disk(SYMBOL, int(rows['ts'][0]), int(rows['ts'][-1]),
                                   timeframe=timeframe, **kwargs)


@pytest.mark.skipif(sys.platform.startswith('win'), reason='parsed-row reuse requires a metadata-change clock')
@pytest.mark.parametrize('timeframe,step', [('1m',60_000), ('5m',300_000), ('1h',3_600_000)])
def test_unchanged_shards_are_not_read_again(manager, monkeypatch, timeframe, step):
    """Warm reuse preserves exact rows and does no further shard parsing."""
    rows = candles(step=step)
    persist(manager, rows, timeframe)
    for _ in range(3):
        np.testing.assert_array_equal(load(manager, rows, timeframe), rows)
    def unexpected_read(*args, **kwargs):
        raise AssertionError('unchanged source shard was reread')
    monkeypatch.setattr(manager, '_load_shard', unexpected_read)
    for _ in range(4):
        np.testing.assert_array_equal(load(manager, rows, timeframe), rows)


@pytest.mark.parametrize('change', ['replace', 'same_size_same_mtime', 'inplace_same_mtime', 'delete', 'corrupt', 'strict'])
def test_changed_or_unusable_sources_match_uncached_loader(manager, tmp_path, change):
    """Corrections, failures and forced reads cannot inherit a warm certificate."""
    rows = candles()
    persist(manager, rows)
    for _ in range(3):
        load(manager, rows)
    path = Path(manager._shard_path(SYMBOL, manager._date_key(int(rows['ts'][-1])), tf='1m'))
    stat = path.stat()
    corrected = rows.copy(); corrected['c'][-1] = 88.0
    if change in ('replace', 'same_size_same_mtime'):
        replacement = tmp_path / 'replacement.npy'
        with replacement.open('wb') as handle:
            np.save(handle, corrected)
        os.replace(replacement, path)
        if change == 'same_size_same_mtime':
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'inplace_same_mtime':
        with path.open('wb') as handle:
            np.save(handle, corrected)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'delete':
        path.unlink()
    elif change == 'corrupt':
        path.write_bytes(b'invalid numpy source')
    kwargs = dict(strict=change == 'strict')
    # Identical RAM starting state for the warm and reference reads.
    before = manager._cache[SYMBOL].copy()
    actual = load(manager, rows, **kwargs)
    manager._disk_rows_cache.clear(); manager._cache[SYMBOL] = before
    expected = load(manager, rows, **kwargs)
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)



@pytest.mark.parametrize('timeframe,step', [('1m', 60_000), ('5m', 300_000)])
@pytest.mark.parametrize('merge_memory_cache', [False, True])
def test_windows_creation_time_cannot_certify_same_metadata_correction(
    manager, monkeypatch, timeframe, step, merge_memory_cache
):
    """Windows metadata can stay identical across a real same-size correction."""
    rows = candles(step=step)
    persist(manager, rows, timeframe)
    path = Path(manager._shard_path(
        SYMBOL, manager._date_key(int(rows['ts'][-1])), tf=timeframe
    ))
    original_stat = os.stat
    before = original_stat(path)

    def creation_time_stat(candidate, *args, **kwargs):
        result = original_stat(candidate, *args, **kwargs)
        if str(candidate) != str(path):
            return result
        # Native Windows 3.12 exposes creation time as ctime, so an
        # in-place rewrite cannot change any field in the reuse signature.
        return SimpleNamespace(st_dev=result.st_dev, st_ino=result.st_ino,
                               st_size=result.st_size, st_mtime_ns=result.st_mtime_ns,
                               st_ctime_ns=before.st_ctime_ns)

    monkeypatch.setattr(candle_module.os, 'stat', creation_time_stat)
    # Seed an otherwise valid reuse entry, then ensure the Windows policy
    # also rejects an existing certificate instead of returning stale rows.
    monkeypatch.setattr(candle_module, 'sys', SimpleNamespace(platform='linux'))
    for _ in range(3):
        np.testing.assert_array_equal(
            load(manager, rows, timeframe, merge_memory_cache=merge_memory_cache), rows
        )
    assert manager._disk_rows_cache
    corrected = rows.copy()
    corrected['c'][-1] = 88.0
    with path.open('wb') as handle:
        np.save(handle, corrected)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
    )
    monkeypatch.setattr(candle_module, 'sys', SimpleNamespace(platform='win32'))
    for _ in range(2):
        np.testing.assert_array_equal(
            load(manager, rows, timeframe, merge_memory_cache=merge_memory_cache), corrected
        )
        assert not manager._disk_rows_cache



@pytest.mark.skipif(sys.platform.startswith('win'), reason='parsed-row reuse requires a metadata-change clock')
@pytest.mark.parametrize('failed_source', ['primary', 'legacy'])
@pytest.mark.parametrize('failure', ['empty', 'open_error'])
@pytest.mark.parametrize('merge_memory_cache', [False, True])
def test_partial_merged_shard_read_recovers_before_reuse(
    manager, tmp_path, monkeypatch, failed_source, failure, merge_memory_cache
):
    """A surviving source cannot certify a failed constituent as reusable history."""
    monkeypatch.chdir(tmp_path)
    rows = candles(3)
    rows['c'] = [100.0, 101.0, 102.0]
    primary_rows = rows[1:].copy()
    primary_rows['c'][0] = 999.0  # Legacy remains canonical on overlap.
    persist(manager, primary_rows)
    day = manager._date_key(int(rows['ts'][0]))
    legacy_dir = tmp_path / 'historical_data' / 'ohlcvs_synthetic' / 'TEST'
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / f'{day}.npy'
    np.save(legacy_path, rows[:2])
    primary_path = Path(manager._shard_path(SYMBOL, day, tf='1m'))
    manager._legacy_shard_paths_cache.clear()
    manager._legacy_day_quality_cache.clear()
    np.testing.assert_array_equal(load(manager, rows, merge_memory_cache=False), rows)
    manager._disk_rows_cache.clear()
    manager._cache.clear()
    failed_path = primary_path if failed_source == 'primary' else legacy_path
    original_read = manager._load_shard
    original_open = builtins.open
    failing = True
    source_reads = []

    def controlled_read(path, **kwargs):
        source_reads.append(Path(path).resolve())
        if failing and failure == 'empty' and Path(path).resolve() == failed_path:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return original_read(path, **kwargs)

    def controlled_open(path, *args, **kwargs):
        if failing and failure == 'open_error' and Path(path).resolve() == failed_path:
            raise OSError('synthetic transient shard read failure')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(manager, '_load_shard', controlled_read)
    monkeypatch.setattr(builtins, 'open', controlled_open)
    partial = load(manager, rows, merge_memory_cache=merge_memory_cache)
    np.testing.assert_array_equal(
        partial, rows[:2] if failed_source == 'primary' else primary_rows
    )
    assert not manager._disk_rows_cache

    # Restore readability without changing either file's identity or timestamps.
    failing = False
    source_reads.clear()
    recovered = load(manager, rows, merge_memory_cache=merge_memory_cache)
    np.testing.assert_array_equal(recovered, rows)
    assert set(source_reads) == {primary_path, legacy_path}
    assert len(manager._disk_rows_cache) == 1
    cached_rows = next(iter(manager._disk_rows_cache.values()))[1]
    assert not cached_rows.flags.writeable
    assert not np.shares_memory(cached_rows, recovered)

    source_reads.clear()
    recovered['c'][-1] = 9.0
    np.testing.assert_array_equal(
        load(manager, rows, merge_memory_cache=merge_memory_cache), rows
    )
    assert not source_reads
    manager._disk_rows_cache.clear()
    manager._cache.clear()
    np.testing.assert_array_equal(
        load(manager, rows, merge_memory_cache=merge_memory_cache), rows
    )

@pytest.mark.parametrize('timeframe,step', [('1m',60_000), ('5m',300_000)])
def test_returned_array_mutation_does_not_modify_cached_disk_rows(manager, timeframe, step):
    """Mutable consumer arrays and in-place RAM changes never mutate source reuse."""
    rows = candles(step=step); persist(manager, rows, timeframe)
    for _ in range(3):
        result = load(manager, rows, timeframe)
    result['c'][-1] = 9.0
    np.testing.assert_array_equal(load(manager, rows, timeframe), rows)


@pytest.mark.parametrize('case', range(48))
def test_merge_shortcut_matches_last_write_wins_reference(manager, case):
    """Exact unchanged subsets, corrections, duplicates and disorder retain semantics."""
    rng = np.random.default_rng(case)
    existing = candles(100)
    incoming = existing[sorted(rng.choice(len(existing), size=20, replace=False))].copy()
    if case % 6 == 1:
        incoming['c'][0] += 1
    elif case % 6 == 2:
        incoming = np.concatenate([incoming, incoming[:2]])
        incoming['c'][-1] += 1
    elif case % 6 == 3:
        rng.shuffle(incoming)
    elif case % 6 == 4:
        rng.shuffle(existing)
    elif case % 6 == 5:
        incoming['c'][0] = np.nan
    combined = np.concatenate([existing, incoming])
    ordered = combined[np.argsort(combined['ts'], kind='stable')]
    keep = np.r_[ordered['ts'][:-1] != ordered['ts'][1:], True]
    result = manager._merge_overwrite(existing, incoming)
    for field in CANDLE_DTYPE.names:
        np.testing.assert_array_equal(result[field], ordered[keep][field])
    assert not np.shares_memory(result, existing)
    assert not np.shares_memory(result, incoming)


def test_reuse_bounds_and_restart_reconstruction(manager):
    """Reuse is disposable and bounded even with many requested ranges."""
    rows = candles(64); persist(manager, rows)
    for offset in range(24):
        manager._load_from_disk(SYMBOL, int(rows['ts'][offset]), int(rows['ts'][-1]))
    assert len(manager._disk_rows_cache) <= 16
    assert sum(v[1].nbytes for v in manager._disk_rows_cache.values()) <= 8*1024*1024
    warm = load(manager, rows).copy()
    manager._disk_rows_cache.clear(); manager._cache.clear()
    np.testing.assert_array_equal(load(manager, rows), warm)



def test_strict_read_propagates_failure_after_warm_reuse(manager, monkeypatch):
    """A strict source audit cannot substitute old rows for a failed read."""
    rows = candles(); persist(manager, rows)
    for _ in range(3): load(manager, rows)
    def failed_read(*args, **kwargs):
        raise OSError('synthetic unreadable shard')
    monkeypatch.setattr(manager, '_load_shard', failed_read)
    with pytest.raises(OSError): load(manager, rows, strict=True)


def test_row_reuse_byte_budget_and_immutable_storage(manager):
    """The byte budget is enforced independently of the entry-count cap."""
    rows = candles(200_000)
    manager._remember_disk_rows(('a',), ('source',), rows)
    manager._remember_disk_rows(('b',), ('source',), rows)
    assert len(manager._disk_rows_cache) == 1
    cached = next(iter(manager._disk_rows_cache.values()))[1]
    assert not cached.flags.writeable
    assert not np.shares_memory(cached, rows)
    manager._remember_disk_rows(('oversized',), ('source',), candles(400_000))
    assert ('oversized',) not in manager._disk_rows_cache



def test_concurrent_source_rewrite_is_not_certified(manager, monkeypatch):
    """A shard changed during parsing must be reread on the next request."""
    rows = candles(); persist(manager, rows)
    real_read = manager._load_shard
    corrected = rows.copy(); corrected['c'][-1] = 77
    rewritten = False
    def changing_read(path, **kwargs):
        nonlocal rewritten
        result = real_read(path, **kwargs)
        if not rewritten:
            rewritten = True
            with Path(path).open('wb') as handle:
                np.save(handle, corrected)
        return result
    monkeypatch.setattr(manager, '_load_shard', changing_read)
    load(manager, rows)
    assert not manager._disk_rows_cache
    np.testing.assert_array_equal(load(manager, rows), corrected)


@pytest.mark.asyncio
async def test_disk_reuse_retains_ema_timing_attribution(manager):
    """Warm reuse still accounts for the complete loader in EMA diagnostics."""
    from live.ema_timing import EmaBundleTimings
    rows = candles(); persist(manager, rows)
    for _ in range(3): load(manager, rows)
    timing = EmaBundleTimings()
    async def read(symbol):
        return load(manager, rows)
    await timing.run_symbol(SYMBOL, read)
    totals = timing.summary()['stage_totals']
    assert totals['disk_load_calls'] == 1
    assert totals['disk_load_ms'] >= 0


def hsl_case(index):
    """Public synthetic corpus, shared by tests and before/after binary comparison."""
    now = 1_800_000 + [0, 1, 12_345, 59_999][index % 4]
    start = [0, 1, 59_999][(index // 4) % 3]
    short = bool(index & 16); side = 'short' if short else 'long'; sign = -1 if short else 1
    mode = ['coin','pside','unified'][(index//32)%3]
    flat = bool(index & 128)
    first = start + (60_000-start%60_000)%60_000
    prices = {str(t): float([100,120,80,105][(t//60000+index)%4])
              for t in range(first,now,60000)}
    shape = (index//256)%6
    if shape == 1:
        prices.pop(next(iter(prices)), None)
    elif shape == 2:
        prices[str(60_001)] = 99.0
    elif shape == 3:
        prices.clear()
    elif shape == 4:
        prices[str(now)] = 110.0
    elif shape == 5:
        prices['120000'] = 0.0
    fills = [dict(identity='entry', timestamp=120_000, delta=10*sign,
                  price=100, realized=0, fee=-.25, sequence=0, revision=0)]
    if flat:
        fills.append(dict(identity='close',timestamp=900_000,delta=-10*sign,
                          price=80 if not short else 120,realized=-200,fee=-.25,
                          sequence=1,revision=0))
    if index & 64:
        fills.append(dict(fills[0], revision=1, price=105))
    pair = dict(symbol=SYMBOL,position=dict(size=0 if flat else 10*sign,basis=100,
                mark=[100,70,130][index%3],multiplier=1,inverse=bool(index&512),
                pside=side,quantity_step=.1),position_at=now,mark_at=now,
                fills_started_at=now,fills_at=now,prices_at=now,
                fills=fills,prices=prices,revisions=[0,0,0,0])
    snapshot = dict(now=now,start=start,balance=[1000,250,100][index%3],balance_at=now,
                    config_at=now,max_current_age_ms=10000,mode=mode,pairs=[pair])
    if mode != 'unified':snapshot['pside']=side
    if mode == 'coin':snapshot['symbol']=SYMBOL
    return dict(snapshot=snapshot,slots=0 if index%17==0 else 1,
                span=[1,2.5,703][index%3],threshold=[.01,.2,.5][index%3],
                cooldown_ms=[0,60000,3600000][index%3],
                restart='never' if index&32 else 'always')


@pytest.mark.parametrize('index', range(64))
def test_real_native_binding_matches_json_reference(index):
    """Native-grid transport and JSON input retain identical complete HSL outputs."""
    import passivbot_rust as pbr
    assert not getattr(pbr, '__is_stub__', False)
    value = hsl_case(index)
    snap = value['snapshot']; pair = snap['pairs'][0]
    rows = [(int(ts)-60000,1,price,price,price,price,snap['now'])
            for ts,price in pair['prices'].items() if int(ts)%60000 == 0]
    grid = pbr.HslCandleSource(rows).project(snap['start'],snap['now'])[0]
    pair['prices'] = grid.values()
    expected = json.loads(pbr.hsl_evaluate(json.dumps(value)))
    pair['prices'] = {}
    assert expected == json.loads(pbr.hsl_evaluate_grids(json.dumps(value),[grid]))
    # Reuse the factual grid, but re-evaluate changed current account/market facts.
    snap['balance'] *= .5
    pair['position']['mark'] *= .75
    pair['prices'] = grid.values()
    expected = json.loads(pbr.hsl_evaluate(json.dumps(value)))
    pair['prices'] = {}
    assert expected == json.loads(pbr.hsl_evaluate_grids(json.dumps(value),[grid]))
