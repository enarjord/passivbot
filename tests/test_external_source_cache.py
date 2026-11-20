"""External source revisions invalidate prepared datasets without reading raw arrays."""
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import backtest as bt
from external_ohlcv_source import ExternalSourceChangedError


def source_day(root, venue='bybit', coin='HYPE', date='2025-01-01', value=100):
    """Write one valid fixture day under the same symbol directory as the native loader."""
    quote='USDC' if venue=='hyperliquid' else 'USDT'
    path=root/venue/'1m'/f'{coin}_{quote}:{quote}'/(date+'.npy')
    path.parent.mkdir(parents=True,exist_ok=True)
    timestamp=int(np.datetime64(date,'ms').astype(np.int64))
    np.save(path,np.array([[timestamp,value,value+1,value-1,value,3]],dtype=np.float64))
    return path


@pytest.fixture
def cache_source(tmp_path,monkeypatch):
    """Use real cache artifacts and only deterministic local market identifiers."""
    root=tmp_path/'source';source=source_day(root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bt,'HLCVS_CACHE_ROOT',tmp_path/'prepared')
    monkeypatch.setattr(bt,'coin_to_symbol',lambda coin,venue,verbose=False:
                        f'{coin}/USDC:USDC' if venue=='hyperliquid' else f'{coin}/USDT:USDT')
    config={'backtest':{'base_dir':str(tmp_path/'results'),'start_date':'2025-01-01','end_date':'2025-01-02',
            'gap_tolerance_ohlcvs_minutes':120,'exchanges':['bybit'],'ohlcv_source_dir':str(root),'compress_cache':False},
            'bot':{'long':{'enabled':True,'n_positions':1,'total_wallet_exposure_limit_ratio':1,
                'universe':{'minimum_coin_age_days':0,'approved_coins':['HYPE']}},
                   'short':{'enabled':False,'universe':{'minimum_coin_age_days':0}},'global':{'universe':{}}},'live':{}}
    return config,root,source


def save_fixture(config, exchange='bybit', **kwargs):
    """Publish a small real prepared cache without fetching or copying raw source candles."""
    values=np.arange(12,dtype=np.float64).reshape(3,1,4)
    timestamps=np.array([1735689600000,1735689660000,1735689720000],dtype=np.int64)
    settings={'HYPE':{'exchange':'bybit','first_valid_index':0,'last_valid_index':2},
              '__meta__':{'btc_source_exchange':'bybit'}}
    return bt.save_coins_hlcvs_to_cache(config,['HYPE'],values,exchange,settings,
                                      np.ones(3),timestamps=timestamps,**kwargs)


@pytest.mark.parametrize('cache_exchange', ['bybit', 'combined'])
@pytest.mark.parametrize('mutation',['earlier_day','edit','delete','replace_same_size_and_mtime'])
def test_external_source_change_invalidates_real_prepared_cache(cache_source, mutation, cache_exchange):
    """Changed sources cannot reuse arrays indexed by the previous source revision."""
    config,root,path=cache_source
    config['backtest']['exchanges'] = ['bybit', 'hyperliquid']
    original_hash=bt.get_cache_hash(config,cache_exchange)
    saved=save_fixture(config, exchange=cache_exchange)
    assert bt.load_coins_hlcvs_from_cache(config,cache_exchange) is not None
    if mutation=='earlier_day':source_day(root,date='2024-12-31')
    elif mutation=='edit':source_day(root,value=200)
    elif mutation=='delete':path.unlink()
    else:
        before=path.stat();replacement=path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement,ns=(before.st_atime_ns,before.st_mtime_ns));replacement.replace(path)
    assert bt.get_cache_hash(config,cache_exchange)!=original_hash
    assert bt.load_coins_hlcvs_from_cache(config,cache_exchange) is None
    assert saved.exists()  # Invalidation must not delete a historical prepared snapshot.


def test_source_revision_ignores_unrelated_coins_and_includes_btc_fallback(cache_source):
    """Only selected source directories and eligible BTC inputs participate."""
    config,root,_=cache_source
    initial=bt.get_cache_hash(config,'bybit')
    source_day(root,coin='UNRELATED')
    assert bt.get_cache_hash(config,'bybit')==initial
    source_day(root,venue='binance',coin='BTC')
    assert bt.get_cache_hash(config,'bybit')!=initial


def test_native_store_identity_never_scans_external_directories(cache_source,monkeypatch):
    """The ordinary download-cache contract is unchanged without an external source."""
    config,root,_=cache_source;config['backtest']['ohlcv_source_dir']=''
    def forbidden(*args,**kwargs):pytest.fail('native store scanned an external directory')
    monkeypatch.setattr(bt,'external_source_revision',forbidden)
    initial=bt.get_cache_hash(config,'bybit');source_day(root,date='2024-12-31')
    assert bt.get_cache_hash(config,'bybit')==initial


def test_changed_source_is_rejected_before_cache_publication(cache_source,monkeypatch):
    """A file update during materialization cannot label old arrays with a new source hash."""
    config,root,_=cache_source
    original=bt._save_coins_hlcvs_artifacts_to_cache_dir
    def changed(**kwargs):
        """Publish fixture artifacts privately, then simulate an independent source writer."""
        original(**kwargs);source_day(root,value=200)
    monkeypatch.setattr(bt,'_save_coins_hlcvs_artifacts_to_cache_dir',changed)
    with pytest.raises(ExternalSourceChangedError,match='source changed'):
        save_fixture(config)
    assert not any(path.is_dir() for path in bt.HLCVS_CACHE_ROOT.iterdir())


def test_changed_source_is_rejected_while_loading_cache(cache_source,monkeypatch):
    """Recheck the source after reading cached arrays, before returning a stale snapshot."""
    config,root,_=cache_source;save_fixture(config)
    original=bt.verify_hlcvs_manifest
    def changed(*args,**kwargs):
        """Change the source only after the verified array bytes have been read."""
        result=original(*args,**kwargs);source_day(root,value=200);return result
    monkeypatch.setattr(bt,'verify_hlcvs_manifest',changed)
    with pytest.raises(ExternalSourceChangedError,match='source changed'):
        bt.load_coins_hlcvs_from_cache(config,'bybit')


def test_preparation_snapshot_cannot_be_saved_after_source_change(cache_source):
    """The caller binds its raw-read revision across the complete preparation stage."""
    config,root,_=cache_source
    revision=bt._external_source_revision(config,'bybit');source_day(root,date='2024-12-31')
    with pytest.raises(ExternalSourceChangedError,match='source changed'):
        save_fixture(config,source_revision=revision)
    assert not bt.HLCVS_CACHE_ROOT.exists()


def test_short_member_warns_even_with_another_tradable_coin(caplog):
    """A short data window remains visible while the existing global admission gate applies."""
    config={'backtest':{'start_date':'2025-01-01','end_date':'2025-01-02'}}
    timestamps=np.array([1735689600000,1735689660000],dtype=np.int64)
    settings={'SHORT':{'first_valid_index':1,'last_valid_index':1,'warmup_minutes':5},
              'READY':{'first_valid_index':0,'last_valid_index':1,'warmup_minutes':0}}
    with caplog.at_level(logging.WARNING):bt.warn_hlcv_valid_range_coverage(config,list(settings),settings,timestamps)
    assert 'SHORT has only 1 valid candles for 5 warmup minutes' in caplog.text
    bt.assert_hlcv_has_tradable_coverage(list(settings),settings)



def test_revision_scan_failure_releases_loaded_payload(cache_source, monkeypatch):
    """A disappearing source during the final check cannot leak materialized arrays."""
    config, _root, _path = cache_source
    save_fixture(config)
    original = bt.external_source_revision
    calls = 0
    released = []

    def disappearing(*args, **kwargs):
        """Simulate a source scan race only after the cache arrays were verified."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ExternalSourceChangedError("External OHLCV source changed during inspection; retry")
        return original(*args, **kwargs)

    monkeypatch.setattr(bt, "external_source_revision", disappearing)
    monkeypatch.setattr(bt, "release_materialized_payload", released.append)
    with pytest.raises(ExternalSourceChangedError, match="during inspection"):
        bt.load_coins_hlcvs_from_cache(config, "bybit")
    assert len(released) == 1
    assert released[0].shape == (3, 1, 4)


def test_source_change_while_waiting_for_publish_lock_is_rejected(cache_source, monkeypatch):
    """Recheck external inputs after acquiring the current upstream publication lock."""
    from contextlib import contextmanager

    config, root, _path = cache_source

    @contextmanager
    def changed_at_lock(_cache_dir):
        """Simulate a source writer completing while the publisher waited for its lock."""
        source_day(root, value=200)
        yield

    monkeypatch.setattr(bt, "_hlcvs_cache_publish_lock", changed_at_lock)
    with pytest.raises(ExternalSourceChangedError, match="source changed"):
        save_fixture(config)
    assert list(bt.HLCVS_CACHE_ROOT.iterdir()) == []
