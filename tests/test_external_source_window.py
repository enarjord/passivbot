"""Explicit external candle windows are independent of exchange inception metadata."""
import hashlib
import socket

import numpy as np
import pytest

import hlcv_preparation as hp
from simulation_data import OfflineDataError, data_manifest, simulation_data_policy

START = 1_774_915_200_000
END = START + 8 * 60_000


def write_source(root, exchange, coin, quote, timestamps):
    """Write ordinary synthetic source candles, never production fixtures."""
    directory = root / exchange / '1m' / f'{coin}_{quote}:{quote}'
    directory.mkdir(parents=True, exist_ok=True)
    from utils import ts_to_date
    path = directory / (ts_to_date(int(timestamps[0]))[:10] + '.npz')
    values = np.zeros(len(timestamps), dtype=[('ts','i8'),('o','f8'),('h','f8'),('l','f8'),('c','f8'),('bv','f8')])
    values['ts']=timestamps
    for name,value in [('o',100),('h',101),('l',99),('c',100),('bv',3)]:values[name]=value
    np.savez_compressed(path,candles=values)
    return path


def manager(root, tmp_path, exchange, coins, start):
    """Use the real source loader and gap standardizer without an exchange client."""
    quote='USDC' if exchange=='hyperliquid' else 'USDT'
    instance=hp.HLCVManager(exchange,ohlcv_source_dir=str(root),gap_tolerance_ohlcvs_minutes=0,verbose=False)
    instance.update_timestamp_range(start,END)
    instance.markets={f'{coin}/{quote}:{quote}':{'id':coin,'symbol':f'{coin}/{quote}:{quote}',
        'base':coin,'quote':quote,'swap':True} for coin in coins}
    instance.cm=hp.CandlestickManager(exchange=None,exchange_name=exchange,cache_dir=str(tmp_path/'cm'))
    instance.load_cc=lambda:None
    return instance


@pytest.fixture(autouse=True)
def no_network(monkeypatch,tmp_path):
    """Metadata and transport must never be needed for a caller-selected window."""
    monkeypatch.chdir(tmp_path)
    def forbidden(*args,**kwargs):pytest.fail('External window attempted network access')
    async def forbidden_metadata(*args,**kwargs):pytest.fail('External window attempted inception discovery')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(hp,'_corroborated_source_listing_boundary',forbidden_metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize('exchange',['bybit','hyperliquid'])
@pytest.mark.parametrize('offline',[False,True])
@pytest.mark.parametrize('earlier_request',[False,True])
async def test_external_sources_keep_independent_coin_starts(tmp_path,exchange,offline,earlier_request):
    """Each coin starts at real available data, without shifting another coin's request."""
    root=tmp_path/'source';quote='USDC' if exchange=='hyperliquid' else 'USDT'
    starts={'ALPHA':START+3*60000,'BETA':START+5*60000}
    files=[write_source(root,exchange,coin,quote,np.arange(first,END+60000,60000)) for coin,first in starts.items()]
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    requested=START-4*365*86400000 if earlier_request else START
    instance=manager(root,tmp_path,exchange,starts,requested)
    try:
        with simulation_data_policy({'backtest':{'offline':offline}}):
            for coin,first in starts.items():
                frame=await instance.get_ohlcvs(coin,source_dir_only=True)
                assert int(frame.timestamp.iloc[0])==first
                assert int(frame.timestamp.iloc[-1])==END
                assert frame.valid.all()
                assert instance.start_ts==requested
            if offline:
                ranges=data_manifest()['ranges']
                assert [entry['requested_start_ms'] for entry in ranges]==[requested,requested]
                assert [entry['actual_start_ms'] for entry in ranges]==list(starts.values())
    finally:await instance.aclose()
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==before


@pytest.mark.asyncio
@pytest.mark.parametrize('defect',['internal_gap','missing_tail'])
async def test_external_window_keeps_offline_integrity_checks(tmp_path,defect):
    """Accepting a local start does not waive holes or an incomplete offline end."""
    root=tmp_path/'source'
    timestamps=np.arange(START+3*60000,END+60000,60000)
    timestamps=np.delete(timestamps,2) if defect=='internal_gap' else timestamps[:-1]
    source=write_source(root,'hyperliquid','ALPHA','USDC',timestamps)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    instance=manager(root,tmp_path,'hyperliquid',['ALPHA'],START)
    try:
        with simulation_data_policy({'backtest':{'offline':True}}),pytest.raises(OfflineDataError):
            await instance.get_ohlcvs('ALPHA',source_dir_only=True)
    finally:await instance.aclose()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before


@pytest.mark.asyncio
@pytest.mark.parametrize('offline',[False,True])
@pytest.mark.parametrize('suffix',['.npz','.npy'])
@pytest.mark.parametrize('defect',['corrupt','empty'])
async def test_present_invalid_early_day_is_not_silently_skipped(tmp_path,offline,suffix,defect):
    """An existing broken day cannot be reclassified as a normal late coin start."""
    from utils import ts_to_date
    root=tmp_path/'source'
    later=write_source(root,'hyperliquid','ALPHA','USDC',np.arange(START+3*60000,END+60000,60000))
    early=later.parent/(ts_to_date(START-86400000)[:10]+suffix)
    if defect=='corrupt':early.write_bytes(b'corrupted candle fixture')
    elif suffix=='.npy':np.save(early,np.empty((0,6)))
    else:
        with np.load(later,allow_pickle=False) as data:rows=data['candles'][:0]
        np.savez_compressed(early,candles=rows)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (early,later)}
    instance=manager(root,tmp_path,'hyperliquid',['ALPHA'],START-86400000)
    try:
        with simulation_data_policy({'backtest':{'offline':offline}}),pytest.raises(hp.HlcvsDataIntegrityError,match='Invalid external OHLCV day'):
            await instance.get_ohlcvs('ALPHA',source_dir_only=False)
    finally:await instance.aclose()
    assert {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (early,later)}==before


@pytest.mark.asyncio
async def test_healthy_same_day_alternative_is_still_usable(tmp_path):
    """A broken preferred extension does not hide a complete healthy same-day variant."""
    root=tmp_path/'source'
    path=write_source(root,'hyperliquid','ALPHA','USDC',np.arange(START+3*60000,END+60000,60000))
    with np.load(path,allow_pickle=False) as data:rows=data['candles']
    np.save(path.with_suffix('.npy'),np.column_stack([rows[name] for name in ('ts','o','h','l','c','bv')]))
    path.write_bytes(b'broken preferred extension')
    instance=manager(root,tmp_path,'hyperliquid',['ALPHA'],START)
    try:
        with simulation_data_policy({'backtest':{'offline':True}}):
            result=await instance.get_ohlcvs('ALPHA',source_dir_only=True)
            assert len(result)==len(rows)
    finally:await instance.aclose()
