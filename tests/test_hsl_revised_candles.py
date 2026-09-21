"""Offline source acquisition; no credentials or network clients are used."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from candlestick_manager import CANDLE_DTYPE, CandlestickManager, OhlcvFetchError
from live.hsl_revised_candles import CandleSourceReader

M = 60_000


async def acquire_sources(manager, symbol, **kwargs):
    # One-shot fixture. Repeated-cycle tests below deliberately reuse one reader.
    return await CandleSourceReader(manager).acquire(symbol, **kwargs)


def rows(*minutes):
    data = np.zeros(len(minutes), dtype=CANDLE_DTYPE)
    data['ts'] = [x*M for x in minutes]
    for name in ['o','h','l','c']:
        data[name] = [100+x for x in minutes]
    return data


@pytest.mark.asyncio
async def test_manager_unstandardized_read_is_sparse_detached_and_has_no_outside_seed(tmp_path):
    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: 20*M
    manager._cache['TEST'] = rows(0,2,4)
    observed = await manager.get_candles('TEST', start_ts=M, end_ts=5*M,
        allow_remote_fetch=False, standardize=False)
    assert list(observed['ts']) == [2*M,4*M]
    observed['c'][0] = 999
    assert manager._cache['TEST']['c'][1] == 102
    assert list(manager._cache['TEST']['ts']) == [0,2*M,4*M]


@pytest.mark.asyncio
async def test_native_cache_reads_without_exchange_never_relabel_one_minute_data(tmp_path):
    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: 120*M
    manager._cache['TEST'] = rows(1,2,3)
    calls=[]
    def disk(symbol,start,end,timeframe='1m'):
        calls.append(timeframe)
        return rows(0,5) if timeframe == '5m' else rows()
    manager._load_from_disk = disk
    source = await manager.get_candles('TEST',start_ts=0,end_ts=10*M,
        timeframe='5m',allow_remote_fetch=False,standardize=False)
    assert calls == ['5m']
    assert list(source['ts']) == [0,5*M]


@pytest.mark.asyncio
async def test_all_supported_resolutions_are_raw_full_window_and_immutable():
    calls=[]
    source=rows(10,12)
    async def get(symbol, **kw):
        calls.append(kw)
        return source
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1,'5m':5,'1h':60}),
        get_candles=get,_now_ms=lambda:20*M)
    result=await acquire_sources(manager,'TEST',start=0,end=15*M,timeout_seconds=1)
    assert [c['timeframe'] for c in calls] == [None,'5m','1h']
    assert all(c['start_ts']==0 and c['end_ts']==15*M and c['standardize'] is False
               and c['allow_remote_fetch'] is False for c in calls)
    assert {c['minutes'] for c in result.payload()} == {1,5,60}
    assert result.failures == ()
    source['c'][:] = 999
    assert all(c['close'] != 999 for c in result.payload())


@pytest.mark.asyncio
async def test_failed_remote_history_keeps_cache_and_other_resolutions():
    calls=[]
    async def get(symbol, **kw):
        tf=kw['timeframe'];calls.append((tf,kw['allow_remote_fetch']))
        if tf is None and kw['allow_remote_fetch']:
            raise OhlcvFetchError('private payload must not appear')
        return rows(1) if tf is None else rows(0)
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1,'5m':5}),
        get_candles=get,_now_ms=lambda:20*M)
    result=await acquire_sources(manager,'TEST',start=0,end=10*M,timeout_seconds=1,
        allow_remote_fetch=True)
    assert (None,False) in calls
    assert {c['minutes'] for c in result.payload()} == {1,5}
    assert [(f.timeframe,f.stage,f.error_type) for f in result.failures] == [
        ('1m','fetch','OhlcvFetchError')]
    assert 'private' not in repr(result)


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_does_not_discard_available_coarse_history():
    async def get(symbol, **kw):
        if kw['timeframe'] is None:
            await asyncio.Event().wait()
        return rows(0)
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1,'15m':15}),
        get_candles=get,_now_ms=lambda:20*M)
    result=await asyncio.wait_for(acquire_sources(manager,'TEST',start=0,end=15*M,
        timeout_seconds=.01),timeout=1)
    assert [c['minutes'] for c in result.payload()] == [15]
    assert result.failures[0].error_type == 'TimeoutError'


@pytest.mark.asyncio
async def test_programming_error_cancels_sibling_reads_and_propagates():
    cancelled=asyncio.Event()
    async def get(symbol, **kw):
        if kw['timeframe'] is None:
            await asyncio.sleep(0)
            raise ValueError('bad producer')
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1,'5m':5}),
        get_candles=get,_now_ms=lambda:20*M)
    with pytest.raises(ValueError,match='bad producer'):
        await acquire_sources(manager,'TEST',start=0,end=M,timeout_seconds=1)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_cancellation_propagates_without_background_reads():
    started=asyncio.Event();ended=asyncio.Event()
    async def get(symbol,**kw):
        started.set()
        try: await asyncio.Event().wait()
        finally: ended.set()
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1}),
        get_candles=get,_now_ms=lambda:20*M)
    task=asyncio.create_task(acquire_sources(manager,'TEST',start=0,end=M,timeout_seconds=1))
    await started.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert ended.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes',[{'start':-1},{'end':-1},{'end':91*1440*M},
    {'timeout_seconds':0},{'timeout_seconds':float('nan')},{'timeout_seconds':True},
    {'allow_remote_fetch':'false'},{'start':2**63,'end':2**63+M}])
async def test_invalid_acquisition_settings_fail_before_io(changes):
    with pytest.raises(ValueError,match='acquisition bounds'):
        await acquire_sources(None,'TEST',**(dict(start=0,end=M,timeout_seconds=1)|changes))


@pytest.mark.asyncio
async def test_real_manager_sparse_sources_reach_rust_finest_whole_window_projection(tmp_path):
    import json
    import passivbot_rust as pbr
    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: 20*M
    manager._cache['TEST'] = rows(0,9)
    coarse=rows(5)
    coarse['o'],coarse['h'],coarse['l'],coarse['c']=120,200,100,150
    manager._load_from_disk = lambda symbol,start,end,timeframe='1m': (
        coarse.copy() if timeframe == '5m' else rows())
    sources=await acquire_sources(manager,'TEST',start=0,end=20*M,timeout_seconds=1)
    actual=json.loads(pbr.hsl_revised_prices(json.dumps(dict(
        start=0,end=20*M,candles=sources.payload()))))
    by_time={r['timestamp']:r for r in actual['rows']}
    # Retained 1m observations remain exact; a later coarse bucket supplies an
    # interior gap, not merely the prefix before the first available 1m candle.
    assert by_time[M]['close'] == 100
    assert by_time[10*M]['close'] == 109
    assert by_time[6*M]['resolution_minutes'] == 5
    assert by_time[9*M]['resolution_minutes'] == 5
    assert by_time[20*M]['close'] == 109
    assert list(manager._cache['TEST']['ts']) == [0,9*M]


@pytest.mark.asyncio
async def test_inclusive_left_edge_keeps_only_the_in_window_close(tmp_path):
    import json
    import passivbot_rust as pbr
    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: 20*M
    manager._cache['TEST'] = rows(0,1,2)
    sources=await acquire_sources(manager,'TEST',start=2*M,end=20*M,timeout_seconds=1)
    actual=json.loads(pbr.hsl_revised_prices(json.dumps(dict(
        start=2*M,end=20*M,candles=sources.payload()))))
    assert actual['rows'][0]['timestamp'] == 2*M
    assert actual['rows'][0]['close'] == 101
    assert not actual['rows'][0]['carried']
    assert all(row['timestamp'] >= 2*M for row in actual['rows'])


@pytest.mark.asyncio
async def test_resistant_read_expires_uses_cache_and_cannot_accumulate_on_retry():
    release=asyncio.Event(); calls=[]
    async def get(symbol, **kw):
        calls.append((symbol,kw['allow_remote_fetch']))
        if not kw['allow_remote_fetch']:
            return rows(1)
        try: await release.wait()
        except asyncio.CancelledError: await release.wait()
        return rows(999)
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1}),
        get_candles=get,_now_ms=lambda:20*M)
    reader=CandleSourceReader(manager)
    task=asyncio.create_task(reader.acquire('TEST',start=0,end=M,
        timeout_seconds=.01,allow_remote_fetch=True))
    try:
        done,_=await asyncio.wait((task,),timeout=1)
        assert task in done, 'deadline must not wait for resistant cancellation'
        result=task.result()
        assert result.payload()[0]['close'] == 101
        assert result.failures[0].error_type == 'TimeoutError'
        assert result.pending_reads == 1
        for _ in range(5):
            retry=await reader.acquire('TEST',start=0,end=M,timeout_seconds=.01,
                allow_remote_fetch=True)
            assert retry.payload()[0]['close'] == 101
            assert retry.failures[0].error_type == 'CandleReadBusy'
        assert calls.count(('TEST',True)) == 1
    finally:
        release.set()
        await task
        await asyncio.gather(*list(reader._pending.values()),return_exceptions=True)
    assert reader.pending_reads == 0
    assert result.payload()[0]['close'] == 101  # late rows cannot mutate published capture


@pytest.mark.asyncio
async def test_resistant_read_capacity_is_bounded_across_symbols():
    release=asyncio.Event(); calls=[]
    async def get(symbol,**kw):
        calls.append(symbol)
        try: await release.wait()
        except asyncio.CancelledError: await release.wait()
        return rows(0)
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1}),
        get_candles=get,_now_ms=lambda:20*M)
    reader=CandleSourceReader(manager,max_pending_reads=2)
    try:
        for symbol in ('A','B','C','D'):
            result=await reader.acquire(symbol,start=0,end=M,timeout_seconds=.01)
            assert not result.tapes
            assert result.pending_reads <= 2
        assert calls == ['A','B']
        assert result.failures[0].error_type == 'CandleReadBusy'
    finally:
        release.set()
        await asyncio.gather(*list(reader._pending.values()),return_exceptions=True)
    # Completion frees capacity for later acquisitions; no stale task latch.
    assert (await reader.acquire('C',start=0,end=M,timeout_seconds=1)).payload()


@pytest.mark.asyncio
async def test_late_programming_failure_is_fatal_on_next_capture():
    release=asyncio.Event()
    async def get(symbol,**kw):
        if not kw['allow_remote_fetch']: return rows(1)
        try: await release.wait()
        except asyncio.CancelledError: await release.wait()
        raise ValueError('late producer failure')
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1}),
        get_candles=get,_now_ms=lambda:20*M)
    reader=CandleSourceReader(manager)
    try:
        await reader.acquire('TEST',start=0,end=M,timeout_seconds=.01,allow_remote_fetch=True)
    finally:
        release.set()
        await asyncio.gather(*list(reader._pending.values()),return_exceptions=True)
    with pytest.raises(ValueError,match='late producer failure'):
        await reader.acquire('TEST',start=0,end=M,timeout_seconds=1)


@pytest.mark.asyncio
async def test_outer_cancellation_does_not_wait_for_resistant_read():
    release=asyncio.Event();started=asyncio.Event()
    async def get(symbol,**kw):
        started.set()
        try: await release.wait()
        except asyncio.CancelledError: await release.wait()
        return rows(0)
    manager=SimpleNamespace(exchange=SimpleNamespace(timeframes={'1m':1}),
        get_candles=get,_now_ms=lambda:20*M)
    reader=CandleSourceReader(manager)
    task=asyncio.create_task(reader.acquire('TEST',start=0,end=M,timeout_seconds=10))
    try:
        await started.wait();task.cancel()
        done,_=await asyncio.wait((task,),timeout=1)
        assert task in done and task.cancelled()
        assert reader.pending_reads == 1
    finally:
        release.set()
        await asyncio.gather(task,*list(reader._pending.values()),return_exceptions=True)


@pytest.mark.parametrize('limit',[0,-1,1.5,True])
def test_read_capacity_requires_positive_integer(limit):
    with pytest.raises(ValueError,match='max_pending_reads'):
        CandleSourceReader(None,max_pending_reads=limit)
