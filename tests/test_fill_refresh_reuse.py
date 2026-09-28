"""Empty-refresh replay equivalence and invalidation, without exchange access."""
from copy import deepcopy
from dataclasses import replace

import pytest

import fill_events_manager as fem


class Fetcher:
    def __init__(self):
        self.events = [dict(id='open', timestamp=1_700_000_000_000, symbol='A/USDT:USDT',
                            side='buy', qty=2., price=100., pnl=0., pb_order_type='entry',
                            position_side='long', client_order_id='entry', raw=[])]
        self.observations = []
        self.calls = []
        self.error = None

    async def fetch(self, start, end, details, on_batch=None):
        self.calls.append((start, end, deepcopy(details)))
        if self.error:
            raise self.error
        self.pnl_observations = deepcopy(self.observations)
        result = deepcopy(self.events)
        if on_batch:
            on_batch(result)
        return result


async def ready(path):
    fetcher = Fetcher()
    manager = fem.FillEventsManager(exchange='example', user='test', fetcher=fetcher, cache_path=path)
    await manager.refresh()
    fetcher.events = []
    await manager.refresh()
    assert manager._empty_refresh_proof is not None
    return manager


@pytest.mark.asyncio
async def test_empty_refresh_skips_replay_but_preserves_fetch_and_metadata(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    before = deepcopy(manager.get_events())
    checkpoint = manager.cache.load_metadata()['last_refresh_ms']
    start, end = 1_700_000_000_000, 1_700_000_001_000
    manager.cache.add_known_gap(start, end, reason=fem.GAP_REASON_FETCH_FAILED)
    monkeypatch.setattr(fem, 'compute_psize_pprice', lambda *args: pytest.fail('redundant replay'))
    for _ in range(3):
        await manager.refresh(start_ms=start, end_ms=end, mark_refreshed=False)
        assert manager.get_events() == before
        assert manager.cache.load_metadata()['last_refresh_ms'] == checkpoint
    assert len(manager.fetcher.calls) == 5
    assert manager.cache.load_metadata()['known_gaps'] == []
    await manager.refresh()
    assert manager.cache.load_metadata()['last_refresh_ms'] >= checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['fee_policy', 'raw', 'fees', 'provenance', 'source_ids',
                                  'position', 'pending', 'late_fill', 'duplicate', 'observation'])
async def test_changed_inputs_match_full_replay(tmp_path, monkeypatch, change):
    cached = await ready(tmp_path / 'cached')
    reference = await ready(tmp_path / 'reference')
    for manager in (cached, reference):
        event = manager._events[0]
        if change == 'fee_policy':
            manager.fee_pct_sanity_abs_max = .00001
        elif change == 'raw':
            event.raw.append({'startPosition': '4', 'dir': 'Open Long'})
        elif change == 'fees':
            manager._events[0] = replace(event, fees=[{'cost': .1, 'currency': 'USDT'}])
        elif change == 'provenance':
            event.provenance['test'] = {'nested': [1]}
        elif change == 'source_ids':
            event.source_ids.append('additional-component')
        elif change == 'position':
            manager._events[0] = replace(event, psize=999., pprice=999.)
        elif change == 'pending':
            manager._events[0] = replace(event, pnl_status='pending')
        elif change == 'late_fill':
            # A late execution in the same timestamp cohort changes replay.
            manager.fetcher.events = [{**event.to_dict(), 'id': 'late', 'source_ids': ['late'], 'price': 110.}]
        elif change == 'duplicate':
            manager.fetcher.events = [event.to_dict()]
        else:
            manager.fetcher.observations = [fem.PnlObservation(
                symbol=event.symbol, position_side='long', scope='position_cycle',
                realized_pnl=5., source='test', source_id='cycle', close_time=event.timestamp,
            )]
    calls = []
    original = fem.compute_psize_pprice
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(fem, 'compute_psize_pprice', counted)
    reference._empty_refresh_proof = None
    await cached.refresh(mark_refreshed=False)
    await reference.refresh(mark_refreshed=False)
    assert len(calls) == 2
    assert cached.get_events() == reference.get_events()
    assert cached.fetcher.calls == reference.fetcher.calls
    for name in ('oldest_event_ts', 'newest_event_ts', 'known_gaps', 'pnl_contract'):
        assert cached.cache.load_metadata().get(name) == reference.cache.load_metadata().get(name)


@pytest.mark.asyncio
async def test_nested_mutation_does_not_alias_proof(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    manager._events[0].raw.append({'nested': {'value': [1]}})
    await manager.refresh()
    assert manager._empty_refresh_proof is not None
    manager.get_events()[0].raw[0]['nested']['value'].append(2)
    calls = []
    original = fem.compute_psize_pprice
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(fem, 'compute_psize_pprice', counted)
    await manager.refresh()
    assert calls == [1]


@pytest.mark.asyncio
async def test_fetch_failure_and_checkpoint_failure_do_not_become_success(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    checkpoint = manager.cache.load_metadata()['last_refresh_ms']
    manager.fetcher.error = RuntimeError('offline synthetic failure')
    with pytest.raises(RuntimeError, match='synthetic failure'):
        await manager.refresh(start_ms=10, end_ms=20)
    assert manager.cache.load_metadata()['last_refresh_ms'] == checkpoint
    assert manager.cache.load_metadata()['known_gaps']
    manager.fetcher.error = None
    monkeypatch.setattr(manager.cache, 'update_metadata_from_events', lambda *a, **k: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError, match='disk failure'):
        await manager.refresh()
    assert manager._empty_refresh_proof is None


@pytest.mark.asyncio
async def test_restart_reconstructs_without_proof(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    events = manager.get_events()
    restarted = fem.FillEventsManager(exchange='example', user='test', fetcher=Fetcher(), cache_path=tmp_path)
    restarted.fetcher.events = []
    assert restarted._empty_refresh_proof is None
    await restarted.refresh()
    assert restarted.get_events() == events
    assert restarted._empty_refresh_proof is not None


@pytest.mark.asyncio
async def test_pnl_only_enrichment_after_cached_empty_refresh(tmp_path):
    manager = await ready(tmp_path)
    opened = manager._events[0]
    manager.fetcher.events = [{**opened.to_dict(), 'id': 'close', 'source_ids': ['close'], 'timestamp': opened.timestamp+1000,
                               'qty': -2., 'side': 'sell', 'price': 110., 'pb_order_type': 'close',
                               'pnl': 0., 'pnl_status': 'pending'}]
    await manager.refresh()
    manager.fetcher.events = []
    await manager.refresh()
    await manager.refresh()
    assert manager._empty_refresh_proof is not None
    assert manager.get_events()[-1].pnl == 20.
    manager.fetcher.observations = [fem.PnlObservation(
        scope='position_cycle', source='test', symbol=opened.symbol, position_side='long',
        realized_pnl=25., source_id='cycle', open_time=opened.timestamp,
        close_time=opened.timestamp+1000, close_size=2.,
    )]
    await manager.refresh()
    assert sum(event.pnl + event.fee_paid for event in manager.get_events()) == pytest.approx(25.)
    assert manager.get_events()[-1].pnl_source == fem.PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
    assert manager._empty_refresh_proof is None


@pytest.mark.asyncio
async def test_cancelled_fetch_preserves_checkpoint(tmp_path):
    import asyncio
    manager = await ready(tmp_path)
    checkpoint = manager.cache.load_metadata()['last_refresh_ms']
    manager.fetcher.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await manager.refresh()
    assert manager.cache.load_metadata()['last_refresh_ms'] == checkpoint
