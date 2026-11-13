"""Fill-refresh replay equivalence and invalidation, without exchange access."""
import json
import math
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

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
@pytest.mark.parametrize("with_overlap", [False, True])
async def test_encoded_proof_avoids_recursive_tape_comparison(tmp_path, monkeypatch, with_overlap):
    manager = await ready(tmp_path)
    assert manager._empty_refresh_proof.evidence_bytes is not None
    if with_overlap:
        manager.fetcher.events = [manager.get_events()[0].to_dict()]
    monkeypatch.setattr(fem, "_same_fill_tape", lambda *_: pytest.fail("recursive tape comparison"))
    work = await manager.refresh()
    assert work.replay_reused
    assert work.duplicate_rows == int(with_overlap)
    assert work.day_replacements == 0


@pytest.mark.asyncio
async def test_encoding_failure_uses_exact_replay_and_recursive_fallback(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    assert manager._empty_refresh_proof.evidence_bytes is not None

    def unavailable(*_):
        raise ValueError("unsupported evidence encoding")

    monkeypatch.setattr(fem, "marshal", SimpleNamespace(dumps=unavailable))
    work = await manager.refresh()
    assert not work.replay_reused
    assert manager._empty_refresh_proof.evidence_bytes is None
    assert (await manager.refresh()).replay_reused

    manager.get_events()[0].raw.append({"changed": True})
    assert not (await manager.refresh()).replay_reused


class _PretendsToBeList(type):
    __hash__ = type.__hash__

    def __eq__(self, other):
        return other is list


class _ListSubclass(list, metaclass=_PretendsToBeList):
    def __buffer__(self, _flags):
        return memoryview(b"list")


class _BufferEvidence(int):
    def __buffer__(self, _flags):
        return memoryview(b"1")


@pytest.mark.asyncio
async def test_unsupported_container_seed_keeps_recursive_tape_proof(tmp_path):
    manager = await ready(tmp_path)
    manager.get_events()[0].raw.append({"nested": _ListSubclass([True])})
    assert not (await manager.refresh()).replay_reused
    assert manager._empty_refresh_proof.evidence_bytes is None
    assert (await manager.refresh()).replay_reused

    manager.get_events()[0].raw[0]["nested"][0] = 1
    assert not (await manager.refresh()).replay_reused


@pytest.mark.asyncio
async def test_encoded_tape_ignores_equivalent_evidence_sharing(tmp_path):
    manager = await ready(tmp_path)
    shared = [1, False]
    manager.get_events()[0].raw.append({"first": shared, "second": shared})
    assert not (await manager.refresh()).replay_reused
    assert manager._empty_refresh_proof.evidence_bytes is not None

    manager.get_events()[0].raw[0]["second"] = list(shared)
    assert (await manager.refresh()).replay_reused


@pytest.mark.asyncio
@pytest.mark.parametrize("old, new", [(False, 0), (1, 1.0), (-0.0, 0.0)])
async def test_same_id_evidence_correction_replays_and_persists(tmp_path, old, new):
    manager = await ready(tmp_path)
    event, = manager.get_events()
    manager.fetcher.events = [{**event.to_dict(), "raw": [{"fact": old}]}]
    await manager.refresh()
    manager.fetcher.events = []
    await manager.refresh()
    assert manager._empty_refresh_proof.evidence_bytes is not None

    manager.fetcher.events = [{**event.to_dict(), "raw": [{"fact": new}]}]
    work = await manager.refresh()
    assert not work.replay_reused and work.changed_overlap_rows == 1
    for saved in (manager.get_events()[0], manager.cache.load()[0]):
        result = saved.raw[0]["fact"]
        assert type(result) is type(new)
        if type(new) is float:
            assert math.copysign(1, result) == math.copysign(1, new)


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['fee_policy', 'raw', 'fees', 'provenance', 'source_ids',
                                  'position', 'pending', 'late_fill', 'explicit_position',
                                  'raw_type', 'observation'])
async def test_changed_inputs_match_full_replay(tmp_path, monkeypatch, change):
    cached = await ready(tmp_path / 'cached')
    reference = await ready(tmp_path / 'reference')
    for manager in (cached, reference):
        event = manager._events[0]
        if change == 'fee_policy':
            manager.max_abs_fee_fraction = .00001
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
        elif change == 'explicit_position':
            manager.fetcher.events = [{**event.to_dict(), 'psize': 999.0}]
        elif change == 'raw_type':
            event.raw.append({'nested': {'reported': True}})
            # True == 1 in Python, but they are distinct raw evidence.
            manager.fetcher.events = [{**event.to_dict(), 'raw': [
                {'nested': {'reported': 1}},
            ]}]
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
@pytest.mark.parametrize("as_batch", [True, False])
async def test_duplicate_overlap_skips_replay_and_verifies_disk_once(tmp_path, monkeypatch, as_batch):
    manager = await ready(tmp_path)
    event, = manager.get_events()
    day = fem._day_key(event.timestamp)
    path = tmp_path / f"{day}.json"
    assert event.psize > 0 and event.pprice > 0
    raw = event.to_dict()
    raw.pop("psize")
    raw.pop("pprice")
    manager.fetcher.events = [raw]
    if not as_batch:
        fetch = manager.fetcher.fetch

        async def returns_rows(start, end, details, on_batch=None):
            return await fetch(start, end, details)

        monkeypatch.setattr(manager.fetcher, "fetch", returns_rows)

    checked = []
    original = manager.cache._stage_day_writes

    def stage(days):
        checked.append(list(days))
        return original(days)

    monkeypatch.setattr(manager.cache, "_stage_day_writes", stage)
    monkeypatch.setattr(fem, "compute_psize_pprice", lambda *_: pytest.fail("redundant replay"))
    before = path.stat()
    start, end = event.timestamp, event.timestamp + 1_000
    manager.cache.add_known_gap(start, end, reason=fem.GAP_REASON_FETCH_FAILED)
    checkpoint = manager.cache.load_metadata()["last_refresh_ms"]
    work = await manager.refresh(start_ms=start, end_ms=end, mark_refreshed=False)
    assert work == fem.FillRefreshWork(
        received_rows=1,
        duplicate_rows=1,
        changed_overlap_rows=0,
        ignored_overlap_rows=0,
        pnl_observations=0,
        duplicate_days_checked=1,
        day_replacements=0,
        replay_reused=True,
    )
    assert checked == [[day]]
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert manager.get_events() == [event]
    assert manager.cache.load_metadata()["last_refresh_ms"] == checkpoint
    assert manager.cache.load_metadata()["known_gaps"] == []
    assert len(manager.fetcher.calls) == 3

    # An identical exchange row also repairs missing on-disk coverage.
    path.unlink()
    assert (await manager.refresh()).day_replacements == 1
    assert manager.cache.load() == [event]


@pytest.mark.asyncio
async def test_duplicate_repairs_durable_raw_evidence_type(tmp_path):
    fetcher = Fetcher()
    fetcher.events[0]["raw"] = [{"reported": False}]
    manager = fem.FillEventsManager(
        exchange="example", user="test", fetcher=fetcher, cache_path=tmp_path,
    )
    await manager.refresh()
    raw = deepcopy(fetcher.events)
    fetcher.events = []
    await manager.refresh()
    event, = manager.get_events()
    path = tmp_path / f"{fem._day_key(event.timestamp)}.json"
    incorrect = json.loads(path.read_text())
    incorrect[0]["raw"][0]["reported"] = 0
    assert incorrect == [event.to_dict()]
    path.write_text(json.dumps(incorrect))

    fetcher.events = raw
    work = await manager.refresh()
    assert work.duplicate_rows == 1 and work.replay_reused
    assert work.day_replacements == 1
    assert manager.cache.load()[0].raw[0]["reported"] is False


@pytest.mark.asyncio
async def test_duplicate_only_refresh_can_establish_proof_without_empty_fetch(tmp_path):
    manager = await ready(tmp_path)
    manager._empty_refresh_proof = None
    event, = manager.get_events()
    manager.fetcher.events = [event.to_dict()]
    first = await manager.refresh()
    second = await manager.refresh()
    assert first.duplicate_rows == second.duplicate_rows == 1
    assert first.replay_reused is False and second.replay_reused is True
    assert first.day_replacements == second.day_replacements == 0


@pytest.mark.asyncio
async def test_changed_fill_replays_duplicate_day_and_preserves_same_timestamp_order(tmp_path):
    manager = await ready(tmp_path)
    entry, = manager.get_events()
    later = [
        {**entry.to_dict(), "id": key, "source_ids": [key],
         "timestamp": entry.timestamp + 86_400_000, "datetime": "", "qty": qty}
        for key, qty in (("second", 1.0), ("third", 3.0))
    ]
    manager.fetcher.events = later
    await manager.refresh()
    manager.fetcher.events = []
    await manager.refresh()
    assert manager._empty_refresh_proof is not None
    original = manager.get_events()

    duplicates = [event.to_dict() for event in reversed(original[1:])]
    for event in duplicates:
        event.pop("psize")
        event.pop("pprice")
    correction = {**entry.to_dict(), "qty": 4.0}
    manager.fetcher.events = [*duplicates, correction]
    work = await manager.refresh()
    assert work.duplicate_rows == 2 and work.changed_overlap_rows == 1
    assert work.replay_reused is False
    expected = manager.get_events()
    assert [(ev.id, ev.psize) for ev in expected] == [
        ("open", 4.0), ("second", 5.0), ("third", 8.0),
    ]
    assert manager.cache.load() == expected
    restarted = fem.FillEventsManager(
        exchange="example", user="test", fetcher=Fetcher(), cache_path=tmp_path,
    )
    await restarted.ensure_loaded()
    assert restarted.get_events() == expected


@pytest.mark.asyncio
async def test_duplicate_after_failed_correction_repairs_day_before_checkpoint(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    event, = manager.get_events()
    checkpoint = manager.cache.load_metadata()["last_refresh_ms"]
    save_metadata = manager.cache.save_metadata

    def fail_metadata(_metadata):
        raise OSError("failed refresh checkpoint")

    manager.fetcher.events = [{**event.to_dict(), "price": 101.0}]
    monkeypatch.setattr(manager.cache, "save_metadata", fail_metadata)
    with pytest.raises(OSError, match="failed refresh checkpoint"):
        await manager.refresh()
    assert manager.cache.load_metadata()["last_refresh_ms"] == checkpoint
    assert manager.cache.load()[0].price == 101.0
    assert manager.get_events() == [event]
    assert manager._empty_refresh_proof is None

    monkeypatch.setattr(manager.cache, "save_metadata", save_metadata)
    manager.fetcher.events = [event.to_dict()]
    work = await manager.refresh()
    assert work.duplicate_rows == 1 and work.day_replacements == 1
    assert work.replay_reused is False
    assert manager.cache.load() == [event]


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
@pytest.mark.parametrize("before, after", [
    (False, 0),
    (1, 1.0),
    (-0.0, 0.0),
    ([1], (1,)),
    ([1], _ListSubclass([1])),
    (1, _BufferEvidence(1)),
])
async def test_nested_type_change_outside_overlap_invalidates_replay_proof(
    tmp_path, monkeypatch, before, after
):
    manager = await ready(tmp_path)
    prior, = manager.get_events()
    prior.raw.append({"raw_fact": before})
    later = {**prior.to_dict(), "id": "later", "source_ids": ["later"],
             "timestamp": prior.timestamp + 1000, "raw": []}
    manager.fetcher.events = [later]
    await manager.refresh()
    manager.fetcher.events = []
    await manager.refresh()
    assert manager._empty_refresh_proof.evidence_bytes is not None

    prior, later = manager.get_events()
    prior.raw[0]["raw_fact"] = after
    manager.fetcher.events = [later.to_dict()]
    calls = []
    replay = fem.compute_psize_pprice

    def counted(payload):
        calls.append(1)
        return replay(payload)

    monkeypatch.setattr(fem, "compute_psize_pprice", counted)
    work = await manager.refresh()
    assert work.duplicate_rows == 1 and work.replay_reused is False
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
async def test_pnl_only_enrichment_after_cached_empty_refresh(tmp_path, monkeypatch):
    manager = await ready(tmp_path)
    synthesized = []
    original_synthesize = manager._synthesize_missing_pnls

    def count_synthesis(payload):
        synthesized.append([row['id'] for row in payload if row['pnl_status'] == 'pending'])
        return original_synthesize(payload)

    monkeypatch.setattr(manager, '_synthesize_missing_pnls', count_synthesis)
    opened = manager._events[0]
    manager.fetcher.events = [{**opened.to_dict(), 'id': 'close', 'source_ids': ['close'], 'timestamp': opened.timestamp+1000,
                               'qty': -2., 'side': 'sell', 'price': 110., 'pb_order_type': 'close',
                               'pnl': 0., 'pnl_status': 'pending'}]
    await manager.refresh()
    assert synthesized == [['close']]
    manager.fetcher.events = [manager.get_events()[-1].to_dict()]
    await manager.refresh()
    assert synthesized == [['close']]
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
    assert synthesized == [['close']]


@pytest.mark.asyncio
async def test_cancelled_fetch_preserves_checkpoint(tmp_path):
    import asyncio
    manager = await ready(tmp_path)
    checkpoint = manager.cache.load_metadata()['last_refresh_ms']
    manager.fetcher.error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await manager.refresh()
    assert manager.cache.load_metadata()['last_refresh_ms'] == checkpoint
