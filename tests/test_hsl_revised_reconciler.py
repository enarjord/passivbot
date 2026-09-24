"""General reconciliation invariants, with independent Decimal and lifecycle checks."""
from dataclasses import replace
from decimal import Decimal
import json
import random

import pytest

from hsl_reference import Fill, Position
from test_hsl_revised_history import compare, run
from test_hsl_revised_snapshot import payload, rust
import test_hsl_reference_replay as cases


@pytest.fixture(scope='module')
def native():
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension
    verify_loaded_runtime_extension()
    return passivbot_rust


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('missing_final', [False, True])
def test_martingale_tape_preserves_every_transition(native, side, inverse, missing_final):
    d = 1 if side == 'long' else -1
    rows = [(1,100), (2,98), (4,96), (7,93), (-4,95), (-10,96)]
    fills = [Fill(str(i), i+1, d*q, price, None, 0) for i,(q,price) in enumerate(rows)]
    if missing_final:
        fills.pop()
    result = compare(native, Position(0, 0, 96, inverse=inverse, pside=side), fills,
                     {i+1:price for i,(_,price) in enumerate(rows)})
    assert [e['before'] for e in result['events']] == [0,1,3,7,14,10][:len(fills)]
    assert [e['after'] for e in result['events']] == [1,3,7,14,10,0][:len(fills)]
    assert result['opening_size'] == 0
    if missing_final:
        assert result['reconciliation'] == dict(applied_at=5, before=d*10, after=0,
            basis_before=result['events'][-1]['basis'], basis_after=0, delta=-d*10)
        assert result['samples'][-1]['size'] == 0
        assert 'current_flat_timestamp_estimate' in result['reasons']
    else:
        assert result['reconciliation'] is None


@pytest.mark.parametrize('side', ['long', 'short'])
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_ambiguous_extra_inventory_and_delayed_add_repair(native, side, mode):
    d = 1 if side == 'long' else -1
    fills = [Fill('open', cases.M, d*10, 100, 0),
             Fill('close', 2*cases.M, -d*10, 80 if d==1 else 120, -200),
             Fill('reopen', 3*cases.M, d, 100, 0)]
    p = cases.pair(pside=side, size=d*2, basis=100, mark=100, fills=fills)
    selectors = {} if mode == 'unified' else dict(pside=side)
    if mode == 'coin':
        selectors['symbol'] = 'A'
    request = payload(cases.frame(p, balance=1000), mode, quantity_step=.1, **selectors)
    history = rust(request)
    assert [b['timestamp'] for b in history['boundaries']] == [2*cases.M]
    assert history['pairs'][0]['history']['opening_size'] == 0
    assert [e['after'] for e in history['pairs'][0]['history']['events']] == [10,0,1]
    repaired = json.loads(json.dumps(request))
    repaired['pairs'][0]['fills'].append(dict(identity='delayed-add', timestamp=3*cases.M+1,
        delta=d, price=100, realized=0, fee=0, sequence=None, revision=0))
    fixed = rust(repaired)
    assert [b['timestamp'] for b in fixed['boundaries']] == [2*cases.M]
    assert fixed['pairs'][0]['history']['opening_size'] == 0
    # A fresh evaluation and a clean restart use the same facts, with no memory.
    assert rust(repaired) == fixed
    assert rust(request) == history


@pytest.mark.parametrize('seed', range(40))
def test_generated_missing_and_invalid_fields_preserve_known_deltas(native, seed):
    rng = random.Random(seed)
    d = 1 if seed % 2 else -1
    size = rng.randrange(8)
    fills = []
    for i in range(30):
        quantity = rng.randrange(1, 6) if not size or rng.random() < .6 else -rng.randrange(1,size+1)
        size += quantity
        fills.append(Fill(str(i), i+1, d*quantity, rng.randrange(80,120),
                          rng.randrange(-10,11) if quantity < 0 else 0, -.1, sequence=i))
    damaged = [f for f in fills if rng.random() > .25]
    if damaged:
        damaged[0] = replace(damaged[0], price=None, fee=None)
    position = Position(d*size, 100 if size else 0, 90, pside='long' if d==1 else 'short')
    result = compare(native, position, damaged, {t:100 for t in range(31)})
    before = abs(result['opening_size'])
    for event in result['events']:
        if event['before'] != pytest.approx(before):
            assert 'local_quantity_reconciliation' in event['reasons']
        assert event['after'] == pytest.approx(event['before'] + d*event['fill']['delta'])
        assert event['after'] >= 0
        before = event['after']
    delta = result['reconciliation']['delta'] if result['reconciliation'] else 0
    assert d*before + delta == pytest.approx(position.size)
    assert result['samples'][-1]['size'] == position.size
    assert result['samples'][-1]['basis'] == position.basis
    assert run(native, position, list(reversed(damaged)), {t:100 for t in range(31)}) == result
    # Whole-tape repair converges to a clean independent reconstruction.
    compare(native, position, fills, {t:100 for t in range(31)})


def test_flat_reconciliation_does_not_reappear_as_exposure_after_boundary(native):
    p = cases.pair(fills=[Fill('open', cases.M, 2, 100, 0),
                         Fill('partial', 2*cases.M, -1, 80, -20)],
                   prices={cases.M:100, 2*cases.M:80, 3*cases.M:70}, mark=60)
    request = payload(cases.frame(p))
    prepared = rust(request)
    history = prepared['pairs'][0]['history']
    assert history['events'][-1]['after'] == 1  # factual reduction preserved
    assert history['reconciliation']['applied_at'] == 2*cases.M
    assert [s['size'] for s in history['samples']] == [2,0,0,0]
    trace = json.loads(native.hsl_revised_trace(json.dumps(request)))
    assert all(not point['exposed'] for episode in trace['episodes'][1:] for point in episode['points'])
