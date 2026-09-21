"""Exchange-flat authority with a reproducible best-effort cooldown timestamp."""
from copy import deepcopy
from dataclasses import replace

import pytest

from hsl_reference import Fill, dec
from hsl_reference_controller import replay
from test_hsl_revised_evaluator import evaluate
from test_hsl_revised_trace import oracle
from test_hsl_reference_replay import pair, frame, M

MODES = ['coin', 'pside', 'unified']


def selectors(mode, side):
    return dict(pside=side, symbol='A') if mode == 'coin' else dict(pside=side) if mode == 'pside' else {}


def damaged(side, now=30*M+1, *, repaired=False, size=0):
    direction = 1 if side == 'long' else -1
    price = 100 - direction*20
    fills = [Fill('open', M, direction*3, 100, 0),
             Fill('partial', 20*M, -direction, price, -20)]
    if repaired:
        fills.append(Fill('final', 25*M, -direction*2, price, -40))
    p = pair(size=direction*size, basis=100 if size else 0, mark=price, pside=side,
             fills=fills, now=now,
             prices={i*M: 100 if i < 5 else price for i in range(now//M+1)})
    return frame(p, now=now, balance=100)


def result(snapshot, mode, side, *, restart='always'):
    selected = selectors(mode, side)
    if mode == 'coin':
        selected['symbol'] = snapshot.pairs[0].symbol
    return evaluate(snapshot, mode, span=10, threshold=.1, cooldown_ms=5*M,
                    restart=restart, **selected)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_missing_final_fill_uses_fixed_scoped_anchor_across_time_and_restart(mode, side):
    for now, expected in [(22*M, 'halted'), (25*M-1, 'halted'), (25*M, 'normal'), (40*M, 'normal')]:
        snapshot = damaged(side, now)
        actual = result(snapshot, mode, side)
        assert actual == result(deepcopy(snapshot), mode, side)
        assert actual['decision']['action'] == expected
        assert 'current_flat_timestamp_estimate' in actual['reasons']
        assert [e['timestamp'] for e in actual['events'] if e['kind'] == 'flat'] == [20*M]
        reference = replay(oracle(snapshot, mode, **selectors(mode, side)), now=now, start=0,
                           budget=100, span=10, threshold=.1, cooldown=5*M)
        assert reference[-1].action == expected
        assert actual['decision']['raw'] == pytest.approx(float(reference[-1].raw))
        assert actual['decision']['ema'] == pytest.approx(float(reference[-1].ema))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_delivered_final_fill_replaces_estimate_with_exchange_timestamp(mode, side):
    estimated = result(damaged(side, 27*M), mode, side)
    repaired = result(damaged(side, 27*M, repaired=True), mode, side)
    assert estimated['decision']['action'] == 'normal'
    assert repaired['decision']['action'] == 'halted'
    assert repaired['decision']['flat_at'] == 25*M
    assert 'current_flat_timestamp_estimate' not in repaired['reasons']
    assert result(damaged(side, 30*M, repaired=True), mode, side)['decision']['action'] == 'normal'


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_current_residual_exposure_cannot_use_flat_fallback(mode, side):
    actual = result(damaged(side, size=2), mode, side)
    assert actual['decision']['action'] == 'panic'
    assert actual['decision']['flat_at'] is None
    assert 'current_flat_timestamp_estimate' not in actual['reasons']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_never_and_history_expiry_keep_their_finite_window_contract(mode, side):
    snapshot = damaged(side)
    assert result(snapshot, mode, side, restart='never')['decision']['action'] == 'halted'
    for start in (20*M+1, snapshot.now):
        expired = result(replace(snapshot, start=start), mode, side, restart='never')
        assert expired['decision']['action'] == 'normal'
        assert not expired['events']
    p = snapshot.pairs[0]
    empty = result(replace(snapshot, pairs=(replace(p, fills=()),)), mode, side)
    assert empty['decision']['action'] == 'normal'
    assert not empty['events']


@pytest.mark.parametrize('mode', MODES)
def test_latest_fill_is_scoped_and_aggregate_flat_is_not_net_exposure(mode):
    snapshot = damaged('long', 27*M)
    other = pair('B', pside='short', fills=[Fill('other-open',24*M,-1,100,0),
                  Fill('other-close',26*M,1,100,0)], mark=100, now=snapshot.now,
                 prices={i*M:100 for i in range(28)})
    combined = replace(snapshot, pairs=(*snapshot.pairs, other))
    actual = result(combined, mode, 'long')
    expected = 'halted' if mode == 'unified' else 'normal'
    assert actual['decision']['action'] == expected
    if mode == 'unified':
        assert actual['decision']['flat_at'] == 26*M
    else:
        assert [e['timestamp'] for e in actual['events'] if e['kind'] == 'flat'] == [20*M]
    a = damaged('long', size=2).pairs[0]
    b = pair('B', pside='short',size=-2,basis=100,mark=100,now=30*M+1,
             prices={i*M:100 for i in range(31)})
    held = result(frame(a,b,now=30*M+1,balance=100),mode,'long')
    assert held['decision']['action'] == 'panic'
    assert not any(e['kind']=='flat' for e in held['events'])


@pytest.mark.parametrize('mode', MODES)
def test_stale_current_flat_is_rejected_and_future_fill_cannot_move_anchor(mode):
    snapshot = damaged('long', 22*M)
    p = snapshot.pairs[0]
    stale = replace(snapshot, pairs=(replace(p, position_at=0),))
    with pytest.raises(ValueError,match='position/mark'):
        result(stale,mode,'long')
    future = replace(snapshot,pairs=(replace(p,fills=(*p.fills,Fill('future',40*M,-2,80,-40))),))
    actual = result(future,mode,'long')
    assert actual['decision']['flat_at'] == 20*M
    assert 'post_capture_fill' in actual['reasons']


@pytest.mark.fake_live
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_fake_exchange_flat_position_overrules_delayed_final_fill(mode, side):
    from exchanges.fake import FakeCCXTClient
    direction = 1 if side == 'long' else -1
    opening = 'buy' if direction == 1 else 'sell'
    closing = 'sell' if direction == 1 else 'buy'
    def action(order_side, qty):
        return dict(type='manual_fill', symbol='A/USDT:USDT', position_side=side,
                    side=order_side, qty=qty, reduce_only=order_side==closing)
    client = FakeCCXTClient(dict(name='revised_current_flat',
        start_time='2026-01-01T00:00:00Z', tick_interval_seconds=60,
        account=dict(balance=100), symbols={'A/USDT:USDT': dict(qty_step=.1, price_step=.1,
                                                     min_qty=.1, min_cost=1, taker=0)},
        timeline=[dict(t=i, prices={'A/USDT:USDT':100 if i<5 else 100-direction*20},
                       actions=[action(opening,3)] if i==1 else [action(closing,1)] if i==20
                       else [action(closing,2)] if i==25 else []) for i in range(31)]))
    start = client.now_ms
    def observe(with_final=False):
        events = client.get_fill_events(start,client.now_ms)
        if not with_final:
            events = events[:2]
        fills = [Fill(e['id'],e['timestamp'],e['qty']*(1 if e['side']=='buy' else -1),
                      e['price'],e['pnl'],-e['fees']['cost']) for e in events]
        position = client.positions['A/USDT:USDT',side]
        p = pair(symbol="A/USDT:USDT",size=direction*position['size'],basis=position['entry_price'],
                 mark=100-direction*20, pside=side, now=client.now_ms, fills=fills,
                 prices={start+i*M:100 if i<5 else 100-direction*20
                         for i in range(client.current_index+1)})
        return frame(p,now=client.now_ms,start=start,balance=client.balance_total)
    while client.current_index < 22:
        assert client.advance_time()
    assert result(observe(),mode,side)['decision']['action'] == 'panic'
    while client.current_index < 25:
        assert client.advance_time()
    assert client.positions['A/USDT:USDT',side]['size'] == 0
    first = result(observe(),mode,side)
    assert first['decision']['action'] == 'normal'
    assert [e['timestamp'] for e in first['events'] if e['kind']=='flat'] == [start+20*M]
    assert result(deepcopy(observe()),mode,side) == first  # no local lifecycle receipt
    repaired = result(observe(with_final=True),mode,side)
    assert repaired['decision']['action'] == 'halted'
    assert repaired['decision']['flat_at'] == start+25*M
    while client.advance_time():
        assert result(observe(),mode,side)['decision']['action'] == 'normal'
    assert result(observe(with_final=True),mode,side)['decision']['action'] == 'normal'


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('side', ['long', 'short'])
def test_live_owner_admits_after_estimated_cooldown_and_rechecks_repaired_history(monkeypatch, mode, side):
    import utils
    from live.hsl_revised_live import Owner
    from test_hsl_revised_runtime import bot, quotes, NOW, SYMBOL
    from test_hsl_revised_inputs import event
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    opening, closing = ('buy','sell') if side == 'long' else ('sell','buy')
    price = 80. if side == 'long' else 120.
    events = [event(id='open',timestamp=NOW-30*M,side=opening,qty=3.,position_side=side,
                    c_mult=1.,fee_paid=0.),
              event(id='partial',timestamp=NOW-10*M,side=closing,qty=1.,price=price,pnl=-20.,
                    position_side=side,c_mult=1.,fee_paid=0.)]
    value = bot(mode,side=side,events=events)
    value.positions[SYMBOL][side].update(size=0.,price=0.)
    value.get_raw_balance = lambda: 100.
    policy = value.config['bot']['hsl'] if mode == 'unified' else value.config['bot'][side]['hsl']
    policy['cooldown_minutes_after_red'] = 5.
    value.get_exchange_time = lambda: NOW
    value.approved_coins_minus_ignored_coins = {'long':set(), 'short':set()}
    value.approved_coins_minus_ignored_coins[side].add(SYMBOL)
    value._live_market_snapshot_max_age_ms = lambda: 10_000
    value._ensure_freshness_ledger().stamp('open_orders',now_ms=NOW)
    owner = Owner(value)
    wave = owner.capture(quotes(side))
    assert wave.permission(SYMBOL,side)[0] == 'normal'
    assert 'current_flat_timestamp_estimate' in wave.decisions[0].reasons
    order = dict(symbol=SYMBOL,position_side=side)
    owner.bind(wave,(),(order,))
    assert owner.admit(order)
    # An independently captured owner reaches the same result without old state.
    assert Owner(value).capture(quotes(side)).permission(SYMBOL,side) == wave.permission(SYMBOL,side)
    events.append(event(id='final',timestamp=NOW-2*M,side=closing,qty=2.,price=price,pnl=-40.,
                        position_side=side,c_mult=1.,fee_paid=0.))
    assert owner.capture().permission(SYMBOL,side)[0] == 'halted'
    assert not owner.admit(order)
