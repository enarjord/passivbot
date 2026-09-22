"""Passive revised-HSL observation and presentation, using the native evaluator."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from live import hsl_revised_live, hsl_revised_diagnostics as diagnostics
from live.hsl_revised_runtime import Scope
from passivbot_monitor import _monitor_hsl_section
from test_hsl_revised_runtime import bot as make_bot, quotes, NOW, SYMBOL


def capture_report(owner, quotes=None):
    wave = owner.capture(quotes)
    owner.report(wave)
    return wave


@pytest.fixture
def observed(monkeypatch):
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: NOW)
    def build(mode='coin'):
        bot = make_bot(mode)
        bot.get_exchange_time = lambda: NOW
        bot.approved_coins_minus_ignored_coins = {'long': {SYMBOL}, 'short': set()}
        bot._live_market_snapshot_max_age_ms = lambda: 10_000
        bot.freshness_ledger = bot._ensure_freshness_ledger()
        bot.freshness_ledger.stamp('open_orders', now_ms=NOW-200)
        events = []
        bot._emit_live_event = lambda *args, **kwargs: events.append((args, kwargs)) or True
        owner = hsl_revised_live.owner(bot)
        wave = capture_report(owner, quotes())
        return bot, owner, wave, events
    return build


@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
def test_monitor_topology_is_native_scope_and_numeric_evidence(observed, mode):
    bot, owner, wave, events = observed(mode)
    payload = _monitor_hsl_section(bot, now_ms=NOW)
    assert payload['engine'] == 'revised' and payload['observation_status'] == 'current'
    assert set(payload).isdisjoint({'long', 'short'})
    assert payload['scope_count'] == 1
    row, = payload['scopes']
    assert row['action'] == 'panic' and row['tier'] == 'red'
    assert row['symbol'] == (SYMBOL if mode == 'coin' else None)
    assert row['pside'] == (None if mode == 'unified' else 'long')
    native = json.loads(wave.decisions[0].payload)['decision']
    assert row['score'] == min(native['raw'], native['ema'])
    assert row['threshold'] == wave.decisions[0].threshold
    assert row['estimated'] and row['estimates']
    assert len(events) == 1


def test_same_state_refreshes_numbers_without_repeating_status(observed):
    bot, owner, wave, events = observed()
    original = diagnostics.snapshot(bot, now_ms=NOW)['scopes'][0]['score']
    capture_report(owner, {SYMBOL: replace(quotes()[SYMBOL], bid=80., ask=80., last=80.)})
    assert diagnostics.snapshot(bot, now_ms=NOW)['scopes'][0]['score'] > original
    assert len(events) == 1
    # Readers cannot mutate the diagnostic store or Rust decisions.
    result = diagnostics.snapshot(bot, now_ms=NOW)
    result['scopes'][0]['action'] = 'normal'
    assert diagnostics.snapshot(bot, now_ms=NOW)['scopes'][0]['action'] == 'panic'


@pytest.mark.parametrize('change', ['ttl', 'confirmation', 'generation'])
def test_last_decision_is_explicitly_stale_when_current_inputs_are_not_confirmed(observed, change):
    bot, _, _, _ = observed()
    now = NOW
    if change == 'ttl':
        now += 10_000
    elif change == 'confirmation':
        bot._authoritative_pending_confirmations = {'positions': bot.freshness_ledger.epoch+1}
    else:
        bot._account_invalidation_generation = 1
    result = diagnostics.snapshot(bot, now_ms=now)
    assert result['observation_status'] == 'stale'
    assert result['scopes'][0]['action'] == 'panic'  # Last observation, not current authority.


def test_never_evaluated_monitor_does_not_initialize_trading_owner():
    bot = make_bot('unified')
    result = _monitor_hsl_section(bot, now_ms=NOW)
    assert result['observation_status'] == 'not_evaluated'
    assert result['scopes'] == []
    assert not hasattr(bot, '_hsl_revised_live')


def test_sink_or_projection_failure_does_not_change_native_permission(observed, monkeypatch):
    bot, owner, wave, _ = observed()
    def failure(*args, **kwargs):
        raise OSError('diagnostic sink failure')
    bot._emit_live_event = failure
    bot._hsl_revised_diagnostic_event = None
    assert capture_report(owner, quotes()).permission(SYMBOL, 'long') == wave.permission(SYMBOL, 'long')
    monkeypatch.setattr(diagnostics, '_row', failure)
    assert capture_report(owner, quotes()).permission(SYMBOL, 'long') == wave.permission(SYMBOL, 'long')
    assert diagnostics.snapshot(bot, now_ms=NOW)['observation_status'] == 'diagnostic_unavailable'


def test_large_scope_table_keeps_complete_counts_and_bounded_priority_sample(observed):
    from live.hsl_revised_runtime import Unavailable
    bot, _, wave, events = observed()
    native = wave.decisions[0]
    decisions = tuple(replace(native, scope=Scope('coin', 'long', f'COIN{i:04}/USDT:USDT')) for i in range(300))
    wave = replace(wave, decisions=decisions,
                   unavailable=(Unavailable(Scope('coin', 'short', SYMBOL), 'current_mark_unavailable'),))
    diagnostics.record(bot, wave)
    data = diagnostics.snapshot(bot, now_ms=NOW)
    assert data['scope_count'] == 301 and data['counts']['red'] == 300
    assert data['counts']['unavailable'] == 1
    assert len(data['scopes']) == diagnostics.SCOPE_LIMIT
    assert data['omitted_scopes'] == 301-diagnostics.SCOPE_LIMIT
    event = events[-1][1]['data']
    assert len(event['scopes']) == diagnostics.SAMPLE_LIMIT
    assert event['scope_count'] == 301


def test_current_input_absence_is_not_misreported_green(observed):
    bot, owner, _, _ = observed()
    bot.positions[SYMBOL]['long']['price'] = float('nan')
    capture_report(owner)
    row, = diagnostics.snapshot(bot, now_ms=NOW)['scopes']
    assert row['availability'] == 'unavailable' and row['tier'] is None
    assert row['score'] is None and row['unavailable_reason']


def test_tui_shows_portfolio_scope_and_observation_status(observed):
    from monitor_tui import MonitorTuiState, render_screen
    bot, _, _, _ = observed('unified')
    state = MonitorTuiState(relay_url='http://127.0.0.1:8765', exchange='fake', user='example')
    state.apply_message(dict(type='snapshot', exchange='fake', user='example', seq=1, ts=NOW,
                            payload={'hsl': diagnostics.snapshot(bot, now_ms=NOW+10_000)}))
    screen = render_screen(state, width=180)
    assert 'revised unified' in screen and 'stale' in screen
    assert 'portfolio: panic' in screen
    assert 'long=disabled' not in screen


def test_dashboard_hsl_summary_handles_legacy_and_revised():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard function smoke')
    source = (Path(__file__).resolve().parents[1] / 'src/monitor_dashboard_static/dashboard.js').read_text()
    function = source[source.index('  function hslSummary('):source.index('  function renderBotOverview(')]
    script = function + '''
const assert = require('node:assert/strict');
assert.equal(hslSummary({long: {tier: 'green'}, short: {tier: 'red'}}), 'L green / S red');
const revised = hslSummary({engine: 'revised', signal_mode: 'unified', observation_status: 'stale', counts: {red: 1, estimated: 1}});
assert.match(revised, /revised unified.*stale.*RED 1.*estimated 1/);
assert.ok(!revised.includes('L '));
assert.match(hslSummary({engine: 'revised', counts: {inactive: 2}}), /inactive 2/);
assert.equal(hslScopeStatus({action: null, tier: 'inactive', availability: 'available'}), 'inactive');
assert.equal(hslScopeStatus({action: null, tier: null, availability: 'unavailable'}), 'unavailable');
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)


def test_structured_console_shows_revised_counts_without_legacy_tiers(observed):
    from live.event_bus import LiveEvent, EventTypes, format_console_event
    bot, _, _, _ = observed('unified')
    event = LiveEvent(EventTypes.HSL_STATUS, status='degraded',
                      data=diagnostics.snapshot(bot, now_ms=NOW))
    text = format_console_event(event)
    assert 'engine=revised mode=unified observation=current' in text
    assert 'red=1' in text and 'estimated=1' in text
    assert 'tier=disabled' not in text


def test_clean_red_aggregate_survives_smoke_and_startup_preview_consumers(observed):
    from live.smoke_report import _risk_event_group, _risk_attention_rank, _summarize_hsl_status
    from tools.hsl_startup_preview import _bounded_hsl_data, _status_from_event
    bot, _, wave, events = observed('unified')
    # No approximation/degradation flag should be needed to get RED attention.
    clean = replace(wave, decisions=tuple(replace(d, reasons=()) for d in wave.decisions))
    diagnostics.record(bot, clean)
    event = dict(event_type='hsl.status', **{key: events[-1][1][key]
        for key in ('level', 'status', 'data')})
    assert event['status'] == 'ok' and event['data']['tier'] == 'red'
    group = _risk_event_group(bot_key='fake/example', row={'ts': NOW, 'seq': 1},
                             live_event=event, path=Path('events.ndjson'), line_no=1)
    assert _risk_attention_rank(group) == 35
    assert _summarize_hsl_status({'one': group})['tier_counts'] == {'red': 1}
    assert _status_from_event({'latest_data': _bounded_hsl_data(event)}) == 'red'


def test_inactive_scopes_are_visible_in_tui_and_overview(observed):
    from monitor_tui import MonitorTuiState, render_screen
    bot, owner, _, _ = observed()
    bot.config['bot']['long']['risk']['n_positions'] = 0
    capture_report(owner)
    payload = diagnostics.snapshot(bot, now_ms=NOW)
    assert payload['tier'] == 'inactive'
    state = MonitorTuiState(relay_url='http://127.0.0.1:8765', exchange='fake', user='example')
    state.apply_message(dict(type='snapshot', exchange='fake', user='example', seq=1, ts=NOW,
                            payload={'hsl': payload}))
    screen = render_screen(state, width=200)
    assert 'inactive=1' in screen and f'{SYMBOL} long: inactive' in screen
    assert f'{SYMBOL} long: available' not in screen


def test_expiry_uses_retained_position_observation_not_newer_ledger_stamp(observed, monkeypatch):
    import utils
    bot, owner, original, _ = observed()
    later = NOW+4000
    monkeypatch.setattr(utils, 'utc_ms', lambda: later)
    bot.get_exchange_time = lambda: later
    for name in ('balance', 'positions', 'open_orders'):
        bot.freshness_ledger.stamp(name, now_ms=later)
    wave = capture_report(owner, {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=later)})
    assert wave.position_observed_ms == original.position_observed_ms == NOW-200
    data = diagnostics.snapshot(bot, now_ms=NOW+10_000)
    assert data['account_unavailable'] == []
    assert data['input_expires_at_ms'] == NOW+9800
    assert data['observation_status'] == 'stale'


def test_unavailable_scope_fallback_remains_warning_without_event_sink(observed, caplog):
    import logging
    bot, owner, _, _ = observed()
    bot._emit_live_event = None
    bot.positions[SYMBOL]['long']['price'] = float('nan')
    with caplog.at_level(logging.WARNING):
        capture_report(owner)
    assert any(row.levelno == logging.WARNING and 'revised HSL' in row.message
               and 'unavailable=1' in row.message for row in caplog.records)


def test_initial_projection_failure_is_distinct_from_not_evaluated(observed, monkeypatch):
    bot, owner, _, _ = observed()
    del bot._hsl_revised_diagnostic_observation
    original = diagnostics._row
    def fail(*args, **kwargs):
        raise ValueError('broken diagnostics')
    monkeypatch.setattr(diagnostics, '_row', fail)
    wave = capture_report(owner)
    assert wave.permission(SYMBOL, 'long')[0] == 'panic'
    assert diagnostics.snapshot(bot, now_ms=NOW)['observation_status'] == 'diagnostic_unavailable'
    monkeypatch.setattr(diagnostics, '_row', original)
    capture_report(owner)
    assert diagnostics.snapshot(bot, now_ms=NOW)['observation_status'] == 'current'


def test_account_freshness_recovery_emits_once_without_numeric_churn(observed):
    bot, owner, _, events = observed()
    bot._authoritative_pending_confirmations = {'open_orders': 1}
    capture_report(owner)
    assert events[-1][1]['data']['observation_status'] == 'stale'
    assert events[-1][1]['data']['account_unavailable'] == ['open_orders']
    bot.freshness_ledger.begin_epoch()
    bot.freshness_ledger.stamp('open_orders', now_ms=NOW)
    capture_report(owner)
    assert events[-1][1]['data']['observation_status'] == 'current'
    assert events[-1][1]['data']['account_unavailable'] == []
    assert len(events) == 3
    capture_report(owner)
    assert len(events) == 3


@pytest.mark.parametrize('mode', ['coin', 'pside'])
def test_unused_disabled_position_quote_does_not_expire_evaluated_scopes(observed, mode):
    bot, owner, _, _ = observed(mode)
    other = 'DISABLED/USDT:USDT'
    bot.positions[other] = {'short': {'size': -1., 'price': 100.}}
    owner.quotes[other] = replace(quotes()[SYMBOL], symbol=other, fetched_ms=NOW-100_000)
    bot.c_mults[other] = 1.
    wave = capture_report(owner)
    assert len(wave.decisions) == 1 and not wave.unavailable
    assert wave.mark_observed_ms == (NOW-100,)
    payload = diagnostics.snapshot(bot, now_ms=NOW)
    assert payload['observation_status'] == 'current'
    assert payload['input_expires_at_ms'] == NOW+9800


def test_stale_green_cannot_survive_as_current_green_in_bounded_consumers(observed):
    from live.smoke_report import _risk_event_group, _risk_attention_rank, _summarize_hsl_status
    from tools.hsl_startup_preview import _bounded_hsl_data, _status_from_event
    bot, owner, _, events = observed('unified')
    wave = capture_report(owner, {SYMBOL: replace(quotes()[SYMBOL], bid=100., ask=100., last=100.)})
    clean = replace(wave, decisions=tuple(replace(d, reasons=()) for d in wave.decisions))
    assert clean.decisions[0].action == 'normal'
    bot._authoritative_pending_confirmations = {'open_orders': 1}
    diagnostics.record(bot, clean)
    event = dict(event_type='hsl.status', **{key: events[-1][1][key]
        for key in ('level', 'status', 'data')})
    assert event['status'] == 'degraded'
    assert event['data']['tier'] == 'stale'
    assert event['data']['scopes'][0]['tier'] == 'green'
    group = _risk_event_group(bot_key='fake/example', row={'ts': NOW, 'seq': 1},
                             live_event=event, path=Path('events.ndjson'), line_no=1)
    assert _risk_attention_rank(group) == 20
    assert _summarize_hsl_status({'one': group})['tier_counts'] == {'stale': 1}
    assert _status_from_event({'latest_data': _bounded_hsl_data(event)}) == 'stale'
    bot.freshness_ledger.begin_epoch()
    bot.freshness_ledger.stamp('open_orders', now_ms=NOW)
    diagnostics.record(bot, clean)
    assert events[-1][1]['data']['tier'] == 'green'


@pytest.mark.parametrize('slow_stage', ['projection', 'sink'])
def test_connector_admission_does_not_run_diagnostics_or_sinks(observed, monkeypatch, slow_stage):
    import utils
    bot, owner, wave, events = observed()
    clock = [NOW]
    monkeypatch.setattr(utils, 'utc_ms', lambda: clock[0])
    bot.get_exchange_time = lambda: clock[0]
    calls = []
    original = diagnostics._row if slow_stage == 'projection' else bot._emit_live_event
    def slow(*args, **kwargs):
        calls.append(slow_stage)
        clock[0] += 20_000
        return original(*args, **kwargs)
    if slow_stage == 'projection':
        monkeypatch.setattr(diagnostics, '_row', slow)
    else:
        bot._emit_live_event = slow
    bot._hsl_revised_diagnostic_event = None
    order = dict(symbol=SYMBOL, position_side='long')
    owner.bind(wave, (), (order,))
    assert owner.admit(order)
    assert not calls and clock == [NOW]
    # Neither planning captures nor admission run diagnostic work.
    fresh = owner.capture()
    assert not calls and clock == [NOW]
    owner.report(fresh)
    assert calls == [slow_stage]
    if slow_stage == 'projection':
        assert events[-1][1]['data']['observation_status'] == 'stale'
        assert events[-1][1]['status'] == 'degraded'



def test_inactive_scope_is_visible_in_console_and_fallback(observed, caplog):
    import logging
    from live.event_bus import LiveEvent, EventTypes, format_console_event
    bot, owner, _, _ = observed()
    bot.config['bot']['long']['risk']['n_positions'] = 0
    bot._emit_live_event = None
    with caplog.at_level(logging.INFO):
        capture_report(owner)
    event = LiveEvent(EventTypes.HSL_STATUS, data=diagnostics.snapshot(bot, now_ms=NOW))
    assert 'inactive=1' in format_console_event(event)
    assert any('inactive=1' in record.message for record in caplog.records)


def test_configured_console_sink_failure_keeps_unavailable_warning(observed, caplog):
    import logging
    from live.event_bus import LiveEventPipeline, LiveEvent, EventTypes, EventRoute
    bot, owner, _, _ = observed()
    class FailedConsole:
        def write(self, event):
            raise OSError('private-sink-detail')
    pipeline = LiveEventPipeline(console_sink=FailedConsole(),
        routes={EventTypes.HSL_STATUS: EventRoute(console=True, structured=False, monitor=False)})
    bot._live_event_pipeline = pipeline
    emitted = []
    def emit(event_type, **kwargs):
        result = pipeline.emit(LiveEvent(event_type, **kwargs))
        emitted.append(result)
        return result
    bot._emit_live_event = emit
    bot.positions[SYMBOL]['long']['price'] = float('nan')
    with caplog.at_level(logging.WARNING):
        capture_report(owner)
    assert emitted and emitted[0] is not None
    assert pipeline.sink_error_counters['console'] >= 1
    assert any('revised HSL' in record.message and 'unavailable=1' in record.message
               and record.levelno == logging.WARNING for record in caplog.records)
    assert 'private-sink-detail' not in caplog.text


@pytest.mark.asyncio
async def test_protective_wave_reports_even_when_scope_has_no_exit_work(observed):
    bot, owner, _, events = observed()
    bot.positions[SYMBOL]['long'].update(size=0., price=0.)
    assert not await owner.protect()
    result = diagnostics.snapshot(bot, now_ms=NOW)
    assert result['counts']['green'] == 1
    assert events[-1][1]['data']['counts']['green'] == 1


@pytest.mark.asyncio
async def test_failed_refresh_then_recovery_reports_expired_prior_wave(observed, monkeypatch):
    import asyncio
    import utils
    bot, owner, _, events = observed()
    clock = [NOW]
    monkeypatch.setattr(utils, 'utc_ms', lambda: clock[0])
    bot.get_exchange_time = lambda: clock[0]
    bot.positions[SYMBOL]['long'].update(size=0., price=0.)
    events.clear()
    capture_report(owner)
    assert len(events) == 1
    calls = []

    async def refresh(**kwargs):
        calls.append(clock[0])
        if len(calls) == 1:
            clock[0] += 20_000
            return False
        for surface in ('balance', 'positions', 'open_orders'):
            bot.freshness_ledger.stamp(surface, now_ms=clock[0])
        owner.quotes = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=clock[0])}
        return True

    async def sleep(delay, *, stage):
        if stage == 'revised_current_inputs':
            # Exercise the real skipped-wave branch, without manually reporting it.
            assert len(events) == 1
        else:
            bot.stop_signal_received = True
        await asyncio.sleep(0)

    async def ordinary():
        await asyncio.Event().wait()

    bot.stop_signal_received = False
    bot._begin_live_event_cycle = lambda **kwargs: None
    bot.refresh_protective_authoritative_state = refresh
    bot._sleep_unless_shutdown = sleep
    bot._maybe_log_health_summary = lambda: None
    bot.live_value = lambda key: .05
    owner.schedule_history = owner.schedule_sources = lambda: None
    owner._ordinary_plan = ordinary
    await owner.run()
    assert len(calls) == 2
    assert [e[1]['data']['observation_status'] for e in events] == ['current', 'stale', 'current']
    assert events[1][1]['status'] == 'degraded'
    assert all(e[1]['data']['counts']['green'] == 1 for e in events)
    capture_report(owner)
    assert len(events) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('slow_stage', ['projection', 'sink'])
async def test_ready_ordinary_plan_is_admitted_before_slow_reporting(observed, monkeypatch, slow_stage):
    import asyncio
    import utils
    bot, owner, _, _ = observed()
    clock = [NOW]
    monkeypatch.setattr(utils, 'utc_ms', lambda: clock[0])
    bot.get_exchange_time = lambda: clock[0]
    bot.positions[SYMBOL]['long'].update(size=0., price=0.)
    planned, admitted, report_order = [], [], []

    async def refresh(**kwargs):
        for surface in ('balance', 'positions', 'open_orders'):
            bot.freshness_ledger.stamp(surface, now_ms=clock[0])
        owner.quotes = {SYMBOL: replace(quotes()[SYMBOL], fetched_ms=clock[0])}
        return True

    async def ordinary():
        await refresh()
        wave = owner.capture()
        order = dict(symbol=SYMBOL, position_side='long')
        owner.bind(wave, (), (order,))
        planned.append(True)
        return (), (order,), None, wave

    async def execute(cancels, creates):
        admitted.append(owner.admit(creates[0]))
        bot.stop_signal_received = True

    async def sleep(*args, **kwargs):
        await asyncio.sleep(0)

    original = diagnostics._row if slow_stage == 'projection' else bot._emit_live_event
    def slow(*args, **kwargs):
        report_order.append((bool(planned), bool(admitted)))
        clock[0] += 20_000
        return original(*args, **kwargs)
    if slow_stage == 'projection':
        monkeypatch.setattr(diagnostics, '_row', slow)
    else:
        bot._emit_live_event = slow
    bot.stop_signal_received = False
    bot._begin_live_event_cycle = lambda **kwargs: None
    bot.refresh_protective_authoritative_state = refresh
    bot._sleep_unless_shutdown = sleep
    bot._maybe_log_health_summary = lambda: None
    bot.live_value = lambda key: .05
    bot.execute_order_plan_to_exchange = execute
    owner.schedule_history = owner.schedule_sources = lambda: None
    owner._ordinary_plan = ordinary
    await owner.run()
    assert admitted == [True]
    assert any(ready for ready, _ in report_order)
    assert all(written for ready, written in report_order if ready)
