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
        wave = owner.capture(quotes())
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
    owner.capture({SYMBOL: replace(quotes()[SYMBOL], bid=80., ask=80., last=80.)})
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
    assert owner.capture(quotes()).permission(SYMBOL, 'long') == wave.permission(SYMBOL, 'long')
    monkeypatch.setattr(diagnostics, '_row', failure)
    assert owner.capture(quotes()).permission(SYMBOL, 'long') == wave.permission(SYMBOL, 'long')
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
    owner.capture()
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
