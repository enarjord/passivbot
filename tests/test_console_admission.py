"""Offline console replay: scope transitions survive while detail churn stays durable."""
from dataclasses import replace

import pytest

from live.console_admission import ConsoleAdmission
from live.event_bus import ConsoleSummarySink, EventTypes, ListEventSink, LiveEvent, LiveEventPipeline


class Clock:
    now = 0.0
    def __call__(self):
        return self.now


class Logger:
    def __init__(self):
        self.lines = []
        self.fail = False
    def log(self, level, message):
        if self.fail:
            raise OSError('sink unavailable')
        self.lines.append((level, message))


def event(**overrides):
    data = dict(engine='revised', signal_mode='coin', observation_status='current',
                account_unavailable=[], console_state='a' * 64,
                counts=dict(green=2, red=0, inactive=0, unavailable=0, estimated=2),
                scopes=[dict(symbol='BTC/USDT:USDT', pside='long', estimates=['snapshot_skew'])])
    data.update(overrides.pop('data', {}))
    return LiveEvent(EventTypes.HSL_STATUS, status='degraded', data=data, **overrides)


def test_estimation_churn_is_durable_and_reminders_are_bounded():
    clock, logger, durable = Clock(), Logger(), ListEventSink()
    sink = ConsoleSummarySink(logger, admission=ConsoleAdmission(clock=clock))
    pipeline = LiveEventPipeline(console_sink=sink, structured_sinks=[durable])
    try:
        for n in range(201):
            clock.now = n * 3
            pipeline.emit(event(cycle_id=f'cy_{n}', data={
                'scopes': [dict(estimates=['snapshot_skew' if n % 2 else 'prices_before_mark'])]}))
        assert pipeline.flush()
        assert len(durable.events) == 201
        assert len(logger.lines) == 3  # initial, 5m, 10m
        assert 'repeats=100 over=300s' in logger.lines[1][1]
        assert 'repeats=100 over=300s' in logger.lines[2][1]
        assert all('cycle=' not in line for _, line in logger.lines)
        assert all(len('2026-09-25T00:00:00Z WARNING  [hyperliquid] ' + line) <= 240
                   for _, line in logger.lines)
        assert pipeline.health_snapshot()['event_dropped_total'] == 0
    finally:
        pipeline.close()


@pytest.mark.parametrize('change', [
    {'console_state': 'b' * 64},  # a different affected scope, even with equal counts
    {'observation_status': 'stale'},
    {'account_unavailable': ['positions']},
])
def test_transitions_and_recoveries_bypass_reminder_window(change):
    logger = Logger()
    sink = ConsoleSummarySink(logger)
    sink.write(event())
    sink.write(event(data=change))
    sink.write(event())
    assert len(logger.lines) == 3


def test_severity_and_bot_identity_do_not_share_suppression():
    logger = Logger()
    sink = ConsoleSummarySink(logger)
    sink.write(event(bot_id='one'))
    sink.write(event(bot_id='one', level='warning'))
    sink.write(event(bot_id='two', level='warning'))
    assert len(logger.lines) == 3


@pytest.mark.parametrize('data', [
    {'console_state': None}, {'console_state': 'old-schema'},
    {'account_unavailable': [{}]}, {'observation_status': 'future-state'},
])
def test_unclassifiable_state_stays_visible(data):
    logger = Logger()
    sink = ConsoleSummarySink(logger)
    sink.write(event(data=data))
    sink.write(event(data=data))
    assert len(logger.lines) == 2


def test_failed_delivery_does_not_consume_transition_or_reminder():
    clock, logger = Clock(), Logger()
    sink = ConsoleSummarySink(logger, admission=ConsoleAdmission(clock=clock))
    for timestamp in [0, 300]:
        clock.now = timestamp
        logger.fail = True
        with pytest.raises(OSError):
            sink.write(event())
        logger.fail = False
        sink.write(event())
    assert len(logger.lines) == 2


def test_scope_storage_is_bounded_and_eviction_and_restart_reemit():
    clock, logger = Clock(), Logger()
    sink = ConsoleSummarySink(logger, admission=ConsoleAdmission(clock=clock, capacity=2))
    for bot in ['one', 'two', 'three', 'one']:
        sink.write(event(bot_id=bot))
    assert len(sink.admission._states) == 2
    assert len(logger.lines) == 4
    ConsoleSummarySink(logger).write(event(bot_id='one'))
    assert len(logger.lines) == 5


@pytest.mark.parametrize('kind', [EventTypes.FILL_INGESTED, EventTypes.EXECUTION_CREATE_FAILED,
                                  EventTypes.HSL_TRANSITION, EventTypes.RISK_INPUT_STATUS])
def test_action_and_numbered_recovery_events_are_never_coalesced(kind):
    logger = Logger()
    sink = ConsoleSummarySink(logger)
    other = replace(event(), event_type=kind)
    sink.write(other)
    sink.write(other)
    assert len(logger.lines) == 2


def test_healthy_state_has_no_extra_periodic_reminder():
    clock, logger = Clock(), Logger()
    sink = ConsoleSummarySink(logger, admission=ConsoleAdmission(clock=clock))
    healthy = replace(event(data={'counts': dict(green=2, red=0, inactive=0, unavailable=0, estimated=0)}), status='succeeded')
    sink.write(healthy)
    clock.now = 3600
    sink.write(healthy)
    assert len(logger.lines) == 1


def test_missing_account_surface_and_record_budget_are_visible():
    from live.event_bus import format_console_event
    rendered = format_console_event(event(data={
        'observation_status': 'stale', 'account_unavailable': ['positions', 'open_orders'],
        'scopes': [dict(symbol='X' * 512, pside='long', unavailable_reason='Y' * 512)],
    }))
    assert 'account=positions,open_orders' in rendered
    assert len('2026-09-25T00:00:00Z WARNING  [hyperliquid] ' + rendered) <= 240


def test_repeat_is_admitted_exactly_at_boundary():
    clock, logger = Clock(), Logger()
    sink = ConsoleSummarySink(logger, admission=ConsoleAdmission(clock=clock))
    sink.write(event())
    clock.now = 299.999
    assert sink.write(event()) is None
    clock.now = 300
    assert 'repeats=2 over=300s' in sink.write(event())
