"""Operator health/recovery contracts, entirely offline."""
import logging
from types import SimpleNamespace

from live.console_health import readiness_payload, log_trailing_recovery
from live.event_bus import format_periodic_health_summary, split_health_console
from live.freshness import FreshnessLedger


def test_health_does_not_infer_waiting_from_absent_orders():
    ledger = FreshnessLedger(now_ms=10_000)
    for name in ('positions', 'balance', 'open_orders'):
        ledger.stamp(name, now_ms=9_000)
    bot = SimpleNamespace(user='example', freshness_ledger=ledger,
        positions={symbol: {'long': {'size': 1}, 'short': {'size': 0}}
                   for symbol in ('REST', 'WAIT', 'BLOCK', 'UNKNOWN')},
        open_orders={'REST': [dict(symbol='REST', position_side='long', side='sell')]},
        _orchestrator_trailing_unavailable_reasons={'BLOCK': ['missing_candles']},
        _build_trailing_status_items=lambda: [dict(symbol='WAIT', pside='long', kind='close',
            payload=dict(status='waiting_threshold'))])
    result = readiness_payload(bot, 10_000)
    assert result['close_coverage'] == dict(resting=1, waiting=1, blocked=1, unknown=1)
    assert result['account_age_ms'] == 1000
    bot._authoritative_pending_confirmations = {'positions': ledger.epoch + 1}
    result = readiness_payload(bot, 10_000)
    assert result['account_age_ms'] is None
    assert result['account_pending'] == ['positions']


def test_recovery_is_immediate_once_and_flat_is_not_called_ready(caplog):
    bot = SimpleNamespace(positions={'BTC': {'long': {'size': 1}}})
    with caplog.at_level(logging.INFO):
        log_trailing_recovery(bot, {'BTC': ['pending'], 'ETH': ['candles']}, 1000)
        log_trailing_recovery(bot, {'BTC': ['candles']}, 3000)
        log_trailing_recovery(bot, {}, 5000)
        log_trailing_recovery(bot, {}, 6000)
    messages = [r.message for r in caplog.records]
    assert len(messages) == 2
    assert 'blocker cleared symbol=ETH' in messages[0]
    assert 'inputs recovered symbol=BTC wait=4.0s' in messages[1]
    assert 'trailing_evaluation_resumed' in messages[1]


def test_recovery_sink_failure_isolated_and_retried(monkeypatch):
    bot = SimpleNamespace(positions={'BTC': {'long': {'size': 1}}})
    log_trailing_recovery(bot, {'BTC': ['pending']}, 1000)
    def fail(*args):
        raise OSError('sink failed')
    monkeypatch.setattr(logging, 'info', fail)
    log_trailing_recovery(bot, {}, 3000)
    assert 'BTC' in bot._console_trailing_blocked_since


def test_heartbeat_labels_unknowns_and_preserves_anomalies_with_bounded_rows():
    data = dict(bot_label='example', uptime_ms=86400000, last_loop_duration_ms=26000,
        positions_long=10, positions_short=2, open_order_count=4,
        close_coverage=dict(resting=4, waiting=6, blocked=1, unknown=1),
        account_age_ms=None, account_pending=['positions'], cpu_percent=88.,
        errors_last_hour=1, ws_reconnects=3, health_summary_lag_ms=71000,
        event_queue_depth=8, event_queue_maxsize=128, event_dropped_total=2,
        event_sink_error_total=1, event_pipeline_worker_alive=False)
    message = format_periodic_health_summary(data)
    for expected in ('last_loop=26.0s', 'open_orders=4', 'account_age=?',
                     'summary_late=71.0s', 'ws_reconnects_total=3', 'errors_1h=1/10',
                     'event_drop=2', 'sink_err=1', 'event_worker=dead'):
        assert expected in message
    assert 'ord=+' not in message and 'lag=' not in message
    lines = split_health_console(message)
    assert all(len('2026-01-01T00:00:00Z WARNING  [hyperliquid] ' + line) <= 240 for line in lines)
    assert 'event_worker=dead' in lines[-1]


def test_ws_console_echo_classification_never_hides_fill_progress_or_mutates_batch(monkeypatch):
    from live.console_health import ws_presentation_self_echo
    import utils
    monkeypatch.setattr(utils, 'utc_ms', lambda: 1_000_000)
    order = dict(id='one', symbol='BTC', side='buy', price=100, qty=1,
                 status='open', filled=0, remaining=1, amount=1,
                 _pb_order_update_requires_authoritative_refresh=True)
    recent = [{**order, 'execution_timestamp': 999_000}]
    bot = SimpleNamespace(recent_order_executions=recent)
    assert ws_presentation_self_echo(bot, [order]) is True
    assert order['_pb_order_update_requires_authoritative_refresh'] is True
    assert bot.recent_order_executions is recent
    assert ws_presentation_self_echo(bot, [{**order, 'filled': .25}]) is False
    assert ws_presentation_self_echo(bot, [{**order, 'remaining': .75}]) is False
    assert ws_presentation_self_echo(bot, [{**order, 'filled': 'invalid'}]) is False
    assert ws_presentation_self_echo(bot, [{**order, 'price': 101}]) is False
    assert ws_presentation_self_echo(bot, [{**order, 'id': 'external'}]) is False


def test_normalized_short_close_is_resting_without_reduce_only_flag():
    bot = SimpleNamespace(positions={'BTC': {'short': {'size': -1}}},
        open_orders={'BTC': [dict(symbol='BTC', position_side='short', side='buy')]})
    assert readiness_payload(bot, 1000)['close_coverage'] == dict(
        resting=1, waiting=0, blocked=0, unknown=0)
