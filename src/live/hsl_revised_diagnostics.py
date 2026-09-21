"""Passive, bounded views of revised HSL observations; never execution authority."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import logging

from live.diagnostic_safety import bounded_exception_type
from live.event_bus import EventTags, EventTypes
from live.event_emitters import _safe_emit, _console_sink_error_count

SCOPE_LIMIT = 128
SAMPLE_LIMIT = 3


def _account_unavailable(bot, now):
    ledger = getattr(bot, 'freshness_ledger', None)
    pending = getattr(bot, '_authoritative_pending_confirmations', {})
    max_age = bot._live_market_snapshot_max_age_ms()
    missing = []
    for name in ('balance', 'positions', 'open_orders'):
        state = getattr(ledger, 'surfaces', {}).get(name)
        if (state is None or state.updated_ms <= 0 or not 0 <= now - state.updated_ms <= max_age
                or int(pending.get(name, 0)) > state.epoch):
            missing.append(name)
    return missing


def _row(scope, *, action=None, reason=None, decision=None):
    native = json.loads(decision.payload)['decision'] if decision is not None else None
    estimates = list(decision.reasons) if decision is not None else []
    return dict(signal_mode=scope.mode, symbol=scope.symbol, pside=scope.pside,
                action=action, tier=(None if reason else 'inactive' if action is None
                                    else 'green' if action == 'normal' else 'red'),
                availability='unavailable' if reason else 'available', unavailable_reason=reason,
                estimated=bool(estimates), estimates=estimates[:32], omitted_estimates=max(0, len(estimates)-32),
                raw=None if native is None else native['raw'],
                ema=None if native is None else native['ema'],
                score=None if native is None else min(native['raw'], native['ema']),
                threshold=None if decision is None else decision.threshold,
                red_at=None if native is None else native['red_at'],
                flat_at=None if native is None else native['flat_at'])


def _priority(row):
    rank = 0 if row['tier'] == 'red' else 1 if row['availability'] == 'unavailable' else 2 if row['estimated'] else 3
    return rank, row['signal_mode'], row['pside'] or '', row['symbol'] or ''


def record(bot, wave):
    """Report a completed protective wave and emit bounded status changes."""
    try:
        from utils import utc_ms
        now = int(utc_ms())
        rows = [_row(d.scope, action=d.action, decision=d) for d in wave.decisions]
        rows += [_row(item.scope, reason=item.reason) for item in wave.unavailable]
        rows.sort(key=_priority)
        counts = Counter(row['tier'] or 'unavailable' for row in rows)
        counts['estimated'] = sum(row['estimated'] for row in rows)
        counts = {key: counts[key] for key in ('green', 'red', 'inactive', 'unavailable', 'estimated')}
        stamps = [wave.captured_ms, wave.position_observed_ms, *wave.mark_observed_ms]
        ledger = getattr(bot, 'freshness_ledger', None)
        stamps += [state.updated_ms for name, state in getattr(ledger, 'surfaces', {}).items()
                   if name in ('balance', 'open_orders')]
        observation = dict(engine='revised', schema_version=1,
            signal_mode=bot.config['live']['hsl_signal_mode'], captured_at_ms=wave.captured_ms,
            input_expires_at_ms=min(stamps) + int(bot._live_market_snapshot_max_age_ms()),
            account_generation=wave.generation, counts=counts, scope_count=len(rows),
            # Existing risk reports consume one aggregate tier. Keep RED visible
            # without fabricating a portfolio score from distinct coin signals.
            tier=('red' if counts['red'] else 'unavailable' if counts['unavailable']
                  else 'green' if counts['green'] else 'inactive'),
            scopes=rows[:SCOPE_LIMIT], omitted_scopes=max(0, len(rows)-SCOPE_LIMIT))
        # No runtime or executor consumer reads this attribute.
        bot._hsl_revised_diagnostic_observation = observation
        bot._hsl_revised_diagnostic_failed = False
        data = snapshot(bot, now_ms=now)
        signature = hashlib.sha256(json.dumps([data['observation_status'], data['account_unavailable'], [
            (r['signal_mode'], r['symbol'], r['pside'], r['action'], r['availability'],
             r['unavailable_reason'], r['estimates']) for r in rows]], sort_keys=True).encode()).hexdigest()
        previous = getattr(bot, '_hsl_revised_diagnostic_event', None)
        if previous is not None and previous[0] == signature:
            return
        bot._hsl_revised_diagnostic_event = (signature, now)
        data['scopes'] = data['scopes'][:SAMPLE_LIMIT]
        data['omitted_scopes'] = max(0, len(rows)-SAMPLE_LIMIT)
        console_errors_before = _console_sink_error_count(bot)
        emitted = _safe_emit(bot, EventTypes.HSL_STATUS, component='risk.hsl', tags=(EventTags.RISK, EventTags.SUMMARY),
            level='warning' if counts['unavailable'] else 'info',
            status='degraded' if counts['unavailable'] or counts['estimated'] else 'ok',
            cycle_id=getattr(bot, '_live_event_current_cycle_id', None), data=data)
        console_errors_after = _console_sink_error_count(bot)
        console_failed = (console_errors_before is not None and console_errors_after is not None
                          and console_errors_after > console_errors_before)
        if emitted is None or console_failed:
            logging.log(logging.WARNING if counts['unavailable'] else logging.INFO, '[risk] revised HSL | mode=%s observation=%s green=%d red=%d inactive=%d unavailable=%d estimated=%d',
                         data['signal_mode'], data['observation_status'], counts['green'], counts['red'],
                         counts['inactive'], counts['unavailable'], counts['estimated'])
    except Exception as exc:
        # Optional diagnostics must never inhibit or fabricate a trading decision.
        bot._hsl_revised_diagnostic_failed = True
        logging.debug('[risk] revised HSL diagnostic projection failed | error_type=%s', bounded_exception_type(exc))


def snapshot(bot, *, now_ms):
    """Read prior observations without creating owners, refreshing inputs or evaluating risk."""
    try:
        observed = getattr(bot, '_hsl_revised_diagnostic_observation', None)
        if observed is None:
            return dict(engine='revised', schema_version=1,
                        signal_mode=bot.config['live']['hsl_signal_mode'],
                        observation_status=('diagnostic_unavailable' if getattr(bot, '_hsl_revised_diagnostic_failed', False)
                                            else 'not_evaluated'), scopes=[], counts={}, scope_count=0, omitted_scopes=0)
        result = deepcopy(observed)
        missing = _account_unavailable(bot, now_ms)
        result['account_unavailable'] = missing
        age = now_ms - observed['captured_at_ms']
        stale = (age < 0 or now_ms > observed['input_expires_at_ms'] or bool(missing)
                 or observed['account_generation'] != int(getattr(bot, '_account_invalidation_generation', 0)))
        result['age_ms'] = max(0, age)
        result['observation_status'] = ('diagnostic_unavailable' if getattr(bot, '_hsl_revised_diagnostic_failed', False)
                                        else 'stale' if stale else 'current')
        # Older bounded report/preview consumers retain only the aggregate tier.
        # Keep last RED attention, but never present stale GREEN as current GREEN.
        if result['observation_status'] != 'current' and result['tier'] != 'red':
            result['tier'] = result['observation_status']
        return result
    except Exception as exc:
        logging.debug('[risk] revised HSL diagnostic read failed | error_type=%s', bounded_exception_type(exc))
        return dict(engine='revised', schema_version=1, observation_status='diagnostic_unavailable',
                    scopes=[], counts={}, scope_count=0, omitted_scopes=0)
