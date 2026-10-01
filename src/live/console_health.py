"""Passive console readiness views. Never fetch inputs or change execution state."""
import logging
import re


def token(value, limit=32):
    return re.sub(r"[^a-zA-Z0-9_./:-]", "_", str(value or "unknown"))[:limit]


def readiness_payload(bot, now_ms):
    """Build only for the periodic console heartbeat, using existing diagnostics."""
    held = {(symbol, side) for symbol, sides in getattr(bot, 'positions', {}).items()
            for side in ('long', 'short') if sides.get(side, {}).get('size', 0)}
    orders = [order for rows in getattr(bot, 'open_orders', {}).values() for order in rows]
    resting = {(order.get('symbol'), order.get('position_side')) for order in orders
               if (order.get('reduce_only') is True
                   or (order.get('position_side'), order.get('side'))
                   in (('long', 'sell'), ('short', 'buy')))}
    reasons = getattr(bot, '_orchestrator_trailing_unavailable_reasons', {}) or {}
    unavailable_sides = getattr(bot, '_orchestrator_trailing_unavailable_psides', {}) or {}
    blocked = {(symbol, side) for symbol, side in held if reasons.get(symbol)
               and (not unavailable_sides.get(symbol) or side in unavailable_sides[symbol])}
    # Reuse the native strategy diagnostic; never infer "waiting" from zero orders.
    build = getattr(bot, '_build_trailing_status_items', None)
    items = build() if callable(build) else []
    waiting = {(item.get('symbol'), item.get('pside')) for item in items
               if item.get('kind') == 'close' and item.get('payload', {}).get('status')
               in ('waiting_threshold', 'waiting_retracement')}
    counts = dict(resting=0, waiting=0, blocked=0, unknown=0)
    for pair in held:
        category = ('blocked' if pair in blocked else 'resting' if pair in resting else
                    'waiting' if pair in waiting else 'unknown')
        counts[category] += 1
    ledger = getattr(getattr(bot, 'freshness_ledger', None), 'surfaces', {})
    pending = getattr(bot, '_authoritative_pending_confirmations', {}) or {}
    ages, missing = [], []
    for name in ('balance', 'positions', 'open_orders'):
        state = ledger.get(name)
        if (state is None or state.updated_ms <= 0 or now_ms < state.updated_ms
                or pending.get(name, 0) > state.epoch):
            missing.append(name)
        else:
            ages.append(now_ms - state.updated_ms)
    waits = getattr(bot, '_console_trailing_waits', {}) or {}
    active = [(symbol, row) for symbol, row in waits.items()
              if 'recovered_ms' not in row and any(pair[0] == symbol for pair in held)]
    active.sort(key=lambda item: item[1]['since_ms'])
    details = [dict(symbol=token(symbol, 48), phase=row['phase'],
                    age_ms=max(0, now_ms - row['since_ms']))
               for symbol, row in active[:3]]
    surface_ages = {name: max(0, now_ms - state.updated_ms)
                   for name in ('balance', 'positions', 'open_orders', 'fills')
                   if (state := ledger.get(name)) is not None
                   and 0 < state.updated_ms <= now_ms}
    writes = [row.get('execution_timestamp') for rows in
              (getattr(bot, 'recent_order_executions', ()),
               getattr(bot, 'recent_order_cancellations', ())) for row in rows[-256:]]
    latest_write = max((stamp for stamp in writes if isinstance(stamp, (int, float))
                        and 0 < stamp <= now_ms), default=None)
    completed = getattr(bot, '_console_last_cycle_completed_ms', None)
    owner = getattr(bot, '_hsl_revised_live', None)
    ordinary = getattr(owner, '_ordinary', None)
    ordinary_started = getattr(owner, '_ordinary_started_ms', None)
    ordinary_pending_age = (max(0, now_ms - ordinary_started)
        if ordinary is not None and not ordinary.done() and ordinary_started is not None else None)
    return dict(trailing_wait_count=len(active), trailing_wait_samples=details,
                trailing_wait_max_ms=max((row['age_ms'] for row in details), default=None),
                ordinary_pending_age_ms=ordinary_pending_age,
                account_surface_ages_ms=surface_ages,
                fills_pending=bool(pending.get('fills', 0) > getattr(ledger.get('fills'), 'epoch', 0)),
                last_cycle_completed_age_ms=(now_ms - completed if completed is not None
                                              and 0 < completed <= now_ms else None),
                last_write_age_ms=(now_ms - latest_write if latest_write is not None else None),
                bot_label=token(getattr(bot, 'user', None)), open_order_count=len(orders),
                close_coverage=counts, account_pending=missing,
                account_age_ms=max(ages) if len(ages) == 3 else None)


def _trailing_phase(reasons):
    if 'position_fill_confirmation_pending' in reasons:
        return 'fill_confirmation'
    if any('candle' in reason for reason in reasons):
        return 'candle_input'
    return 'other_input'


def log_trailing_recovery(bot, unavailable, now_ms):
    """Measure observed blocker phases, without profiling or controlling their work.

    Elapsed time is attributed to the last observed phase until the next poll.
    It includes polling, finalization and retrieval; it is not network-only time.
    """
    try:
        prior = getattr(bot, '_console_trailing_blocked_since', {})
        waits = getattr(bot, '_console_trailing_waits', {})
        bot._console_trailing_waits = waits
        bot._console_trailing_blocked_since = prior
        for symbol, row in list(waits.items()):
            if 'recovered_ms' not in row:
                elapsed = max(0, now_ms - row['observed_ms'])
                row['phase_ms'][row['phase']] += elapsed
                row['observed_ms'] = now_ms
                if symbol in unavailable:
                    row['phase'] = _trailing_phase(unavailable[symbol])
                    row['polls'] = min(999999, row['polls'] + 1)
                    continue
                row['recovered_ms'] = now_ms
            held = any(side.get('size', 0) for side in
                       getattr(bot, 'positions', {}).get(symbol, {}).values())
            logging.info(
                '[trailing] %s symbol=%s wait=%.1fs fill=%.1fs candles=%.1fs other=%.1fs polls=%d action=%s',
                'inputs recovered' if held else 'blocker cleared', token(symbol),
                max(0, row['recovered_ms'] - row['since_ms']) / 1000.,
                row['phase_ms']['fill_confirmation'] / 1000.,
                row['phase_ms']['candle_input'] / 1000., row['phase_ms']['other_input'] / 1000.,
                row['polls'], 'trailing_evaluation_resumed' if held else 'position_no_longer_held')
            del waits[symbol]
            prior.pop(symbol, None)
        for symbol in sorted(unavailable):
            if symbol not in waits:
                if len(waits) >= 256:
                    evicted = next(iter(waits))
                    waits.pop(evicted)
                    prior.pop(evicted, None)
                waits[symbol] = dict(since_ms=now_ms, observed_ms=now_ms,
                                    phase=_trailing_phase(unavailable[symbol]), polls=1,
                                    phase_ms=dict(fill_confirmation=0, candle_input=0, other_input=0))
                prior[symbol] = now_ms
    except Exception as exc:
        logging.debug('[trailing] recovery presentation failed | error_type=%s', type(exc).__name__)


def ws_presentation_self_echo(bot, orders):
    """Recognize exact local write echoes without pruning execution-owned caches."""
    try:
        from utils import utc_ms
        cutoff = utc_ms() - 180_000
        if not orders:
            return False
        for order in orders:
            if not isinstance(order, dict) or not order.get('id'):
                return False
            if float(order.get('filled') or 0) > 0:
                return False
            remaining = order.get('remaining')
            amount = order.get('amount', order.get('qty'))
            if remaining is not None and (amount is None or float(remaining) < abs(float(amount))):
                return False
            status = str(order.get('status', '')).lower()
            if status == 'open':
                recent = getattr(bot, 'recent_order_executions', ())
            elif status in ('canceled', 'cancelled'):
                recent = getattr(bot, 'recent_order_cancellations', ())
            else:
                return False
            if not any(row.get('id') == order['id'] and row.get('execution_timestamp', 0) > cutoff
                       and all(order.get(key) is not None and order.get(key) == row.get(key)
                               for key in ('symbol', 'side', 'price', 'qty'))
                       for row in recent[-256:]):
                return False
        return True
    except Exception:
        # Optional presentation evidence: an unclassifiable update stays visible.
        return False
