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
               if order.get('reduce_only') is True}
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
    return dict(bot_label=token(getattr(bot, 'user', None)), open_order_count=len(orders),
                close_coverage=counts, account_pending=missing,
                account_age_ms=max(ages) if len(ages) == 3 else None)


def log_trailing_recovery(bot, unavailable, now_ms):
    """Pair existing input warnings with immediate, scoped recovery/clear notices."""
    try:
        prior = getattr(bot, '_console_trailing_blocked_since', {})
        for symbol, since in list(prior.items()):
            if symbol in unavailable:
                continue
            held = any(side.get('size', 0) for side in
                       getattr(bot, 'positions', {}).get(symbol, {}).values())
            logging.info('[trailing] %s symbol=%s wait=%.1fs action=%s',
                         'inputs recovered' if held else 'blocker cleared', token(symbol),
                         max(0, now_ms - since) / 1000.,
                         'trailing_evaluation_resumed' if held else 'position_no_longer_held')
            del prior[symbol]
        for symbol in sorted(unavailable):
            if symbol not in prior:
                if len(prior) >= 256:
                    prior.pop(next(iter(prior)))
                prior[symbol] = now_ms
        bot._console_trailing_blocked_since = prior
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
