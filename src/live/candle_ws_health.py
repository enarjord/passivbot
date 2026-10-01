"""Bounded passive presentation for shared candle receive failures.

Only the caller's CCXT NetworkError receive branch uses this observer. It owns
no reconnect tasks and never changes retry delays or candle ingestion.
"""
import logging
import time

from live.console_health import token
from live.diagnostic_safety import bounded_exception_type
from live.event_bus import EventTypes, emit_event


def observe_receive_status(bot, symbol, *, recovered=False, retired=False, error=None, retry_s=0.):
    try:
        state = getattr(bot, '_console_candle_receive', None)
        if (recovered or retired) and (state is None or symbol not in state['active']):
            return
        now = time.monotonic()
        if state is None:
            state = dict(active=set(), sample=[], failures=0, started=now, last_warning=now,
                         peak=0, overflow=False)
            bot._console_candle_receive = state
        if recovered or retired:
            state['active'].discard(symbol)
            status = 'retired' if retired else 'recovered'
        else:
            status = 'failed'
            state['failures'] = min(999999, state['failures'] + 1)
            if len(state['active']) < 256:
                state['active'].add(symbol)
            elif symbol not in state['active']:
                state['overflow'] = True
            if symbol not in state['sample'] and len(state['sample']) < 3:
                state['sample'].append(symbol)
            state['peak'] = max(state['peak'], len(state['active']))
        emit_event(bot, dict(event_type=EventTypes.CANDLE_WEBSOCKET_STATUS,
            level='debug', component='candle_ws', tags=['candle', 'websocket'], symbol=symbol,
            status='skipped' if retired else status, data=dict(receive_status=status, stage='receive', retry_s=retry_s,
                **({'error_type': bounded_exception_type(error)} if error is not None else {}))))
        sample = ','.join(token(item.split('/')[0], 12) for item in state['sample'])
        if status == 'failed' and (state['failures'] == 1 or now - state['last_warning'] >= 300.):
            state['last_warning'] = now
            logging.warning('[candle] websocket receive failures symbols=%d sample=%s '
                            'failures=%d error_type=%s action=rest_fallback',
                            len(state['active']), sample, state['failures'],
                            token(bounded_exception_type(error), 32))
        elif not state['active']:
            bot._console_candle_receive = None
            # With overflow, the tracked set cannot prove every symbol recovered.
            logging.info('[candle] websocket %s symbols=%d failures=%d wait=%.1fs%s',
                         'watchers cleared' if retired else 'receive recovered',
                         state['peak'], state['failures'], max(0., now - state['started']),
                         ' coverage=partial' if state['overflow'] else '')
    except Exception as exc:
        logging.debug('[candle] receive presentation unavailable | error_type=%s',
                      bounded_exception_type(exc))
