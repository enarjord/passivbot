"""Bounded passive presentation for shared candle receive failures.

Only the caller's CCXT NetworkError receive branch uses this observer. It owns
no reconnect tasks and never changes retry delays or candle ingestion.
"""
import logging
import time

from ccxt.base.errors import NetworkError

from live.console_health import token
from live.diagnostic_safety import bounded_exception_type
from live.event_bus import EventTypes, emit_event


class MalformedCandleFallback(NetworkError):
    """A receive fallback already counted by a connector's malformed-candle summary."""


def _emit_status(bot, symbol, status, *, error=None, retry_s=0., generation=None):
    data = dict(
        receive_status=status, stage='receive', retry_s=retry_s,
        **({'watcher_generation': generation} if generation is not None else {}),
        **({'error_type': bounded_exception_type(error)} if error is not None else {}),
    )
    if isinstance(error, MalformedCandleFallback):
        data['receive_cause'] = 'malformed_kline'
    emit_event(bot, dict(event_type=EventTypes.CANDLE_WEBSOCKET_STATUS,
        level='debug', component='candle_ws', tags=['candle', 'websocket'], symbol=symbol,
        status='skipped' if status == 'retired' else status, data=data))


def observe_receive_status(bot, symbol, *, recovered=False, retired=False, error=None, retry_s=0.,
                           previously_failed=False, generation=None):
    try:
        state = getattr(bot, '_console_candle_receive', None)
        if (recovered or retired) and (state is None or symbol not in state['active']
                                      or state['active'][symbol] != generation):
            # The watcher's one-bit receive history is independent of the bounded
            # console aggregate, which may have overflowed or already cleared.
            if previously_failed:
                _emit_status(bot, symbol, 'retired' if retired else 'recovered', generation=generation)
            return
        now = time.monotonic()
        if state is None:
            state = dict(active={}, sample=[], failures=0, started=now, last_warning=None,
                         first_malformed_warning=None, peak=0, overflow=False)
            bot._console_candle_receive = state
        if recovered or retired:
            state['active'].pop(symbol, None)
            status = 'retired' if retired else 'recovered'
        else:
            status = 'failed'
            state['failures'] = min(999999, state['failures'] + 1)
            if len(state['active']) < 256 or symbol in state['active']:
                state['active'][symbol] = generation
            elif symbol not in state['active']:
                state['overflow'] = True
            if symbol not in state['sample'] and len(state['sample']) < 3:
                state['sample'].append(symbol)
            state['peak'] = max(state['peak'], len(state['active']))
        _emit_status(bot, symbol, status, error=error, retry_s=retry_s, generation=generation)
        sample = ','.join(token(item.split('/')[0], 12) for item in state['sample'])
        malformed = isinstance(error, MalformedCandleFallback)
        warning_clock = 'first_malformed_warning' if malformed else 'last_warning'
        last = state[warning_clock]
        # The connector continues to summarize malformed drops and fallback requests.
        if status == 'failed' and (last is None or (not malformed and now - last >= 300.)):
            logging.warning('[candle] websocket receive failures symbols=%d sample=%s '
                            'failures=%d error_type=%s%s action=rest_fallback',
                            len(state['active']), sample, state['failures'],
                            token(bounded_exception_type(error), 32),
                            ' cause=malformed_kline' if malformed else '')
            state[warning_clock] = now
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
