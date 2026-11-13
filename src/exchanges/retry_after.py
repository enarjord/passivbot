"""Request-local Retry-After transport for live exchange clients."""

from contextvars import ContextVar
from datetime import timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from math import isfinite
import re
from time import time

from ccxt.async_support.base.exchange import Exchange
from ccxt.base.errors import InvalidNonce, NetworkError


_SHORT_DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_LONG_DAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_CLOCK = r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
_HTTP_DATE = re.compile(
    rf"{_SHORT_DAY}, [0-9]{{2}} {_MONTH} [0-9]{{4}} {_CLOCK} GMT"
    rf"|{_LONG_DAY}, [0-9]{{2}}-{_MONTH}-[0-9]{{2}} {_CLOCK} GMT"
    rf"|{_SHORT_DAY} {_MONTH} (?: [0-9]|[0-9]{{2}}) {_CLOCK} [0-9]{{4}}"
)


def _http_date(value):
    if not value:
        return None
    value = value.strip()
    # CCXT joins repeated response fields. Match one complete HTTP-date before
    # using the email parser, which otherwise accepts trailing input.
    if not _HTTP_DATE.fullmatch(value):
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def retry_after_from_headers(headers):
    """Return positive delta seconds or the interval between response dates."""
    retry_after = None
    response_date = None
    for key, value in headers.items():
        name = key.lower()
        if name == "retry-after":
            if retry_after is not None:
                return 0.0
            retry_after = value.strip()
        elif name == "date":
            response_date = "" if response_date is not None else value
    if not retry_after:
        return 0.0
    if retry_after.isascii() and retry_after.isdecimal():
        seconds = float(retry_after)
        return seconds if isfinite(seconds) else 0.0
    deadline = _http_date(retry_after)
    if deadline is None:
        return 0.0
    # The server's Date shares the deadline's clock. Use the local wall clock
    # when the response provides no usable Date; scheduling itself is monotonic.
    received_at = _http_date(response_date)
    return max(0.0, deadline - (time() if received_at is None else received_at))


class _SkipCcxtRetry(Exception):
    """Unwrap at fetch2, outside CCXT's optional OperationFailed retry loop."""

    def __init__(self, error):
        self.error = error


def install_ccxt_retry_after(client):
    if not isinstance(client, Exchange) or getattr(client, "_pb_retry_after_installed", False):
        return

    fetch = client.fetch
    fetch2 = client.fetch2
    on_rest_response = client.on_rest_response
    response_delay = ContextVar("ccxt_response_retry_after", default=0.0)
    inside_fetch2 = ContextVar("ccxt_retry_after_fetch2", default=False)

    @wraps(on_rest_response)
    def capture_response(code, reason, url, method, headers, body, request_headers, request_body):
        response_delay.set(retry_after_from_headers(headers) if code >= 400 else 0.0)
        return on_rest_response(code, reason, url, method, headers, body, request_headers, request_body)

    @wraps(fetch)
    async def fetch_with_retry_after(*args, **kwargs):
        token = response_delay.set(0.0)
        try:
            return await fetch(*args, **kwargs)
        except InvalidNonce:
            raise
        except NetworkError as exc:
            if delay := response_delay.get():
                exc.retry_after_seconds = delay
                if inside_fetch2.get():
                    raise _SkipCcxtRetry(exc) from None
            raise
        finally:
            response_delay.reset(token)

    @wraps(fetch2)
    async def fetch2_without_early_retry(*args, **kwargs):
        token = inside_fetch2.set(True)
        try:
            return await fetch2(*args, **kwargs)
        except _SkipCcxtRetry as stopped:
            failure = stopped.error
        finally:
            inside_fetch2.reset(token)
        raise failure

    client.on_rest_response = capture_response
    client.fetch = fetch_with_retry_after
    client.fetch2 = fetch2_without_early_retry
    client._pb_retry_after_installed = True
