import asyncio
from types import SimpleNamespace

import ccxt.async_support as ccxt
import pytest
from ccxt.base.errors import AuthenticationError, DDoSProtection, RateLimitExceeded, RequestTimeout

from exchanges import retry_after


class RestResponse:
    def __init__(self, status, headers, body="{}", *, read_body=None):
        self.status = status
        self.headers = headers
        self.reason = "test response"
        self.body = body
        self.read_body = read_body

    async def text(self, **kwargs):
        if self.read_body is not None:
            await self.read_body()
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def client(monkeypatch):
    def build(respond, *, retries=0, exchange_class=ccxt.binanceusdm):
        exchange = exchange_class({
            "enableRateLimit": False,
            "options": {"maxRetriesOnFailure": retries},
        })
        monkeypatch.setattr(exchange, "open", lambda: None)
        monkeypatch.setattr(exchange, "session", SimpleNamespace(get=respond, headers={}))
        retry_after.install_ccxt_retry_after(exchange)
        return exchange

    return build


@pytest.mark.parametrize(("value", "seconds"), [
    ("00090", 90.0), (" 35 ", 35.0), ("0", 0.0),
    ("-1", 0.0), ("0.5", 0.0), ("inf", 0.0),
    ("1e100", 0.0), ("500, 900", 0.0), ("9" * 400, 0.0),
    ("not a date", 0.0),
])
def test_retry_after_accepts_http_delta_seconds(value, seconds):
    assert retry_after.retry_after_from_headers({"rEtRy-AfTeR": value}) == seconds


def test_retry_after_date_uses_response_clock_and_falls_back_to_local_clock(monkeypatch):
    # 2015-10-21 07:28:00 UTC. The first response clock is five minutes ahead.
    monkeypatch.setattr(retry_after, "time", lambda: 1_445_412_480.0)
    headers = {"Date": "Wed, 21 Oct 2015 07:33:00 GMT",
               "Retry-After": "Wed, 21 Oct 2015 07:34:00 GMT"}
    assert retry_after.retry_after_from_headers(headers) == 60.0
    headers["Date"] = "malformed"
    assert retry_after.retry_after_from_headers(headers) == 360.0
    headers["Retry-After"] = "Wed, 21 Oct 2015 07:27:00 GMT"
    assert retry_after.retry_after_from_headers(headers) == 0.0
    headers["retry-after"] = "1"
    assert retry_after.retry_after_from_headers(headers) == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("exchange_class", "error_type"), [
    (ccxt.binanceusdm, DDoSProtection),
    (ccxt.bybit, RateLimitExceeded),
])
async def test_ccxt_request_propagates_http_retry_after_without_inline_retry(
    client, exchange_class, error_type,
):
    calls = []

    def respond(url, **kwargs):
        calls.append(url)
        return RestResponse(429, {"Retry-After": "75"})

    exchange = client(respond, retries=2, exchange_class=exchange_class)
    # Binance raises in handle_errors and Bybit falls through to the base HTTP
    # status mapping for this body. Both use the real CCXT request and fetch2.
    request = (
        exchange.fapiPublicGetTime if exchange_class is ccxt.binanceusdm
        else exchange.publicGetV5MarketTime
    )
    with pytest.raises(error_type) as error:
        await request()

    assert error.value.retry_after_seconds == 75.0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ccxt_ordinary_network_retry_is_unchanged(client):
    calls = []

    def respond(url, **kwargs):
        calls.append(url)
        return RestResponse(429 if len(calls) == 1 else 200, {})

    assert await client(respond, retries=1).fapiPublicGetTime() == {}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_ccxt_retry_after_is_request_local_on_a_shared_client(client, monkeypatch):
    first_request_open = asyncio.Event()
    second_request_done = asyncio.Event()

    async def timeout_after_sibling_response():
        first_request_open.set()
        await second_request_done.wait()
        raise TimeoutError

    def respond(url, **kwargs):
        if url.path == "/slow":
            return RestResponse(200, {}, read_body=timeout_after_sibling_response)
        return RestResponse(429, {"Retry-After": "90", "Server": "private test header"})

    exchange = client(respond)
    original_hook = exchange.on_rest_response
    retry_after.install_ccxt_retry_after(exchange)
    assert exchange.on_rest_response is original_hook
    monkeypatch.setattr(exchange, "enableLastResponseHeaders", True)
    pending = asyncio.create_task(exchange.fetch("https://offline.invalid/slow"))
    await asyncio.wait_for(first_request_open.wait(), timeout=1.0)
    try:
        with pytest.raises(DDoSProtection) as error:
            await exchange.fetch("https://offline.invalid/rate-limited")
        assert error.value.retry_after_seconds == 90.0
        assert not hasattr(error.value, "response_headers")
        assert exchange.last_response_headers["Retry-After"] == "90"
    finally:
        second_request_done.set()
        with pytest.raises(RequestTimeout) as timeout:
            await pending
    assert not hasattr(timeout.value, "retry_after_seconds")


@pytest.mark.asyncio
async def test_ccxt_non_network_error_with_header_preserves_failure_class(client):
    response = RestResponse(401, {"Retry-After": "120"})
    with pytest.raises(AuthenticationError) as error:
        await client(lambda *args, **kwargs: response).fapiPublicGetTime()
    assert not hasattr(error.value, "retry_after_seconds")
