import pytest

from candlestick_manager import CandlestickManager, OhlcvFetchError


class RequestTimeout(Exception):
    pass


class DummyExchange:
    def __init__(self, *, exid: str, fail_times: int):
        self.id = exid
        self._fail_times = int(fail_times)
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, since, limit, params=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RequestTimeout("timed out")
        # Return a minimal valid ccxt OHLCV row
        return [[since, 1.0, 1.0, 1.0, 1.0, 0.0]]


class CapturingExchange:
    def __init__(self, *, exid: str):
        self.id = exid
        self.params_seen = None

    async def fetch_ohlcv(self, symbol, timeframe, since, limit, params=None):
        self.params_seen = dict(params or {})
        return [[since, 1.0, 1.0, 1.0, 1.0, 0.0]]


class PartialThenFailExchange:
    id = "binanceusdm"

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, since, limit, params=None):
        self.calls += 1
        if self.calls == 1:
            return [[since, 1.0, 1.0, 1.0, 1.0, 0.0]]
        raise RequestTimeout("timed out")


class PartialThenEmptyExchange:
    id = "binanceusdm"

    def __init__(self):
        self.calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, since, limit, params=None):
        self.calls += 1
        if self.calls == 1:
            return [[since, 1.0, 1.0, 1.0, 1.0, 0.0]]
        return []


@pytest.mark.asyncio
async def test_bybit_retries_more_than_default(tmp_path, monkeypatch):
    # Avoid real sleeping in retry loop
    async def _nosleep(_):
        return None

    monkeypatch.setattr("candlestick_manager.asyncio.sleep", _nosleep)

    ex = DummyExchange(exid="bybit", fail_times=6)
    cm = CandlestickManager(exchange=ex, exchange_name="bybit", cache_dir=str(tmp_path / "caches"))

    rows = await cm._ccxt_fetch_ohlcv_once(
        "HBAR/USDT:USDT",
        since_ms=1643262960000,
        limit=1000,
        timeframe="1m",
    )

    assert rows and len(rows) == 1
    # Should have retried through 6 failures and then succeed
    assert ex.calls == 7


@pytest.mark.asyncio
async def test_non_bybit_keeps_default_retry_budget(tmp_path, monkeypatch):
    async def _nosleep(_):
        return None

    monkeypatch.setattr("candlestick_manager.asyncio.sleep", _nosleep)

    ex = DummyExchange(exid="binanceusdm", fail_times=6)
    cm = CandlestickManager(
        exchange=ex, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )

    with pytest.raises(OhlcvFetchError, match="exhausted retries"):
        await cm._ccxt_fetch_ohlcv_once(
            "BTC/USDT:USDT",
            since_ms=1643262960000,
            limit=1000,
            timeframe="1m",
        )

    assert ex.calls == 5


@pytest.mark.asyncio
async def test_paginated_fetch_raises_instead_of_returning_partial_on_page_failure(
    tmp_path, monkeypatch
):
    async def _nosleep(_):
        return None

    monkeypatch.setattr("candlestick_manager.asyncio.sleep", _nosleep)

    ex = PartialThenFailExchange()
    cm = CandlestickManager(
        exchange=ex, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )

    with pytest.raises(OhlcvFetchError):
        await cm._fetch_ohlcv_paginated(
            "BTC/USDT:USDT",
            1643262960000,
            1643262960000 + 3 * 60_000,
            timeframe="1m",
        )

    assert ex.calls == 6


@pytest.mark.asyncio
async def test_paginated_fetch_can_raise_on_partial_empty_success_page(tmp_path):
    ex = PartialThenEmptyExchange()
    cm = CandlestickManager(
        exchange=ex, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )

    with pytest.raises(OhlcvFetchError, match="empty page"):
        await cm._fetch_ohlcv_paginated(
            "BTC/USDT:USDT",
            1643262960000,
            1643262960000 + 3 * 60_000,
            timeframe="1m",
            raise_on_partial_empty_page=True,
        )

    assert ex.calls == 2


@pytest.mark.asyncio
async def test_gateio_fetch_ohlcv_omits_until_param(tmp_path):
    ex = CapturingExchange(exid="gateio")
    cm = CandlestickManager(exchange=ex, exchange_name="gateio", cache_dir=str(tmp_path / "caches"))

    rows = await cm._ccxt_fetch_ohlcv_once(
        "ADA/USDT:USDT",
        since_ms=1643262960000,
        limit=1000,
        end_exclusive_ms=1643262960000 + 1000 * 60_000,
        timeframe="1m",
    )

    assert rows and len(rows) == 1
    assert ex.params_seen == {}
