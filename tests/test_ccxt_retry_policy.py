import pytest

from candlestick_manager import CandlestickManager


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

    rows = await cm._ccxt_fetch_ohlcv_once(
        "BTC/USDT:USDT",
        since_ms=1643262960000,
        limit=1000,
        timeframe="1m",
    )

    # Default behavior: 5 attempts -> still fails -> returns empty
    assert rows == []
    assert ex.calls == 5
