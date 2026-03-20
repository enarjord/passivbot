import asyncio

import ccxt
import pytest

from exchanges.bybit import BybitBot


class DummyCCA:
    def __init__(self):
        self.margin_calls = []
        self.leverage_calls = []

    async def set_margin_mode(self, margin_mode, *, symbol=None, params=None):
        self.margin_calls.append((margin_mode, symbol, params))
        return {"retCode": 0}

    async def set_leverage(self, leverage, *, symbol=None):
        self.leverage_calls.append((leverage, symbol))
        raise ccxt.BadRequest('bybit {"retCode":110043,"retMsg":"leverage not modified"}')


@pytest.mark.asyncio
async def test_bybit_exchange_config_updates_symbol_without_create_task(monkeypatch):
    bot = BybitBot.__new__(BybitBot)
    bot.cca = DummyCCA()

    def config_get(path, *, symbol=None):
        assert path == ["live", "leverage"]
        assert symbol == "BTC/USDT:USDT"
        return 10

    bot.config_get = config_get

    def fail_create_task(_coro):
        raise AssertionError("update_exchange_config_by_symbols should not create detached tasks")

    monkeypatch.setattr(asyncio, "create_task", fail_create_task)

    await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert bot.cca.margin_calls == [("cross", "BTC/USDT:USDT", {"leverage": 10})]
    assert bot.cca.leverage_calls == [(10, "BTC/USDT:USDT")]
