import asyncio

import pytest

from exchanges.binance import BinanceBot
from exchanges.bitget import BitgetBot


class DummyTask:
    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        return self._coro.__await__()


class DummyCCA:
    def __init__(self):
        self.margin_calls = []
        self.leverage_calls = []

    async def set_margin_mode(self, margin_mode, *, symbol=None, params=None):
        self.margin_calls.append((margin_mode, symbol, params))
        return {"code": "0"}

    async def set_leverage(self, leverage, *, symbol=None):
        self.leverage_calls.append((leverage, symbol))
        return {"code": "0"}


def _configure_bot(bot, symbol):
    bot.cca = DummyCCA()
    bot.config = {"live": {"margin_mode_preference": "auto_isolated"}}
    bot.markets_dict = {symbol: {"marginModes": {"cross": True, "isolated": True}, "info": {}}}
    bot.max_leverage = {symbol: 20}
    bot.positions = {}
    bot.open_orders = {}
    bot._live_margin_modes = {}
    bot.bot_value = lambda pside, key: 1.0 if key == "total_wallet_exposure_limit" else 0.0
    bot.config_get = lambda path, symbol=None: 7


@pytest.mark.asyncio
async def test_binance_exchange_config_uses_resolved_margin_mode(monkeypatch):
    bot = BinanceBot.__new__(BinanceBot)
    _configure_bot(bot, "BTC/USDT:USDT")

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert bot.cca.margin_calls == [("isolated", "BTC/USDT:USDT", None)]
    assert bot.cca.leverage_calls == [(7, "BTC/USDT:USDT")]


@pytest.mark.asyncio
async def test_bitget_exchange_config_uses_resolved_margin_mode(monkeypatch):
    bot = BitgetBot.__new__(BitgetBot)
    _configure_bot(bot, "ETH/USDT:USDT")

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    await bot.update_exchange_config_by_symbols(["ETH/USDT:USDT"])

    assert bot.cca.margin_calls == [("isolated", "ETH/USDT:USDT", None)]
    assert bot.cca.leverage_calls == [(7, "ETH/USDT:USDT")]
