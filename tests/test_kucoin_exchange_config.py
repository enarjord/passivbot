import asyncio
import logging
import types

import pytest

from exchanges.kucoin import KucoinBot


class DummyTask:
    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        return self._coro.__await__()


class DummyCCA:
    def __init__(self):
        self.position_mode_calls = []
        self.margin_calls = []
        self.leverage_calls = []

    async def set_position_mode(self, hedged):
        self.position_mode_calls.append(hedged)
        return {"code": "200000", "hedged": hedged}

    async def set_margin_mode(self, **params):
        self.margin_calls.append(params)
        return {"symbol": params["symbol"], "marginMode": params["marginMode"]}

    async def set_leverage(self, **params):
        self.leverage_calls.append(params)
        return {"symbol": params["symbol"], "leverage": params["leverage"]}


def make_bot():
    bot = KucoinBot.__new__(KucoinBot)
    bot.cca = DummyCCA()
    bot.hedge_mode = True
    bot.max_leverage = {}
    return bot


@pytest.mark.asyncio
async def test_update_exchange_config_sets_position_mode_when_supported(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    await bot.update_exchange_config()
    assert bot.cca.position_mode_calls == [True]
    assert "set_position_mode hedged=True" in caplog.text


@pytest.mark.asyncio
async def test_update_exchange_config_handles_missing_position_mode(caplog):
    caplog.set_level(logging.INFO)
    bot = make_bot()
    bot.cca = types.SimpleNamespace()
    await bot.update_exchange_config()
    assert "set_position_mode not supported" in caplog.text


@pytest.mark.asyncio
async def test_update_exchange_config_by_symbols_sets_margin_and_leverage(monkeypatch):
    bot = make_bot()
    leverage_cfg = {
        "BTC/USDT:USDT": 5,
        "ETH/USDT:USDT": 3,
    }
    bot.max_leverage = {
        "BTC/USDT:USDT": 10,
        "ETH/USDT:USDT": 2,
    }

    def config_get(path, *, symbol=None):
        if path == ["live", "leverage"]:
            return leverage_cfg[symbol]
        raise KeyError(path)

    bot.config_get = config_get

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    symbols = list(leverage_cfg.keys())
    await bot.update_exchange_config_by_symbols(symbols)

    margin_symbols = [call["symbol"] for call in bot.cca.margin_calls]
    assert margin_symbols == symbols
    assert all(call["marginMode"] == "cross" for call in bot.cca.margin_calls)

    leverage_map = {call["symbol"]: call["leverage"] for call in bot.cca.leverage_calls}
    # leverage is clamped by max_leverage
    assert leverage_map["BTC/USDT:USDT"] == 5  # min(10, 5)
    assert leverage_map["ETH/USDT:USDT"] == 2  # min(2, 3)
