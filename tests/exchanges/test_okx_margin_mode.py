import asyncio

import pytest

from exchanges.okx import OKXBot


class DummyTask:
    def __init__(self, coro):
        self._coro = coro

    def __await__(self):
        return self._coro.__await__()


class DummyCCA:
    def __init__(self):
        self.margin_calls = []

    async def set_margin_mode(self, margin_mode, *, symbol=None, params=None):
        self.margin_calls.append((margin_mode, symbol, params))
        return {"code": "0"}


def _make_bot():
    bot = OKXBot.__new__(OKXBot)
    bot.config = {"live": {"margin_mode_preference": "auto"}}
    bot.markets_dict = {}
    bot._live_margin_modes = {}
    bot.positions = {}
    bot.open_orders = {}
    bot.max_leverage = {}
    bot.bot_value = lambda pside, key: 1.0 if key == "total_wallet_exposure_limit" else 0.0
    bot.config_get = lambda path, symbol=None: 10
    return bot


def test_okx_normalize_positions_keeps_isolated_positions():
    bot = _make_bot()
    fetched = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": 0.5,
            "entryPrice": 50000.0,
            "marginMode": "isolated",
        }
    ]

    positions = bot._normalize_positions(fetched)

    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC/USDT:USDT"
    assert positions[0]["margin_mode"] == "isolated"
    assert bot._live_margin_modes["BTC/USDT:USDT"] == "isolated"


def test_okx_build_order_params_uses_resolved_margin_mode():
    bot = _make_bot()
    bot.config["live"]["time_in_force"] = "post_only"
    bot.broker_code = "broker"
    bot.okx_dual_side = True
    bot.markets_dict = {
        "BTC/USDT:USDT": {"marginModes": {"cross": True, "isolated": True}, "info": {}}
    }
    bot.config["live"]["margin_mode_preference"] = "auto_isolated"

    params = bot._build_order_params(
        {
            "symbol": "BTC/USDT:USDT",
            "reduce_only": False,
            "custom_id": "cid",
            "position_side": "long",
        }
    )

    assert params["marginMode"] == "isolated"
    assert params["positionSide"] == "long"


@pytest.mark.asyncio
async def test_okx_exchange_config_uses_resolved_margin_mode(monkeypatch):
    bot = _make_bot()
    bot.cca = DummyCCA()
    bot.markets_dict = {
        "BTC/USDT:USDT": {"marginModes": {"cross": True, "isolated": True}, "info": {}}
    }
    bot.config["live"]["margin_mode_preference"] = "auto_isolated"

    monkeypatch.setattr(asyncio, "create_task", lambda coro: DummyTask(coro))

    await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    assert bot.cca.margin_calls == [("isolated", "BTC/USDT:USDT", {"lever": 10})]
