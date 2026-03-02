import importlib
import sys
import types

import pytest


@pytest.fixture
def stubbed_modules(monkeypatch):
    # Stub passivbot_rust
    pr_module = types.ModuleType("passivbot_rust")
    pr_module.qty_to_cost = lambda *args, **kwargs: 0.0
    pr_module.round_dynamic = lambda x, y=None: x
    pr_module.calc_order_price_diff = lambda *args, **kwargs: 0.0
    pr_module.hysteresis = lambda x, y, z: x
    pr_module.trailing_bundle_default_py = lambda: (0, 0, 0, 0)
    pr_module.update_trailing_bundle_py = lambda *args, **kwargs: (0, 0, 0, 0)
    pr_module.order_type_id_to_snake = lambda *args, **kwargs: "unknown"
    pr_module.calc_min_entry_qty_py = lambda *args, **kwargs: 0.0
    pr_module.calc_min_close_qty_py = lambda *args, **kwargs: 0.0
    pr_module.__getattr__ = lambda name: (lambda *args, **kwargs: 0)
    monkeypatch.setitem(sys.modules, "passivbot_rust", pr_module)

    # Stub ccxt modules
    errors_module = types.ModuleType("ccxt.base.errors")
    errors_module.NetworkError = Exception
    errors_module.RateLimitExceeded = Exception
    monkeypatch.setitem(sys.modules, "ccxt.base.errors", errors_module)

    ccxt_base = types.ModuleType("ccxt.base")
    ccxt_base.errors = errors_module
    monkeypatch.setitem(sys.modules, "ccxt.base", ccxt_base)

    ccxt_async = types.ModuleType("ccxt.async_support")
    ccxt_async.hyperliquid = None
    monkeypatch.setitem(sys.modules, "ccxt.async_support", ccxt_async)

    ccxt_pro = types.ModuleType("ccxt.pro")
    ccxt_pro.hyperliquid = None
    monkeypatch.setitem(sys.modules, "ccxt.pro", ccxt_pro)

    ccxt_module = types.ModuleType("ccxt")
    ccxt_module.__version__ = "4.4.99"
    ccxt_module.base = ccxt_base
    ccxt_module.async_support = ccxt_async
    ccxt_module.pro = ccxt_pro
    monkeypatch.setitem(sys.modules, "ccxt", ccxt_module)

    # Stub procedures to bypass ccxt version assertion during import
    proc_module = types.ModuleType("procedures")
    proc_module.assert_correct_ccxt_version = lambda *args, **kwargs: None
    proc_module.print_async_exception = lambda *args, **kwargs: None
    proc_module.load_broker_code = lambda *args, **kwargs: {}
    proc_module.load_user_info = lambda *args, **kwargs: {"exchange": "hyperliquid"}
    proc_module.get_first_timestamps_unified = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "procedures", proc_module)

    yield

    # Cleanup reload of hyperliquid module if it was imported
    if "exchanges.hyperliquid" in sys.modules:
        sys.modules.pop("exchanges.hyperliquid", None)


class DummyCCA:
    def __init__(self):
        self.calls = 0

    async def fetch_balance(self):
        self.calls += 1
        return {
            "info": {
                "marginSummary": {"accountValue": 200.0},
                "assetPositions": [
                    {"position": {"coin": "BTC", "szi": 1.0, "entryPx": 100.0, "unrealizedPnl": 10.0}}
                ],
            }
        }


@pytest.mark.asyncio
async def test_hyperliquid_combined_fetch_reused(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    import asyncio

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.quote = "USDT"
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.coin_to_symbol = lambda c: "BTC/USDT:USDT" if c == "BTC" else c
    bot.cm = types.SimpleNamespace(get_current_close=lambda *args, **kwargs: 1.0)
    bot._hl_fetch_lock = asyncio.Lock()
    bot._hl_cache_generation = 0
    dummy = DummyCCA()
    bot.cca = dummy

    # First update_positions should fetch once, cache balance
    ok = await bot.update_positions()
    assert ok is True
    assert dummy.calls == 1
    assert bot.fetched_positions[0]["symbol"] == "BTC/USDT:USDT"

    # update_balance should reuse cached combined fetch; no new call
    ok = await bot.update_balance()
    assert ok is True
    assert dummy.calls == 1

    # Another update_positions triggers another combined fetch
    ok = await bot.update_positions()
    assert ok is True
    assert dummy.calls == 2
