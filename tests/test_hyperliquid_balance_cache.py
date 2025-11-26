import sys
import types

import pytest

# Stub dependencies before importing hyperliquid
sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

# Stub ccxt to bypass version assertion during test import
fake_errors = types.SimpleNamespace(NetworkError=Exception, RateLimitExceeded=Exception)
fake_ccxt_errors_module = types.SimpleNamespace(NetworkError=Exception, RateLimitExceeded=Exception)
sys.modules.setdefault("ccxt.base.errors", fake_ccxt_errors_module)
fake_ccxt_base = types.SimpleNamespace(errors=fake_errors)
fake_ccxt = types.SimpleNamespace(
    __version__="4.4.99",
    base=fake_ccxt_base,
    async_support=types.SimpleNamespace(hyperliquid=None, kucoinfutures=None),  # placeholder
)
fake_ccxt_pro = types.SimpleNamespace(hyperliquid=None, kucoinfutures=None)
fake_ccxt.pro = fake_ccxt_pro
sys.modules.setdefault("ccxt", fake_ccxt)
sys.modules.setdefault("ccxt.async_support", fake_ccxt)
sys.modules.setdefault("ccxt.pro", fake_ccxt_pro)

# Stub procedures to bypass ccxt version assertion during import
fake_procedures = types.SimpleNamespace(
    assert_correct_ccxt_version=lambda *args, **kwargs: None,
    print_async_exception=lambda *args, **kwargs: None,
    load_broker_code=lambda *args, **kwargs: {},
    load_user_info=lambda *args, **kwargs: {"exchange": "hyperliquid"},
    get_first_timestamps_unified=lambda *args, **kwargs: {},
)
sys.modules.setdefault("procedures", fake_procedures)

from exchanges.hyperliquid import HyperliquidBot


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
async def test_hyperliquid_combined_fetch_reused():
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.quote = "USDT"
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.coin_to_symbol = lambda c: "BTC/USDT:USDT" if c == "BTC" else c
    bot.cm = types.SimpleNamespace(get_current_close=lambda *args, **kwargs: 1.0)
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
