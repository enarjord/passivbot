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


@pytest.mark.asyncio
async def test_hyperliquid_snapshot_helpers_return_raw_bundle_on_cold_capture(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot._hl_cache_generation = 0
    bot._last_hl_balance = None
    bot._hl_balance_consumed = True
    bot.fetched_positions = []
    bot.fetched_balance = {}

    raw_snapshot = {
        "balance": {"info": {"marginSummary": {"accountValue": 200.0}}},
        "positions": {
            "core": [{"position": {"coin": "BTC", "szi": "1.0"}}],
            "hip3": [{"fetch_spec": {"params": {"dex": "xyz"}}, "response": [{"symbol": "XYZ-SP500"}]}],
        },
    }
    normalized_positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 1.0,
            "price": 100.0,
        }
    ]

    async def fake_cached(my_gen=0):
        return raw_snapshot, normalized_positions, 190.0

    bot._get_positions_and_balance_cached = fake_cached

    raw_positions, normalized = await bot.capture_positions_snapshot()
    raw_balance, balance = await bot.capture_balance_snapshot()

    assert raw_positions == raw_snapshot["positions"]
    assert normalized == normalized_positions
    assert raw_balance == raw_snapshot["balance"]
    assert balance == 190.0


def _make_probe_bot(HyperliquidBot):
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.quote = "USDC"
    bot.balance_override = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.previous_hysteresis_balance = 51.194323
    bot.balance_raw = 51.194323
    bot.balance = 51.194323
    bot._exchange_reported_balance_raw = 51.194323
    bot.c_mults = {
        "XYZ-SP500/USDC:USDC": 1.0,
        "BTC/USDC:USDC": 1.0,
    }
    bot._get_hl_dex_for_symbol = lambda symbol: "xyz" if symbol == "XYZ-SP500/USDC:USDC" else None
    bot._requires_isolated_margin = lambda symbol: False
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 20 if symbol in {"XYZ-SP500/USDC:USDC", "BTC/USDC:USDC"} else 5
    bot.fetched_positions = []
    bot.open_orders = {}
    bot.stop_signal_received = False
    return bot


def _pb_order_id(type_hex: str = "0000") -> str:
    return f"pb-0x{type_hex}-test"


def test_hyperliquid_reconcile_adds_back_cross_hip3_order_margin(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.003,
                "price": 5088.8,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    expected_reserve = 0.003 * 5088.8 / 20.0 * 1.01
    assert changed is True
    assert bot.balance_raw == pytest.approx(51.194323 + expected_reserve)
    assert bot.balance == pytest.approx(51.194323 + expected_reserve)


def test_hyperliquid_reconcile_adds_back_flat_standard_perp_entry_reserve(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "BTC/USDC:USDC": [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "qty": 0.00022,
                "price": 54161.0,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    expected_reserve = 0.00022 * 54161.0 / 20.0 * 1.01
    assert changed is True
    assert bot.balance_raw == pytest.approx(51.194323 + expected_reserve)
    assert bot.balance == pytest.approx(51.194323 + expected_reserve)


def test_hyperliquid_reconcile_skips_standard_perp_entry_reserve_when_position_exists(
    stubbed_modules,
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.fetched_positions = [
        {
            "symbol": "BTC/USDC:USDC",
            "position_side": "long",
            "size": 0.00016,
            "price": 72029.0,
            "margin_used": 0.57616,
        }
    ]
    bot.open_orders = {
        "BTC/USDC:USDC": [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "qty": 0.00016,
                "price": 54072.0,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    assert changed is False
    assert bot.balance_raw == pytest.approx(51.194323)
    assert bot.balance == pytest.approx(51.194323)


def test_hyperliquid_reconcile_adds_back_hip3_position_margin(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.fetched_positions = [
        {
            "symbol": "XYZ-SP500/USDC:USDC",
            "position_side": "long",
            "size": 0.002,
            "price": 6813.8,
            "margin_used": 0.68139,
        }
    ]

    changed = bot._reconcile_balance_after_positions_and_balance_refresh()

    assert changed is True
    assert bot.balance_raw == pytest.approx(51.194323 + 0.68139)
    assert bot.balance == pytest.approx(51.194323 + 0.68139)


def test_hyperliquid_reconcile_adds_back_hip3_position_and_entry_order_margin(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.fetched_positions = [
        {
            "symbol": "XYZ-SP500/USDC:USDC",
            "position_side": "long",
            "size": 0.002,
            "price": 6814.3,
            "margin_used": 0.68144,
        }
    ]
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.002,
                "price": 5110.7,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    expected = 51.194323 + 0.68144 + (0.002 * 5110.7 / 20.0 * 1.01)
    assert changed is True
    assert bot.balance_raw == pytest.approx(expected)
    assert bot.balance == pytest.approx(expected)


def test_hyperliquid_reconcile_skips_balance_override(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.balance_override = 40.0
    bot.balance_raw = 40.0
    bot.balance = 40.0
    bot._exchange_reported_balance_raw = 35.0
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.003,
                "price": 5088.8,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    assert changed is False
    assert bot.balance_raw == pytest.approx(40.0)
    assert bot.balance == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_update_open_orders_applies_hyperliquid_balance_reconciliation(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    seen_sources = []

    async def fake_fetch_open_orders():
        return [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.003,
                "amount": 0.003,
                "price": 5088.8,
                "timestamp": 1,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]

    async def fake_handle_balance_update(source="REST"):
        seen_sources.append(source)

    bot.fetch_open_orders = fake_fetch_open_orders
    bot.handle_balance_update = fake_handle_balance_update
    bot.order_was_recently_cancelled = lambda order: False
    bot.log_order_action = lambda *args, **kwargs: None

    ok = await bot.update_open_orders()

    expected_reserve = 0.003 * 5088.8 / 20.0 * 1.01
    assert ok is True
    assert seen_sources == ["REST+open_orders"]
    assert bot.balance_raw == pytest.approx(51.194323 + expected_reserve)
    assert bot.open_orders["XYZ-SP500/USDC:USDC"][0]["id"] == "1"


def test_hyperliquid_reconcile_skips_external_standard_perp_entry_order(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "BTC/USDC:USDC": [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "qty": 0.00022,
                "price": 54161.0,
                "reduce_only": False,
                "clientOrderId": "",
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    assert changed is False
    assert bot.balance_raw == pytest.approx(51.194323)
    assert bot.balance == pytest.approx(51.194323)


def test_hyperliquid_reconcile_skips_external_hip3_entry_order(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.003,
                "price": 5088.8,
                "reduce_only": False,
                "clientOrderId": "manual-order-123",
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    assert changed is False
    assert bot.balance_raw == pytest.approx(51.194323)
    assert bot.balance == pytest.approx(51.194323)


def test_hyperliquid_reconcile_skips_external_hex_prefixed_order_ids(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.003,
                "price": 5088.8,
                "reduce_only": False,
                "clientOrderId": "deadbeef",
            }
        ]
    }

    changed = bot._reconcile_balance_after_open_orders_refresh()

    assert changed is False
    assert bot.balance_raw == pytest.approx(51.194323)
    assert bot.balance == pytest.approx(51.194323)


@pytest.mark.asyncio
async def test_update_positions_and_balance_applies_hip3_position_reconciliation(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    seen_sources = []

    async def fake_fetch_balance():
        return 50.499284

    async def fake_fetch_positions():
        return [
            {
                "symbol": "XYZ-SP500/USDC:USDC",
                "position_side": "long",
                "size": 0.002,
                "price": 6813.8,
                "margin_used": 0.68139,
            }
        ]

    async def fake_handle_balance_update(source="REST"):
        seen_sources.append(source)

    async def fake_log_position_changes(*args, **kwargs):
        return None

    bot.fetch_balance = fake_fetch_balance
    bot.fetch_positions = fake_fetch_positions
    bot.handle_balance_update = fake_handle_balance_update
    bot.active_symbols = []
    bot.positions = {}
    bot.log_position_changes = fake_log_position_changes

    balance_ok, positions_ok = await bot.update_positions_and_balance()

    assert (balance_ok, positions_ok) == (True, True)
    assert seen_sources == ["REST"]
    assert bot.balance_raw == pytest.approx(50.499284 + 0.68139)
    assert bot.fetched_positions[0]["symbol"] == "XYZ-SP500/USDC:USDC"
