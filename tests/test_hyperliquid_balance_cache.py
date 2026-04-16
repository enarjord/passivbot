import importlib
import logging as pylogging
import sys
import types

import pytest
from passivbot_exceptions import FatalBotException


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

    async def fetch_positions(self, **kwargs):
        del kwargs
        return []


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


@pytest.mark.asyncio
async def test_hyperliquid_fetch_open_orders_dedupes_parallel_routes(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "BTC/USDC:USDC": {"info": {}},
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
    }
    bot.positions = {
        "BTC/USDC:USDC": {"long": {"size": 0.1}, "short": {"size": 0.0}},
        "XYZ-SP500/USDC:USDC": {"long": {"size": 0.1}, "short": {"size": 0.0}},
    }
    bot._hl_state_fetch_concurrency = lambda: 2
    bot._get_hl_dex_for_symbol = lambda symbol: "xyz" if symbol == "XYZ-SP500/USDC:USDC" else None

    class _OpenOrdersCCA:
        async def fetch_open_orders(self, symbol=None, params=None):
            if params and params.get("dex") == "xyz":
                return [
                    {
                        "id": "2",
                        "symbol": "XYZ-SP500/USDC:USDC",
                        "side": "buy",
                        "amount": 0.003,
                        "price": 5088.8,
                        "timestamp": 2,
                    },
                    {
                        "id": "1",
                        "symbol": "BTC/USDC:USDC",
                        "side": "buy",
                        "amount": 0.001,
                        "price": 70000.0,
                        "timestamp": 1,
                    },
                ]
            return [
                {
                    "id": "1",
                    "symbol": "BTC/USDC:USDC",
                    "side": "buy",
                    "amount": 0.001,
                    "price": 70000.0,
                    "timestamp": 1,
                }
            ]

    bot.cca = _OpenOrdersCCA()

    orders = await bot.fetch_open_orders()

    assert [order["id"] for order in orders] == ["1", "2"]
    assert orders[0]["qty"] == pytest.approx(0.001)
    assert orders[1]["qty"] == pytest.approx(0.003)
    assert orders[0]["position_side"] == "long"
    assert orders[1]["position_side"] == "long"


def test_hyperliquid_selects_active_dex_scope_until_periodic_full_sweep(stubbed_modules, monkeypatch):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
        "XYZ-GOLD/USDC:USDC": {"baseName": "gold:GOLD", "info": {"baseName": "gold:GOLD"}},
    }
    bot.active_symbols = ["XYZ-SP500/USDC:USDC"]
    bot.open_orders = {}
    bot.positions = {}
    bot._hl_force_full_dex_sweep = False
    bot._hl_force_full_dex_sweep_surfaces = set()
    bot._hl_last_full_dex_sweep_ms_by_surface = {"positions": 1_000}

    monkeypatch.setattr("exchanges.hyperliquid.utc_ms", lambda: 50_000)

    dexes, full = bot._hl_select_dex_names_for_state("positions")
    assert (dexes, full) == (["xyz"], False)

    bot._hl_last_full_dex_sweep_ms_by_surface["positions"] = 0
    dexes, full = bot._hl_select_dex_names_for_state("positions")
    assert full is True
    assert dexes == ["gold", "xyz"]


def test_hyperliquid_ws_unknown_dex_activity_forces_full_sweep(stubbed_modules, caplog):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
        "XYZ-GOLD/USDC:USDC": {"baseName": "gold:GOLD", "info": {"baseName": "gold:GOLD"}},
    }
    bot.active_symbols = ["XYZ-SP500/USDC:USDC"]
    bot.open_orders = {}
    bot.positions = {}
    bot._hl_force_full_dex_sweep = False
    bot._hl_force_full_dex_sweep_surfaces = set()

    with caplog.at_level(pylogging.INFO):
        bot._hl_note_ws_symbols_for_dex_scope(
            [{"symbol": "XYZ-GOLD/USDC:USDC", "status": "open", "side": "buy", "amount": 0.1}]
        )

    assert bot._hl_force_full_dex_sweep_surfaces == {"open_orders", "positions"}
    assert any("forcing full hip3 sweep" in record.message for record in caplog.records)


def test_hyperliquid_unknown_dex_full_sweep_sticks_until_positions_consume_it(
    stubbed_modules, monkeypatch
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
        "PARA-TOTAL2/USDC:USDC": {
            "baseName": "para:TOTAL2",
            "info": {"baseName": "para:TOTAL2"},
        },
    }
    bot.active_symbols = ["XYZ-SP500/USDC:USDC"]
    bot.open_orders = {}
    bot.positions = {}
    bot._hl_force_full_dex_sweep = False
    bot._hl_force_full_dex_sweep_surfaces = {"open_orders", "positions"}
    bot._hl_last_full_dex_sweep_ms_by_surface = {}

    monkeypatch.setattr("exchanges.hyperliquid.utc_ms", lambda: 50_000)

    dexes, full = bot._hl_select_dex_names_for_state("open_orders")
    assert full is True
    assert dexes == ["para", "xyz"]
    bot._hl_mark_dex_scope_consumed("open_orders", full_sweep=True)

    assert bot._hl_force_full_dex_sweep_surfaces == {"positions"}

    dexes, full = bot._hl_select_dex_names_for_state("positions")
    assert full is True
    assert dexes == ["para", "xyz"]


def test_hyperliquid_non_unified_approved_hip3_requires_unified(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = _make_probe_bot(HyperliquidBot)
    bot._hl_user_abstraction = "dexAbstraction"
    bot._hl_unified_enabled = False
    bot.approved_coins_minus_ignored_coins = {
        "long": {"XYZ-SP500/USDC:USDC"},
        "short": set(),
    }
    bot.positions = {}
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
    }

    with pytest.raises(FatalBotException, match="require unifiedAccount mode"):
        bot._assert_supported_live_state()


def test_hyperliquid_non_unified_live_hip3_state_requires_unified(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = _make_probe_bot(HyperliquidBot)
    bot._hl_user_abstraction = "dexAbstraction"
    bot._hl_unified_enabled = False
    bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
    bot.positions = {
        "XYZ-SP500/USDC:USDC": {
            "long": {"size": 0.002, "price": 6953.4},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.open_orders = {}
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
    }

    with pytest.raises(FatalBotException, match="Unsupported HIP-3 state detected"):
        bot._assert_supported_live_state()


def test_hyperliquid_unified_allows_hip3_symbols(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = _make_probe_bot(HyperliquidBot)
    bot._hl_user_abstraction = "unifiedAccount"
    bot._hl_unified_enabled = True
    bot.approved_coins_minus_ignored_coins = {
        "long": {"XYZ-SP500/USDC:USDC"},
        "short": set(),
    }
    bot.positions = {
        "XYZ-SP500/USDC:USDC": {
            "long": {"size": 0.002, "price": 6953.4},
            "short": {"size": 0.0, "price": 0.0},
        }
    }
    bot.open_orders = {
        "XYZ-SP500/USDC:USDC": [
            {"id": "1", "symbol": "XYZ-SP500/USDC:USDC", "qty": 0.003, "price": 6600.0}
        ]
    }
    bot.markets_dict = {
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
    }

    bot._assert_supported_live_state()


@pytest.mark.asyncio
async def test_hyperliquid_combined_fetch_handles_unified_balance_payload(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    import asyncio

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.quote = "USDC"
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.coin_to_symbol = lambda c: "BTC/USDC:USDC" if c == "BTC" else c
    bot.cm = types.SimpleNamespace(get_current_close=lambda *args, **kwargs: 1.0)
    bot._hl_fetch_lock = asyncio.Lock()
    bot._hl_cache_generation = 0
    bot.markets_dict = {
        "BTC/USDC:USDC": {"info": {}},
        "XYZ-SP500/USDC:USDC": {"baseName": "xyz:SP500", "info": {"baseName": "xyz:SP500"}},
    }
    bot._get_hl_dex_for_symbol = lambda symbol: "xyz" if symbol == "XYZ-SP500/USDC:USDC" else None
    bot._record_hl_live_margin_mode = lambda *args, **kwargs: None

    class _UnifiedCCA:
        async def fetch_balance(self):
            return {
                "total": {"USDC": 50.92373263},
                "info": {
                    "balances": [
                        {"coin": "USDC", "hold": "6.59768", "total": "50.92373263"},
                    ]
                },
            }

        async def fetch_positions(self, **kwargs):
            if kwargs.get("params", {}).get("dex") == "xyz":
                return [
                    {
                        "symbol": "XYZ-SP500/USDC:USDC",
                        "side": "long",
                        "contracts": 0.002,
                        "entryPrice": 6953.4,
                        "marginMode": "cross",
                        "initialMargin": 0.69617,
                        "info": {
                            "position": {
                                "coin": "xyz:SP500",
                                "leverage": {"type": "cross"},
                                "marginUsed": "0.69617",
                            }
                        },
                    }
                ]
            return [
                {
                    "symbol": "BTC/USDC:USDC",
                    "side": "long",
                    "contracts": 0.00028,
                    "entryPrice": 74741.6,
                    "marginMode": "cross",
                    "initialMargin": 1.039948,
                    "info": {
                        "position": {
                            "coin": "BTC",
                            "leverage": {"type": "cross"},
                            "marginUsed": "1.039948",
                        }
                    },
                }
            ]

    bot.cca = _UnifiedCCA()

    raw_snapshot, positions, balance = await bot._fetch_positions_and_balance()

    assert balance == pytest.approx(50.92373263)
    assert raw_snapshot["balance_mode"] == "unified_total"
    assert {p["symbol"] for p in positions} == {"BTC/USDC:USDC", "XYZ-SP500/USDC:USDC"}
    assert any(p["symbol"] == "BTC/USDC:USDC" and p["margin_used"] == pytest.approx(1.039948) for p in positions)
    assert any(p["symbol"] == "XYZ-SP500/USDC:USDC" and p["margin_used"] == pytest.approx(0.69617) for p in positions)


@pytest.mark.asyncio
async def test_hyperliquid_fetch_user_abstraction_state_sets_unified_options(stubbed_modules):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.user = "hyperliquid_01"
    bot.user_info = {"wallet_address": "0xabc"}
    bot.cca = types.SimpleNamespace(
        options={},
        publicPostInfo=lambda payload: _return_async('"unifiedAccount"', payload),
    )
    bot.ccp = types.SimpleNamespace(options={})

    abstraction = await bot.fetch_user_abstraction_state()

    assert abstraction == "unifiedAccount"
    assert bot._hl_user_abstraction == "unifiedAccount"
    assert bot._hl_unified_enabled is True
    assert bot.cca.options["enableUnifiedMargin"] is True
    assert bot.ccp.options["enableUnifiedMargin"] is True


@pytest.mark.asyncio
async def test_hyperliquid_refresh_and_log_user_abstraction_logs_initial_and_change(
    stubbed_modules, caplog
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot

    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.user = "hyperliquid_01"
    bot.user_info = {"wallet_address": "0xabc"}
    responses = iter(['"disabled"', '"unifiedAccount"'])

    async def _public_post_info(_payload):
        return next(responses)

    bot.cca = types.SimpleNamespace(options={}, publicPostInfo=_public_post_info)
    bot.ccp = types.SimpleNamespace(options={})

    with caplog.at_level(pylogging.INFO):
        first = await bot.refresh_and_log_user_abstraction_state()
        second = await bot.refresh_and_log_user_abstraction_state()

    assert first == "disabled"
    assert second == "unifiedAccount"
    assert bot._hl_last_logged_user_abstraction == "unifiedAccount"
    assert "[account] Hyperliquid abstraction=disabled | unified=no" in caplog.text
    assert (
        "[account] Hyperliquid abstraction changed disabled -> unifiedAccount | unified=yes"
        in caplog.text
    )


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


async def _return_async(value, _payload):
    return value
@pytest.mark.asyncio
async def test_update_open_orders_suppresses_missing_log_for_exact_recent_bot_cancel(
    stubbed_modules,
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.open_orders = {
        "BTC/USDC:USDC": [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "side": "buy",
                "position_side": "long",
                "qty": 0.00042,
                "amount": 0.00042,
                "price": 69257.0,
                "timestamp": 1,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }
    bot.recent_order_cancellations = [
        {
            "id": "1",
            "symbol": "BTC/USDC:USDC",
            "side": "buy",
            "position_side": "long",
            "qty": 0.00042,
            "amount": 0.00042,
            "price": 69257.0,
            "timestamp": 1,
            "reduce_only": False,
            "clientOrderId": _pb_order_id(),
            "execution_timestamp": 1_000_000,
        }
    ]
    seen = []

    async def fake_fetch_open_orders():
        return []

    async def fail_update_positions_and_balance():
        raise AssertionError("should not schedule positions refresh for confirmed bot cancel")

    bot.fetch_open_orders = fake_fetch_open_orders
    bot.handle_balance_update = lambda source="REST": None
    bot.update_positions_and_balance = fail_update_positions_and_balance
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda order, action, source, **kwargs: seen.append((action, kwargs))

    import passivbot as pb_mod

    original_utc_ms = pb_mod.utc_ms
    pb_mod.utc_ms = lambda: 1_060_000
    try:
        ok = await bot.update_open_orders()
    finally:
        pb_mod.utc_ms = original_utc_ms

    assert ok is True
    assert [action for action, _kwargs in seen] == ["removed order"]
    assert seen[0][1].get("context") == "bot_cancel_confirmed"


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_hyperliquid_publishes_final_balance_once(
    stubbed_modules,
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "hyperliquid"
    bot.active_symbols = []
    bot.positions = {}
    bot.fetched_positions = []
    bot.fetched_open_orders = []
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot.recent_order_cancellations = []
    bot.previous_hysteresis_balance = 0.0
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    seen_sources = []
    seen = {}

    async def fake_capture_positions_balance_staged_snapshot():
        return (
            {"raw": "snapshot"},
            [
                {
                    "symbol": "XYZ-SP500/USDC:USDC",
                    "position_side": "long",
                    "size": 0.002,
                    "price": 6813.8,
                    "margin_used": 0.68139,
                }
            ],
            50.499284,
        )

    async def fake_fetch_open_orders():
        return [
            {
                "id": "1",
                "symbol": "XYZ-SP500/USDC:USDC",
                "qty": 0.002,
                "amount": 0.002,
                "price": 5110.7,
                "timestamp": 1,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]

    async def fake_log_position_changes(*args, **kwargs):
        del args, kwargs
        seen["balance_raw_when_logged"] = bot.balance_raw

    async def fake_handle_balance_update(source="REST"):
        seen_sources.append(source)

    async def fake_update_pnls():
        return True

    bot._capture_positions_balance_staged_snapshot = fake_capture_positions_balance_staged_snapshot
    bot.fetch_open_orders = fake_fetch_open_orders
    bot.update_pnls = fake_update_pnls
    bot.log_position_changes = fake_log_position_changes
    bot.handle_balance_update = fake_handle_balance_update
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None

    ok = await bot.refresh_authoritative_state()

    expected_balance = 50.499284
    assert ok is True
    assert bot.balance_raw == pytest.approx(expected_balance)
    assert bot.balance == pytest.approx(expected_balance)
    assert seen["balance_raw_when_logged"] == pytest.approx(expected_balance)
    assert seen_sources == ["REST"]
    assert bot.open_orders["XYZ-SP500/USDC:USDC"][0]["id"] == "1"


@pytest.mark.asyncio
async def test_hyperliquid_open_orders_refresh_does_not_republish_same_hip3_effective_balance(
    stubbed_modules,
):
    HyperliquidBot = importlib.import_module("exchanges.hyperliquid").HyperliquidBot
    bot = _make_probe_bot(HyperliquidBot)
    seen_sources = []
    bot.fetched_positions = []
    bot.open_orders = {
        "BTC/USDC:USDC": [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "qty": 0.002,
                "amount": 0.002,
                "price": 5110.7,
                "timestamp": 1,
                "reduce_only": False,
                "clientOrderId": _pb_order_id(),
            }
        ]
    }
    bot.balance_raw = 50.499284
    bot.balance = bot.balance_raw
    bot._exchange_reported_balance_raw = 50.499284

    async def fake_fetch_open_orders():
        return [
            {
                "id": "1",
                "symbol": "BTC/USDC:USDC",
                "qty": 0.002,
                "amount": 0.002,
                "price": 5110.7,
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

    assert ok is True
    assert seen_sources == []
    assert bot.balance_raw == pytest.approx(50.499284)
