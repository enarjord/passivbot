"""
Tests for HIP-3 Stock Perpetuals support.

Tests cover:
- Symbol detection (xyz: prefix, onlyIsolated flag)
- Leverage capping (10x max for isolated-only HIP-3)
- Margin mode selection (cross for cross-capable HIP-3, isolated metadata handling)
- Symbol mapping (xyz:TSLA <-> TSLA)
- Config enablement (stock_perps.enabled)
"""

import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def stubbed_modules(monkeypatch):
    """Stub external dependencies for isolated testing."""
    # Stub passivbot_rust
    pr_module = types.ModuleType("passivbot_rust")
    pr_module.qty_to_cost = lambda *args, **kwargs: 0.0
    pr_module.round_ = lambda x, step: round(x / step) * step if step else x
    pr_module.round_dynamic = lambda x, y=None: x
    pr_module.round_dynamic_up = lambda x, y=None: x
    pr_module.round_dynamic_dn = lambda x, y=None: x
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

    # Stub procedures
    proc_module = types.ModuleType("procedures")
    proc_module.assert_correct_ccxt_version = lambda *args, **kwargs: None
    proc_module.print_async_exception = lambda *args, **kwargs: None
    proc_module.load_broker_code = lambda *args, **kwargs: {}
    proc_module.load_user_info = lambda *args, **kwargs: {"exchange": "hyperliquid"}
    proc_module.get_first_timestamps_unified = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "procedures", proc_module)

    yield

    # Cleanup
    if "exchanges.hyperliquid" in sys.modules:
        sys.modules.pop("exchanges.hyperliquid", None)


class TestSymbolMapping:
    """Tests for HIP-3 symbol mapping functions."""

    def test_hip3_to_tradfi_simple(self):
        """Test conversion from HIP-3 to TradFi symbol."""
        from tradfi_data import hip3_to_tradfi_symbol

        assert hip3_to_tradfi_symbol("xyz:TSLA") == "TSLA"
        assert hip3_to_tradfi_symbol("xyz:NVDA") == "NVDA"
        assert hip3_to_tradfi_symbol("xyz:AAPL") == "AAPL"

    def test_hip3_to_tradfi_ccxt_format(self):
        """Test conversion from full CCXT symbol format."""
        from tradfi_data import hip3_to_tradfi_symbol

        assert hip3_to_tradfi_symbol("xyz:TSLA/USDC:USDC") == "TSLA"
        assert hip3_to_tradfi_symbol("xyz:NVDA/USDC:USDC") == "NVDA"

    def test_hip3_to_tradfi_passthrough(self):
        """Test that non-HIP3 symbols pass through unchanged."""
        from tradfi_data import hip3_to_tradfi_symbol

        assert hip3_to_tradfi_symbol("TSLA") == "TSLA"
        assert hip3_to_tradfi_symbol("BTC") == "BTC"

    def test_tradfi_to_hip3(self):
        """Test conversion from TradFi to HIP-3 symbol."""
        from tradfi_data import tradfi_to_hip3_symbol

        assert tradfi_to_hip3_symbol("TSLA") == "xyz:TSLA/USDC:USDC"
        assert tradfi_to_hip3_symbol("NVDA") == "xyz:NVDA/USDC:USDC"

    def test_is_stock_perp_symbol(self):
        """Test stock perp symbol detection."""
        from tradfi_data import is_stock_perp_symbol

        # Stock perps
        assert is_stock_perp_symbol("xyz:TSLA") is True
        assert is_stock_perp_symbol("xyz:TSLA/USDC:USDC") is True
        assert is_stock_perp_symbol("xyz:NVDA") is True

        # Crypto perps
        assert is_stock_perp_symbol("BTC") is False
        assert is_stock_perp_symbol("BTC/USDC:USDC") is False
        assert is_stock_perp_symbol("ETH") is False


class TestHyperliquidBotHIP3:
    """Tests for HyperliquidBot HIP-3 support."""

    @pytest.fixture
    def bot_class(self, stubbed_modules):
        """Get HyperliquidBot class with stubbed dependencies."""
        # Import after stubbing
        from exchanges.hyperliquid import HyperliquidBot

        return HyperliquidBot

    def test_hip3_prefix_constant(self, bot_class):
        """Test HIP-3 prefix constant is correct."""
        assert bot_class.HIP3_PREFIX == "xyz:"

    def test_hip3_max_leverage_constant(self, bot_class):
        """Test HIP-3 max leverage constant is 10."""
        assert bot_class.HIP3_MAX_LEVERAGE == 10

    def test_requires_isolated_margin_uses_market_metadata(self, bot_class):
        """HIP-3 isolated requirement must come from market metadata, not just prefix."""
        # Create minimal bot instance for method testing
        bot = object.__new__(bot_class)
        bot.markets_dict = {
            "xyz:TSLA/USDC:USDC": {
                "info": {"onlyIsolated": True},
                "marginModes": {"cross": False, "isolated": True},
            },
            "XYZ-XYZ100/USDC:USDC": {
                "baseName": "xyz:XYZ100",
                "info": {"marginMode": "normal"},
                "marginModes": {"cross": True, "isolated": True},
            },
        }
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES

        assert bot._requires_isolated_margin("xyz:TSLA/USDC:USDC") is True
        assert bot._requires_isolated_margin("XYZ-XYZ100/USDC:USDC") is False
        assert bot._requires_isolated_margin("BTC/USDC:USDC") is False
        assert bot._requires_isolated_margin("ETH/USDC:USDC") is False

    def test_requires_isolated_margin_by_flag(self, bot_class):
        """Test isolated margin detection by onlyIsolated flag."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.markets_dict = {
            "NEWSTOCK/USDC:USDC": {"info": {"onlyIsolated": True}},
            "BTC/USDC:USDC": {"info": {"onlyIsolated": False}},
        }

        # Market with onlyIsolated=True should require isolated margin
        assert bot._requires_isolated_margin("NEWSTOCK/USDC:USDC") is True

        # Market with onlyIsolated=False should not require isolated margin
        assert bot._requires_isolated_margin("BTC/USDC:USDC") is False

    @pytest.mark.asyncio
    async def test_update_exchange_config_uses_cross_for_cross_capable_hip3(self, bot_class):
        """Cross-capable HIP-3 symbols must stay on cross mode."""
        bot = object.__new__(bot_class)
        bot.exchange = "hyperliquid"
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.user_info = {"is_vault": False}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {
                "baseName": "xyz:XYZ100",
                "info": {"marginMode": "normal"},
                "marginModes": {"cross": True, "isolated": True},
            }
        }
        bot.cca = MagicMock()
        bot.cca.set_margin_mode = AsyncMock(return_value={"status": "ok"})
        bot._calc_leverage_for_symbol = lambda symbol: 7

        await bot.update_exchange_config_by_symbols(["XYZ-XYZ100/USDC:USDC"])

        bot.cca.set_margin_mode.assert_awaited_once_with(
            "cross",
            symbol="XYZ-XYZ100/USDC:USDC",
            params={"leverage": 7},
        )

    @pytest.mark.asyncio
    async def test_fetch_positions_and_balance_uses_dex_scoped_fetch_for_hip3_state(
        self, bot_class
    ):
        """Symbol-less HIP-3 state refresh must use dex-scoped fetch_positions calls."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = ["XYZ-XYZ100/USDC:USDC"]
        bot.open_orders = {}
        bot.positions = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
        }
        bot.coin_to_symbol = lambda coin: f"{coin}/USDC:USDC"
        bot.cca = MagicMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "info": {
                    "assetPositions": [],
                    "marginSummary": {"accountValue": "1000.0"},
                }
            }
        )
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "XYZ-XYZ100/USDC:USDC",
                    "side": "long",
                    "contracts": 0.0009,
                    "entryPrice": 24982.0,
                }
            ]
        )

        positions, balance = await bot._fetch_positions_and_balance()

        assert balance == 1000.0
        assert positions == [
            {
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "position_side": "long",
                "size": 0.0009,
                "price": 24982.0,
                "margin_mode": None,
                "margin_used": 0.0,
            }
        ]
        bot.cca.fetch_positions.assert_awaited_once_with(params={"dex": "xyz"})

    @pytest.mark.asyncio
    async def test_fetch_positions_and_balance_bootstraps_approved_hip3_positions_via_dex_on_startup(
        self, bot_class
    ):
        """Startup must use dex-scoped discovery even when HIP-3 approvals exist."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.coin_to_symbol = lambda coin: f"{coin}/USDC:USDC"
        bot.cca = MagicMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "info": {
                    "assetPositions": [],
                    "marginSummary": {"accountValue": "1000.0"},
                }
            }
        )
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "XYZ-XYZ100/USDC:USDC",
                    "side": "long",
                    "contracts": 0.0009,
                    "entryPrice": 24982.0,
                }
            ]
        )

        positions, balance = await bot._fetch_positions_and_balance()

        assert balance == 1000.0
        assert positions == [
            {
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "position_side": "long",
                "size": 0.0009,
                "price": 24982.0,
                "margin_mode": None,
                "margin_used": 0.0,
            }
        ]
        bot.cca.fetch_positions.assert_awaited_once_with(params={"dex": "xyz"})

    @pytest.mark.asyncio
    async def test_fetch_positions_and_balance_bootstraps_mixed_approved_unapproved_hip3_positions_via_dex(
        self, bot_class
    ):
        """Startup must not let approved HIP-3 symbols hide different unapproved live HIP-3 positions."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.coin_to_symbol = lambda coin: f"{coin}/USDC:USDC"
        bot.cca = MagicMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "info": {
                    "assetPositions": [],
                    "marginSummary": {"accountValue": "1000.0"},
                }
            }
        )
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "XYZ-XYZ200/USDC:USDC",
                    "side": "long",
                    "contracts": 0.0009,
                    "entryPrice": 24982.0,
                    "marginMode": "cross",
                }
            ]
        )

        positions, balance = await bot._fetch_positions_and_balance()

        assert balance == 1000.0
        assert positions == [
            {
                "symbol": "XYZ-XYZ200/USDC:USDC",
                "position_side": "long",
                "size": 0.0009,
                "price": 24982.0,
                "margin_mode": "cross",
                "margin_used": 0.0,
            }
        ]
        bot.cca.fetch_positions.assert_awaited_once_with(params={"dex": "xyz"})

    @pytest.mark.asyncio
    async def test_fetch_positions_and_balance_steady_state_discovers_untracked_hip3_positions_via_dex(
        self, bot_class
    ):
        """Once any HIP-3 symbol is tracked, dex-wide discovery must still find new external HIP-3 state."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = ["XYZ-XYZ100/USDC:USDC"]
        bot.open_orders = {"XYZ-XYZ100/USDC:USDC": [{"id": "tracked"}]}
        bot.positions = {
            "XYZ-XYZ100/USDC:USDC": {
                "long": {"size": 0.001, "price": 24000.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.coin_to_symbol = lambda coin: f"{coin}/USDC:USDC"
        bot.cca = MagicMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "info": {
                    "assetPositions": [],
                    "marginSummary": {"accountValue": "1000.0"},
                }
            }
        )
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "XYZ-XYZ200/USDC:USDC",
                    "side": "long",
                    "contracts": 0.0009,
                    "entryPrice": 24982.0,
                    "marginMode": "cross",
                }
            ]
        )

        positions, balance = await bot._fetch_positions_and_balance()

        assert balance == 1000.0
        assert positions == [
            {
                "symbol": "XYZ-XYZ200/USDC:USDC",
                "position_side": "long",
                "size": 0.0009,
                "price": 24982.0,
                "margin_mode": "cross",
                "margin_used": 0.0,
            }
        ]
        bot.cca.fetch_positions.assert_awaited_once_with(params={"dex": "xyz"})

    @pytest.mark.asyncio
    async def test_fetch_positions_and_balance_bootstraps_unapproved_hip3_positions_via_dex(
        self, bot_class
    ):
        """Startup must discover pre-existing unapproved HIP-3 positions via dex-scoped queries."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.coin_to_symbol = lambda coin: f"{coin}/USDC:USDC"
        bot.cca = MagicMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "info": {
                    "assetPositions": [],
                    "marginSummary": {"accountValue": "1000.0"},
                }
            }
        )
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "XYZ-XYZ100/USDC:USDC",
                    "side": "long",
                    "contracts": 0.0009,
                    "entryPrice": 24982.0,
                    "marginMode": "cross",
                }
            ]
        )

        positions, balance = await bot._fetch_positions_and_balance()

        assert balance == 1000.0
        assert positions == [
            {
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "position_side": "long",
                "size": 0.0009,
                "price": 24982.0,
                "margin_mode": "cross",
                "margin_used": 0.0,
            }
        ]
        bot.cca.fetch_positions.assert_awaited_once_with(params={"dex": "xyz"})

    def test_isolated_only_hip3_is_ignored_for_new_entries(self, bot_class, caplog):
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.markets_dict = {
            "xyz:TSLA/USDC:USDC": {
                "info": {"onlyIsolated": True},
                "marginModes": {"cross": False, "isolated": True},
            }
        }

        with caplog.at_level("WARNING"):
            filtered = bot._filter_approved_symbols("long", {"xyz:TSLA/USDC:USDC"})

        assert filtered == set()
        assert "isolated margin is currently unsupported" in caplog.text

    def test_isolated_only_hip3_open_orders_hard_fail(self, bot_class):
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.markets_dict = {
            "xyz:SP500/USDC:USDC": {
                "baseName": "xyz:SP500",
                "info": {"onlyIsolated": True},
                "marginModes": {"cross": False, "isolated": True},
            }
        }
        bot.positions = {
            "xyz:SP500/USDC:USDC": {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        bot.open_orders = {"xyz:SP500/USDC:USDC": [{"id": "1"}]}
        bot._hl_live_margin_modes = {}

        with pytest.raises(NotImplementedError, match="Unsupported live state detected"):
            bot._assert_supported_live_state()

    @pytest.mark.asyncio
    async def test_fetch_open_orders_queries_explicit_hip3_symbol_with_symbol_scope(self, bot_class):
        """Explicit HIP-3 symbol queries must still use symbol-scoped order routes."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = ["XYZ-XYZ100/USDC:USDC"]
        bot.open_orders = {}
        bot.positions = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
        }
        bot.determine_pos_side = lambda order: "long"
        bot.cca = MagicMock()

        async def fetch_open_orders(symbol=None, params=None):
            if params is not None:
                return []
            if symbol is None:
                return []
            return [
                {
                    "id": "abc123",
                    "symbol": symbol,
                    "side": "buy",
                    "amount": 0.0009,
                    "price": 24980.0,
                    "timestamp": 1000,
                }
            ]

        bot.cca.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)

        orders = await bot.fetch_open_orders(symbol="XYZ-XYZ100/USDC:USDC")

        assert orders == [
            {
                "id": "abc123",
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "side": "buy",
                "amount": 0.0009,
                "price": 24980.0,
                "timestamp": 1000,
                "position_side": "long",
                "qty": 0.0009,
            }
        ]
        assert bot.cca.fetch_open_orders.await_args_list[0].kwargs == {
            "symbol": "XYZ-XYZ100/USDC:USDC"
        }

    @pytest.mark.asyncio
    async def test_fetch_open_orders_bootstraps_approved_hip3_orders_via_dex_on_startup(
        self, bot_class
    ):
        """Startup must use dex-scoped order discovery even when HIP-3 approvals exist."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.determine_pos_side = lambda order: "long"
        bot.cca = MagicMock()

        async def fetch_open_orders(symbol=None, params=None):
            if symbol is None and params is None:
                return []
            if params != {"dex": "xyz"}:
                return []
            return [
                {
                    "id": "abc123",
                    "symbol": "XYZ-XYZ100/USDC:USDC",
                    "side": "buy",
                    "amount": 0.0009,
                    "price": 24980.0,
                    "timestamp": 1000,
                }
            ]

        bot.cca.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)

        orders = await bot.fetch_open_orders()

        assert orders == [
            {
                "id": "abc123",
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "side": "buy",
                "amount": 0.0009,
                "price": 24980.0,
                "timestamp": 1000,
                "position_side": "long",
                "qty": 0.0009,
            }
        ]
        assert bot.cca.fetch_open_orders.await_args_list[0].kwargs == {"symbol": None}
        assert bot.cca.fetch_open_orders.await_args_list[1].kwargs == {"params": {"dex": "xyz"}}

    @pytest.mark.asyncio
    async def test_fetch_open_orders_bootstraps_mixed_approved_unapproved_hip3_orders_via_dex(
        self, bot_class
    ):
        """Startup must not let approved HIP-3 symbols hide unapproved live HIP-3 orders."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.determine_pos_side = lambda order: "long"
        bot.cca = MagicMock()

        async def fetch_open_orders(symbol=None, params=None):
            if symbol is None and params is None:
                return []
            if params != {"dex": "xyz"}:
                return []
            return [
                {
                    "id": "abc123",
                    "symbol": "XYZ-XYZ200/USDC:USDC",
                    "side": "buy",
                    "amount": 0.0009,
                    "price": 24980.0,
                    "timestamp": 1000,
                }
            ]

        bot.cca.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)

        orders = await bot.fetch_open_orders()

        assert orders == [
            {
                "id": "abc123",
                "symbol": "XYZ-XYZ200/USDC:USDC",
                "side": "buy",
                "amount": 0.0009,
                "price": 24980.0,
                "timestamp": 1000,
                "position_side": "long",
                "qty": 0.0009,
            }
        ]
        assert bot.cca.fetch_open_orders.await_args_list[0].kwargs == {"symbol": None}
        assert bot.cca.fetch_open_orders.await_args_list[1].kwargs == {"params": {"dex": "xyz"}}

    @pytest.mark.asyncio
    async def test_fetch_open_orders_steady_state_discovers_untracked_hip3_orders_via_dex(
        self, bot_class
    ):
        """Once any HIP-3 symbol is tracked, dex-wide order discovery must still find new external HIP-3 orders."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = ["XYZ-XYZ100/USDC:USDC"]
        bot.open_orders = {"XYZ-XYZ100/USDC:USDC": [{"id": "tracked"}]}
        bot.positions = {
            "XYZ-XYZ100/USDC:USDC": {
                "long": {"size": 0.001, "price": 24000.0},
                "short": {"size": 0.0, "price": 0.0},
            }
        }
        bot.approved_coins_minus_ignored_coins = {
            "long": {"XYZ-XYZ100/USDC:USDC"},
            "short": set(),
        }
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.determine_pos_side = lambda order: "long"
        bot.cca = MagicMock()

        async def fetch_open_orders(symbol=None, params=None):
            if symbol is None and params is None:
                return []
            if params != {"dex": "xyz"}:
                return []
            return [
                {
                    "id": "abc123",
                    "symbol": "XYZ-XYZ200/USDC:USDC",
                    "side": "buy",
                    "amount": 0.0009,
                    "price": 24980.0,
                    "timestamp": 1000,
                }
            ]

        bot.cca.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)

        orders = await bot.fetch_open_orders()

        assert orders == [
            {
                "id": "abc123",
                "symbol": "XYZ-XYZ200/USDC:USDC",
                "side": "buy",
                "amount": 0.0009,
                "price": 24980.0,
                "timestamp": 1000,
                "position_side": "long",
                "qty": 0.0009,
            }
        ]
        assert bot.cca.fetch_open_orders.await_args_list[0].kwargs == {"symbol": None}
        assert bot.cca.fetch_open_orders.await_args_list[1].kwargs == {"params": {"dex": "xyz"}}

    @pytest.mark.asyncio
    async def test_fetch_open_orders_bootstraps_unapproved_hip3_orders_via_dex(self, bot_class):
        """Startup must discover pre-existing unapproved HIP-3 open orders via dex-scoped queries."""
        bot = object.__new__(bot_class)
        bot.HIP3_PREFIX = bot_class.HIP3_PREFIX
        bot.HIP3_ALT_PREFIXES = bot_class.HIP3_ALT_PREFIXES
        bot.active_symbols = []
        bot.open_orders = {}
        bot.positions = {}
        bot.approved_coins_minus_ignored_coins = {"long": set(), "short": set()}
        bot.coin_overrides = {}
        bot._hl_live_margin_modes = {}
        bot.markets_dict = {
            "XYZ-XYZ100/USDC:USDC": {"baseName": "xyz:XYZ100", "info": {}},
            "XYZ-XYZ200/USDC:USDC": {"baseName": "xyz:XYZ200", "info": {}},
            "BTC/USDC:USDC": {"baseName": "BTC", "info": {}},
        }
        bot.determine_pos_side = lambda order: "long"
        bot.cca = MagicMock()

        async def fetch_open_orders(symbol=None, params=None):
            if symbol is None and params is None:
                return []
            if params == {"dex": "xyz"}:
                return [
                    {
                        "id": "abc123",
                        "symbol": "XYZ-XYZ100/USDC:USDC",
                        "side": "buy",
                        "amount": 0.0009,
                        "price": 24980.0,
                        "timestamp": 1000,
                    }
                ]
            return []

        bot.cca.fetch_open_orders = AsyncMock(side_effect=fetch_open_orders)

        orders = await bot.fetch_open_orders()

        assert orders == [
            {
                "id": "abc123",
                "symbol": "XYZ-XYZ100/USDC:USDC",
                "side": "buy",
                "amount": 0.0009,
                "price": 24980.0,
                "timestamp": 1000,
                "position_side": "long",
                "qty": 0.0009,
            }
        ]
        assert bot.cca.fetch_open_orders.await_args_list[0].kwargs == {"symbol": None}
        assert bot.cca.fetch_open_orders.await_args_list[1].kwargs == {"params": {"dex": "xyz"}}


class TestIsolatedMarginLeverageCapping:
    """Tests for isolated margin leverage capping."""

    def test_leverage_capped_for_isolated_symbols(self, stubbed_modules):
        """Test that isolated margin symbols have leverage capped appropriately."""
        from exchanges.hyperliquid import HyperliquidBot

        # Create minimal bot instance
        bot = object.__new__(HyperliquidBot)
        bot.HIP3_PREFIX = HyperliquidBot.HIP3_PREFIX
        bot.HIP3_MAX_LEVERAGE = HyperliquidBot.HIP3_MAX_LEVERAGE
        bot.max_leverage = {}
        bot.min_costs = {}
        bot.min_qtys = {}
        bot.qty_steps = {}
        bot.price_steps = {}
        bot.c_mults = {}
        bot.symbol_ids = {}

        # Mock markets with different max leverage values
        bot.markets_dict = {
            "xyz:TSLA/USDC:USDC": {
                "id": "TSLA",
                "info": {"onlyIsolated": True, "maxLeverage": "20"},
                "limits": {"cost": {"min": 10}, "amount": {"min": 0.01}},
                "precision": {"amount": 0.01, "price": 0.01},
                "contractSize": 1,
            },
            "BTC/USDC:USDC": {
                "id": "BTC",
                "info": {"onlyIsolated": False, "maxLeverage": "50"},
                "limits": {"cost": {"min": 10}, "amount": {"min": 0.001}},
                "precision": {"amount": 0.001, "price": 0.1},
                "contractSize": 1,
            },
        }

        # Simulate set_market_specific_settings logic
        import passivbot_rust as pbr

        for symbol, elm in bot.markets_dict.items():
            bot.symbol_ids[symbol] = elm["id"]
            bot.min_costs[symbol] = pbr.round_(10.0 * 1.01, 0.01)
            bot.qty_steps[symbol] = elm["precision"]["amount"]
            bot.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            bot.price_steps[symbol] = elm["precision"]["price"]
            bot.c_mults[symbol] = elm["contractSize"]

            if bot._requires_isolated_margin(symbol):
                bot.max_leverage[symbol] = min(
                    bot.HIP3_MAX_LEVERAGE,
                    int(elm["info"]["maxLeverage"]),
                )
            else:
                bot.max_leverage[symbol] = int(elm["info"]["maxLeverage"])

        # Isolated-only symbol should be capped at 10x
        assert bot.max_leverage["xyz:TSLA/USDC:USDC"] == 10
        assert bot._requires_isolated_margin("xyz:TSLA/USDC:USDC") is True

        # Cross margin symbol should use full leverage
        assert bot.max_leverage["BTC/USDC:USDC"] == 50
        assert bot._requires_isolated_margin("BTC/USDC:USDC") is False


class TestStockTickerDetection:
    """Tests for automatic stock ticker detection."""

    def test_is_stock_ticker_known_tickers(self):
        """Test detection of known stock tickers."""
        from tradfi_data import is_stock_ticker

        assert is_stock_ticker("TSLA") is True
        assert is_stock_ticker("NVDA") is True
        assert is_stock_ticker("AAPL") is True
        assert is_stock_ticker("xyz:TSLA") is True  # With prefix
        assert is_stock_ticker("tsla") is True  # Case insensitive

    def test_is_stock_ticker_crypto(self):
        """Test that crypto coins are not detected as stock tickers."""
        from tradfi_data import is_stock_ticker

        assert is_stock_ticker("BTC") is False
        assert is_stock_ticker("ETH") is False
        assert is_stock_ticker("SOL") is False

    def test_is_stock_ticker_with_ccxt_format(self):
        """Test stock ticker detection with CCXT symbol format."""
        from tradfi_data import is_stock_ticker

        assert is_stock_ticker("TSLA/USDC") is True
        assert is_stock_ticker("xyz:TSLA/USDC:USDC") is True


class TestTradFiProvider:
    """Tests for TradFi data provider."""

    def test_get_provider_finnhub(self):
        """Test getting Finnhub provider."""
        from tradfi_data import get_provider, FinnhubProvider

        provider = get_provider("finnhub", api_key="test_key")
        assert isinstance(provider, FinnhubProvider)
        assert provider.api_key == "test_key"
        assert provider.name == "finnhub"

    def test_get_provider_alphavantage(self):
        """Test getting Alpha Vantage provider."""
        from tradfi_data import get_provider, AlphaVantageProvider

        provider = get_provider("alphavantage", api_key="test_key")
        assert isinstance(provider, AlphaVantageProvider)
        assert provider.api_key == "test_key"
        assert provider.name == "alphavantage"

    def test_get_provider_unknown(self):
        """Test that unknown provider raises error."""
        from tradfi_data import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_candles_to_array_empty(self):
        """Test converting empty candle list."""
        from tradfi_data import candles_to_array

        arr = candles_to_array([])
        assert arr.size == 0
        assert arr.dtype.names == ("ts", "o", "h", "l", "c", "bv")

    def test_candles_to_array(self):
        """Test converting candle list to array."""
        from tradfi_data import candles_to_array, TradFiCandle

        candles = [
            TradFiCandle(
                timestamp_ms=1704067200000,
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000.0,
            ),
            TradFiCandle(
                timestamp_ms=1704067260000,
                open=103.0,
                high=107.0,
                low=102.0,
                close=106.0,
                volume=1500.0,
            ),
        ]

        arr = candles_to_array(candles)

        assert len(arr) == 2
        assert arr[0]["ts"] == 1704067200000
        assert arr[0]["o"] == pytest.approx(100.0)
        assert arr[0]["h"] == pytest.approx(105.0)
        assert arr[0]["l"] == pytest.approx(99.0)
        assert arr[0]["c"] == pytest.approx(103.0)
        assert arr[0]["bv"] == pytest.approx(1000.0)


class TestMinIsolatedLeverage:
    """Tests for automatic isolated margin leverage calculation."""

    def test_calc_min_isolated_leverage_basic(self):
        """Test minimum leverage calculation based on TWEL."""
        import math

        # TWEL 1.25 -> ceil(1.25) = 2
        assert math.ceil(1.25) == 2

        # TWEL 2.0 -> ceil(2.0) = 2
        assert math.ceil(2.0) == 2

        # TWEL 2.5 -> ceil(2.5) = 3
        assert math.ceil(2.5) == 3

    def test_leverage_formula_ensures_margin_coverage(self):
        """Test that the leverage formula ensures margin can always be covered."""
        import math

        test_cases = [
            # (TWEL, expected_min_leverage)
            (1.0, 1),
            (1.25, 2),
            (1.5, 2),
            (2.0, 2),
            (2.5, 3),
            (3.0, 3),
            (5.0, 5),
        ]

        balance = 100_000  # $100k

        for twel, expected_min_lev in test_cases:
            min_leverage = max(1, math.ceil(twel))
            assert (
                min_leverage == expected_min_lev
            ), f"TWEL {twel} should need {expected_min_lev}x leverage"

            # Verify margin requirement <= balance
            max_exposure = twel * balance
            margin_required = max_exposure / min_leverage
            assert (
                margin_required <= balance
            ), f"TWEL {twel} with {min_leverage}x: margin {margin_required} > balance {balance}"


class TestAvailableStockPerps:
    """Tests for available stock perps list."""

    def test_available_stock_perps_list(self):
        """Test that available stock perps list contains expected symbols."""
        from tradfi_data import AVAILABLE_STOCK_PERPS

        assert "xyz:TSLA" in AVAILABLE_STOCK_PERPS
        assert "xyz:NVDA" in AVAILABLE_STOCK_PERPS
        assert "xyz:AAPL" in AVAILABLE_STOCK_PERPS
        assert "xyz:MSFT" in AVAILABLE_STOCK_PERPS

    def test_all_stock_perps_have_xyz_prefix(self):
        """Test that all available stock perps have xyz: prefix."""
        from tradfi_data import AVAILABLE_STOCK_PERPS

        for symbol in AVAILABLE_STOCK_PERPS:
            assert symbol.startswith("xyz:"), f"{symbol} missing xyz: prefix"
