# tests/exchanges/test_ccxt_bot.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestCCXTBotSessionCreation:
    """Test CCXTBot CCXT session initialization."""

    def test_build_ccxt_config_with_api_key(self):
        """Should build CCXT config from api-keys with CCXT credential fields."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to create instance without calling __init__
        # This bypasses complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        # Use CCXT field names directly (apiKey, secret, password)
        bot.user_info = {
            "exchange": "binance",
            "apiKey": "test_key",
            "secret": "test_secret",
            "password": "test_pass",
            "quote": "USDT",
        }

        config = bot._build_ccxt_config()

        assert config["apiKey"] == "test_key"
        assert config["secret"] == "test_secret"
        assert config["password"] == "test_pass"
        assert config["enableRateLimit"] is True
        assert "walletAddress" not in config  # Not provided

    def test_build_ccxt_config_with_wallet(self):
        """Should build CCXT config from api-keys with CCXT wallet fields."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to create instance without calling __init__
        bot = CCXTBot.__new__(CCXTBot)
        # Use CCXT field names directly (walletAddress, privateKey)
        bot.user_info = {
            "exchange": "hyperliquid",
            "walletAddress": "0xABC123",
            "privateKey": "0xDEF456",
            "quote": "USDC",
        }

        config = bot._build_ccxt_config()

        assert config["walletAddress"] == "0xABC123"
        assert config["privateKey"] == "0xDEF456"
        assert "apiKey" not in config  # Not provided


class TestCCXTBotFetchBalance:
    """Test CCXTBot balance fetching."""

    @pytest.mark.asyncio
    async def test_fetch_balance_returns_quote_total(self):
        """Should return total balance for quote currency."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to bypass complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.quote = "USDT"
        bot.cca = AsyncMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "total": {"USDT": 1000.50, "BTC": 0.5},
                "free": {"USDT": 900.0},
            }
        )

        balance = await bot.fetch_balance()

        assert balance == 1000.50

    @pytest.mark.asyncio
    async def test_fetch_balance_returns_zero_when_missing(self):
        """Should return 0 when quote currency not in balance."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to bypass complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.quote = "USDC"
        bot.cca = AsyncMock()
        bot.cca.fetch_balance = AsyncMock(
            return_value={
                "total": {"USDT": 1000.0},
            }
        )

        balance = await bot.fetch_balance()

        assert balance == 0.0


class TestCCXTBotFetchPositions:
    """Test CCXTBot position fetching."""

    @pytest.mark.asyncio
    async def test_fetch_positions_returns_open_positions(self):
        """Should return list of open positions with normalized fields."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to bypass complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "long",
                    "contracts": 0.5,
                    "entryPrice": 50000.0,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "short",
                    "contracts": 2.0,
                    "entryPrice": 3000.0,
                },
                {
                    "symbol": "SOL/USDT:USDT",
                    "side": "long",
                    "contracts": 0,  # No position
                    "entryPrice": 0,
                },
            ]
        )

        positions = await bot.fetch_positions()

        assert len(positions) == 2
        assert positions[0]["symbol"] == "BTC/USDT:USDT"
        assert positions[0]["position_side"] == "long"
        assert positions[0]["size"] == 0.5
        assert positions[0]["price"] == 50000.0
        assert positions[1]["position_side"] == "short"

    @pytest.mark.asyncio
    async def test_fetch_positions_handles_empty(self):
        """Should return empty list when no positions."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to bypass complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_positions = AsyncMock(return_value=[])

        positions = await bot.fetch_positions()

        assert positions == []


class TestCCXTBotFetchOpenOrders:
    """Test CCXTBot open order fetching."""

    @pytest.mark.asyncio
    async def test_fetch_open_orders_normalizes_fields(self):
        """Should return orders with position_side and qty fields."""
        from exchanges.ccxt_bot import CCXTBot

        # Use __new__ to bypass complex Passivbot initialization
        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_open_orders = AsyncMock(
            return_value=[
                {
                    "id": "123",
                    "symbol": "BTC/USDT:USDT",
                    "side": "buy",
                    "amount": 0.1,
                    "price": 50000.0,
                    "timestamp": 1000,
                    "info": {"positionSide": "LONG"},
                },
                {
                    "id": "456",
                    "symbol": "BTC/USDT:USDT",
                    "side": "sell",
                    "amount": 0.2,
                    "price": 55000.0,
                    "timestamp": 2000,
                    "info": {},  # No positionSide
                },
            ]
        )

        orders = await bot.fetch_open_orders()

        assert len(orders) == 2
        assert orders[0]["id"] == "123"
        assert orders[0]["position_side"] == "long"
        assert orders[0]["qty"] == 0.1
        assert orders[1]["position_side"] == "both"  # Fallback
        assert orders[1]["qty"] == 0.2
        # Should be sorted by timestamp
        assert orders[0]["timestamp"] < orders[1]["timestamp"]


class TestCCXTBotWatchOrders:
    """Tests for watch_orders WebSocket handler."""

    @pytest.mark.asyncio
    async def test_watch_orders_normalizes_fields(self):
        """Test that watch_orders normalizes position_side and qty."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.stop_websocket = False
        bot.exchange = "test_exchange"

        # Track calls to handle_order_update
        handled_orders = []
        bot.handle_order_update = lambda orders: handled_orders.append(orders)

        # Mock WebSocket client
        call_count = [0]

        async def mock_watch_orders():
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"id": "order1", "amount": 0.5, "info": {"positionSide": "LONG"}}]
            else:
                # Stop after processing first batch
                bot.stop_websocket = True
                # Return order that triggers stop on next iteration
                return [{"id": "order2", "amount": 1.0, "info": {}}]

        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": True}  # Required for can_watch_orders()
        bot.ccp.watch_orders = mock_watch_orders

        await bot.watch_orders()

        # First batch processed and has proper normalization
        assert len(handled_orders) >= 1
        order = handled_orders[0][0]
        assert order["position_side"] == "long"
        assert order["qty"] == 0.5


class TestCCXTBotUpdateExchangeConfig:
    """Tests for update_exchange_config."""

    @pytest.mark.asyncio
    async def test_update_exchange_config_sets_hedge_mode(self):
        """Test that hedge mode is set when exchange supports it."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"

        bot.cca = MagicMock()
        bot.cca.has = {"setPositionMode": True}
        bot.cca.set_position_mode = AsyncMock(return_value={"result": "success"})

        await bot.update_exchange_config()

        bot.cca.set_position_mode.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_update_exchange_config_skips_when_unsupported(self):
        """Test that method skips gracefully when exchange doesn't support position mode."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"

        bot.cca = MagicMock()
        bot.cca.has = {"setPositionMode": False}
        bot.cca.set_position_mode = AsyncMock()

        await bot.update_exchange_config()

        # Should not be called when not supported
        bot.cca.set_position_mode.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_exchange_config_skips_when_key_missing(self):
        """Test that method skips gracefully when setPositionMode key is missing."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"

        bot.cca = MagicMock()
        bot.cca.has = {}  # No setPositionMode key at all
        bot.cca.set_position_mode = AsyncMock()

        await bot.update_exchange_config()

        # Should not be called when key is missing
        bot.cca.set_position_mode.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_exchange_config_reraises_exception(self):
        """Test that method logs and re-raises exceptions."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"

        bot.cca = MagicMock()
        bot.cca.has = {"setPositionMode": True}
        bot.cca.set_position_mode = AsyncMock(side_effect=Exception("API error"))

        # Should log and re-raise (fail loudly)
        with pytest.raises(Exception, match="API error"):
            await bot.update_exchange_config()


class TestCCXTBotUpdateExchangeConfigBySymbols:
    """Tests for update_exchange_config_by_symbols."""

    @pytest.mark.asyncio
    async def test_sets_leverage_when_supported(self):
        """Test that leverage is set for each symbol when supported."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.has = {"setLeverage": True, "setMarginMode": False}
        bot.cca.set_leverage = AsyncMock()
        bot.config_get = MagicMock(return_value=10)

        await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT", "ETH/USDT:USDT"])

        assert bot.cca.set_leverage.call_count == 2
        bot.cca.set_leverage.assert_any_call(10, symbol="BTC/USDT:USDT")
        bot.cca.set_leverage.assert_any_call(10, symbol="ETH/USDT:USDT")

    @pytest.mark.asyncio
    async def test_sets_margin_mode_when_supported(self):
        """Test that margin mode is set for each symbol when supported."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.has = {"setLeverage": False, "setMarginMode": True}
        bot.cca.set_margin_mode = AsyncMock()
        bot.config_get = MagicMock(return_value=10)

        await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

        bot.cca.set_margin_mode.assert_called_once_with("cross", symbol="BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_skips_unsupported_operations(self):
        """Test that unsupported operations are skipped gracefully."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.cca = MagicMock()
        bot.cca.has = {}  # Neither supported
        bot.cca.set_leverage = AsyncMock()
        bot.cca.set_margin_mode = AsyncMock()

        await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

        bot.cca.set_leverage.assert_not_called()
        bot.cca.set_margin_mode.assert_not_called()

    @pytest.mark.asyncio
    async def test_leverage_error_is_logged_and_reraised(self):
        """Test that leverage errors are logged and re-raised."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.has = {"setLeverage": True}
        bot.cca.set_leverage = AsyncMock(side_effect=Exception("Leverage API error"))
        bot.config_get = MagicMock(return_value=10)

        with pytest.raises(Exception, match="Leverage API error"):
            await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])

    @pytest.mark.asyncio
    async def test_margin_mode_error_is_logged_and_reraised(self):
        """Test that margin mode errors are logged and re-raised."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.has = {"setMarginMode": True}
        bot.cca.set_margin_mode = AsyncMock(side_effect=Exception("Margin API error"))
        bot.config_get = MagicMock(return_value=10)

        with pytest.raises(Exception, match="Margin API error"):
            await bot.update_exchange_config_by_symbols(["BTC/USDT:USDT"])


class TestCCXTBotSetMarketSpecificSettings:
    """Tests for set_market_specific_settings."""

    def test_extracts_market_settings(self):
        """Test that market settings are extracted from CCXT market info."""
        from exchanges.ccxt_bot import CCXTBot
        from passivbot import Passivbot

        bot = CCXTBot.__new__(CCXTBot)
        # Initialize empty dicts that parent class would create
        bot.symbol_ids = {}
        bot.min_costs = {}
        bot.min_qtys = {}
        bot.qty_steps = {}
        bot.price_steps = {}
        bot.c_mults = {}

        bot.markets_dict = {
            "BTC/USDT:USDT": {
                "id": "BTCUSDT",
                "limits": {
                    "cost": {"min": 5.0},
                    "amount": {"min": 0.001},
                },
                "precision": {
                    "amount": 0.001,
                    "price": 0.01,
                },
                "contractSize": 1,
            }
        }

        # Mock parent's method
        with patch.object(Passivbot, "set_market_specific_settings", lambda self: None):
            bot.set_market_specific_settings()

        assert bot.symbol_ids["BTC/USDT:USDT"] == "BTCUSDT"
        assert bot.min_costs["BTC/USDT:USDT"] == 5.0
        assert bot.min_qtys["BTC/USDT:USDT"] == 0.001
        assert bot.qty_steps["BTC/USDT:USDT"] == 0.001
        assert bot.price_steps["BTC/USDT:USDT"] == 0.01
        assert bot.c_mults["BTC/USDT:USDT"] == 1

    def test_uses_default_min_cost_when_none(self):
        """Test that min_cost defaults to 0.1 when exchange returns None."""
        from exchanges.ccxt_bot import CCXTBot
        from passivbot import Passivbot

        bot = CCXTBot.__new__(CCXTBot)
        bot.symbol_ids = {}
        bot.min_costs = {}
        bot.min_qtys = {}
        bot.qty_steps = {}
        bot.price_steps = {}
        bot.c_mults = {}

        bot.markets_dict = {
            "ETH/USDT:USDT": {
                "id": "ETHUSDT",
                "limits": {
                    "cost": {"min": None},  # Some exchanges don't provide this
                    "amount": {"min": 0.01},
                },
                "precision": {
                    "amount": 0.01,
                    "price": 0.01,
                },
            }
        }

        with patch.object(Passivbot, "set_market_specific_settings", lambda self: None):
            bot.set_market_specific_settings()

        assert bot.min_costs["ETH/USDT:USDT"] == 0.1  # Default fallback
        assert bot.c_mults["ETH/USDT:USDT"] == 1  # Default when missing


class TestCCXTBotFetchTickers:
    """Tests for fetch_tickers."""

    @pytest.mark.asyncio
    async def test_fetch_tickers_returns_filtered_data(self):
        """Test that tickers are filtered to markets_dict symbols."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.markets_dict = {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}
        bot.cca = AsyncMock()
        bot.cca.fetch_tickers = AsyncMock(
            return_value={
                "BTC/USDT:USDT": {"bid": 50000.0, "ask": 50010.0, "last": 50005.0},
                "ETH/USDT:USDT": {"bid": 3000.0, "ask": 3001.0, "last": 3000.5},
                "DOGE/USDT:USDT": {"bid": 0.1, "ask": 0.11, "last": 0.105},  # Not in markets_dict
            }
        )

        tickers = await bot.fetch_tickers()

        assert len(tickers) == 2
        assert "BTC/USDT:USDT" in tickers
        assert "ETH/USDT:USDT" in tickers
        assert "DOGE/USDT:USDT" not in tickers
        assert tickers["BTC/USDT:USDT"]["bid"] == 50000.0
        assert tickers["BTC/USDT:USDT"]["ask"] == 50010.0
        assert tickers["BTC/USDT:USDT"]["last"] == 50005.0

    @pytest.mark.asyncio
    async def test_fetch_tickers_handles_missing_values(self):
        """Test that missing ticker values default to 0 or fallback."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.markets_dict = {"BTC/USDT:USDT": {}}
        bot.cca = AsyncMock()
        bot.cca.fetch_tickers = AsyncMock(
            return_value={
                "BTC/USDT:USDT": {"bid": 50000.0, "ask": None, "last": None},
            }
        )

        tickers = await bot.fetch_tickers()

        assert tickers["BTC/USDT:USDT"]["bid"] == 50000.0
        assert tickers["BTC/USDT:USDT"]["ask"] == 0  # None -> 0
        assert tickers["BTC/USDT:USDT"]["last"] == 50000.0  # Falls back to bid

    @pytest.mark.asyncio
    async def test_fetch_tickers_raises_on_error(self):
        """Test that method raises on error (caller handles exceptions)."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.markets_dict = {"BTC/USDT:USDT": {}}
        bot.cca = AsyncMock()
        bot.cca.fetch_tickers = AsyncMock(side_effect=Exception("Network error"))

        with pytest.raises(Exception, match="Network error"):
            await bot.fetch_tickers()


class TestCCXTBotFetchOHLCV:
    """Tests for fetch_ohlcv and fetch_ohlcvs_1m."""

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_returns_candles(self):
        """Test that fetch_ohlcv returns candlestick data."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_ohlcv = AsyncMock(
            return_value=[
                [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
                [1704067260000, 42050.0, 42150.0, 42000.0, 42100.0, 150.0],
            ]
        )

        result = await bot.fetch_ohlcv("BTC/USDT:USDT", "1m")

        assert len(result) == 2
        assert result[0][0] == 1704067200000  # timestamp
        bot.cca.fetch_ohlcv.assert_called_with("BTC/USDT:USDT", timeframe="1m", limit=1000)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_raises_on_error(self):
        """Test that fetch_ohlcv raises on error (caller handles exceptions)."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_ohlcv = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await bot.fetch_ohlcv("BTC/USDT:USDT")

    @pytest.mark.asyncio
    async def test_fetch_ohlcvs_1m_without_since(self):
        """Test fetch_ohlcvs_1m without since parameter."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()
        bot.cca.fetch_ohlcv = AsyncMock(
            return_value=[
                [1704067200000, 42000.0, 42100.0, 41900.0, 42050.0, 100.0],
            ]
        )

        result = await bot.fetch_ohlcvs_1m("BTC/USDT:USDT")

        assert len(result) == 1
        bot.cca.fetch_ohlcv.assert_called_with("BTC/USDT:USDT", timeframe="1m", limit=1000)

    @pytest.mark.asyncio
    async def test_fetch_ohlcvs_1m_with_pagination(self):
        """Test fetch_ohlcvs_1m paginates when needed."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = AsyncMock()

        # Simulate two pages of results
        call_count = [0]

        async def mock_fetch(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [[1704067200000, 1, 2, 3, 4, 5]] * 1000  # Full page
            else:
                return [[1704067260000, 1, 2, 3, 4, 5]] * 500  # Partial page (stops)

        bot.cca.fetch_ohlcv = mock_fetch

        result = await bot.fetch_ohlcvs_1m("BTC/USDT:USDT", since=1704067200000)

        assert call_count[0] == 2  # Two paginated calls


class TestCCXTBotValidateWebSocketSupport:
    """Tests for validate_websocket_support."""

    @pytest.mark.asyncio
    async def test_validates_watch_orders_support(self):
        """Test that method passes when watchOrders is supported."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": True}

        # Should not raise
        await bot.validate_websocket_support()
