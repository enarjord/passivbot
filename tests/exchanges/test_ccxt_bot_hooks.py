"""Tests for CCXTBot Template Method hooks."""

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestCanWatchOrders:
    """Test can_watch_orders() hook."""

    def test_returns_false_when_ccp_is_none(self):
        """Should return False when WebSocket client is None."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = None

        assert bot.can_watch_orders() is False

    def test_returns_false_when_watch_orders_not_supported(self):
        """Should return False when CCXT doesn't support watchOrders."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": False}

        assert bot.can_watch_orders() is False

    def test_returns_true_when_watch_orders_supported(self):
        """Should return True when CCXT supports watchOrders."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": True}

        assert bot.can_watch_orders() is True


class TestNormalizeOrderUpdate:
    """Test _normalize_order_update() hook."""

    def test_adds_position_side_from_info(self):
        """Should extract position_side from info.positionSide."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        order = {
            "amount": 1.5,
            "info": {"positionSide": "LONG"},
        }

        result = bot._normalize_order_update(order)

        assert result["position_side"] == "long"
        assert result["qty"] == 1.5

    def test_defaults_position_side_to_both(self):
        """Should default position_side to 'both' when not in info."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        order = {"amount": 2.0, "info": {}}

        result = bot._normalize_order_update(order)

        assert result["position_side"] == "both"


class TestWatchOrdersTemplateMethod:
    """Test watch_orders() template method."""

    @pytest.mark.asyncio
    async def test_exits_gracefully_when_cannot_watch(self):
        """Should return immediately when can_watch_orders() is False."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = None
        bot.exchange = "test_exchange"

        # Should not raise, should return immediately
        await bot.watch_orders()

    @pytest.mark.asyncio
    async def test_calls_hooks_when_can_watch(self):
        """Should use hooks when watching orders."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": True}
        bot.exchange = "test_exchange"
        bot.stop_websocket = False  # Start running

        # Mock the hooks
        bot._do_watch_orders = AsyncMock(
            return_value=[{"amount": 1.0, "info": {"positionSide": "LONG"}}]
        )

        # Set stop_websocket = True after first call to exit loop
        def stop_after_call(orders):
            bot.stop_websocket = True

        bot.handle_order_update = MagicMock(side_effect=stop_after_call)

        await bot.watch_orders()

        bot._do_watch_orders.assert_called_once()
        bot.handle_order_update.assert_called_once()
        # Verify normalization happened
        call_args = bot.handle_order_update.call_args[0][0]
        assert call_args[0]["position_side"] == "long"
        assert call_args[0]["qty"] == 1.0


class TestCreateCcxtSessionsWebSocketOptional:
    """Test that WebSocket is optional in create_ccxt_sessions()."""

    def test_prefers_exchange_ccxt_id_when_available(self):
        """Should instantiate CCXT clients with futures-specific id when present."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.user_info = {"exchange": "binance", "apiKey": "test", "options": {"user_opt": "x"}}
        bot.exchange = "binance"
        bot.exchange_ccxt_id = "binanceusdm"
        bot.ws_enabled = True
        bot.endpoint_override = None
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={"cfg_opt": "y"})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async
        import ccxt.pro as ccxt_pro

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {}
        ws_ctor = MagicMock()
        ws_ctor.return_value = MagicMock()
        ws_ctor.return_value.options = {}
        generic_rest_ctor = MagicMock()
        generic_ws_ctor = MagicMock()

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "binanceusdm", rest_ctor, raising=False)
            m.setattr(ccxt_pro, "binanceusdm", ws_ctor, raising=False)
            m.setattr(ccxt_async, "binance", generic_rest_ctor, raising=False)
            m.setattr(ccxt_pro, "binance", generic_ws_ctor, raising=False)
            bot.create_ccxt_sessions()

        rest_ctor.assert_called_once_with({"apiKey": "test"})
        ws_ctor.assert_called_once_with({"apiKey": "test"})
        generic_rest_ctor.assert_not_called()
        generic_ws_ctor.assert_not_called()
        assert bot.cca.options["cfg_opt"] == "y"
        assert bot.cca.options["user_opt"] == "x"
        assert bot.cca.options["defaultType"] == "swap"
        assert bot.ccp.options["cfg_opt"] == "y"
        assert bot.ccp.options["user_opt"] == "x"
        assert bot.ccp.options["defaultType"] == "swap"

    def test_falls_back_to_exchange_when_exchange_ccxt_id_missing(self):
        """Should support test-created instances without exchange_ccxt_id."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.user_info = {"exchange": "paradex", "apiKey": "test"}
        bot.exchange = "paradex"
        bot.ws_enabled = True
        bot.endpoint_override = None
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async
        import ccxt.pro as ccxt_pro

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {}
        ws_ctor = MagicMock()
        ws_ctor.return_value = MagicMock()
        ws_ctor.return_value.options = {}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "paradex", rest_ctor, raising=False)
            m.setattr(ccxt_pro, "paradex", ws_ctor, raising=False)
            bot.create_ccxt_sessions()

        rest_ctor.assert_called_once_with({"apiKey": "test"})
        ws_ctor.assert_called_once_with({"apiKey": "test"})

    def test_sets_ccp_none_when_ws_disabled(self):
        """Should set ccp=None instead of raising when ws_enabled=False."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.user_info = {"exchange": "paradex", "apiKey": "test"}
        bot.exchange = "paradex"
        bot.ws_enabled = False
        bot.endpoint_override = None
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        # Mock ccxt_async to avoid real exchange instantiation
        import ccxt.async_support as ccxt_async

        mock_exchange = MagicMock()
        mock_exchange.return_value = MagicMock()
        mock_exchange.return_value.options = {}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "paradex", mock_exchange, raising=False)
            bot.create_ccxt_sessions()

        assert bot.ccp is None


class TestBybitCreateCcxtSessions:
    """Test Bybit broker attribution during CCXT session creation."""

    def test_sets_passivbot_broker_id_on_rest_and_ws_clients(self):
        from exchanges.bybit import BybitBot

        bot = BybitBot.__new__(BybitBot)
        bot.user_info = {"exchange": "bybit", "apiKey": "test", "options": {"brokerId": "CCXT"}}
        bot.exchange = "bybit"
        bot.exchange_ccxt_id = "bybit"
        bot.ws_enabled = True
        bot.endpoint_override = None
        bot.broker_code = "passivbotbybit"
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async
        import ccxt.pro as ccxt_pro

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {"brokerId": "CCXT"}
        ws_ctor = MagicMock()
        ws_ctor.return_value = MagicMock()
        ws_ctor.return_value.options = {"brokerId": "CCXT"}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "bybit", rest_ctor, raising=False)
            m.setattr(ccxt_pro, "bybit", ws_ctor, raising=False)
            bot.create_ccxt_sessions()

        assert bot.cca.options["brokerId"] == "passivbotbybit"
        assert bot.ccp.options["brokerId"] == "passivbotbybit"

    def test_rejects_missing_broker_code(self):
        from exchanges.bybit import BybitBot

        bot = BybitBot.__new__(BybitBot)
        bot.user_info = {"exchange": "bybit", "apiKey": "test"}
        bot.exchange = "bybit"
        bot.exchange_ccxt_id = "bybit"
        bot.ws_enabled = False
        bot.endpoint_override = None
        bot.broker_code = ""
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {"brokerId": "CCXT"}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "bybit", rest_ctor, raising=False)
            with pytest.raises(ValueError, match="Bybit broker code"):
                bot.create_ccxt_sessions()

    def test_ccxt_signed_order_request_contains_passivbot_referer(self):
        import ccxt

        client = ccxt.bybit({"apiKey": "test_key", "secret": "test_secret"})
        client.options["brokerId"] = "passivbotbybit"

        signed = client.sign(
            "v5/order/create",
            "private",
            "POST",
            {
                "category": "linear",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Limit",
                "qty": "0.001",
                "price": "10000",
            },
        )

        assert signed["headers"]["Referer"] == "passivbotbybit"


class TestBitgetCreateCcxtSessions:
    """Test Bitget broker attribution during CCXT session creation."""

    def test_sets_passivbot_broker_on_rest_and_ws_clients(self):
        from exchanges.bitget import BitgetBot

        bot = BitgetBot.__new__(BitgetBot)
        bot.user_info = {"exchange": "bitget", "apiKey": "test", "options": {"broker": "p4sve"}}
        bot.exchange = "bitget"
        bot.exchange_ccxt_id = "bitget"
        bot.ws_enabled = True
        bot.endpoint_override = None
        bot.broker_code = "Passivbot"
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async
        import ccxt.pro as ccxt_pro

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {"broker": "p4sve"}
        ws_ctor = MagicMock()
        ws_ctor.return_value = MagicMock()
        ws_ctor.return_value.options = {"broker": "p4sve"}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "bitget", rest_ctor, raising=False)
            m.setattr(ccxt_pro, "bitget", ws_ctor, raising=False)
            bot.create_ccxt_sessions()

        assert bot.cca.options["broker"] == "Passivbot"
        assert bot.ccp.options["broker"] == "Passivbot"

    def test_rejects_missing_broker_code(self):
        from exchanges.bitget import BitgetBot

        bot = BitgetBot.__new__(BitgetBot)
        bot.user_info = {"exchange": "bitget", "apiKey": "test"}
        bot.exchange = "bitget"
        bot.exchange_ccxt_id = "bitget"
        bot.ws_enabled = False
        bot.endpoint_override = None
        bot.broker_code = ""
        bot._build_ccxt_config = MagicMock(return_value={"apiKey": "test"})
        bot._build_ccxt_options = MagicMock(return_value={})
        bot._apply_endpoint_override = MagicMock()

        import ccxt.async_support as ccxt_async

        rest_ctor = MagicMock()
        rest_ctor.return_value = MagicMock()
        rest_ctor.return_value.options = {"broker": "p4sve"}

        with pytest.MonkeyPatch().context() as m:
            m.setattr(ccxt_async, "bitget", rest_ctor, raising=False)
            with pytest.raises(ValueError, match="Bitget broker code"):
                bot.create_ccxt_sessions()

    def test_ccxt_signed_order_request_contains_passivbot_channel_code(self):
        import ccxt

        client = ccxt.bitget(
            {"apiKey": "test_key", "secret": "test_secret", "password": "test_passphrase"}
        )
        client.options["broker"] = "Passivbot"

        signed = client.sign(
            "v2/mix/order/place-order",
            ["private", "mix"],
            "POST",
            {
                "symbol": "BTCUSDT",
                "productType": "USDT-FUTURES",
                "marginMode": "crossed",
                "marginCoin": "USDT",
                "size": "0.001",
                "price": "10000",
                "side": "buy",
                "orderType": "limit",
                "force": "gtc",
            },
        )

        assert signed["headers"]["X-CHANNEL-API-CODE"] == "Passivbot"


class TestValidateWebsocketSupport:
    """Test validate_websocket_support() is non-fatal."""

    @pytest.mark.asyncio
    async def test_does_not_raise_when_watch_orders_not_supported(self):
        """Should log warning instead of raising when watchOrders not supported."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = MagicMock()
        bot.ccp.has = {"watchOrders": False}
        bot.exchange = "test_exchange"

        # Should not raise
        await bot.validate_websocket_support()

    @pytest.mark.asyncio
    async def test_does_not_raise_when_ccp_is_none(self):
        """Should handle ccp=None gracefully."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.ccp = None
        bot.exchange = "test_exchange"

        # Should not raise
        await bot.validate_websocket_support()


class TestGetPnlFromTrade:
    """Test _get_pnl_from_trade() hook."""

    def test_extracts_realized_pnl_from_info(self):
        """Should extract realized_pnl from info."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"info": {"realized_pnl": "12.50"}}

        result = bot._get_pnl_from_trade(trade)

        assert result == 12.50

    def test_tries_alternative_field_names(self):
        """Should try realizedPnl, pnl, profit as fallbacks."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)

        # Test realizedPnl
        assert bot._get_pnl_from_trade({"info": {"realizedPnl": "5.0"}}) == 5.0
        # Test pnl
        assert bot._get_pnl_from_trade({"info": {"pnl": "-3.5"}}) == -3.5
        # Test profit
        assert bot._get_pnl_from_trade({"info": {"profit": "100"}}) == 100.0

    def test_returns_zero_when_no_pnl_field(self):
        """Should return 0.0 when no PnL field found."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"info": {"other_field": "value"}}

        result = bot._get_pnl_from_trade(trade)

        assert result == 0.0


class TestGetPositionSideFromTrade:
    """Test _get_position_side_from_trade() hook."""

    def test_buy_with_zero_pnl_is_long_entry(self):
        """Buy with 0 PnL = opening long position."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"side": "buy", "info": {}}

        result = bot._get_position_side_from_trade(trade)

        assert result == "long"

    def test_buy_with_nonzero_pnl_is_short_exit(self):
        """Buy with PnL != 0 = closing short position."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"side": "buy", "info": {"realized_pnl": "10.0"}}

        result = bot._get_position_side_from_trade(trade)

        assert result == "short"

    def test_sell_with_zero_pnl_is_short_entry(self):
        """Sell with 0 PnL = opening short position."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"side": "sell", "info": {}}

        result = bot._get_position_side_from_trade(trade)

        assert result == "short"

    def test_sell_with_nonzero_pnl_is_long_exit(self):
        """Sell with PnL != 0 = closing long position."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        trade = {"side": "sell", "info": {"realized_pnl": "-5.0"}}

        result = bot._get_position_side_from_trade(trade)

        assert result == "long"


class TestFetchPnls:
    """Test fetch_pnls() template method."""

    @pytest.mark.asyncio
    async def test_fetches_trades_and_adds_pnl_fields(self):
        """Should fetch trades via CCXT and add pnl/position_side/qty."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.fetch_my_trades = AsyncMock(
            return_value=[
                {
                    "id": "trade1",
                    "timestamp": 1000,
                    "side": "buy",
                    "amount": 1.5,
                    "info": {"realized_pnl": "0"},
                },
                {
                    "id": "trade2",
                    "timestamp": 2000,
                    "side": "sell",
                    "amount": 1.5,
                    "info": {"realized_pnl": "25.50"},
                },
            ]
        )

        result = await bot.fetch_pnls(start_time=500)

        assert len(result) == 2
        # First trade: buy entry (pnl=0 -> long)
        assert result[0]["qty"] == 1.5
        assert result[0]["pnl"] == 0.0
        assert result[0]["position_side"] == "long"
        # Second trade: sell exit (pnl!=0 -> long)
        assert result[1]["qty"] == 1.5
        assert result[1]["pnl"] == 25.50
        assert result[1]["position_side"] == "long"

    @pytest.mark.asyncio
    async def test_raises_on_error(self):
        """Should raise on error (caller handles via restart_bot_on_too_many_errors)."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.fetch_my_trades = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await bot.fetch_pnls()

    @pytest.mark.asyncio
    async def test_passes_params_to_ccxt(self):
        """Should pass start_time, end_time, limit to CCXT."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.cca.fetch_my_trades = AsyncMock(return_value=[])

        await bot.fetch_pnls(start_time=1000, end_time=2000, limit=50)

        bot.cca.fetch_my_trades.assert_called_once_with(
            symbol=None,
            since=1000,
            limit=50,
            params={"until": 2000},
        )


class TestFetchBalanceHooks:
    """Tests for fetch_balance template method and hooks."""

    @pytest.fixture
    def bot_with_mock_cca(self):
        """Bot with mocked cca.fetch_balance."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.quote = "USDT"
        return bot

    @pytest.mark.asyncio
    async def test_do_fetch_balance_calls_cca(self, bot_with_mock_cca):
        """_do_fetch_balance should call cca.fetch_balance."""
        bot = bot_with_mock_cca
        bot.cca.fetch_balance = AsyncMock(return_value={"total": {"USDT": 1000.0}})

        result = await bot._do_fetch_balance()

        bot.cca.fetch_balance.assert_called_once()
        assert result == {"total": {"USDT": 1000.0}}

    def test_get_balance_extracts_from_total(self, bot_with_mock_cca):
        """_get_balance should extract quote currency from total."""
        bot = bot_with_mock_cca
        fetched = {"total": {"USDT": 1234.56, "BTC": 0.5}}

        result = bot._get_balance(fetched)

        assert result == 1234.56

    def test_get_balance_raises_when_quote_missing(self, bot_with_mock_cca):
        """_get_balance should fail loudly when quote not in total."""
        bot = bot_with_mock_cca
        fetched = {"total": {"BTC": 0.5}}

        with pytest.raises(KeyError, match="missing total\\['USDT'\\]"):
            bot._get_balance(fetched)

    def test_get_balance_raises_when_total_missing(self, bot_with_mock_cca):
        """_get_balance should fail loudly when total mapping is absent."""
        bot = bot_with_mock_cca
        fetched = {"free": {"USDT": 12.0}}

        with pytest.raises(KeyError, match="missing 'total' mapping"):
            bot._get_balance(fetched)

    @pytest.mark.asyncio
    async def test_fetch_balance_uses_hooks(self, bot_with_mock_cca):
        """fetch_balance should use _do_fetch_balance and _get_balance."""
        bot = bot_with_mock_cca
        bot.cca.fetch_balance = AsyncMock(return_value={"total": {"USDT": 500.0}})

        result = await bot.fetch_balance()

        assert result == 500.0


class TestFetchPositionsHooks:
    """Tests for fetch_positions template method and hooks."""

    @pytest.fixture
    def bot_with_mock_cca(self):
        """Bot with mocked cca.fetch_positions."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        return bot

    @pytest.mark.asyncio
    async def test_do_fetch_positions_calls_cca(self, bot_with_mock_cca):
        """_do_fetch_positions should call cca.fetch_positions."""
        bot = bot_with_mock_cca
        bot.cca.fetch_positions = AsyncMock(return_value=[])

        result = await bot._do_fetch_positions()

        bot.cca.fetch_positions.assert_called_once()
        assert result == []

    def test_get_position_side_from_side_field(self, bot_with_mock_cca):
        """_get_position_side should use CCXT unified 'side' field."""
        bot = bot_with_mock_cca

        assert bot._get_position_side({"side": "long"}) == "long"
        assert bot._get_position_side({"side": "short"}) == "short"
        assert bot._get_position_side({"side": "SHORT"}) == "short"

    def test_get_position_side_defaults_to_long(self, bot_with_mock_cca):
        """_get_position_side should default to 'long' when missing."""
        bot = bot_with_mock_cca

        assert bot._get_position_side({}) == "long"

    def test_normalize_positions_transforms_list(self, bot_with_mock_cca):
        """_normalize_positions should transform to passivbot format."""
        bot = bot_with_mock_cca
        fetched = [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.5, "entryPrice": 50000},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 2.0, "entryPrice": 3000},
        ]

        result = bot._normalize_positions(fetched)

        assert len(result) == 2
        assert result[0] == {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.5,
            "price": 50000,
        }
        assert result[1] == {
            "symbol": "ETH/USDT:USDT",
            "position_side": "short",
            "size": 2.0,
            "price": 3000,
        }

    def test_normalize_positions_skips_zero_contracts(self, bot_with_mock_cca):
        """_normalize_positions should skip positions with 0 contracts."""
        bot = bot_with_mock_cca
        fetched = [
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0, "entryPrice": 50000},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 1.0, "entryPrice": 3000},
        ]

        result = bot._normalize_positions(fetched)

        assert len(result) == 1
        assert result[0]["symbol"] == "ETH/USDT:USDT"

    @pytest.mark.asyncio
    async def test_fetch_positions_uses_hooks(self, bot_with_mock_cca):
        """fetch_positions should use _do_fetch_positions and _normalize_positions."""
        bot = bot_with_mock_cca
        bot.cca.fetch_positions = AsyncMock(
            return_value=[
                {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 1.0, "entryPrice": 45000}
            ]
        )

        result = await bot.fetch_positions()

        assert len(result) == 1
        assert result[0]["position_side"] == "long"
        assert result[0]["size"] == 1.0


class TestFetchPnlsHooks:
    """Tests for fetch_pnls template method and hooks."""

    @pytest.fixture
    def bot_with_mock_cca(self):
        """Bot with mocked cca.fetch_my_trades."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        return bot

    @pytest.mark.asyncio
    async def test_do_fetch_pnls_calls_cca(self, bot_with_mock_cca):
        """_do_fetch_pnls should call cca.fetch_my_trades."""
        bot = bot_with_mock_cca
        bot.cca.fetch_my_trades = AsyncMock(return_value=[])

        result = await bot._do_fetch_pnls(start_time=1000, end_time=2000, limit=100)

        bot.cca.fetch_my_trades.assert_called_once()
        call_kwargs = bot.cca.fetch_my_trades.call_args
        assert call_kwargs.kwargs["since"] == 1000
        assert call_kwargs.kwargs["limit"] == 100
        assert call_kwargs.kwargs["params"]["until"] == 2000

    @pytest.mark.asyncio
    async def test_do_fetch_pnls_omits_until_when_no_end_time(self, bot_with_mock_cca):
        """_do_fetch_pnls should not pass 'until' when end_time is None."""
        bot = bot_with_mock_cca
        bot.cca.fetch_my_trades = AsyncMock(return_value=[])

        await bot._do_fetch_pnls(start_time=1000, end_time=None, limit=100)

        call_kwargs = bot.cca.fetch_my_trades.call_args
        assert "until" not in call_kwargs.kwargs["params"]

    def test_normalize_pnls_adds_fields(self, bot_with_mock_cca):
        """_normalize_pnls should add qty, pnl, position_side to each trade."""
        bot = bot_with_mock_cca
        trades = [
            {"amount": 1.0, "side": "buy", "timestamp": 1000, "info": {"realized_pnl": "0"}},
            {"amount": 2.0, "side": "sell", "timestamp": 2000, "info": {"realized_pnl": "50"}},
        ]

        result = bot._normalize_pnls(trades)

        assert result[0]["qty"] == 1.0
        assert result[0]["pnl"] == 0.0
        assert result[0]["position_side"] == "long"  # buy + pnl=0 = entry long
        assert result[1]["qty"] == 2.0
        assert result[1]["pnl"] == 50.0
        assert result[1]["position_side"] == "long"  # sell + pnl!=0 = exit long

    def test_normalize_pnls_sorts_by_timestamp(self, bot_with_mock_cca):
        """_normalize_pnls should sort trades by timestamp."""
        bot = bot_with_mock_cca
        trades = [
            {"amount": 1.0, "side": "buy", "timestamp": 2000, "info": {}},
            {"amount": 2.0, "side": "sell", "timestamp": 1000, "info": {}},
        ]

        result = bot._normalize_pnls(trades)

        assert result[0]["timestamp"] == 1000
        assert result[1]["timestamp"] == 2000

    @pytest.mark.asyncio
    async def test_fetch_pnls_uses_hooks(self, bot_with_mock_cca):
        """fetch_pnls should use _do_fetch_pnls and _normalize_pnls."""
        bot = bot_with_mock_cca
        bot.cca.fetch_my_trades = AsyncMock(
            return_value=[
                {"amount": 1.0, "side": "buy", "timestamp": 1000, "info": {"realized_pnl": "100"}}
            ]
        )

        result = await bot.fetch_pnls(start_time=500)

        assert len(result) == 1
        assert result[0]["qty"] == 1.0
        assert result[0]["pnl"] == 100.0


class TestFetchTickersHooks:
    """Tests for fetch_tickers template method and hooks."""

    @pytest.fixture
    def bot_with_mock_cca(self):
        """Bot with mocked cca.fetch_tickers."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.exchange = "testexchange"
        bot.cca = MagicMock()
        bot.markets_dict = {"BTC/USDT:USDT": {}, "ETH/USDT:USDT": {}}
        return bot

    @pytest.mark.asyncio
    async def test_do_fetch_tickers_calls_cca(self, bot_with_mock_cca):
        """_do_fetch_tickers should call cca.fetch_tickers."""
        bot = bot_with_mock_cca
        bot.cca.fetch_tickers = AsyncMock(return_value={})

        result = await bot._do_fetch_tickers()

        bot.cca.fetch_tickers.assert_called_once()
        assert result == {}

    def test_normalize_tickers_transforms_dict(self, bot_with_mock_cca):
        """_normalize_tickers should transform to {bid, ask, last} format."""
        bot = bot_with_mock_cca
        fetched = {
            "BTC/USDT:USDT": {"bid": 50000, "ask": 50001, "last": 50000.5},
            "ETH/USDT:USDT": {"bid": 3000, "ask": 3001, "last": 3000.5},
        }

        result = bot._normalize_tickers(fetched)

        assert result["BTC/USDT:USDT"] == {"bid": 50000, "ask": 50001, "last": 50000.5}
        assert result["ETH/USDT:USDT"] == {"bid": 3000, "ask": 3001, "last": 3000.5}

    def test_normalize_tickers_filters_to_markets_dict(self, bot_with_mock_cca):
        """_normalize_tickers should only include symbols in markets_dict."""
        bot = bot_with_mock_cca
        fetched = {
            "BTC/USDT:USDT": {"bid": 50000, "ask": 50001, "last": 50000.5},
            "UNKNOWN/USDT:USDT": {"bid": 100, "ask": 101, "last": 100.5},
        }

        result = bot._normalize_tickers(fetched)

        assert "BTC/USDT:USDT" in result
        assert "UNKNOWN/USDT:USDT" not in result

    def test_normalize_tickers_handles_none_values(self, bot_with_mock_cca):
        """_normalize_tickers should handle None bid/ask/last values."""
        bot = bot_with_mock_cca
        fetched = {
            "BTC/USDT:USDT": {"bid": None, "ask": 50001, "last": None},
        }

        result = bot._normalize_tickers(fetched)

        assert result["BTC/USDT:USDT"]["bid"] == 0
        assert result["BTC/USDT:USDT"]["ask"] == 50001
        assert result["BTC/USDT:USDT"]["last"] == 0  # Falls back to bid which is also 0

    @pytest.mark.asyncio
    async def test_fetch_tickers_uses_hooks(self, bot_with_mock_cca):
        """fetch_tickers should use _do_fetch_tickers and _normalize_tickers."""
        bot = bot_with_mock_cca
        bot.cca.fetch_tickers = AsyncMock(
            return_value={"BTC/USDT:USDT": {"bid": 45000, "ask": 45001, "last": 45000}}
        )

        result = await bot.fetch_tickers()

        assert "BTC/USDT:USDT" in result
        assert result["BTC/USDT:USDT"]["bid"] == 45000


class TestBuildOrderParams:
    """Tests for _build_order_params hook."""

    @pytest.fixture
    def bot_with_config(self):
        """Bot with config for time_in_force."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)
        bot.config = {"live": {"time_in_force": "post_only"}}
        return bot

    def test_build_order_params_sets_position_side(self, bot_with_config):
        """_build_order_params should set positionSide from order."""
        bot = bot_with_config
        order = {"position_side": "long", "type": "market"}

        result = bot._build_order_params(order)

        assert result["positionSide"] == "LONG"

    def test_build_order_params_sets_client_order_id(self, bot_with_config):
        """_build_order_params should set clientOrderId from custom_id."""
        bot = bot_with_config
        order = {"custom_id": "my_order_123", "type": "market"}

        result = bot._build_order_params(order)

        assert result["clientOrderId"] == "my_order_123"

    def test_build_order_params_sets_post_only_for_limit(self, bot_with_config):
        """_build_order_params should set postOnly for limit orders with post_only tif."""
        bot = bot_with_config
        order = {"type": "limit"}

        result = bot._build_order_params(order)

        assert result["postOnly"] is True

    def test_build_order_params_sets_gtc_for_limit(self, bot_with_config):
        """_build_order_params should set GTC for limit orders with non-post_only tif."""
        bot = bot_with_config
        bot.config = {"live": {"time_in_force": "gtc"}}
        order = {"type": "limit"}

        result = bot._build_order_params(order)

        assert result["timeInForce"] == "GTC"
