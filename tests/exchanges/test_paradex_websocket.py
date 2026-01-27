"""Tests for ParadexBot native WebSocket implementation."""

import json
import pytest
import ccxt
from unittest.mock import MagicMock, AsyncMock, patch


class TestParadexWebSocketHooks:
    """Test WebSocket hook overrides."""

    def test_can_watch_orders_returns_true(self):
        """ParadexBot always supports order watching via native WebSocket."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.user_info = {"quote": "USDC"}
            bot.quote = "USDC"
            bot._ws = None

            assert bot.can_watch_orders() is True


class TestParadexWebSocketUrl:
    """Test WebSocket URL selection."""

    def test_get_ws_url_returns_production_by_default(self):
        """Default to production WebSocket URL."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.cca = MagicMock()
            bot.cca.urls = {"api": {"public": "https://api.prod.paradex.trade"}}

            assert bot._get_ws_url() == "wss://ws.api.prod.paradex.trade/v1"

    def test_get_ws_url_returns_testnet_when_configured(self):
        """Use testnet URL when CCXT is configured for testnet."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.cca = MagicMock()
            bot.cca.urls = {"api": {"public": "https://api.testnet.paradex.trade"}}

            assert bot._get_ws_url() == "wss://ws.api.testnet.paradex.trade/v1"


class TestParadexWsSendAndExpect:
    """Test WebSocket send/expect helper."""

    @pytest.mark.asyncio
    async def test_send_and_expect_success(self):
        """Successful send/expect returns response."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps({"jsonrpc": "2.0", "result": {"node_id": "abc123"}, "id": 1})
            )

            response = await bot._ws_send_and_expect(
                method="auth", params={"bearer": "token"}, msg_id=1, success_log="auth ok"
            )

            assert response["result"]["node_id"] == "abc123"
            bot._ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_and_expect_auth_error_raises_authentication_error(self):
        """Auth errors (40111) raise ccxt.AuthenticationError."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": 40111, "message": "Invalid Bearer Token"},
                        "id": 1,
                    }
                )
            )

            with pytest.raises(ccxt.AuthenticationError, match="40111"):
                await bot._ws_send_and_expect(
                    method="auth", params={"bearer": "bad_token"}, msg_id=1, success_log="auth ok"
                )

    @pytest.mark.asyncio
    async def test_send_and_expect_other_error_raises_exception(self):
        """Non-auth errors raise generic Exception (retryable)."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32601, "message": "method not found"},
                        "id": 1,
                    }
                )
            )

            with pytest.raises(Exception, match="-32601"):
                await bot._ws_send_and_expect(
                    method="bad_method", params={}, msg_id=1, success_log="ok"
                )

    @pytest.mark.asyncio
    async def test_send_and_expect_connection_closed_sets_ws_none(self):
        """ConnectionClosed sets _ws to None for reconnection."""
        from websockets.exceptions import ConnectionClosed

        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.send = AsyncMock(side_effect=ConnectionClosed(None, None))

            with pytest.raises(ConnectionError):
                await bot._ws_send_and_expect(method="auth", params={}, msg_id=1, success_log="ok")

            assert bot._ws is None


class TestParadexWsConnect:
    """Test WebSocket connection establishment."""

    @pytest.mark.asyncio
    async def test_ws_connect_authenticates_and_subscribes(self):
        """_ws_connect establishes connection, authenticates, and subscribes."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            with patch("exchanges.paradex.websockets") as mock_ws:
                from exchanges.paradex import ParadexBot

                # Setup mock WebSocket
                mock_conn = AsyncMock()
                mock_conn.recv = AsyncMock(
                    side_effect=[
                        json.dumps({"jsonrpc": "2.0", "result": {"node_id": "abc"}, "id": 1}),
                        json.dumps({"jsonrpc": "2.0", "result": {}, "id": 2}),
                    ]
                )
                mock_ws.connect = AsyncMock(return_value=mock_conn)

                bot = ParadexBot.__new__(ParadexBot)
                bot._ws = None
                bot.cca = MagicMock()
                bot.cca.urls = {"api": {"public": "https://api.prod.paradex.trade"}}
                bot.cca.authenticate_rest = AsyncMock(return_value="jwt_token_123")

                await bot._ws_connect()

                # Verify connection
                mock_ws.connect.assert_called_once_with("wss://ws.api.prod.paradex.trade/v1")

                # Verify auth message sent
                calls = mock_conn.send.call_args_list
                auth_msg = json.loads(calls[0][0][0])
                assert auth_msg["method"] == "auth"
                assert auth_msg["params"]["bearer"] == "jwt_token_123"

                # Verify subscribe message sent
                sub_msg = json.loads(calls[1][0][0])
                assert sub_msg["method"] == "subscribe"
                assert sub_msg["params"]["channel"] == "orders.ALL"


class TestParadexWsReceiveOrders:
    """Test WebSocket order message receiving."""

    @pytest.mark.asyncio
    async def test_receive_orders_returns_order_data(self):
        """Order subscription messages return order data."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            order_data = {
                "id": "order123",
                "market": "ETH-USD-PERP",
                "side": "BUY",
                "size": "0.5",
                "price": "2000.00",
                "status": "OPEN",
            }

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "subscription",
                        "params": {"channel": "orders.ETH-USD-PERP", "data": order_data},
                    }
                )
            )

            result = await bot._ws_receive_orders()

            assert len(result) == 1
            assert result[0]["id"] == "order123"

    @pytest.mark.asyncio
    async def test_receive_orders_skips_non_subscription_messages(self):
        """Non-subscription messages return empty list."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps({"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 5})
            )

            result = await bot._ws_receive_orders()

            assert result == []

    @pytest.mark.asyncio
    async def test_receive_orders_skips_non_order_channels(self):
        """Non-order channel messages return empty list."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(
                return_value=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "subscription",
                        "params": {"channel": "trades.ETH-USD-PERP", "data": {"id": "trade123"}},
                    }
                )
            )

            result = await bot._ws_receive_orders()

            assert result == []

    @pytest.mark.asyncio
    async def test_receive_orders_connection_closed_sets_ws_none(self):
        """ConnectionClosed sets _ws to None for reconnection."""
        from websockets.exceptions import ConnectionClosed

        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.recv = AsyncMock(side_effect=ConnectionClosed(None, None))

            with pytest.raises(ConnectionError):
                await bot._ws_receive_orders()

            assert bot._ws is None


class TestParadexDoWatchOrders:
    """Test _do_watch_orders hook."""

    @pytest.mark.asyncio
    async def test_do_watch_orders_connects_on_first_call(self):
        """First call to _do_watch_orders triggers connection."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = None
            bot._ws_connect = AsyncMock()
            bot._ws_receive_orders = AsyncMock(return_value=[{"id": "order1"}])

            # Mock _ws after connect
            async def set_ws():
                bot._ws = AsyncMock()
                bot._ws.closed = False

            bot._ws_connect.side_effect = set_ws

            result = await bot._do_watch_orders()

            bot._ws_connect.assert_called_once()
            assert result == [{"id": "order1"}]

    @pytest.mark.asyncio
    async def test_do_watch_orders_reuses_existing_connection(self):
        """Subsequent calls reuse existing connection."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.closed = False
            bot._ws_connect = AsyncMock()
            bot._ws_receive_orders = AsyncMock(return_value=[{"id": "order2"}])

            result = await bot._do_watch_orders()

            bot._ws_connect.assert_not_called()
            assert result == [{"id": "order2"}]

    @pytest.mark.asyncio
    async def test_do_watch_orders_reconnects_when_closed(self):
        """Reconnects when connection is closed."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot._ws = AsyncMock()
            bot._ws.closed = True  # Connection closed
            bot._ws_connect = AsyncMock()
            bot._ws_receive_orders = AsyncMock(return_value=[])

            # Mock reconnection
            async def reconnect():
                bot._ws = AsyncMock()
                bot._ws.closed = False

            bot._ws_connect.side_effect = reconnect

            await bot._do_watch_orders()

            bot._ws_connect.assert_called_once()


class TestParadexNormalizeOrderUpdate:
    """Test order normalization."""

    def test_normalize_order_update_maps_all_fields_with_client_id(self):
        """All Paradex fields are mapped to passivbot format, position_side from client_id."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.markets_dict = {"ETH/USD:USD": {"id": "ETH-USD-PERP"}}

            # order-0x0004 = entry_grid_normal_long (matches passivbot_rust)
            paradex_order = {
                "id": "order123",
                "market": "ETH-USD-PERP",
                "side": "BUY",
                "type": "LIMIT",
                "size": "0.5",
                "price": "2000.00",
                "status": "OPEN",
                "created_at": 1704240000000,
                "client_id": "order-0x0004",
            }

            result = bot._normalize_order_update(paradex_order)

            assert result["id"] == "order123"
            assert result["symbol"] == "ETH/USD:USD"
            assert result["side"] == "buy"
            assert result["type"] == "limit"
            assert result["qty"] == 0.5
            assert result["amount"] == 0.5
            assert result["price"] == 2000.0
            assert result["status"] == "open"
            assert result["timestamp"] == 1704240000000
            assert result["position_side"] == "long"
            assert result["custom_id"] == "order-0x0004"
            assert result["info"] == paradex_order

    def test_normalize_order_update_close_long_has_position_side_long(self):
        """close_grid_long (SELL) orders get position_side='long'."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.markets_dict = {}

            # 0x0007 = close_grid_long - a SELL order to close a LONG position
            result = bot._normalize_order_update(
                {"side": "SELL", "market": "X", "client_id": "order-0x0007"}
            )

            assert result["position_side"] == "long"  # NOT short!
            assert result["side"] == "sell"

    def test_normalize_order_update_no_client_id_returns_both(self):
        """Orders without client_id get position_side='both'."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.markets_dict = {}

            result = bot._normalize_order_update({"side": "SELL", "market": "X"})

            assert result["position_side"] == "both"

    def test_normalize_status_mappings(self):
        """Paradex statuses map to CCXT-style statuses."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)

            assert bot._normalize_status("NEW") == "open"
            assert bot._normalize_status("UNTRIGGERED") == "open"
            assert bot._normalize_status("OPEN") == "open"
            assert bot._normalize_status("CLOSED") == "closed"
            assert bot._normalize_status("UNKNOWN") == "unknown"

    def test_paradex_market_to_symbol_direct_match(self):
        """Market in markets_dict returns directly."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.markets_dict = {"ETH-USD-PERP": {"id": "ETH-USD-PERP"}}

            assert bot._paradex_market_to_symbol("ETH-USD-PERP") == "ETH-USD-PERP"

    def test_paradex_market_to_symbol_id_lookup(self):
        """Market matching symbol's id field returns symbol."""
        with patch("exchanges.paradex.CCXTBot.__init__", return_value=None):
            from exchanges.paradex import ParadexBot

            bot = ParadexBot.__new__(ParadexBot)
            bot.markets_dict = {"ETH/USD:USD": {"id": "ETH-USD-PERP"}}

            assert bot._paradex_market_to_symbol("ETH-USD-PERP") == "ETH/USD:USD"
