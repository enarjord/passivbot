"""Tests for LighterBot exchange adapter."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════ Task 2: Skeleton tests ═══════════════════


class TestLighterBotInit:
    """Test LighterBot initialization and basic attributes."""

    def _make_bot(self, user_info=None):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.user_info = user_info or {
                "private_key": "0xdeadbeef",
                "api_key_index": "1",
                "account_index": "42",
            }
            bot.quote = "USDC"
            bot.hedge_mode = False
            bot._ws = None
            return bot

    def test_quote_is_usdc(self):
        bot = self._make_bot()
        assert bot.quote == "USDC"

    def test_hedge_mode_is_false(self):
        bot = self._make_bot()
        assert bot.hedge_mode is False

    def test_ws_state_initialized(self):
        bot = self._make_bot()
        assert bot._ws is None


class TestLighterBotCcxtConfig:
    """Test _build_ccxt_config maps auth fields correctly."""

    def _make_bot(self, user_info):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.user_info = user_info
            return bot

    def test_build_ccxt_config_maps_fields(self):
        bot = self._make_bot({
            "private_key": "0xabc123",
            "api_key_index": "5",
            "account_index": "99",
        })
        config = bot._build_ccxt_config()
        assert config["enableRateLimit"] is True
        assert config["privateKey"] == "0xabc123"
        assert config["options"]["apiKeyIndex"] == 5
        assert config["options"]["accountIndex"] == 99

    def test_build_ccxt_config_with_library_path(self):
        bot = self._make_bot({
            "private_key": "0xabc123",
            "api_key_index": "5",
            "account_index": "99",
            "library_path": "/usr/lib/lighter.so",
        })
        config = bot._build_ccxt_config()
        assert config["options"]["libraryPath"] == "/usr/lib/lighter.so"

    def test_build_ccxt_config_without_library_path(self):
        bot = self._make_bot({
            "private_key": "0xabc123",
            "api_key_index": "5",
            "account_index": "99",
        })
        config = bot._build_ccxt_config()
        assert "libraryPath" not in config["options"]

    def test_build_ccxt_config_sets_default_timeout(self):
        """Match CCXTBot base — CCXT's ~10s default is too tight on cold boot (see PR 576)."""
        bot = self._make_bot({
            "private_key": "0xabc123",
            "api_key_index": "5",
            "account_index": "99",
        })
        config = bot._build_ccxt_config()
        assert config["timeout"] == 30000


class TestLighterBotMarginMode:
    """Test that LighterBot inherits base class margin mode support."""

    def test_no_margin_mode_override(self):
        """LighterBot should NOT override _should_set_margin_mode (supports both modes)."""
        from exchanges.lighter import LighterBot
        from exchanges.ccxt_bot import CCXTBot

        # LighterBot should use the base class implementation, not override it
        assert "_should_set_margin_mode" not in LighterBot.__dict__


# ═══════════════════ Task 1: Signer init ═══════════════════


class TestLighterSignerInit:
    """Test create_ccxt_sessions loads native signer and injects it."""

    def _make_bot(self, user_info=None):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.user_info = user_info or {
                "private_key": "0xdeadbeef",
                "api_key_index": "1",
                "account_index": "42",
                "library_path": "/usr/lib/lighter.so",
            }
            bot.exchange = "lighter"
            bot.ws_enabled = False
            bot.cca = MagicMock()
            bot.cca.privateKey = "0xdeadbeef"
            bot.cca.options = {"chainId": 304}
            bot.cca.urls = {"api": {"public": "https://mainnet.zklighter.elliot.ai"}}
            bot.cca.implode_hostname = MagicMock(
                return_value="https://mainnet.zklighter.elliot.ai"
            )
            return bot

    @patch("exchanges.lighter.CCXTBot.create_ccxt_sessions")
    @patch("exchanges.lighter.load_lighter_library")
    def test_creates_client_and_injects_signer(self, mock_load, mock_super):
        mock_signer = MagicMock()
        mock_signer.CreateClient.return_value = None  # success
        mock_load.return_value = mock_signer

        bot = self._make_bot()
        bot.create_ccxt_sessions()

        mock_load.assert_called_once_with("/usr/lib/lighter.so")
        mock_signer.CreateClient.assert_called_once_with(
            b"https://mainnet.zklighter.elliot.ai",
            b"0xdeadbeef",
            304,
            1,
            42,
        )
        assert bot._signer is mock_signer
        assert bot.cca.options["signer"] is mock_signer

    @patch("exchanges.lighter.CCXTBot.create_ccxt_sessions")
    @patch("exchanges.lighter.load_lighter_library")
    def test_raises_on_create_client_error(self, mock_load, mock_super):
        mock_signer = MagicMock()
        mock_signer.CreateClient.return_value = b"error: invalid key"
        mock_load.return_value = mock_signer

        bot = self._make_bot()
        with pytest.raises(Exception, match="CreateClient failed"):
            bot.create_ccxt_sessions()

    @patch("exchanges.lighter.CCXTBot.create_ccxt_sessions")
    @patch("exchanges.lighter.load_lighter_library")
    def test_falls_back_to_ccxt_library_path(self, mock_load, mock_super):
        mock_signer = MagicMock()
        mock_signer.CreateClient.return_value = None
        mock_load.return_value = mock_signer

        bot = self._make_bot(user_info={
            "private_key": "0xdeadbeef",
            "api_key_index": "1",
            "account_index": "42",
        })
        bot.cca.options["libraryPath"] = "/fallback/lighter.so"
        bot.create_ccxt_sessions()

        mock_load.assert_called_once_with("/fallback/lighter.so")


# ═══════════════════ Task 2: update_exchange_config_by_symbols ═══════════════════


class TestLighterUpdateExchangeConfig:
    """Test update_exchange_config_by_symbols calls signer directly."""

    def _make_bot(self):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.user_info = {
                "api_key_index": "1",
                "account_index": "42",
            }
            bot.config = {"live": {"TWE_long": 1.5, "TWE_short": 1.5}}
            bot._signer = MagicMock()
            bot.cca = MagicMock()
            bot.cca.market.return_value = {"id": "3"}
            bot.cca.fetch_nonce = AsyncMock(return_value=100)
            bot.cca.publicPostSendTx = AsyncMock(return_value={})
            return bot

    @pytest.mark.asyncio
    @patch("exchanges.lighter.decode_tx_info")
    async def test_signs_and_submits_leverage_tx(self, mock_decode):
        mock_decode.return_value = (7, "signed_data", "hash", None)
        bot = self._make_bot()
        bot._calc_leverage_for_symbol = MagicMock(return_value=5)
        bot._get_margin_mode_for_symbol = MagicMock(return_value="cross")

        await bot.update_exchange_config_by_symbols(["HYPE/USDC:USDC"])

        bot._signer.SignUpdateLeverage.assert_called_once_with(3, 2000, 0, 100, 1, 42)
        bot.cca.publicPostSendTx.assert_called_once_with({
            "tx_type": 7,
            "tx_info": "signed_data",
        })

    @pytest.mark.asyncio
    @patch("exchanges.lighter.decode_tx_info")
    async def test_isolated_margin_mode_value(self, mock_decode):
        mock_decode.return_value = (7, "signed_data", "hash", None)
        bot = self._make_bot()
        bot._calc_leverage_for_symbol = MagicMock(return_value=10)
        bot._get_margin_mode_for_symbol = MagicMock(return_value="isolated")

        await bot.update_exchange_config_by_symbols(["HYPE/USDC:USDC"])

        # margin_mode=1 for isolated
        call_args = bot._signer.SignUpdateLeverage.call_args[0]
        assert call_args[2] == 1
        # initial_margin_fraction = 10000 / 10 = 1000
        assert call_args[1] == 1000

    @pytest.mark.asyncio
    @patch("exchanges.lighter.decode_tx_info")
    async def test_signer_error_propagates(self, mock_decode):
        """Generic signer errors must raise so the orchestrator's retry/backoff
        loop in passivbot.update_exchange_config can handle them."""
        mock_decode.return_value = (None, None, None, "some signer error")
        bot = self._make_bot()
        bot._calc_leverage_for_symbol = MagicMock(return_value=5)
        bot._get_margin_mode_for_symbol = MagicMock(return_value="cross")

        with pytest.raises(Exception, match="some signer error"):
            await bot.update_exchange_config_by_symbols(["HYPE/USDC:USDC"])
        bot.cca.publicPostSendTx.assert_not_called()

    @pytest.mark.asyncio
    @patch("exchanges.lighter.decode_tx_info")
    async def test_already_set_is_info_not_error(self, mock_decode):
        mock_decode.return_value = (None, None, None, "leverage already set")
        bot = self._make_bot()
        bot._calc_leverage_for_symbol = MagicMock(return_value=5)
        bot._get_margin_mode_for_symbol = MagicMock(return_value="cross")

        # Should not raise — "already" is expected
        await bot.update_exchange_config_by_symbols(["HYPE/USDC:USDC"])


# ═══════════════════ Task 3: Position side logic ═══════════════════


class TestLighterPositionSide:
    """Test _get_position_side_for_order one-way mode logic."""

    def _make_bot(self, positions=None):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.positions = positions or {}
            return bot

    def test_has_long_position_returns_long(self):
        bot = self._make_bot({
            "BTC/USDC:USDC": {"long": {"size": 1.0}, "short": {"size": 0.0}},
        })
        order = {"symbol": "BTC/USDC:USDC", "side": "sell", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "long"

    def test_has_short_position_returns_short(self):
        bot = self._make_bot({
            "BTC/USDC:USDC": {"long": {"size": 0.0}, "short": {"size": -1.0}},
        })
        order = {"symbol": "BTC/USDC:USDC", "side": "buy", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "short"

    def test_no_position_reduce_only_buy_returns_short(self):
        """When symbol not in positions dict, reduceOnly buy -> short."""
        bot = self._make_bot({})
        order = {"symbol": "BTC/USDC:USDC", "side": "buy", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "short"

    def test_no_position_reduce_only_sell_returns_long(self):
        """When symbol not in positions dict, reduceOnly sell -> long."""
        bot = self._make_bot({})
        order = {"symbol": "BTC/USDC:USDC", "side": "sell", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "long"

    def test_no_position_buy_returns_long(self):
        bot = self._make_bot({
            "BTC/USDC:USDC": {"long": {"size": 0.0}, "short": {"size": 0.0}},
        })
        order = {"symbol": "BTC/USDC:USDC", "side": "buy", "reduceOnly": False}
        assert bot._get_position_side_for_order(order) == "long"

    def test_no_position_sell_returns_short(self):
        bot = self._make_bot({
            "BTC/USDC:USDC": {"long": {"size": 0.0}, "short": {"size": 0.0}},
        })
        order = {"symbol": "BTC/USDC:USDC", "side": "sell", "reduceOnly": False}
        assert bot._get_position_side_for_order(order) == "short"

    def test_symbol_not_in_positions_buy_returns_long(self):
        bot = self._make_bot({})
        order = {"symbol": "BTC/USDC:USDC", "side": "buy"}
        assert bot._get_position_side_for_order(order) == "long"


# ═══════════════════ Task 4: Order params ═══════════════════


class TestLighterOrderParams:
    """Test _build_order_params for limit/market orders."""

    def _make_bot(self, tif="post_only"):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.config = {"live": {"time_in_force": tif}}
            return bot

    def test_limit_order_post_only(self):
        bot = self._make_bot(tif="post_only")
        order = {
            "type": "limit",
            "reduce_only": False,
            "custom_id": "abc123",
        }
        params = bot._build_order_params(order)
        assert params["reduceOnly"] is False
        assert params["clientOrderId"] == "abc123"
        assert params["postOnly"] is True
        assert "timeInForce" not in params

    def test_limit_order_gtc(self):
        bot = self._make_bot(tif="gtc")
        order = {
            "type": "limit",
            "reduce_only": True,
            "custom_id": "def456",
        }
        params = bot._build_order_params(order)
        assert params["reduceOnly"] is True
        assert params["clientOrderId"] == "def456"
        # Lighter uses GTT (Good Till Time with expiry=-1) for GTC-equivalent behavior
        assert params["timeInForce"] == "GTT"
        assert "postOnly" not in params

    def test_market_order_no_tif_params(self):
        bot = self._make_bot(tif="post_only")
        order = {
            "type": "market",
            "reduce_only": True,
            "custom_id": "ghi789",
        }
        params = bot._build_order_params(order)
        assert params["reduceOnly"] is True
        assert params["clientOrderId"] == "ghi789"
        assert "postOnly" not in params
        assert "timeInForce" not in params


# ═══════════════════ Task 5: WebSocket tests ═══════════════════


class TestLighterWebSocketUrl:
    """Test _get_ws_url returns correct URL based on testnet config."""

    def _make_bot(self, test_urls=None):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.cca = MagicMock()
            bot.cca.urls = {"test": test_urls or {}}
            return bot

    def test_mainnet_url(self):
        bot = self._make_bot(test_urls={})
        assert bot._get_ws_url() == "wss://mainnet.zklighter.elliot.ai/stream"

    def test_testnet_url(self):
        bot = self._make_bot(test_urls={"public": "https://testnet.zklighter.elliot.ai"})
        assert bot._get_ws_url() == "wss://testnet.zklighter.elliot.ai/stream"


class TestLighterCanWatchOrders:
    """Test can_watch_orders always returns True."""

    def test_can_watch_orders_returns_true(self):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            assert bot.can_watch_orders() is True


class TestLighterWsConnect:
    """Test _ws_connect establishes connection and subscribes."""

    def _make_bot(self, account_index=42):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot._ws = None
            bot.cca = MagicMock()
            bot.cca.urls = {"test": {}}
            bot.user_info = {"account_index": str(account_index)}
            return bot

    @pytest.mark.asyncio
    async def test_ws_connect_subscribes_to_account_all(self):
        bot = self._make_bot(account_index=42)
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value='{"type": "subscribed"}')

        with patch("exchanges.lighter.websockets.connect", new_callable=AsyncMock, return_value=mock_ws):
            await bot._ws_connect()

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "subscribe"
        assert sent["channel"] == "account_all/42"
        assert bot._ws is mock_ws


class TestLighterWsReceive:
    """Test _ws_receive routes messages correctly."""

    def _make_bot(self):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot._ws = AsyncMock()
            bot.positions = {}
            return bot

    @pytest.mark.asyncio
    async def test_receive_order_updates(self):
        bot = self._make_bot()
        msg = {"type": "order_update", "orders": [{"order_id": "1"}]}
        bot._ws.recv = AsyncMock(return_value=json.dumps(msg))
        result = await bot._ws_receive()
        assert result == [{"order_id": "1"}]

    @pytest.mark.asyncio
    async def test_receive_null_orders_returns_empty_list(self):
        """Regression: msg.get('orders', []) returned None when the server sent
        JSON null, crashing the watch loop downstream. Use `or []` as a guard."""
        bot = self._make_bot()
        msg = {"type": "account_update", "orders": None}
        bot._ws.recv = AsyncMock(return_value=json.dumps(msg))
        result = await bot._ws_receive()
        assert result == []

    @pytest.mark.asyncio
    async def test_receive_handles_ping(self):
        bot = self._make_bot()
        ping_msg = json.dumps({"type": "ping"})
        real_msg = json.dumps({"type": "order_update", "orders": [{"order_id": "2"}]})
        bot._ws.recv = AsyncMock(side_effect=[ping_msg, real_msg])
        result = await bot._ws_receive()
        # Should have sent pong
        bot._ws.send.assert_called_once()
        pong = json.loads(bot._ws.send.call_args[0][0])
        assert pong["type"] == "pong"
        # Should return the real message orders
        assert result == [{"order_id": "2"}]

    @pytest.mark.asyncio
    async def test_receive_connection_closed_sets_ws_none(self):
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        bot = self._make_bot()
        close_frame = Close(1000, "normal")
        bot._ws.recv = AsyncMock(
            side_effect=ConnectionClosed(close_frame, None)
        )
        with pytest.raises(ConnectionClosed):
            await bot._ws_receive()
        assert bot._ws is None


# ═══════════════════ Task 6: Order normalization tests ═══════════════════


class TestLighterNormalizeOrderUpdate:
    """Test _normalize_order_update maps fields correctly."""

    def _make_bot(self, positions=None):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            bot.positions = positions or {}
            return bot

    def test_normalize_order_maps_fields(self):
        bot = self._make_bot()
        order = {
            "order_id": "12345",
            "symbol": "BTC/USDC:USDC",
            "side": "Buy",
            "order_type": "Limit",
            "price": "50000.0",
            "size": "0.5",
            "status": "open",
            "timestamp": 1700000000,
            "client_order_id": "custom_abc",
        }
        result = bot._normalize_order_update(order)
        assert result["id"] == "12345"
        assert result["symbol"] == "BTC/USDC:USDC"
        assert result["side"] == "buy"
        assert result["type"] == "limit"
        assert result["price"] == 50000.0
        assert result["amount"] == 0.5
        assert result["qty"] == 0.5
        assert result["status"] == "open"
        assert result["timestamp"] == 1700000000
        assert result["clientOrderId"] == "custom_abc"
        assert result["custom_id"] == "custom_abc"
        assert result["info"] is order
        assert result["position_side"] == "long"

    def test_normalize_order_sell_no_position_is_short(self):
        bot = self._make_bot()
        order = {
            "order_id": "99",
            "symbol": "ETH/USDC:USDC",
            "side": "Sell",
            "price": "3000",
            "size": "1.0",
            "status": "open",
        }
        result = bot._normalize_order_update(order)
        assert result["position_side"] == "short"

    def test_normalize_order_filled_status(self):
        bot = self._make_bot()
        order = {
            "order_id": "100",
            "symbol": "BTC/USDC:USDC",
            "side": "Buy",
            "price": "50000",
            "size": "1.0",
            "status": "fully_filled",
        }
        result = bot._normalize_order_update(order)
        assert result["status"] == "closed"


class TestLighterNormalizeStatus:
    """Test _normalize_status maps all statuses correctly."""

    def _make_bot(self):
        with patch("exchanges.lighter.CCXTBot.__init__", return_value=None):
            from exchanges.lighter import LighterBot

            bot = LighterBot.__new__(LighterBot)
            return bot

    def test_normalize_status_mappings(self):
        bot = self._make_bot()
        assert bot._normalize_status("open") == "open"
        assert bot._normalize_status("new") == "open"
        assert bot._normalize_status("partially_filled") == "open"
        assert bot._normalize_status("fully_filled") == "closed"
        assert bot._normalize_status("filled") == "closed"
        assert bot._normalize_status("canceled") == "canceled"
        assert bot._normalize_status("cancelled") == "canceled"
        assert bot._normalize_status("expired") == "canceled"
        assert bot._normalize_status("some_other") == "some_other"


