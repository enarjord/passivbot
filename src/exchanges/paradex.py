"""
ParadexBot: Paradex-specific exchange connector.

Extends CCXTBot with Paradex-specific onboarding logic.
Paradex requires users to call the /onboarding endpoint once
before any authenticated API calls will work.

See: https://docs.paradex.trade/api-reference/general-information/authentication
"""

import json

import ccxt
import websockets
from websockets.exceptions import ConnectionClosed

from exchanges.ccxt_bot import CCXTBot
from passivbot import logging


class ParadexBot(CCXTBot):
    """Paradex exchange bot with automatic onboarding support."""

    # Auth errors that should not trigger retry
    WS_AUTH_ERROR_CODES = {40110, 40111, 40112}

    def __init__(self, config: dict):
        super().__init__(config)
        # Paradex only supports USDC as quote currency - must set after super()
        # to override CCXTBot's default of USDT
        self.quote = "USDC"
        self.hedge_mode = False  # Paradex doesn't support two-way mode
        self._ws = None

    def _build_ccxt_config(self) -> dict:
        """Convert wallet_address/private_key to CCXT's paradexAccount format."""
        config = {"enableRateLimit": True}

        # Build the nested structure CCXT expects for Paradex L2 auth
        if self.user_info.get("wallet_address") and self.user_info.get("private_key"):
            config["options"] = {
                "paradexAccount": {
                    "address": self.user_info["wallet_address"],
                    "privateKey": self.user_info["private_key"],
                }
            }

        return config

    async def init_markets(self, verbose=True):
        """Override to ensure onboarding before market initialization.

        Paradex requires the account to be onboarded before any authenticated
        API calls work. CCXT throws:
        - BadRequest with 'ETHEREUM_ADDRESS_ALREADY_ONBOARDED' if already done
        - BadRequest with 'NOT_ONBOARDED' if not done (but we call onboarding first)

        When using L2-only auth (wallet_address/private_key), onboarding is skipped
        since the account is already onboarded via the Paradex UI.
        """
        # Skip onboarding if using L2-only auth (wallet_address/private_key)
        if self.user_info.get("wallet_address"):
            logging.info("paradex: using L2-only auth, skipping onboarding")
        else:
            try:
                await self.cca.onboarding()
                logging.info("paradex: onboarding successful")
            except ccxt.BadRequest as e:
                if "ALREADY_ONBOARDED" not in str(e):
                    raise
                logging.info("paradex: account already onboarded")

        await super().init_markets(verbose)

    def _should_set_margin_mode(self, symbol: str) -> bool:
        """Paradex uses cross margin only — no API to set it."""
        return False

    def can_watch_orders(self) -> bool:
        """Override: Always True - we implement native WebSocket."""
        return True

    async def _do_watch_orders(self) -> list:
        """Hook: Connect if needed, then receive order updates."""
        if self._ws is None or self._ws.closed:
            await self._ws_connect()
        return await self._ws_receive_orders()

    def _get_ws_url(self) -> str:
        """Return WebSocket URL, respecting testnet setting."""
        api_url = self.cca.urls.get("api", {}).get("public", "")
        if "testnet" in api_url:
            return "wss://ws.api.testnet.paradex.trade/v1"
        return "wss://ws.api.prod.paradex.trade/v1"

    async def _ws_send_and_expect(self, method: str, params: dict, msg_id: int, success_log: str):
        """Send JSON-RPC message and validate response."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params, "id": msg_id}
        try:
            await self._ws.send(json.dumps(msg))
            raw = await self._ws.recv()
        except ConnectionClosed as e:
            self._ws = None
            raise ConnectionError(f"paradex WS closed during {method}: {e}")
        except Exception as e:
            raise ConnectionError(f"paradex WS {method} error: {e}")

        response = json.loads(raw)
        if "error" in response:
            code = response["error"].get("code", 0)
            message = response["error"].get("message", "unknown")
            if code in self.WS_AUTH_ERROR_CODES:
                raise ccxt.AuthenticationError(f"paradex WS {method}: [{code}] {message}")
            raise Exception(f"paradex WS {method}: [{code}] {message}")

        logging.info(success_log)
        return response

    async def _ws_connect(self):
        """Establish WebSocket connection, authenticate, and subscribe."""
        url = self._get_ws_url()
        try:
            self._ws = await websockets.connect(url)
        except Exception as e:
            raise ConnectionError(f"paradex WS connect failed: {e}")
        logging.info("paradex: WebSocket connected")

        jwt = await self.cca.authenticate_rest()
        await self._ws_send_and_expect("auth", {"bearer": jwt}, 1, "paradex: WS authenticated")
        await self._ws_send_and_expect(
            "subscribe", {"channel": "orders.ALL"}, 2, "paradex: subscribed to orders.ALL"
        )

    async def _ws_receive_orders(self) -> list:
        """Receive next message and extract order updates."""
        try:
            raw = await self._ws.recv()
        except ConnectionClosed as e:
            self._ws = None
            raise ConnectionError(f"paradex WS closed: {e}")

        msg = json.loads(raw)
        if msg.get("method") != "subscription":
            return []

        params = msg.get("params", {})
        if not params.get("channel", "").startswith("orders."):
            return []

        data = params.get("data")
        return [data] if data else []

    def _normalize_order_update(self, order: dict) -> dict:
        """Transform Paradex order format to passivbot format."""
        custom_id = order.get("client_id") or ""
        normalized = {
            "id": order.get("id"),
            "symbol": self._paradex_market_to_symbol(order.get("market")),
            "side": (order.get("side") or "").lower(),
            "type": (order.get("type") or "").lower(),
            "price": float(order.get("price") or 0),
            "amount": float(order.get("size") or 0),
            "qty": float(order.get("size") or 0),
            "status": self._normalize_status(order.get("status")),
            "timestamp": order.get("created_at"),
            "clientOrderId": custom_id,
            "custom_id": custom_id,
            "info": order,
        }
        normalized["position_side"] = self._get_position_side_for_order(normalized)
        return normalized

    def _normalize_status(self, status: str) -> str:
        """Map Paradex status to CCXT-style status."""
        mapping = {"NEW": "open", "UNTRIGGERED": "open", "OPEN": "open", "CLOSED": "closed"}
        return mapping.get(status, (status or "").lower())

    def _paradex_market_to_symbol(self, market: str) -> str:
        """Convert Paradex market to CCXT symbol."""
        if market in self.markets_dict:
            return market
        for symbol, info in self.markets_dict.items():
            if info.get("id") == market:
                return symbol
        return market

    def did_cancel_order(self, executed, order=None) -> bool:
        """Paradex returns 204 No Content on successful cancellation.

        CCXT translates this to an order structure with all None values.
        We detect success by checking that we received the expected structure
        (has 'id' key) rather than an empty error dict {}.
        """
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        # Success: CCXT returns {'id': None, 'status': None, ...} (full structure)
        # Error: execute_cancellation returns {} (empty dict)
        return "id" in executed
