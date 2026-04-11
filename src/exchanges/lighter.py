"""
LighterBot: Lighter exchange adapter with native WebSocket.

Lighter is a zero-fee decentralized trading platform using wallet-based auth.
Requires a platform-specific native binary (libraryPath) for transaction signing.

Auth fields (api-keys.json):
  - private_key: API key private key (hex) from Lighter's API keys page
  - api_key_index: integer (0-254) assigned to the API key
  - account_index: integer from Lighter API
  - library_path: path to platform-specific native binary

WebSocket: Native connection to wss://mainnet.zklighter.elliot.ai/stream
  - Subscribes to account_all/{account_index} for orders, positions, balance

Reference: https://github.com/ccxt/ccxt/wiki/FAQ#how-to-use-the-lighter-exchange-in-ccxt
"""

import asyncio
import json

import websockets
from websockets.exceptions import ConnectionClosed

from ccxt.static_dependencies.lighter_client.signer import load_lighter_library, decode_tx_info
from exchanges.ccxt_bot import CCXTBot
from passivbot import logging, custom_id_to_snake
from config.access import require_live_value


class LighterBot(CCXTBot):

    def __init__(self, config: dict):
        super().__init__(config)
        self.quote = "USDC"
        self.hedge_mode = False
        self._ws = None

    def _build_ccxt_config(self) -> dict:
        config = {"enableRateLimit": True}
        config.setdefault("timeout", 30000)  # match CCXTBot base; CCXT default ~10s too tight on cold boot
        config["privateKey"] = self.user_info["private_key"]
        config["options"] = {
            "apiKeyIndex": int(self.user_info["api_key_index"]),
            "accountIndex": int(self.user_info["account_index"]),
        }
        if self.user_info.get("library_path"):
            config["options"]["libraryPath"] = self.user_info["library_path"]
        return config

    def create_ccxt_sessions(self):
        super().create_ccxt_sessions()
        library_path = (
            self.user_info.get("library_path") or self.cca.options.get("libraryPath")
        )
        signer = load_lighter_library(library_path)
        url = self.cca.implode_hostname(self.cca.urls["api"]["public"])
        api_key_index = int(self.user_info["api_key_index"])
        account_index = int(self.user_info["account_index"])
        res = signer.CreateClient(
            url.encode("utf-8"),
            self.cca.privateKey.encode("utf-8"),
            self.cca.options["chainId"],
            api_key_index,
            account_index,
        )
        if res is not None:
            err = res.decode("utf-8") if isinstance(res, bytes) else str(res)
            if "error" in err.lower():
                raise Exception(f"lighter: CreateClient failed: {err}")
        self.cca.options["signer"] = signer
        self._signer = signer

    async def update_exchange_config_by_symbols(self, symbols):
        """Set leverage and margin mode via direct signer call.

        Bypasses CCXT's buggy load_account/lighter_sign_update_leverage
        by calling self._signer.SignUpdateLeverage directly.
        """
        for symbol in symbols:
            leverage = None
            margin_mode = None
            try:
                leverage = self._calc_leverage_for_symbol(symbol)
                margin_mode = self._get_margin_mode_for_symbol(symbol)
                market = self.cca.market(symbol)
                api_key_index = int(self.user_info["api_key_index"])
                account_index = int(self.user_info["account_index"])
                nonce = await self.cca.fetch_nonce(account_index, api_key_index)
                tx_type, tx_info, _, error = decode_tx_info(
                    self._signer.SignUpdateLeverage(
                        int(market["id"]),
                        int(10000 / leverage),
                        0 if margin_mode == "cross" else 1,
                        nonce,
                        api_key_index,
                        account_index,
                    )
                )
                if error:
                    raise Exception(error)
                await self.cca.publicPostSendTx({
                    "tx_type": tx_type,
                    "tx_info": tx_info,
                })
                logging.info(f"{symbol}: set {margin_mode} margin, {leverage}x leverage")
            except Exception as e:
                err_str = str(e)
                if "already" in err_str.lower() or "no change" in err_str.lower():
                    logging.info(
                        f"{symbol}: margin/leverage unchanged ({margin_mode}, {leverage}x)"
                    )
                else:
                    # Propagate so the orchestrator's per-symbol retry/backoff
                    # loop (see passivbot.py update_exchange_config) can handle it.
                    raise
            await asyncio.sleep(0.2)

    def _build_order_params(self, order: dict) -> dict:
        params = {
            "reduceOnly": order["reduce_only"],
            "clientOrderId": order["custom_id"],
        }
        if order.get("type") == "limit":
            tif = require_live_value(self.config, "time_in_force")
            if tif == "post_only":
                params["postOnly"] = True
            else:
                # CCXT's Lighter adapter only maps 'gtt' and 'ioc' (not 'gtc');
                # passing 'gtt' internally sets order_expiry=-1, giving GTC semantics.
                params["timeInForce"] = "GTT"
        return params

    # ── REST overrides ────────────────────────────────────────

    async def fetch_open_orders(self, symbol: str = None) -> list:
        """Lighter requires per-symbol queries; gather across tracked symbols."""
        if symbol is not None:
            return await super().fetch_open_orders(symbol=symbol)
        symbols_ = set()
        symbols_.update(s for s in self.open_orders if self.open_orders[s])
        symbols_.update(self.get_symbols_with_pos())
        if hasattr(self, "active_symbols") and self.active_symbols:
            symbols_.update(self.active_symbols)
        if not symbols_:
            return []
        results = await asyncio.gather(
            *[super().fetch_open_orders(symbol=s) for s in sorted(symbols_)]
        )
        return sorted(
            [order for sublist in results for order in sublist],
            key=lambda x: x["timestamp"],
        )

    # ── WebSocket support ──────────────────────────────────────

    def can_watch_orders(self) -> bool:
        """Override: Always True - we implement native WebSocket."""
        return True

    def _get_ws_url(self) -> str:
        """Return WebSocket URL, respecting testnet setting."""
        if self.cca.urls.get("test", {}).get("public"):
            return "wss://testnet.zklighter.elliot.ai/stream"
        return "wss://mainnet.zklighter.elliot.ai/stream"

    async def _ws_connect(self):
        """Establish WebSocket connection and subscribe to account channel."""
        url = self._get_ws_url()
        self._ws = await websockets.connect(url)
        logging.info(f"lighter: WebSocket connected to {url}")

        account_index = int(self.user_info["account_index"])
        subscribe_msg = json.dumps({
            "type": "subscribe",
            "channel": f"account_all/{account_index}",
        })
        await self._ws.send(subscribe_msg)
        raw = await self._ws.recv()
        logging.info(f"lighter: subscription confirmation: {raw}")

    async def _ws_receive(self) -> list:
        """Receive and route WebSocket messages, returning order updates."""
        while True:
            try:
                raw = await self._ws.recv()
            except ConnectionClosed:
                self._ws = None
                raise

            msg = json.loads(raw)

            if msg.get("type") == "ping":
                await self._ws.send(json.dumps({"type": "pong"}))
                continue

            if "update" not in msg.get("type", ""):
                continue

            return msg.get("orders") or []

    async def _do_watch_orders(self) -> list:
        """Hook: Connect if needed, then receive order updates."""
        if self._ws is None or self._ws.closed:
            await self._ws_connect()
        return await self._ws_receive()

    # ── Order normalization ────────────────────────────────────

    def _normalize_order_update(self, order: dict) -> dict:
        """Transform Lighter WS order format to passivbot format."""
        custom_id = order.get("client_order_id") or ""
        reduce_only = bool(order.get("reduce_only"))
        if not reduce_only and custom_id:
            snake = custom_id_to_snake(custom_id)
            reduce_only = "close" in snake
        price = float(order.get("price") or 0)
        size = float(order.get("size") or 0)
        normalized = {
            "id": order.get("order_id"),
            "symbol": order.get("symbol", ""),
            "side": (order.get("side") or "").lower(),
            "type": (order.get("order_type") or "limit").lower(),
            "price": price,
            "amount": size,
            "qty": size,
            "reduceOnly": reduce_only,
            "status": self._normalize_status(order.get("status")),
            "timestamp": order.get("timestamp"),
            "clientOrderId": custom_id,
            "custom_id": custom_id,
            "info": order,
        }
        normalized["position_side"] = self._get_position_side_for_order(normalized)
        return normalized

    def _normalize_status(self, status: str) -> str:
        """Map Lighter status strings to passivbot statuses."""
        mapping = {
            "open": "open",
            "new": "open",
            "partially_filled": "open",
            "fully_filled": "closed",
            "filled": "closed",
            "canceled": "canceled",
            "cancelled": "canceled",
            "expired": "canceled",
        }
        return mapping.get(status, (status or "").lower())
