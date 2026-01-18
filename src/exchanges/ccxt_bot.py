"""
CCXTBot: Universal exchange connector using CCXT unified API.

This is a base class for quickly onboarding new exchanges. Subclass this
and override only the methods that need exchange-specific behavior.

See docs/plans/2026-01-02-ccxtbot-design.md for design rationale.

Hook Taxonomy
=============
CCXTBot uses a consistent naming convention for extension points:

    can_*        - Capability checks (return bool)
                   Example: can_watch_orders() -> True if WebSocket supported

    _do_*        - Async actions that call the exchange API
                   Example: _do_fetch_balance() -> dict from CCXT

    _get_*       - Value extraction from API responses
                   Example: _get_balance(fetched) -> float

    _normalize_* - Data transformation to passivbot format
                   Example: _normalize_positions(fetched) -> list[dict]

    _build_*     - Config/parameter construction
                   Example: _build_order_params(order) -> dict for CCXT

To customize behavior for a new exchange:
1. Subclass CCXTBot
2. Override only the hooks that need exchange-specific logic
3. Template methods (fetch_balance, fetch_positions, etc.) orchestrate the hooks
"""

import asyncio
import time
import traceback

from passivbot import Passivbot, logging, custom_id_to_snake
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
from procedures import assert_correct_ccxt_version
from config_utils import require_live_value

assert_correct_ccxt_version(ccxt=ccxt_async)


class CCXTBot(Passivbot):
    """Generic exchange bot using CCXT unified API.

    See module docstring for hook taxonomy and extension patterns.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.quote = self.user_info.get("quote", "USDT")

    # ═══════════════════ ORDER WATCHING HOOKS ═══════════════════

    def can_watch_orders(self) -> bool:
        """Hook: Can this exchange watch orders in real-time?

        Default: Check CCXT's has['watchOrders']
        Override: Return True if implementing native WebSocket
        """
        if self.ccp is None:
            return False
        return bool(self.ccp.has.get("watchOrders"))

    async def _do_watch_orders(self) -> list:
        """Hook: Fetch the next batch of order updates.

        Default: Use CCXT's watchOrders()
        Override: Implement native WebSocket
        """
        return await self.ccp.watch_orders()

    def _normalize_order_update(self, order: dict) -> dict:
        """Hook: Transform raw order to passivbot format.

        Default: Handle CCXT unified format
        Override: Handle exchange-specific format
        """
        order["position_side"] = self._get_position_side_for_order(order)
        order["qty"] = order["amount"]
        return order

    def _get_position_side_for_order(self, order: dict) -> str:
        """Hook: Derive position_side from order data.

        Default: Use CCXT unified fields, fall back to custom_id derivation.
        Override: Exchange-specific logic when neither source is available.
        """
        info = order.get("info", {})

        # 1. Exchange provides positionSide directly
        pos_side = info.get("positionSide", "")
        if pos_side:
            return pos_side.lower()

        # 2. Derive from CCXT unified clientOrderId
        custom_id = order.get("clientOrderId", "")
        if custom_id:
            order_type = custom_id_to_snake(custom_id)
            if order_type.endswith("_long"):
                return "long"
            if order_type.endswith("_short"):
                return "short"

        return "both"

    # ═══════════════════ PNL FETCHING HOOKS ═══════════════════

    def _get_pnl_from_trade(self, trade: dict) -> float:
        """Hook: Extract realized PnL from trade.

        Default: Look for common CCXT info fields
        Override: Exchange-specific field names
        """
        info = trade.get("info", {})
        for field in ["realized_pnl", "realizedPnl", "pnl", "profit"]:
            if field in info:
                return float(info[field])
        return 0.0

    def _get_position_side_from_trade(self, trade: dict) -> str:
        """Hook: Determine position side from trade.

        Default: Infer from side + PnL (entry vs exit)
        Override: Exchange-specific logic

        Logic: PnL=0 means entry, PnL!=0 means exit
        - buy + entry = long
        - buy + exit = short (closing short)
        - sell + entry = short
        - sell + exit = long (closing long)
        """
        pnl = self._get_pnl_from_trade(trade)
        if trade["side"] == "buy":
            return "long" if pnl == 0.0 else "short"
        else:
            return "short" if pnl == 0.0 else "long"

    def _build_ccxt_config(self) -> dict:
        """Build CCXT config by passing through all user_info fields.

        CCXT ignores unknown fields, so we pass everything except
        passivbot-specific fields. Users can use any CCXT-supported
        credential field directly in api-keys.json (apiKey, secret,
        password, walletAddress, privateKey, etc.).
        """
        # Fields used by passivbot, not CCXT
        passivbot_fields = {"exchange", "options", "quote"}

        config = {k: v for k, v in self.user_info.items() if k not in passivbot_fields}
        config["enableRateLimit"] = True

        # Remap legacy credential field names to CCXT-native names for backwards compatibility
        legacy_mappings = {
            "key": "apiKey",
            "api_key": "apiKey",
            "wallet": "walletAddress",
            "private_key": "privateKey",
            "passphrase": "password",
            "wallet_address": "walletAddress",
        }
        for old_name, new_name in legacy_mappings.items():
            if old_name in config and new_name not in config:
                logging.warning(
                    f"{self.exchange}: '{old_name}' in api-keys.json is deprecated, "
                    f"use '{new_name}' instead (CCXT-native field name)"
                )
                config[new_name] = config.pop(old_name)

        return config

    def create_ccxt_sessions(self):
        """Initialize REST and WebSocket CCXT clients.

        The REST client (cca) is always created. The WebSocket client (ccp)
        is created only when ws_enabled=True; otherwise ccp is set to None
        and the bot falls back to REST polling for order updates.
        """
        ccxt_config = self._build_ccxt_config()
        user_options = self.user_info.get("options", {})

        # REST client
        exchange_class = getattr(ccxt_async, self.exchange)
        self.cca = exchange_class(ccxt_config)
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options.update(user_options)
        self.cca.options["defaultType"] = "swap"
        self._apply_endpoint_override(self.cca)

        # WebSocket client - optional, enables faster order updates
        if self.ws_enabled:
            ws_class = getattr(ccxt_pro, self.exchange)
            self.ccp = ws_class(ccxt_config)
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options.update(user_options)
            self.ccp.options["defaultType"] = "swap"
            self._apply_endpoint_override(self.ccp)
        else:
            self.ccp = None
            logging.info(f"{self.exchange}: WebSocket disabled, using REST polling")

    async def validate_websocket_support(self):
        """Check WebSocket capabilities (informational, non-fatal).

        Logs whether watchOrders is available. Does not raise.
        Subclasses can override to set ws_orders_supported=True if
        implementing native WebSocket.
        """
        if self.ccp is None:
            logging.info(f"{self.exchange}: WebSocket client not initialized")
            return

        if self.ccp.has.get("watchOrders"):
            logging.info(f"{self.exchange}: watchOrders support confirmed")
        else:
            logging.info(
                f"{self.exchange}: watchOrders not supported in CCXT, using REST polling"
            )

    async def determine_utc_offset(self, verbose=True):
        """Derive the exchange server time offset using CCXT's fetch_time().

        Overrides base class which expects fetch_balance() to return timestamp.
        """
        from utils import utc_ms

        try:
            server_time = await self.cca.fetch_time()
            self.utc_offset = round((server_time - utc_ms()) / (1000 * 60 * 60)) * (
                1000 * 60 * 60
            )
            if verbose:
                logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")
        except Exception as e:
            logging.warning(f"Could not fetch server time: {e}, using 0 offset")
            self.utc_offset = 0

    async def fetch_balance(self) -> float:
        """Template method: Fetch account balance for quote currency.

        Uses hooks:
        - _do_fetch_balance(): Call exchange API
        - _get_balance(): Extract balance value

        Returns:
            float: Total balance in quote currency, or 0.0 if not found.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        fetched = await self._do_fetch_balance()
        return self._get_balance(fetched)

    async def _do_fetch_balance(self) -> dict:
        """Hook: Call exchange API for balance.

        Default: Use CCXT's fetch_balance()
        Override: Custom API call or different endpoint
        """
        logging.debug(f"{self.exchange}: fetching balance via CCXT fetch_balance()")
        t0 = time.time()
        result = await self.cca.fetch_balance()
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_balance completed in {elapsed_ms:.1f}ms")
        return result

    def _get_balance(self, fetched: dict) -> float:
        """Hook: Extract balance value from response.

        Default: CCXT unified format total[quote]
        Override: Exchange-specific field paths (e.g., info.totalCrossWalletBalance)
        """
        return float(fetched.get("total", {}).get(self.quote, 0))

    async def fetch_positions(self) -> list:
        """Template method: Fetch all open positions.

        Uses hooks:
        - _do_fetch_positions(): Call exchange API
        - _normalize_positions(): Transform to passivbot format
        - _get_position_side(): Derive position_side per position

        Returns:
            list: List of position dicts with normalized fields.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        fetched = await self._do_fetch_positions()
        return self._normalize_positions(fetched)

    async def _do_fetch_positions(self) -> list:
        """Hook: Call exchange API for positions.

        Default: Use CCXT's fetch_positions()
        Override: Custom API call
        """
        logging.debug(f"{self.exchange}: fetching positions via CCXT fetch_positions()")
        t0 = time.time()
        result = await self.cca.fetch_positions()
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_positions completed in {elapsed_ms:.1f}ms, {len(result)} raw positions")
        return result

    def _normalize_positions(self, fetched: list) -> list:
        """Hook: Transform raw positions to passivbot format.

        Default: Use CCXT unified fields (contracts, entryPrice, side)
        Override: Exchange-specific field mappings
        """
        positions = []
        for elm in fetched:
            contracts = float(elm.get("contracts", 0))
            if contracts != 0:
                positions.append({
                    "symbol": elm["symbol"],
                    "position_side": self._get_position_side(elm),
                    "size": contracts,
                    "price": float(elm.get("entryPrice", 0)),
                })
        return positions

    def _get_position_side(self, elm: dict) -> str:
        """Hook: Derive position_side from position data.

        Default: CCXT unified 'side' field
        Override: Exchange-specific logic (e.g., info.positionSide)
        """
        return elm.get("side", "long").lower()

    async def fetch_open_orders(self, symbol: str = None) -> list:
        """Fetch open orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol to filter orders.

        Returns:
            list: Orders sorted by timestamp with normalized fields.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        sym_str = symbol if symbol else "all symbols"
        logging.debug(f"{self.exchange}: fetching open orders for {sym_str}")
        t0 = time.time()
        fetched = await self.cca.fetch_open_orders(symbol=symbol)
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_open_orders completed in {elapsed_ms:.1f}ms, {len(fetched)} orders")
        for elm in fetched:
            elm["position_side"] = self._get_position_side_for_order(elm)
            elm["qty"] = elm["amount"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def watch_orders(self):
        """Template method: Watch for order updates.

        Uses hooks for customization:
        - can_watch_orders(): Check if watching is available
        - _do_watch_orders(): Get raw order updates
        - _normalize_order_update(): Transform to passivbot format

        If watching is not available, exits gracefully (polling handles updates).
        """
        if not self.can_watch_orders():
            logging.info(f"[ws] {self.exchange}: watch_orders not available, using REST polling")
            return

        logging.info(f"[ws] {self.exchange}: starting order watch")
        while True:
            try:
                if self.stop_websocket:
                    break
                raw_orders = await self._do_watch_orders()
                normalized = [self._normalize_order_update(o) for o in raw_orders]
                self.handle_order_update(normalized)
            except Exception as e:
                logging.error(f"[ws] exception in watch_orders: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def update_exchange_config(self):
        """Set exchange to hedge mode if supported.

        Uses capability check to determine if exchange supports position mode setting.
        Skips gracefully if not supported.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        if not self.cca.has.get("setPositionMode"):
            logging.info(f"{self.exchange} does not support setPositionMode, skipping")
            return

        logging.debug(f"{self.exchange}: setting position mode to hedge via CCXT set_position_mode(True)")
        t0 = time.time()
        res = await self.cca.set_position_mode(True)
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: set_position_mode completed in {elapsed_ms:.1f}ms")
        logging.info(f"Set hedge mode: {res}")

    def _should_set_margin_mode(self, symbol: str) -> bool:
        """Hook: Should we call set_margin_mode for this symbol?

        Default: Check CCXT's has['setMarginMode']
        Override: Return False if exchange doesn't support/need it
        """
        return self.cca.has.get("setMarginMode", False)

    async def update_exchange_config_by_symbols(self, symbols: list):
        """Set leverage and margin mode for each symbol.

        Args:
            symbols: List of symbols to configure.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        can_set_leverage = self.cca.has.get("setLeverage", False)

        for symbol in symbols:
            if can_set_leverage:
                leverage = int(self.config_get(["live", "leverage"], symbol=symbol))
                logging.debug(f"{self.exchange}: setting leverage for {symbol} to {leverage}x")
                t0 = time.time()
                await self.cca.set_leverage(leverage, symbol=symbol)
                elapsed_ms = (time.time() - t0) * 1000
                logging.debug(f"{self.exchange}: set_leverage completed in {elapsed_ms:.1f}ms")
                logging.info(f"{symbol}: set leverage to {leverage}x")

            if self._should_set_margin_mode(symbol):
                logging.debug(f"{self.exchange}: setting margin mode for {symbol} to cross")
                t0 = time.time()
                await self.cca.set_margin_mode("cross", symbol=symbol)
                elapsed_ms = (time.time() - t0) * 1000
                logging.debug(f"{self.exchange}: set_margin_mode completed in {elapsed_ms:.1f}ms")
                logging.info(f"{symbol}: set cross margin mode")

    def set_market_specific_settings(self):
        """Extract market-specific settings from CCXT market info.

        Populates symbol_ids, min_costs, min_qtys, qty_steps, price_steps, and c_mults
        from CCXT's unified market structure.
        """
        super().set_market_specific_settings()
        for symbol, market in self.markets_dict.items():
            self.symbol_ids[symbol] = market["id"]
            self.min_costs[symbol] = market["limits"]["cost"]["min"] or 0.1
            self.min_qtys[symbol] = (
                market["precision"]["amount"]
                if market["limits"]["amount"]["min"] is None
                else market["limits"]["amount"]["min"]
            )
            self.qty_steps[symbol] = market["precision"]["amount"]
            self.price_steps[symbol] = market["precision"]["price"]
            self.c_mults[symbol] = market.get("contractSize", 1)

    async def fetch_tickers(self) -> dict:
        """Template method: Fetch current ticker data for all markets.

        Uses hooks:
        - _do_fetch_tickers(): Call exchange API
        - _normalize_tickers(): Transform to {symbol: {bid, ask, last}}

        Returns:
            dict: Ticker data keyed by symbol with bid/ask/last prices.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        fetched = await self._do_fetch_tickers()
        return self._normalize_tickers(fetched)

    async def _do_fetch_tickers(self) -> dict:
        """Hook: Call exchange API for tickers.

        Default: Use CCXT's fetch_tickers()
        Override: Custom API call or different endpoint
        """
        logging.debug(f"{self.exchange}: fetching tickers via CCXT fetch_tickers()")
        t0 = time.time()
        result = await self.cca.fetch_tickers()
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_tickers completed in {elapsed_ms:.1f}ms, {len(result)} tickers")
        return result

    def _normalize_tickers(self, fetched: dict) -> dict:
        """Hook: Transform to {symbol: {bid, ask, last}} format.

        Default: Use CCXT unified fields, filter to markets_dict
        Override: Exchange-specific field mappings
        """
        tickers = {}
        for symbol, data in fetched.items():
            if symbol in self.markets_dict:
                tickers[symbol] = {
                    "bid": float(data.get("bid") or 0),
                    "ask": float(data.get("ask") or 0),
                    "last": float(data.get("last") or data.get("bid") or 0),
                }
        return tickers

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m") -> list:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe (default "1m").

        Returns:
            list: OHLCV data.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        logging.debug(f"{self.exchange}: fetching OHLCV for {symbol} ({timeframe})")
        t0 = time.time()
        result = await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_ohlcv completed in {elapsed_ms:.1f}ms, {len(result)} candles")
        return result

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit: int = None) -> list:
        """Fetch 1-minute OHLCV data with pagination support.

        Args:
            symbol: Trading pair symbol.
            since: Start timestamp in milliseconds.
            limit: Maximum number of candles.

        Returns:
            list: Sorted OHLCV candles by timestamp.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        n_limit = limit or 1000
        logging.debug(f"{self.exchange}: fetching 1m OHLCV for {symbol}, since={since}, limit={n_limit}")
        t0 = time.time()

        if since is None:
            result = await self.cca.fetch_ohlcv(symbol, timeframe="1m", limit=n_limit)
            elapsed_ms = (time.time() - t0) * 1000
            logging.debug(f"{self.exchange}: fetch_ohlcvs_1m completed in {elapsed_ms:.1f}ms, {len(result)} candles")
            return result

        since = int(since // 60000 * 60000)  # Round to minute
        all_candles = {}
        page_count = 0
        for _ in range(5):  # Max 5 paginated requests
            fetched = await self.cca.fetch_ohlcv(
                symbol, timeframe="1m", since=since, limit=n_limit
            )
            page_count += 1
            if not fetched:
                break
            for candle in fetched:
                all_candles[candle[0]] = candle
            if len(fetched) < n_limit:
                break
            since = fetched[-1][0]

        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(f"{self.exchange}: fetch_ohlcvs_1m completed in {elapsed_ms:.1f}ms, {len(all_candles)} candles ({page_count} pages)")
        return sorted(all_candles.values(), key=lambda x: x[0])

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None) -> list:
        """Template method: Fetch trade history for PnL tracking.

        Uses hooks:
        - _do_fetch_pnls(): Call exchange API
        - _normalize_pnls(): Add pnl, position_side, qty to each trade
        - _get_pnl_from_trade(): Extract PnL value
        - _get_position_side_from_trade(): Derive position_side

        Args:
            start_time: Start timestamp in milliseconds.
            end_time: End timestamp in milliseconds.
            limit: Maximum number of trades to fetch.

        Returns:
            list: Trades sorted by timestamp with pnl, position_side, qty fields.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        trades = await self._do_fetch_pnls(start_time, end_time, limit)
        return self._normalize_pnls(trades)

    async def _do_fetch_pnls(self, start_time, end_time, limit) -> list:
        """Hook: Call exchange API for trades.

        Default: Use CCXT's fetch_my_trades()
        Override: Custom API call or different endpoint
        """
        logging.debug(
            f"{self.exchange}: fetching PnLs via CCXT fetch_my_trades(), "
            f"since={start_time}, end_time={end_time}, limit={limit}"
        )
        t0 = time.time()
        params = {}
        if end_time:
            params["until"] = int(end_time)
        result = await self.cca.fetch_my_trades(
            symbol=None,
            since=int(start_time) if start_time else None,
            limit=limit,
            params=params,
        )
        elapsed_ms = (time.time() - t0) * 1000
        logging.debug(
            f"{self.exchange}: fetch_my_trades completed in {elapsed_ms:.1f}ms, {len(result)} trades"
        )
        return result

    def _normalize_pnls(self, trades: list) -> list:
        """Hook: Add pnl, position_side, qty to each trade.

        Default: Use _get_pnl_from_trade and _get_position_side_from_trade
        Override: Exchange-specific normalization
        """
        for trade in trades:
            trade["qty"] = trade["amount"]
            trade["pnl"] = self._get_pnl_from_trade(trade)
            trade["position_side"] = self._get_position_side_from_trade(trade)
        return sorted(trades, key=lambda x: x["timestamp"])

    def _build_order_params(self, order: dict) -> dict:
        """Hook: Build execution parameters for CCXT order creation.

        Default: Handle positionSide, clientOrderId, postOnly/timeInForce
        Override: Exchange-specific parameter requirements

        Args:
            order: Order dict with type, position_side, custom_id, etc.

        Returns:
            dict: Parameters for CCXT create_order.
        """
        params = {}

        if order.get("position_side"):
            params["positionSide"] = order["position_side"].upper()

        if order.get("custom_id"):
            params["clientOrderId"] = order["custom_id"]

        if order.get("type") == "limit":
            tif = require_live_value(self.config, "time_in_force")
            if tif == "post_only":
                params["postOnly"] = True
            else:
                params["timeInForce"] = "GTC"

        return params

    async def execute_orders(self, orders: list[dict]) -> list[dict]:
        """Execute order creations in parallel using asyncio.gather.

        Unlike the base class sequential approach, this fires all orders
        concurrently for better latency on exchanges with good rate limits.
        """
        if not orders:
            return []

        tasks = [self.execute_order(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for exceptions and trigger error handling if needed
        any_exceptions = any(isinstance(r, Exception) for r in results)
        if any_exceptions:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"error executing order {orders[i]}: {result}")
            await self.restart_bot_on_too_many_errors()

        return results

    async def execute_cancellations(self, orders: list[dict]) -> list[dict]:
        """Execute order cancellations in parallel using asyncio.gather."""
        if not orders:
            return []

        tasks = [self.execute_cancellation(order) for order in orders]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        any_exceptions = any(isinstance(r, Exception) for r in results)
        if any_exceptions:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"error cancelling order {orders[i]}: {result}")
            await self.restart_bot_on_too_many_errors()

        return results