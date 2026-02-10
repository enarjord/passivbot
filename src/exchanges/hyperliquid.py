import asyncio
import json
import traceback

import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import passivbot_rust as pbr

from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
from utils import ts_to_date, utc_ms
from config_utils import require_live_value
from pure_funcs import calc_hash
from procedures import print_async_exception, assert_correct_ccxt_version

round_ = pbr.round_
round_dynamic = pbr.round_dynamic
round_dynamic_up = pbr.round_dynamic_up
round_dynamic_dn = pbr.round_dynamic_dn

assert_correct_ccxt_version(ccxt=ccxt_async)


class HyperliquidBot(CCXTBot):
    # HIP-3 stock perps have a max leverage of 10x
    HIP3_MAX_LEVERAGE = 10
    # HIP-3 symbols use "xyz:" prefix (TradeXYZ builder)
    HIP3_PREFIX = "xyz:"

    def __init__(self, config: dict):
        super().__init__(config)
        self.quote = "USDC"
        self.hedge_mode = False
        self.significant_digits = {}
        if "is_vault" not in self.user_info or self.user_info["is_vault"] == "":
            logging.info(
                f"parameter 'is_vault' missing from api-keys.json for user {self.user}. Setting to false"
            )
            self.user_info["is_vault"] = False
        self.max_n_concurrent_ohlcvs_1m_updates = 2
        self.custom_id_max_length = 34

    def create_ccxt_sessions(self):
        creds = {
            "walletAddress": self.user_info["wallet_address"],
            "privateKey": self.user_info["private_key"],
        }
        # Configure fetchMarkets to include HIP-3 stock perps from TradeXYZ
        fetch_markets_config = {
            "types": ["swap", "hip3"],  # Include HIP-3 markets
            "hip3": {
                "dex": ["xyz"],  # TradeXYZ DEX for stock perps (TSLA, NVDA, etc.)
            },
        }
        if self.ws_enabled:
            self.ccp = getattr(ccxt_pro, self.exchange)(creds)
            self.ccp.options.update(self._build_ccxt_options())
            self.ccp.options["defaultType"] = "swap"
            self.ccp.options["fetchMarkets"] = fetch_markets_config
            self._apply_endpoint_override(self.ccp)
        elif self.endpoint_override:
            logging.info("Skipping Hyperliquid websocket session due to custom endpoint override.")
        self.cca = getattr(ccxt_async, self.exchange)(creds)
        self.cca.options.update(self._build_ccxt_options())
        self.cca.options["defaultType"] = "swap"
        self.cca.options["fetchMarkets"] = fetch_markets_config
        self._apply_endpoint_override(self.cca)
        self._apply_builder_code_options()

    # ═══════════════════ BUILDER CODE SUPPORT ═══════════════════

    def _apply_builder_code_options(self):
        """Apply Hyperliquid builder code and referral options to CCXT sessions.

        Sets builder/ref CCXT options from broker_codes.hjson, but suppresses
        CCXT's auto-approval (we handle it ourselves in _init_builder_codes).
        """
        if not self.broker_code:
            return
        hl_opts = {}
        if isinstance(self.broker_code, dict):
            if "ref" in self.broker_code:
                hl_opts["ref"] = self.broker_code["ref"]
            if "builder" in self.broker_code:
                hl_opts["builder"] = self.broker_code["builder"]
                hl_opts["feeRate"] = self.broker_code.get("fee_rate", "0.02%")
                hl_opts["feeInt"] = self.broker_code.get("fee_int", 20)
        else:
            hl_opts["ref"] = str(self.broker_code)
        # Suppress CCXT's auto-approval; we handle it in _init_builder_codes
        hl_opts["builderFee"] = False
        hl_opts["approvedBuilderFee"] = False
        for client in [self.cca, getattr(self, "ccp", None)]:
            if client is not None:
                client.options.update(hl_opts)

    def _has_builder_config(self) -> bool:
        return isinstance(self.broker_code, dict) and "builder" in self.broker_code

    async def _check_builder_fee_approved(self) -> bool:
        """Query Hyperliquid info endpoint to check if builder fee is approved."""
        if not self._has_builder_config():
            return False
        try:
            builder_addr = self.broker_code["builder"]
            wallet_addr = self.user_info["wallet_address"]
            res = await self.cca.fetch(
                "https://api.hyperliquid.xyz/info",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({
                    "type": "maxBuilderFee",
                    "user": wallet_addr,
                    "builder": builder_addr,
                }),
            )
            max_fee = int(res) if res else 0
            required_fee = self.broker_code.get("fee_int", 20)
            return max_fee >= required_fee
        except Exception as e:
            logging.debug(f"[builder] fee approval check failed: {e}")
            return False

    def _set_builder_approved(self, approved: bool):
        """Set approvedBuilderFee on all CCXT clients."""
        self.cca.options["approvedBuilderFee"] = approved
        if hasattr(self, "ccp") and self.ccp is not None:
            self.ccp.options["approvedBuilderFee"] = approved

    async def _init_builder_codes(self):
        """One-time async builder code initialization. Three paths:

        Path A: Already approved → thank-you banner
        Path B: Not approved + main wallet → auto-approve, notice + thank-you
        Path C: Not approved + agent wallet → force-enable, will nag on order failure
        """
        if not self._has_builder_config():
            return

        builder_addr = self.broker_code["builder"]
        fee_rate = self.broker_code.get("fee_rate", "0.02%")

        # Path A: check if already approved
        if await self._check_builder_fee_approved():
            self._set_builder_approved(True)
            logging.info(
                "[builder] Builder code active. Thank you for supporting Passivbot development!"
            )
            return

        # Not yet approved - try to approve (succeeds only with main wallet key)
        try:
            await self.cca.approve_builder_fee(builder_addr, fee_rate)
            # Path B: approval succeeded (main wallet)
            self._set_builder_approved(True)
            self._print_boxed_banner([
                "NOTICE: Passivbot builder code approved",
                " ",
                f"Builder: {builder_addr}",
                f"Fee: {fee_rate} per trade, to support Passivbot development.",
                " ",
                "You can revoke this at any time via the Hyperliquid web interface.",
                "To adjust the fee, edit broker_codes.hjson.",
            ])
            logging.info(
                "[builder] Builder code active. Thank you for supporting Passivbot development!"
            )
            return
        except Exception as e:
            logging.debug(f"[builder] auto-approval failed (expected for agent wallets): {e}")

        # Path C: agent wallet, not approved - force-enable to trigger order failure nag
        self._set_builder_approved(True)
        self._builder_pending_approval = True
        logging.info(
            "[builder] Builder code not yet approved. Will prompt on first order."
        )

    def _is_builder_fee_error(self, error) -> bool:
        """Check if an error is a Hyperliquid builder-fee-not-approved error."""
        err_str = str(error).lower()
        return "builder fee" in err_str and "not been approved" in err_str

    def _print_builder_nag_banner(self):
        """Print the nag banner for Path C users who haven't approved builder codes."""
        builder_addr = self.broker_code["builder"]
        fee_rate = self.broker_code.get("fee_rate", "0.02%")
        self._print_boxed_banner([
            "Passivbot builder code is NOT approved on your Hyperliquid account.",
            " ",
            f"Builder codes help fund Passivbot development at a small fee ({fee_rate} per trade).",
            " ",
            "To approve (one-time), choose one of:",
            " ",
            "  1. Switch to main wallet key in api-keys.json (bot will auto-approve on restart)",
            "  2. Run: python3 src/tools/approve_builder_fee.py",
            "  3. Approve via the Hyperliquid web interface (Settings > Approvals)",
            " ",
            "To remove this message, approve the builder code.",
            "To disable builder codes entirely, remove 'hyperliquid' from broker_codes.hjson.",
        ])

    def _print_boxed_banner(self, lines: list):
        front_pad = " " * 4 + "##"
        back_pad = "##"
        max_len = max(len(line) for line in lines)
        print("\n\n")
        print(front_pad + "#" * (max_len + 2) + back_pad)
        for line in lines:
            print(front_pad + " " + line + " " * (max_len - len(line) + 1) + back_pad)
        print(front_pad + "#" * (max_len + 2) + back_pad)
        print("\n\n")

    async def _maybe_reenable_builder_codes(self):
        """Periodically re-enable builder codes for Path C users (every ~30 min)."""
        reenable_interval_ms = 1000 * 60 * 30
        disabled_ts = getattr(self, "_builder_disabled_ts", None)
        if disabled_ts is None:
            return
        if utc_ms() - disabled_ts < reenable_interval_ms:
            return

        # Check if user approved externally since last check
        if await self._check_builder_fee_approved():
            self._set_builder_approved(True)
            self._builder_pending_approval = False
            self._builder_disabled_ts = None
            logging.info(
                "[builder] Builder code active. Thank you for supporting Passivbot development!"
            )
            return

        # Not yet approved - re-enable to try again on next order
        self._set_builder_approved(True)
        self._builder_disabled_ts = None

    async def execute_to_exchange(self):
        # One-time builder code initialization
        if not hasattr(self, "_builder_initialized"):
            await self._init_builder_codes()
            self._builder_initialized = True

        res = await super().execute_to_exchange()

        # Periodic re-enable for Path C users
        if getattr(self, "_builder_pending_approval", False):
            await self._maybe_reenable_builder_codes()

        return res

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        isolated_count = 0
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                10.0 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_costs[symbol] = pbr.round_(self.min_costs[symbol] * 1.01, 0.01)
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.min_qtys[symbol] = (
                self.qty_steps[symbol]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]

            # For isolated-only markets (HIP-3), cap at 10x leverage
            if self._requires_isolated_margin(symbol):
                isolated_count += 1
                self.max_leverage[symbol] = min(
                    self.HIP3_MAX_LEVERAGE,
                    int(elm["info"]["maxLeverage"]) if "maxLeverage" in elm["info"] else self.HIP3_MAX_LEVERAGE,
                )
            else:
                self.max_leverage[symbol] = (
                    int(elm["info"]["maxLeverage"]) if "maxLeverage" in elm["info"] else 0
                )
        self.n_decimal_places = 6
        self.n_significant_figures = 5
        if isolated_count:
            logging.info(f"Detected {isolated_count} isolated-margin-only symbols (HIP-3/stock perps)")

    def _requires_isolated_margin(self, symbol: str) -> bool:
        """Check if a symbol requires isolated margin mode.

        On Hyperliquid, this includes:
        1. Symbols with xyz: prefix (HIP-3 stock perps from TradeXYZ)
        2. Markets with onlyIsolated=True flag

        Args:
            symbol: CCXT-style symbol (e.g., "xyz:TSLA/USDC:USDC")

        Returns:
            True if this symbol requires isolated margin mode
        """
        # Check for xyz: prefix in symbol or base
        if symbol.startswith(self.HIP3_PREFIX):
            return True
        base = symbol.split("/")[0] if "/" in symbol else symbol
        if base.startswith(self.HIP3_PREFIX):
            return True

        # Fall back to base class check (onlyIsolated flag, etc.)
        return super()._requires_isolated_margin(symbol)

    async def watch_orders(self):
        res = None
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = self.determine_pos_side(res[i])
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                self._health_ws_reconnects += 1
                logging.warning(
                    "[ws] %s: connection lost (reconnect #%d), retrying in 1s: %s",
                    self.exchange,
                    self._health_ws_reconnects,
                    type(e).__name__,
                )
                logging.debug("[ws] %s: full exception: %s", self.exchange, e)
                logging.debug("".join(traceback.format_exc()))
                await asyncio.sleep(1)
                logging.info("[ws] %s: reconnecting...", self.exchange)

    def determine_pos_side(self, order):
        # hyperliquid is not hedge mode
        if order["symbol"] in self.positions:
            if self.positions[order["symbol"]]["long"]["size"] != 0.0:
                return "long"
            elif self.positions[order["symbol"]]["short"]["size"] != 0.0:
                return "short"
            else:
                return "long" if order["side"] == "buy" else "short"
        else:
            if "reduceOnly" in order:
                if order["side"] == "buy":
                    return "short" if order["reduceOnly"] else "long"
                if order["side"] == "sell":
                    return "long" if order["reduceOnly"] else "short"
            return "long" if order["side"] == "buy" else "short"

    def _get_position_side_for_order(self, order: dict) -> str:
        """Hook: Derive position_side from order data for Hyperliquid (one-way mode)."""
        return self.determine_pos_side(order)

    async def fetch_open_orders(self, symbol: str = None):
        fetched = await self.cca.fetch_open_orders()
        for elm in fetched:
            elm["position_side"] = self.determine_pos_side(elm)
            elm["qty"] = elm["amount"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def _fetch_positions_and_balance(self):
        info = await self.cca.fetch_balance()
        positions = [
            {
                "symbol": self.coin_to_symbol(x["position"]["coin"]),
                "position_side": ("long" if (size := float(x["position"]["szi"])) > 0.0 else "short"),
                "size": size,
                "price": float(x["position"]["entryPx"]),
            }
            for x in info["info"]["assetPositions"]
        ]
        balance = float(info["info"]["marginSummary"]["accountValue"]) - sum(
            [float(x["position"]["unrealizedPnl"]) for x in info["info"]["assetPositions"]]
        )
        return positions, balance

    async def fetch_positions(self):
        positions, balance = await self._fetch_positions_and_balance()
        self._last_hl_positions_balance = (positions, balance)
        self._hl_positions_balance_applied = False
        return positions

    async def fetch_balance(self):
        cached = getattr(self, "_last_hl_positions_balance", None)
        applied = getattr(self, "_hl_positions_balance_applied", False)
        if cached and not applied:
            positions, balance = cached
            self._hl_positions_balance_applied = True
            return balance
        positions, balance = await self._fetch_positions_and_balance()
        self._last_hl_positions_balance = (positions, balance)
        self._hl_positions_balance_applied = True
        return balance

    async def fetch_tickers(self):
        fetched = await self.cca.fetch(
            "https://api.hyperliquid.xyz/info",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"type": "allMids"}),
        )
        return {
            self.coin_to_symbol(coin): {
                "bid": float(fetched[coin]),
                "ask": float(fetched[coin]),
                "last": float(fetched[coin]),
            }
            for coin in fetched
        }

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        # fetches latest ohlcvs
        str2int = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60 * 4}
        n_candles = 480
        since = int(utc_ms() - 1000 * 60 * str2int[timeframe] * n_candles)
        return await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, since=since)

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 5000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
            since=int(self.get_exchange_time() - 1000 * 60 * n_candles_limit * 0.95),
        )
        return result

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit=None,
    ):
        # hyperliquid fetches from past to future
        if limit is None:
            limit = 2000
        if start_time is None:
            # hyperliquid returns latest trades if no time frame is passed
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        prev_hash = ""
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, limit=limit)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            if end_time and fetched[-1]["timestamp"] >= end_time:
                break
            new_hash = calc_hash(fetched)
            if prev_hash == new_hash:
                logging.debug(f"pnls hash unchanged: {prev_hash}")
                break
            prev_hash = new_hash
            logging.info(
                f"debug fetching pnls {ts_to_date(fetched[-1]['timestamp'])} len {len(fetched)}"
            )
            start_time = fetched[-1]["timestamp"] - 1000
            limit = 2000
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for Hyperliquid (draft placeholder)."""
        events = []
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time, limit=limit)
        for fill in fills:
            events.append(
                {
                    "id": fill.get("id"),
                    "timestamp": fill.get("timestamp"),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "position_side": fill.get("position_side"),
                    "qty": fill.get("amount"),
                    "price": fill.get("price"),
                    "pnl": fill.get("pnl"),
                    "fee": fill.get("fee"),
                    "info": fill.get("info"),
                }
            )
        return events

    async def fetch_pnl(
        self,
        start_time: int = None,
        limit=None,
    ):
        if start_time is None:
            fetched = await self.cca.fetch_my_trades(limit=limit)
        else:
            fetched = await self.cca.fetch_my_trades(since=max(1, int(start_time)), limit=limit)
        for elm in fetched:
            elm["pnl"] = float(elm["info"]["closedPnl"])
            elm["position_side"] = "long" if "long" in elm["info"]["dir"].lower() else "short"
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def execute_cancellation(self, order: dict) -> dict:
        """Hyperliquid: Cancel order with vault support."""
        params = (
            {"vaultAddress": self.user_info["wallet_address"]} if self.user_info["is_vault"] else {}
        )
        def _is_already_gone(payload) -> bool:
            try:
                text = str(payload)
            except Exception:
                text = ""
            text_l = text.lower()
            if "order was never placed" in text_l or "already canceled" in text_l or "already cancelled" in text_l:
                return True
            return False

        try:
            res = await self.cca.cancel_order(order["id"], symbol=order["symbol"], params=params)
            # Sometimes hyperliquid returns an "ok" wrapper with an embedded error; treat as non-fatal.
            if _is_already_gone(res):
                logging.info("Order already canceled/filled on exchange; treating as success.")
                return {"status": "success"}
            return res
        except Exception as e:
            if _is_already_gone(e):
                logging.info("Order already canceled/filled on exchange; treating as success.")
                return {"status": "success"}
            raise

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return "status" in executed and executed["status"] == "success"
        except (TypeError, KeyError):
            return False

    def _build_order_params(self, order: dict) -> dict:
        params = {
            "reduceOnly": order["reduce_only"],
            "timeInForce": (
                "Alo" if require_live_value(self.config, "time_in_force") == "post_only" else "Gtc"
            ),
            "clientOrderId": order["custom_id"],
        }
        if self.user_info["is_vault"]:
            params["vaultAddress"] = self.user_info["wallet_address"]
        return params

    async def execute_order(self, order: dict) -> dict:
        """Hyperliquid: Execute order with builder-fee and min_cost error handling."""
        try:
            return await super().execute_order(order)
        except Exception as e:
            # Path C: catch builder-fee-not-approved errors
            if getattr(self, "_builder_pending_approval", False) and self._is_builder_fee_error(e):
                logging.warning(
                    f"[builder] Order failed: builder fee not approved. "
                    f"Temporarily disabling builder codes. Symbol: {order['symbol']}"
                )
                self._print_builder_nag_banner()
                self._set_builder_approved(False)
                self._builder_disabled_ts = utc_ms()
                return {}
            # Try to recover from Hyperliquid's "$10 minimum" errors by adjusting min_cost
            try:
                if self.adjust_min_cost_on_error(e, order):
                    logging.info(f"Adjusted min_cost for order, will retry: {order['symbol']}")
                    return {}
            except Exception as e0:
                logging.error(f"error with adjust_min_cost_on_error {e0}")
            # Could not recover - re-raise to trigger restart_bot_on_too_many_errors
            raise

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(orders, "execute_order")

    def did_create_order(self, executed) -> bool:
        did_create = super().did_create_order(executed)
        try:
            return did_create and (
                "info" in executed and ("filled" in executed["info"] or "resting" in executed["info"])
            )
        except (TypeError, KeyError):
            return False

    def adjust_min_cost_on_error(self, error, order=None):
        any_adjusted = False
        successful_orders = []
        str_e = str(error)
        brace_idx = str_e.find("{")
        if brace_idx == -1:
            return False
        try:
            error_json = json.loads(str_e[brace_idx:])
        except json.JSONDecodeError:
            return False
        if (
            "response" in error_json
            and "data" in error_json["response"]
            and "statuses" in error_json["response"]["data"]
        ):
            for elm in error_json["response"]["data"]["statuses"]:
                if "error" in elm:
                    if "Order must have minimum value of $10" in elm["error"]:
                        asset_id = int(elm["error"][elm["error"].find("asset=") + 6 :])
                        for symbol in self.markets_dict:
                            if (
                                "baseId" in self.markets_dict[symbol]["info"]
                                and self.markets_dict[symbol]["info"]["baseId"] == asset_id
                            ):
                                break
                        else:
                            raise Exception(f"No symbol match for asset_id={asset_id}")
                        new_min_cost = pbr.round_(self.min_costs[symbol] * 1.1, 0.1)
                        logging.info(
                            f"caught {elm['error']} {symbol}. Upping min_cost from {self.min_costs[symbol]} to {new_min_cost}. Order: {order}"
                        )
                        self.min_costs[symbol] = new_min_cost
                        any_adjusted = True
        return any_adjusted

    def symbol_is_eligible(self, symbol):
        """Check if a symbol is eligible for trading.

        HIP-3 stock perps (onlyIsolated=True) are eligible - they use isolated margin
        automatically and have leverage capped at 10x.
        """
        try:
            market_info = self.markets_dict[symbol]["info"]

            # Zero open interest means market is inactive
            if float(market_info.get("openInterest", 0)) == 0.0:
                return False
        except Exception as e:
            logging.error(f"error with symbol_is_eligible {e} {symbol}")
            return False
        return True

    async def update_exchange_config_by_symbols(self, symbols):
        """Set leverage and margin mode for Hyperliquid symbols.

        Uses base class methods for isolated margin detection and leverage calculation.
        Adds Hyperliquid-specific vault address handling.
        """
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                # Use base class method for leverage calculation (handles isolated margin)
                leverage = self._calc_leverage_for_symbol(symbol)
                margin_mode = self._get_margin_mode_for_symbol(symbol)

                params = {"leverage": leverage}
                if self.user_info["is_vault"]:
                    params["vaultAddress"] = self.user_info["wallet_address"]

                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(margin_mode, symbol=symbol, params=params)
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting margin mode and leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"margin={format_exchange_config_response(res)} ({margin_mode})"
            except Exception as e:
                if '"code":"59107"' in str(e):
                    to_print += f"margin=ok (unchanged, {margin_mode})"
                else:
                    logging.error(f"{symbol} error setting {margin_mode} mode {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        pass

    async def calc_ideal_orders(self):
        # hyperliquid needs custom price rounding
        ideal_orders = await super().calc_ideal_orders()
        for sym in ideal_orders:
            for i in range(len(ideal_orders[sym])):
                if ideal_orders[sym][i]["side"] == "sell":
                    ideal_orders[sym][i]["price"] = round_dynamic_up(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                elif ideal_orders[sym][i]["side"] == "buy":
                    ideal_orders[sym][i]["price"] = round_dynamic_dn(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                else:
                    ideal_orders[sym][i]["price"] = round_dynamic(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                ideal_orders[sym][i]["price"] = round_(
                    ideal_orders[sym][i]["price"], self.price_steps[sym]
                )
        return ideal_orders

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (formatted)[: self.custom_id_max_length]
