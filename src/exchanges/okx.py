from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
import passivbot_rust as pbr

import asyncio
from utils import ts_to_date, utc_ms
from config_utils import require_live_value

calc_order_price_diff = pbr.calc_order_price_diff


class OKXBot(CCXTBot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.order_side_map = {
            "buy": {"long": "open_long", "short": "close_short"},
            "sell": {"long": "close_long", "short": "open_short"},
        }
        self.custom_id_max_length = 32
        # Track whether dual-side/hedge mode is available; default to True.
        self.okx_dual_side = True
        self.okx_pm_account = False

    async def _detect_account_config(self):
        """
        Inspect account configuration to detect portfolio margin (PM) and position mode.
        Falls back silently if the endpoint is unavailable.
        """
        try:
            cfg = await self.cca.private_get_account_config()
            data = cfg.get("data", [{}])
            data0 = data[0] if data else {}
            pos_mode = str(data0.get("posMode", "")).lower()  # "long_short_mode" or "net_mode"
            acct_lv = str(data0.get("acctLv", "")).lower()  # "pm" for portfolio margin accounts
            if pos_mode == "net_mode":
                self.okx_dual_side = False
                self.hedge_mode = False
            elif pos_mode == "long_short_mode":
                self.okx_dual_side = True
            # If unknown, keep default True and let later failures flip it off.
            self.okx_pm_account = acct_lv == "pm"
            if self.okx_pm_account:
                logging.info(
                    "OKX account detected as Portfolio Margin (PM); mode/leverage changes may be restricted."
                )
            if not self.okx_dual_side:
                logging.info("OKX account is in net (one-way) mode; running without posSide/hedge.")
        except Exception as e:
            logging.warning(f"Unable to detect OKX account configuration: {e}")

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    def _get_position_side_for_order(self, order: dict) -> str:
        """OKX provides posSide in info."""
        return order.get("info", {}).get("posSide", "long").lower()

    def _normalize_positions(self, fetched: list) -> list:
        """OKX: Filter to cross margin positions only."""
        positions = []
        for elm in fetched:
            if elm.get("marginMode") != "cross":
                continue
            contracts = float(elm.get("contracts", 0))
            if contracts != 0:
                positions.append(
                    {
                        "symbol": elm["symbol"],
                        "position_side": elm.get("side", "long").lower(),
                        "size": contracts,
                        "price": float(elm.get("entryPrice", 0)),
                    }
                )
        return positions

    def _get_pnl_from_trade(self, trade: dict) -> float:
        """OKX uses fillPnl in info."""
        return float(trade.get("info", {}).get("fillPnl", 0))

    def _get_position_side_from_trade(self, trade: dict) -> str:
        """OKX provides posSide in info."""
        return trade.get("info", {}).get("posSide", "long").lower()

    # ═══════════════════ OKX-SPECIFIC METHODS ═══════════════════

    async def fetch_balance(self) -> float:
        """OKX: Complex multi-asset mode balance calculation.

        OKX has a unique balance structure that requires summing collateral
        across multiple assets, converting each to quote currency.
        """
        fetched_balance = await self.cca.fetch_balance()
        balance = 0.0

        is_multi_asset_mode = True
        if len(fetched_balance["info"]["data"]) == 1:
            if len(fetched_balance["info"]["data"][0]["details"]) == 1:
                if fetched_balance["info"]["data"][0]["details"][0]["ccy"] == self.quote:
                    if not fetched_balance["info"]["data"][0]["details"][0]["collateralEnabled"]:
                        is_multi_asset_mode = False

        if is_multi_asset_mode:
            for elm in fetched_balance["info"]["data"]:
                for elm2 in elm["details"]:
                    if elm2["collateralEnabled"]:
                        balance += float(elm2["cashBal"]) * (
                            (
                                await self.cm.get_current_close(
                                    self.coin_to_symbol(elm2["ccy"]), max_age_ms=10_000
                                )
                            )
                            if elm2["ccy"] != self.quote
                            else 1.0
                        )
        else:
            balance = float(fetched_balance["info"]["data"][0]["details"][0]["cashBal"])
        return balance

    async def fetch_pnls(self, start_time: int = None, end_time: int = None, limit=None):
        if limit is None:
            limit = 100
        if start_time is None and end_time is None:
            return await self.fetch_pnl()
        all_fetched = {}
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, end_time=end_time)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            logging.debug(f"fetching income {ts_to_date(fetched[-1]['timestamp'])}")
            end_time = fetched[0]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for OKX."""
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
        end_time: int = None,
    ):
        """Fetch trades from OKX. If there are more than 100 fills, fetches latest."""
        if end_time is None:
            end_time = utc_ms() + 1000 * 60 * 60 * 24
        if start_time is None:
            start_time = end_time - 1000 * 60 * 60 * 24 * 7
        fetched = await self.cca.fetch_my_trades(
            since=int(start_time), params={"until": int(end_time)}
        )
        for i in range(len(fetched)):
            fetched[i]["pnl"] = float(fetched[i]["info"]["fillPnl"])
            fetched[i]["position_side"] = fetched[i]["info"]["posSide"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    async def execute_cancellation(self, order: dict) -> dict:
        """OKX: Cancel order with special handling for 51400 (already cancelled/filled)."""
        try:
            return await self.cca.cancel_order(order["id"], symbol=order["symbol"])
        except Exception as e:
            # 51400 = order already cancelled or filled - not an error
            if '"sCode":"51400"' in str(e):
                logging.info(f"Order already cancelled/filled: {e}")
                return {}
            raise

    def _build_order_params(self, order: dict) -> dict:
        params = {
            "postOnly": require_live_value(self.config, "time_in_force") == "post_only",
            "reduceOnly": order["reduce_only"],
            "hedged": True,
            "tag": self.broker_code,
            "clOrdId": order["custom_id"],
            "marginMode": "cross",
        }
        # Only send positionSide when dual-side mode is confirmed.
        if self.okx_dual_side:
            params["positionSide"] = order["position_side"]
        return params

    async def update_exchange_config_by_symbols(self, symbols: [str]):
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        "cross",
                        symbol=symbol,
                        params={"lever": int(self.config_get(["live", "leverage"], symbol=symbol))},
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode and leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"margin={format_exchange_config_response(res)}"
            except Exception as e:
                err_str = str(e)
                if '"code":"59107"' in err_str:
                    to_print += f"margin=ok (unchanged)"
                elif '"code":"51039"' in err_str:
                    logging.warning(
                        f"{symbol}: unable to adjust margin mode/leverage (possibly PM or open positions)"
                    )
                    continue
                else:
                    logging.error(f"{symbol} error setting cross mode {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        # Detect current account mode; adjust expectations before attempting changes.
        await self._detect_account_config()
        if not self.okx_dual_side:
            # One-way mode: skip attempting to set hedge mode; orders will omit posSide.
            return
        try:
            res = await self.cca.set_position_mode(True)
            logging.debug("[config] set hedge mode response: %s", res)
        except Exception as e:
            err_str = str(e)
            if '"code":"59000"' in err_str:
                logging.info("[config] hedge mode update skipped: %s", e)
            elif '"code":"51039"' in err_str or '"code":"51000"' in err_str:
                # Cannot switch to dual/hedge (often due to PM or open orders/positions).
                self.okx_dual_side = False
                self.hedge_mode = False
                logging.warning(
                    "[config] OKX rejected hedge/dual-side switch (51039/51000). Continuing in net mode without posSide."
                )
            else:
                logging.error("[config] error setting hedge mode: %s", e)

    async def calc_ideal_orders(self):
        # okx has max 100 open orders. Drop orders whose pprice diff is greatest.
        ideal_orders = await super().calc_ideal_orders()
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append(
                    (
                        calc_order_price_diff(
                            x["side"],
                            x["price"],
                            await self.cm.get_current_close(s, max_age_ms=10_000),
                        ),
                        {**x, **{"symbol": s}},
                    )
                )
        ideal_orders_tmp = [x[1] for x in sorted(ideal_orders_tmp, key=lambda x: x[0])][:100]
        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for x in ideal_orders_tmp:
            ideal_orders[x["symbol"]].append(x)
        return ideal_orders

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (self.broker_code + formatted)[: self.custom_id_max_length]
