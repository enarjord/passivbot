from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging
import passivbot_rust as pbr

import asyncio
from utils import symbol_to_coin, ts_to_date, utc_ms
from config.access import require_live_value

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
        Startup must know whether OKX is in dual-side or net mode before building orders.
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
                raise RuntimeError(
                    "OKX account is in net (one-way) mode; Passivbot requires "
                    "dual-side/hedge mode for live order side safety"
                )
            elif pos_mode == "long_short_mode":
                self.okx_dual_side = True
            # If unknown, keep default True and let later failures flip it off.
            self.okx_pm_account = acct_lv == "pm"
            if self.okx_pm_account:
                logging.info(
                    "OKX account detected as Portfolio Margin (PM); mode/leverage changes may be restricted."
                )
        except Exception as e:
            if (
                isinstance(e, RuntimeError)
                and "Passivbot requires dual-side/hedge mode" in str(e)
            ):
                raise
            raise RuntimeError(
                "Unable to detect OKX account configuration before live order setup"
            ) from e

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    def _get_position_side_for_order(self, order: dict) -> str:
        """OKX provides posSide in info."""
        return order.get("info", {}).get("posSide", "long").lower()

    def _normalize_positions(self, fetched: list) -> list:
        """OKX: Preserve live positions across both cross and isolated margin modes."""
        positions = []
        for elm in fetched:
            contracts = float(elm.get("contracts", 0))
            if contracts != 0:
                normalized = {
                    "symbol": elm["symbol"],
                    "position_side": elm.get("side", "long").lower(),
                    "size": contracts,
                    "price": float(elm.get("entryPrice", 0)),
                }
                margin_mode = self._extract_live_margin_mode(elm)
                if margin_mode is not None:
                    normalized["margin_mode"] = margin_mode
                    self._record_live_margin_mode(normalized["symbol"], margin_mode)
                positions.append(normalized)
        return positions

    def _get_pnl_from_trade(self, trade: dict) -> float:
        """OKX uses fillPnl in info."""
        return float(trade.get("info", {}).get("fillPnl", 0))

    def _get_position_side_from_trade(self, trade: dict) -> str:
        """OKX provides posSide in info."""
        return trade.get("info", {}).get("posSide", "long").lower()

    # ═══════════════════ OKX-SPECIFIC METHODS ═══════════════════

    async def fetch_balance(self) -> float:
        """OKX: return wallet balance in quote terms, including multi-asset collateral."""
        fetched = await self._do_fetch_balance()
        return self._get_balance(fetched)

    def _get_balance(self, fetched: dict) -> float:
        """OKX account equity includes UPL; Passivbot raw balance should not."""
        info = fetched["info"]
        data = info["data"]
        if not data:
            raise KeyError("okx: fetch_balance response missing info.data[0]")
        account = data[0]
        details = account.get("details", [])
        if "totalEq" in account:
            upl_sum = sum(float(detail.get("upl") or 0.0) for detail in details)
            return float(account["totalEq"]) - upl_sum

        balance = 0.0
        for detail in details:
            collateral_enabled = detail.get("collateralEnabled")
            if str(collateral_enabled).lower() not in {"true", "1"} and collateral_enabled is not True:
                continue
            ccy = detail["ccy"]
            if ccy == self.quote:
                balance += float(detail["cashBal"])
            elif "eqUsd" in detail:
                balance += float(detail["eqUsd"]) - float(detail.get("upl") or 0.0)
            else:
                raise KeyError(f"okx: collateral detail for {ccy} missing eqUsd")
        if balance == 0.0:
            total = fetched.get("total")
            if not isinstance(total, dict) or self.quote not in total:
                raise KeyError(f"okx: fetch_balance response missing total[{self.quote!r}]")
            return float(total[self.quote])
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
                return self._ambiguous_cancel_success_result(order)
            raise

    def _build_order_params(self, order: dict) -> dict:
        if not bool(getattr(self, "okx_dual_side", True)):
            raise RuntimeError(
                "OKX order construction requires dual-side/hedge mode; "
                "refusing to build net-mode order params"
            )
        margin_mode = self._get_margin_mode_for_symbol(order["symbol"])
        params = {
            "postOnly": require_live_value(self.config, "time_in_force") == "post_only",
            "reduceOnly": order["reduce_only"],
            "hedged": True,
            "tag": self.broker_code,
            "clOrdId": order["custom_id"],
            "marginMode": margin_mode,
        }
        params["positionSide"] = order["position_side"]
        return params

    async def update_exchange_config_by_symbols(self, symbols: [str]):
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            margin_mode = self._get_margin_mode_for_symbol(symbol)
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
            try:
                leverage = self._calc_leverage_for_symbol(symbol)
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        margin_mode,
                        symbol=symbol,
                        params={"lever": leverage},
                    )
                )
            except Exception as e:
                logging.error(f"{log_symbol}: error setting {margin_mode} mode and leverage {e}")
        for symbol in symbols:
            log_symbol = symbol_to_coin(symbol, verbose=False) or symbol
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
                        f"{log_symbol}: unable to adjust margin mode/leverage (possibly PM or open positions)"
                    )
                    continue
                else:
                    logging.error(f"{log_symbol} error setting cross mode {e}")
            if to_print:
                logging.info(f"{log_symbol}: {to_print}")

    async def update_exchange_config(self):
        # Detect current account mode; adjust expectations before attempting changes.
        await self._detect_account_config()
        if not self.okx_dual_side:
            raise RuntimeError(
                "OKX account is not in dual-side/hedge mode; Passivbot refuses "
                "to continue in net mode"
            )
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
                raise RuntimeError(
                    "OKX rejected hedge/dual-side switch (51039/51000); "
                    "Passivbot refuses to continue in net mode"
                ) from e
            else:
                raise

    async def calc_ideal_orders(self):
        # okx has max 100 open orders. Drop orders whose pprice diff is greatest.
        ideal_orders = await super().calc_ideal_orders()
        market_prices = await self._get_live_last_prices(
            ideal_orders.keys(),
            max_age_ms=10_000,
            context="okx_order_cap_sort",
            allow_completed_candle_fallback=True,
        )
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append(
                    (
                        calc_order_price_diff(
                            x["side"],
                            x["price"],
                            market_prices[s],
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
