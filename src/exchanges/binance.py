from exchanges.ccxt_bot import CCXTBot, format_exchange_config_response
from passivbot import logging

import asyncio
import random
from utils import ts_to_date, utc_ms
from pure_funcs import flatten
from procedures import load_broker_code
from config_utils import require_live_value


class BinanceBot(CCXTBot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36

    def create_ccxt_sessions(self):
        """Binance: Add broker codes after standard setup."""
        self.broker_code_spot = load_broker_code("binance_spot")
        super().create_ccxt_sessions()
        for client in [self.cca, self.ccp]:
            if client is None:
                continue
            if self.broker_code:
                for key in ["future", "delivery", "swap", "option"]:
                    client.options["broker"][key] = "x-" + self.broker_code
            if self.broker_code_spot:
                for key in ["spot", "margin"]:
                    client.options["broker"][key] = "x-" + self.broker_code_spot

    async def print_new_user_suggestion(self):
        between_print_wait_ms = 1000 * 60 * 60 * 4
        if hasattr(self, "previous_user_suggestion_print_ts"):
            if utc_ms() - self.previous_user_suggestion_print_ts < between_print_wait_ms:
                return
        self.previous_user_suggestion_print_ts = utc_ms()

        try:
            res = await self.cca.fapiprivate_get_apireferral_ifnewuser(
                params={"brokerid": self.broker_code}
            )
        except Exception as e:
            # This endpoint may not be available on all accounts - expected failure
            logging.debug(f"fapiprivate_get_apireferral_ifnewuser not available: {e}")
            return
        if res["ifNewUser"] and res["rebateWorking"]:
            return
        import json

        lines = [
            "To support continued Passivbot development, please use a Binance account which",
            "1) was created after 2024-09-21 and",
            "2) either:",
            "  a) was created without a referral link, or",
            '  b) was created with referral ID: "TII4B07C".',
            " ",
            "Passivbot receives commissions from trades only for accounts meeting these criteria.",
            " ",
            json.dumps(res),
        ]
        front_pad = " " * 8 + "##"
        back_pad = "##"
        max_len = max([len(line) for line in lines])
        print("\n\n")
        print(front_pad + "#" * (max_len + 2) + back_pad)
        for line in lines:
            print(front_pad + " " + line + " " * (max_len - len(line) + 1) + back_pad)
        print(front_pad + "#" * (max_len + 2) + back_pad)
        print("\n\n")

    async def execute_to_exchange(self):
        res = await super().execute_to_exchange()
        await self.print_new_user_suggestion()
        return res

    # ═══════════════════ HOOK OVERRIDES ═══════════════════

    def _get_position_side_for_order(self, order: dict) -> str:
        """Binance provides ps (positionSide) in info."""
        return order.get("info", {}).get("ps", "long").lower()

    async def _do_fetch_positions(self) -> list:
        """Binance: Use fapiprivatev3_get_positionrisk endpoint."""
        return await self.cca.fapiprivatev3_get_positionrisk()

    def _normalize_positions(self, fetched: list) -> list:
        """Binance: Parse positionrisk response format."""
        positions = []
        for elm in fetched:
            if float(elm["positionAmt"]) != 0.0:
                positions.append(
                    {
                        "symbol": self.get_symbol_id_inv(elm["symbol"]),
                        "position_side": elm["positionSide"].lower(),
                        "size": float(elm["positionAmt"]),
                        "price": float(elm["entryPrice"]),
                    }
                )
        return positions

    def _get_balance(self, fetched: dict) -> float:
        """Binance uses totalCrossWalletBalance in info."""
        return float(fetched["info"]["totalCrossWalletBalance"])

    # ═══════════════════ BINANCE-SPECIFIC METHODS ═══════════════════

    async def fetch_open_orders(self, symbol: str = None, all=False) -> list:
        """Binance: Parallel fetch per-symbol to avoid expensive all-symbols query."""
        if all:
            self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
            logging.info("fetching all open orders for binance")
            fetched = await self.cca.fetch_open_orders()
            self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = True
        else:
            symbols_ = set()
            symbols_.update([s for s in self.open_orders if self.open_orders[s]])
            symbols_.update([s for s in self.get_symbols_with_pos()])
            if hasattr(self, "active_symbols") and self.active_symbols:
                symbols_.update(list(self.active_symbols))
            results = await asyncio.gather(
                *[self.cca.fetch_open_orders(symbol=s) for s in sorted(symbols_)]
            )
            fetched = [x for sublist in results for x in sublist]

        open_orders = {}
        for elm in fetched:
            elm["position_side"] = elm["info"]["positionSide"].lower()
            elm["qty"] = elm["amount"]
            open_orders[elm["id"]] = elm
        return sorted(open_orders.values(), key=lambda x: x["timestamp"])

    async def fetch_tickers(self) -> dict:
        """Binance: Use bookticker endpoint for efficiency."""
        fetched = await self.cca.fapipublic_get_ticker_bookticker()
        tickers = {}
        for elm in fetched:
            symbol = self.get_symbol_id_inv(elm["symbol"])
            if symbol in self.markets_dict:
                bid = float(elm["bidPrice"])
                ask = float(elm["askPrice"])
                tickers[symbol] = {
                    "bid": bid,
                    "ask": ask,
                    "last": random.choice([bid, ask]),
                }
        return tickers

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        pnls = await self.fetch_pnls_sub(start_time, end_time, limit)
        symbols = sorted(set(self.positions) | set([x["symbol"] for x in pnls]))
        tasks = {}
        for symbol in symbols:
            tasks[symbol] = asyncio.create_task(
                self.fetch_fills_sub(symbol, start_time, end_time, limit)
            )
        fills = {}
        for symbol in tasks:
            fills[symbol] = await tasks[symbol]
        fills = flatten(fills.values())
        if start_time:
            pnls = [x for x in pnls if x["timestamp"] >= start_time]
            fills = [x for x in fills if x["timestamp"] >= start_time]
        unified = {x["id"]: x for x in pnls}
        for x in fills:
            if x["id"] in unified:
                unified[x["id"]].update(x)
            else:
                unified[x["id"]] = x
        result = []
        for x in sorted(unified.values(), key=lambda x: x["timestamp"]):
            if "position_side" not in x:
                logging.info(f"debug: pnl without corresponding fill {x}")
                x["position_side"] = "unknown"
            result.append(x)
        return result

    async def fetch_pnls_sub(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        # binance needs symbol specified for fetch fills
        # but can fetch pnls for all symbols
        # fetch fills for all symbols with pos
        # fetch pnls for all symbols
        # binance returns at most 7 days worth of pnls per fetch unless both start_time and end_time are given
        if limit is None:
            limit = 1000
        else:
            limit = min(limit, 1000)
        if end_time is None:
            if start_time is None:
                return await self.fetch_pnl(limit=limit)
            end_time = self.get_exchange_time() + 1000 * 60 * 60
        all_fetched = {}
        week = 1000 * 60 * 60 * 24 * 7
        while True:
            fetched = await self.fetch_pnl(start_time, end_time, limit)
            if fetched == []:
                break
            if fetched[0]["tradeId"] in all_fetched and fetched[-1]["tradeId"] in all_fetched:
                break
            for elm in fetched:
                all_fetched[elm["tradeId"]] = elm
            if len(fetched) < limit:
                if start_time:
                    if end_time:
                        if end_time - start_time < week:
                            break
                    else:
                        if self.get_exchange_time() - start_time < week:
                            break
            logging.info(
                f"fetched {len(fetched)} pnls from {ts_to_date(fetched[0]['timestamp'])[:19]} until {ts_to_date(fetched[-1]['timestamp'])[:19]}"
            )
            start_time = fetched[-1]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def gather_fill_events(self, start_time=None, end_time=None, limit=None):
        """Return canonical fill events for Binance."""
        events = []
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time, limit=limit)
        for fill in fills:
            events.append(
                {
                    "id": fill.get("id") or fill.get("tradeId"),
                    "timestamp": fill.get("timestamp"),
                    "symbol": fill.get("symbol"),
                    "side": fill.get("side"),
                    "position_side": fill.get("position_side", fill.get("pside")),
                    "qty": fill.get("qty") or fill.get("amount"),
                    "price": fill.get("price"),
                    "pnl": fill.get("pnl"),
                    "fee": fill.get("fee"),
                    "info": fill.get("info"),
                }
            )
        return events

    async def fetch_fills_sub(self, symbol, start_time=None, end_time=None, limit=None):
        if symbol not in self.markets_dict:
            return []
        # limit is max 1000
        # fetches at most 7 days worth
        max_limit = 1000
        limit = min(max_limit, limit) if limit else max_limit
        if start_time is None and end_time is None:
            fills = await self.cca.fetch_my_trades(symbol, limit=limit)
            all_fills = {x["id"]: x for x in fills}
        elif start_time is None:
            fills = await self.cca.fetch_my_trades(
                symbol, limit=limit, params={"endTime": int(end_time)}
            )
            all_fills = {x["id"]: x for x in fills}
        else:
            if end_time is None:
                end_time = self.get_exchange_time() + 1000 * 60 * 60
            all_fills = {}
            week = 1000 * 60 * 60 * 24 * 7.0
            start_time_sub = start_time
            while True:
                param_start_time = int(min(start_time_sub, self.get_exchange_time() - 1000 * 60))
                param_end_time = max(
                    param_start_time, int(min(end_time, start_time_sub + week * 0.999))
                )
                fills = await self.cca.fetch_my_trades(
                    symbol,
                    limit=limit,
                    params={
                        "startTime": param_start_time,
                        "endTime": param_end_time,
                    },
                )
                if not fills:
                    if end_time - start_time_sub < week * 0.9:
                        self.debug_print("debug fetch_fills_sub a", symbol)
                        break
                    else:
                        logging.info(
                            f"fetched 0 fills for {symbol} between {ts_to_date(start_time_sub)[:19]} and {ts_to_date(end_time)[:19]}"
                        )
                        start_time_sub += week
                        continue
                if fills[0]["id"] in all_fills and fills[-1]["id"] in all_fills:
                    if end_time - start_time_sub < week * 0.9:
                        self.debug_print("debug fetch_fills_sub b", symbol)
                        break
                    else:
                        logging.info(
                            f"fetched 0 new fills for {symbol} between {ts_to_date(start_time_sub)[:19]} and {ts_to_date(end_time)[:19]}"
                        )
                        start_time_sub += week
                        continue
                else:
                    for x in fills:
                        all_fills[x["id"]] = x
                if end_time - start_time_sub < week * 0.9 and len(fills) < limit:
                    self.debug_print("debug fetch_fills_sub c", symbol)
                    break
                start_time_sub = fills[-1]["timestamp"]
                logging.info(
                    f"fetched {len(fills)} fill{'s' if len(fills) > 1 else ''} for {symbol} {ts_to_date(fills[0]['timestamp'])[:19]}"
                )
        all_fills = sorted(all_fills.values(), key=lambda x: x["timestamp"])
        for i in range(len(all_fills)):
            all_fills[i]["pnl"] = float(all_fills[i]["info"]["realizedPnl"])
            all_fills[i]["position_side"] = all_fills[i]["info"]["positionSide"].lower()
        return all_fills

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        # will fetch from start_time until end_time, earliest first
        # if start_time is None and end_time is None, will only fetch for last 7 days
        # if end_time is None, will fetch for more than 7 days
        # if start_time is None, will only fetch for last 7 days
        max_limit = 1000
        if limit is None:
            limit = max_limit
        params = {"incomeType": "REALIZED_PNL", "limit": min(max_limit, limit)}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        fetched = await self.cca.fapiprivate_get_income(params=params)
        for i in range(len(fetched)):
            fetched[i]["symbol"] = self.get_symbol_id_inv(fetched[i]["symbol"])
            fetched[i]["pnl"] = float(fetched[i]["income"])
            fetched[i]["timestamp"] = float(fetched[i]["time"])
            fetched[i]["id"] = fetched[i]["tradeId"]
        return sorted(fetched, key=lambda x: x["timestamp"])

    def _build_order_params(self, order: dict) -> dict:
        order_type = order.get("type", "limit")
        params = {
            "positionSide": order["position_side"].upper(),
            "newClientOrderId": order["custom_id"],
        }
        if order_type == "limit":
            tif = require_live_value(self.config, "time_in_force")
            params["timeInForce"] = "GTX" if tif == "post_only" else "GTC"
        return params

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            coros_to_call_margin_mode[symbol] = asyncio.create_task(
                self.cca.set_margin_mode("cross", symbol=symbol)
            )
            coros_to_call_lev[symbol] = asyncio.create_task(
                self.cca.set_leverage(
                    int(self.config_get(["live", "leverage"], symbol=symbol)), symbol=symbol
                )
            )
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f"leverage={format_exchange_config_response(res)} "
            except Exception as e:
                logging.error(f"{symbol}: error setting leverage {e}")
            try:
                res_margin = await coros_to_call_margin_mode[symbol]
                to_print += f"margin={format_exchange_config_response(res_margin)}"
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        try:
            res = await self.cca.set_position_mode(True)
            logging.debug("[config] set hedge mode response: %s", res)
        except Exception as e:
            if '"code":-4059' in str(e):
                logging.debug("[config] hedge mode unchanged: %s", e)
            else:
                logging.error("[config] error setting hedge mode: %s", e)

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ticker("BTC/USDT:USDT")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return ("x-" + self.broker_code + formatted)[: self.custom_id_max_length]
