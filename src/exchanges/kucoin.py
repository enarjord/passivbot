from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import asyncio
import traceback
import numpy as np
import passivbot_rust as pbr
from pure_funcs import (
    floatify,
    ts_to_date_utc,
    calc_hash,
    shorten_custom_id,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version
from collections import defaultdict

assert_correct_ccxt_version(ccxt=ccxt_async)


class KucoinBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36  # adjust if needed
        self.quote = "USDT"
        self.hedge_mode = False

    def create_ccxt_sessions(self):
        self.ccp = ccxt_pro.kucoinfutures(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
        self.cca = ccxt_async.kucoinfutures(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca.options["defaultType"] = "swap"

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = int(elm["limits"]["leverage"]["max"])

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for order in res:
                    order["position_side"] = self.determine_pos_side(order)
                    order["qty"] = order["amount"]
                self.handle_order_update(res)
            except Exception as e:
                logging.error(f"exception watch_orders {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_pos_side(self, order):
        # non hedge mode
        if self.has_position("long", order["symbol"]):
            return "long"
        elif self.has_position("short", order["symbol"]):
            return "short"
        elif order["side"] == "buy":
            return "long"
        elif order["side"] == "sell":
            return "short"
        raise Exception(f"unknown side {order['side']}")

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol)
            for order in fetched:
                order["position_side"] = self.determine_pos_side(order)
                order["qty"] = order["amount"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions, fetched_balance = None, None
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(),
                self.cca.fetch_balance(),
            )
            positions = []
            for p in fetched_positions:
                positions.append(
                    {
                        **p,
                        **{
                            "symbol": p["symbol"],
                            "position_side": p["side"],
                            "size": float(p["contracts"]),
                            "price": float(p["entryPrice"]),
                        },
                    }
                )
            balance = fetched_balance["info"]["data"]["marginBalance"]
            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(fetched_positions)
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fetch_tickers()
            return fetched
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        params = {}
        if end_time:
            params["until"] = int(end_time)
        if start_time:
            start_time = int(start_time)
            params["paginate"] = True
        # fetch fills...
        mt = await self.cca.fetch_my_trades(since=start_time, params=params)
        for i in range(len(mt)):
            mt[i]["qty"] = mt[i]["amount"]
            mt[i]["pnl"] = 0.0
            if mt[i]["side"] == "buy":
                mt[i]["position_side"] = (
                    "long" if float(mt[i]["info"]["closeFeePay"]) == 0.0 else "short"
                )
            elif mt[i]["side"] == "sell":
                mt[i]["position_side"] = (
                    "short" if float(mt[i]["info"]["closeFeePay"]) == 0.0 else "long"
                )
            else:
                raise Exception(f"invalid side {mt[i]}")
        mt = sorted(mt, key=lambda x: x["timestamp"])
        closes = [
            x
            for x in mt
            if (x["side"] == "sell" and x["position_side"] == "long")
            or (x["side"] == "buy" and x["position_side"] == "short")
        ]
        if not closes:
            return mt

        # fetch pos history for pnls
        ph = await self.cca.fetch_positions_history(
            since=closes[0]["timestamp"] - 60000, params={"until": closes[-1]["timestamp"] + 60000}
        )

        # match up...
        cld, phd = defaultdict(list), defaultdict(list)
        for x in closes:
            cld[x["symbol"]].append(x)
        for x in ph:
            phd[x["symbol"]].append(x)
        matches = []
        seen_trade_id = set()
        for symbol in phd:
            if symbol not in cld:
                print(f"debug no fills for pos close {symbol} {phd[symbol]}")
                continue
            for p in phd[symbol]:
                with_td = sorted(
                    [x for x in cld[symbol] if x["id"] not in seen_trade_id],
                    key=lambda x: abs(p["lastUpdateTimestamp"] - x["timestamp"]),
                )
                best_match = with_td[0]
                matches.append((p, best_match))
                print(
                    f"debug best match fill and pos close {symbol} timedelta {best_match['timestamp'] - p['lastUpdateTimestamp']}ms"
                )
                seen_trade_id.add(best_match["id"])
            if len(phd[symbol]) != len(cld[symbol]):
                print(
                    f"debug len mismatch between closes and positions_history for {symbol}: {len(closes[symbol])} {len(phd[symbol])}"
                )
        # add pnls, dedup and return
        deduped = {}
        for p, c in matches:
            c["pnl"] = p["realizedPnl"]
            if c["id"] in deduped:
                print(f"debug unexpected duplicate {c}")
                continue
            deduped[c["id"]] = c
        for t in mt:
            if t["id"] not in deduped:
                deduped[t["id"]] = t

        return sorted(deduped.values(), key=lambda x: x["timestamp"])

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(orders, "execute_order")

    async def execute_order(self, order: dict) -> dict:
        order_type = order["type"] if "type" in order else "limit"
        reduce_only = order["reduce_only"] if "reduce_only" in order else False
        params = {
            "symbol": order["symbol"],
            "type": order_type,
            "side": order["side"],
            "amount": abs(order["qty"]),
            "price": order["price"],
            "params": {
                "timeInForce": "GTC",
                "reduceOnly": reduce_only,
                "marginMode": "CROSS",
            },
        }
        # print(params)
        executed = await self.cca.create_order(**params)
        # print(executed)
        return executed

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return order is not None and order["id"] in executed.get("cancelledOrderIds", [])
        except:
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cca.cancel_order(order["id"], symbol=order["symbol"])
            return executed
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_cancellation(orders[0])]
        return await self.execute_multiple(orders, "execute_cancellation")

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

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call = []
        for symbol in symbols:
            try:
                params = {
                    "marginMode": "cross",
                    "symbol": symbol,
                }
                coros_to_call.append(
                    (
                        symbol,
                        "set_margin_mode",
                        asyncio.create_task(self.cca.set_margin_mode(**params)),
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error set_margin_mode {e}")
        for symbol, task_name, task in coros_to_call:
            res = None
            to_print = ""
            try:
                res = await task
                to_print += f"{task_name} {res}"
            except Exception as e:
                logging.error(f"{symbol} error {task_name} {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

        coros_to_call = []
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.config_get(["live", "leverage"], symbol=symbol),
                        )
                    ),
                    "symbol": symbol,
                    "params": {"marginMode": "cross"},
                }
                coros_to_call.append(
                    (symbol, "set_leverage", asyncio.create_task(self.cca.set_leverage(**params)))
                )
            except Exception as e:
                logging.error(f"{symbol}: error set_margin_mode {e}")
        for symbol, task_name, task in coros_to_call:
            res = None
            to_print = ""
            try:
                res = await task
                to_print += f"{task_name} {res}"
            except Exception as e:
                logging.error(f"{symbol} error {task_name} {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")
