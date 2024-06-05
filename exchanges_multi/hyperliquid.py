from passivbot_multi import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import json
import numpy as np
from pure_funcs import (
    multi_replace,
    floatify,
    ts_to_date_utc,
    calc_hash,
    shorten_custom_id,
    coin2symbol,
)
from njit_funcs import (
    calc_diff,
    round_,
    round_up,
    round_dn,
    round_dynamic,
    round_dynamic_up,
    round_dynamic_dn,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class HyperliquidBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "walletAddress": self.user_info["wallet_address"],
                "privateKey": self.user_info["private_key"],
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "walletAddress": self.user_info["wallet_address"],
                "privateKey": self.user_info["private_key"],
            }
        )
        self.cca.options["defaultType"] = "swap"
        self.quote = "USDC"
        self.hedge_mode = False
        self.significant_digits = {}
        if "is_vault" not in self.user_info or self.user_info["is_vault"] == "":
            logging.info(
                f"parameter 'is_vault' missing from api-keys.json for user {self.user}. Setting to false"
            )
            self.user_info["is_vault"] = False

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                10.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = (
                elm["precision"]["amount"]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = round_(10 ** -elm["precision"]["price"], 0.0000000001)
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = (
                int(elm["info"]["maxLeverage"]) if "maxLeverage" in elm["info"] else 0
            )
        self.n_decimal_places = 6
        self.n_significant_figures = 5

    async def start_websockets(self):
        await asyncio.gather(
            self.watch_balance(),
            self.watch_orders(),
            self.watch_tickers(),
        )

    async def watch_balance(self):
        # hyperliquid ccxt watch balance not supported.
        # relying instead on periodic REST updates
        res = None
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.cca.fetch_balance()
                res[self.quote]["total"] = float(res["info"]["marginSummary"]["accountValue"]) - sum(
                    [float(x["position"]["unrealizedPnl"]) for x in res["info"]["assetPositions"]]
                )
                self.handle_balance_update(res)
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"exception watch_balance {res} {e}")
                traceback.print_exc()

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
                logging.error(f"exception watch_orders {res} {e}")
                traceback.print_exc()

    async def watch_tickers(self):
        self.WS_ticker_tasks = {}
        while not self.stop_websocket:
            current_symbols = set(self.active_symbols)
            started_symbols = set(self.WS_ticker_tasks.keys())

            # Start watch_ticker tasks for new symbols
            for symbol in current_symbols - started_symbols:
                task = asyncio.create_task(self.watch_ticker(symbol))
                self.WS_ticker_tasks[symbol] = task
                logging.info(f"Started watching ticker for symbol: {symbol}")

            # Cancel tasks for symbols that are no longer active
            for symbol in started_symbols - current_symbols:
                self.WS_ticker_tasks[symbol].cancel()
                del self.WS_ticker_tasks[symbol]
                logging.info(f"Stopped watching ticker for symbol: {symbol}")

            # Wait a bit before checking again
            await asyncio.sleep(1)  # Adjust sleep time as needed

    async def watch_ticker(self, symbol):
        while not self.stop_websocket and symbol in self.active_symbols:
            try:
                res = await self.ccp.watch_order_book(symbol)
                if res["bids"] and res["asks"]:
                    res["bid"], res["ask"] = res["bids"][0][0], res["asks"][0][0]
                    res["last"] = (res["bid"] + res["ask"]) / 2
                    self.handle_ticker_update(res)
            except Exception as e:
                logging.error(f"exception watch_ticker {symbol} {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(1)
            await asyncio.sleep(0.1)

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
            return "long" if order["side"] == "buy" else "short"

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders()
            for i in range(len(fetched)):
                fetched[i]["position_side"] = self.determine_pos_side(fetched[i])
                fetched[i]["qty"] = fetched[i]["amount"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self) -> ([dict], float):
        info = None
        try:
            info = await self.cca.fetch_balance()
            balance = float(info["info"]["marginSummary"]["accountValue"]) - sum(
                [float(x["position"]["unrealizedPnl"]) for x in info["info"]["assetPositions"]]
            )
            positions = [
                {
                    "symbol": x["position"]["coin"] + "/USDC:USDC",
                    "position_side": (
                        "long" if (size := float(x["position"]["szi"])) > 0.0 else "short"
                    ),
                    "size": size,
                    "price": float(x["position"]["entryPx"]),
                }
                for x in info["info"]["assetPositions"]
            ]

            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(info)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fetch(
                "https://api.hyperliquid.xyz/info",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"type": "allMids"}),
            )
            return {
                coin2symbol(coin, self.quote): {
                    "bid": float(fetched[coin]),
                    "ask": float(fetched[coin]),
                    "last": float(fetched[coin]),
                }
                for coin in fetched
            }
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        # fetches latest ohlcvs
        fetched = None
        str2int = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60 * 4}
        n_candles = 480
        try:
            since = int(utc_ms() - 1000 * 60 * str2int[timeframe] * n_candles)
            fetched = await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
        limit = 2000
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
            logging.info(f"debug fetching income {ts_to_date_utc(fetched[-1]['timestamp'])}")
            end_time = fetched[0]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        # if there are more fills in timeframe than 100, it will fetch latest
        try:
            if end_time is None:
                end_time = utc_ms() + 1000 * 60 * 60 * 24
            if start_time is None:
                start_time = end_time - 1000 * 60 * 60 * 24 * 7
            fetched = await self.cca.fetch_my_trades(
                since=int(start_time), params={"endTime": int(end_time)}
            )
            for i in range(len(fetched)):
                fetched[i]["pnl"] = float(fetched[i]["info"]["closedPnl"])
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching pnl {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        return await self.execute_cancellations([order])

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        res = None
        try:
            if len(orders) > self.config["max_n_cancellations_per_batch"]:
                # prioritize cancelling reduce-only orders
                try:
                    reduce_only_orders = [x for x in orders if x["reduce_only"]]
                    rest = [x for x in orders if not x["reduce_only"]]
                    orders = (reduce_only_orders + rest)[
                        : self.config["max_n_cancellations_per_batch"]
                    ]
                except Exception as e:
                    logging.error(f"debug filter cancellations {e}")
            by_symbol = {}
            for order in orders:
                if order["symbol"] not in by_symbol:
                    by_symbol[order["symbol"]] = []
                by_symbol[order["symbol"]].append(order)
            syms = sorted(by_symbol)
            res = await asyncio.gather(
                *[
                    self.cca.cancel_orders(
                        [x["id"] for x in by_symbol[sym]],
                        symbol=sym,
                        params=(
                            {"vaultAddress": self.user_info["wallet_address"]}
                            if self.user_info["is_vault"]
                            else {}
                        ),
                    )
                    for sym in syms
                ]
            )
            cancellations = []
            for sym, elm in zip(syms, res):
                if "status" in elm and elm["status"] == "ok":
                    for status, order in zip(elm["response"]["data"]["statuses"], by_symbol[sym]):
                        if status == "success":
                            cancellations.append(order)
            return cancellations
        except Exception as e:
            logging.error(f"error executing cancellations {e}")
            print_async_exception(res)
            traceback.print_exc()

    async def execute_order(self, order: dict) -> dict:
        return await self.execute_orders([order])

    async def execute_orders(self, orders: [dict]) -> [dict]:
        res = None
        try:
            if len(orders) == 0:
                return []
            to_execute = []
            for order in orders:
                to_execute.append(
                    {
                        "symbol": order["symbol"],
                        "type": "limit",
                        "side": order["side"],
                        "amount": order["qty"],
                        "price": order["price"],
                        "params": {
                            # "orderType": {"limit": {"tif": "Alo"}},
                            # "cloid": order["custom_id"],
                            "reduceOnly": order["reduce_only"],
                            "timeInForce": "Alo",
                        },
                    }
                )
            res = await self.cca.create_orders(
                to_execute,
                params=(
                    {"vaultAddress": self.user_info["wallet_address"]}
                    if self.user_info["is_vault"]
                    else {}
                ),
            )
            executed = []
            for ex, order in zip(res, orders):
                if "info" in ex and "filled" in ex["info"] or "resting" in ex["info"]:
                    executed.append({**ex, **order})
            return executed
        except Exception as e:
            logging.error(f"error executing orders {e}")
            print_async_exception(res)
            traceback.print_exc()

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.live_configs[symbol]["leverage"],
                        )
                    )
                }
                if self.user_info["is_vault"]:
                    params["vaultAddress"] = self.user_info["wallet_address"]
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode("cross", symbol=symbol, params=params)
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode and leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                if '"code":"59107"' in e.args[0]:
                    to_print += f" cross mode and leverage: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting cross mode {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        pass

    def calc_ideal_orders(self):
        # hyperliquid needs custom price rounding
        ideal_orders = super().calc_ideal_orders()
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
