import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"


import logging
import traceback
import argparse
import asyncio
import json
import pprint

from procedures import load_broker_code, load_user_info, utc_ms, make_get_filepath


class Passivbot:
    def __init__(self, config: dict):
        self.config = config
        self.user = config["user"]
        self.user_info = load_user_info(config["user"])
        self.exchange = self.user_info["exchange"]
        self.broker_code = load_broker_code(self.user_info["exchange"])
        self.balance = 0.0
        self.upd_timestamps = {"balance": 0.0, "open_orders": {}, "tickers": {}}
        self.positions = {}
        self.open_orders = {}
        self.pnls = []
        self.tickers = {}
        self.emas_long = {}
        self.emas_short = {}
        self.symbol_ids = {}
        self.min_costs = {}
        self.min_qtys = {}
        self.qty_steps = {}
        self.price_steps = {}
        self.c_mults = {}
        self.debug_event_log = []
        self.stop_websocket = False
        self.pnls_cache_filepath = make_get_filepath(f"caches/{self.exchange}/{self.user}_pnls.json")
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    async def execute_orders_refresh(self, symbols: [str] = []):
        # 1. fetch open orders and position via REST
        # 2. if stuck, fetch pnls via REST
        # 3. calc orders_to_create and orders_to_cancel
        # 4. cancel wrong orders
        # 5. create missing orders

        # if symbols is empty list: update for all symbols
        # else: update for given symbols
        pass

    async def handle_order_update(self, upd_list):
        try:
            self.debug_event_log.append(upd_list)
            for upd in upd_list:
                if upd["symbol"] not in self.symbols:
                    print("debug unknown symbol", upd["symbol"])
                    return
                if upd["filled"] > 0.0:
                    # There was a fill, partial or full. Schedule update of open orders, pnls, position.
                    pass
                elif upd["status"] == "canceled":
                    # remove order from open_orders
                    self.open_orders[upd["symbol"]] = [
                        elm for elm in self.open_orders[upd["symbol"]] if elm["id"] != upd["id"]
                    ]
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                elif upd["status"] == "open":
                    # add order to open_orders
                    if upd["id"] not in {x["id"] for x in self.open_orders[upd["symbol"]]}:
                        self.open_orders[upd["symbol"]].append(upd)
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                else:
                    print("debug open orders unknown type", upd)
        except Exception as e:
            logging.error(f"error updating open orders from websocket {upd_list} {e}")
            traceback.print_exc()
        pprint.pprint(upd_list)

    async def handle_balance_update(self, upd):
        try:
            self.debug_event_log.append(upd)
            self.balance = upd["USDT"]["total"]
            self.upd_timestamps["balance"] = utc_ms()
        except Exception as e:
            logging.error(f"error updating balance from websocket {upd} {e}")
            traceback.print_exc()
        pprint.pprint(upd)

    async def handle_ticker_update(self, upd):
        self.upd_timestamps["tickers"][upd["symbol"]] = utc_ms()  # update timestamp
        if (
            upd["bid"] != self.tickers[upd["symbol"]]["bid"]
            or upd["ask"] != self.tickers[upd["symbol"]]["ask"]
        ):
            self.tickers[upd["symbol"]]["bid"] = upd["bid"]
            self.tickers[upd["symbol"]]["ask"] = upd["ask"]
            # pprint.pprint(upd)

    async def update_pnls(self):
        # fetch latest pnls
        # dump new pnls to cache
        if len(self.pnls) == 0:
            # load pnls from cache
            pnls_cache = []
            try:
                if os.path.exists(self.pnls_cache_filepath):
                    pnls_cache = json.load(open(self.pnls_cache_filepath))
            except Exception as e:
                logging.error(f"error loading {self.pnls_cache_filepath} {e}")
            # fetch pnls since latest timestamp
            self.pnls = pnls_cache
        start_time = (
            self.pnls[-1]["updatedTime"]
            if self.pnls
            else utc_ms() - 1000 * 60 * 60 * 24 * self.config["pnls_max_lookback_days"]
        )
        new_pnls = await self.fetch_pnls(start_time=start_time)
        len_pnls = len(self.pnls)
        self.pnls = sorted(
            {elm["orderId"] + str(elm["qty"]): elm for elm in self.pnls + new_pnls}.values(),
            key=lambda x: x["updatedTime"],
        )
        if len(self.pnls) > len_pnls:
            logging.debug(f"{len(self.pnls) - len_pnls} new pnls")
            print(f"{len(self.pnls) - len_pnls} new pnls")
            try:
                json.dump(self.pnls, open(self.pnls_cache_filepath, "w"))
            except Exception as e:
                logging.error(f"error dumping pnls to {self.pnls_cache_filepath} {e}")
        return True

    async def update_open_orders(self):
        open_orders = await self.fetch_open_orders()
        oo_ids_old = {elm["id"] for sublist in self.open_orders.values() for elm in sublist}
        for oo in open_orders:
            if oo["id"] not in oo_ids_old:
                # there was a new open order not caught by websocket
                logging.debug(f"new open order {oo['symbol']} {oo['position_side']} {oo['id']}")
                print(f"new open order {oo['symbol']} {oo['position_side']} {oo['id']}")
        oo_ids_new = {elm["id"] for elm in open_orders}
        for oo in [elm for sublist in self.open_orders.values() for elm in sublist]:
            if oo["id"] not in oo_ids_new:
                # there was an order cancellation not caught by websocket
                logging.debug(f"cancelled open order {oo['symbol']} {oo['position_side']} {oo['id']}")
                print(f"cancelled open order {oo['symbol']} {oo['position_side']} {oo['id']}")
        self.open_orders = {symbol: [] for symbol in self.open_orders}
        for elm in open_orders:
            if elm["symbol"] in self.open_orders:
                self.open_orders[elm["symbol"]].append(elm)
            else:
                logging.debug(
                    f"{elm['symbol']} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
                print(
                    f"debug {elm['symbol']} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
        return True

    async def update_positions(self):
        positions_list_new = await self.fetch_positions()
        positions_new = {
            symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0, "price": 0.0}}
            for symbol in self.positions
        }
        for elm in positions_list_new:
            if elm["symbol"] not in self.positions:
                print(
                    f"debug {elm['symbol']} has a {elm['position_side']} position, but is not under passivbot management"
                )
                logging.debug(
                    f"debug {elm['symbol']} has a {elm['position_side']} position, but is not under passivbot management"
                )
            else:
                p_ = self.positions[elm["symbol"]][elm["position_side"]]
                new_pos = {
                    "size": abs(elm["contracts"])
                    * (-1.0 if elm["position_side"] == "short" else 1.0),
                    "price": elm["entryPrice"],
                }
                if new_pos != p_:
                    print(f"{elm['symbol']} {elm['position_side']} changed: {p_} -> {new_pos}")
                    logging.debug(
                        f"{elm['symbol']} {elm['position_side']} changed: {p_} -> {new_pos}"
                    )
                positions_new[elm["symbol"]][elm["position_side"]] = new_pos

        for symbol in self.positions:
            for side in self.positions[symbol]:
                if self.positions[symbol][side] != positions_new[symbol][side]:
                    print(
                        f"{symbol} {side} changed: {self.positions[symbol][side]} -> {positions_new[symbol][side]}"
                    )
                    logging.debug(
                        f"{symbol} {side} changed: {self.positions[symbol][side]} -> {positions_new[symbol][side]}"
                    )
        self.positions = positions_new
        return True
