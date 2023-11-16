import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"


import logging
import traceback
import argparse
import asyncio
import json
import pprint

from procedures import load_broker_code, load_user_info, utc_ms


class Passivbot:
    def __init__(self, config: dict):
        self.config = config
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
        if upd["bid"] != self.tickers[upd["symbol"]]["bid"] or upd["ask"] != self.tickers[upd["symbol"]]["ask"]:
            self.tickers[upd["symbol"]]["bid"] = upd["bid"]
            self.tickers[upd["symbol"]]["ask"] = upd["ask"]
            # pprint.pprint(upd)
