import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import traceback
import argparse
import asyncio
import json
import signal
import pprint
import numpy as np
import time
import random
from procedures import (
    load_live_config,
    make_get_filepath,
    load_exchange_key_secret_passphrase,
    numpyize,
    print_async_exception,
    utc_ms,
    load_broker_code,
)
from pure_funcs import (
    filter_orders,
    create_xk,
    round_dynamic,
    denumpyize,
    spotify_config,
    determine_passivbot_mode,
    config_pretty_str,
    shorten_custom_id,
)
from njit_funcs import (
    qty_to_cost,
    calc_diff,
    round_,
    calc_close_grid_long,
    calc_close_grid_short,
    calc_upnl,
    calc_entry_grid_long,
    calc_entry_grid_short,
    calc_samples,
    calc_emas_last,
    calc_ema,
)
from njit_funcs_neat_grid import (
    calc_neat_grid_long,
    calc_neat_grid_short,
)
from njit_funcs_recursive_grid import (
    calc_recursive_entries_long,
    calc_recursive_entries_short,
)
from njit_clock import (
    calc_clock_entry_long,
    calc_clock_entry_short,
    calc_clock_close_long,
    calc_clock_close_short,
)
from typing import Union, Dict, List

import websockets
import logging

TEST_MODE_SUPPORTED_EXCHANGES = ["bybit"]


class Bot:
    def __init__(self, config: dict):
        self.spot = False
        self.config = config
        self.config["max_leverage"] = 25
        self.xk = {}

        self.ws_user = None
        self.ws_market = None

        self.hedge_mode = self.config["hedge_mode"] = True
        self.set_config(self.config)

        self.ts_locked = {
            k: 0.0
            for k in [
                "cancel_orders",
                "update_open_orders",
                "cancel_and_create",
                "update_position",
                "create_orders",
                "check_fills",
                "update_fills",
                "update_last_fills_timestamps",
                "force_update",
            ]
        }
        self.ts_released = {k: 1.0 for k in self.ts_locked}
        self.error_halt = {
            "update_open_orders": False,
            "update_fills": False,
            "update_position": False,
        }
        self.heartbeat_ts = 0
        self.heartbeat_interval_seconds = 60 * 60
        self.listen_key = None

        self.position = {}
        self.open_orders = []
        self.fills = []
        self.last_fills_timestamps = {
            "clock_entry_long": 0,
            "clock_entry_short": 0,
            "clock_close_long": 0,
            "clock_close_short": 0,
        }
        self.price = 0.0
        self.ob = [0.0, 0.0]
        self.emas_long = np.zeros(3)
        self.emas_short = np.zeros(3)
        self.ema_sec = 0

        self.n_orders_per_execution = 2
        self.delay_between_executions = 3
        self.force_update_interval = 30

        self.c_mult = self.config["c_mult"] = 1.0

        self.log_filepath = make_get_filepath(f"logs/{self.exchange}/{config['config_name']}.log")

        self.api_keys = config["api_keys"] if "api_keys" in config else None
        _, self.key, self.secret, self.passphrase = load_exchange_key_secret_passphrase(
            self.user, self.api_keys
        )
        self.broker_code = load_broker_code(self.exchange)

        self.log_level = 0

        self.user_stream_task = None
        self.market_stream_task = None

        self.stop_websocket = False
        self.process_websocket_ticks = True

    def set_config(self, config):
        for k, v in [
            ("long_mode", None),
            ("short_mode", None),
            ("test_mode", False),
            ("assigned_balance", None),
            ("cross_wallet_pct", 1.0),
            ("price_distance_threshold", 0.5),
            ("c_mult", 1.0),
            ("leverage", 7.0),
            ("countdown", False),
            ("countdown_offset", 0),
            ("ohlcv", True),
        ]:
            if k not in config:
                config[k] = v
        self.passivbot_mode = config["passivbot_mode"] = determine_passivbot_mode(config)
        if config["cross_wallet_pct"] > 1.0 or config["cross_wallet_pct"] <= 0.0:
            logging.warning(
                f"Invalid cross_wallet_pct given: {config['cross_wallet_pct']}.  "
                + "It must be greater than zero and less than or equal to one.  Defaulting to 1.0."
            )
            config["cross_wallet_pct"] = 1.0
        self.config["do_long"] = config["long"]["enabled"]
        self.config["do_short"] = config["short"]["enabled"]
        self.ema_spans_long = np.array(
            sorted(
                [
                    config["long"]["ema_span_0"],
                    (config["long"]["ema_span_0"] * config["long"]["ema_span_1"]) ** 0.5,
                    config["long"]["ema_span_1"],
                ]
            )
        )
        self.ema_spans_short = np.array(
            sorted(
                [
                    config["short"]["ema_span_0"],
                    (config["short"]["ema_span_0"] * config["short"]["ema_span_1"]) ** 0.5,
                    config["short"]["ema_span_1"],
                ]
            )
        )
        self.config = config
        for key in config:
            setattr(self, key, config[key])
            if key in self.xk:
                self.xk[key] = config[key]

    def set_config_value(self, key, value):
        self.config[key] = value
        setattr(self, key, self.config[key])

    async def _init(self):
        self.xk = create_xk(self.config)
        if self.passivbot_mode == "clock":
            self.xk["auto_unstuck_ema_dist"] = (0.0, 0.0)
            self.xk["auto_unstuck_wallet_exposure_threshold"] = (0.0, 0.0)
            self.xk["delay_between_fills_ms_entry"] = (
                self.config["long"]["delay_between_fills_minutes_entry"] * 60 * 1000.0,
                self.config["short"]["delay_between_fills_minutes_entry"] * 60 * 1000.0,
            )
            self.xk["delay_between_fills_ms_close"] = (
                self.config["long"]["delay_between_fills_minutes_close"] * 60 * 1000.0,
                self.config["short"]["delay_between_fills_minutes_close"] * 60 * 1000.0,
            )
        print("initiating position, open orders, fills, exchange config, order book, and emas...")
        await asyncio.gather(
            self.update_position(),
            self.update_open_orders(),
            self.init_fills(),
            self.init_exchange_config(),
            self.init_order_book(),
            self.init_emas(),
        )
        print("done")
        if "price_step_custom" in self.config and self.config["price_step_custom"] is not None:
            new_price_step = max(
                self.price_step, round_(self.config["price_step_custom"], self.price_step)
            )
            if new_price_step != self.price_step:
                logging.info(f"changing price step from {self.price_step} to {new_price_step}")
                self.price_step = self.config["price_step"] = self.xk["price_step"] = new_price_step
        elif (
            "price_precision_multiplier" in self.config
            and self.config["price_precision_multiplier"] is not None
        ):
            new_price_step = max(
                self.price_step,
                round_(self.ob[0] * self.config["price_precision_multiplier"], self.price_step),
            )
            if new_price_step != self.price_step:
                logging.info(f"changing price step from {self.price_step} to {new_price_step}")
                self.price_step = self.config["price_step"] = self.xk["price_step"] = new_price_step

    def dump_log(self, data) -> None:
        if "logging_level" in self.config and self.config["logging_level"] > 0:
            with open(self.log_filepath, "a") as f:
                f.write(
                    json.dumps({**{"log_timestamp": time.time(), "symbol": self.symbol}, **data})
                    + "\n"
                )

    async def init_emas_1m(self, ohlcvs: dict) -> None:
        samples1m = calc_samples(
            numpyize(
                [
                    [o["timestamp"], o["volume"], o["close"]]
                    for o in sorted(ohlcvs.values(), key=lambda x: x["timestamp"])
                ]
            ),
            60000,
        )
        self.emas_long = calc_emas_last(samples1m[:, 2], self.ema_spans_long)
        self.emas_short = calc_emas_last(samples1m[:, 2], self.ema_spans_short)
        self.alpha_long = 2 / (self.ema_spans_long + 1)
        self.alpha__long = 1 - self.alpha_long
        self.alpha_short = 2 / (self.ema_spans_short + 1)
        self.alpha__short = 1 - self.alpha_short
        self.ema_min = int(round(time.time() // 60 * 60))
        return samples1m

    async def init_emas(self) -> None:
        ohlcvs1m = await self.fetch_ohlcvs(interval="1m")
        max_span = max(list(self.ema_spans_long) + list(self.ema_spans_short))
        for mins, interval in zip([5, 15, 30, 60, 60 * 4], ["5m", "15m", "30m", "1h", "4h"]):
            if max_span <= len(ohlcvs1m) * mins:
                break
        ohlcvs = await self.fetch_ohlcvs(interval=interval)
        ohlcvs = {ohlcv["timestamp"]: ohlcv for ohlcv in ohlcvs + ohlcvs1m}
        if self.ohlcv:
            return await self.init_emas_1m(ohlcvs)
        samples1s = calc_samples(
            numpyize(
                [
                    [o["timestamp"], o["volume"], o["close"]]
                    for o in sorted(ohlcvs.values(), key=lambda x: x["timestamp"])
                ]
            )
        )
        spans1s_long = np.array(self.ema_spans_long) * 60
        spans1s_short = np.array(self.ema_spans_short) * 60
        self.emas_long = calc_emas_last(samples1s[:, 2], spans1s_long)
        self.emas_short = calc_emas_last(samples1s[:, 2], spans1s_short)
        self.alpha_long = 2 / (spans1s_long + 1)
        self.alpha__long = 1 - self.alpha_long
        self.alpha_short = 2 / (spans1s_short + 1)
        self.alpha__short = 1 - self.alpha_short
        self.ema_sec = int(time.time())
        # return samples1s

    def update_emas_1m(self, price: float, prev_price: float) -> None:
        now_min = int(round(time.time() // 60 * 60))
        if now_min <= self.ema_min:
            return
        while self.ema_min < int(round(now_min - 60)):
            self.emas_long = calc_ema(self.alpha_long, self.alpha__long, self.emas_long, prev_price)
            self.emas_short = calc_ema(
                self.alpha_short, self.alpha__short, self.emas_short, prev_price
            )
            self.ema_min += 60
        self.emas_long = calc_ema(self.alpha_long, self.alpha__long, self.emas_long, price)
        self.emas_short = calc_ema(self.alpha_short, self.alpha__short, self.emas_short, price)
        self.ema_min = now_min

    def update_emas(self, price: float, prev_price: float) -> None:
        if self.ohlcv:
            return self.update_emas_1m(price, prev_price)
        now_sec = int(time.time())
        if now_sec <= self.ema_sec:
            return
        while self.ema_sec < int(round(now_sec - 1)):
            self.emas_long = calc_ema(self.alpha_long, self.alpha__long, self.emas_long, prev_price)
            self.emas_short = calc_ema(
                self.alpha_short, self.alpha__short, self.emas_short, prev_price
            )
            self.ema_sec += 1
        self.emas_long = calc_ema(self.alpha_long, self.alpha__long, self.emas_long, price)
        self.emas_short = calc_ema(self.alpha_short, self.alpha__short, self.emas_short, price)
        self.ema_sec = now_sec

    async def update_open_orders(self) -> None:
        if self.ts_locked["update_open_orders"] > self.ts_released["update_open_orders"]:
            return
        try:
            open_orders = await self.fetch_open_orders()
            open_orders = [x for x in open_orders if x["symbol"] == self.symbol]
            if self.open_orders != open_orders:
                self.dump_log({"log_type": "open_orders", "data": open_orders})
            self.open_orders = open_orders
            self.error_halt["update_open_orders"] = False
            return True
        except Exception as e:
            self.error_halt["update_open_orders"] = True

            logging.error(f"error with update open orders {e}")
            traceback.print_exc()
            return False
        finally:
            self.ts_released["update_open_orders"] = time.time()

    def adjust_wallet_balance(self, balance: float) -> float:
        return (
            balance if self.assigned_balance is None else self.assigned_balance
        ) * self.cross_wallet_pct

    def add_wallet_exposures_to_pos(self, position_: dict):
        position = position_.copy()
        position["long"]["wallet_exposure"] = (
            (
                qty_to_cost(
                    position["long"]["size"],
                    position["long"]["price"],
                    self.xk["inverse"],
                    self.xk["c_mult"],
                )
                / position["wallet_balance"]
            )
            if position["wallet_balance"]
            else 0.0
        )
        position["short"]["wallet_exposure"] = (
            (
                qty_to_cost(
                    position["short"]["size"],
                    position["short"]["price"],
                    self.xk["inverse"],
                    self.xk["c_mult"],
                )
                / position["wallet_balance"]
            )
            if position["wallet_balance"]
            else 0.0
        )
        return position

    async def update_position(self) -> None:
        if self.ts_locked["update_position"] > self.ts_released["update_position"]:
            return
        self.ts_locked["update_position"] = time.time()
        try:
            position = await self.fetch_position()
            position["wallet_balance"] = self.adjust_wallet_balance(position["wallet_balance"])
            # isolated equity, not cross equity
            position["equity"] = position["wallet_balance"] + calc_upnl(
                position["long"]["size"],
                position["long"]["price"],
                position["short"]["size"],
                position["short"]["price"],
                self.price,
                self.inverse,
                self.c_mult,
            )
            position = self.add_wallet_exposures_to_pos(position)
            if self.position != position:
                if (
                    self.position
                    and "spot" in self.market_type
                    and (
                        self.position["long"]["size"] != position["long"]["size"]
                        or self.position["short"]["size"] != position["short"]["size"]
                    )
                ):
                    # update fills if position size changed
                    await self.update_fills()
                self.dump_log({"log_type": "position", "data": position})
            self.position = position
            self.error_halt["update_position"] = False
            return True
        except Exception as e:
            self.error_halt["update_position"] = True
            logging.error(f"error with update position {e}")
            traceback.print_exc()
            return False
        finally:
            self.ts_released["update_position"] = time.time()

    async def init_fills(self, n_days_limit=60):
        self.fills = await self.fetch_fills()

    async def update_fills(self) -> [dict]:
        """
        fetches recent fills
        returns list of new fills
        """
        if self.ts_locked["update_fills"] > self.ts_released["update_fills"]:
            return
        self.ts_locked["update_fills"] = time.time()
        try:
            fetched = await self.fetch_fills()
            seen = set()
            updated_fills = []
            for fill in fetched + self.fills:
                if fill["order_id"] not in seen:
                    updated_fills.append(fill)
                    seen.add(fill["order_id"])
            self.fills = sorted(updated_fills, key=lambda x: x["order_id"])[-5000:]
            self.error_halt["update_fills"] = False
        except Exception as e:
            self.error_halt["update_fills"] = True
            logging.error(f"error with update fills {e}")
            traceback.print_exc()
        finally:
            self.ts_released["update_fills"] = time.time()

    async def create_orders(self, orders_to_create: [dict]) -> [dict]:
        if not orders_to_create:
            return []
        if self.ts_locked["create_orders"] > self.ts_released["create_orders"]:
            return []
        self.ts_locked["create_orders"] = time.time()
        try:
            orders = None
            orders_to_create = [order for order in orders_to_create if self.order_is_valid(order)]
            orders = await self.execute_orders(orders_to_create)
            for order in sorted(orders, key=lambda x: calc_diff(x["price"], self.price)):
                if "side" in order:
                    logging.info(
                        f'  created order {order["symbol"]} {order["side"]: <4} '
                        + f'{order["position_side"]: <5} {float(order["qty"])} {float(order["price"])}'
                    )
            return orders
        except Exception as e:
            print(f"error creating orders {e}")
            print_async_exception(orders)
            traceback.print_exc()
            return []
        finally:
            self.ts_released["create_orders"] = time.time()

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if self.ts_locked["cancel_orders"] > self.ts_released["cancel_orders"]:
            return
        self.ts_locked["cancel_orders"] = time.time()
        try:
            if not orders_to_cancel:
                return
            deletions, orders_to_cancel_dedup, oo_ids = [], [], set()
            for o in orders_to_cancel:
                if o["order_id"] not in oo_ids:
                    oo_ids.add(o["order_id"])
                    orders_to_cancel_dedup.append(o)
            cancellations = None
            try:
                cancellations = await self.execute_cancellations(orders_to_cancel_dedup)
                for cancellation in cancellations:
                    if "order_id" in cancellation:
                        logging.info(
                            f'cancelled order {cancellation["symbol"]} {cancellation["side"]: <4} '
                            + f'{cancellation["position_side"]: <5} {cancellation["qty"]} {cancellation["price"]}'
                        )
                        self.open_orders = [
                            oo
                            for oo in self.open_orders
                            if oo["order_id"] != cancellation["order_id"]
                        ]
                return cancellations
            except Exception as e:
                logging.error(f"error cancelling orders {cancellations} {e}")
                print_async_exception(cancellations)
                return []
        finally:
            self.ts_released["cancel_orders"] = time.time()

    def stop(self, signum=None, frame=None) -> None:
        logging.info("Stopping passivbot, please wait...")
        self.stop_websocket = True
        if not self.ohlcv:
            try:
                self.user_stream_task.cancel()
                self.market_stream_task.cancel()

            except Exception as e:
                logging.error(f"An error occurred during shutdown: {e}")

    def pause(self) -> None:
        self.process_websocket_ticks = False

    def resume(self) -> None:
        self.process_websocket_ticks = True

    def calc_orders(self):
        balance = self.position["wallet_balance"]
        psize_long = self.position["long"]["size"]
        pprice_long = self.position["long"]["price"]
        psize_short = self.position["short"]["size"]
        pprice_short = self.position["short"]["price"]

        if self.hedge_mode:
            do_long = self.do_long or psize_long != 0.0
            do_short = self.do_short or psize_short != 0.0
        else:
            no_pos = psize_long == 0.0 and psize_short == 0.0
            do_long = (no_pos and self.do_long) or psize_long != 0.0
            do_short = (no_pos and self.do_short) or psize_short != 0.0
        self.xk["do_long"] = do_long
        self.xk["do_short"] = do_short

        orders = []

        if self.long_mode == "panic":
            if psize_long != 0.0:
                orders.append(
                    {
                        "side": "sell",
                        "position_side": "long",
                        "qty": abs(psize_long),
                        "price": float(self.ob[1]),
                        "type": "limit",
                        "reduce_only": True,
                        "custom_id": "long_panic_close",
                    }
                )
        else:
            if do_long:
                if self.passivbot_mode == "recursive_grid":
                    entries_long = calc_recursive_entries_long(
                        balance,
                        psize_long,
                        pprice_long,
                        self.ob[0],
                        min(self.emas_long),
                        self.xk["inverse"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["initial_qty_pct"][0],
                        self.xk["initial_eprice_ema_dist"][0],
                        self.xk["ddown_factor"][0],
                        self.xk["rentry_pprice_dist"][0],
                        self.xk["rentry_pprice_dist_wallet_exposure_weighting"][0],
                        self.xk["wallet_exposure_limit"][0],
                        self.xk["auto_unstuck_ema_dist"][0],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][0],
                    )
                elif self.passivbot_mode == "static_grid":
                    entries_long = calc_entry_grid_long(
                        balance,
                        psize_long,
                        pprice_long,
                        self.ob[0],
                        min(self.emas_long),
                        self.xk["inverse"],
                        self.xk["do_long"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["grid_span"][0],
                        self.xk["wallet_exposure_limit"][0],
                        self.xk["max_n_entry_orders"][0],
                        self.xk["initial_qty_pct"][0],
                        self.xk["initial_eprice_ema_dist"][0],
                        self.xk["eprice_pprice_diff"][0],
                        self.xk["secondary_allocation"][0],
                        self.xk["secondary_pprice_diff"][0],
                        self.xk["eprice_exp_base"][0],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][0],
                        self.xk["auto_unstuck_ema_dist"][0],
                    )
                elif self.passivbot_mode == "neat_grid":
                    entries_long = calc_neat_grid_long(
                        balance,
                        psize_long,
                        pprice_long,
                        self.ob[0],
                        min(self.emas_long),
                        self.xk["inverse"],
                        self.xk["do_long"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["grid_span"][0],
                        self.xk["wallet_exposure_limit"][0],
                        self.xk["max_n_entry_orders"][0],
                        self.xk["initial_qty_pct"][0],
                        self.xk["initial_eprice_ema_dist"][0],
                        self.xk["eqty_exp_base"][0],
                        self.xk["eprice_exp_base"][0],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][0],
                        self.xk["auto_unstuck_ema_dist"][0],
                    )
                elif self.passivbot_mode == "clock":
                    entries_long = [
                        calc_clock_entry_long(
                            balance,
                            psize_long,
                            pprice_long,
                            self.ob[0],
                            min(self.emas_long),
                            utc_ms(),
                            0
                            if psize_long == 0.0
                            else self.last_fills_timestamps["clock_entry_long"],
                            self.xk["inverse"],
                            self.xk["qty_step"],
                            self.xk["price_step"],
                            self.xk["min_qty"],
                            self.xk["min_cost"],
                            self.xk["c_mult"],
                            self.xk["ema_dist_entry"][0],
                            self.xk["qty_pct_entry"][0],
                            self.xk["we_multiplier_entry"][0],
                            self.xk["delay_weight_entry"][0],
                            self.xk["delay_between_fills_ms_entry"][0],
                            self.xk["wallet_exposure_limit"][0],
                        )
                    ]
                else:
                    raise Exception(f"unknown passivbot mode {self.passivbot_mode}")
                orders += [
                    {
                        "side": "buy",
                        "position_side": "long",
                        "qty": abs(float(o[0])),
                        "price": float(o[1]),
                        "type": "limit",
                        "reduce_only": False,
                        "custom_id": o[2],
                    }
                    for o in entries_long
                    if o[0] > 0.0
                ]
            if do_long or self.long_mode == "tp_only":
                closes_long = calc_close_grid_long(
                    self.xk["backwards_tp"][0],
                    balance,
                    psize_long,
                    pprice_long,
                    self.ob[1],
                    max(self.emas_long),
                    self.xk["inverse"],
                    self.xk["qty_step"],
                    self.xk["price_step"],
                    self.xk["min_qty"],
                    self.xk["min_cost"],
                    self.xk["c_mult"],
                    self.xk["wallet_exposure_limit"][0],
                    self.xk["min_markup"][0],
                    self.xk["markup_range"][0],
                    self.xk["n_close_orders"][0],
                    self.xk["auto_unstuck_wallet_exposure_threshold"][0],
                    self.xk["auto_unstuck_ema_dist"][0],
                )
                if self.passivbot_mode == "clock":
                    clock_close_long = calc_clock_close_long(
                        balance,
                        psize_long,
                        pprice_long,
                        self.ob[1],
                        max(self.emas_long),
                        utc_ms(),
                        self.last_fills_timestamps["clock_close_long"],
                        self.xk["inverse"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["ema_dist_close"][0],
                        self.xk["qty_pct_close"][0],
                        self.xk["we_multiplier_close"][0],
                        self.xk["delay_weight_close"][0],
                        self.xk["delay_between_fills_ms_close"][0],
                        self.xk["wallet_exposure_limit"][0],
                    )
                    if clock_close_long[0] != 0.0 and (
                        not closes_long or clock_close_long[1] <= closes_long[0][1]
                    ):
                        closes_long = [clock_close_long]
                        closes_long += calc_close_grid_long(
                            True,
                            balance,
                            max(0.0, round_(psize_long - abs(clock_close_long[0]), self.qty_step)),
                            pprice_long,
                            self.ob[1],
                            max(self.emas_long),
                            self.xk["inverse"],
                            self.xk["qty_step"],
                            self.xk["price_step"],
                            self.xk["min_qty"],
                            self.xk["min_cost"],
                            self.xk["c_mult"],
                            self.xk["wallet_exposure_limit"][0],
                            self.xk["min_markup"][0],
                            self.xk["markup_range"][0],
                            self.xk["n_close_orders"][0],
                            self.xk["auto_unstuck_wallet_exposure_threshold"][0],
                            self.xk["auto_unstuck_ema_dist"][0],
                        )
                orders += [
                    {
                        "side": "sell",
                        "position_side": "long",
                        "qty": abs(float(o[0])),
                        "price": float(o[1]),
                        "type": "limit",
                        "reduce_only": True,
                        "custom_id": o[2],
                    }
                    for o in closes_long
                    if o[0] < 0.0
                ]
        if self.short_mode == "panic":
            if psize_short != 0.0:
                orders.append(
                    {
                        "side": "buy",
                        "position_side": "short",
                        "qty": abs(psize_short),
                        "price": float(self.ob[0]),
                        "type": "limit",
                        "reduce_only": True,
                        "custom_id": "short_panic_close",
                    }
                )
        else:
            if do_short:
                if self.passivbot_mode == "recursive_grid":
                    entries_short = calc_recursive_entries_short(
                        balance,
                        psize_short,
                        pprice_short,
                        self.ob[1],
                        max(self.emas_short),
                        self.xk["inverse"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["initial_qty_pct"][1],
                        self.xk["initial_eprice_ema_dist"][1],
                        self.xk["ddown_factor"][1],
                        self.xk["rentry_pprice_dist"][1],
                        self.xk["rentry_pprice_dist_wallet_exposure_weighting"][1],
                        self.xk["wallet_exposure_limit"][1],
                        self.xk["auto_unstuck_ema_dist"][1],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][1],
                    )
                elif self.passivbot_mode == "neat_grid":
                    entries_short = calc_neat_grid_short(
                        balance,
                        psize_short,
                        pprice_short,
                        self.ob[1],
                        max(self.emas_short),
                        self.xk["inverse"],
                        self.xk["do_short"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["grid_span"][1],
                        self.xk["wallet_exposure_limit"][1],
                        self.xk["max_n_entry_orders"][1],
                        self.xk["initial_qty_pct"][1],
                        self.xk["initial_eprice_ema_dist"][1],
                        self.xk["eqty_exp_base"][1],
                        self.xk["eprice_exp_base"][1],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][1],
                        self.xk["auto_unstuck_ema_dist"][1],
                    )
                elif self.passivbot_mode == "static_grid":
                    entries_short = calc_entry_grid_short(
                        balance,
                        psize_short,
                        pprice_short,
                        self.ob[1],
                        max(self.emas_short),
                        self.xk["inverse"],
                        self.xk["do_short"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["grid_span"][1],
                        self.xk["wallet_exposure_limit"][1],
                        self.xk["max_n_entry_orders"][1],
                        self.xk["initial_qty_pct"][1],
                        self.xk["initial_eprice_ema_dist"][1],
                        self.xk["eprice_pprice_diff"][1],
                        self.xk["secondary_allocation"][1],
                        self.xk["secondary_pprice_diff"][1],
                        self.xk["eprice_exp_base"][1],
                        self.xk["auto_unstuck_wallet_exposure_threshold"][1],
                        self.xk["auto_unstuck_ema_dist"][1],
                    )
                elif self.passivbot_mode == "clock":
                    entries_short = [
                        calc_clock_entry_short(
                            balance,
                            psize_short,
                            pprice_short,
                            self.ob[1],
                            max(self.emas_short),
                            utc_ms(),
                            0
                            if psize_short == 0.0
                            else self.last_fills_timestamps["clock_entry_short"],
                            self.xk["inverse"],
                            self.xk["qty_step"],
                            self.xk["price_step"],
                            self.xk["min_qty"],
                            self.xk["min_cost"],
                            self.xk["c_mult"],
                            self.xk["ema_dist_entry"][1],
                            self.xk["qty_pct_entry"][1],
                            self.xk["we_multiplier_entry"][1],
                            self.xk["delay_weight_entry"][1],
                            self.xk["delay_between_fills_ms_entry"][1],
                            self.xk["wallet_exposure_limit"][1],
                        )
                    ]
                else:
                    raise Exception(f"unknown passivbot mode {self.passivbot_mode}")
                orders += [
                    {
                        "side": "sell",
                        "position_side": "short",
                        "qty": abs(float(o[0])),
                        "price": float(o[1]),
                        "type": "limit",
                        "reduce_only": False,
                        "custom_id": o[2],
                    }
                    for o in entries_short
                    if o[0] < 0.0
                ]
            if do_short or self.short_mode == "tp_only":
                closes_short = calc_close_grid_short(
                    self.xk["backwards_tp"][1],
                    balance,
                    psize_short,
                    pprice_short,
                    self.ob[0],
                    min(self.emas_short),
                    self.xk["inverse"],
                    self.xk["qty_step"],
                    self.xk["price_step"],
                    self.xk["min_qty"],
                    self.xk["min_cost"],
                    self.xk["c_mult"],
                    self.xk["wallet_exposure_limit"][1],
                    self.xk["min_markup"][1],
                    self.xk["markup_range"][1],
                    self.xk["n_close_orders"][1],
                    self.xk["auto_unstuck_wallet_exposure_threshold"][1],
                    self.xk["auto_unstuck_ema_dist"][1],
                )
                if self.passivbot_mode == "clock":
                    clock_close_short = calc_clock_close_short(
                        balance,
                        psize_short,
                        pprice_short,
                        self.ob[0],
                        min(self.emas_short),
                        utc_ms(),
                        self.last_fills_timestamps["clock_close_short"],
                        self.xk["inverse"],
                        self.xk["qty_step"],
                        self.xk["price_step"],
                        self.xk["min_qty"],
                        self.xk["min_cost"],
                        self.xk["c_mult"],
                        self.xk["ema_dist_close"][1],
                        self.xk["qty_pct_close"][1],
                        self.xk["we_multiplier_close"][1],
                        self.xk["delay_weight_close"][1],
                        self.xk["delay_between_fills_ms_close"][1],
                        self.xk["wallet_exposure_limit"][1],
                    )
                    if clock_close_short[0] != 0.0 and (
                        not closes_short or clock_close_short[1] >= closes_short[0][1]
                    ):
                        closes_short = [clock_close_short]
                        closes_short += calc_close_grid_short(
                            True,
                            balance,
                            -max(
                                0.0,
                                round_(abs(psize_short) - abs(clock_close_short[0]), self.qty_step),
                            ),
                            pprice_short,
                            self.ob[0],
                            min(self.emas_short),
                            self.xk["inverse"],
                            self.xk["qty_step"],
                            self.xk["price_step"],
                            self.xk["min_qty"],
                            self.xk["min_cost"],
                            self.xk["c_mult"],
                            self.xk["wallet_exposure_limit"][1],
                            self.xk["min_markup"][1],
                            self.xk["markup_range"][1],
                            self.xk["n_close_orders"][1],
                            self.xk["auto_unstuck_wallet_exposure_threshold"][1],
                            self.xk["auto_unstuck_ema_dist"][1],
                        )
                orders += [
                    {
                        "side": "buy",
                        "position_side": "short",
                        "qty": abs(float(o[0])),
                        "price": float(o[1]),
                        "type": "limit",
                        "reduce_only": True,
                        "custom_id": o[2],
                    }
                    for o in closes_short
                    if o[0] > 0.0
                ]
        return sorted(orders, key=lambda x: calc_diff(x["price"], self.price))

    async def cancel_and_create(self):
        if self.ts_locked["cancel_and_create"] > self.ts_released["cancel_and_create"]:
            return
        self.ts_locked["cancel_and_create"] = time.time()
        try:
            if any(self.error_halt.values()):
                logging.warning(
                    f"warning:  error in rest api fetch {self.error_halt}, "
                    + "halting order creations/cancellations"
                )
                return []
            ideal_orders = []
            all_orders = self.calc_orders()
            for o in all_orders:
                if (
                    not self.ohlcv
                    and "ientry" in o["custom_id"]
                    and calc_diff(o["price"], self.price) < 0.002
                ):
                    # call update_position() before making initial entry orders
                    # in case websocket has failed
                    logging.info(
                        f"updating position with REST API before creating initial entries.  Last price {self.price}"
                    )
                    await self.update_position()
                    all_orders = self.calc_orders()
                    break
            for o in all_orders:
                if any(x in o["custom_id"] for x in ["ientry", "unstuck"]) and not self.ohlcv:
                    if calc_diff(o["price"], self.price) < 0.01:
                        # EMA based orders must be closer than 1% of current price unless ohlcv mode
                        ideal_orders.append(o)
                else:
                    if calc_diff(o["price"], self.price) < self.price_distance_threshold:
                        # all orders must be closer than x% of current price
                        ideal_orders.append(o)
            to_cancel_, to_create_ = filter_orders(
                self.open_orders,
                ideal_orders,
                keys=["side", "position_side", "qty", "price"],
            )
            to_cancel, to_create = [], []
            for elm in to_cancel_:
                if elm["position_side"] == "long":
                    if self.long_mode == "tp_only":
                        if elm["side"] == "sell":
                            to_cancel.append(elm)
                    elif self.long_mode != "manual":
                        to_cancel.append(elm)
                if elm["position_side"] == "short":
                    if self.short_mode == "tp_only":
                        if elm["side"] == "buy":
                            to_cancel.append(elm)
                    elif self.short_mode != "manual":
                        to_cancel.append(elm)
                else:
                    to_cancel.append(elm)
            for elm in to_create_:
                if elm["position_side"] == "long":
                    if self.long_mode == "tp_only":
                        if elm["side"] == "sell":
                            to_create.append(elm)
                    elif self.long_mode != "manual":
                        to_create.append(elm)
                if elm["position_side"] == "short":
                    if self.short_mode == "tp_only":
                        if elm["side"] == "buy":
                            to_create.append(elm)
                    elif self.short_mode != "manual":
                        to_create.append(elm)

            to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x["price"], self.price))
            to_create = sorted(to_create, key=lambda x: calc_diff(x["price"], self.price))

            """
            logging.info(f"to_cancel {to_cancel}")
            logging.info(f"to create {to_create}")
            return
            """

            results = []
            if to_cancel:
                results.append(
                    asyncio.create_task(
                        self.cancel_orders(to_cancel[: self.max_n_cancellations_per_batch])
                    )
                )
                await asyncio.sleep(
                    0.1
                )  # sleep 10 ms between sending cancellations and sending creations
            if to_create:
                results.append(await self.create_orders(to_create[: self.max_n_orders_per_batch]))
            return results
        finally:
            await asyncio.sleep(self.delay_between_executions)  # sleep before releasing lock
            self.ts_released["cancel_and_create"] = time.time()

    async def on_market_stream_event(self, ticks: [dict]):
        if ticks:
            for tick in ticks:
                if tick["is_buyer_maker"]:
                    self.ob[0] = tick["price"]
                else:
                    self.ob[1] = tick["price"]
            self.update_emas(ticks[-1]["price"], self.price)
            self.price = ticks[-1]["price"]

        now = time.time()
        if now - self.ts_released["force_update"] > self.force_update_interval:
            self.ts_released["force_update"] = now
            # force update pos and open orders thru rest API every x sec (default 30)
            await asyncio.gather(self.update_position(), self.update_open_orders())
        if now - self.heartbeat_ts > self.heartbeat_interval_seconds:
            # print heartbeat once an hour
            self.heartbeat_print()
            self.heartbeat_ts = time.time()
        await self.cancel_and_create()

    def heartbeat_print(self):
        logging.info(f"heartbeat {self.symbol}")
        self.log_position_long()
        self.log_position_short()
        liq_price = self.position["long"]["liquidation_price"]
        if calc_diff(self.position["short"]["liquidation_price"], self.price) < calc_diff(
            liq_price, self.price
        ):
            liq_price = self.position["short"]["liquidation_price"]
        logging.info(
            f'balance: {round_dynamic(self.position["wallet_balance"], 6)}'
            + f' equity: {round_dynamic(self.position["equity"], 6)} last price: {self.price}'
            + f" liq: {round_(liq_price, self.price_step)}"
        )

    def log_position_long(self, prev_pos=None):
        closes_long = sorted(
            [o for o in self.open_orders if o["side"] == "sell" and o["position_side"] == "long"],
            key=lambda x: x["price"],
        )
        entries_long = sorted(
            [o for o in self.open_orders if o["side"] == "buy" and o["position_side"] == "long"],
            key=lambda x: x["price"],
        )
        leqty, leprice = (
            (entries_long[-1]["qty"], entries_long[-1]["price"]) if entries_long else (0.0, 0.0)
        )
        lcqty, lcprice = (
            (closes_long[0]["qty"], closes_long[0]["price"]) if closes_long else (0.0, 0.0)
        )
        prev_pos_line = (
            (
                f'long: {prev_pos["long"]["size"]} @'
                + f' {round_(prev_pos["long"]["price"], self.price_step)} -> '
            )
            if prev_pos
            else ""
        )
        logging.info(
            prev_pos_line
            + f'long: {self.position["long"]["size"]} @'
            + f' {round_(self.position["long"]["price"], self.price_step)}'
            + f' lWE: {self.position["long"]["wallet_exposure"]:.4f}'
            + f' pprc diff {self.position["long"]["price"] / self.price - 1:.3f}'
            + f" EMAs: {[round_dynamic(e, 5) for e in self.emas_long]}"
            + f" e {leqty} @ {leprice} | c {lcqty} @ {lcprice}"
        )

    def log_position_short(self, prev_pos=None):
        closes_short = sorted(
            [o for o in self.open_orders if o["side"] == "buy" and o["position_side"] == "short"],
            key=lambda x: x["price"],
        )
        entries_short = sorted(
            [o for o in self.open_orders if o["side"] == "sell" and o["position_side"] == "short"],
            key=lambda x: x["price"],
        )
        leqty, leprice = (
            (entries_short[0]["qty"], entries_short[0]["price"]) if entries_short else (0.0, 0.0)
        )
        lcqty, lcprice = (
            (closes_short[-1]["qty"], closes_short[-1]["price"]) if closes_short else (0.0, 0.0)
        )
        pprice_diff = (
            (self.price / self.position["short"]["price"] - 1)
            if self.position["short"]["price"] != 0.0
            else 1.0
        )
        prev_pos_line = (
            (
                f'short: {prev_pos["short"]["size"]} @'
                + f' {round_(prev_pos["short"]["price"], self.price_step)} -> '
            )
            if prev_pos
            else ""
        )
        logging.info(
            prev_pos_line
            + f'short: {self.position["short"]["size"]} @'
            + f' {round_(self.position["short"]["price"], self.price_step)}'
            + f' sWE: {self.position["short"]["wallet_exposure"]:.4f}'
            + f" pprc diff {pprice_diff:.3f}"
            + f" EMAs: {[round_dynamic(e, 5) for e in self.emas_short]}"
            + f" e {leqty} @ {leprice} | c {lcqty} @ {lcprice}"
        )

    async def on_user_stream_events(self, events: Union[List[Dict], List]) -> None:
        if type(events) == list:
            for event in events:
                await self.on_user_stream_event(event)
        else:
            await self.on_user_stream_event(events)

    async def on_user_stream_event(self, event: dict) -> None:
        try:
            if "logged_in" in event:
                # bitget needs to login before sending subscribe requests
                await self.subscribe_to_user_stream(self.ws_user)
            pos_change = False
            if "wallet_balance" in event:
                new_wallet_balance = self.adjust_wallet_balance(event["wallet_balance"])
                if new_wallet_balance != self.position["wallet_balance"]:
                    liq_price = self.position["long"]["liquidation_price"]
                    if calc_diff(self.position["short"]["liquidation_price"], self.price) < calc_diff(
                        liq_price, self.price
                    ):
                        liq_price = self.position["short"]["liquidation_price"]
                    logging.info(
                        f"balance: {round_dynamic(new_wallet_balance, 6)}"
                        + f' equity: {round_dynamic(self.position["equity"], 6)} last price: {self.price}'
                        + f" liq: {round_(liq_price, self.price_step)}"
                    )
                self.position["wallet_balance"] = new_wallet_balance
                pos_change = True
            if "psize_long" in event:
                do_log = False
                if event["psize_long"] != self.position["long"]["size"]:
                    do_log = True
                self.position["long"]["size"] = event["psize_long"]
                self.position["long"]["price"] = event["pprice_long"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
                if do_log:
                    self.log_position_long()
            if "psize_short" in event:
                do_log = False
                if event["psize_short"] != self.position["short"]["size"]:
                    do_log = True
                self.position["short"]["size"] = event["psize_short"]
                self.position["short"]["price"] = event["pprice_short"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
                if do_log:
                    self.log_position_short()
            if "new_open_order" in event:
                if event["new_open_order"]["order_id"] not in {
                    x["order_id"] for x in self.open_orders
                }:
                    self.open_orders.append(event["new_open_order"])
            if "deleted_order_id" in event:
                self.open_orders = [
                    oo for oo in self.open_orders if oo["order_id"] != event["deleted_order_id"]
                ]
            if "partially_filled" in event:
                logging.info(f"partial fill {list(event.values())}")
                await self.update_open_orders()
            if pos_change:
                self.position["equity"] = self.position["wallet_balance"] + calc_upnl(
                    self.position["long"]["size"],
                    self.position["long"]["price"],
                    self.position["short"]["size"],
                    self.position["short"]["price"],
                    self.price,
                    self.inverse,
                    self.c_mult,
                )
                await asyncio.sleep(
                    0.01
                )  # sleep 10 ms to catch both pos update and open orders update
                await self.cancel_and_create()
        except Exception as e:
            logging.error(f"error handling user stream event, {e}")
            traceback.print_exc()

    def flush_stuck_locks(self, timeout: float = 5.0) -> None:
        timeout = max(timeout, self.delay_between_executions + 1)
        now = time.time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    logging.warning(f"flushing stuck lock {key}")
                    self.ts_released[key] = now

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        self.process_websocket_ticks = True
        logging.info("starting websockets...")
        self.user_stream_task = asyncio.create_task(self.start_websocket_user_stream())
        self.market_stream_task = asyncio.create_task(self.start_websocket_market_stream())
        await asyncio.gather(self.user_stream_task, self.market_stream_task)

    async def beat_heart_user_stream(self) -> None:
        pass

    async def beat_heart_market_stream(self) -> None:
        pass

    async def init_user_stream(self) -> None:
        pass

    async def init_market_stream(self) -> None:
        pass

    async def start_websocket_user_stream(self) -> None:
        await self.init_user_stream()
        asyncio.create_task(self.beat_heart_user_stream())
        logging.info(f"url {self.endpoints['websocket_user']}")
        async with websockets.connect(self.endpoints["websocket_user"]) as ws:
            self.ws_user = ws
            await self.subscribe_to_user_stream(ws)
            async for msg in ws:
                # print('debug user stream', msg)
                if msg is None or msg == "pong":
                    continue
                if "type" in msg and "welcome" in msg or "ack" in msg:
                    continue
                try:
                    if self.stop_websocket:
                        break
                    asyncio.create_task(
                        self.on_user_stream_events(
                            self.standardize_user_stream_event(json.loads(msg))
                        )
                    )
                except Exception as e:
                    logging.error(f"error in websocket user stream {e}")
                    traceback.print_exc()

    async def start_websocket_market_stream(self) -> None:
        await self.init_market_stream()
        k = 1
        asyncio.create_task(self.beat_heart_market_stream())
        async with websockets.connect(self.endpoints["websocket_market"]) as ws:
            self.ws_market = ws
            await self.subscribe_to_market_stream(ws)
            async for msg in ws:
                # print('debug market stream', msg)
                if msg is None or msg == "pong":
                    continue
                if "type" in msg and "welcome" in msg or "ack" in msg:
                    continue
                try:
                    if self.stop_websocket:
                        break
                    ticks = self.standardize_market_stream_event(json.loads(msg))
                    if self.process_websocket_ticks:
                        asyncio.create_task(self.on_market_stream_event(ticks))
                    if k % 10 == 0:
                        self.flush_stuck_locks()
                        k = 1
                    k += 1

                except Exception as e:
                    if "success" not in msg:
                        logging.error(f"error in websocket {e} {msg}")

    async def subscribe_to_market_stream(self, ws):
        pass

    async def subscribe_to_user_stream(self, ws):
        pass

    async def update_last_fills_timestamps(self):
        if (
            self.ts_locked["update_last_fills_timestamps"]
            > self.ts_released["update_last_fills_timestamps"]
        ):
            return
        self.ts_locked["update_last_fills_timestamps"] = time.time()
        try:
            fills = await self.fetch_latest_fills()
            keys_done = set()
            all_keys = set(self.last_fills_timestamps)
            for fill in sorted(fills, key=lambda x: x["timestamp"], reverse=True):
                # print("debug fills", fill["custom_id"])
                for key in all_keys - keys_done:
                    if any(k in fill["custom_id"] for k in [key, shorten_custom_id(key)]):
                        self.last_fills_timestamps[key] = fill["timestamp"]
                        keys_done.add(key)
                        if all_keys == keys_done:
                            break
            return True
        except Exception as e:
            logging.error(f"error with update last fills timestamps {e}")
            traceback.print_exc()
            return False
        finally:
            self.ts_released["update_last_fills_timestamps"] = time.time()

    async def start_ohlcv_mode(self):
        logging.info("starting bot...")
        while True:
            now = time.time()
            # print('secs until next', ((now + 60) - now % 60) - now)
            while int(now) % 60 != self.countdown_offset:
                if self.stop_websocket:
                    break
                await asyncio.sleep(0.5)
                now = time.time()
                if self.countdown:
                    print(
                        f"\rcountdown: {((now + 60) - now % 60) - now:.1f} last price: {self.price}      ",
                        end=" ",
                    )
            if self.stop_websocket:
                break
            await asyncio.sleep(1.0)
            await self.on_minute_mark()
            await asyncio.sleep(1.0)

    async def on_minute_mark(self):
        # called each whole minute
        try:
            if self.countdown:
                print("\r", end="")
            if time.time() - self.heartbeat_ts > self.heartbeat_interval_seconds:
                # print heartbeat once an hour
                self.heartbeat_print()
                self.heartbeat_ts = time.time()
            self.prev_price = self.ob[0]
            prev_pos = self.position.copy()
            to_update = [self.update_position(), self.update_open_orders(), self.init_order_book()]
            if self.passivbot_mode == "clock":
                to_update.append(self.update_last_fills_timestamps())
            res = await asyncio.gather(*to_update)
            self.update_emas(self.ob[0], self.prev_price)
            """
            print(self.last_fills_timestamps)
            print(self.emas_long)
            print(self.emas_short)
            orders = self.calc_orders()
            print(orders)
            print(res)
            """
            if not all(res):
                reskeys = ["pos", "open orders", "order book", "last fills"]
                line = "error with "
                for i in range(len(to_update)):
                    if not to_update[i]:
                        line += reskeys[i]
                logging.error(line)
                return
            await self.cancel_and_create()
            if prev_pos["wallet_balance"] != self.position["wallet_balance"]:
                logging.info(
                    f"balance: {round_dynamic(prev_pos['wallet_balance'], 7)}"
                    + f" -> {round_dynamic(self.position['wallet_balance'], 7)}"
                )
            if prev_pos["long"]["size"] != self.position["long"]["size"]:
                plp = prev_pos["long"]["size"], round_(prev_pos["long"]["price"], self.price_step)
                clp = self.position["long"]["size"], round_(
                    self.position["long"]["price"], self.price_step
                )
                self.log_position_long(prev_pos)
            if prev_pos["short"]["size"] != self.position["short"]["size"]:
                psp = prev_pos["short"]["size"], round_(prev_pos["short"]["price"], self.price_step)
                csp = self.position["short"]["size"], round_(
                    self.position["short"]["price"], self.price_step
                )
                self.log_position_short(prev_pos)
        except Exception as e:
            logging.error(f"error on minute mark {e}")
            traceback.print_exc()

    def order_is_valid(self, order: dict) -> bool:

        # perform checks to detect abnormal orders
        # such abnormal orders were observed in bitget bots where short entries exceeded exposure limit

        try:
            order_good = True
            fault = ""
            if order["position_side"] == "long":
                if order["side"] == "buy":
                    max_cost = self.position["wallet_balance"] * self.xk["wallet_exposure_limit"][0]
                    # check if order cost is too big
                    order_cost = qty_to_cost(
                        order["qty"],
                        order["price"],
                        self.xk["inverse"],
                        self.xk["c_mult"],
                    )
                    position_cost = qty_to_cost(
                        self.position["long"]["size"],
                        self.position["long"]["price"],
                        self.xk["inverse"],
                        self.xk["c_mult"],
                    )
                    if order_cost + position_cost > max_cost * 1.2:
                        fault = "Long pos cost would be more than 20% greater than max allowed"
                        order_good = False
                elif order["side"] == "sell":
                    # check if price is above pos price
                    if "n_close" in order["custom_id"]:
                        if order["price"] < self.position["long"]["price"]:
                            fault = "long nclose price below pos price"
                            order_good = False

            elif order["position_side"] == "short":
                max_cost = self.position["wallet_balance"] * self.xk["wallet_exposure_limit"][1]
                if order["side"] == "sell":
                    order_cost = qty_to_cost(
                        order["qty"],
                        order["price"],
                        self.xk["inverse"],
                        self.xk["c_mult"],
                    )
                    position_cost = qty_to_cost(
                        self.position["short"]["size"],
                        self.position["short"]["price"],
                        self.xk["inverse"],
                        self.xk["c_mult"],
                    )
                    if order_cost + position_cost > max_cost * 1.2:
                        fault = "Short pos cost would be more than 20% greater than max allowed"
                        order_good = False
                elif order["side"] == "buy":
                    # check if price is below pos price
                    if "n_close" in order["custom_id"]:
                        if order["price"] > self.position["short"]["price"]:
                            fault = "short nclose price above pos price"
                            order_good = False

            if not order_good:
                logging.error(f"invalid order: {fault} {order}")
                info = {
                    "timestamp": utc_ms(),
                    "date": ts_to_date(utc_ms()),
                    "fault": fault,
                    "order": order,
                    "open_orders": self.open_orders,
                    "position": self.position,
                    "order_book": self.ob,
                    "emas_long": self.emas_long,
                    "emas_short": self.emas_short,
                }
                with open(self.log_filepath, "a") as f:
                    f.write(json.dumps(denumpyize(info)) + "\n")
            return order_good
        except Exception as e:
            logging.error(f"error validating order")
            traceback.print_exc()
            return False


async def start_bot(bot):
    if bot.ohlcv:
        await bot.start_ohlcv_mode()
    else:
        while not bot.stop_websocket:
            try:
                await bot.start_websocket()
            except Exception as e:
                logging.warning(
                    "Websocket connection has been lost, attempting to reinitialize the bot... {e}",
                )
                traceback.print_exc()
                await asyncio.sleep(10)


async def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
    parser.add_argument("user", type=str, help="user/account_name defined in api-keys.json")
    parser.add_argument("symbol", type=str, help="symbol to trade")
    parser.add_argument("live_config_path", type=str, help="live config to use")
    parser.add_argument(
        "-m",
        "--market_type",
        type=str,
        required=False,
        dest="market_type",
        default=None,
        help="specify whether spot or futures (default), overriding value from backtest config",
    )
    parser.add_argument(
        "-gs",
        "--graceful_stop",
        action="store_true",
        help="if passed, set graceful stop to both long and short",
    )
    parser.add_argument(
        "-sm",
        "--short_mode",
        "--short-mode",
        type=str,
        required=False,
        dest="short_mode",
        default=None,
        help="specify one of following short modes: [n (normal), m (manual), gs (graceful_stop), p (panic), t (tp_only)]",
    )
    parser.add_argument(
        "-lm",
        "--long_mode",
        "--long-mode",
        type=str,
        required=False,
        dest="long_mode",
        default=None,
        help="specify one of following long modes: [n (normal), m (manual), gs (graceful_stop), p (panic), t (tp_only)]",
    )
    parser.add_argument(
        "-ab",
        "--assigned_balance",
        type=float,
        required=False,
        dest="assigned_balance",
        default=None,
        help="add assigned_balance to live config, overriding balance fetched from exchange",
    )
    parser.add_argument(
        "-pt",
        "--price-distance-threshold",
        "--price_distance_threshold",
        type=float,
        required=False,
        dest="price_distance_threshold",
        default=0.5,
        help="only create limit orders closer to price than threshold.  default=0.5 (50%)",
    )
    parser.add_argument(
        "-ak",
        "--api-keys",
        "--api_keys",
        type=str,
        required=False,
        dest="api_keys",
        default="api-keys.json",
        help="File containing users/accounts and api-keys for each exchange",
    )
    parser.add_argument(
        "-lev",
        "--leverage",
        type=int,
        required=False,
        dest="leverage",
        default=7,
        help="Leverage set on exchange, if applicable.  Default is 7.",
    )
    parser.add_argument(
        "-tm",
        "--test_mode",
        action="store_true",
        help=f"if true, run on the test net instead of normal exchange. Supported exchanges: {TEST_MODE_SUPPORTED_EXCHANGES}",
    )
    parser.add_argument(
        "-cd",
        "--countdown",
        action="store_true",
        help=f"if true, print a countdown in ohlcv mode",
    )
    parser.add_argument(
        "-pp",
        "--price-precision",
        "--price_precision",
        type=float,
        required=False,
        dest="price_precision_multiplier",
        default=None,
        help="Override price step with round_dynamic(market_price * price_precision, 1).  Suggested val 0.0001",
    )
    parser.add_argument(
        "-ps",
        "--price-step",
        "--price_step",
        type=float,
        required=False,
        dest="price_step_custom",
        default=None,
        help="Override price step with custom price step.  Takes precedence over -pp",
    )
    parser.add_argument(
        "-co",
        "--countdown-offset",
        "--countdown_offset",
        type=int,
        required=False,
        dest="countdown_offset",
        default=random.randrange(60),
        help="when in ohlcv mode, offset execution cycle in seconds from whole minute",
    )
    parser.add_argument(
        "-oh",
        "--ohlcv",
        type=str,
        required=False,
        dest="ohlcv",
        default=None,
        nargs="?",
        const="y",
        help="if no arg or [y/yes], use 1m ohlcv instead of 1s ticks, overriding param ohlcv from config/backtest/default.hjson",
    )

    float_kwargs = [
        ("-lmm", "--long_min_markup", "--long-min-markup", "long_min_markup"),
        ("-smm", "--short_min_markup", "--short-min-markup", "short_min_markup"),
        ("-lmr", "--long_markup_range", "--long-markup-range", "long_markup_range"),
        ("-smr", "--short_markup_range", "--short-markup-range", "short_markup_range"),
        (
            "-lw",
            "--long_wallet_exposure_limit",
            "--long-wallet-exposure-limit",
            "long_wallet_exposure_limit",
        ),
        (
            "-sw",
            "--short_wallet_exposure_limit",
            "--short-wallet-exposure-limit",
            "short_wallet_exposure_limit",
        ),
    ]
    for k0, k1, k2, dest in float_kwargs:
        parser.add_argument(
            k0,
            k1,
            k2,
            type=float,
            required=False,
            dest=dest,
            default=None,
            help=f"specify {dest}, overriding value from live config",
        )

    args = parser.parse_args()
    try:
        exchange = load_exchange_key_secret_passphrase(args.user, args.api_keys)[0]
    except Exception as e:
        logging.error(f"{e} failed to load api-keys.json file")
        return
    try:
        config = load_live_config(args.live_config_path)
    except Exception as e:
        logging.error(f"{e} failed to load config {args.live_config_path}")
        return
    config["exchange"] = exchange
    for k in [
        "user",
        "api_keys",
        "symbol",
        "leverage",
        "price_distance_threshold",
        "test_mode",
        "countdown",
        "countdown_offset",
        "price_precision_multiplier",
        "price_step_custom",
    ]:
        config[k] = getattr(args, k)
    if config["test_mode"] and config["exchange"] not in TEST_MODE_SUPPORTED_EXCHANGES:
        raise IOError(f"Exchange {config['exchange']} is not supported in test mode.")
    config["market_type"] = args.market_type if args.market_type is not None else "futures"
    config["passivbot_mode"] = determine_passivbot_mode(config)
    if config["passivbot_mode"] == "clock":
        config["ohlcv"] = True
    elif hasattr(args, "ohlcv"):
        if args.ohlcv is None:
            config["ohlcv"] = True
        else:
            if args.ohlcv.lower() in ["y", "yes", "t", "true"]:
                config["ohlcv"] = True
            else:
                config["ohlcv"] = False
    else:
        config["ohlcv"] = True
    if args.assigned_balance is not None:
        logging.info(f"assigned balance set to {args.assigned_balance}")
        config["assigned_balance"] = args.assigned_balance

    if args.long_mode is None:
        if config["long"]["enabled"]:
            logging.info("long normal mode")
        else:
            config["long_mode"] = "manual"
            logging.info("long manual mode enabled; will neither cancel nor create long orders")
    else:
        if args.long_mode in ["gs", "graceful_stop", "graceful-stop"]:
            logging.info(
                "long graceful stop enabled; will not make new entries once existing positions are closed"
            )
            config["long"]["enabled"] = config["do_long"] = False
        elif args.long_mode in ["m", "manual"]:
            logging.info("long manual mode enabled; will neither cancel nor create long orders")
            config["long_mode"] = "manual"
        elif args.long_mode in ["n", "normal"]:
            logging.info("long normal mode")
            config["long"]["enabled"] = config["do_long"] = True
        elif args.long_mode in ["p", "panic"]:
            logging.info("long panic mode enabled")
            config["long_mode"] = "panic"
            config["long"]["enabled"] = config["do_long"] = False
        elif args.long_mode.lower() in ["t", "tp_only", "tp-only"]:
            logging.info("long tp only mode enabled")
            config["long_mode"] = "tp_only"

    if args.short_mode is None:
        if config["short"]["enabled"]:
            logging.info("short normal mode")
        else:
            config["short_mode"] = "manual"
            logging.info("short manual mode enabled; will neither cancel nor create short orders")
    else:
        if args.short_mode in ["gs", "graceful_stop", "graceful-stop"]:
            logging.info(
                "short graceful stop enabled; "
                + "will not make new entries once existing positions are closed"
            )
            config["short"]["enabled"] = config["do_short"] = False
        elif args.short_mode in ["m", "manual"]:
            logging.info("short manual mode enabled; will neither cancel nor create short orders")
            config["short_mode"] = "manual"
        elif args.short_mode in ["n", "normal"]:
            logging.info("short normal mode")
            config["short"]["enabled"] = config["do_short"] = True
        elif args.short_mode in ["p", "panic"]:
            logging.info("short panic mode enabled")
            config["short_mode"] = "panic"
            config["short"]["enabled"] = config["do_short"] = False
        elif args.short_mode.lower() in ["t", "tp_only", "tp-only"]:
            logging.info("short tp only mode enabled")
            config["short_mode"] = "tp_only"
    if args.graceful_stop:
        logging.info(
            "\n\ngraceful stop enabled for both long and short; will not make new entries once existing positions are closed\n"
        )
        config["long"]["enabled"] = config["do_long"] = False
        config["short"]["enabled"] = config["do_short"] = False
        config["long_mode"] = None
        config["short_mode"] = None
    for _, _, _, dest in float_kwargs:
        if getattr(args, dest) is not None:
            side, key = dest[: dest.find("_")], dest[dest.find("_") + 1 :]
            old_val = config[side][key]
            config[side][key] = getattr(args, dest)
            logging.info(f"overriding {dest} {old_val} " + f"with new value: {getattr(args, dest)}")

    if "spot" in config["market_type"]:
        config = spotify_config(config)
    logging.info(f"using config \n{config_pretty_str(denumpyize(config))}")

    if config["exchange"] == "binance":
        if "spot" in config["market_type"]:
            from procedures import create_binance_bot_spot

            bot = await create_binance_bot_spot(config)
        else:
            from procedures import create_binance_bot

            bot = await create_binance_bot(config)
    elif config["exchange"] == "binance_us":
        from procedures import create_binance_bot_spot

        bot = await create_binance_bot_spot(config)
    elif config["exchange"] == "bybit":
        from procedures import create_bybit_bot

        bot = await create_bybit_bot(config)
    elif config["exchange"] == "bitget":
        from procedures import create_bitget_bot

        config["ohlcv"] = True
        bot = await create_bitget_bot(config)
    elif config["exchange"] == "okx":
        from procedures import create_okx_bot

        config["ohlcv"] = True
        bot = await create_okx_bot(config)
    elif config["exchange"] == "kucoin":
        from procedures import create_kucoin_bot

        config["ohlcv"] = True
        bot = await create_kucoin_bot(config)
    else:
        raise Exception("unknown exchange", config["exchange"])

    if config["ohlcv"]:
        logging.info(
            "starting passivbot in ohlcv mode, using REST API only and updating once a minute"
        )

    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    await start_bot(bot)
    if hasattr(bot, "session"):
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"There was an error starting the bot: {e}")
        traceback.print_exc()
    finally:
        logging.error("Passivbot was stopped succesfully")
        os._exit(0)
