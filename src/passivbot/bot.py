import argparse
import asyncio
import json
import os
import pathlib
import pprint
import signal
import time
import traceback
from typing import Dict
from typing import List
from typing import Union

import websockets

from passivbot.utils.funcs.njit import calc_diff
from passivbot.utils.funcs.njit import calc_long_close_grid
from passivbot.utils.funcs.njit import calc_long_entry_grid
from passivbot.utils.funcs.njit import calc_upnl
from passivbot.utils.funcs.njit import qty_to_cost
from passivbot.utils.funcs.njit import round_
from passivbot.utils.funcs.njit import round_dynamic
from passivbot.utils.funcs.pure import create_xk
from passivbot.utils.funcs.pure import denumpyize
from passivbot.utils.funcs.pure import filter_orders
from passivbot.utils.funcs.pure import spotify_config
from passivbot.utils.procedures import load_exchange_key_secret
from passivbot.utils.procedures import load_live_config
from passivbot.utils.procedures import print_

SUBPARSER_NAME: str = "live"

log = logging.getLogger(__name__)


class Bot:
    def __init__(self, config: dict):
        self.spot = False
        self.config = config
        self.basedir: pathlib.Path = config["basedir"]
        self.config["do_long"] = config["long"]["enabled"]
        self.config["do_short"] = config["short"]["enabled"]
        self.config["max_leverage"] = 25
        self.xk = {}

        self.ws = None

        self.hedge_mode = self.config["hedge_mode"] = True
        self.set_config(self.config)

        self.ts_locked = {
            k: 0.0
            for k in [
                "cancel_orders",
                "update_open_orders",
                "cancel_and_create",
                "update_position",
                "print",
                "create_orders",
                "check_fills",
                "update_fills",
                "force_update",
            ]
        }
        self.ts_released = {k: 1.0 for k in self.ts_locked}
        self.heartbeat_ts = 0
        self.listen_key = None

        self.position = {}
        self.open_orders = []
        self.fills = []
        self.price = 0.0
        self.ob = [0.0, 0.0]

        self.n_orders_per_execution = 2
        self.delay_between_executions = 2
        self.force_update_interval = 30

        self.c_mult = self.config["c_mult"] = 1.0

        self.log_filepath = self.basedir / "logs" / self.exchange / f"{config['config_name']}.log"
        self.log_filepath.parent.mkdir(parents=True, exist_ok=True)

        _, self.key, self.secret = load_exchange_key_secret(self.user)

        self.log_level = 0

        self.user_stream_task = None
        self.market_stream_task = None

        self.stop_websocket = False
        self.process_websocket_ticks = True

    def set_config(self, config):
        if "stop_mode" not in config:
            config["stop_mode"] = None
        if "last_price_diff_limit" not in config:
            config["last_price_diff_limit"] = 0.3
        if "profit_trans_pct" not in config:
            config["profit_trans_pct"] = 0.0
        if "assigned_balance" not in config:
            config["assigned_balance"] = None
        if "cross_wallet_pct" not in config:
            config["cross_wallet_pct"] = 1.0
        if config["cross_wallet_pct"] > 1.0 or config["cross_wallet_pct"] <= 0.0:
            print(
                f'Invalid cross_wallet_pct given: {config["cross_wallet_pct"]}.  It must be greater'
                " than zero and less than or equal to one.  Defaulting to 1.0."
            )
            config["cross_wallet_pct"] = 1.0
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
        await self.init_fills()

    def dump_log(self, data) -> None:
        if self.config["logging_level"] > 0:
            self.log_filepath.write_text(
                json.dumps({**{"log_timestamp": time.time()}, **data}) + "\n"
            )

    async def update_open_orders(self) -> None:
        if self.ts_locked["update_open_orders"] > self.ts_released["update_open_orders"]:
            return
        try:
            open_orders = await self.fetch_open_orders()
            open_orders = [x for x in open_orders if x["symbol"] == self.symbol]
            if self.open_orders != open_orders:
                self.dump_log({"log_type": "open_orders", "data": open_orders})
            self.open_orders = open_orders
        except Exception as e:
            print("error with update open orders", e)
            traceback.print_exc()
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
        except Exception as e:
            print("error with update position", e)
            traceback.print_exc()
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
        except Exception as e:
            print("error with update fills", e)
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
            creations = []
            for oc in sorted(orders_to_create, key=lambda x: calc_diff(x["price"], self.price)):
                try:
                    creations.append((oc, asyncio.create_task(self.execute_order(oc))))
                except Exception as e:
                    print_(["error creating order a", oc, e], n=True)
            created_orders = []
            for oc, c in creations:
                try:
                    o = await c
                    created_orders.append(o)
                    if "side" in o:
                        print_(
                            [
                                "  created order",
                                o["symbol"],
                                o["side"],
                                o["position_side"],
                                o["qty"],
                                o["price"],
                            ],
                            n=True,
                        )
                        if o["order_id"] not in {x["order_id"] for x in self.open_orders}:
                            self.open_orders.append(o)
                    else:
                        print_(["error creating order b", o, oc], n=True)
                    self.dump_log({"log_type": "create_order", "data": o})
                except Exception as e:
                    print_(["error creating order c", oc, c.exception(), e], n=True)
                    self.dump_log(
                        {
                            "log_type": "create_order",
                            "data": {"result": str(c.exception()), "error": repr(e), "data": oc},
                        }
                    )
            return created_orders
        finally:
            self.ts_released["create_orders"] = time.time()

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if not orders_to_cancel:
            return
        if self.ts_locked["cancel_orders"] > self.ts_released["cancel_orders"]:
            return
        self.ts_locked["cancel_orders"] = time.time()
        try:
            deletions = []
            for oc in orders_to_cancel:
                try:
                    deletions.append((oc, asyncio.create_task(self.execute_cancellation(oc))))
                except Exception as e:
                    print_(["error cancelling order a", oc, e])
            cancelled_orders = []
            for oc, c in deletions:
                try:
                    o = await c
                    cancelled_orders.append(o)
                    if "order_id" in o:
                        print_(
                            [
                                "cancelled order",
                                o["symbol"],
                                o["side"],
                                o["position_side"],
                                o["qty"],
                                o["price"],
                            ],
                            n=True,
                        )
                        self.open_orders = [
                            oo for oo in self.open_orders if oo["order_id"] != o["order_id"]
                        ]

                    else:
                        print_(["error cancelling order", o], n=True)
                    self.dump_log({"log_type": "cancel_order", "data": o})
                except Exception as e:
                    print_(["error cancelling order b", oc, c.exception(), e], n=True)
                    self.dump_log(
                        {
                            "log_type": "cancel_order",
                            "data": {"result": str(c.exception()), "error": repr(e), "data": oc},
                        }
                    )
            return cancelled_orders
        finally:
            self.ts_released["cancel_orders"] = time.time()

    def stop(self, signum=None, frame=None) -> None:
        print("\nStopping passivbot, please wait...")
        try:

            self.stop_websocket = True
            self.user_stream_task.cancel()
            self.market_stream_task.cancel()
        except Exception as e:
            print(f"An error occurred during shutdown: {e}")

    def pause(self) -> None:
        self.process_websocket_ticks = False

    def resume(self) -> None:
        self.process_websocket_ticks = True

    def calc_orders(self):
        balance = self.position["wallet_balance"]
        long_psize = self.position["long"]["size"]
        long_pprice = self.position["long"]["price"]
        short_psize = self.position["short"]["size"]

        if self.stop_mode in ["panic"]:
            if self.exchange == "bybit":
                print("\n\npanic mode temporarily disabled for bybit\n\n")
                return []
            panic_orders = []
            if long_psize != 0.0:
                panic_orders.append(
                    {
                        "side": "sell",
                        "position_side": "long",
                        "qty": abs(long_psize),
                        "price": self.ob[1],
                        "type": "market",
                        "reduce_only": True,
                        "custom_id": "long_panic",
                    }
                )
            if short_psize != 0.0:
                panic_orders.append(
                    {
                        "side": "buy",
                        "position_side": "short",
                        "qty": abs(short_psize),
                        "price": self.ob[0],
                        "type": "market",
                        "reduce_only": True,
                        "custom_id": "short_panic",
                    }
                )
            return panic_orders

        if self.hedge_mode:
            do_long = self.do_long or long_psize != 0.0
            do_short = self.do_short or short_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and short_psize == 0.0
            do_long = (no_pos and self.do_long) or long_psize != 0.0
            do_short = (no_pos and self.do_short) or short_psize != 0.0
        do_short = self.do_short = False  # shorts currently disabled for v5
        self.xk["do_long"] = do_long
        self.xk["do_short"] = do_short

        long_entries = calc_long_entry_grid(
            balance,
            long_psize,
            long_pprice,
            self.ob[0],
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
            self.xk["eprice_pprice_diff"][0],
            self.xk["secondary_allocation"][0],
            self.xk["secondary_pprice_diff"][0],
            self.xk["eprice_exp_base"][0],
        )
        long_closes = calc_long_close_grid(
            balance,
            long_psize,
            long_pprice,
            self.ob[1],
            self.xk["spot"],
            self.xk["inverse"],
            self.xk["qty_step"],
            self.xk["price_step"],
            self.xk["min_qty"],
            self.xk["min_cost"],
            self.xk["c_mult"],
            self.xk["wallet_exposure_limit"][0],
            self.xk["initial_qty_pct"][0],
            self.xk["min_markup"][0],
            self.xk["markup_range"][0],
            self.xk["n_close_orders"][0],
        )
        orders = [
            {
                "side": "buy",
                "position_side": "long",
                "qty": abs(float(o[0])),
                "price": float(o[1]),
                "type": "limit",
                "reduce_only": False,
                "custom_id": o[2],
            }
            for o in long_entries
            if o[0] > 0.0
        ]
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
            for o in long_closes
            if o[0] < 0.0
        ]
        return sorted(orders, key=lambda x: calc_diff(x["price"], self.price))

    async def cancel_and_create(self):
        if self.ts_locked["cancel_and_create"] > self.ts_released["cancel_and_create"]:
            return
        self.ts_locked["cancel_and_create"] = time.time()
        try:
            to_cancel, to_create = filter_orders(
                self.open_orders, self.calc_orders(), keys=["side", "position_side", "qty", "price"]
            )
            to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x["price"], self.price))
            to_create = sorted(to_create, key=lambda x: calc_diff(x["price"], self.price))
            results = []
            if self.stop_mode not in ["manual"]:
                if to_cancel:
                    # to avoid building backlog, cancel n+1 orders, create n orders
                    results.append(
                        asyncio.create_task(
                            self.cancel_orders(to_cancel[: self.n_orders_per_execution + 1])
                        )
                    )
                    await asyncio.sleep(
                        0.01
                    )  # sleep 10 ms between sending cancellations and sending creations
                if to_create:
                    results.append(
                        await self.create_orders(to_create[: self.n_orders_per_execution])
                    )
            if any(results):
                print()
            await asyncio.sleep(self.delay_between_executions)  # sleep before releasing lock
            return results
        finally:
            self.ts_released["cancel_and_create"] = time.time()

    async def on_market_stream_event(self, ticks: [dict]):
        if ticks:
            for tick in ticks:
                if tick["is_buyer_maker"]:
                    self.ob[0] = tick["price"]
                else:
                    self.ob[1] = tick["price"]
            self.price = ticks[-1]["price"]

        if self.stop_mode is not None:
            print(f"{self.stop_mode} stop mode is active")

        now = time.time()
        if now - self.ts_released["force_update"] > self.force_update_interval:
            self.ts_released["force_update"] = now
            # force update pos and open orders thru rest API every 30 sec
            await asyncio.gather(self.update_position(), self.update_open_orders())
        if now - self.ts_released["print"] >= 0.5:
            self.update_output_information()
        if now - self.heartbeat_ts > 60 * 60:
            # print heartbeat once an hour
            print_(["heartbeat\n"], n=True)
            self.heartbeat_ts = time.time()
        await self.cancel_and_create()

    async def on_user_stream_events(self, events: Union[List[Dict], List]) -> None:
        if isinstance(events, list):
            for event in events:
                await self.on_user_stream_event(event)
        else:
            await self.on_user_stream_event(events)

    async def on_user_stream_event(self, event: dict) -> None:
        try:
            pos_change = False
            if "wallet_balance" in event:
                self.position["wallet_balance"] = self.adjust_wallet_balance(
                    event["wallet_balance"]
                )
                pos_change = True
            if "long_psize" in event:
                self.position["long"]["size"] = event["long_psize"]
                self.position["long"]["price"] = event["long_pprice"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
            if "short_psize" in event:
                self.position["short"]["size"] = event["short_psize"]
                self.position["short"]["price"] = event["short_pprice"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
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
            print(["error handling user stream event", e])
            traceback.print_exc()

    def update_output_information(self):
        self.ts_released["print"] = time.time()
        line = f"{self.symbol} "
        line += f"l {self.position['long']['size']} @ "
        line += f"{round_(self.position['long']['price'], self.price_step)}, "
        long_closes = sorted(
            (o for o in self.open_orders if o["side"] == "sell" and o["position_side"] == "long"),
            key=lambda x: x["price"],
        )
        long_entries = sorted(
            (o for o in self.open_orders if o["side"] == "buy" and o["position_side"] == "long"),
            key=lambda x: x["price"],
        )
        leqty, leprice = (
            (long_entries[-1]["qty"], long_entries[-1]["price"]) if long_entries else (0.0, 0.0)
        )
        lcqty, lcprice = (
            (long_closes[0]["qty"], long_closes[0]["price"]) if long_closes else (0.0, 0.0)
        )
        line += f"e {leqty} @ {leprice}, c {lcqty} @ {lcprice} "
        if self.position["long"]["size"] > abs(self.position["short"]["size"]):
            liq_price = self.position["long"]["liquidation_price"]
        else:
            liq_price = self.position["short"]["liquidation_price"]
        line += f"|| last {self.price} "
        line += f"pprc diff {calc_diff(self.position['long']['price'], self.price):.3f} "
        line += f"liq {round_dynamic(liq_price, 5)} "
        line += f"lwallet_exposure {self.position['long']['wallet_exposure']:.3f} "
        line += f"bal {round_dynamic(self.position['wallet_balance'], 5)} "
        line += f"eq {round_dynamic(self.position['equity'], 5)} "
        print_([line], r=True)

    def flush_stuck_locks(self, timeout: float = 5.0) -> None:
        now = time.time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print("flushing stuck lock", key)
                    self.ts_released[key] = now

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        self.process_websocket_ticks = True
        await asyncio.gather(self.update_position(), self.update_open_orders())
        await self.init_exchange_config()
        await self.init_order_book()
        self.user_stream_task = asyncio.create_task(self.start_websocket_user_stream())
        self.market_stream_task = asyncio.create_task(self.start_websocket_market_stream())
        await asyncio.gather(self.user_stream_task, self.market_stream_task)

    async def beat_heart_user_stream(self) -> None:
        pass

    async def init_user_stream(self) -> None:
        pass

    async def start_websocket_user_stream(self) -> None:
        await self.init_user_stream()
        asyncio.create_task(self.beat_heart_user_stream())
        print_(["url", self.endpoints["websocket_user"]])
        async with websockets.connect(self.endpoints["websocket_user"]) as ws:
            self.ws = ws
            await self.subscribe_to_user_stream(ws)
            async for msg in ws:
                if msg is None:
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
                    print(["error in websocket user stream", e])
                    traceback.print_exc()

    async def start_websocket_market_stream(self) -> None:
        k = 1
        async with websockets.connect(self.endpoints["websocket_market"]) as ws:
            await self.subscribe_to_market_stream(ws)
            async for msg in ws:
                if msg is None:
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
                        print("error in websocket", e, msg)

    async def subscribe_to_market_stream(self, ws):
        pass

    async def subscribe_to_user_stream(self, ws):
        pass


async def start_bot(bot):
    while not bot.stop_websocket:
        try:
            await bot.start_websocket()
        except Exception as e:
            print("Websocket connection has been lost, attempting to reinitialize the bot...", e)
            traceback.print_exc()
            await asyncio.sleep(10)


async def _main(args: argparse.Namespace) -> None:
    try:
        accounts = json.load(args.api_keys.open())
    except FileNotFoundError:
        log.error("The '%s' file does not exist.", args.api_keys)
    except ValueError as exc:
        log.error("Failed to load JSON from '%s': %s", args.api_keys, exc)
        return
    except Exception:
        log.error("Failed to load %s", args.api_keys, exc_info=True)
        return
    try:
        account = accounts[args.user]
    except Exception as e:
        print("unrecognized account name", args.user, e)
        return
    try:
        config = load_live_config(args.live_config_path)
    except Exception as e:
        print(e, "failed to load config", args.live_config_path)
        return
    config["basedir"] = args.basedir
    config["user"] = args.user
    config["exchange"] = account["exchange"]
    config["symbol"] = args.symbol
    config["live_config_path"] = args.live_config_path
    config["market_type"] = args.market_type if args.market_type is not None else "futures"
    if args.assigned_balance is not None:
        print(f"\nassigned balance set to {args.assigned_balance}\n")
        config["assigned_balance"] = args.assigned_balance

    if args.graceful_stop:
        print(
            "\n\ngraceful stop enabled, will not make new entries once existing positions are"
            " closed\n"
        )
        config["long"]["enabled"] = config["do_long"] = False
        config["short"]["enabled"] = config["do_short"] = False

    if "spot" in config["market_type"]:
        config = spotify_config(config)

    if account["exchange"] == "binance":
        if "spot" in config["market_type"]:
            from passivbot.utils.procedures import create_binance_bot_spot

            bot = await create_binance_bot_spot(config)
        else:
            from passivbot.utils.procedures import create_binance_bot

            bot = await create_binance_bot(config)
    elif account["exchange"] == "bybit":
        from passivbot.utils.procedures import create_bybit_bot

        bot = await create_bybit_bot(config)
    else:
        raise Exception("unknown exchange", account["exchange"])

    print("using config")
    pprint.pprint(denumpyize(config))

    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    await start_bot(bot)
    await bot.session.close()


def main(args: argparse.Namespace) -> None:
    try:
        asyncio.run(_main(args))
    except Exception as e:
        print(f"\nThere was an error starting the bot: {e}")
        traceback.print_exc()
    finally:
        print("\nPassivbot was stopped succesfully")
        os._exit(0)


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        SUBPARSER_NAME, help="Run PassivBot Live"
    )
    parser.add_argument("user", type=str, help="user/account_name defined in api-keys.json")
    parser.add_argument("symbol", type=str, help="symbol to trade")
    parser.add_argument("live_config_path", type=str, help="live config to use")
    parser.add_argument(
        "-m",
        "--market_type",
        "--market-type",
        type=str,
        required=False,
        dest="market_type",
        default=None,
        help="specify whether spot or futures (default), overriding value from backtest config",
    )
    parser.add_argument(
        "-gs",
        "--graceful_stop",
        "--graceful-stop",
        action="store_true",
        help="if true, disable long and short",
    )
    parser.add_argument(
        "-ab",
        "--assigned_balance",
        "--assigned-balance",
        type=float,
        required=False,
        dest="assigned_balance",
        default=None,
        help="add assigned_balance to live config",
    )
    parser.set_defaults(func=main)


def validate_argparse_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    pass
