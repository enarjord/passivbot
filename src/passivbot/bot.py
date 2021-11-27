from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from typing import Any

from passivbot.datastructures import Fill
from passivbot.datastructures import Order
from passivbot.datastructures import Position
from passivbot.datastructures import StopMode
from passivbot.datastructures import Tick
from passivbot.datastructures.config import NamedConfig
from passivbot.datastructures.runtime import RuntimeFuturesConfig
from passivbot.datastructures.runtime import RuntimeSpotConfig
from passivbot.utils.funcs.njit import calc_diff
from passivbot.utils.funcs.njit import calc_long_close_grid
from passivbot.utils.funcs.njit import calc_long_entry_grid
from passivbot.utils.funcs.njit import calc_upnl
from passivbot.utils.funcs.njit import qty_to_cost
from passivbot.utils.funcs.njit import round_
from passivbot.utils.funcs.njit import round_dynamic
from passivbot.utils.funcs.pure import filter_orders
from passivbot.utils.httpclient import HTTPClient
from passivbot.utils.httpclient import HTTPRequestError

log = logging.getLogger(__name__)


class Bot:
    httpclient: HTTPClient
    rtc: RuntimeFuturesConfig | RuntimeSpotConfig

    def __init__(self, config: NamedConfig):
        self.spot = False
        self.config = config
        self._stop_mode_log_message_iterations = 0
        self.rtc = self.get_initial_runtime_config(config)
        self.__bot_init__()
        self.rtc.long = self.config.long
        if isinstance(self.rtc, RuntimeFuturesConfig):
            self.rtc.short = self.config.short
            self.rtc.max_leverage = 25

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
        self.heartbeat_ts: float = 0.0
        self.listen_key: str | None = None

        self.position: Position = Position()
        self.open_orders: list[Order] = []
        self.fills: list[Fill] = []
        self.ob: list[float] = [0.0, 0.0]

        self.n_orders_per_execution = 2
        self.delay_between_executions = 2
        self.force_update_interval = 30

        self.user_stream_task: asyncio.Task | None = None
        self.market_stream_task: asyncio.Task | None = None

        self.stop_websocket = False
        self.process_websocket_ticks = True

    def __bot_init__(self):
        """
        Subclass initialization routines
        """
        raise NotImplementedError

    @staticmethod
    def get_initial_runtime_config(config: NamedConfig) -> RuntimeFuturesConfig | RuntimeSpotConfig:
        raise NotImplementedError

    @staticmethod
    async def get_httpclient(config: NamedConfig) -> HTTPClient:
        raise NotImplementedError

    async def _init(self):
        await self.init_fills()

    def dump_log(self, data) -> None:
        return
        if self.config["logging_level"] > 0:
            self.log_filepath.write_text(
                json.dumps({**{"log_timestamp": time.time()}, **data}) + "\n"
            )

    async def update_open_orders(self) -> None:
        if self.ts_locked["update_open_orders"] > self.ts_released["update_open_orders"]:
            return
        try:
            open_orders = await self.fetch_open_orders()
            open_orders = [x for x in open_orders if x.symbol == self.config.symbol.name]
            if self.open_orders != open_orders:
                self.dump_log({"log_type": "open_orders", "data": open_orders})
            self.open_orders = open_orders
        except Exception as e:
            log.error("error with update open orders: %s", e, exc_info=True)
        finally:
            self.ts_released["update_open_orders"] = time.time()

    def adjust_wallet_balance(self, balance: float) -> float:
        return (
            balance if self.config.assigned_balance is None else self.config.assigned_balance
        ) * self.config.cross_wallet_pct

    def add_wallet_exposures_to_pos(self, position: Position) -> Position:
        position = position.copy()
        position.long.wallet_exposure = (
            (
                qty_to_cost(
                    position.long.size,
                    position.long.price,
                    self.rtc.inverse,
                    self.rtc.c_mult,
                )
                / position.wallet_balance
            )
            if position.wallet_balance
            else 0.0
        )
        position.short.wallet_exposure = (
            (
                qty_to_cost(
                    position.short.size,
                    position.short.price,
                    self.rtc.inverse,
                    self.rtc.c_mult,
                )
                / position.wallet_balance
            )
            if position.wallet_balance
            else 0.0
        )
        return position

    async def update_position(self) -> None:
        if self.ts_locked["update_position"] > self.ts_released["update_position"]:
            return
        self.ts_locked["update_position"] = time.time()
        try:
            position = await self.fetch_position()
            position.wallet_balance = self.adjust_wallet_balance(position.wallet_balance)
            # isolated equity, not cross equity
            position.equity = position.wallet_balance + calc_upnl(
                position.long.size,
                position.long.price,
                position.short.size,
                position.short.price,
                self.rtc.price,
                self.rtc.inverse,
                self.rtc.c_mult,
            )
            position = self.add_wallet_exposures_to_pos(position)
            if self.position != position:
                if (
                    self.position
                    and "spot" in self.rtc.market_type
                    and (
                        self.position.long.size != position.long.size
                        or self.position.short.size != position.short.size
                    )
                ):
                    # update fills if position size changed
                    await self.update_fills()
                self.dump_log({"log_type": "position", "data": position})
            self.position = position
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
        except Exception as e:
            log.error("error with update position: %s", e, exc_info=True)
        finally:
            self.ts_released["update_position"] = time.time()

    async def init_fills(self, n_days_limit=60):
        self.fills = await self.fetch_fills()

    async def update_fills(self) -> None:
        """
        fetches recent fills
        returns list of new fills
        """
        if self.ts_locked["update_fills"] > self.ts_released["update_fills"]:
            return None
        self.ts_locked["update_fills"] = time.time()
        try:
            fetched = await self.fetch_fills()
            seen = set()
            updated_fills = []
            for fill in fetched + self.fills:
                if fill.order_id not in seen:
                    updated_fills.append(fill)
                    seen.add(fill.order_id)
            self.fills = sorted(updated_fills, key=lambda x: x.order_id)[-5000:]
        except Exception as e:
            log.error("error with update fills: %s", e, exc_info=True)
        finally:
            self.ts_released["update_fills"] = time.time()
        return None

    async def create_orders(self, orders_to_create: list[Order]) -> list[Order]:
        if not orders_to_create:
            return []
        if self.ts_locked["create_orders"] > self.ts_released["create_orders"]:
            return []
        self.ts_locked["create_orders"] = time.time()
        try:
            creations = []
            for oc in sorted(orders_to_create, key=lambda x: calc_diff(x.price, self.rtc.price)):  # type: ignore[no-any-return]
                try:
                    creations.append((oc, asyncio.create_task(self.execute_order(oc))))
                except Exception as e:
                    log.error("error creating order a: %r; error: %s", oc, e)
            created_orders: list[Order] = []
            for oc, c in creations:
                try:
                    o = await c
                    if not o:
                        log.error("error creating order: %r // %s", oc, o)
                        continue
                    created_orders.append(o)
                    log.info("created order: %r", o)
                    if o.order_id not in {x.order_id for x in self.open_orders}:
                        self.open_orders.append(o)
                    self.dump_log({"log_type": "create_order", "data": o})
                except Exception as e:
                    log.error(
                        "error creating order c: %r; error: %s, exc:\n%s", oc, e, c.exception()
                    )
                    self.dump_log(
                        {
                            "log_type": "create_order",
                            "data": {"result": str(c.exception()), "error": repr(e), "data": oc},
                        }
                    )
            return created_orders
        finally:
            self.ts_released["create_orders"] = time.time()

    async def cancel_orders(self, orders_to_cancel: list[Order]) -> list[Order] | None:
        if not orders_to_cancel:
            return None
        if self.ts_locked["cancel_orders"] > self.ts_released["cancel_orders"]:
            return None
        self.ts_locked["cancel_orders"] = time.time()
        try:
            deletions = []
            for oc in orders_to_cancel:
                try:
                    deletions.append((oc, asyncio.create_task(self.execute_cancellation(oc))))
                except Exception as exc:
                    log.error("error cancelling order a:: %r, error: %s", oc, exc, exc_info=True)
            cancelled_orders: list[Order] = []
            for oc, c in deletions:
                try:
                    o = await c
                    if o is None:
                        log.error("error cancelling order: %r", oc)
                        continue
                    cancelled_orders.append(o)
                    log.info("cancelled order: %r", o)
                    self.open_orders = [oo for oo in self.open_orders if oo.order_id != o.order_id]

                except Exception as e:
                    log.error(
                        "error cancelling order b: %r; error: %s, exc:\n%s",
                        oc,
                        e,
                        c.exception(),
                    )
                    self.dump_log(
                        {
                            "log_type": "cancel_order",
                            "data": {"result": str(c.exception()), "error": repr(e), "data": oc},
                        }
                    )
            return cancelled_orders
        finally:
            self.ts_released["cancel_orders"] = time.time()

    async def init_exchange_config(self) -> None:
        raise NotImplementedError

    async def init_order_book(self) -> None:
        raise NotImplementedError

    async def init_market_type(self) -> None:
        raise NotImplementedError

    async def fetch_open_orders(self) -> list[Order]:
        raise NotImplementedError

    async def fetch_fills(
        self,
        symbol: str | None = None,
        limit: int = 1000,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Fill]:
        raise NotImplementedError

    async def fetch_position(self) -> Position:
        raise NotImplementedError

    async def execute_order(self, order: Order) -> Order | None:
        raise NotImplementedError

    async def execute_cancellation(self, order: Order) -> Order | None:
        raise NotImplementedError

    def standardize_user_stream_event(self, event: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def standardize_market_stream_event(self, data: dict[str, Any]) -> list[Tick]:
        raise NotImplementedError

    def stop(self, signum=None, frame=None) -> None:
        log.info("Stopping passivbot, please wait...")
        try:

            self.stop_websocket = True
            if self.user_stream_task is not None:
                self.user_stream_task.cancel()
            if self.market_stream_task is not None:
                self.market_stream_task.cancel()
        except Exception as e:
            log.info("An error occurred during shutdown: %s", e, exc_info=True)

    def pause(self) -> None:
        self.process_websocket_ticks = False

    def resume(self) -> None:
        self.process_websocket_ticks = True

    def calc_orders(self) -> list[Order]:
        balance = self.position.wallet_balance
        long_psize = self.position.long.size
        long_pprice = self.position.long.price
        short_psize = self.position.short.size

        if self.config.stop_mode == StopMode.PANIC:
            if self.config.api_key.exchange == "bybit":
                log.warning("panic mode temporarily disabled for bybit")
                return []
            panic_orders = []
            if long_psize != 0.0:
                panic_orders.append(
                    Order.parse_obj(
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
                )
            if short_psize != 0.0:
                panic_orders.append(
                    Order.parse_obj(
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
                )
            return panic_orders

        if self.rtc.hedge_mode:
            do_long = self.rtc.do_long or long_psize != 0.0
            # do_short = self.rtc.do_short or short_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and short_psize == 0.0
            do_long = (no_pos and self.rtc.do_long) or long_psize != 0.0
            # do_short = (no_pos and self.rtc.do_short) or short_psize != 0.0
        # do_short = self.rtc.do_short = False  # shorts currently disabled for v5
        self.rtc.do_short = False  # shorts currently disabled for v5

        long_entries = calc_long_entry_grid(
            balance,
            long_psize,
            long_pprice,
            self.ob[0],
            self.rtc.inverse,
            do_long,
            self.rtc.qty_step,
            self.rtc.price_step,
            self.rtc.min_qty,
            self.rtc.min_cost,
            self.rtc.c_mult,
            self.rtc.long.grid_span,
            self.rtc.long.wallet_exposure_limit,
            self.rtc.long.max_n_entry_orders,
            self.rtc.long.initial_qty_pct,
            self.rtc.long.eprice_pprice_diff,
            self.rtc.long.secondary_allocation,
            self.rtc.long.secondary_pprice_diff,
            self.rtc.long.eprice_exp_base,
        )
        long_closes = calc_long_close_grid(
            balance,
            long_psize,
            long_pprice,
            self.ob[1],
            self.rtc.spot,
            self.rtc.inverse,
            self.rtc.qty_step,
            self.rtc.price_step,
            self.rtc.min_qty,
            self.rtc.min_cost,
            self.rtc.c_mult,
            self.rtc.long.wallet_exposure_limit,
            self.rtc.long.initial_qty_pct,
            self.rtc.long.min_markup,
            self.rtc.long.markup_range,
            self.rtc.long.n_close_orders,
        )
        orders = [
            Order.parse_obj(
                {
                    "symbol": self.config.symbol.name,
                    "side": "buy",
                    "position_side": "long",
                    "qty": abs(float(o[0])),
                    "price": float(o[1]),
                    "type": "limit",
                    "reduce_only": False,
                    "custom_id": o[2],
                }
            )
            for o in long_entries
            if o[0] > 0.0
        ]
        orders += [
            Order.parse_obj(
                {
                    "symbol": self.config.symbol.name,
                    "side": "sell",
                    "position_side": "long",
                    "qty": abs(float(o[0])),
                    "price": float(o[1]),
                    "type": "limit",
                    "reduce_only": True,
                    "custom_id": o[2],
                }
            )
            for o in long_closes
            if o[0] < 0.0
        ]
        return sorted(orders, key=lambda x: calc_diff(x.price, self.rtc.price))  # type: ignore[no-any-return]

    async def cancel_and_create(self) -> None:
        if self.ts_locked["cancel_and_create"] > self.ts_released["cancel_and_create"]:
            return
        self.ts_locked["cancel_and_create"] = time.time()
        try:
            to_cancel, to_create = filter_orders(
                self.open_orders, self.calc_orders(), keys=["side", "position_side", "qty", "price"]
            )
            to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x.price, self.rtc.price))  # type: ignore[no-any-return]
            to_create = sorted(to_create, key=lambda x: calc_diff(x.price, self.rtc.price))  # type: ignore[no-any-return]
            if self.config.stop_mode != StopMode.MANUAL:
                if to_cancel:
                    # to avoid building backlog, cancel n+1 orders, create n orders
                    asyncio.create_task(
                        self.cancel_orders(to_cancel[: self.n_orders_per_execution + 1])
                    )
                    await asyncio.sleep(
                        0.01
                    )  # sleep 10 ms between sending cancellations and sending creations
                if to_create:
                    await self.create_orders(to_create[: self.n_orders_per_execution])
            await asyncio.sleep(self.delay_between_executions)  # sleep before releasing lock
        finally:
            self.ts_released["cancel_and_create"] = time.time()

    async def on_market_stream_event(self, ticks: list[Tick]):
        if ticks:
            for tick in ticks:
                if tick.is_buyer_maker:
                    self.ob[0] = tick.price
                else:
                    self.ob[1] = tick.price
            self.rtc.price = ticks[-1].price

        if self.config.stop_mode and self.config.stop_mode != StopMode.NORMAL:
            if not self._stop_mode_log_message_iterations:
                log.info("%s stop mode is active", self.config.stop_mode)
                self._stop_mode_log_message_iterations = 100
            else:
                self._stop_mode_log_message_iterations -= 1

        now = time.time()
        if now - self.ts_released["force_update"] > self.force_update_interval:
            self.ts_released["force_update"] = now
            # force update pos and open orders thru rest API every 30 sec
            await asyncio.gather(self.update_position(), self.update_open_orders())
        if now - self.ts_released["print"] >= 0.5:
            self.update_output_information()
        if now - self.heartbeat_ts > 60 * 60:
            # print heartbeat once an hour
            log.info("heartbeat")
            self.heartbeat_ts = time.time()
        await self.cancel_and_create()

    async def on_user_stream_event(self, event: dict[str, Any]) -> None:
        try:
            pos_change = False
            if "wallet_balance" in event:
                self.position.wallet_balance = self.adjust_wallet_balance(event["wallet_balance"])
                pos_change = True
            if "long_psize" in event:
                self.position.long.size = event["long_psize"]
                self.position.long.price = event["long_pprice"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
            if "short_psize" in event:
                self.position.short.size = event["short_psize"]
                self.position.short.price = event["short_pprice"]
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
            if "new_open_order" in event:
                if event["new_open_order"]["order_id"] not in {
                    x.order_id for x in self.open_orders
                }:
                    self.open_orders.append(
                        Order.from_binance_payload(event["new_open_order"], futures=True)
                    )
            if "deleted_order_id" in event:
                self.open_orders = [
                    oo for oo in self.open_orders if oo.order_id != event["deleted_order_id"]
                ]
            if "partially_filled" in event:
                await self.update_open_orders()
            if pos_change:
                self.position.equity = self.position.wallet_balance + calc_upnl(
                    self.position.long.size,
                    self.position.long.price,
                    self.position.short.size,
                    self.position.short.price,
                    self.rtc.price,
                    self.rtc.inverse,
                    self.rtc.c_mult,
                )
                await asyncio.sleep(
                    0.01
                )  # sleep 10 ms to catch both pos update and open orders update
                await self.cancel_and_create()
        except Exception as e:
            log.error("error handling user stream event: %s", e, exc_info=True)

    def update_output_information(self):
        self.ts_released["print"] = time.time()
        line = f"{self.config.symbol.name} "
        line += f"l {self.position.long.size} @ "
        line += f"{round_(self.position.long.price, self.rtc.price_step)}, "
        long_closes = sorted(
            (o for o in self.open_orders if o.side == "sell" and o.position_side == "long"),
            key=lambda x: x.price,
        )
        long_entries = sorted(
            (o for o in self.open_orders if o.side == "buy" and o.position_side == "long"),
            key=lambda x: x.price,
        )
        leqty, leprice = (
            (long_entries[-1].qty, long_entries[-1].price) if long_entries else (0.0, 0.0)
        )
        lcqty, lcprice = (long_closes[0].qty, long_closes[0].price) if long_closes else (0.0, 0.0)
        line += f"e {leqty} @ {leprice}, c {lcqty} @ {lcprice} "
        if self.position.long.size > abs(self.position.short.size):
            liq_price = self.position.long.liquidation_price
        else:
            liq_price = self.position.short.liquidation_price
        line += f"|| last {self.rtc.price} "
        line += f"pprc diff {calc_diff(self.position.long.price, self.rtc.price):.3f} "
        line += f"liq {round_dynamic(liq_price, 5)} "
        line += f"wallet_exposure {self.position.long.wallet_exposure:.3f} "
        line += f"bal {round_dynamic(self.position.wallet_balance, 5)} "
        line += f"eq {round_dynamic(self.position.equity, 5)} "
        log.info(line, wipe_line=True)

    def flush_stuck_locks(self, timeout: float = 5.0) -> None:
        now = time.time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    log.info("flushing stuck lock: %s", key)
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

    async def beat_heart_user_stream(self, ws) -> None:
        pass

    async def init_user_stream(self) -> None:
        pass

    async def start_websocket_user_stream(self) -> None:
        await self.init_user_stream()
        log.info("Websocket stream URL: %s", self.httpclient.endpoints["websocket_user"])
        async with self.httpclient.ws_connect("websocket_user") as ws:
            asyncio.create_task(self.beat_heart_user_stream(ws))
            await self.subscribe_to_user_stream(ws)
            async for msg in ws:
                if msg is None:
                    continue
                try:
                    if self.stop_websocket:
                        break
                    event: dict[str, Any] = self.standardize_user_stream_event(json.loads(msg))
                    asyncio.create_task(self.on_user_stream_event(event))
                except Exception as e:
                    log.error("error in websocket user stream: %s", e, exc_info=True)

    async def start_websocket_market_stream(self) -> None:
        k = 1
        async with self.httpclient.ws_connect("websocket_market") as ws:
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
                        log.error("error in websocket: %s message: %s", e, msg, exc_info=True)

    async def subscribe_to_market_stream(self, ws):
        pass

    async def subscribe_to_user_stream(self, ws):
        pass


async def start_bot(bot):
    while not bot.stop_websocket:
        try:
            await bot.start_websocket()
        except Exception as e:
            log.error(
                "Websocket connection has been lost, attempting to reinitialize the bot: %s",
                e,
                exc_info=True,
            )
            await asyncio.sleep(10)


async def _main(config: NamedConfig) -> None:

    from passivbot.exchanges.bybit import Bybit
    from passivbot.exchanges.binance import BinanceBot
    from passivbot.exchanges.binance_spot import BinanceBotSpot

    bot: BinanceBot | BinanceBotSpot | Bybit

    if config.api_key.exchange == "binance":
        if config.market_type == "spot":
            from passivbot.utils.procedures import create_binance_bot_spot

            bot = await create_binance_bot_spot(config)
        else:
            from passivbot.utils.procedures import create_binance_bot

            bot = await create_binance_bot(config)
    elif config.api_key.exchange == "bybit":
        from passivbot.utils.procedures import create_bybit_bot

        bot = await create_bybit_bot(config)

    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    await start_bot(bot)
    await bot.httpclient.close()


def main(config: NamedConfig) -> None:
    try:
        asyncio.run(_main(config))
    except Exception as e:
        log.error("There was an error starting the bot: %s", e, exc_info=True)
    finally:
        log.info("Passivbot was stopped successfully")
        os._exit(0)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("user", type=str, help="user/account_name defined in api-keys.json")
    parser.add_argument("symbol", type=str, help="symbol to trade")
    parser.add_argument("live_config_path", type=str, help="live config to use")
    parser.add_argument(
        "-gs",
        "--graceful_stop",
        "--graceful-stop",
        action="store_true",
        help="if true, disable long and short",
    )
    parser.add_argument(
        "-lw",
        "--long_wallet_exposure_limit",
        "--long-wallet-exposure-limit",
        type=float,
        required=False,
        dest="long_wallet_exposure_limit",
        default=None,
        help="specify long wallet exposure limit, overriding value from live config",
    )
    parser.add_argument(
        "-sw",
        "--short_wallet_exposure_limit",
        "--short-wallet-exposure-limit",
        type=float,
        required=False,
        dest="short_wallet_exposure_limit",
        default=None,
        help="specify short wallet exposure limit, overriding value from live config",
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
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: NamedConfig
) -> None:

    if args.assigned_balance is not None:
        log.info("assigned balance set to: %s", args.assigned_balance)
        config.assigned_balance = args.assigned_balance

    if args.graceful_stop:
        log.info(
            "Graceful stop enabled, will not make new entries once existing positions are closed"
        )
        config.stop_mode = StopMode.GRACEFUL
        config.long.enabled = False
        config.short.enabled = False

    if args.long_wallet_exposure_limit is not None:
        log.info(
            f"overriding long wallet exposure limit ({config.long.wallet_exposure_limit}) "
            f"with new value: {args.long_wallet_exposure_limit}"
        )
        config.long.wallet_exposure_limit = args.long_wallet_exposure_limit
    if args.short_wallet_exposure_limit is not None:
        log.info(
            f"overriding short wallet exposure limit ({config.short.wallet_exposure_limit}) "
            f"with new value: {args.short_wallet_exposure_limit}"
        )
        config.short.wallet_exposure_limit = args.short_wallet_exposure_limit
