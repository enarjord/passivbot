from typing import Tuple, List

import numpy as np
from numba import njit, types, typeof

from definitions.candle import Candle, empty_candle_list
from definitions.order import Order, empty_order_list, NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, LONG, SHORT, \
    NEW_INSURANCE, NEW_ADL
from definitions.order_list import OrderList
from definitions.position import Position
from definitions.position_list import PositionList
from definitions.tick import Tick, empty_tick_list
from functions import print_

ORDER_UPDATE = 'order'
ACCOUNT_UPDATE = 'account'


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_up(n, step, safety_rounding=10) -> float:
    return np.round(np.ceil(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def aggregate_candle(tick_list: List[Tick], candle_list: List[Candle], lowest_time: int, last_candle: Candle,
                     tick_interval: float) -> List[Candle]:
    if tick_list:
        prices = []
        qty = []
        for t in tick_list:
            prices.append(t.price)
            qty.append(t.qty)
        prices = np.asarray(prices)
        qty = np.asarray(qty)
        candle = Candle(lowest_time + int(tick_interval * 1000), prices[0], np.max(prices), np.min(prices), prices[-1],
                        np.sum(qty))
        candle_list.append(candle)
    else:
        if candle_list:
            last_candle = candle_list[-1]
            candle = Candle(lowest_time + int(tick_interval * 1000), last_candle.close, last_candle.close,
                            last_candle.close, last_candle.close, 0.0)
            candle_list.append(candle)
        else:
            candle = Candle(lowest_time + int(tick_interval * 1000), last_candle.close, last_candle.close,
                            last_candle.close, last_candle.close, 0.0)
            candle_list.append(candle)
    return candle_list


@njit
def prepare_candles(ticks: List[Tick], last_update_time: int, max_update_time: int, last_candle: Candle,
                    tick_interval: float) -> Tuple[List[Candle], List[Tick], int]:
    tmp_tick_list = empty_tick_list()
    candle_list = empty_candle_list()
    current_lowest_time = last_update_time
    for tick in ticks:
        if tick.timestamp < (current_lowest_time + tick_interval * 1000):
            tmp_tick_list.append(tick)
        else:
            while (current_lowest_time + tick_interval * 1000 - 1) < int(
                    tick.timestamp - (tick.timestamp % (tick_interval * 1000))) and (
                    current_lowest_time + tick_interval * 1000) < max_update_time:
                candle_list = aggregate_candle(tmp_tick_list, candle_list, current_lowest_time, last_candle,
                                               tick_interval)
                tmp_tick_list = empty_tick_list()
                current_lowest_time += int(tick_interval * 1000)
            tmp_tick_list.append(tick)
    return candle_list, tmp_tick_list, current_lowest_time


base_bot_spec = [
    ("balance", types.float64),
    ("position", typeof(PositionList())),
    ("open_orders", typeof(OrderList())),
    ("long", types.boolean),
    ("short", types.boolean),
    ("qty_step", types.float64),
    ("price_step", types.float64),
    ("call_interval", types.float64),
    ("tick_interval", types.float64),
    ("last_filled_order", typeof(Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', ''))),
    ("position_change", types.boolean),
    ("order_fill_change", types.boolean)
]


class Bot:
    def __init__(self):
        self.balance = 0.0

        self.position = PositionList()
        self.open_orders = OrderList()

        self.long = True
        self.short = False

        self.qty_step = 0.0
        self.price_step = 0.0
        self.call_interval = 1.0
        self.tick_interval = 0.25

        self.last_filled_order = Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', '')
        self.position_change = False
        self.order_fill_change = False

    def init(self):
        self.strategy.update_steps(self.qty_step, self.price_step, self.call_interval)

    def prepare_order(self, msg) -> Order:
        raise NotImplementedError

    def prepare_account(self, msg) -> Tuple[float, Position, Position]:
        raise NotImplementedError

    def prepare_candles(self, ticks: List[Tick], last_update_time: int, max_update_time: int, last_candle: Candle) -> \
            Tuple[List[Candle], List[Tick], int]:
        candle_list, ticks, current_lowest_time = prepare_candles(ticks, last_update_time, max_update_time, last_candle,
                                                                  self.tick_interval)
        return candle_list, ticks, current_lowest_time

    def prepare_tick(self, msg) -> Tick:
        raise NotImplementedError

    def update_heartbeat(self):
        pass

    def determine_update_type(self, msg) -> str:
        raise NotImplementedError

    def precompile(self):
        print_(['Precompiling...'], n=True)
        tick_list = empty_tick_list()
        times = [1, 200, 750]
        tick_interval = 0.25
        for t in range(len(times)):
            tick_list.append(Tick(times[t], t, 1.0, False))
        tick = tick_list[-1]
        max_time = int(tick.timestamp - (tick.timestamp % (tick_interval * 1000))) + int(tick_interval * 1000)
        candle_list, tmp_tick_list, current_lowest_time = prepare_candles(tick_list, 0, max_time,
                                                                          Candle(0, 0.0, 0.0, 0.0, 0.0, 0.0),
                                                                          tick_interval)
        c = Candle(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t = Tick(0, 0.0, 0.0, False)
        p = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        p.equal(p)
        o = Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', '')
        o.equal(o)
        pl = PositionList()
        pl.update_long(p)
        pl.update_short(p)
        pl.copy()
        ol = OrderList()
        ol.add_long(empty_order_list())
        ol.add_short(empty_order_list())
        ol.delete_long(empty_order_list())
        ol.delete_short(empty_order_list())
        ol.update_long(empty_order_list())
        ol.update_short(empty_order_list())
        ol.copy()
        self.strategy.precompile()
        print_(['Precompile finished.'], n=True)

    def reset(self):
        self.precompile()
        self.balance = 0
        self.position = PositionList()
        self.open_orders = OrderList()
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

    def init_orders(self):
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        self.update_orders(add_orders, delete_orders)

    def init_position(self):
        self.update_position(Position('XYZ', 0.0, 0.0, 0.0, 0.0, 1, LONG),
                             Position('XYZ', 0.0, 0.0, 0.0, 0.0, 1, SHORT))

    def init_balance(self):
        self.update_balance(0.0)

    def update_orders(self, add_orders: List[Order], delete_orders: List[Order]):
        add_long = empty_order_list()
        add_short = empty_order_list()
        delete_long = empty_order_list()
        delete_short = empty_order_list()
        for order in delete_orders:
            if order.position_side == LONG:
                delete_long.append(order)
            elif order.position_side == SHORT:
                delete_short.append(order)
        for order in add_orders:
            if order.position_side == LONG:
                add_long.append(order)
            elif order.position_side == SHORT:
                add_short.append(order)
        self.open_orders.delete_long(delete_long)
        self.open_orders.delete_short(delete_short)
        self.open_orders.add_long(add_long)
        self.open_orders.add_short(add_short)

    def update_position(self, long: Position, short: Position):
        self.position.update_long(long)
        self.position.update_short(short)

    def update_balance(self, balance: float = None):
        if balance:
            self.balance = balance

    def get_orders(self):
        open_orders = self.open_orders.copy()
        return open_orders

    def get_position(self):
        position = self.position.copy()
        return position

    def get_balance(self):
        balance = self.balance
        return balance

    def handle_order_update(self, order: Order):
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        if order.action in [CANCELED, FILLED, EXPIRED, NEW_INSURANCE, NEW_ADL]:
            delete_orders.append(order)
        if order.action in [NEW]:
            add_orders.append(order)
        if order.action in [PARTIALLY_FILLED]:
            delete_orders.append(order)
            add_orders.append(order)
        if order.action == FILLED:
            self.last_filled_order = order
            self.order_fill_change = True
        self.update_orders(add_orders, delete_orders)

    def handle_account_update(self, balance: float, long: Position, short: Position):
        self.update_balance(balance)
        if not self.position.long.equal(long) or not self.position.short.equal(short):
            self.position_change = True
        self.update_position(long, short)

    def create_orders(self, orders_to_create: List[Order]):
        pass

    def cancel_orders(self, orders_to_cancel: List[Order]):
        pass

    def correct_float_precision(self, order):
        if not np.isclose(order.price, round_dn(order.price, self.price_step), rtol=1e-60, atol=1e-60):
            if order.price > round_dn(order.price, self.price_step):
                order.price = round_up(order.price, self.price_step)
            else:
                order.price = round_dn(order.price, self.price_step)
        if not np.isclose(order.stop_price, round_dn(order.stop_price, self.price_step), rtol=1e-60, atol=1e-60):
            if order.stop_price > round_dn(order.stop_price, self.price_step):
                order.stop_price = round_up(order.stop_price, self.price_step)
            else:
                order.stop_price = round_dn(order.stop_price, self.price_step)
        if not np.isclose(order.qty, round_dn(order.qty, self.qty_step), rtol=1e-60, atol=1e-60):
            if order.qty > round_dn(order.qty, self.qty_step):
                order.qty = round_up(order.qty, self.qty_step)
            else:
                order.qty = round_dn(order.qty, self.qty_step)
        return order

    def execute_strategy_update(self):
        self.strategy.update_balance(self.get_balance())
        self.strategy.update_orders(self.get_orders())
        add_orders, delete_orders = self.strategy.on_update(self.get_position(), self.last_filled_order)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())
        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        self.order_fill_change = False
        self.position_change = False
        return add_orders, delete_orders

    def decide(self, prices: List[Candle]):
        add_orders, delete_orders = self.strategy.make_decision(self.get_balance(), self.get_position(),
                                                                self.get_orders(), prices)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        return add_orders, delete_orders
