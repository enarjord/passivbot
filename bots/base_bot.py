from typing import Tuple, List

from numba import types, typeof

from definitions.candle import Candle
from definitions.order import Order, empty_order_list, NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, LONG, SHORT, \
    NEW_INSURANCE, NEW_ADL
from definitions.order_list import OrderList
from definitions.position import Position
from definitions.position_list import PositionList
from definitions.tick import Tick, empty_tick_list
from helpers.optimized import prepare_candles, correct_order_float_precision
from helpers.print_functions import print_

ORDER_UPDATE = 'order'
ACCOUNT_UPDATE = 'account'

base_bot_spec = [
    ("balance", types.float64),
    ("position", typeof(PositionList())),
    ("open_orders", typeof(OrderList())),
    ("long", types.boolean),
    ("short", types.boolean),
    ("quantity_step", types.float64),
    ("price_step", types.float64),
    ("call_interval", types.float64),
    ("tick_interval", types.float64),
    ("last_filled_order", typeof(Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', ''))),
    ("position_change", types.boolean),
    ("order_fill_change", types.boolean)
]


class Bot:
    """
    Base bot class that contains functions and attributes for a bot.
    """

    def __init__(self):
        """
        Initializes the base attributes of the bot.
        """
        self.balance = 0.0

        self.position = PositionList()
        self.open_orders = OrderList()

        self.long = True
        self.short = False

        self.quantity_step = 0.0
        self.price_step = 0.0
        self.call_interval = 1.0
        self.tick_interval = 0.25

        self.last_filled_order = Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', '')
        self.position_change = False
        self.order_fill_change = False

    def init(self):
        """
        Updates the quantity step, price step, and call interval of the strategy.
        :return:
        """
        self.strategy.update_steps(self.quantity_step, self.price_step, self.call_interval)

    def prepare_order(self, msg) -> Order:
        """
        Function to get an order in the correct format.
        :param msg: Message that needs to be translated.
        :return: An order object.
        """
        raise NotImplementedError

    def prepare_account(self, msg) -> Tuple[float, Position, Position]:
        """
        Function to get an account update in the correct format.
        :param msg: Message that needs to be translated.
        :return: A tuple of balance, long position, and short position.
        """
        raise NotImplementedError

    def prepare_candles(self, tick_list: List[Tick], last_candle_start_time: int, max_candle_start_time: int,
                        last_candle: Candle) -> Tuple[List[Candle], List[Tick], int]:
        """
        Prepares a list of ticks into one or more candles. Fills in gap in candles if update time is longer than the
        tick interval.
        :param tick_list: The list of ticks to aggregate.
        :param last_candle_start_time: The last time this function was called.
        :param max_candle_start_time: The stop time to aggregate. This represents the current, not finished, tick
        interval.
        :param last_candle: The last candle of the last aggregation.
        :return: A list of candles, a list of not yet aggregated ticks, the timestamp of the current candle.
        """
        candle_list, ticks, current_lowest_time = prepare_candles(tick_list, last_candle_start_time,
                                                                  max_candle_start_time, last_candle,
                                                                  self.tick_interval)
        return candle_list, ticks, current_lowest_time

    def prepare_tick(self, msg) -> Tick:
        """
        Function to get a tick update in the correct format.
        :param msg: Message that needs to be translated.
        :return: A tick object.
        """
        raise NotImplementedError

    def update_heartbeat(self):
        """
        Function that triggers an update of the websocket, if needed.
        :return:
        """
        pass

    def determine_update_type(self, msg) -> str:
        """
        Function that determines whether the message is an order or account update.
        :param msg: Message that needs to be identified.
        :return: ORDER_UPDATE or ACCOUNT_UPDATE.
        """
        raise NotImplementedError

    def precompile(self):
        """
        Triggers the compilation of numba classes and functions by calling them. Used to avoid compilation at first use.
        :return:
        """
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
        """
        Resets the bot to an empty state.
        :return:
        """
        self.precompile()
        self.balance = 0
        self.position = PositionList()
        self.open_orders = OrderList()
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

    def init_orders(self):
        """
        Base function to initialize orders.
        :return:
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        self.update_orders(add_orders, delete_orders)

    def init_position(self):
        """
        Base function to initialize positions.
        :return:
        """
        self.update_position(Position('XYZ', 0.0, 0.0, 0.0, 0.0, 1, LONG),
                             Position('XYZ', 0.0, 0.0, 0.0, 0.0, 1, SHORT))

    def init_balance(self):
        """
        Base function to initialize balance.
        :return:
        """
        self.update_balance(0.0)

    def update_orders(self, add_orders: List[Order], delete_orders: List[Order]):
        """
        Sorts the orders into long and short orders and adds/removes them from the current open orders.
        :param add_orders: Orders to add to the open orders.
        :param delete_orders: Orders to remove from the open orders.
        :return:
        """
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
        """
        Updates positions on both sides.
        :param long: The new long position.
        :param short: The new short position.
        :return:
        """
        self.position.update_long(long)
        self.position.update_short(short)

    def update_balance(self, balance: float = None):
        """
        Updates the balance.
        :param balance: The new balance.
        :return:
        """
        if balance:
            self.balance = balance

    def get_orders(self) -> OrderList:
        """
        Returns a copy of the current open orders.
        :return: Copy of the open orders.
        """
        open_orders = self.open_orders.copy()
        return open_orders

    def get_position(self) -> PositionList:
        """
        Returns a copy of the current positions.
        :return: Copy of current positions.
        """
        position = self.position.copy()
        return position

    def get_balance(self) -> float:
        """
        Returns the current balance.
        :return: Current balance.
        """
        balance = self.balance
        return balance

    def handle_order_update(self, order: Order):
        """
        Handles an orders update by either deleting, adding, or changing the open orders. Also sets the attribute
        order_fill_change to True if the order was FILLED and last_filled_order to the processed order.
        :param order: The order to process.
        :return:
        """
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
        """
        Handles an account update which includes balance and position changes. Also sets the attribute position_change
        to True.
        :param balance: The new balance.
        :param long: The new long position.
        :param short: The new short position.
        :return:
        """
        self.update_balance(balance)
        if not self.position.long.equal(long) or not self.position.short.equal(short):
            self.position_change = True
        self.update_position(long, short)

    def create_orders(self, orders_to_create: List[Order]):
        """
        Base function to execute order creation.
        :param orders_to_create: Orders to create/send to the exchange.
        :return:
        """
        pass

    def cancel_orders(self, orders_to_cancel: List[Order]):
        """
        Base function to execute order cancellation.
        :param orders_to_cancel: Orders to cancel/send to the exchange.
        :return:
        """
        pass

    def correct_orders(self, add_orders: List[Order], delete_orders: List[Order]) -> Tuple[List[Order], List[Order]]:
        """
        Corrects the floating point attributes of all orders.
        :param add_orders: Orders to create/send to the exchange.
        :param delete_orders: Orders to cancel/send to the exchange.
        :return: A tuple of two lists of Orders.
        """
        for i in range(len(add_orders)):
            o = correct_order_float_precision(add_orders[i], self.price_step, self.quantity_step)
            add_orders[i] = o
        for i in range(len(delete_orders)):
            o = correct_order_float_precision(delete_orders[i], self.price_step, self.quantity_step)
            delete_orders[i] = o

        return add_orders, delete_orders

    def execute_strategy_update(self) -> Tuple[List[Order], List[Order]]:
        """
        Executes the update function of the strategy. Updates the balance and orders before but not the position to
        give the opportunity of using the change between position in the strategy. Updates all values including the
        position after the strategy update function was called.
        Executes the creation and cancellation of orders and resets order_fill_change and position_change.
        :return:
        """
        self.strategy.update_balance(self.get_balance())
        self.strategy.update_orders(self.get_orders())
        add_orders, delete_orders = self.strategy.on_update(self.get_position(), self.last_filled_order)
        add_orders, delete_orders = self.correct_orders(add_orders, delete_orders)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())
        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        self.order_fill_change = False
        self.position_change = False
        return add_orders, delete_orders

    def decide(self, prices: List[Candle]) -> Tuple[List[Order], List[Order]]:
        """
        Executes the decision making function of the strategy. Afterward, it updates all values of the strategy.
        Executes the creation and cancellation of orders and resets order_fill_change and position_change.
        :param prices:
        :return:
        """
        add_orders, delete_orders = self.strategy.make_decision(self.get_balance(), self.get_position(),
                                                                self.get_orders(), prices)
        add_orders, delete_orders = self.correct_orders(add_orders, delete_orders)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        return add_orders, delete_orders
