from typing import Tuple, List

from numba import types, typeof

from definitions.candle import Candle, empty_candle, precompile_candle
from definitions.order import Order, empty_order, empty_order_list, precompile_order, NEW, PARTIALLY_FILLED, FILLED, \
    CANCELED, EXPIRED, LONG, SHORT, NEW_INSURANCE, NEW_ADL
from definitions.order_list import OrderList, precompile_order_list
from definitions.position import Position, empty_long_position, empty_short_position, precompile_position
from definitions.position_list import PositionList, precompile_position_list
from definitions.tick import Tick, empty_tick_list, precompile_tick
from helpers.optimized import prepare_candles, correct_order_float_precision
from helpers.print_functions import print_, print_order

ORDER_UPDATE = 'order'
ACCOUNT_UPDATE = 'account'

base_bot_spec = [
    ("balance", types.float64),
    ("position", typeof(PositionList())),
    ("open_orders", typeof(OrderList())),
    ("quantity_step", types.float64),
    ("minimal_quantity", types.float64),
    ("minimal_cost", types.float64),
    ("price_step", types.float64),
    ("call_interval", types.float64),
    ("tick_interval", types.float64),
    ("leverage", types.float64),
    ("symbol", types.string)
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

        self.quantity_step = 0.0
        self.price_step = 0.0
        self.minimal_quantity = 0.0
        self.minimal_cost = 0.0
        self.call_interval = 1.0
        self.tick_interval = 0.25
        self.leverage = 1.0
        self.symbol = ''

    def init(self):
        """
        Updates the quantity step, price step, and call interval of the strategy.
        :return:
        """
        self.strategy.update_steps(self.quantity_step, self.price_step, self.minimal_quantity, self.minimal_cost,
                                   self.call_interval)
        self.strategy.update_symbol(self.symbol)
        self.strategy.update_leverage(self.leverage)

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
        precompile_tick()
        precompile_candle()
        precompile_order()
        precompile_position()
        precompile_order_list()
        precompile_position_list()
        tick_list = empty_tick_list()
        times = [1, 200, 750]
        tick_interval = 0.25
        for t in range(len(times)):
            tick_list.append(Tick(times[t], t, 1.0, False))
        tick = tick_list[-1]
        max_time = int(tick.timestamp - (tick.timestamp % (tick_interval * 1000))) + int(tick_interval * 1000)
        prepare_candles(tick_list, 0, max_time, empty_candle(), tick_interval)
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
        self.update_position(empty_long_position(), empty_short_position())

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

    def handle_order_update(self, order: Order) -> Order:
        """
        Handles an order update by either deleting, adding, or changing the open orders. If the order was FILLED it
        returns the last filled order, otherwise an empty order.
        :param order: The order to process.
        :return: An empty order with default values or the last filled order.
        """
        last_filled_order = empty_order()
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
            last_filled_order = order
        self.update_orders(add_orders, delete_orders)
        return last_filled_order

    def handle_account_update(self, balance: float, long: Position, short: Position) -> Tuple[
        float, float, PositionList, PositionList]:
        """
        Handles an account update which includes balance and position changes. Returns the old and new balance as well
        as the old and new position list.
        :param balance: The new balance.
        :param long: The new long position.
        :param short: The new short position.
        :return: The old balance, the new balance, the old position, and the new position.
        """
        old_balance = self.get_balance()
        self.update_balance(balance)
        new_balance = self.get_balance()
        old_position = self.get_position()
        self.update_position(long, short)
        new_position = self.get_position()
        return old_balance, new_balance, old_position, new_position

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

    def filter_orders(self, add_orders: List[Order], delete_orders: List[Order], print=False) -> Tuple[
        List[Order], List[Order]]:
        """
        Filters out orders that have either a quantity below the minimum quantity or a cost (price times quantity) that
        is below the minimal cost.
        :param add_orders: Orders to create/send to the exchange.
        :param delete_orders: Orders to cancel/send to the exchange.
        :param print: Whether to print that an order was removed.
        :return: The filtered add and delete orders.
        """
        add_orders_new = empty_order_list()
        delete_orders_new = empty_order_list()

        for order in add_orders:
            if order.quantity < self.minimal_quantity:
                # if print:
                #     print_(["Quantity too small"])
                #     print_order(order)
                pass
            elif order.price != 0.0 and order.price * order.quantity < self.minimal_cost:
                # if print:
                #     print_(["Cost too small"])
                #     print_order(order)
                pass
            elif order.stop_price != 0.0 and order.stop_price * order.quantity < self.minimal_cost:
                # if print:
                #     print_(["Cost too small"])
                #     print_order(order)
                pass
            else:
                add_orders_new.append(order)

        for order in delete_orders:
            if order.quantity < self.minimal_quantity:
                # if print:
                #     print_(["Quantity too small"])
                #     print_order(order)
                pass
            elif order.price != 0.0 and order.price * order.quantity < self.minimal_cost:
                # if print:
                #     print_(["Cost too small"])
                #     print_order(order)
                pass
            elif order.stop_price != 0.0 and order.stop_price * order.quantity < self.minimal_cost:
                # if print:
                #     print_(["Cost too small"])
                #     print_order(order)
                pass
            else:
                delete_orders_new.append(order)

        return add_orders_new, delete_orders_new

    def execute_strategy_order_update(self, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        """
        Executes the order update function of the strategy. Updates all values before the function is called.
        Executes the creation and cancellation of orders and resets order_fill_change and position_change.
        :return:
        """
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        add_orders, delete_orders = self.strategy.on_order_update(last_filled_order)
        add_orders, delete_orders = self.correct_orders(add_orders, delete_orders)
        add_orders, delete_orders = self.filter_orders(add_orders, delete_orders)

        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        return add_orders, delete_orders

    def execute_strategy_account_update(self, old_balance: float, new_balance: float, old_position: PositionList,
                                        new_position: PositionList) -> Tuple[List[Order], List[Order]]:
        """
        Executes the account update function of the strategy. Updates all values before the function is called.
        Executes the creation and cancellation of orders and resets order_fill_change and position_change.
        :return:
        """
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        add_orders, delete_orders = self.strategy.on_account_update(old_balance, new_balance, old_position,
                                                                    new_position)
        add_orders, delete_orders = self.correct_orders(add_orders, delete_orders)
        add_orders, delete_orders = self.filter_orders(add_orders, delete_orders)

        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        return add_orders, delete_orders

    def execute_strategy_decision_making(self, prices: List[Candle]) -> Tuple[List[Order], List[Order]]:
        """
        Executes the decision making function of the strategy. Before, it updates all values of the strategy.
        Executes the creation and cancellation of orders and resets order_fill_change and position_change.
        :param prices:
        :return:
        """
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        add_orders, delete_orders = self.strategy.make_decision(prices)
        add_orders, delete_orders = self.correct_orders(add_orders, delete_orders)
        add_orders, delete_orders = self.filter_orders(add_orders, delete_orders)

        self.cancel_orders(delete_orders)
        self.create_orders(add_orders)
        return add_orders, delete_orders
