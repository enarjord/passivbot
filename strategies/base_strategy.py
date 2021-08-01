from typing import List, Tuple

from numba import typeof, types
from numba.experimental import jitclass

from definitions.candle import Candle
from definitions.order import Order, empty_order_list
from definitions.order_list import OrderList
from definitions.position_list import PositionList


@jitclass([])
class StrategyConfig:
    """
    Base strategy config. Needs to be implemented by every strategy.
    DO NOT INHERIT FROM IT!
    """

    def __init__(self):
        pass


def convert_dict_to_config(config: dict) -> StrategyConfig:
    """
    Converts the strategy config from a dictionary to a specific strategy config.
    Needs to be implemented by every strategy.
    :param config: The strategy part of the config.
    :return: A jitclass StrategyConfig implemented by the strategy.
    """
    strategy_config = StrategyConfig()
    for key, item in config.items():
        pass
    return strategy_config


base_strategy_spec = [
    # ("config", typeof(StrategyConfig())),
    ("symbol", types.string),
    ("leverage", types.float64),
    ("balance", types.float64),
    ("position", typeof(PositionList())),
    ("open_orders", typeof(OrderList())),
    ("call_interval", types.float64),
    ("quantity_step", types.float64),
    ("price_step", types.float64),
    ("minimal_quantity", types.float64),
    ("minimal_cost", types.float64)
]


class Strategy:
    """
    Base strategy class specifying base functionality like updating values.
    """

    def __init__(self, config):
        """
        Initializes a base strategy with Strategy config and empty values for balance, position, open orders, qty step,
        and price step.
        :param config: Containing the content for the strategy in form of a StrategyConfig.
        """
        self.config = config
        self.symbol = ''
        self.leverage = 1.0
        self.balance = 0.0
        self.position = PositionList()
        self.open_orders = OrderList()
        self.call_interval = 1.0
        self.quantity_step = 0.0
        self.price_step = 0.0
        self.minimal_quantity = 0.0
        self.minimal_cost = 0.0

    def precompile(self):
        """
        Dummy function that should call all functions in a strategy, including external function to force numba to
        compile them. Avoids compilation at first use, which can be during execution of trade logic.
        :return:
        """
        pass

    def update_symbol(self, symbol: str):
        """
        Assigns the symbol to the strategy.
        :param symbol: The symbol to assign.
        :return:
        """
        self.symbol = symbol

    def update_leverage(self, leverage: float):
        """
        Assigns the leverage to the strategy.
        :param leverage: The leverage to assign.
        :return:
        """
        self.leverage = leverage

    def update_steps(self, quantity_step: float, price_step: float, minimal_quantity: float, minimal_cost: float,
                     call_interval: float):
        """
        Assigns the qty and price step to the strategy. Depending on pair and exchange.
        :param quantity_step: The step size of the quantity of an order for this pair and exchange.
        :param price_step: The step size of the price of an order for this pair and exchange.
        :param minimal_quantity: The minimal size of the quantity of an order for this pair and exchange.
        :param minimal_cost: The minimal cost of an order for this pair and exchange.
        :param call_interval: The call interval for the strategy. A multiple of 0.25 seconds.
        :return:
        """
        self.quantity_step = quantity_step
        self.price_step = price_step
        self.minimal_quantity = minimal_quantity
        self.minimal_cost = minimal_cost
        self.call_interval = call_interval

    def update_balance(self, balance: float):
        """
        Updates the balance of the strategy. Is called in the bot when the balance changes.
        :param balance: The new balance.
        :return:
        """
        self.balance = balance

    def update_position(self, position: PositionList):
        """
        Updates the positions of the strategy. Is called in the bot when the position changes.
        :param position: New positions.
        :return:
        """
        self.position.update_long(position.long)
        self.position.update_short(position.short)

    def update_orders(self, orders: OrderList):
        """
        Updates the orders of the strategy. Is called in the bot when the orders change.
        :param orders: New orders.
        :return:
        """
        self.open_orders.update_long(orders.long)
        self.open_orders.update_short(orders.short)

    def update_values(self, balance: float, position: PositionList, orders: OrderList):
        """
        Updates all value of the strategy (balance, positions, orders) at once.
        :param balance: New balance.
        :param position: New positions.
        :param orders: New orders.
        :return:
        """
        self.update_balance(balance)
        self.update_position(position)
        self.update_orders(orders)

    def make_decision(self, balance: float, position: PositionList, orders: OrderList, prices: List[Candle]) -> Tuple[
        List[Order], List[Order]]:
        """
        Base function to make a decision on a price update. Before the function is called, all values of the strategy
        are updated.
        :param balance: Current balance.
        :param position: Current position.
        :param orders: Current orders.
        :param prices: Current price list.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_order_update(self, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        """
        Base function to call when there is an order update. Before the function is called, all values of the strategy
        are updated.
        :param last_filled_order: The last filled order.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_account_update(self, old_balance: float, new_balance: float, old_position: PositionList,
                          new_position: PositionList) -> Tuple[List[Order], List[Order]]:
        """
        Base function to call when there is a position and/or balance update. Before the function is called, all values
        of the strategy are updated.
        :param old_balance: The old balance.
        :param new_balance: The new balance.
        :param old_position: The old position.
        :param new_position: The new position.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders


# Strategy definition used in initializing the backtesting bot.
# Ensure that this is correctly defined.
strategy_definition = Strategy(StrategyConfig())
