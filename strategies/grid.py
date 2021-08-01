from typing import List, Tuple

import numpy as np
from numba import types, njit, typeof
from numba.experimental import jitclass

from definitions.candle import Candle, empty_candle_list
from definitions.order import Order, empty_order, empty_order_list, TP, SELL, LONG, LIMIT, BUY, FILLED, MARKET
from definitions.position import empty_long_position
from definitions.position_list import PositionList
from helpers.optimized import round_down, calculate_new_position_size_position_price
from strategies.base_strategy import Strategy, base_strategy_spec


@njit
def get_initial_position(current_price: float, leverage: float, reentry_grid: np.ndarray, wallet_balance: float,
                         wallet_percent: float, quantity_step: float, price_step: float) -> float:
    """
    Calculates the initial leveraged position size. Uses a percentage of the wallet balance as the limit of the
    cumulative entries.
    :param current_price: The current price to consider.
    :param leverage: The leverage to use.
    :param reentry_grid: The grid in form of a 2D array.
    :param wallet_balance: The current wallet balance.
    :param wallet_percent: The percentage of the wallet balance to use.
    :param quantity_step: The quantity step of the pair.
    :param price_step: The price step of the pair.
    :return: Initial leveraged position size.
    """
    available_leveraged_balance = wallet_balance * wallet_percent * leverage
    position_price = current_price
    position_size = 1.0
    sums = np.zeros(len(reentry_grid) + 1)
    sums[0] = current_price
    for i in range(len(reentry_grid)):
        old_position_price = position_price
        old_position_size = position_size
        position_price = round_down(position_price * (1 - reentry_grid[i][0] / 100), price_step)
        position_size = round_down(position_size * reentry_grid[i][1], quantity_step)
        sums[i + 1] = position_price * position_size
        position_size, position_price = calculate_new_position_size_position_price(old_position_size,
                                                                                   old_position_price, position_size,
                                                                                   position_price, quantity_step)
    initial_size = round_down(available_leveraged_balance / np.sum(sums), quantity_step)
    return initial_size


@njit
def get_dca_grid(position_size: float, position_price: float, leverage: float, reentry_grid: np.ndarray,
                 wallet_balance: float, wallet_percent: float, quantity_step: float, price_step: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Calculates the leveraged reentry grid based on the current position size and the current price.
    :param position_size: The size of the current position.
    :param position_price: The price of the current position.
    :param leverage: The leverage of the position.
    :param reentry_grid: The grid in form of a 2D array.
    :param wallet_balance: The current wallet balance.
    :param wallet_percent: The percentage of the wallet balance to use.
    :param quantity_step: The quantity step of the pair.
    :param price_step: The price step of the pair.
    :return: The leveraged reenry values, including price and size.
    """
    available_leveraged_balance = wallet_balance * wallet_percent * leverage
    reentry_prices = []
    reentry_sizes = []
    for v in reentry_grid:
        old_position_price = position_price
        old_position_size = position_size
        position_price = round_down(position_price * (1 - v[0] / 100), price_step)
        position_size = round_down(position_size * v[1], quantity_step)
        if old_position_price * old_position_size + position_price * position_size > available_leveraged_balance:
            break
        reentry_prices.append(position_price)
        reentry_sizes.append(position_size)
        position_size, position_price = calculate_new_position_size_position_price(old_position_size,
                                                                                   old_position_price, position_size,
                                                                                   position_price, quantity_step)
    reentry_prices = np.array(reentry_prices)
    reentry_sizes = np.array(reentry_sizes)
    return reentry_prices, reentry_sizes


@njit
def get_tp_grid(position_price: float, position_size: float, tp_grid: np.ndarray, quantity_step: float,
                price_step: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the leveraged take profit grid based on the current position size and current price.
    :param position_price: The current price to consider. Either the price where the initial position was entered, or
    the price where a reentry enter triggered.
    :param position_size: The size of the current position.
    :param tp_grid: The grid in form of a 2D array.
    :param quantity_step: The quantity step of the pair.
    :param price_step: The price step of the pair.
    :return: The leveraged take profit values, including price and size.
    """
    tp_prices = np.zeros(len(tp_grid))
    tp_sizes = np.zeros(len(tp_grid))
    for i in range(len(tp_grid)):
        tp_prices[i] = round_down(position_price * (1 + tp_grid[i][0] / 100), price_step)
        tp_sizes[i] = round_down(position_size * tp_grid[i][1], quantity_step)
    if np.sum(tp_sizes) < position_size:
        tp_sizes[-1] = round_down(tp_sizes[-1] + position_size - np.sum(tp_sizes), quantity_step)
    return tp_prices, tp_sizes


@jitclass([
    ('reentry_grid', types.float64[:, :]),
    ('tp_grid', types.float64[:, :]),
    ('percent', types.float64),
])
class StrategyConfig:
    """
    Strategy config for the grid trading strategy.
    """

    def __init__(self, reentry_grid: np.ndarray, tp_grid: np.ndarray, percent: float):
        """
        Initializes a strategy config for the grid trading strategy.
        :param reentry_grid: The reentry grid to use.
        :param tp_grid: The take profit grid to use.
        :param percent: The wallet percentage to use.
        """
        self.reentry_grid = reentry_grid
        self.tp_grid = tp_grid
        self.percent = percent


def convert_dict_to_config(config: dict) -> StrategyConfig:
    """
    Converts the strategy config from a dictionary to a grid strategy config.
    :param config: The strategy part of the config.
    :return: A jitclass grid StrategyConfig.
    """
    grid = []
    for k, v in config['reentry_grid'].items():
        grid.append([v[0], v[1]])
    reentry_grid = np.array(grid)
    grid = []
    for k, v in config['tp_grid'].items():
        grid.append([v[0], v[1]])
    tp_grid = np.array(grid)
    percent = config['percent']
    strategy_config = StrategyConfig(reentry_grid, tp_grid, percent)
    return strategy_config


@jitclass([
              ("config", typeof(StrategyConfig(np.asarray([[0.0, 0.0]]), np.asarray([[0.0, 0.0]]), 0.0))),
          ]
          + base_strategy_spec +
          [
              ('reentry_grid', types.float64[:, :]),
              ('tp_grid', types.float64[:, :]),
              ('percent', types.float64),
              ("last_filled_order", typeof(empty_order())),
              ("last_position", typeof(PositionList())),
              ("position_change", types.boolean),
              ("order_fill_change", types.boolean)
          ])
class Grid(Strategy):
    """
    Grid trading strategy using a fixed grid.
    """
    __init_Base = Strategy.__init__

    def __init__(self, config: StrategyConfig):
        """
        Initializes a grid strategy with a grid strategy config.
        :param config:
        """
        self.__init_Base(config)
        self.reentry_grid = self.config.reentry_grid
        self.tp_grid = self.config.tp_grid
        self.percent = self.config.percent

        self.last_filled_order = empty_order()
        self.last_position = PositionList()
        self.position_change = False
        self.order_fill_change = False

    def precompile(self):
        """
        Compiles all used functions.
        :return:
        """
        round_down(0.01, 0.001)
        get_initial_position(0.01, 1.0, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_dca_grid(0.01, 0.01, 1, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_tp_grid(0.01, 0.01, np.array([[0.1, 1.0]]), 0.01, 0.01)
        self.quantity_step = 0.001
        self.price_step = 0.01
        self.on_update(PositionList(), empty_order())
        price_list = empty_candle_list()
        self.make_decision(price_list)
        self.prepare_tp_orders(PositionList())
        p = PositionList()
        p.update_long(empty_long_position())
        self.prepare_reentry_orders(p)
        self.calculate_dca_tp(p)
        self.quantity_step = 0.0
        self.price_step = 0.0
        self.last_filled_order = empty_order()

    def make_decision(self, prices: List[Candle]) -> Tuple[List[Order], List[Order]]:
        """
        Makes a decision based on a price update. All values are updated before the function is called.
        :param prices: Current price list.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        if self.position.long.empty() and len(self.open_orders.long) == 0:
            if len(prices) > 0:
                size = get_initial_position(prices[-1].close, self.leverage, self.reentry_grid, self.balance,
                                            self.percent,
                                            self.quantity_step, self.price_step)
                add_orders.append(Order(self.symbol, 0, 0.0, 0.0, size, MARKET, BUY, 0, '', LONG))
        return add_orders, delete_orders

    def on_order_update(self, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        """
        Checks whether the last order was a filled order. In that case, set order_fill_change to True and
        last_filled_order to the order. If position_change is also True, execute the update.
        :param last_filled_order: The last filled order.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        if last_filled_order.action == FILLED:
            self.last_filled_order = last_filled_order
            self.order_fill_change = True
        if self.order_fill_change and self.position_change:
            add_orders, delete_orders = self.on_update(self.last_position, self.last_filled_order)
            self.order_fill_change = False
            self.position_change = False
            self.last_filled_order = empty_order()
            self.last_position = PositionList()
        else:
            add_orders = empty_order_list()
            delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_account_update(self, old_balance: float, new_balance: float, old_position: PositionList,
                          new_position: PositionList) -> Tuple[List[Order], List[Order]]:
        """
        Checks whether the position changed. If that's the case, set position_change to True and last_position to the
        new position. If order_fill_change is also True, execute the update.
        :param old_balance: The old balance.
        :param new_balance: The new balance.
        :param old_position: The old position.
        :param new_position: The new position.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        if not old_position.long.equal(new_position.long) or not old_position.short.equal(new_position.short):
            self.last_position = new_position
            self.position_change = True
        if self.order_fill_change and self.position_change:
            add_orders, delete_orders = self.on_update(self.last_position, self.last_filled_order)
            self.order_fill_change = False
            self.position_change = False
            self.last_filled_order = empty_order()
            self.last_position = PositionList()
        else:
            add_orders = empty_order_list()
            delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_update(self, position: PositionList, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        """
        Called on a position and order update. Creates the full grid when the order is new. Removes the grid when the
        position closed. Updates the take profit grid when a reentry triggered and vice versa the reentry grid when a
        take profit triggered.
        :param position: The new position.
        :param last_filled_order: The last filled order.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        if self.position.long.size == 0.0 and position.long.size != 0.0:
            add_orders.extend(self.calculate_dca_tp(position))
        elif self.position.long.size != 0.0 and position.long.size == 0.0:
            for order in self.open_orders.long:
                delete_orders.append(order)
        else:
            if last_filled_order.order_type == LIMIT and last_filled_order.position_side == LONG \
                    and last_filled_order.side == BUY:
                # Reentry triggered
                # Update TP grid
                for order in self.open_orders.long:
                    if order.order_type == TP:
                        delete_orders.append(order)
                add_orders.extend(self.prepare_tp_orders(position))
            elif last_filled_order.order_type == TP and last_filled_order.position_side == LONG \
                    and last_filled_order.side == SELL:
                # TP triggered
                # Update reentry grid
                for order in self.open_orders.long:
                    if order.order_type == LIMIT:
                        delete_orders.append(order)
                add_orders.extend(self.prepare_reentry_orders(position))
            else:
                # Position was changed but not by the strategy
                # Recalculate both
                for order in self.open_orders.long:
                    delete_orders.append(order)
                add_orders.extend(self.calculate_dca_tp(position))
        return add_orders, delete_orders

    def prepare_tp_orders(self, position: PositionList) -> List[Order]:
        """
        Prepares the take profit orders. Calculates the grid and converts it to order list.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        tp_prices, tp_sizes = get_tp_grid(position.long.price, position.long.size, self.tp_grid, self.quantity_step,
                                          self.price_step)
        for i in range(len(tp_prices)):
            order = Order(position.long.symbol, 0, float(tp_prices[i]), float(tp_prices[i]), float(tp_sizes[i]), TP,
                          SELL, 0, '', LONG)
            orders.append(order)
        return orders

    def prepare_reentry_orders(self, position: PositionList) -> List[Order]:
        """
        Prepares the reentry orders. Calculates the grid and converts it to order list.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        reentry_prices, reentry_sizes = get_dca_grid(position.long.size, position.long.price, position.long.leverage,
                                                     self.reentry_grid, self.balance, self.percent, self.quantity_step,
                                                     self.price_step)
        for i in range(len(reentry_prices)):
            order = Order(position.long.symbol, 0, float(reentry_prices[i]), float(reentry_prices[i]),
                          float(reentry_sizes[i]), LIMIT, BUY, 0, '', LONG)
            orders.append(order)
        return orders

    def calculate_dca_tp(self, position: PositionList) -> List[Order]:
        """
        Calculates both grids.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        orders.extend(self.prepare_reentry_orders(position))
        orders.extend(self.prepare_tp_orders(position))
        return orders


# Strategy definition used in initializing the backtesting bot.
# Ensure that this is correctly defined.
strategy_definition = Grid(StrategyConfig(np.zeros((1, 1)), np.zeros((1, 1)), 0.0))
