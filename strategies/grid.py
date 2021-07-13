from typing import List, Tuple

import numpy as np
from numba import types, typeof, njit
from numba.experimental import jitclass

from definitions.order import Order, TP, SELL, LONG, LIMIT, BUY
from definitions.order_list import OrderList, empty_order_list
from definitions.position_list import PositionList
from strategies.base_strategy import Strategy, base_strategy_spec


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def get_initial_position(current_price, reentry_grid, wallet_balance, wallet_percent, qty_step, price_step):
    available_balance = wallet_balance * wallet_percent
    price = current_price
    mult = 1.0
    sums = np.zeros(len(reentry_grid) + 1)
    sums[0] = current_price
    for i in range(len(reentry_grid)):
        price = round_dn(price * (1 - reentry_grid[i][0] / 100), price_step)
        mult *= reentry_grid[i][1]
        sums[i + 1] = price * mult
    initial_size = round_dn(available_balance / np.sum(sums), qty_step)
    return initial_size


@njit
def get_dca_grid(current_price, position_size, position_price, leverage, reentry_grid, wallet_balance, wallet_percent,
                 qty_step, price_step):
    available_balance = wallet_balance * wallet_percent
    reentry_prices = []
    reentry_sizes = []
    position_size /= leverage
    max_pos = position_size * position_price
    for v in reentry_grid:
        current_price = round_dn(current_price * (1 - v[0] / 100), price_step)
        position_size = round_dn(position_size * v[1], qty_step)
        if max_pos + current_price * position_size > available_balance:
            break
        reentry_prices.append(current_price)
        reentry_sizes.append(position_size)
        max_pos += current_price * position_size
    reentry_prices = np.array(reentry_prices)
    reentry_sizes = np.array(reentry_sizes)
    return reentry_prices, reentry_sizes


@njit
def get_tp_grid(current_price, position_size, tp_grid, qty_step, price_step):
    tp_prices = np.zeros(len(tp_grid))
    tp_sizes = np.zeros(len(tp_grid))
    for i in range(len(tp_grid)):
        tp_prices[i] = round_dn(current_price * (1 + tp_grid[i][0] / 100), price_step)
        tp_sizes[i] = round_dn(position_size * tp_grid[i][1], qty_step)
    if np.sum(tp_sizes) < position_size:
        tp_sizes[-1] = round_dn(tp_sizes[-1] + position_size - np.sum(tp_sizes), qty_step)
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

    def precompile(self):
        """
        Compiles all used functions.
        :return:
        """
        round_dn(0.0, 0.01)
        get_initial_position(0.01, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_dca_grid(0.01, 0.01, 0.01, 1, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_tp_grid(0.01, 0.01, np.array([[0.1, 1.0]]), 0.01, 0.01)

    def make_decision(self, balance: float, position: PositionList, orders: OrderList, price: float) -> Tuple[
        List[Order], List[Order]]:
        """
        Makes a decision based on a price update.
        :param balance: Current balance.
        :param position: Current position.
        :param orders: Current orders.
        :param price: Current price.
        :return: Two typed lists of orders, orders to add and orders to delete.
        """
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
            add_orders.extend(self.calculate_dca_tp(last_filled_order.price, position))
        elif self.position.long.size != 0.0 and position.long.size == 0.0:
            for order in self.open_orders.long:
                delete_orders.append(order)
        else:
            if last_filled_order.type == LIMIT and last_filled_order.position_side == LONG \
                    and last_filled_order.side == BUY:
                # Reentry triggered
                # Update TP grid
                for order in self.open_orders.long:
                    if order.type == TP:
                        delete_orders.append(order)
                add_orders.extend(self.prepare_tp_orders(last_filled_order.price, position))
            elif last_filled_order.type == TP and last_filled_order.position_side == LONG \
                    and last_filled_order.side == SELL:
                # TP triggered
                # Update reentry grid
                for order in self.open_orders.long:
                    if order.type == LIMIT:
                        delete_orders.append(order)
                add_orders.extend(self.prepare_reentry_orders(last_filled_order.price, position))
            else:
                # Position was changed but not by the strategy
                # Recalculate both
                for order in self.open_orders.long:
                    delete_orders.append(order)
                add_orders.extend(self.calculate_dca_tp(last_filled_order.price, position))
        return add_orders, delete_orders

    def prepare_tp_orders(self, price: float, position: PositionList) -> List[Order]:
        """
        Prepares the take profit orders. Calculates the grid and converts it to order list.
        :param price: Price to use for grid calculation.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        tp_prices, tp_sizes = get_tp_grid(price, position.long.size, self.tp_grid, self.qty_step, self.price_step)
        for i in range(len(tp_prices)):
            order = Order(position.long.symbol, 0, float(tp_prices[i]), float(tp_prices[i]), float(tp_sizes[i]), TP,
                          SELL, 0, '', LONG)
            orders.append(order)
        return orders

    def prepare_reentry_orders(self, price: float, position: PositionList) -> List[Order]:
        """
        Prepares the reentry orders. Calculates the grid and converts it to order list.
        :param price: Price to use for grid calculation.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        reentry_prices, reentry_sizes = get_dca_grid(price, position.long.size, position.long.price,
                                                     position.long.leverage, self.reentry_grid, self.balance,
                                                     self.percent, self.qty_step, self.price_step)
        for i in range(len(reentry_prices)):
            order = Order(position.long.symbol, 0, float(reentry_prices[i]), float(reentry_prices[i]),
                          float(reentry_sizes[i] * float(position.long.leverage)), LIMIT, BUY, 0, '', LONG)
            orders.append(order)
        return orders

    def calculate_dca_tp(self, price: float, position: PositionList) -> List[Order]:
        """
        Calculates both grids.
        :param price: Price to use for grid calculation.
        :param position: Current position.
        :return: Typed list of orders.
        """
        orders = empty_order_list()
        orders.extend(self.prepare_reentry_orders(price, position))
        orders.extend(self.prepare_tp_orders(price, position))
        return orders
