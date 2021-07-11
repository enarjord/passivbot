from typing import List, Tuple

import numpy as np
from numba import njit

from definitions.order import Order, TP, SELL, LONG, LIMIT, BUY
from definitions.order_list import OrderList, empty_order_list
from definitions.position_list import PositionList
from functions import print_
from strategies.base_strategy import Strategy


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


class Grid(Strategy):
    def __init__(self, config):
        super().__init__(config)
        self.reentry_grid = self.config['reentry_grid']
        self.tp_grid = self.config['tp_grid']
        self.percent = self.config['percent']

    def precompile(self):
        round_dn(0.0, 0.01)
        get_initial_position(0.01, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_dca_grid(0.01, 0.01, 0.01, 1, np.array([[0.1, 1.0]]), 1, 0.1, 0.01, 0.01)
        get_tp_grid(0.01, 0.01, np.array([[0.1, 1.0]]), 0.01, 0.01)

    def make_decision(self, balance: float, position: PositionList, orders: OrderList, price: float) -> Tuple[
        List[Order], List[Order]]:
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        return add_orders, delete_orders

    def on_update(self, position: PositionList, last_filled_order: Order) -> Tuple[List[Order], List[Order]]:
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        try:
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
        except Exception as e:
            print_(['On update error', e], n=True)
        return add_orders, delete_orders

    def prepare_tp_orders(self, price: float, position: PositionList) -> List[Order]:
        orders = empty_order_list()
        tp_prices, tp_sizes = get_tp_grid(price, position.long.size, self.tp_grid, self.qty_step, self.price_step)
        for i in range(len(tp_prices)):
            order = Order(position.long.symbol, 0, float(tp_prices[i]), float(tp_prices[i]), float(tp_sizes[i]), TP,
                          SELL, 0, '', LONG)
            orders.append(order)
        return orders

    def prepare_reentry_orders(self, price: float, position: PositionList) -> List[Order]:
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
        orders = empty_order_list()
        orders.extend(self.prepare_reentry_orders(price, position))
        orders.extend(self.prepare_tp_orders(price, position))
        return orders

    def load_strategy_config(self, config: dict) -> dict:
        try:
            grid = []
            for k, v in config['reentry_grid'].items():
                grid.append([v[0], v[1]])
            config['reentry_grid'] = np.array(grid)
            grid = []
            for k, v in config['tp_grid'].items():
                grid.append([v[0], v[1]])
            config['tp_grid'] = np.array(grid)
            return config
        except Exception as e:
            print_(['Could not read strategy config', e], n=True)
            return {}
