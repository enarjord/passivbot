import numpy as np
from numba import njit

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

    def make_decision(self, balance, position, orders, price) -> dict:
        changed_orders = {'cancel': [],
                          'add': []}
        return changed_orders

    def on_update(self, position, last_filled_order):
        changed_orders = {'cancel': [],
                          'add': []}
        try:
            if not self.position['LONG'] and position['LONG']:
                changed_orders['add'].extend(self.calculate_dca_tp(last_filled_order['price'], position))
            elif self.position['LONG'] and not position['LONG']:
                for order in self.open_orders['LONG']:
                    changed_orders['cancel'].append(order)
            else:
                if last_filled_order['type'] == 'LIMIT' and last_filled_order['position_side'] == 'LONG' and \
                        last_filled_order['side'] == 'BUY':
                    # Reentry triggered
                    # Update TP grid
                    for order in self.open_orders['LONG']:
                        if order['type'] == 'TAKE_PROFIT':
                            changed_orders['cancel'].append(order)
                    changed_orders['add'].extend(self.prepare_tp_orders(last_filled_order['price'], position))
                elif last_filled_order['type'] == 'TAKE_PROFIT' and last_filled_order['position_side'] == 'LONG' and \
                        last_filled_order['side'] == 'SELL':
                    # TP triggered
                    # Update reentry grid
                    for order in self.open_orders['LONG']:
                        if order['type'] == 'LIMIT':
                            changed_orders['cancel'].append(order)
                    changed_orders['add'].extend(self.prepare_reentry_orders(last_filled_order['price'], position))
                else:
                    # Position was changed but not by the strategy
                    # Recalculate both
                    for order in self.open_orders['LONG']:
                        changed_orders['cancel'].append(order)
                    changed_orders['add'].extend(self.calculate_dca_tp(last_filled_order['price'], position))
        except Exception as e:
            print_(['On update error', e], n=True)
        return changed_orders

    def prepare_tp_orders(self, price, position):
        orders = []
        tp_prices, tp_sizes = get_tp_grid(price, position['LONG']['size'], self.tp_grid, self.qty_step, self.price_step)
        for i in range(len(tp_prices)):
            order = {'side': 'SELL',
                     'position_side': 'LONG',
                     'type': 'TAKE_PROFIT',
                     'qty': tp_sizes[i],
                     'price': tp_prices[i],
                     'stop_price': tp_prices[i]}
            orders.append(order)
        return orders

    def prepare_reentry_orders(self, price, position):
        orders = []
        reentry_prices, reentry_sizes = get_dca_grid(price, position['LONG']['size'], position['LONG']['price'],
                                                     position['LONG']['leverage'], self.reentry_grid, self.balance,
                                                     self.percent, self.qty_step, self.price_step)
        for i in range(len(reentry_prices)):
            order = {'side': 'BUY',
                     'position_side': 'LONG',
                     'type': 'LIMIT',
                     'qty': reentry_sizes[i] * position['LONG']['leverage'],
                     'price': reentry_prices[i]}
            orders.append(order)
        return orders

    def calculate_dca_tp(self, price, position):
        orders = []
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
