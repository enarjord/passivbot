import asyncio
import datetime
import json
import os
import sys
from time import time

import git
import numpy as np

import telegram_bot

if '--nojit' in sys.argv:
    print('not using numba')


    def njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap
else:
    print('using numba')
    from numba import njit


@njit
def nan_to_0(x: float) -> float:
    return x if x == x else 0.0


@njit
def round_up(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.ceil(n / step) * step, safety_rounding)


@njit
def round_dn(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.floor(n / step) * step, safety_rounding)


@njit
def round_(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.round(n / step) * step, safety_rounding)


@njit
def calc_diff(x, y):
    return abs(x - y) / abs(y)


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if type(v) == dict:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@njit
def calc_ema(alpha: float, alpha_: float, prev_ema: float, new_val: float) -> float:
    return prev_ema * alpha_ + new_val * alpha


@njit
def calc_initial_long_entry_price(price_step: float, spread: float, ema: float,
                                  highest_bid: float) -> float:
    return min(highest_bid, round_dn(ema * (1 - spread), price_step))


@njit
def calc_initial_shrt_entry_price(price_step: float, spread: float, ema: float,
                                  lowest_ask: float) -> float:
    return max(lowest_ask, round_up(ema * (1 + spread), price_step))


#################
# inverse calcs #
#################


@njit
def calc_min_qty_inverse(qty_step: float, min_qty: float, min_cost: float, price: float) -> float:
    return min_qty


@njit
def calc_long_pnl_inverse(entry_price: float, close_price: float, qty: float) -> float:
    return abs(qty) * (1 / entry_price - 1 / close_price)


@njit
def calc_shrt_pnl_inverse(entry_price: float, close_price: float, qty: float) -> float:
    return abs(qty) * (1 / close_price - 1 / entry_price)


@njit
def calc_cost_inverse(qty: float, price: float) -> float:
    return abs(qty / price)


@njit
def calc_margin_cost_inverse(leverage: float, qty: float, price: float) -> float:
    return calc_cost_inverse(qty, price) / leverage


@njit
def calc_max_pos_size_inverse(leverage: float, balance: float, price: float) -> float:
    return balance * price * leverage


@njit
def calc_min_order_qty_inverse(qty_step: float, min_qty: float, min_cost: float,
                               qty_pct: float, leverage: float, balance: float,
                               price: float) -> float:
    return calc_min_order_qty(calc_min_qty_inverse(qty_step, min_qty, min_cost, price),
                              qty_step,
                              balance * leverage * price,
                              qty_pct)


@njit
def calc_available_margin_inverse(contract_multiplier: float,
                                  leverage: float,
                                  balance: float,
                                  long_psize: float,
                                  long_pprice: float,
                                  shrt_psize: float,
                                  shrt_pprice: float,
                                  last_price: float) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        long_psize_real = long_psize * contract_multiplier
        equity += calc_long_pnl_inverse(long_pprice, last_price, long_psize_real)
        used_margin += calc_cost_inverse(long_psize_real, long_pprice) / leverage
    if shrt_pprice and shrt_psize:
        shrt_psize_real = shrt_psize * contract_multiplier
        equity += calc_shrt_pnl_inverse(shrt_pprice, last_price, shrt_psize_real)
        used_margin += calc_cost_inverse(shrt_psize_real, shrt_pprice) / leverage
    return equity - used_margin


@njit
def calc_stop_loss_inverse(qty_step: float,
                           min_qty: float,
                           min_cost: float,
                           contract_multiplier: float,
                           qty_pct: float,
                           leverage: float,
                           stop_loss_liq_diff: float,
                           stop_loss_pos_pct: float,
                           balance: float,
                           long_psize: float,
                           long_pprice: float,
                           shrt_psize: float,
                           shrt_pprice: float,
                           liq_price: float,
                           highest_bid: float,
                           lowest_ask: float,
                           last_price: float,
                           available_margin: float,
                           do_long: bool = True,
                           do_shrt: bool = True,):
    abs_shrt_psize = abs(shrt_psize)
    if calc_diff(liq_price, last_price) < stop_loss_liq_diff:
        if long_psize > abs_shrt_psize:
            stop_loss_qty = min(long_psize,
                                max(calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct,
                                                               leverage,
                                                               balance / contract_multiplier,
                                                               lowest_ask),
                                    round_dn(long_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase short pos, otherwise, reduce long pos
            margin_cost = calc_margin_cost_inverse(leverage, stop_loss_qty * contract_multiplier,
                                                   lowest_ask)
            if margin_cost < available_margin and do_shrt:
                # add to shrt pos
                new_shrt_psize = round_(shrt_psize - stop_loss_qty, qty_step)
                shrt_pprice = (nan_to_0(shrt_pprice) * (shrt_psize / new_shrt_psize) +
                               lowest_ask * (-stop_loss_qty / new_shrt_psize))
                shrt_psize = new_shrt_psize
                return -stop_loss_qty, lowest_ask, shrt_psize, shrt_pprice, 'stop_loss_shrt_entry'
                available_margin -= margin_cost
            else:
                # reduce long pos
                long_psize = round_(long_psize - stop_loss_qty, qty_step)
                return -stop_loss_qty, lowest_ask, long_psize, long_pprice, 'stop_loss_long_close'
                available_margin += margin_cost
        else:
            stop_loss_qty = min(abs_shrt_psize,
                                max(calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct,
                                                               leverage,
                                                               balance / contract_multiplier,
                                                               highest_bid),
                                    round_dn(abs_shrt_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase long pos, otherwise, reduce shrt pos
            margin_cost = calc_margin_cost_inverse(leverage, stop_loss_qty * contract_multiplier,
                                                   highest_bid)
            if margin_cost < available_margin and do_long:
                # add to long pos
                new_long_psize = round_(long_psize + stop_loss_qty, qty_step)
                long_pprice = (nan_to_0(long_pprice) * (long_psize / new_long_psize) +
                               highest_bid * (stop_loss_qty / new_long_psize))
                long_psize = new_long_psize
                return stop_loss_qty, highest_bid, long_psize, long_pprice, 'stop_loss_long_entry'
                available_margin -= margin_cost
            else:
                # reduce shrt pos
                shrt_psize = round_(shrt_psize + stop_loss_qty, qty_step)
                return stop_loss_qty, highest_bid, shrt_psize, shrt_pprice, 'stop_loss_shrt_close'
                available_margin += margin_cost
    return 0.0, 0.0, 0.0, 0.0, ''


@njit
def iter_entries_inverse(price_step: float,
                         qty_step: float,
                         min_qty: float,
                         min_cost: float,
                         contract_multiplier: float,
                         ddown_factor: float,
                         qty_pct: float,
                         leverage: float,
                         grid_spacing: float,
                         grid_coefficient: float,
                         ema_spread: float,
                         stop_loss_liq_diff: float,
                         stop_loss_pos_pct: float,
                         balance: float,
                         long_psize: float,
                         long_pprice: float,
                         shrt_psize: float,
                         shrt_pprice: float,
                         liq_price: float,
                         highest_bid: float,
                         lowest_ask: float,
                         ema: float,
                         last_price: float,
                         do_long: bool = True,
                         do_shrt: bool = True,):
    # yields both long and short entries
    # (qty, price, new_psize, new_pprice, comment)

    available_margin = calc_available_margin_inverse(contract_multiplier, leverage, balance,
                                                     long_psize, long_pprice,
                                                     shrt_psize, shrt_pprice, last_price)


    stop_loss_order = calc_stop_loss_inverse(qty_step, min_qty, min_cost, contract_multiplier,
                                             qty_pct, leverage, stop_loss_liq_diff,
                                             stop_loss_pos_pct, balance, long_psize, long_pprice,
                                             shrt_psize, shrt_pprice, liq_price, highest_bid,
                                             lowest_ask, last_price, available_margin, do_long,
                                             do_shrt)
    if stop_loss_order[0] != 0.0:
        yield stop_loss_order

    while True:

        long_entry = calc_next_long_entry_inverse(
            price_step, qty_step, min_qty, min_cost, contract_multiplier, ddown_factor, qty_pct,
            leverage, grid_spacing, grid_coefficient, ema_spread, balance, long_psize, long_pprice,
            shrt_psize, highest_bid, ema, available_margin
        ) if do_long else (0.0, np.nan, long_psize, long_pprice, '')

        shrt_entry = calc_next_shrt_entry_inverse(
            price_step, qty_step, min_qty, min_cost, contract_multiplier, ddown_factor, qty_pct,
            leverage, grid_spacing, grid_coefficient, ema_spread, balance, long_psize, shrt_psize,
            shrt_pprice, lowest_ask, ema, available_margin
        ) if do_shrt else (0.0, np.nan, shrt_psize, shrt_pprice, '')

        if long_entry[0] > 0.0:
            if shrt_entry[0] == 0.0:
                long_first = True
            else:
                long_first = calc_diff(long_entry[1], last_price) < calc_diff(shrt_entry[1], last_price)
        elif shrt_entry[0] < 0.0:
            long_first = False
        else:
            break
        if long_first:
            yield long_entry
            long_psize = long_entry[2]
            long_pprice = long_entry[3]
            if long_entry[1]:
                available_margin -= calc_margin_cost_inverse(leverage,
                                                             long_entry[0] * contract_multiplier,
                                                             long_entry[1])
        else:
            yield shrt_entry
            shrt_psize = shrt_entry[2]
            shrt_pprice = shrt_entry[3]
            if shrt_entry[1]:
                available_margin -= calc_margin_cost_inverse(leverage,
                                                             shrt_entry[0] * contract_multiplier,
                                                             shrt_entry[1])


@njit
def calc_next_long_entry_inverse(price_step: float,
                                 qty_step: float,
                                 min_qty: float,
                                 min_cost: float,
                                 contract_multiplier: float,
                                 ddown_factor: float,
                                 qty_pct: float,
                                 leverage: float,
                                 grid_spacing: float,
                                 grid_coefficient: float,
                                 ema_spread: float,
                                 balance: float,
                                 long_psize: float,
                                 long_pprice: float,
                                 shrt_psize: float,
                                 highest_bid: float,
                                 ema: float,
                                 available_margin: float):
    if long_psize == 0.0:
        price = min(highest_bid, round_dn(ema * (1 - ema_spread), price_step))
        long_qty = min(round_dn((available_margin / contract_multiplier) * price * leverage, qty_step),
                       calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct, leverage,
                                                  balance / contract_multiplier, price))
        if long_qty < calc_min_qty_inverse(qty_step, min_qty, min_cost, price):
            long_qty = 0.0
        long_pprice = price
        return long_qty, price, long_qty, long_pprice, 'initial_long_entry'
    else:
        long_pmargin = calc_margin_cost_inverse(leverage, long_psize * contract_multiplier, long_pprice)
        price = min(round_(highest_bid, price_step),
                    calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, long_pmargin, long_pprice))
        if price <= 0.0:
            return 0.0, np.nan, long_psize, long_pprice, 'long_reentry'
        max_order_qty = round_dn((available_margin / contract_multiplier) * price * leverage, qty_step)
        min_long_order_qty = calc_min_qty_inverse(qty_step, min_qty, min_cost, price)
        long_qty = calc_reentry_qty(qty_step, ddown_factor, min_long_order_qty,
                                    max_order_qty,
                                    long_psize)
        if long_qty >= min_long_order_qty:
            new_long_psize = round_(long_psize + long_qty, qty_step)
            long_pprice = nan_to_0(long_pprice) * (long_psize / new_long_psize) + \
                          price * (long_qty / new_long_psize)
            return long_qty, price, new_long_psize, long_pprice, 'long_reentry'
        else:
            return 0.0, np.nan, long_psize, long_pprice, 'long_reentry'

@njit
def calc_next_shrt_entry_inverse(price_step: float,
                                 qty_step: float,
                                 min_qty: float,
                                 min_cost: float,
                                 contract_multiplier: float,
                                 ddown_factor: float,
                                 qty_pct: float,
                                 leverage: float,
                                 grid_spacing: float,
                                 grid_coefficient: float,
                                 ema_spread: float,
                                 balance: float,
                                 long_psize: float,
                                 shrt_psize: float,
                                 shrt_pprice: float,
                                 lowest_ask: float,
                                 ema: float,
                                 available_margin: float):
    if shrt_psize == 0.0:
        price = max(lowest_ask, round_up(ema * (1 + ema_spread), price_step))
        shrt_qty = min(round_dn((available_margin / contract_multiplier) * price * leverage, qty_step),
                       calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct, leverage,
                                                  balance / contract_multiplier, price))
        if shrt_qty < calc_min_qty_inverse(qty_step, min_qty, min_cost, price):
            shrt_qty = 0.0
        shrt_pprice = price
        return -shrt_qty, price, -shrt_qty, shrt_pprice, 'initial_shrt_entry'
    else:
        pos_margin = calc_margin_cost_inverse(leverage, shrt_psize * contract_multiplier, shrt_pprice)
        price = max(round_(lowest_ask, price_step),
                    calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, pos_margin, shrt_pprice))
        '''
        min_order_qty = -calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct,
                                                    leverage, balance, price)
        '''
        min_order_qty = calc_min_qty_inverse(qty_step, min_qty, min_cost, price)

        max_order_qty = round_dn((available_margin / contract_multiplier) * price * leverage, qty_step)
        qty = calc_reentry_qty(qty_step, ddown_factor, min_order_qty, max_order_qty, shrt_psize)
        if qty >= min_order_qty:
            new_pos_size = shrt_psize - qty
            shrt_pprice = nan_to_0(shrt_pprice) * (shrt_psize / new_pos_size) + price * (-qty / new_pos_size)
            margin_cost = calc_margin_cost_inverse(leverage, qty, price)
            return -qty, price, round_(new_pos_size, qty_step), shrt_pprice, 'shrt_reentry'
        else:
            return 0.0, np.nan, shrt_psize, shrt_pprice, 'shrt_reentry'


@njit
def iter_long_closes_inverse(price_step: float,
                             qty_step: float,
                             min_qty: float,
                             min_cost: float,
                             contract_multiplier: float,
                             qty_pct: float,
                             leverage: float,
                             min_markup: float,
                             markup_range: float,
                             n_orders: int,
                             balance: float,
                             pos_size: float,
                             pos_price: float,
                             lowest_ask: float):

    # yields tuple (qty, price, new_pos_size)
    if pos_size == 0.0:
        return

    minm = pos_price * (1 + min_markup)
    prices = np.linspace(minm, pos_price * (1 + min_markup + markup_range), int(n_orders))
    prices = [p for p in sorted(set([round_up(p_, price_step) for p_ in prices]))
              if p >= lowest_ask]
    if len(prices) == 0:
        yield -pos_size, max(lowest_ask, round_up(minm, price_step)), 0.0
    else:
        n_orders = int(min([n_orders, len(prices), int(pos_size / min_qty)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = -min(pos_size, max(calc_min_order_qty_inverse(qty_step, min_qty, min_cost,
                                                                    qty_pct, leverage,
                                                                    balance / contract_multiplier,
                                                                    lowest_ask),
                                         round_up(pos_size / n_orders, qty_step)))
                if pos_size != 0.0 and -qty / pos_size > 0.75:
                    qty = -pos_size
            if qty == 0.0:
                break
            pos_size = round_(pos_size + qty, qty_step)
            yield qty, price, pos_size
            lowest_ask = price
            n_orders -= 1
        if pos_size > 0.0:
            yield -pos_size, max(lowest_ask, round_up(minm, price_step)), 0.0


@njit
def iter_shrt_closes_inverse(price_step: float,
                             qty_step: float,
                             min_qty: float,
                             min_cost: float,
                             contract_multiplier: float,
                             qty_pct: float,
                             leverage: float,
                             min_markup: float,
                             markup_range: float,
                             n_orders: int,
                             balance: float,
                             pos_size: float,
                             pos_price: float,
                             highest_bid: float):
    # yields tuple (qty, price, new_pos_size)

    if pos_size == 0.0:
        return

    abs_pos_size = abs(pos_size)
    minm = pos_price * (1 - min_markup)

    prices = np.linspace(minm, pos_price * (1 - (min_markup + markup_range)), int(n_orders))
    prices = [p for p in sorted(set([round_dn(p_, price_step) for p_ in prices]), reverse=True)
              if p <= highest_bid]

    if len(prices) == 0:
        yield abs_pos_size, min(highest_bid, round_dn(minm, price_step)), 0.0
        abs_pos_size = 0.0
    else:
        n_orders = int(min([n_orders, len(prices), int(abs_pos_size / min_qty)]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(abs_pos_size, max(calc_min_order_qty_inverse(qty_step, min_qty, min_cost,
                                                                       qty_pct, leverage,
                                                                       balance / contract_multiplier,
                                                                       highest_bid),
                                            round_up(abs_pos_size / n_orders, qty_step)))
                if abs_pos_size != 0.0 and qty / abs_pos_size > 0.75:
                    qty = abs_pos_size
            if qty == 0.0:
                break
            abs_pos_size = round_(abs_pos_size - qty, qty_step)
            yield qty, price, abs_pos_size * -1
            highest_bid = price
            n_orders -= 1
        if abs_pos_size > 0.0:
            yield abs_pos_size, min(highest_bid, round_dn(minm, price_step)), 0.0


@njit
def calc_cross_long_liq_price_bybit_inverse(balance,
                                            pos_size,
                                            pos_price,
                                            leverage,
                                            mm=0.005) -> float:
    order_cost = pos_size / pos_price
    order_margin = order_cost / leverage
    bankruptcy_price = calc_cross_long_bankruptcy_price_bybit_inverse(pos_size, order_cost, balance, order_margin)
    if bankruptcy_price == 0.0:
        return 0.0
    rhs = -(balance - order_margin - (pos_size / pos_price) * mm -
            (pos_size * 0.00075) / bankruptcy_price)
    return (pos_price * pos_size) / (pos_size - pos_price * rhs)


@njit
def calc_cross_long_bankruptcy_price_bybit_inverse(pos_size, order_cost, balance, order_margin) -> float:
    return (1.00075 * pos_size) / (order_cost + (balance - order_margin))


@njit
def calc_cross_shrt_liq_price_bybit_inverse(balance,
                                            pos_size,
                                            pos_price,
                                            leverage,
                                            mm=0.005) -> float:
    _pos_size = abs(pos_size)
    order_cost = _pos_size / pos_price
    order_margin = order_cost / leverage
    bankruptcy_price = calc_cross_shrt_bankruptcy_price_bybit_inverse(_pos_size, order_cost, balance, order_margin)
    if bankruptcy_price == 0.0:
        return 0.0
    rhs = -(balance - order_margin - (_pos_size / pos_price) * mm -
            (_pos_size * 0.00075) / bankruptcy_price)
    shrt_liq_price = (pos_price * _pos_size) / (pos_price * rhs + _pos_size)
    if shrt_liq_price <= 0.0:
        return 0.0
    return shrt_liq_price


@njit
def calc_cross_shrt_bankruptcy_price_bybit_inverse(pos_size, order_cost, balance, order_margin) -> float:
    return (0.99925 * pos_size) / (order_cost - (balance - order_margin))


@njit
def calc_cross_hedge_liq_price_bybit_inverse(balance: float,
                                             long_pos_size: float,
                                             long_pos_price: float,
                                             shrt_pos_size: float,
                                             shrt_pos_price: float,
                                             leverage: float) -> float:
    if long_pos_size > abs(shrt_pos_size):
        return calc_cross_long_liq_price_bybit_inverse(balance, long_pos_size, long_pos_price, leverage)
    else:
        return calc_cross_shrt_liq_price_bybit_inverse(balance, shrt_pos_size, shrt_pos_price, leverage)


@njit
def calc_cross_hedge_liq_price_binance_inverse(balance: float,
                                               long_pos_size: float,
                                               long_pos_price: float,
                                               shrt_pos_size: float,
                                               shrt_pos_price: float,
                                               leverage: float,
                                               contract_multiplier: float = 1.0) -> float:
    abs_long_pos_size = abs(long_pos_size)
    abs_shrt_pos_size = abs(shrt_pos_size)
    long_pos_price = long_pos_price if long_pos_price == long_pos_price else 0.0
    shrt_pos_price = shrt_pos_price if shrt_pos_price == shrt_pos_price else 0.0
    mml = 0.02
    mms = 0.02
    numerator = abs_long_pos_size * mml + abs_shrt_pos_size * mms + abs_long_pos_size - abs_shrt_pos_size
    long_pos_cost = abs_long_pos_size / long_pos_price if long_pos_price > 0.0 else 0.0
    shrt_pos_cost = abs_shrt_pos_size / shrt_pos_price if shrt_pos_price > 0.0 else 0.0
    denom = balance / contract_multiplier + long_pos_cost - shrt_pos_cost
    if denom == 0.0:
        return 0.0
    return max(0.0, numerator / denom)


################
# linear calcs #
################


@njit
def calc_min_qty_linear(qty_step: float, min_qty: float, min_cost: float, price: float) -> float:
    return max(min_qty, round_up(min_cost / price, qty_step))


@njit
def calc_long_pnl_linear(entry_price: float, close_price: float, qty: float) -> float:
    return abs(qty) * (close_price - entry_price)


@njit
def calc_shrt_pnl_linear(entry_price: float, close_price: float, qty: float) -> float:
    return abs(qty) * (entry_price - close_price)


@njit
def calc_cost_linear(qty: float, price: float) -> float:
    return abs(qty * price)


@njit
def calc_margin_cost_linear(leverage: float, qty: float, price: float) -> float:
    return calc_cost_linear(qty, price) / leverage


@njit
def calc_max_pos_size_linear(leverage: float, balance: float, price: float) -> float:
    return (balance / price) * leverage


@njit
def calc_min_order_qty_linear(qty_step: float, min_qty: float, min_cost: float,
                              qty_pct: float, leverage: float, balance: float,
                              price: float) -> float:
    return calc_min_order_qty(calc_min_qty_linear(qty_step, min_qty, min_cost, price),
                              qty_step,
                              (balance * leverage) / price,
                              qty_pct)


@njit
def calc_cross_hedge_liq_price_binance_linear(balance: float,
                                              long_pos_size: float,
                                              long_pos_price: float,
                                              shrt_pos_size: float,
                                              shrt_pos_price: float,
                                              leverage: float) -> float:
    abs_long_pos_size = abs(long_pos_size)
    abs_shrt_pos_size = abs(shrt_pos_size)
    long_pos_price = long_pos_price if long_pos_price == long_pos_price else 0.0
    shrt_pos_price = shrt_pos_price if shrt_pos_price == shrt_pos_price else 0.0
    long_pos_margin = abs_long_pos_size * long_pos_price / leverage
    shrt_pos_margin = abs_shrt_pos_size * shrt_pos_price / leverage
    mml = 0.006
    mms = 0.006
    # tmm = max(long_pos_margin, shrt_pos_margin)
    tmm = long_pos_margin + shrt_pos_margin
    numerator = (balance - tmm + long_pos_margin + shrt_pos_margin -
                 abs_long_pos_size * long_pos_price + abs_shrt_pos_size * shrt_pos_price)
    denom = (abs_long_pos_size * mml + abs_shrt_pos_size * mms - abs_long_pos_size + abs_shrt_pos_size)
    if denom == 0.0:
        return 0.0
    return max(0.0, numerator / denom)


@njit
def calc_cross_hedge_liq_price_bybit_linear(balance: float,
                                            long_psize: float,
                                            long_pprice: float,
                                            shrt_psize: float,
                                            shrt_pprice: float,
                                            leverage: float) -> float:
    
    raise Exception('bybit linear not yet implemented')


@njit
def calc_available_margin_linear(contract_multiplier: float,
                                 leverage: float,
                                 balance: float,
                                 long_psize: float,
                                 long_pprice: float,
                                 shrt_psize: float,
                                 shrt_pprice: float,
                                 last_price: float) -> float:
    used_margin = 0.0
    equity = balance
    if long_pprice and long_psize:
        equity += calc_long_pnl_linear(long_pprice, last_price, long_psize)
        used_margin += calc_cost_linear(long_psize, long_pprice) / leverage
    if shrt_pprice and shrt_psize:
        equity += calc_shrt_pnl_linear(shrt_pprice, last_price, shrt_psize)
        used_margin += calc_cost_linear(shrt_psize, shrt_pprice) / leverage
    return equity - used_margin



@njit
def calc_stop_loss_linear(qty_step: float,
                          min_qty: float,
                          min_cost: float,
                          contract_multiplier: float,
                          qty_pct: float,
                          leverage: float,
                          stop_loss_liq_diff: float,
                          stop_loss_pos_pct: float,
                          balance: float,
                          long_psize: float,
                          long_pprice: float,
                          shrt_psize: float,
                          shrt_pprice: float,
                          liq_price: float,
                          highest_bid: float,
                          lowest_ask: float,
                          last_price: float,
                          available_margin: float,
                          do_long: bool = True,
                          do_shrt: bool = True,):
    abs_shrt_psize = abs(shrt_psize)
    if calc_diff(liq_price, last_price) < stop_loss_liq_diff:
        if long_psize > abs_shrt_psize:
            stop_loss_qty = min(long_psize,
                                max(calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct,
                                                              leverage, balance, lowest_ask),
                                    round_dn(long_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase short pos, otherwise, reduce long pos
            margin_cost = calc_margin_cost_linear(leverage, stop_loss_qty, lowest_ask)
            if margin_cost < available_margin and do_shrt:
                # add to shrt pos
                new_shrt_psize = round_(shrt_psize - stop_loss_qty, qty_step)
                shrt_pprice = (nan_to_0(shrt_pprice) * (shrt_psize / new_shrt_psize) +
                               lowest_ask * (-stop_loss_qty / new_shrt_psize))
                shrt_psize = new_shrt_psize
                return -stop_loss_qty, lowest_ask, shrt_psize, shrt_pprice, 'stop_loss_shrt_entry'
                available_margin -= margin_cost
            else:
                # reduce long pos
                long_psize = round_(long_psize - stop_loss_qty, qty_step)
                return -stop_loss_qty, lowest_ask, long_psize, long_pprice, 'stop_loss_long_close'
                available_margin += margin_cost
        else:
            stop_loss_qty = min(abs_shrt_psize,
                                max(calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct,
                                                              leverage, balance, highest_bid),
                                    round_dn(abs_shrt_psize * stop_loss_pos_pct, qty_step)))
            # if sufficient margin available, increase long pos, otherwise, reduce shrt pos
            margin_cost = calc_margin_cost_linear(leverage, stop_loss_qty, highest_bid)
            if margin_cost < available_margin and do_long:
                # add to long pos
                new_long_psize = round_(long_psize + stop_loss_qty, qty_step)
                long_pprice = (nan_to_0(long_pprice) * (long_psize / new_long_psize) +
                               highest_bid * (stop_loss_qty / new_long_psize))
                long_psize = new_long_psize
                return stop_loss_qty, highest_bid, long_psize, long_pprice, 'stop_loss_long_entry'
                available_margin -= margin_cost
            else:
                # reduce shrt pos
                shrt_psize = round_(shrt_psize + stop_loss_qty, qty_step)
                return stop_loss_qty, highest_bid, shrt_psize, shrt_pprice, 'stop_loss_shrt_close'
                available_margin += margin_cost
    return 0.0, 0.0, 0.0, 0.0, ''


@njit
def iter_entries_linear(price_step: float,
                        qty_step: float,
                        min_qty: float,
                        min_cost: float,
                        contract_multiplier: float,
                        ddown_factor: float,
                        qty_pct: float,
                        leverage: float,
                        grid_spacing: float,
                        grid_coefficient: float,
                        ema_spread: float,
                        stop_loss_liq_diff: float,
                        stop_loss_pos_pct: float,
                        balance: float,
                        long_psize: float,
                        long_pprice: float,
                        shrt_psize: float,
                        shrt_pprice: float,
                        liq_price: float,
                        highest_bid: float,
                        lowest_ask: float,
                        ema: float,
                        last_price: float,
                        do_long: bool = True,
                        do_shrt: bool = True,):
    # yields both long and short entries
    # also yields stop loss orders if triggered
    # (qty, price, new_psize, new_pprice, comment)

    available_margin = calc_available_margin_linear(contract_multiplier, leverage, balance,
                                                    long_psize, long_pprice,
                                                    shrt_psize, shrt_pprice, last_price)

    stop_loss_order = calc_stop_loss_linear(qty_step, min_qty, min_cost, contract_multiplier, qty_pct,
                                            leverage, stop_loss_liq_diff,
                                            stop_loss_pos_pct, balance, long_psize, long_pprice,
                                            shrt_psize, shrt_pprice, liq_price, highest_bid,
                                            lowest_ask, last_price, available_margin, do_long,
                                            do_shrt)

    if stop_loss_order[0] != 0.0:
        yield stop_loss_order

    while True:
        long_entry = calc_next_long_entry_linear(
            price_step, qty_step, min_qty, min_cost, contract_multiplier, ddown_factor, qty_pct,
            leverage, grid_spacing, grid_coefficient, ema_spread, balance, long_psize, long_pprice,
            shrt_psize, highest_bid, ema, available_margin
        ) if do_long else (0.0, np.nan, long_psize, long_pprice, 'long_entry')

        shrt_entry = calc_next_shrt_entry_linear(
            price_step, qty_step, min_qty, min_cost, contract_multiplier, ddown_factor, qty_pct,
            leverage, grid_spacing, grid_coefficient, ema_spread, balance, long_psize, shrt_psize,
            shrt_pprice, lowest_ask, ema, available_margin
        ) if do_shrt else (0.0, np.nan, shrt_psize, shrt_pprice, 'shrt_entry')

        if long_entry[0] > 0.0:
            if shrt_entry[0] == 0.0:
                long_first = True
            else:
                long_first = calc_diff(long_entry[1], last_price) < calc_diff(shrt_entry[1], last_price)
        elif shrt_entry[0] < 0.0:
            long_first = False
        else:
            break
        if long_first:
            yield long_entry
            long_psize = long_entry[2]
            long_pprice = long_entry[3]
            available_margin -= calc_margin_cost_linear(leverage, long_entry[0], long_entry[1])
        else:
            yield shrt_entry
            shrt_psize = shrt_entry[2]
            shrt_pprice = shrt_entry[3]
            available_margin -= calc_margin_cost_linear(leverage, shrt_entry[0], shrt_entry[1])


@njit
def calc_next_long_entry_linear(price_step: float,
                                qty_step: float,
                                min_qty: float,
                                min_cost: float,
                                contract_multiplier: float,
                                ddown_factor: float,
                                qty_pct: float,
                                leverage: float,
                                grid_spacing: float,
                                grid_coefficient: float,
                                ema_spread: float,
                                balance: float,
                                long_psize: float,
                                long_pprice: float,
                                shrt_psize: float,
                                highest_bid: float,
                                ema: float,
                                available_margin: float):
    if long_psize == 0.0:
        price = min(highest_bid, round_dn(ema * (1 - ema_spread), price_step))
        long_qty = min(round_dn((available_margin / price) * leverage, qty_step),
                       calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct, leverage,
                                                 balance, price))
        if long_qty < calc_min_qty_linear(qty_step, min_qty, min_cost, price):
            long_qty = 0.0
        long_pprice = price
        return long_qty, price, long_qty, long_pprice, 'initial_long_entry'
    else:
        long_pmargin = calc_margin_cost_linear(leverage, long_psize, long_pprice)
        price = min(round_(highest_bid, price_step),
                    calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, long_pmargin, long_pprice))
        if price <= 0.0:
            return 0.0, np.nan, long_psize, long_pprice, 'long_reentry'
        max_order_qty = round_dn((available_margin / price) * leverage, qty_step)
        min_long_order_qty = calc_min_qty_linear(qty_step, min_qty, min_cost, price)
        long_qty = calc_reentry_qty(qty_step, ddown_factor, min_long_order_qty,
                                    max_order_qty,
                                    long_psize)
        if long_qty >= min_long_order_qty:
            new_long_psize = round_(long_psize + long_qty, qty_step)
            long_pprice = long_pprice * (long_psize / new_long_psize) + \
                          price * (long_qty / new_long_psize)
            return long_qty, price, new_long_psize, long_pprice, 'long_reentry'
        else:
            return 0.0, np.nan, long_psize, long_pprice, 'long_reentry'


@njit
def calc_next_shrt_entry_linear(price_step: float,
                                qty_step: float,
                                min_qty: float,
                                min_cost: float,
                                contract_multiplier: float,
                                ddown_factor: float,
                                qty_pct: float,
                                leverage: float,
                                grid_spacing: float,
                                grid_coefficient: float,
                                ema_spread: float,
                                balance: float,
                                long_psize: float,
                                shrt_psize: float,
                                shrt_pprice: float,
                                lowest_ask: float,
                                ema: float,
                                available_margin: float):
    if shrt_psize == 0.0:
        price = max(lowest_ask, round_up(ema * (1 + ema_spread), price_step))
        shrt_qty = min(round_dn(available_margin / price * leverage, qty_step),
                       calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct, leverage,
                                                 balance, price))
        if shrt_qty < calc_min_qty_linear(qty_step, min_qty, min_cost, price):
            shrt_qty = 0.0
        shrt_pprice = price
        return -shrt_qty, price, -shrt_qty, shrt_pprice, 'initial_shrt_entry'
    else:
        pos_margin = calc_margin_cost_linear(leverage, shrt_psize, shrt_pprice)
        price = max(round_(lowest_ask, price_step),
                    calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, pos_margin, shrt_pprice))
        min_order_qty = calc_min_qty_linear(qty_step, min_qty, min_cost, price)
        max_order_qty = round_dn((available_margin / price) * leverage, qty_step)
        qty = calc_reentry_qty(qty_step, ddown_factor, min_order_qty, max_order_qty, shrt_psize)
        if qty >= min_order_qty:
            new_pos_size = shrt_psize - qty
            shrt_pprice = nan_to_0(shrt_pprice) * (shrt_psize / new_pos_size) + price * (-qty / new_pos_size)
            return -qty, price, round_(new_pos_size, qty_step), shrt_pprice, 'shrt_reentry'
        else:
            return 0.0, np.nan, shrt_psize, shrt_pprice, 'shrt_reentry'


@njit
def iter_long_closes_linear(price_step: float,
                            qty_step: float,
                            min_qty: float,
                            min_cost: float,
                            contract_multiplier: float,
                            qty_pct: float,
                            leverage: float,
                            min_markup: float,
                            markup_range: float,
                            n_orders: int,
                            balance: float,
                            pos_size: float,
                            pos_price: float,
                            lowest_ask: float):

    # yields tuple (qty, price, new_pos_size)

    if pos_size == 0.0:
        return

    minm = pos_price * (1 + min_markup)
    prices = np.linspace(minm, pos_price * (1 + min_markup + markup_range), int(n_orders))
    prices = [p for p in sorted(set([round_up(p_, price_step) for p_ in prices]))
              if p >= lowest_ask]
    if len(prices) == 0:
        yield -pos_size, max(lowest_ask, round_up(minm, price_step)), 0.0
    else:
        n_orders = int(min([n_orders, len(prices), pos_size / min_qty]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = -min(pos_size, max(calc_min_order_qty_linear(qty_step, min_qty, min_cost,
                                                                   qty_pct, leverage,
                                                                   balance, lowest_ask),
                                         round_up(pos_size / n_orders, qty_step)))
                if pos_size != 0.0 and -qty / pos_size > 0.75:
                    qty = -pos_size
            if qty == 0.0:
                break
            pos_size = round_(pos_size + qty, qty_step)
            yield qty, price, pos_size
            lowest_ask = price
            n_orders -= 1
        if pos_size > 0.0:
            yield -pos_size, max(lowest_ask, round_up(minm, price_step)), 0.0


@njit
def iter_shrt_closes_linear(price_step: float,
                            qty_step: float,
                            min_qty: float,
                            min_cost: float,
                            contract_multiplier: float,
                            qty_pct: float,
                            leverage: float,
                            min_markup: float,
                            markup_range: float,
                            n_orders: int,
                            balance: float,
                            pos_size: float,
                            pos_price: float,
                            highest_bid: float):
    
    # yields tuple (qty, price, new_pos_size)

    if pos_size == 0.0:
        return

    abs_pos_size = abs(pos_size)
    minm = pos_price * (1 - min_markup)

    prices = np.linspace(minm, pos_price * (1 - (min_markup + markup_range)), int(n_orders))
    prices = [p for p in sorted(set([round_dn(p_, price_step) for p_ in prices]), reverse=True)
              if p <= highest_bid]

    if len(prices) == 0:
        yield abs_pos_size, min(highest_bid, round_dn(minm, price_step)), 0.0
        abs_pos_size = 0.0
    else:
        n_orders = int(min([n_orders, len(prices), abs_pos_size / min_qty]))
        for price in prices:
            if n_orders == 0:
                break
            else:
                qty = min(abs_pos_size, max(calc_min_order_qty_linear(qty_step, min_qty, min_cost,
                                                                      qty_pct, leverage,
                                                                      balance, highest_bid),
                                            round_up(abs_pos_size / n_orders, qty_step)))
                if abs_pos_size != 0.0 and qty / abs_pos_size > 0.75:
                    qty = abs_pos_size
            if qty == 0.0:
                break
            abs_pos_size = round_(abs_pos_size - qty, qty_step)
            yield qty, price, abs_pos_size * -1
            highest_bid = price
            n_orders -= 1
        if abs_pos_size > 0.0:
            yield abs_pos_size, min(highest_bid, round_dn(minm, price_step)), 0.0


##################
##################


@njit
def calc_long_reentry_price(price_step: float,
                            grid_spacing: float,
                            grid_coefficient: float,
                            balance: float,
                            pos_margin: float,
                            pos_price: float) -> float:
    modified_grid_spacing = grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)
    return round_dn(pos_price * (1 - modified_grid_spacing), price_step)


@njit
def calc_shrt_reentry_price(price_step: float,
                            grid_spacing: float,
                            grid_coefficient: float,
                            balance: float,
                            pos_margin: float,
                            pos_price: float) -> float:
    modified_grid_spacing = grid_spacing * (1 + (pos_margin / balance) * grid_coefficient)
    return round_up(pos_price * (1 + modified_grid_spacing), price_step)


@njit
def calc_min_order_qty(min_qty: float,
                       qty_step: float,
                       leveraged_balance_ito_contracts: float,
                       qty_pct: float) -> float:
    return max(min_qty, round_dn(leveraged_balance_ito_contracts * qty_pct, qty_step))


@njit
def calc_reentry_qty(qty_step: float,
                     ddown_factor: float,
                     min_order_qty: float,
                     max_order_qty: float,
                     pos_size: float) -> float:
    qty_available = max(0.0, round_dn(max_order_qty, qty_step))
    return min(qty_available, max(min_order_qty, round_dn(abs(pos_size) * ddown_factor, qty_step)))


def make_get_filepath(filepath: str) -> str:
    '''
    if not is path, creates dir and subdirs for path, returns path
    '''
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_key_secret(exchange: str, user: str) -> (str, str):
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:

            # If we need to get the `market` key:
            # market = keyfile[user]["market"]
            # print("The Market Type is " + str(market))

            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]

            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        print('\n' + line, end=' ')
    elif r:
        print('\r' + line, end=' ')
    else:
        print(line)
    return line


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def filter_orders(actual_orders: [dict],
                  ideal_orders: [dict],
                  keys: [str] = ['symbol', 'side', 'qty', 'price']) -> ([dict], [dict]):
    # returns (orders_to_delete, orders_to_create)

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []
    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_orders_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_orders_cropped = [{k: o[k] for k in keys} for o in actual_orders]
    for ioc, io in zip(ideal_orders_cropped, ideal_orders):
        matches = [(aoc, ao) for aoc, ao in zip(actual_orders_cropped, actual_orders) if aoc == ioc]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_orders_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(io)
    return actual_orders, orders_to_create


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


class Bot:
    def __init__(self, user: str, settings: dict):
        self.settings = settings
        self.user = user

        for key in settings:
            setattr(self, key, settings[key])

        self.ema_alpha = 2 / (settings['ema_span'] + 1)
        self.ema_alpha_ = 1 - self.ema_alpha

        self.ts_locked = {'cancel_orders': 0, 'decide': 0, 'update_open_orders': 0,
                          'update_position': 0, 'print': 0, 'create_orders': 0}
        self.ts_released = {k: 1 for k in self.ts_locked}

        self.position = {}
        self.open_orders = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0
        self.ob = [0.0, 0.0]
        self.ema = 0.0
        self.fills = []

        self.hedge_mode = True
        self.contract_multiplier = 1.0

        self.log_filepath = make_get_filepath(f"logs/{self.exchange}/{settings['config_name']}.log")

        self.my_trades = []
        self.my_trades_cache_filepath = \
            make_get_filepath(os.path.join('historical_data', self.exchange, 'my_trades',
                                           self.symbol, 'my_trades.txt'))

        self.log_level = 0

        self.stop_websocket = False

    def dump_log(self, data) -> None:
        if self.settings['logging_level'] > 0:
            with open(self.log_filepath, 'a') as f:
                f.write(json.dumps({**{'log_timestamp': time()}, **data}) + '\n')

    async def update_open_orders(self) -> None:
        if self.ts_locked['update_open_orders'] > self.ts_released['update_open_orders']:
            return
        try:
            open_orders = await self.fetch_open_orders()
            self.highest_bid, self.lowest_ask = 0.0, 9.9e9
            for o in open_orders:
                if o['side'] == 'buy':
                    self.highest_bid = max(self.highest_bid, o['price'])
                elif o['side'] == 'sell':
                    self.lowest_ask = min(self.lowest_ask, o['price'])
            if self.open_orders != open_orders:
                self.dump_log({'log_type': 'open_orders', 'data': open_orders})
            self.open_orders = open_orders
            self.ts_released['update_open_orders'] = time()
        except Exception as e:
            print('error with update open orders', e)

    async def update_position(self) -> None:
        # also updates open orders
        if self.ts_locked['update_position'] > self.ts_released['update_position']:
            return
        self.ts_locked['update_position'] = time()
        try:
            position, _ = await asyncio.gather(self.fetch_position(),
                                               self.update_open_orders())
            position['used_margin'] = \
                ((self.cost_f(position['long']['size'], position['long']['price'])
                  if position['long']['price'] else 0.0) +
                 (self.cost_f(position['shrt']['size'], position['shrt']['price'])
                  if position['shrt']['price'] else 0.0)) / self.leverage
            position['available_margin'] = (position['equity'] - position['used_margin']) * 0.9
            position['long']['liq_diff'] = calc_diff(position['long']['liquidation_price'], self.price)
            position['shrt']['liq_diff'] = calc_diff(position['shrt']['liquidation_price'], self.price)
            if self.position != position:
                self.dump_log({'log_type': 'position', 'data': position})
            self.position = position
            self.ts_released['update_position'] = time()
        except Exception as e:
            print('error with update position', e)

    async def create_orders(self, orders_to_create: [dict]) -> dict:
        if self.ts_locked['create_orders'] > self.ts_released['create_orders']:
            return
        self.ts_locked['create_orders'] = time()
        creations = []
        for oc in sorted(orders_to_create, key=lambda x: x['qty']):
            try:
                creations.append((oc, asyncio.create_task(self.execute_order(oc))))
            except Exception as e:
                print_(['error creating order a', oc, e], n=True)
        created_orders = []
        for oc, c in creations:
            try:
                o = await c
                created_orders.append(o)
                if 'side' in o:
                    print_([' created order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error creating order b', o, oc], n=True)
                self.dump_log({'log_type': 'create_order', 'data': o})
            except Exception as e:
                print_(['error creating order c', oc, c.exception(), e], n=True)
                self.dump_log({'log_type': 'create_order', 'data': {'result': str(c.exception()),
                                                                    'error': repr(e), 'data': oc}})
        self.ts_released['create_orders'] = time()
        return created_orders

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if self.ts_locked['cancel_orders'] > self.ts_released['cancel_orders']:
            return
        self.ts_locked['cancel_orders'] = time()
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletions.append((oc,
                                  asyncio.create_task(self.execute_cancellation(oc))))
            except Exception as e:
                print_(['error cancelling order a', oc, e])
        canceled_orders = []
        for oc, c in deletions:
            try:
                o = await c
                canceled_orders.append(o)
                if 'side' in o:
                    print_(['cancelled order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error cancelling order', o], n=True)
                self.dump_log({'log_type': 'cancel_order', 'data': o})
            except Exception as e:
                print_(['error cancelling order b', oc, c.exception(), e], n=True)
                self.dump_log({'log_type': 'cancel_order', 'data': {'result': str(c.exception()),
                                                                    'error': repr(e), 'data': oc}})
        self.ts_released['cancel_orders'] = time()
        return canceled_orders

    def stop(self) -> None:
        self.stop_websocket = True

    def calc_orders(self):
        last_price_diff_limit = 0.15
        balance = self.position['wallet_balance'] * 0.98
        long_psize = self.position['long']['size']
        long_pprice = self.position['long']['price']
        shrt_psize = self.position['shrt']['size']
        shrt_pprice = self.position['shrt']['price']

        if self.hedge_mode:
            do_long = self.do_long or long_psize != 0.0
            do_shrt = self.do_shrt or shrt_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and shrt_psize == 0.0
            do_long = (no_pos and self.do_long) or long_psize != 0.0
            do_shrt = (no_pos and self.do_shrt) or shrt_psize != 0.0

        liq_price = self.position['long']['liquidation_price'] if long_psize > abs(shrt_psize) \
            else self.position['shrt']['liquidation_price']

        long_entry_orders, shrt_entry_orders, long_close_orders, shrt_close_orders = [], [], [], []
        stop_loss_close = False

        for tpl in self.iter_entries(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                     liq_price, self.ob[0], self.ob[1], self.ema, self.price,
                                     do_long, do_shrt):
            if (len(long_entry_orders) >= self.n_entry_orders and
                len(shrt_entry_orders) >= self.n_entry_orders) or \
                    calc_diff(tpl[1], self.price) > last_price_diff_limit:
                break
            if tpl[4] == 'stop_loss_shrt_close':
                shrt_close_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(tpl[0]),
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': True,
                                          'custom_id': tpl[4]})
                shrt_psize = tpl[2]
                stop_loss_close = True
            elif tpl[4] == 'stop_loss_long_close':
                long_close_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(tpl[0]),
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': True,
                                          'custom_id': tpl[4]})
                long_psize = tpl[2]
                stop_loss_close = True
            elif tpl[0] > 0.0:
                long_entry_orders.append({'side': 'buy', 'position_side': 'long', 'qty': tpl[0],
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                          'custom_id': tpl[4]})
            else:
                shrt_entry_orders.append({'side': 'sell', 'position_side': 'shrt', 'qty': abs(tpl[0]),
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                          'custom_id': tpl[4]})

        for ask_qty, ask_price, _ in self.iter_long_closes(balance, long_psize, long_pprice, self.ob[1]):
            if len(long_close_orders) >= self.n_entry_orders or \
                    calc_diff(ask_price, self.price) > last_price_diff_limit or \
                    stop_loss_close:
                break
            long_close_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(ask_qty),
                                      'price': float(ask_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})

        for bid_qty, bid_price, _ in self.iter_shrt_closes(balance, shrt_psize, shrt_pprice, self.ob[0]):
            if len(shrt_close_orders) >= self.n_entry_orders or \
                    calc_diff(bid_price, self.price) > last_price_diff_limit or \
                    stop_loss_close:
                break
            shrt_close_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(bid_qty),
                                      'price': float(bid_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})
        return long_entry_orders + shrt_entry_orders + long_close_orders + shrt_close_orders

    async def cancel_and_create(self):
        await asyncio.sleep(0.01)
        await self.update_position()
        await asyncio.sleep(0.01)
        if any([self.ts_locked[k_] > self.ts_released[k_]
                for k_ in [x for x in self.ts_locked if x != 'decide']]):
            return
        n_orders_limit = 4
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_orders(),
                                             keys=['side', 'position_side', 'qty', 'price'])
        to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x['price'], self.price))
        to_create = sorted(to_create, key=lambda x: calc_diff(x['price'], self.price))
        tasks = []
        if to_cancel:
            tasks.append(self.cancel_orders(to_cancel[:n_orders_limit]))
        tasks.append(self.create_orders(to_create[:n_orders_limit]))
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(0.01)
        await self.update_position()
        if any(results):
            print()
        return results

    async def decide(self):
        if self.price <= self.highest_bid:
            self.ts_locked['decide'] = time()
            print_(['bid maybe taken'], n=True)
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if self.price >= self.lowest_ask:
            self.ts_locked['decide'] = time()
            print_(['ask maybe taken'], n=True)
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_locked['decide'] > 5:
            self.ts_locked['decide'] = time()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_released['print'] >= 0.5:
            self.ts_released['print'] = time()
            line = f"{self.symbol} "
            line += f"long {self.position['long']['size']} @ "
            line += f"{round_(self.position['long']['price'], self.price_step)} "
            long_closes = sorted([o for o in self.open_orders if o['side'] == 'sell'
                                  and o['position_side'] == 'long'], key=lambda x: x['price'])
            long_entries = sorted([o for o in self.open_orders if o['side'] == 'buy'
                                   and o['position_side'] == 'long'], key=lambda x: x['price'])
            line += f"close @ {long_closes[0]['price'] if long_closes else 0.0} "
            line += f"enter @ {long_entries[-1]['price'] if long_entries else 0.0} "
            line += f"|| shrt {self.position['shrt']['size']} @ "
            line += f"{round_(self.position['shrt']['price'], self.price_step)} "
            shrt_closes = sorted([o for o in self.open_orders if o['side'] == 'buy'
                                  and o['position_side'] == 'shrt'], key=lambda x: x['price'])
            shrt_entries = sorted([o for o in self.open_orders if o['side'] == 'sell'
                                   and o['position_side'] == 'shrt'], key=lambda x: x['price'])
            line += f"close @ {shrt_closes[-1]['price'] if shrt_closes else 0.0} "
            line += f"enter @ {shrt_entries[0]['price'] if shrt_entries else 0.0} "
            line += f"|| last {self.price} ema {round_(self.ema, self.price_step)} "
            print_([line], r=True)

    def load_cached_my_trades(self) -> [dict]:
        if os.path.exists(self.my_trades_cache_filepath):
            with open(self.my_trades_cache_filepath) as f:
                mtd = {(t := json.loads(line))['order_id']: t for line in f.readlines()}
            return sorted(mtd.values(), key=lambda x: x['timestamp'])
        return []

    async def update_my_trades(self):
        mt = await self.fetch_my_trades()
        if self.my_trades:
            mt = [e for e in mt if e['timestamp'] >= self.my_trades[-1]['timestamp']]
            if mt[0]['order_id'] == self.my_trades[-1]['order_id']:
                mt = mt[1:]
        with open(self.my_trades_cache_filepath, 'a') as f:
            for t in mt:
                f.write(json.dumps(t) + '\n')
        self.my_trades += mt

    def flush_stuck_locks(self, timeout: float = 4.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now

    async def update_fills(self):
        try:
            fills = self.fetch_my_fills()
            self.fills = fills
        except Exception as e:
            print('failed to fetch fills', e)


async def start_bot(bot):
    await bot.start_websocket()


async def create_binance_bot(user: str, settings: str):
    from binance import BinanceBot
    bot = BinanceBot(user, settings)
    await bot._init()
    return bot


async def create_bybit_bot(user: str, settings: str):
    from bybit import Bybit
    bot = Bybit(user, settings)
    await bot._init()
    return bot


async def _start_telegram(account: dict, bot: Bot):
    try:
        telegram = telegram_bot.Telegram(token=account['telegram']['token'],
                                         chat_id=account['telegram']['chat_id'], bot=bot)
        msg = f'<b>Passivbot started</b>'
        telegram.send_msg(msg=msg)

        telegram.show_config()
        return telegram
    except Exception as e:
        print(e, 'failed to initialize telegram')
        return

async def main() -> None:
    try:
        accounts = json.load(open('api-keys.json'))
    except Exception as e:
        print(e, 'failed to load api-keys.json file')
        return
    if sys.argv[1] in accounts:
        account = accounts[sys.argv[1]]
    else:
        print('unrecognized account name', sys.argv[1])
        return
    try:
        config = json.load(open(sys.argv[3]))
    except Exception as e:
        print(e, 'failed to load config', sys.argv[3])
        return
    config['symbol'] = sys.argv[2]

    if account['exchange'] == 'binance':
        bot = await create_binance_bot(sys.argv[1], config)
    elif account['exchange'] == 'bybit':
        bot = await create_bybit_bot(sys.argv[1], config)
    else:
        raise Exception('unknown exchange', account['exchange'])
    print('using config')
    print(json.dumps(config, indent=4))

    if config['telegram']:
        await _start_telegram(account=account, bot=bot)
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())
