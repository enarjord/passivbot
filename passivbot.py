import ccxt.async_support as ccxt_async
import json
import os
import datetime
import numpy as np
import pandas as pd
import pprint
import asyncio
import traceback
import sys
from time import time, sleep
from typing import Iterator, Tuple


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
def calc_max_order_qty_inverse(leverage: float,
                               balance: float,
                               long_pos_size: float,
                               shrt_pos_size: float,
                               price: float) -> float:
    return calc_max_pos_size_inverse(leverage, balance, price) - (long_pos_size + abs(shrt_pos_size))


@njit
def calc_min_order_qty_inverse(qty_step: float, min_qty: float, min_cost: float,
                               qty_pct: float, leverage: float, balance: float,
                               price: float) -> float:
    return calc_min_order_qty(calc_min_qty_inverse(qty_step, min_qty, min_cost, price),
                              qty_step,
                              balance * leverage * price,
                              qty_pct)

@njit
def iter_entries_inverse(price_step: float,
                         qty_step: float,
                         min_qty: float,
                         min_cost: float,
                         ddown_factor: float,
                         qty_pct: float,
                         leverage: float,
                         grid_spacing: float,
                         grid_coefficient: float,
                         balance: float,
                         long_psize: float,
                         long_pprice: float,
                         shrt_psize: float,
                         shrt_pprice: float,
                         highest_bid: float,
                         lowest_ask: float,
                         last_price: float,
                         do_long: bool = True,
                         do_shrt: bool = True,):
    # yields both long and short entries
    # (long_qty, long_price, new_long_psize, new_long_pprice,
    #  shrt_qty, shrt_price, new_shrt_psize, new_shrt_pprice)
    while True:
        long_entry = calc_next_long_entry_inverse(
            price_step, qty_step, min_qty, min_cost, ddown_factor, qty_pct, leverage, grid_spacing,
            grid_coefficient, balance, long_psize, long_pprice, shrt_psize, highest_bid
        ) if do_long else (0.0, np.nan, long_psize, long_pprice, 0)
        shrt_entry = calc_next_shrt_entry_inverse(
            price_step, qty_step, min_qty, min_cost, ddown_factor, qty_pct, leverage, grid_spacing,
            grid_coefficient, balance, long_psize, shrt_psize, shrt_pprice, lowest_ask
        ) if do_shrt else (0.0, np.nan, shrt_psize, shrt_pprice, 0)
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
        else:
            yield shrt_entry
            shrt_psize = shrt_entry[2]
            shrt_pprice = shrt_entry[3]


@njit
def calc_next_long_entry_inverse(price_step: float,
                                 qty_step: float,
                                 min_qty: float,
                                 min_cost: float,
                                 ddown_factor: float,
                                 qty_pct: float,
                                 leverage: float,
                                 grid_spacing: float,
                                 grid_coefficient: float,
                                 balance: float,
                                 long_psize: float,
                                 long_pprice: float,
                                 shrt_psize: float,
                                 highest_bid: float,):
    if long_psize == 0.0:
        long_qty = calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct, leverage,
                                             balance, highest_bid)
        long_price = highest_bid
        long_psize = round_(long_qty, qty_step)
        long_pprice = long_price
        return long_qty, long_price, long_psize, long_pprice, 0
    else:
        long_pmargin = calc_margin_cost_inverse(leverage, long_psize, long_pprice)
        long_price = min(round_(highest_bid, price_step),
                         calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                                 balance, long_pmargin, long_pprice))

        min_long_order_qty = calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct,
                                                       leverage, balance, long_price)
        long_qty = calc_reentry_qty(qty_step, ddown_factor, min_long_order_qty,
                                    calc_max_order_qty_inverse(leverage, balance, long_psize,
                                                              shrt_psize, long_price),
                                    long_psize)
        if long_qty >= min_long_order_qty:
            new_long_psize = round_(long_psize + long_qty, qty_step)
            long_pprice = long_pprice * (long_psize / new_long_psize) + \
                long_price * (long_qty / new_long_psize)
            long_psize = new_long_psize
            return long_qty, long_price, long_psize, long_pprice, 1
        else:
            return 0.0, np.nan, long_psize, long_pprice, 1

@njit
def calc_next_shrt_entry_inverse(price_step: float,
                                 qty_step: float,
                                 min_qty: float,
                                 min_cost: float,
                                 ddown_factor: float,
                                 qty_pct: float,
                                 leverage: float,
                                 grid_spacing: float,
                                 grid_coefficient: float,
                                 balance: float,
                                 long_psize: float,
                                 shrt_psize: float,
                                 shrt_pprice: float,
                                 lowest_ask: float,):
    if shrt_psize == 0.0:
        qty = -calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct, leverage,
                                         balance, lowest_ask)
        price = lowest_ask
        shrt_psize = qty
        shrt_pprice = price
        return qty, price, round_(shrt_psize, qty_step), shrt_pprice, 0
    else:
        pos_margin = calc_margin_cost_inverse(leverage, shrt_psize, shrt_pprice)
        price = max(round_(lowest_ask, price_step),
                    calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, pos_margin, shrt_pprice))
        min_order_qty = -calc_min_order_qty_inverse(qty_step, min_qty, min_cost, qty_pct,
                                                   leverage, balance, price)
        qty = -calc_reentry_qty(qty_step, ddown_factor, abs(min_order_qty),
                                calc_max_order_qty_inverse(leverage, balance, long_psize, shrt_psize,
                                                          price),
                                shrt_psize)
        if qty <= min_order_qty:
            new_pos_size = shrt_psize + qty
            shrt_pprice = shrt_pprice * (shrt_psize / new_pos_size) + price * (qty / new_pos_size)
            shrt_psize = new_pos_size
            margin_cost = calc_margin_cost_inverse(leverage, qty, price)
            return qty, price, round_(shrt_psize, qty_step), shrt_pprice, 1
        else:
            return 0.0, np.nan, shrt_psize, shrt_pprice, 1


@njit
def iter_long_closes_inverse(price_step: float,
                             qty_step: float,
                             min_qty: float,
                             min_cost: float,
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
def iter_shrt_closes_inverse(price_step: float,
                             qty_step: float,
                             min_qty: float,
                             min_cost: float,
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
    rhs = -(balance - order_margin - (pos_size / pos_price) * mm - \
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
    rhs = -(balance - order_margin - (_pos_size / pos_price) * mm - \
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
def calc_max_order_qty_linear(leverage: float,
                              balance: float,
                              long_pos_size: float,
                              shrt_pos_size: float,
                              price: float) -> float:
    return calc_max_pos_size_linear(leverage, balance, price) - (long_pos_size + abs(shrt_pos_size))


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
    #tmm = max(long_pos_margin, shrt_pos_margin)
    tmm = long_pos_margin + shrt_pos_margin
    numerator = (balance - tmm + long_pos_margin + shrt_pos_margin -
                 abs_long_pos_size * long_pos_price + abs_shrt_pos_size * shrt_pos_price)
    denom = (abs_long_pos_size * mml + abs_shrt_pos_size * mms - abs_long_pos_size + abs_shrt_pos_size)
    if denom == 0.0:
        return 0.0
    return max(0.0, numerator / denom)


@njit
def iter_entries_linear(price_step: float,
                        qty_step: float,
                        min_qty: float,
                        min_cost: float,
                        ddown_factor: float,
                        qty_pct: float,
                        leverage: float,
                        grid_spacing: float,
                        grid_coefficient: float,
                        balance: float,
                        long_psize: float,
                        long_pprice: float,
                        shrt_psize: float,
                        shrt_pprice: float,
                        highest_bid: float,
                        lowest_ask: float,
                        last_price: float,
                        do_long: bool = True,
                        do_shrt: bool = True,):
    # yields both long and short entries
    # (long_qty, long_price, new_long_psize, new_long_pprice,
    #  shrt_qty, shrt_price, new_shrt_psize, new_shrt_pprice)
    while True:
        if do_long:
            long_entry = calc_next_long_entry_linear(
                price_step, qty_step, min_qty, min_cost, ddown_factor, qty_pct, leverage, grid_spacing,
                grid_coefficient, balance, long_psize, long_pprice, shrt_psize, highest_bid
            )
        else:
            long_entry = (0.0, np.nan, long_psize, long_pprice, 0)
        if do_shrt:
            shrt_entry = calc_next_shrt_entry_linear(
                price_step, qty_step, min_qty, min_cost, ddown_factor, qty_pct, leverage, grid_spacing,
                grid_coefficient, balance, long_psize, shrt_psize, shrt_pprice, lowest_ask
            )
        else:
            shrt_entry = (0.0, np.nan, shrt_psize, shrt_pprice, 0)
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
        else:
            yield shrt_entry
            shrt_psize = shrt_entry[2]
            shrt_pprice = shrt_entry[3]


@njit
def calc_next_long_entry_linear(price_step: float,
                                qty_step: float,
                                min_qty: float,
                                min_cost: float,
                                ddown_factor: float,
                                qty_pct: float,
                                leverage: float,
                                grid_spacing: float,
                                grid_coefficient: float,
                                balance: float,
                                long_psize: float,
                                long_pprice: float,
                                shrt_psize: float,
                                highest_bid: float,):
    if long_psize == 0.0:
        long_qty = calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct, leverage,
                                             balance, highest_bid)
        long_price = highest_bid
        long_psize = round_(long_qty, qty_step)
        long_pprice = long_price
        return long_qty, long_price, long_psize, long_pprice, 0
    else:
        long_pmargin = calc_margin_cost_linear(leverage, long_psize, long_pprice)
        long_price = min(round_(highest_bid, price_step),
                         calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                                 balance, long_pmargin, long_pprice))
        if long_price <= 0.0:
            return 0.0, np.nan, long_psize, long_pprice, 1
        min_long_order_qty = calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct,
                                                       leverage, balance, long_price)
        long_qty = calc_reentry_qty(qty_step, ddown_factor, min_long_order_qty,
                                    calc_max_order_qty_linear(leverage, balance, long_psize,
                                                              shrt_psize, long_price),
                                    long_psize)
        if long_qty >= min_long_order_qty:
            new_long_psize = round_(long_psize + long_qty, qty_step)
            long_pprice = long_pprice * (long_psize / new_long_psize) + \
                long_price * (long_qty / new_long_psize)
            long_psize = new_long_psize
            return long_qty, long_price, long_psize, long_pprice, 1
        else:
            return 0.0, np.nan, long_psize, long_pprice, 1

@njit
def calc_next_shrt_entry_linear(price_step: float,
                                qty_step: float,
                                min_qty: float,
                                min_cost: float,
                                ddown_factor: float,
                                qty_pct: float,
                                leverage: float,
                                grid_spacing: float,
                                grid_coefficient: float,
                                balance: float,
                                long_psize: float,
                                shrt_psize: float,
                                shrt_pprice: float,
                                lowest_ask: float,):
    if shrt_psize == 0.0:
        qty = -calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct, leverage,
                                         balance, lowest_ask)
        price = lowest_ask
        shrt_psize = qty
        shrt_pprice = price
        return qty, price, round_(shrt_psize, qty_step), shrt_pprice, 0
    else:
        pos_margin = calc_margin_cost_linear(leverage, shrt_psize, shrt_pprice)
        price = max(round_(lowest_ask, price_step),
                    calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                            balance, pos_margin, shrt_pprice))
        min_order_qty = -calc_min_order_qty_linear(qty_step, min_qty, min_cost, qty_pct,
                                                   leverage, balance, price)
        qty = -calc_reentry_qty(qty_step, ddown_factor, abs(min_order_qty),
                                calc_max_order_qty_linear(leverage, balance, long_psize, shrt_psize,
                                                          price),
                                shrt_psize)
        if qty <= min_order_qty:
            new_pos_size = shrt_psize + qty
            shrt_pprice = shrt_pprice * (shrt_psize / new_pos_size) + price * (qty / new_pos_size)
            shrt_psize = new_pos_size
            margin_cost = calc_margin_cost_linear(leverage, qty, price)
            return qty, price, round_(shrt_psize, qty_step), shrt_pprice, 1
        else:
            return 0.0, np.nan, shrt_psize, shrt_pprice, 1


@njit
def iter_long_closes_linear(price_step: float,
                            qty_step: float,
                            min_qty: float,
                            min_cost: float,
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

            # Get the keys, format as a string list for ccxt, then return
            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]

            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


def init_ccxt(exchange: str = None, user: str = None):
    if user is None:
        cc = getattr(ccxt_async, exchange)
    try:
        cc = getattr(ccxt_async, exchange)({'apiKey': (ks := load_key_secret(exchange, user))[0],
                                            'secret': ks[1]})
    except Exception as e:
        print('error init ccxt', e)
        cc = getattr(ccxt_async, exchange)
    #print('ccxt enableRateLimit true')
    #cc.enableRateLimit = True
    return cc


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


def load_live_settings(exchange: str, user: str = 'default', do_print=True) -> dict:
    settings = json.load(open(f'live_configs/{exchange}_default.json'))
    if do_print:
        print('\nloaded settings:')
        print(json.dumps(settings, indent=4))
    return settings


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
        self.symbol = settings['symbol']
        self.leverage = settings['leverage']
        self.grid_coefficient = settings['grid_coefficient']
        self.grid_spacing = settings['grid_spacing']
        self.markup_range = settings['markup_range']
        self.min_markup = settings['min_markup']
        self.n_entry_orders = settings['n_entry_orders']
        self.n_close_orders = settings['n_close_orders']
        self.qty_pct = settings['qty_pct']
        self.ddown_factor = settings['ddown_factor']
        self.do_long = settings['do_long']
        self.do_shrt = settings['do_shrt']

        self.ema_alpha = 2 / (settings['ema_span'] + 1)
        self.ema_alpha_ = 1 - self.ema_alpha
        self.ema_spread = settings['ema_spread']

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
                print_([' created order', o['symbol'], o['side'], o['position_side'], o['qty'],
                        o['price']], n=True)
                self.dump_log({'log_type': 'create_order', 'data': o})
            except Exception as e:
                print_(['error creating order b', oc, c.exception(), e], n=True)
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
                print_(['cancelled order', o['symbol'], o['side'], o['position_side'], o['qty'],
                        o['price']], n=True)
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

        if self.hedge_mode:
            do_long = self.do_long or self.position['long']['size'] != 0.0
            do_shrt = self.do_shrt or self.position['shrt']['size'] != 0.0
        else:
            no_pos = self.position['long']['size'] == 0.0 and self.position['shrt']['size'] == 0.0
            do_long = (no_pos and self.do_long) or self.position['long']['size'] != 0.0
            do_shrt = (no_pos and self.do_shrt) or self.position['shrt']['size'] != 0.0
        highest_bid = min(self.ob[0], round_dn(self.ema * (1 - self.ema_spread), self.price_step)) \
            if self.position['long']['size'] == 0.0 else self.ob[0]
        lowest_ask = max(self.ob[1], round_up(self.ema * (1 + self.ema_spread), self.price_step)) if \
            self.position['shrt']['size'] == 0.0 else self.ob[1]

        entry_orders = []

        for tpl in self.iter_entries(balance,
                                     self.position['long']['size'],
                                     self.position['long']['price'],
                                     self.position['shrt']['size'],
                                     self.position['shrt']['price'],
                                     highest_bid, lowest_ask, self.price, do_long, do_shrt):
            if len(entry_orders) >= self.n_entry_orders * 2 or \
                    calc_diff(tpl[1], self.price) > last_price_diff_limit:
                break
            if tpl[0] > 0.0:
                entry_orders.append({'side': 'buy', 'position_side': 'long', 'qty': tpl[0],
                                     'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                     'custom_id': 'entry'})
            else:
                entry_orders.append({'side': 'sell', 'position_side': 'shrt', 'qty': abs(tpl[0]),
                                     'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                     'custom_id': 'entry'})

        long_close_orders = []
        for ask_qty, ask_price, _ in self.iter_long_closes(balance, self.position['long']['size'],
                                                           self.position['long']['price'],
                                                           self.ob[1]):
            if len(long_close_orders) >= self.n_entry_orders or \
                    calc_diff(ask_price, self.price) > last_price_diff_limit:
                break
            long_close_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(ask_qty),
                                      'price': float(ask_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})
        shrt_close_orders = []
        for bid_qty, bid_price, _ in self.iter_shrt_closes(balance, self.position['shrt']['size'],
                                                           self.position['shrt']['price'],
                                                           self.ob[0]):
            if len(shrt_close_orders) >= self.n_entry_orders or \
                    calc_diff(bid_price, self.price) > last_price_diff_limit:
                break
            shrt_close_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(bid_qty),
                                      'price': float(bid_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})
        return entry_orders + long_close_orders + shrt_close_orders


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
        bot = await create_binance_bot(sys.argv[1], config)
    else:
        raise Exception('unknown exchange', account['exchange'])
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())
