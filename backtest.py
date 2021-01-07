import sys
import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings, \
    sort_dict_keys, round_up, round_dn, round_
from passivbot import calc_long_entry_price as calc_long_entry_price_
from passivbot import calc_shrt_entry_price as calc_shrt_entry_price_
from passivbot import calc_entry_qty as calc_entry_qty_
from binance import fetch_trades as binance_fetch_trades
from bybit import fetch_trades as bybit_fetch_trades, calc_long_liq_price, calc_shrt_liq_price
from typing import Iterator
from bisect import insort_left


def calc_bid_ask(price_step, pos_price, ob, take_loss=False):
    if take_loss:
        bid_price = round_dn(ob[0], price_step)
        ask_price = round_up(ob[1], price_step)
    else:
        bid_price = round_dn(min(ob[0], pos_price if pos_price else ob[0]), price_step)
        ask_price = round_up(max(ob[1], pos_price), price_step)
    return bid_price, ask_price


def calc_long_closes(price_step: float,
                     qty_step: float,
                     min_qty: float,
                     min_markup: float,
                     max_markup: float,
                     pos_size: float,
                     pos_price: float,
                     lowest_ask: float,
                     n_orders: int = 10):
    prices = round_up(np.linspace(pos_price * (1 + min_markup), pos_price * (1 + max_markup),
                                  min(n_orders, int(pos_size / min_qty))),
                      price_step)
    prices = prices[np.where(prices >= lowest_ask)]
    qtys = round_up(np.repeat(pos_size / len(prices), len(prices)), qty_step)
    qtys_sum = qtys.sum()
    while qtys_sum > pos_size:
        for i in range(len(qtys)):
            qtys[i] = round_(qtys[i] - min_qty, qty_step)
            qtys_sum = round_(qtys_sum - min_qty, qty_step)
            if qtys_sum <= pos_size:
                break
    return qtys, prices


def backtest_controlled_loss(df: pd.DataFrame, settings: dict):

    price_step = settings['price_step']

    entry_qty = settings['entry_qty']
    loss_qty = settings['loss_qty']
    close_qty = settings['close_qty']

    leverage = settings['leverage']
    pos_margin_loss_threshold = settings['pos_margin_loss_threshold']
    maker_fee = -0.00025

    trades = []

    ob = [df.iloc[:2].price.min(), df.iloc[:2].price.max()]

    pos_size = 0.0
    pos_price = 0.0

    bid_price = round_dn(ob[0], price_step)
    ask_price = round_up(ob[1], price_step)


    for row in df.itertuples():
        if row.buyer_maker:
            ob[0] = row.price
            if row.price < bid_price:
                if pos_size >= 0.0:
                    # create or add to long pos
                    cost = bid_qty / bid_price
                    margin_cost = cost * leverage
                    pnl = -cost * maker_fee
                    new_pos_size = pos_size + bid_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        bid_price * (bid_qty / new_pos_size)
                    pos_size = new_pos_size
                    trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'entry',
                                   'price': bid_price, 'qty': bid_qty, 'pnl': pnl, 'pos_size': pos_size,
                                   'pos_price': pos_price, 'roe': np.nan,
                                   'margin_cost': margin_cost})
                else:
                    # close short pos
                    '''
                    if bid_price < pos_price:
                        bid_qty = -pos_size
                    '''
                    cost = bid_qty / bid_price
                    margin_cost = cost * leverage
                    gain = (pos_price / bid_price - 1)
                    pnl = cost * gain - cost * maker_fee
                    pos_size += bid_qty
                    roe = gain * leverage
                    trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'close',
                                   'price': bid_price, 'qty': bid_qty, 'pnl': pnl, 'pos_size': pos_size,
                                   'pos_price': pos_price, 'roe': roe,
                                   'margin_cost': margin_cost})
        else:
            ob[1] = row.price
            if row.price > ask_price:
                if pos_size <= 0.0:
                    # add to or create short pos
                    cost = -ask_qty / ask_price
                    margin_cost = cost * leverage
                    pnl = -cost * maker_fee
                    new_pos_size = pos_size + ask_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        ask_price * (ask_qty / new_pos_size)
                    pos_size = new_pos_size
                    trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'entry',
                                   'price': ask_price, 'qty': ask_qty, 'pnl': pnl, 'pos_size': pos_size,
                                   'pos_price': pos_price, 'roe': np.nan,
                                   'margin_cost': margin_cost})
                else:
                    # close long pos
                    '''
                    if ask_price > pos_price:
                        ask_qty = -pos_size
                    '''
                    cost = -ask_qty / ask_price
                    margin_cost = cost * leverage
                    gain = (ask_price / pos_price - 1)
                    pnl = cost * gain - cost * maker_fee
                    pos_size += ask_qty
                    roe = gain * leverage
                    trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'close',
                                   'price': ask_price, 'qty': ask_qty, 'pnl': pnl, 'pos_size': pos_size,
                                   'pos_price': pos_price, 'roe': roe,
                                   'margin_cost': margin_cost})
        if pos_size == 0.0:
            bid_price = round_dn(ob[0], price_step)
            ask_price = round_up(ob[1], price_step)
            bid_qty, ask_qty = entry_qty, -entry_qty
        elif pos_size > 0.0:
            if pos_size / pos_price > pos_margin_loss_threshold * leverage:
                bid_price = round_dn(ob[0], price_step)
                ask_price = round_up(ob[1], price_step)
                bid_qty, ask_qty = entry_qty, -loss_qty
            else:
                bid_price = round_dn(min(ob[0], pos_price), price_step)
                ask_price = round_up(max(ob[1], pos_price), price_step)
                bid_qty, ask_qty = entry_qty, -close_qty
        else:
            if -pos_size / pos_price > pos_margin_loss_threshold * leverage:
                bid_price = round_dn(ob[0], price_step)
                ask_price = round_up(ob[1], price_step)
                bid_qty, ask_qty = loss_qty, -entry_qty
            else:
                bid_price = round_dn(min(ob[0], pos_price), price_step)
                ask_price = round_up(max(ob[1], pos_price), price_step)
                bid_qty, ask_qty = close_qty, -entry_qty
    return trades

    







def jackrabbit(agg_trades: pd.DataFrame):
    settings = {
        "compounding": False,
        "ddown_factor": 0.2087,
        "grid_spacing": 0.0017,
        "grid_spacing_coefficient": 42.55,
        "margin_limit": 0.0006,
        "isolated_mode": False,
        "leverage": 100.0,
        "liq_modifier": 0.001,
        "maker_fee": -0.00025,
        "markup": 0.002149,
        "min_qty": 1.0,
        "n_days": 0.0,
        "price_step": 0.5,
        "qty_step": 1.0,
        "symbol": "BTCUSD"
    }

    ranges = {
        'ddown_factor': (0.0, 2.0, 0.0001),
        'grid_spacing': (0.0001, 0.1, 0.0001),
        'grid_spacing_coefficient': (0.0, 100.0, 0.01),
        'margin_limit': (0.0001, 0.001, 0.0001),
        'markup': (0.0001, 0.005, 0.000001),
    }

    tweakable = {
        'ddown_factor': 0.0,
        'grid_spacing': 0.0,
        'grid_spacing_coefficient': 0.0,
        'margin_limit': 0.0,
        'markup': 0.0
    }

    best = {}

    for key in tweakable:
        if type(tweakable[key]) == tuple:
            best[key] = tuple(sorted([
                calc_new_val((ranges[key][1] - ranges[key][0]) / 2, ranges[key], 1.0)
                for _ in tweakable[key]
            ]))
        else:
            best[key] = calc_new_val((ranges[key][1] - ranges[key][0]) / 2, ranges[key], 1.0)

    best = {k_: settings[k_] for k_ in sorted(ranges)}
    # optional: uncomment to manually set starting settings.

    settings = sort_dict_keys(settings)
    best = sort_dict_keys(best)

    results = {}
    best_gain = -99999999
    candidate = best

    ks = 200
    k = 0
    ms = np.array([1/(i/2 + 16) for i in range(ks)])
    ms = ((ms - ms.min()) / (ms.max() - ms.min()))

    results_filename = make_get_filepath(
        f'jackrabbit_results_grid/{ts_to_date(time())[:19]}'
    )

    n_days = (agg_trades.timestamp.iloc[-1] - agg_trades.timestamp.iloc[0]) / 1000 / 60 / 60 / 24
    settings['n_days'] = n_days
    print('n_days', n_days)

    # conditions for result approval
    conditions = [
        lambda r: r['n_trades'] > 100 * n_days,
        lambda r: r['n_liqs'] == 0,
        lambda r: True,
    ]

    df = prep_df(agg_trades)

    while k < ks - 1:
        try:
            k += 1
            key = tuple([candidate[k_] for k_ in sorted(candidate)])
            if key in results:
                print('skipping', key)
                candidate = get_new_candidate(ranges, best)
                continue
            line = f'\n{k} m={ms[k]:.4f} best {tuple(best.values())}, '
            line += f'candidate {tuple(candidate.values())}'
            print(line)
            settings_ = {k_: candidate[k_] if k_ in candidate else settings[k_]
                         for k_ in sorted(settings)}
            trades = backtest(df, settings_)
            if not trades:
                print('\nno trades')
                candidate = get_new_candidate(ranges, best)
                continue
            tdf = pd.DataFrame(trades).set_index('trade_id')
            n_liqs = len(tdf[tdf.type == 'liq'])
            n_closes = len(tdf[tdf.type == 'close'])
            pnl_sum = tdf.pnl.sum()
            max_margin_cost = tdf.margin_cost.max()
            gain = (pnl_sum + max_margin_cost) / max_margin_cost
            n_trades = len(tdf)
            result = {'n_liqs': n_liqs, 'n_closes': n_closes, 'pnl_sum': pnl_sum,
                      'max_margin_cost': max_margin_cost,
                      'gain': gain, 'n_trades': n_trades}
            print('\n', result)
            results[key] = result

            if gain > best_gain and all([c(results[key]) for c in conditions]):
                best = candidate
                best_gain = gain
                print('\n\nnew best', best, '\n', gain, '\n')
                print(settings_)
                print(results[key], '\n\n')
            candidate = get_new_candidate(ranges, best, m=ms[k])
            pd.DataFrame(results).T.to_csv(results_filename + '.csv')
        except KeyboardInterrupt:
            return results
    return results


def prep_df(adf: pd.DataFrame):
    # bybit
    dfc = adf[adf.price != adf.price.shift(1)]
    buyer_maker = dfc.side == 'Sell'
    buyer_maker.name = 'buyer_maker'
    df = pd.concat([dfc.price, buyer_maker], axis=1)
    df.index = np.arange(len(df))
    return df



def calc_long_closes_old(min_bet: float,
                         amount_step: float,
                         price_step: float,
                         markup: float,
                         pos_amount: float,
                         pos_price: float):
    n = 10
    prices = round_up(np.linspace(pos_price,
                                  round_up(pos_price * (1 + markup),
                                  price_step), n), price_step)
    if pos_amount < n:
        amounts = np.zeros(n)
        remainder = int(pos_amount)
    else:
        amounts = round_dn(np.linspace(min_bet, pos_amount / n * 2 - 1, n), amount_step)
        remainder = pos_amount % amounts.sum()
    print(amounts, remainder)
    for i in range(n - 1, -1, -1):
        if remainder < min_bet:
            break
        amounts[i] += min_bet
        remainder -= min_bet
    return list(zip(-amounts, prices))


def calc_shrt_closes_old(min_bet: float,
                         amount_step: float,
                         price_step: float,
                         markup: float,
                         pos_amount: float,
                         pos_price: float):
    n = 10
    prices = round_dn(np.linspace(pos_price,
                                  round_up(pos_price * (1 - markup),
                                  price_step), n), price_step)
    if -pos_amount < n:
        amounts = np.zeros(n)
        remainder = int(-pos_amount)
    else:
        amounts = round_dn(np.linspace(min_bet, -pos_amount / n * 2 - 1, n), amount_step)
        remainder = -pos_amount % amounts.sum()
    for i in range(n - 1, -1, -1):
        amounts[i] += min_bet
        remainder -= min_bet
        if remainder < min_bet:
            break
    return list(zip(amounts, prices))


def backtest(df: pd.DataFrame, settings: dict):

    def calc_long_entry_price(equity_, pos_size_, pos_price_):
        return calc_long_entry_price_(price_step,
                                      leverage,
                                      grid_spacing,
                                      grid_spacing_coefficient,
                                      equity_,
                                      pos_size_,
                                      pos_price_)

    def calc_shrt_entry_price(equity_, pos_size_, pos_price_):
        return calc_shrt_entry_price_(price_step,
                                      leverage,
                                      grid_spacing,
                                      grid_spacing_coefficient,
                                      equity_,
                                      pos_size_,
                                      pos_price_)

    def calc_entry_qty(equity_, pos_size_, pos_price_):
        return calc_entry_qty_(qty_step, min_qty, ddown_factor, leverage,
                               equity_, pos_size_, pos_price_)

    # bybit
    maker_fee = settings['maker_fee']
    min_qty = settings['min_qty']
    qty_step = settings['qty_step']
    price_step = settings['price_step']

    ddown_factor = settings['ddown_factor']
    leverage = settings['leverage']
    markup = settings['markup']
    grid_spacing = settings['grid_spacing']

    compounding = settings['compounding']
    liq_modifier = settings['liq_modifier']
    n_days = settings['n_days']
    isolated_mode = settings['isolated_mode']
    grid_spacing_coefficient = settings['grid_spacing_coefficient']

    pos_size = 0.0
    liq_price = np.nan
    pos_price = 0.0
    equity = settings['margin_limit']
    ob = [df.iloc[:2].price.min(), df.iloc[:2].price.max()]
    bid_price = ob[0]
    bid_qty = min_qty
    ask_price = ob[1]
    ask_qty = -min_qty

    trades = []
    prev_len_trades = 0

    # df needs columns price: float, buyer_maker: bool
    for row in df.itertuples():
        if row.buyer_maker:
            ob[0] = row.price
            if pos_size == 0:
                ask_price, ask_qty = ob[1], -min_qty
            elif pos_size < 0:
                ask_price = calc_shrt_entry_price(equity, pos_size, pos_price)
                ask_qty = -calc_entry_qty(equity, pos_size, pos_price)
            if liq_price and pos_size > 0 and row.price <= liq_price:
                # liquidate long pos
                cost = pos_size / liq_price
                margin_cost = cost / leverage
                pnl = -margin_cost
                if compounding:
                    equity = max(settings['margin_limit'], equity + pnl) # reset equity
                trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'liq',
                               'qty': -pos_size, 'price': liq_price,
                               'pos_size': 0, 'pos_price': np.nan,
                               'liq_price': liq_price, 'pnl': pnl,
                               'margin_cost': margin_cost, 'equity': equity,
                               'equity_available': equity,
                               'close_price': round_up(pos_price * (1 + markup), price_step)})
                liq_price = np.nan
                pos_size = 0
                bid_price, bid_qty = ob[0], min_qty
                ask_price, ask_qty = ob[1], -min_qty
            elif row.price < bid_price:
                if pos_size >= 0:
                    # add more to or create long pos
                    while row.price < bid_price:
                        if bid_qty < min_qty:
                            bid_price, bid_qty = 0.0, 0.0
                            liq_price = calc_long_liq_price(leverage, pos_price) * (1 + liq_modifier)
                            break
                        new_pos_size = pos_size + bid_qty
                        cost = bid_qty / bid_price
                        margin_cost = cost / leverage
                        pnl = -cost * maker_fee
                        if compounding:
                            equity += pnl
                        if isolated_mode:
                            liq_price = calc_long_liq_price(
                                leverage, pos_price) * (1 + liq_modifier
                            )
                        pos_price = pos_price * (pos_size / new_pos_size) + \
                            bid_price * (bid_qty / new_pos_size)
                        pos_size = new_pos_size
                        pos_cost = pos_size / pos_price
                        close_price = round_up(pos_price * (1 + markup), price_step)
                        trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'entry',
                                       'qty': bid_qty, 'price': bid_price,
                                       'pos_size': pos_size, 'pos_price': pos_price,
                                       'liq_price': liq_price, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity - pos_size / row.price / leverage,
                                       'close_price': close_price})
                        bid_price = calc_long_entry_price(equity, pos_size, pos_price)
                        bid_qty = calc_entry_qty(equity, pos_size, pos_price)
                    ask_price = round_up(pos_price * (1 + markup), price_step)
                    ask_qty = -pos_size
                else:
                    # close short pos
                    if row.price < bid_price:
                        cost = -pos_size / bid_price
                        margin_cost = cost / leverage
                        pnl = cost * (pos_price / bid_price - 1) - cost * maker_fee
                        if compounding:
                            equity += pnl
                        liq_price = np.nan
                        trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'close',
                                       'qty': -pos_size, 'price': bid_price,
                                       'pos_size': 0, 'pos_price': np.nan,
                                       'liq_price': np.nan, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity, 'close_price': np.nan})
                        pos_size = 0
                        bid_price, bid_qty = ob[0], min_qty
                        ask_price, ask_qty = ob[1], -min_qty
        else:
            ob[1] = row.price
            if pos_size == 0:
                bid_price, bid_qty = ob[0], min_qty
            elif pos_size > 0:
                bid_price = calc_long_entry_price(equity, pos_size, pos_price)
                bid_qty = calc_entry_qty(equity, pos_size, pos_price)
            if liq_price and pos_size < 0 and row.price > liq_price:
                # liquidate shrt pos
                cost = -pos_size / liq_price
                margin_cost = cost / leverage
                pnl = -margin_cost
                if compounding:
                    equity = max(settings['margin_limit'], equity + pnl)
                trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'liq',
                               'qty': -pos_size, 'price': liq_price,
                               'pos_size': 0, 'pos_price': np.nan,
                               'liq_price': liq_price, 'pnl': pnl,
                               'margin_cost': margin_cost, 'equity': equity,
                               'equity_available': equity,
                               'close_price': round_dn(pos_price * (1 - markup), price_step)})
                liq_price = np.nan
                pos_size = 0
                bid_price, bid_qty = ob[0], min_qty
                ask_price, ask_qty = ob[1], -min_qty
            if row.price > ask_price:
                if pos_size <= 0:
                    # adding more to or creating shrt pos
                    while row.price > ask_price:
                        if -ask_qty < min_qty:
                            ask = [0.0, 9.9e-9]
                            liq_price = calc_shrt_liq_price(leverage, pos_price) * (1 - liq_modifier)
                            break
                        new_pos_size = pos_size + ask_qty
                        cost = -ask_qty / ask_price
                        margin_cost = cost / leverage
                        pnl = -cost * maker_fee
                        if compounding:
                            equity += pnl
                        if isolated_mode:
                            liq_price = calc_shrt_liq_price(
                                leverage, pos_price) * (1 - liq_modifier
                            )
                        pos_price = pos_price * (pos_size / new_pos_size) + \
                            ask_price * (ask_qty / new_pos_size)
                        pos_size = new_pos_size

                        close_price = round_dn(pos_price * (1 - markup), price_step)
                        trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'entry',
                                       'qty': ask_qty, 'price': ask_price,
                                       'pos_size': pos_size, 'pos_price': pos_price,
                                       'liq_price': liq_price, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity + pos_size / row.price / leverage,
                                       'close_price': close_price})
                        ask_price = calc_shrt_entry_price(equity, pos_size, pos_price)
                        ask_qty = -calc_entry_qty(equity, pos_size, pos_price)

                    bid_price = round_dn(pos_price * (1 - markup), price_step)
                    bid_qty = -pos_size
                else:
                    if row.price > ask_price:
                        # close long pos
                        cost = pos_size / ask_price
                        margin_cost = cost / leverage
                        pnl = cost * (ask_price / pos_price - 1) - cost * maker_fee
                        if compounding:
                            equity += pnl
                        trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'close',
                                       'qty': -pos_size, 'price': ask_price,
                                       'pos_size': 0, 'pos_price': np.nan,
                                       'liq_price': np.nan, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity, 'close_price': np.nan})
                        pos_size = 0
                        liq_price = np.nan
                        bid_price, bid_qty = ob[0], min_qty
                        ask_price, ask_qty = ob[1], -min_qty
        if len(trades) > prev_len_trades + 100:
            prev_len_trades = len(trades)
            n_liqs = len([t for t in trades if t['type'] == 'liq'])
            max_margin_cost = max([abs(t['margin_cost']) for t in trades])
            line = f'\r{row.Index / len(df):.2f} '
            line += f"start_equity: {settings['margin_limit']} equity {equity:.6f} "
            line += f"pnl sum {sum([t['pnl'] for t in trades]):.6f} "
            line += f"n liqs {n_liqs} "
            line += f"max margin cost {max_margin_cost:.6f} "
            print(line, end=' ')
            max_n_liqs_per_day = 1
            max_n_liqs = n_days * max_n_liqs_per_day
            if n_liqs > max_n_liqs:
                break
    return trades


def calc_new_val(val, range_, m):
    new_val = val + (np.random.random() - 0.5) * (range_[1] - range_[0]) * max(0.0001, m)
    return round(round(max(min(new_val, range_[1]), range_[0]) / range_[2]) * range_[2], 10)


def get_new_candidate(ranges: dict, best: dict, m=0.2):
    new_candidate = {}
    for key in best:
        if key not in ranges:
            continue
        if type(best[key]) == tuple:
            new_candidate[key] = tuple(sorted([calc_new_val(e, ranges[key], m) for e in best[key]]))
        else:
            new_candidate[key] = calc_new_val(best[key], ranges[key], m)
    return {k_: new_candidate[k_] for k_ in sorted(new_candidate)}


def iter_chunks(exchange: str, symbol: str) -> Iterator[pd.DataFrame]:
    chunk_size = 100000
    filepath = f'historical_data/{exchange}/agg_trades_futures/{symbol}/'
    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')])
        for f in filenames[::-1]:
            chunk = pd.read_csv(filepath + f).set_index('trade_id')
            if chunk is not None:
                print('loaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000))
                yield chunk
            else:
                yield None
        yield None
    else:
        yield None


async def load_trades(exchange: str, user: str, symbol: str, n_days: float) -> pd.DataFrame:

    def skip_ids(id_, ids_):
        if id_ in ids_:
            print('skipping from', id_)
            while id_ in ids_:
                id_ -= 1
            print('           to', from_id)
        return id_

    cc = init_ccxt(exchange, user)
    try:
        if exchange == 'binance':
            fetch_trades_func = binance_fetch_trades
        elif exchange == 'bybit':
            fetch_trades_func = bybit_fetch_trades
        else:
            print(exchange, 'not found')
            return
        filepath = make_get_filepath(f'historical_data/{exchange}/agg_trades_futures/{symbol}/')
        cache_filepath = make_get_filepath(
            f'historical_data/{exchange}/agg_trades_futures/{symbol}_cache/'
        )
        cache_filenames = [f for f in os.listdir(cache_filepath) if f.endswith('.csv')]
        ids = set()
        if cache_filenames:
            print('loading cached trades...')
            cached_trades = pd.concat([pd.read_csv(cache_filepath + f) for f in cache_filenames],
                                      axis=0)
            cached_trades = cached_trades.set_index('trade_id').sort_index()
            cached_trades = cached_trades[~cached_trades.index.duplicated()]
            ids.update(cached_trades.index)
        else:
            cached_trades = None
        age_limit = time() - 60 * 60 * 24 * n_days
        age_limit_millis = age_limit * 1000
        print('age_limit', ts_to_date(age_limit))
        chunk_iterator = iter_chunks(exchange, symbol)
        chunk = next(chunk_iterator)
        chunks = {} if chunk is None else {int(chunk.index[0]): chunk}
        if chunk is not None:
            ids.update(chunk.index)
        min_id = min(ids) if ids else 0
        new_trades = await fetch_trades_func(cc, symbol)
        cached_ids = set()
        k = 0
        while True:
            if new_trades[0]['timestamp'] <= age_limit_millis:
                break
            from_id = new_trades[0]['trade_id'] - 1
            while True:
                if chunk is None:
                    min_id = 0
                    break
                from_id = skip_ids(from_id, ids)
                if from_id < min_id:
                    chunk = next(chunk_iterator)
                    if chunk is None:
                        min_id = 0
                        break
                    else:
                        chunks[int(chunk.index[0])] = chunk
                        ids.update(chunk.index)
                        min_id = min(ids)
                        if chunk.timestamp.max() < age_limit_millis:
                            break
                else:
                    break
            from_id = skip_ids(from_id, ids)
            from_id -= 999
            new_trades = await fetch_trades_func(cc, symbol, from_id=from_id) + new_trades
            k += 1
            if k % 20 == 0:
                print('dumping cache')
                cache_df = pd.DataFrame([t for t in new_trades
                                         if t['trade_id'] not in cached_ids]).set_index('trade_id')
                cache_df.to_csv(cache_filepath + str(int(time() * 1000)) + '.csv')
                cached_ids.update(cache_df.index)
        new_trades_df = pd.DataFrame(new_trades).set_index('trade_id')
        trades_updated = pd.concat(list(chunks.values()) + [new_trades_df, cached_trades], axis=0)
        no_dup = trades_updated[~trades_updated.index.duplicated()]
        no_dup_sorted = no_dup.sort_index()
        chunk_size = 100000
        chunk_ids = no_dup_sorted.index // chunk_size * chunk_size
        for g in no_dup_sorted.groupby(chunk_ids):
            if g[0] not in chunks or len(chunks[g[0]]) != chunk_size:
                print('dumping chunk', g[0])
                g[1].to_csv(f'{filepath}{str(g[0])}.csv')
        for f in [f_ for f_ in os.listdir(cache_filepath) if f_.endswith('.csv')]:
            os.remove(cache_filepath + f)
        await cc.close()
        return no_dup_sorted[no_dup_sorted.timestamp >= age_limit_millis]
    except KeyboardInterrupt:
        await cc.close()


async def main():
    exchange = sys.argv[1]
    user = sys.argv[2]
    n_days = int(sys.argv[3])
    if exchange == 'bybit':
        symbol = 'BTCUSD'
    elif exchange == 'binance':
        symbol = 'BTCUSDT'
    else:
        raise Exception(f'exchange {exchange} not found')
    filename = f'btcusdt_agg_trades_{exchange}_{n_days}_days_{ts_to_date(time())[:10]}.csv'
    if os.path.isfile(filename):
        print('loading trades...')
        adf = pd.read_csv(filename).set_index('trade_id')
    else:
        print('fetching trades')
        agg_trades = await load_trades(exchange, user, symbol, n_days)
        adf = agg_trades.loc[agg_trades.price != agg_trades.price.shift(1)]
        adf.to_csv(filename)
    jackrabbit(adf)


if __name__ == '__main__':
    asyncio.run(main())

