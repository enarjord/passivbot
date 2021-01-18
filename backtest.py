import sys
import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings, \
    sort_dict_keys, round_up, round_dn, round_, calc_long_closes, calc_shrt_closes, \
    calc_long_reentry_price, calc_shrt_reentry_price, calc_diff, calc_default_qty, \
    calc_entry_qty
from binance import fetch_trades as binance_fetch_trades
from bybit import fetch_trades as bybit_fetch_trades
from bybit import calc_cross_long_liq_price as bybit_calc_cross_long_liq_price
from bybit import calc_cross_shrt_liq_price as bybit_calc_cross_shrt_liq_price
from binance import calc_cross_long_liq_price as binance_calc_cross_long_liq_price
from binance import calc_cross_shrt_liq_price as binance_calc_cross_shrt_liq_price

from typing import Iterator


def backtest(df: pd.DataFrame, settings: dict):

    grid_spacing = settings['grid_spacing']
    grid_coefficient = settings['grid_coefficient']
    price_step = settings['price_step']
    qty_step = settings['qty_step']
    inverse = settings['inverse']
    break_on_loss = settings['break_on_loss']
    liq_diff_threshold = settings['liq_diff_threshold']
    stop_loss_pos_reduction = settings['stop_loss_pos_reduction']
    min_qty = settings['min_qty']
    grid_step = settings['grid_step']
    ddown_factor = settings['ddown_factor']

    leverage = settings['leverage']
    margin_limit = settings['margin_limit']
    compounding = settings['compounding']

    if inverse:
        calc_cost = lambda qty_, price_: qty_ / price_
        calc_liq_price = lambda balance_, pos_size_, pos_price_: \
            bybit_calc_cross_shrt_liq_price(balance_, pos_size_, pos_price_) \
                if pos_size_ < 0.0 else \
                bybit_calc_cross_long_liq_price(balance_, pos_size_, pos_price_)
        if settings['default_qty'] <= 0.0:
            calc_default_qty_ = lambda balance_, last_price: \
                calc_default_qty(min_qty, qty_step, balance_ * last_price, settings['default_qty'])
        else:
            calc_default_qty_ = lambda balance_, last_price: settings['default_qty']
        calc_max_pos_size = lambda margin_limit_, price_: margin_limit_ * price_ * leverage
    else:
        calc_cost = lambda qty_, price_: qty_ * price_
        calc_liq_price = lambda balance_, pos_size_, pos_price_: \
            binance_calc_cross_shrt_liq_price(balance_, pos_size_, pos_price_, leverage=leverage) \
                if pos_size_ < 0.0 else \
                binance_calc_cross_long_liq_price(balance_, pos_size_, pos_price_, leverage=leverage)
        if settings['default_qty'] <= 0.0:
            calc_default_qty_ = lambda balance_, last_price: \
                calc_default_qty(min_qty, qty_step, balance_ / last_price, settings['default_qty'])
        else:
            calc_default_qty_ = lambda balance_, last_price: settings['default_qty']
        calc_max_pos_size = lambda margin_limit_, price_: margin_limit_ / price_ * leverage


    if settings['dynamic_grid']:
        calc_long_initial_bid = lambda highest_bid: highest_bid
        calc_shrt_initial_ask = lambda lowest_ask: lowest_ask
        calc_long_reentry_price_ = lambda balance_, pos_margin_, pos_price_, highest_bid_: \
            min(highest_bid_, calc_long_reentry_price(price_step, grid_spacing, grid_coefficient,
                                               balance_, pos_margin_, pos_price_))
        calc_shrt_reentry_price_ = lambda balance_, pos_margin_, pos_price_, lowest_ask_: \
            max(lowest_ask_, calc_shrt_reentry_price(price_step, grid_spacing, grid_coefficient,
                                                     balance_, pos_margin_, pos_price_))
    else:
        calc_long_initial_bid = lambda highest_bid: round_dn(highest_bid, grid_step)
        calc_shrt_initial_ask = lambda lowest_ask: round_up(lowest_ask, grid_step)
        calc_long_reentry_price_ = lambda balance_, pos_margin_, pos_price_, highest_bid_: \
            round_dn(min(pos_price_ - 9e-9, highest_bid_), grid_step)
        calc_shrt_reentry_price_ = lambda balance_, pos_margin_, pos_price_, lowest_ask_: \
            round_up(max(pos_price_ + 9e-9, lowest_ask_), grid_step)


    balance = margin_limit
    print('default_qty', calc_default_qty_(balance, df.price.iloc[0]))

    maker_fee = settings['maker_fee']
    taker_fee = settings['taker_fee']

    min_markup = settings['min_markup']
    max_markup = settings['max_markup']
    print('dynamic grid' if settings['dynamic_grid'] else 'static grid')
    n_close_orders = settings['n_close_orders']


    trades = []

    ob = [df.iloc[:2].price.min(), df.iloc[:2].price.max()]

    pos_size = 0.0
    pos_price = 0.0

    bid_price = ob[0]
    ask_price = ob[1]

    liq_price = 0.0

    pnl_sum = 0.0

    for row in df.itertuples():
        if row.buyer_maker:
            if pos_size == 0.0:                                     # no pos
                bid_qty = calc_default_qty_(balance, ob[0])
                bid_price = calc_long_initial_bid(ob[0])
            elif pos_size > 0.0:                                    # long pos
                if calc_diff(liq_price, row.price) < liq_diff_threshold:
                    # limit reached; enter no more
                    bid_qty = 0.0
                    bid_price = 0.0
                else:                                               # long reentry
                    pos_margin = calc_cost(pos_size, pos_price) / leverage
                    bid_price = calc_long_reentry_price_(balance, pos_margin, pos_price, ob[0])
                    max_pos_size = calc_max_pos_size(balance, bid_price)
                    bid_qty = calc_entry_qty(qty_step, ddown_factor,
                                             calc_default_qty_(balance, ob[0]), max_pos_size,
                                             pos_size)
            else:                                                   # shrt pos
                if row.price <= pos_price:                          # shrt close
                    qtys, prices = calc_shrt_closes(price_step, qty_step, min_qty, min_markup,
                                                    max_markup, pos_size, pos_price, ob[0],
                                                    n_close_orders)
                    bid_qty = qtys[0]
                    bid_price = prices[0]
                elif calc_diff(liq_price, row.price) < liq_diff_threshold:
                    if break_on_loss:
                        print('shrt break on loss')
                        return []
                    # controlled shrt loss
                    bid_qty = round_up(-pos_size * stop_loss_pos_reduction, qty_step)
                    if settings['market_stop_loss']:
                        bid_price = round_up(row.price + 9e-9, price_step) # market order
                        cost = calc_cost(bid_qty, bid_price)
                        margin_cost = cost / leverage
                        gain = (pos_price / bid_price - 1)
                        pnl = cost * gain - cost * taker_fee
                        pos_size += bid_qty
                        roe = gain * leverage
                        liq_price = calc_liq_price(balance, pos_size, pos_price)
                        trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'stop_loss',
                                       'price': bid_price, 'qty': bid_qty, 'pnl': pnl,
                                       'pos_size': pos_size, 'pos_price': pos_price, 'roe': roe,
                                       'margin_cost': margin_cost, 'liq_price': liq_price})
                        pnl_sum += pnl
                        if compounding:
                            balance = max(margin_limit, balance + pnl)
                        continue
                    bid_price = ob[0]
                else:                                               # no shrt close
                    bid_qty = 0.0
                    bid_price = 0.0
            ob[0] = row.price
            if pos_size > 0.0 and liq_price and row.price < liq_price:
                print('long liquidation', liq_price)
                return []
            if row.price < bid_price:
                if pos_size >= 0.0:
                    # create or add to long pos
                    cost = calc_cost(bid_qty, bid_price)
                    margin_cost = cost / leverage
                    pnl = -cost * maker_fee
                    new_pos_size = pos_size + bid_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        bid_price * (bid_qty / new_pos_size)
                    pos_size = new_pos_size
                    liq_price = calc_liq_price(balance, pos_size, pos_price)
                    trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'entry',
                                   'price': bid_price, 'qty': bid_qty, 'pnl': pnl,
                                   'pos_size': pos_size, 'pos_price': pos_price, 'roe': np.nan,
                                   'margin_cost': margin_cost, 'liq_price': liq_price})
                    pnl_sum += pnl
                    if compounding:
                        balance = max(margin_limit, balance + pnl)
                    line = f'\r{row.Index / len(df):.2f} pnl sum {pnl_sum:.6f} '
                    liq_diff = abs(liq_price - row.price) / row.price
                    line += f'balance {balance:.6f} '
                    line += f'liq diff {liq_diff:.2f} '
                    line += f'qty {calc_default_qty_(balance, ob[0]):.3f} '
                    line += f'pos_size {pos_size:.3f} '
                    print(line, end='    ')
                else:
                    # close shrt pos
                    cost = calc_cost(bid_qty, bid_price)
                    margin_cost = cost / leverage
                    gain = (pos_price / bid_price - 1)
                    pnl = cost * gain - cost * maker_fee
                    pos_size += bid_qty
                    roe = gain * leverage
                    liq_price = calc_liq_price(balance, pos_size, pos_price)
                    trades.append({'trade_id': row.Index, 'side': 'shrt',
                                   'type': 'close' if gain > 0.0 else 'stop_loss',
                                   'price': bid_price, 'qty': bid_qty, 'pnl': pnl,
                                   'pos_size': pos_size, 'pos_price': pos_price, 'roe': roe,
                                   'margin_cost': margin_cost, 'liq_price': liq_price})
                    pnl_sum += pnl
                    if compounding:
                        balance = max(margin_limit, balance + pnl)
                    line = f'\r{row.Index / len(df):.2f} pnl sum {pnl_sum:.6f} '
                    liq_diff = abs(liq_price - row.price) / row.price

                    line += f'balance {balance:.6f} '
                    line += f'liq diff {liq_diff:.2f} '
                    line += f'qty {calc_default_qty_(balance, ob[0]):.3f} '
                    line += f'pos_size {pos_size:.3f} '
                    print(line, end='    ')
        else:
            if pos_size == 0.0:                                      # no pos
                ask_qty = -calc_default_qty_(balance, ob[1])
                ask_price = calc_shrt_initial_ask(ob[1])
            elif pos_size < 0.0:                                     # shrt pos
                if abs(liq_price - row.price) / row.price < liq_diff_threshold:
                    # limit reached; enter no more
                    ask_qty = 0.0
                    ask_price = 9.9e9
                else:                                                # shrt reentry
                    pos_margin = calc_cost(-pos_size, pos_price) / leverage
                    ask_price = calc_shrt_reentry_price_(balance, pos_margin, pos_price, ob[1])
                    max_pos_size = calc_max_pos_size(margin_limit, ask_price)
                    ask_qty = -calc_entry_qty(qty_step, ddown_factor,
                                              calc_default_qty_(balance, ob[1]), max_pos_size,
                                              pos_size)
            else:                                                    # long pos
                if row.price >= pos_price:                           # close long pos
                    qtys, prices = calc_long_closes(price_step, qty_step, min_qty, min_markup,
                                                    max_markup, pos_size, pos_price, ob[1],
                                                    n_close_orders)
                    ask_qty = qtys[0]
                    ask_price = prices[0]
                elif abs(liq_price - row.price) / row.price < liq_diff_threshold:
                    if break_on_loss:
                        print('break on loss')
                        return []
                    # controlled long loss
                    ask_qty = -round_up(pos_size * stop_loss_pos_reduction, qty_step)
                    if settings['market_stop_loss']:
                        ask_price = round_dn(ob[0] - 9e-9, price_step) # market order
                        cost = -calc_cost(ask_qty, ask_price)
                        margin_cost = cost / leverage
                        gain = (ask_price / pos_price - 1)
                        pnl = cost * gain - cost * taker_fee
                        pos_size += ask_qty
                        roe = gain * leverage
                        liq_price = calc_liq_price(balance, pos_size, pos_price)
                        trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'stop_loss',
                                       'price': ask_price, 'qty': ask_qty, 'pnl': pnl,
                                       'pos_size': pos_size, 'pos_price': pos_price, 'roe': roe,
                                       'margin_cost': margin_cost, 'liq_price': liq_price})
                        pnl_sum += pnl
                        if compounding:
                            balance = max(margin_limit, balance + pnl)
                        continue
                    ask_price = ob[1]
                else:                                                # no close
                    ask_qty = 0.0
                    ask_price = 9.9e9
            ob[1] = row.price
            if pos_size < 0.0 and liq_price and row.price > liq_price:
                print('shrt liquidation', row.price, liq_price, pos_size, pos_price, balance)
                return []
            if row.price > ask_price:
                if pos_size <= 0.0:
                    # add to or create short pos
                    cost = -calc_cost(ask_qty, ask_price)
                    margin_cost = cost / leverage
                    pnl = -cost * maker_fee
                    new_pos_size = pos_size + ask_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        ask_price * (ask_qty / new_pos_size)
                    pos_size = new_pos_size
                    liq_price = calc_liq_price(balance, pos_size, pos_price)
                    trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'entry',
                                   'price': ask_price, 'qty': ask_qty, 'pnl': pnl,
                                   'pos_size': pos_size, 'pos_price': pos_price, 'roe': np.nan,
                                   'margin_cost': margin_cost, 'liq_price': liq_price})
                    pnl_sum += pnl
                    if compounding:
                        balance = max(margin_limit, balance + pnl)
                    line = f'\r{row.Index / len(df):.2f} pnl sum {pnl_sum:.6f} '
                    liq_diff = abs(liq_price - row.price) / row.price
                    line += f'balance {balance:.6f} '
                    line += f'liq diff {liq_diff:.2f} '
                    line += f'qty {calc_default_qty_(balance, ob[0]):.3f} '
                    line += f'pos_size {pos_size:.3f} '
                    print(line, end='    ')
                else:
                    # close long pos
                    cost = -calc_cost(ask_qty, ask_price)
                    margin_cost = cost / leverage
                    gain = (ask_price / pos_price - 1)
                    pnl = cost * gain - cost * maker_fee
                    pos_size += ask_qty
                    roe = gain * leverage
                    liq_price = calc_liq_price(balance, pos_size, pos_price)
                    trades.append({'trade_id': row.Index, 'side': 'long',
                                   'type': 'close' if gain > 0.0 else 'stop_loss',
                                   'price': ask_price, 'qty': ask_qty, 'pnl': pnl,
                                   'pos_size': pos_size, 'pos_price': pos_price, 'roe': roe,
                                   'margin_cost': margin_cost, 'liq_price': liq_price})
                    pnl_sum += pnl
                    if compounding:
                        balance = max(margin_limit, balance + pnl)
                    line = f'\r{row.Index / len(df):.2f} pnl sum {pnl_sum:.6f} '
                    liq_diff = abs(liq_price - row.price) / row.price
                    line += f'balance {balance:.6f} '
                    line += f'liq diff {liq_diff:.2f} '
                    line += f'qty {calc_default_qty_(balance, ob[0]):.3f} '
                    line += f'pos_size {pos_size:.3f} '
                    print(line, end='    ')
    return trades


def format_dict(d: dict):
    r = ''
    for key in sorted(d):
        r += f'&{key}={round(d[key], 10) if type(d[key]) in [float, int] else str(d[key])}'
    return r[1:]

def unformat_dict(d: str):
    kv = d.split('&')
    kvs = [kv.split('=') for kv in d.split('&')]
    result = {}
    for kv in kvs:
        try:
            result[kv[0]] = eval(kv[1])
        except:
            result[kv[0]] = kv[1]
    return result


def jackrabbit(df: pd.DataFrame,
               backtesting_settings: dict,
               ranges: dict,
               starting_candidate: dict = None):
    if starting_candidate is None:
        # randomized starting settings
        best = {key: calc_new_val((ranges[key][1] - ranges[key][0]) / 2, ranges[key], 1.0)
                for key in sorted(ranges)}
    else:
        best = sort_dict_keys(starting_candidate)

    n_days = backtesting_settings['n_days']
    results = {}
    best_gain = -9e9
    candidate = best

    ks = 130
    k = 0
    ms = np.array([1/(i/2 + 16) for i in range(ks)])
    ms = ((ms - ms.min()) / (ms.max() - ms.min()))
    base_filepath = make_get_filepath(
        os.path.join('backtesting_results', backtesting_settings['exchange'],
                     ts_to_date(time())[:19],
                     int(round(n_days)),
                     '')
    )
    trades_filepath = make_get_filepath(os.path.join(base_filepath, 'trades', ''))
    json.dump(backtesting_settings, open(base_filepath + 'backtesting_settings.json', 'w'),
              indent=4, sort_keys=True)

    print(backtesting_settings)

    while k < ks - 1:

        if candidate['min_markup'] >= candidate['max_markup']:
            candidate['min_markup'] = candidate['max_markup']

        k += 1
        settings_ = {**backtesting_settings, **candidate}
        key = format_dict(candidate)
        if key in results:
            print('\nskipping', key)
            candidate = get_new_candidate(ranges, best)
            continue
        print(f'k={k}, m={ms[k]:.4f} candidate', candidate)
        trades = backtest(df, settings_)
        if not trades:
            print('\nno trades')
            candidate = get_new_candidate(ranges, best)
            continue
        tdf = pd.DataFrame(trades).set_index('trade_id')
        tdf.to_csv(trades_filepath + key + '.csv')
        closest_liq = ((tdf.price - tdf.liq_price).abs() / tdf.price).min()
        biggest_pos_size = tdf.pos_size.abs().max()
        n_closes = len(tdf[tdf.type == 'close'])
        pnl_sum = tdf.pnl.sum()
        loss_sum = tdf[tdf.type == 'stop_loss'].pnl.sum()
        abs_pos_sizes = tdf.pos_size.abs()
        if backtesting_settings['inverse']:
            max_margin_cost = (abs_pos_sizes / tdf.pos_price / settings_['leverage']).max()
        else:
            max_margin_cost = (abs_pos_sizes * tdf.pos_price / settings_['leverage']).max()
        gain = (pnl_sum + settings_['margin_limit']) / settings_['margin_limit']
        average_daily_gain = gain ** (1 / n_days)
        n_trades = len(tdf)
        result = {'n_closes': n_closes, 'pnl_sum': pnl_sum, 'loss_sum': loss_sum,
                  'max_margin_cost': max_margin_cost, 'average_daily_gain': average_daily_gain,
                  'gain': gain, 'n_trades': n_trades, 'closest_liq': closest_liq,
                  'biggest_pos_size': biggest_pos_size, 'n_days': n_days}
        print('\n', result)
        results[key] = {**result, **candidate}

        if gain > best_gain:
            best = candidate
            best_gain = gain
            print('\n\n\n###############\nnew best', best, '\n', gain, '\n\n')
            print(settings_)
            print(results[key], '\n\n')
        candidate = get_new_candidate(ranges, best, m=ms[k])
        pd.DataFrame(results).T.to_csv(base_filepath + 'results.csv')


def prep_df(adf: pd.DataFrame, settings: dict):
    dfc = adf.drop('price', axis=1).join(round_(adf.price, settings['price_step']))
    dfc = dfc[dfc.price != dfc.price.shift(1)]
    if 'side' in dfc.columns:
        # bybit
        buyer_maker = dfc.side == 'Sell'
        buyer_maker.name = 'buyer_maker'
    elif 'is_buyer_maker' in dfc.columns:
        # binance
        buyer_maker = dfc.is_buyer_maker
        buyer_maker.name = 'buyer_maker'
    else:
        raise Exception('trades of unknown format')
    df = pd.concat([dfc.price, buyer_maker], axis=1)
    df.index = np.arange(len(df))
    return df


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
    filepath = os.path.join('historical_data', exchange, 'agg_trades_futures', symbol, '')

    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')],
                           key=lambda x: int(x.replace('.csv', '')))
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
            print('           to', id_)
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
        filepath = make_get_filepath(os.path.join('historical_data', exchange, 'agg_trades_futures',
                                                  symbol, ''))
        cache_filepath = make_get_filepath(
            os.path.join('historical_data', exchange, 'agg_trades_futures', symbol + '_cache', '')
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
    base_filepath = os.path.join('backtesting_settings', exchange, '')
    backtesting_settings = json.load(open(os.path.join(base_filepath, 'backtesting_settings.json')))
    symbol = backtesting_settings['symbol']
    n_days = backtesting_settings['n_days']
    ranges = json.load(open(os.path.join(base_filepath, 'ranges.json')))
    print(base_filepath)
    if 'random' in sys.argv:
        print('using randomized starting candidate')
        starting_candidate = None
    else:
        starting_candidate = {k: backtesting_settings[k] for k in ranges}
    trades_filename = f'{symbol}_agg_trades_{exchange}_{n_days}_days_{ts_to_date(time())[:10]}'
    trades_filename += f"_price_step_{str(backtesting_settings['price_step']).replace('.', '_')}"
    trades_filename += ".csv"
    if os.path.exists(trades_filename):
        print('loading cached trade dataframe')
        df = pd.read_csv(trades_filename)
    else:
        agg_trades = await load_trades(exchange, user, symbol, n_days)
        df = prep_df(agg_trades, backtesting_settings)
        df.to_csv(trades_filename)
    jackrabbit(df, backtesting_settings, ranges, starting_candidate)


if __name__ == '__main__':
    asyncio.run(main())

