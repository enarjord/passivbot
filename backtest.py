import sys
import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings, \
    sort_dict_keys, calc_long_liq_price, calc_shrt_liq_price, calc_long_entry, calc_shrt_entry, \
    round_up, round_dn, calc_initial_long_entry_qty, calc_initial_shrt_entry_qty
from binance import fetch_trades as binance_fetch_trades
from bybit import fetch_trades as bybit_fetch_trades
from typing import Iterator
from bisect import insort_left


def jackrabbit(agg_trades: pd.DataFrame):
    settings = {
        'ema_spans': (2500, 5000),
        'ema_spread': -0.0004,
        'entry_qty_equity_multiplier': 0.002,
        'entry_qty_scaling_factor': 0.05,
        'grid_spacing': 0.002,
        'initial_equity': 0.001,
        'leverage': 100.0,
        'maker_fee': -0.00025,
        'markup': 0.003,
        'min_qty': 1.0,
        'price_step': 0.5,
        'qty_step': 1.0,
        'symbol': 'BTCUSD'
    }

    ranges = {
        'ema_spans': (50, 300000, 1),
        'ema_spread': (-0.002, 0.002, 0.000001),
        'entry_qty_equity_multiplier': (0.00001, 0.01, 0.00001),
        'entry_qty_scaling_factor': (0.001, 1.0, 0.001),
        'grid_spacing': (0.0001, 0.01, 0.0001),
        'leverage': (10, 100, 1),
        'markup': (0.0001, 0.05, 0.000001),
    }

    best = {
        'ema_spans': (0, 0),
        'ema_spread': 0.0,
        'entry_qty_equity_multiplier': 0.0,
        'entry_qty_scaling_factor': 0.0,
        'grid_spacing': 0.0,
        'leverage': 0.0,
        'markup': 0.0,
    }

    for key in best:
        if type(best[key]) == tuple:
            best[key] = tuple(sorted([calc_new_val((ranges[key][1] - ranges[key][0]) / 2,
                                                   ranges[key], 1.0)
                                      for _ in best[key]]))
        else:
            best[key] = calc_new_val((ranges[key][1] - ranges[key][0]) / 2, ranges[key], 1.0)
    '''
    best = {'bet_m': 0.05,
            'bet_purse_pct': 0.001,
            'ema_spans': (10000, 20000),
            'markup': 0.002,
            'spread': 0.001,
            'price_step': 3.5}
    best = {'ema_spans': (129788.0, 275785.0),
            'ema_spread': 0.001437,
            'markup': 0.002345,
            'grid_spacing': 0.0026,
            'entry_qty_scaling_factor': 0.38,
            'entry_qty_equity_multiplier': 0.00171}
    best = {k_: settings[k_] for k_ in sorted(ranges)}
    '''

    settings = sort_dict_keys(settings)
    best = sort_dict_keys(best)

    results = {}
    best_gain = -99999999
    candidate = best

    ks = 200
    k = 0
    ms = np.array([1/(i/2 + 16) for i in range(ks)])
    ms = ((ms - ms.min()) / (ms.max() - ms.min())) / 2

    results_filename = make_get_filepath(
        f'jackrabbit_results_grid/{ts_to_date(time())[:19]}'
    )

    n_days = (agg_trades.timestamp.iloc[-1] - agg_trades.timestamp.iloc[0]) / 1000 / 60 / 60 / 24
    print('n_days', n_days)

    while k < ks - 1:
        try:
            k += 1
            adf = agg_trades
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
            df = prep_df(adf, settings_)
            trades = backtest(df, settings_)
            tdf = pd.DataFrame(trades).set_index('trade_id')
            n_liqs = len(tdf[tdf.type == 'liq'])
            n_closes = len(tdf[tdf.type == 'close'])
            pnl_sum = tdf.pnl.sum()
            gain = (pnl_sum + settings['initial_equity']) / settings['initial_equity']
            results[key] = {'n_liqs': n_liqs, 'n_closes': n_closes, 'pnl_sum': pnl_sum,
                            'equity_start': settings['initial_equity'], 
                            'equity_end': tdf.equity.iloc[-1],
                            'gain': gain, 'n_trades': len(tdf)}
            print(results[key])
            if not trades:
                print('\nno trades')
                candidate = get_new_candidate(best)
                continue

            if gain > best_gain:
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


def prep_df(adf: pd.DataFrame, settings: dict):
    # bybit
    # assumes consecutive duplicates in price are removed
    spread = settings['ema_spread']
    ema_spans = settings['ema_spans']
    emas = pd.concat([adf.price.ewm(span=span, adjust=False).mean() for span in ema_spans], axis=1)
    bid_thr = emas.min(axis=1) * (1 - spread / 2)
    bid_thr.name = 'bid_thr'
    ask_thr = emas.max(axis=1) * (1 + spread / 2)
    ask_thr.name = 'ask_thr'
    buyer_maker = adf.side == 'Sell'
    buyer_maker.name = 'buyer_maker'
    df = pd.concat([adf.price, buyer_maker, bid_thr, ask_thr], axis=1)
    df.index = np.arange(len(df))
    return df


def calc_long_close(markup: float, price_step: float, pos_size: float, pos_price: float):
    return [-pos_size, round_up(pos_price * (1 + markup), price_step)]


def calc_shrt_close(markup: float, price_step: float, pos_size: float, pos_price: float):
    return [-pos_size, round_dn(pos_price * (1 - markup), price_step)]


def calc_long_closes(min_bet: float,
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


def calc_shrt_closes(min_bet: float,
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

    # bybit
    maker_fee = settings['maker_fee']
    min_qty = settings['min_qty']
    qty_step = settings['qty_step']
    price_step = settings['price_step']

    entry_qty_scaling_factor = settings['entry_qty_scaling_factor']
    leverage = settings['leverage']
    markup = settings['markup']
    grid_spacing = settings['grid_spacing']
    entry_qty_equity_multiplier = settings['entry_qty_equity_multiplier']

    calc_long_entry_ = lambda equity_, pos_size_, pos_price_: calc_long_entry(
        min_qty, qty_step, price_step, leverage, entry_qty_scaling_factor, grid_spacing,
        entry_qty_equity_multiplier, equity_, pos_size_, pos_price_
    )

    calc_shrt_entry_ = lambda equity_, pos_size_, pos_price_: calc_shrt_entry(
        min_qty, qty_step, price_step, leverage, entry_qty_scaling_factor, grid_spacing,
        entry_qty_equity_multiplier, equity_, pos_size_, pos_price_
    )

    calc_long_close_ = lambda pos_size_, pos_price_: calc_long_close(
        markup, price_step, pos_size_, pos_price
    )
    calc_shrt_close_ = lambda pos_size_, pos_price_: calc_shrt_close(
        markup, price_step, pos_size_, pos_price
    )

    calc_initial_long_entry_qty_ = lambda equity_, price_: calc_initial_long_entry_qty(
        min_qty, qty_step, entry_qty_equity_multiplier, equity_, price_
    )

    calc_initial_shrt_entry_qty_ = lambda equity_, price_: calc_initial_shrt_entry_qty(
        min_qty, qty_step, entry_qty_equity_multiplier, equity_, price_
    )

    pos_size = 0.0
    liq_price = 0.0
    pos_price = 0.0
    equity = settings['initial_equity']
    ob = [df.iloc[:2].price.min(), df.iloc[:2].price.max()]
    bid = [calc_initial_long_entry_qty_(equity, ob[0]), ob[0]]
    ask = [calc_initial_shrt_entry_qty_(equity, ob[1]), ob[1]]

    trades = []

    # df needs columns price: float, buyer_maker: bool, bid_thr: float, ask_thr: float
    for row in df.itertuples():
        if row.buyer_maker:
            ob[0] = row.price
            if pos_size == 0:
                ask = [calc_initial_shrt_entry_qty_(equity, ob[1]),
                       max(ob[1], round_up(row.ask_thr, price_step))]
            elif pos_size < 0:
                ask = calc_shrt_entry_(equity, pos_size, pos_price)
                ask[1] = max(ask[1], ob[1])
            if liq_price and pos_size > 0 and row.price <= liq_price:
                # liquidate long pos
                cost = pos_size / liq_price
                margin_cost = cost / leverage
                pnl = -margin_cost
                equity = max(settings['initial_equity'], equity + pnl) # reset equity
                trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'liq',
                               'qty': -pos_size, 'price': liq_price,
                               'pos_size': 0, 'pos_price': np.nan,
                               'liq_price': np.nan, 'pnl': pnl,
                               'margin_cost': margin_cost, 'equity': equity,
                               'equity_available': equity})
                liq_price = 0
                pos_size = 0
                bid = [calc_initial_long_entry_qty_(equity, ob[0]), min(ob[0], row.bid_thr)]
                ask = [calc_initial_shrt_entry_qty_(equity, ob[1]), max(ob[1], row.ask_thr)]
            elif row.price < bid[1]:
                if pos_size >= 0:
                    # add more to or create long pos
                    while row.price < bid[1]:
                        if bid[0] < min_qty:
                            bid = [0.0, 0.0]
                            break
                        new_pos_size = pos_size + bid[0]
                        cost = bid[0] / bid[1]
                        margin_cost = cost / leverage
                        pnl = -cost * maker_fee
                        equity += pnl
                        pos_price = pos_price * (pos_size / new_pos_size) + \
                            bid[1] * (bid[0] / new_pos_size)
                        pos_size = new_pos_size
                        liq_price = calc_long_liq_price(pos_price, leverage)
                        close_price = round_up(pos_price * (1 + markup), price_step)
                        trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'entry',
                                       'qty': bid[0], 'price': bid[1],
                                       'pos_size': pos_size, 'pos_price': pos_price,
                                       'liq_price': liq_price, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity - pos_size / row.price / leverage,
                                       'close_price': close_price})
                        bid = calc_long_entry_(equity, pos_size, bid[1])
                    ask = calc_long_close_(pos_size, pos_price)
                else:
                    # close short pos
                    if row.price < bid[1]:
                        cost = -pos_size / bid[1]
                        margin_cost = cost / leverage
                        pnl = cost * (pos_price / bid[1] - 1) - cost * maker_fee
                        equity += pnl
                        liq_price = 0
                        trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'close',
                                       'qty': -pos_size, 'price': bid[1],
                                       'pos_size': 0, 'pos_price': np.nan,
                                       'liq_price': np.nan, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity, 'close_price': np.nan})
                        pos_size = 0
                        bid = [calc_initial_long_entry_qty_(equity, ob[0]), min(ob[0], row.bid_thr)]
                        ask = [calc_initial_shrt_entry_qty_(equity, ob[1]), max(ob[1], row.ask_thr)]
        else:
            ob[1] = row.price
            if pos_size == 0:
                bid = [calc_initial_long_entry_qty_(equity, ob[0]),
                       round_dn(min(ob[0], row.bid_thr), price_step)]
            elif pos_size > 0:
                bid = calc_long_entry_(equity, pos_size, pos_price)
                bid[1] = min(bid[1], ob[0])
            if liq_price and pos_size < 0 and row.price > liq_price:
                # liquidate shrt pos
                cost = -pos_size / liq_price
                margin_cost = cost / leverage
                pnl = -margin_cost
                equity = max(settings['initial_equity'], equity + pnl)
                trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'liq',
                               'qty': -pos_size, 'price': liq_price,
                               'pos_size': 0, 'pos_price': np.nan,
                               'liq_price': np.nan, 'pnl': pnl,
                               'margin_cost': margin_cost, 'equity': equity,
                               'equity_available': equity, 'close_price': np.nan})
                liq_price = 0
                pos_size = 0
                bid = [calc_initial_long_entry_qty_(equity, ob[0]), min(ob[0], row.bid_thr)]
                ask = [calc_initial_shrt_entry_qty_(equity, ob[1]), max(ob[1], row.ask_thr)]
            if row.price > ask[1]:
                if pos_size <= 0:
                    # adding more to or creating shrt pos
                    while row.price > ask[1]:
                        if -ask[0] < min_qty:
                            ask = [0.0, 9.9e-9]
                            break
                        new_pos_size = pos_size + ask[0]
                        cost = -ask[0] / ask[1]
                        margin_cost = cost / leverage
                        pnl = -cost * maker_fee
                        equity += pnl
                        pos_price = pos_price * (pos_size / new_pos_size) + \
                            ask[1] * (ask[0] / new_pos_size)
                        pos_size = new_pos_size
                        liq_price = calc_shrt_liq_price(pos_price, leverage)
                        close_price = round_dn(pos_price * (1 - markup), price_step)
                        trades.append({'trade_id': row.Index, 'side': 'shrt', 'type': 'entry',
                                       'qty': ask[0], 'price': ask[1],
                                       'pos_size': pos_size, 'pos_price': pos_price,
                                       'liq_price': liq_price, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity + pos_size / row.price / leverage,
                                       'close_price': close_price})
                        ask = calc_shrt_entry_(equity, pos_size, ask[1])
                    bid = calc_shrt_close_(pos_size, pos_price)
                else:
                    if row.price > ask[1]:
                        # close long pos
                        cost = pos_size / ask[1]
                        margin_cost = cost / leverage
                        pnl = cost * (ask[1] / pos_price - 1) - cost * maker_fee
                        equity += pnl
                        trades.append({'trade_id': row.Index, 'side': 'long', 'type': 'close',
                                       'qty': -pos_size, 'price': ask[1],
                                       'pos_size': 0, 'pos_price': np.nan,
                                       'liq_price': np.nan, 'pnl': pnl,
                                       'margin_cost': margin_cost, 'equity': equity,
                                       'equity_available': equity, 'close_price': np.nan})
                        pos_size = 0
                        liq_price = 0
                        bid = [calc_initial_long_entry_qty_(equity, ob[0]), min(ob[0], row.bid_thr)]
                        ask = [calc_initial_shrt_entry_qty_(equity, ob[1]), max(ob[1], row.ask_thr)]
        if row.Index % 1000 == 0:
            line = f'\r{row.Index / len(df):.2f} '
            line += f"start_equity: {settings['initial_equity']} equity {equity:.6f} "
            print(line, end=' ')
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

