import sys
import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings
from binance import fetch_trades as binance_fetch_trades
from bybit import fetch_trades as bybit_fetch_trades
from bybit import round_up, round_dn
from typing import Iterator


def backtest(adf: pd.DataFrame,
             settings: dict,
             margin_cost_limit: float = 0.0,
             maker_fee: float = 0.00018,
             taker_fee: float = 0.00036,
             price_step: float = 0.01,
             inverse: bool = False) -> ([dict], [dict], pd.DataFrame):
    ema_spans = sorted(settings['ema_spans'])
    markup = settings['markup']
    leverage = settings['leverage']
    entry_amount = settings['entry_amount']
    enter_long = settings['enter_long']
    enter_shrt = settings['enter_shrt']
    spread_minus = 1 - settings['spread']
    spread_plus = 1 + settings['spread']

    rdn = lambda n: round_dn(n, price_step)
    rup = lambda n: round_up(n, price_step)

    if inverse:
        calc_cost = lambda amount, price: amount / price
    else:
        calc_cost = lambda amount, price: amount * price

    assert enter_long or enter_shrt

    liq_multiplier = (1 / leverage) / 2

    margin_cost_max = 0
    realized_pnl_sum = 0.0
    n_double_downs = 0

    pos_amount = 0.0
    entry_price = 0.0
    exit_price = 0.0
    liq_price = 0.0

    trades = []

    for span in ema_spans:
        ema_name = f"ema_{int(span)}"
        if ema_name not in adf.columns:
            print(f'calculating {ema_name}...')
            ema = adf.price.ewm(span=span, adjust=False).mean()
            ema.name = ema_name
            adf = adf.join(ema)
    min_name = f"min_" + '_'.join([f'{int(span)}' for span in ema_spans])
    max_name = f"max_" + '_'.join([f'{int(span)}' for span in ema_spans])
    if min_name not in adf.columns:
        ema_min = adf[[c for c in adf.columns if 'ema' in c]].min(axis=1)
        ema_min.name = min_name
        ema_max = adf[[c for c in adf.columns if 'ema' in c]].max(axis=1)
        ema_max.name = max_name
        adf = adf.join(ema_min).join(ema_max)

    idxrange = adf.index[-1] - adf.index[0]
    do_print = False

    for row in adf.drop([c for c in adf.columns if 'ema' in c], axis=1).itertuples():
        do_print = False
        if pos_amount == 0.0:
            if row.price < rdn(getattr(row, min_name) * spread_minus):
                pos_amount = entry_amount
                entry_price = row.price
                liq_price = rdn(entry_price * (1 - liq_multiplier))
                exit_price = rup(entry_price * (1 + markup))
                cost = calc_cost(entry_amount, entry_price)
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'entry',
                               'trade_id': row.Index, 'price': entry_price,
                               'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': 0.0,
                               'fee': cost * taker_fee,
                               'n_double_downs': -1})
                do_print = True
            elif row.price > rup(getattr(row, max_name) * spread_plus):
                pos_amount = -entry_amount
                entry_price = row.price
                liq_price = rup(entry_price * (1 + liq_multiplier))
                exit_price = rdn(entry_price * (1 - markup))
                cost = calc_cost(entry_amount, entry_price)
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'entry',
                               'trade_id': row.Index, 'price': entry_price,
                               'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': 0.0,
                               'fee': cost * taker_fee,
                               'n_double_downs': -1})
                do_print = True
        elif pos_amount > 0.0:
            # long position
            if row.price > exit_price:
                cost = calc_cost(pos_amount, exit_price)
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'exit',
                               'trade_id': row.Index, 'price': exit_price, 'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': cost * markup,
                               'fee': cost * maker_fee,
                               'n_double_downs': -1})
                do_print = True
                pos_amount = 0.0
                n_double_downs = 0
            elif row.price < liq_price:
                cost = calc_cost(pos_amount, liq_price)
                margin_cost = cost / leverage
                if margin_cost >= margin_cost_limit / 2:
                    # liquidation
                    trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'liquidation',
                                   'trade_id': row.Index, 'price': liq_price,
                                   'amount': pos_amount,
                                   'margin_cost': margin_cost, 'realized_pnl': -margin_cost,
                                   'fee': 0.0,
                                   'n_double_downs': -1})
                    do_print = True
                    pos_amount = 0.0
                    n_double_downs = 0
                else:
                    # double down
                    n_double_downs += 1
                    trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'ddown',
                                   'trade_id': row.Index, 'price': liq_price,
                                   'amount': pos_amount, 'margin_cost': margin_cost,
                                   'realized_pnl': 0.0, 'fee': cost * maker_fee,
                                   'n_double_downs': n_double_downs})
                    entry_price = (entry_price + liq_price) / 2
                    pos_amount *= 2
                    liq_price = rdn(entry_price * (1 - liq_multiplier))
                    exit_price = rup(entry_price * (1 + markup))
                    do_print = True
        else:
            # shrt position
            if row.price < exit_price:
                cost = calc_cost(-pos_amount, exit_price)
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'exit',
                               'trade_id': row.Index, 'price': exit_price, 'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': cost * markup,
                               'fee': cost * maker_fee,
                               'n_double_downs': -1})
                do_print = True
                pos_amount = 0.0
                n_double_downs = 0
            elif row.price > liq_price:
                cost = calc_cost(-pos_amount, liq_price)
                margin_cost = cost / leverage
                if margin_cost >= margin_cost_limit / 2:
                    # liquidation
                    trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'liquidation',
                                   'trade_id': row.Index, 'price': exit_price, 'amount': pos_amount,
                                   'margin_cost': margin_cost, 'realized_pnl': -margin_cost,
                                   'fee': 0.0,
                                   'n_double_downs': -1})
                    do_print = True
                    pos_amount = 0.0
                    n_double_downs = 0
                else:
                    n_double_downs += 1
                    trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'ddown',
                                   'trade_id': row.Index, 'price': liq_price,
                                   'amount': pos_amount, 'margin_cost': cost / leverage,
                                   'realized_pnl': 0.0, 'fee': cost * maker_fee,
                                   'n_double_downs': n_double_downs})
                    entry_price = (entry_price + liq_price) / 2
                    pos_amount *= 2
                    liq_price = rup(entry_price * (1 + liq_multiplier))
                    exit_price = rdn(entry_price * (1 - markup))
                    do_print = True
        if do_print:
            realized_pnl_sum += trades[-1]['realized_pnl'] - trades[-1]['fee']
            margin_cost_max = max(margin_cost_max, trades[-1]['margin_cost'])
            rounding = 8 if inverse else 2
            line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.{rounding}f} '
            line += f'{margin_cost_max:.{rounding}f} '
            print(line, end=' ')

    return trades, adf


def get_new_candidate(ranges: dict, best: dict, m=0.2):

    def calc_new_val(v, r):
        new_val = v + (np.random.random() - 0.5) * (r[1] - r[0]) * max(0.0001, m)
        return round(max(min(new_val, r[1]), r[0]), r[2])

    new_candidate = {}
    for key in best:
        if key not in ranges:
            continue
        if type(best[key]) == tuple:
            new_candidate[key] = tuple(sorted([calc_new_val(e, ranges[key]) for e in best[key]]))
        else:
            new_candidate[key] = calc_new_val(best[key], ranges[key])
    return {k_: new_candidate[k_] for k_ in sorted(new_candidate)}


def jackrabbit(exchange: str, agg_trades: pd.DataFrame):


    settings = {'ema_spans': [10000, 38036],
                'enter_long': True,
                'enter_shrt': True,
                'entry_amount': 0.001,
                'leverage': 108,
                'markup': 0.00143,
                'spread': 0.00001,
                'symbol': 'BTCUSDT'}



    '''
    best = {'ema_spans': (39256.0, 90333.0),
            'leverage': 79.0,
            'markup': 0.002025,
            'spread': 0.000239}

    best = {"ema_spans": (45876, 67689),
            "spread": -0.000149,
            "leverage": 91,
            "markup": 0.00093}
    '''


    if exchange == 'bybit':
        best = {'ema_spans': (20174, 61286),
                'leverage': 86.7,
                'markup': 0.0009,
                'spread': 0.0}
        ranges = {'ema_spans': (2000, 250000, 0),
                  'leverage':  (40, 100, 1),
                  'markup': (0.0003, 0.006, 6),
                  'spread': (-0.002, 0.002, 6)}
        price_step = 0.5
        inverse = True
        margin_cost_limit = 0.004
        maker_fee = -0.00025
        taker_fee = 0.00075
        settings['entry_amount'] = 1
    elif exchange == 'binance':
        best = {'ema_spans': (50000, 50000),
                'leverage': 90,
                'markup': 0.0015,
                'spread': 0.0}
        ranges = {'ema_spans': (2000, 250000, 0),
                  'leverage':  (40, 125, 0),
                  'markup': (0.0006, 0.006, 6),
                  'spread': (-0.002, 0.002, 6)}
        price_step = 0.01
        inverse = False
        margin_cost_limit = 160
        maker_fee = 0.00018
        taker_fee = 0.00036
        settings['entry_amount'] = 0.001
    else:
        raise Exception(f'exchage {exchange} not found')

    best = {k_: best[k_] for k_ in sorted(best)}

    results = {}
    best_gain = -99999
    candidate = best

    ks = 200
    k = 0
    ms = np.array([1 / (i * 0.1 + 1) for i in range(ks)])
    ms = (ms - ms.min()) / (ms.max() - ms.min())

    results_filename = make_get_filepath(
        f'jackrabbit_results/{exchange}/{ts_to_date(time())[:19]}'
    )

    min_n_trades = len(agg_trades) / 5000
    print('min_n_trades', min_n_trades)
    conditions = [
        lambda r: r['n_trades'] > min_n_trades
    ]

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
            trades, adf = backtest(adf,
                                   {k_: candidate[k_] if k_ in candidate else settings[k_]
                                    for k_ in settings},
                                   margin_cost_limit=margin_cost_limit,
                                   maker_fee=maker_fee,
                                   taker_fee=taker_fee,
                                   inverse=inverse,
                                   price_step=price_step)
            if not trades:
                print('\nno trades')
                candidate = get_new_candidate(best)
                continue
            tdf = pd.DataFrame(trades).set_index('trade_id')
            result = {'net_pnl': tdf.realized_pnl.sum() - tdf.fee.sum()}
            result['amount_max'] = tdf.amount.max()
            result['amount_min'] = tdf.amount.min()
            result['amount_abs_max'] = tdf.amount.abs().max()
            result['amount_abs_sum'] = tdf.amount.abs().sum()
            result['n_trades'] = len(trades)
            result['max_n_ddown'] = tdf.n_double_downs.max()
            result['mean_n_ddown'] = tdf[tdf.n_double_downs >= 0].n_double_downs.mean()
            result['margin_cost_max'] = tdf.margin_cost.max()
            result['n_liquidations'] = len(tdf[tdf.type == 'liquidation'])
            result['gain'] = (result['net_pnl'] + result['margin_cost_max']) / \
                result['margin_cost_max']
            results[key] = result
            print(f'\n{result}')
            with open(results_filename + '.txt', 'a') as f:
                f.write(str(key) + ' ' + str(results[key]) + '\n')
            if result['gain'] > best_gain:
                if all([condition(result) for condition in conditions]):
                    best = candidate
                    best_gain = result['gain']
                    print('\n\nnew best', best, '\n', result, '\n\n')
            candidate = get_new_candidate(ranges, best, m=ms[k])
            pd.DataFrame(results).to_csv(results_filename + '.csv')
        except KeyboardInterrupt:
            return results
    return results


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
        agg_trades = pd.read_csv(filename).set_index('trade_id')
    else:
        print('fetching trades')
        agg_trades = await load_trades(exchange, user, symbol, n_days)
        agg_trades.to_csv(filename)
    jackrabbit(exchange, agg_trades)


if __name__ == '__main__':
    asyncio.run(main())

