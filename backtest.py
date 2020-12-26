import json
import numpy as np
import pandas as pd
import asyncio
import os
from time import time
from passivbot import init_ccxt, make_get_filepath, ts_to_date, print_, load_settings
from binance import load_trades
from typing import Iterator


def backtest(adf: pd.DataFrame,
             settings: dict,
             margin_cost_limit: float = 0.0,
             maker_fee: float = 0.00018,
             taker_fee: float = 0.00036) -> ([dict], [dict], pd.DataFrame):
    ema_spans = sorted(settings['ema_spans'])
    markup = settings['markup']
    leverage = settings['leverage']
    entry_amount = settings['entry_amount']
    enter_long = settings['enter_long']
    enter_shrt = settings['enter_shrt']
    spread_minus = 1 - settings['spread']
    spread_plus = 1 + settings['spread']

    assert enter_long or enter_shrt

    liq_multiplier = (1 / leverage) / 2

    margin_cost_max = 0
    margin_cost = 0.0
    realized_pnl_sum = 0.0
    n_double_downs = 0

    pos_amount = 0.0
    entry_price = 0.0
    exit_price = 0.0
    liq_price = 0.0
    double_down_price = 0.0

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
            if row.price < getattr(row, min_name) * spread_minus:
                pos_amount = entry_amount
                entry_price = row.price
                liq_price = entry_price * (1 - liq_multiplier)
                exit_price = entry_price * (1 + markup)
                double_down_price = liq_price
                cost = entry_amount * entry_price
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'entry',
                               'trade_id': row.Index, 'price': entry_price,
                               'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': 0.0,
                               'fee': cost * taker_fee,
                               'n_double_downs': -1})
                realized_pnl_sum -= trades[-1]['fee']
            elif row.price > getattr(row, max_name) * spread_plus:
                pos_amount = -entry_amount
                entry_price = row.price
                liq_price = entry_price * (1 + liq_multiplier)
                exit_price = entry_price * (1 - markup)
                double_down_price = liq_price
                cost = entry_amount * entry_price
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'entry',
                               'trade_id': row.Index, 'price': entry_price,
                               'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': 0.0,
                               'fee': cost * taker_fee,
                               'n_double_downs': -1})
                realized_pnl_sum -= trades[-1]['fee']
        elif pos_amount > 0.0:
            # long position
            if row.price >= exit_price:
                cost = pos_amount * exit_price
                realized_pnl = cost * markup
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'exit',
                               'trade_id': row.Index, 'price': exit_price, 'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': realized_pnl,
                               'fee': cost * maker_fee,
                               'n_double_downs': -1})
                realized_pnl_sum += trades[-1]['realized_pnl'] - trades[-1]['fee']
                margin_cost_max = max(margin_cost_max, trades[-1]['margin_cost'])
                do_print = True
                pos_amount = 0.0
                n_double_downs = 0
            elif row.price <= double_down_price:
                n_double_downs += 1
                cost = pos_amount * double_down_price
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'entry',
                               'trade_id': row.Index, 'price': double_down_price,
                               'amount': pos_amount, 'margin_cost': cost / leverage,
                               'realized_pnl': 0.0, 'fee': cost * maker_fee,
                               'n_double_downs': n_double_downs})
                realized_pnl_sum += trades[-1]['realized_pnl'] - trades[-1]['fee']
                margin_cost_max = max(margin_cost_max, trades[-1]['margin_cost'])
                entry_price = (entry_price + double_down_price) / 2
                pos_amount *= 2
                liq_price = entry_price * (1 - liq_multiplier)
                exit_price = entry_price * (1 + markup)
                double_down_price = liq_price
                do_print = True
        else:
            # shrt position
            if row.price <= exit_price:
                cost = -pos_amount * exit_price
                realized_pnl = cost * markup
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'exit',
                               'trade_id': row.Index, 'price': exit_price, 'amount': pos_amount,
                               'margin_cost': cost / leverage, 'realized_pnl': realized_pnl,
                               'fee': cost * maker_fee,
                               'n_double_downs': -1})
                realized_pnl_sum += trades[-1]['realized_pnl'] - trades[-1]['fee']
                margin_cost_max = max(margin_cost_max, trades[-1]['margin_cost'])
                do_print = True
                pos_amount = 0.0
                n_double_downs = 0
            elif row.price >= double_down_price:
                n_double_downs += 1
                cost = -pos_amount * double_down_price
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'entry',
                               'trade_id': row.Index, 'price': double_down_price,
                               'amount': pos_amount, 'margin_cost': cost / leverage,
                               'realized_pnl': 0.0, 'fee': cost * maker_fee,
                               'n_double_downs': n_double_downs})
                realized_pnl_sum += trades[-1]['realized_pnl'] - trades[-1]['fee']
                margin_cost_max = max(margin_cost_max, trades[-1]['margin_cost'])
                entry_price = (entry_price + double_down_price) / 2
                pos_amount *= 2
                liq_price = entry_price * (1 + liq_multiplier)
                exit_price = entry_price * (1 - markup)
                double_down_price = liq_price
                do_print = True
        if do_print:
            line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
            line += f'{margin_cost_max:.2f} '
            print(line, end=' ')
            if margin_cost_limit and margin_cost_max >= margin_cost_limit:
                print('margin_cost_limit exceeded')
                break


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
    return new_candidate


def jackrabbit(agg_trades: pd.DataFrame):


    settings = {'ema_spans': [10000, 38036],
                'enter_long': True,
                'enter_shrt': True,
                'entry_amount': 0.001,
                'leverage': 108,
                'markup': 0.00143,
                'spread': 0.00001,
                'symbol': 'BTCUSDT'}



    #((57096.0, 84029.0), 85.0, 0.00295)
    best = {'ema_spans': (57096, 84029),
            'leverage': 85,
            'markup': 0.00295,
            'spread': 0.0}


    ranges = {'ema_spans': (1000, 300000, 0),
              'leverage':  (50, 125, 0),
              'markup': (0.0002, 0.006, 5),
              'spread': (-0.002, 0.002, 5)}

    margin_cost_limit = 500

    results = {}
    best_gain = 0
    candidate = best

    ks = 200
    k = 0
    ms = np.array([1 / (i * 0.1 + 1) for i in range(ks)])
    ms = (ms - ms.min()) / (ms.max() - ms.min())

    #maker_fee = -0.00025
    #taker_fee = 0.00075
    maker_fee = 0.00018
    taker_fee = 0.00036

    results_filename = make_get_filepath(f'jackrabbit_results/{ts_to_date(time())[:19]}.txt')
    results_filename_csv = make_get_filepath(f'jackrabbit_results/{ts_to_date(time())[:19]}.csv')

    symbol = 'BTCUSDT'
    n_days = 10

    conditions = [
        lambda r: r['n_trades'] > 100
    ]

    while k < ks:
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
                                   taker_fee=taker_fee)
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
            result['gain'] = (result['net_pnl'] + result['margin_cost_max']) / \
                result['margin_cost_max']
            results[key] = result
            print(f'\n{result}')
            with open(results_filename, 'a') as f:
                f.write(str(key) + ' ' + str(results[key]) + '\n')
            if result['gain'] > best_gain:
                if all([condition(result) for condition in conditions]):
                    best = candidate
                    best_gain = result['gain']
                    print('\n\nnew best', best, '\n', result, '\n\n')
            candidate = get_new_candidate(ranges, best, m=ms[k])
            pd.DataFrame(results).to_csv(results_filename_csv)
        except KeyboardInterrupt:
            return results
    return results


async def main():
    n_days = 40
    symbol = 'BTCUSDT'
    agg_trades = await load_trades(symbol, n_days)
    jackrabbit(agg_trades)


if __name__ == '__main__':
    asyncio.run(main())

