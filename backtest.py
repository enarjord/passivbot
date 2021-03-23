import sys
import json
import hjson
import numpy as np
import pandas as pd
import asyncio
import os
import pprint
import gc
import matplotlib.pyplot as plt
import aiomultiprocess
from hashlib import sha256
from multiprocessing import cpu_count, Lock, Value, Array
from time import time
from passivbot import *
from bybit_inverse_futures import create_bot as create_bot_bybit
from bybit_inverse_futures import fetch_trades as bybit_fetch_trades
from binance import create_bot as create_bot_binance
from binance import fetch_trades as binance_fetch_trades

from typing import Iterator, Callable

try:
    aiomultiprocess.set_start_method("fork")
except Exception as e:
    aiomultiprocess.set_start_method("spawn")
    print('failed to set fork method for aiomultiprocess', e)
    print('using spawn method instead')


def score_func_avg_adg(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['avg']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg']
    except:
        return True
    return candidate_score > best_score

def score_func_min_adg(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['min']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['min']
    except:
        return True
    return candidate_score > best_score

def score_func_avg_adg_min_adg(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['avg'] * \
            candidate['average_daily_gain']['min']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg'] * \
            best['average_daily_gain']['min']
    except:
        return True
    return candidate_score > best_score

def score_func_avg_adg_min_liq(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['avg'] * candidate['closest_liq']['min']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg'] * best['closest_liq']['min']
    except:
        return True
    return candidate_score > best_score

def score_func_avg_adg_min_adg_min_liq(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['avg'] * \
            candidate['average_daily_gain']['min'] * candidate['closest_liq']['min']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg'] * best['average_daily_gain']['min'] * \
            best['closest_liq']['min']
    except:
        return True
    return candidate_score > best_score

def score_func_avg_adg_min_adg_min_liq_std_adg(best: dict, candidate: dict) -> bool:
    try:
        candidate_score = candidate['average_daily_gain']['avg'] * \
            candidate['average_daily_gain']['min'] * candidate['closest_liq']['min'] / \
            candidate['average_daily_gain']['std']
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg'] * best['average_daily_gain']['min'] * \
            best['closest_liq']['min'] / best['average_daily_gain']['std']
    except:
        return True
    return candidate_score > best_score

def score_func_avg_adg_min_adg_min_liq_capped(best: dict, candidate: dict) -> bool:
    cap = 0.2
    try:
        candidate_score = candidate['average_daily_gain']['avg'] * \
            candidate['average_daily_gain']['min'] * min(candidate['closest_liq']['min'], cap)
    except:
        return False
    try:
        best_score = best['average_daily_gain']['avg'] * \
            best['average_daily_gain']['min'] * min(best['closest_liq']['min'], cap)
    except:
        return True
    return candidate_score > best_score

def score_func_gain_liq_stuck(best: dict, candidate: dict) -> bool:
    liq_cap = 0.16
    hours_stuck_cap = 72
    try:
        candidate_score = (candidate['average_daily_gain']['avg'] *
                           candidate['average_daily_gain']['min'] *
                           min(1.0, candidate['closest_liq']['min'] / liq_cap) /
                           max(1.0, candidate['max_n_hours_stuck']['max'] / hours_stuck_cap))
    except:
        return False
    try:
        best_score = (best['average_daily_gain']['avg'] *
                      best['average_daily_gain']['min'] *
                      min(1.0, best['closest_liq']['min'] / liq_cap) /
                      max(1.0, best['max_n_hours_stuck']['max'] / hours_stuck_cap))
    except:
        return True
    return candidate_score > best_score


def get_score_func(key: str) -> Callable:

    if key == 'avg adg':
        return score_func_avg_adg
    elif key == 'min adg':
        return score_func_min_adg
    elif key == 'avg adg * min adg':
        return score_func_avg_adg_min_adg
    elif key == 'avg adg * min liq':
        return score_func_avg_adg_min_liq
    elif key == 'avg adg * min adg * min liq':
        return score_func_avg_adg_min_adg_min_liq
    elif key == 'avg adg * min adg * min liq capped':
        return score_func_avg_adg_min_adg_min_liq_capped
    elif key == 'avg adg * min adg * min liq / std adg':
        return score_func_avg_adg_min_adg_min_liq_std_adg
    elif key == 'gain liq stuck':
        return score_func_gain_liq_stuck
    raise Exception('unknown score metric', key)


def dump_plots(result: dict, fdf: pd.DataFrame, df: pd.DataFrame):
    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)

    def gain_conv(x):
        return x * 100 - 100
    
    lines = []
    lines.append(f"net pnl plus fees {result['net_pnl_plus_fees']:.6f}")
    lines.append(f"profit sum {result['profit_sum']:.6f}")
    lines.append(f"loss sum {result['loss_sum']:.6f}")
    lines.append(f"fee sum {result['fee_sum']:.6f}")
    lines.append(f"gain percentage {gain_conv(result['gain']):.2f}%")
    lines.append(f"n_days {result['n_days']}")
    lines.append(f"average_daily_gain percentage {(result['average_daily_gain'] - 1) * 100:.2f}%")
    lines.append(f"n fills {result['n_fills']}")
    lines.append(f"n closes {result['n_closes']}")
    lines.append(f"n reentries {result['n_reentries']}")
    lines.append(f"n stop loss closes {result['n_stop_losses']}")
    lines.append(f"biggest_pos_size {round(result['biggest_pos_size'], 10)}")
    lines.append(f"closest liq percentage {result['closest_liq'] * 100:.4f}%")
    lines.append(f"max n hours stuck {result['max_n_hours_stuck']:.2f}")
    lines.append(f"starting balance {result['starting_balance']}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    print('plotting price with bid ask entry thresholds')
    ema = df.price.ewm(span=result['ema_span'], adjust=False).mean()
    bids_ = ema * (1 - result['ema_spread'])
    asks_ = ema * (1 + result['ema_spread'])
    
    plt.clf()
    df.price.iloc[::100].plot()
    bids_.iloc[::100].plot()
    asks_.iloc[::100].plot()
    plt.savefig(f"{result['session_dirpath']}ema_spread_plot.png")

    print('writing backtest_result.txt...')
    with open(f"{result['session_dirpath']}backtest_result.txt", 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')

    print('plotting balance and equity...')
    plt.clf()
    fdf.actual_balance.plot()
    fdf.equity.plot()
    plt.savefig(f"{result['session_dirpath']}balance_and_equity.png")

    print('plotting backtest whole and in chunks...')
    n_parts = 7
    for z in range(n_parts):
        start_ = z / n_parts
        end_ = (z + 1) / n_parts
        print(start_, end_)
        fig = plot_fills(df, fdf.iloc[int(len(fdf) * start_):int(len(fdf) * end_)], liq_thr=0.1)
        fig.savefig(f"{result['session_dirpath']}backtest_{z + 1}of{n_parts}.png")
    fig = plot_fills(df, fdf, liq_thr=0.1)
    fig.savefig(f"{result['session_dirpath']}whole_backtest.png")

    print('plotting pos sizes...')
    plt.clf()
    fdf.long_pos_size.plot()
    fdf.shrt_pos_size.plot()
    plt.savefig(f"{result['session_dirpath']}pos_sizes_plot.png")

    print('plotting average daily gain...')
    adg_ = fdf.average_daily_gain
    adg_.index = np.linspace(0.0, 1.0, len(fdf))
    plt.clf()
    adg_c = adg_.iloc[int(len(fdf) * 0.1):] # skipping first 10%
    print('min max', adg_c.min(), adg_c.max())
    adg_c.plot()
    plt.savefig(f"{result['session_dirpath']}average_daily_gain_plot.png")
    


def plot_fills(df, fdf, side_: int = 0, liq_thr=0.1):
    plt.clf()

    df.loc[fdf.index[0]:fdf.index[-1]].price.plot(style='y-')

    if side_ >= 0:
        longs = fdf[fdf.pos_side == 'long']
        le = longs[longs.type.str.endswith('entry')]
        lc = longs[longs.type == 'close']
        le.price.plot(style='b.')
        lc.price.plot(style='r.')
        longs.long_pos_price.fillna(method='ffill').plot(style='b--')
    if side_ <= 0:
        shrts = fdf[fdf.pos_side == 'shrt']
        se = shrts[shrts.type.str.endswith('entry')]
        sc = shrts[shrts.type == 'close']
        se.price.plot(style='r.')
        sc.price.plot(style='b.')
        shrts.shrt_pos_price.fillna(method='ffill').plot(style='r--')


    stops = fdf[fdf.type.str.startswith('stop_loss')]
    stops.price.plot(style='gx')
    if 'liq_price' in fdf.columns:
        liq_diff = ((fdf.liq_price - fdf.price).abs() / fdf.price)
        fdf.liq_price.where(liq_diff < liq_thr, np.nan).plot(style='k--')
    return plt


def prep_ticks(df: pd.DataFrame) -> np.ndarray:
    dfc = df[df.price != df.price.shift(1)] # drop consecutive same price trades
    dfc.index = np.arange(len(dfc))
    buyer_maker = dfc.is_buyer_maker
    buyer_maker.name = 'buyer_maker'
    dfcc = pd.concat([dfc.price, buyer_maker, dfc.timestamp], axis=1)
    return dfcc.values


def backtest(ticks: np.ndarray, settings: dict) -> [dict]:
    ss = settings

    long_pos_size, long_pos_price, long_reentry_price = 0.0, np.nan, np.nan
    shrt_pos_size, shrt_pos_price, shrt_reentry_price = 0.0, np.nan, np.nan
    liq_price = 0.0
    actual_balance = ss['starting_balance']
    apparent_balance = actual_balance * ss['balance_pct']

    pnl_plus_fees_cumsum, loss_cumsum, profit_cumsum, fee_paid_cumsum = 0.0, 0.0, 0.0, 0.0

    if ss['inverse']:
        long_pnl_f = calc_long_pnl_inverse
        shrt_pnl_f = calc_shrt_pnl_inverse
        cost_f = calc_cost_inverse
    else:
        iter_long_entries = lambda balance, long_psize, long_pprice, shrt_psize, highest_bid: \
            iter_long_entries_linear(
                ss['price_step'], ss['qty_step'], ss['min_qty'], ss['min_cost'], ss['ddown_factor'],
                ss['entry_qty_pct'], ss['leverage'], ss['grid_spacing'], ss['grid_coefficient'],
                balance, long_psize, long_pprice, shrt_psize, highest_bid
            )
        iter_shrt_entries = lambda balance, long_psize, shrt_psize, shrt_pprice, lowest_ask: \
            iter_shrt_entries_linear(
                ss['price_step'], ss['qty_step'], ss['min_qty'], ss['min_cost'], ss['ddown_factor'],
                ss['entry_qty_pct'], ss['leverage'], ss['grid_spacing'], ss['grid_coefficient'],
                balance, long_psize, shrt_psize, shrt_pprice, lowest_ask
            )
        iter_long_closes = lambda balance, pos_size, pos_price, lowest_ask: \
            iter_long_closes_linear(ss['price_step'], ss['qty_step'], ss['min_qty'],
                                    ss['close_qty_pct'], ss['leverage'], ss['min_markup'],
                                    ss['max_markup'], ss['n_close_orders'],
                                    balance, pos_size, pos_price, lowest_ask)
        iter_shrt_closes = lambda balance, pos_size, pos_price, highest_bid: \
            iter_shrt_closes_linear(ss['price_step'], ss['qty_step'], ss['min_qty'],
                                    ss['close_qty_pct'], ss['leverage'], ss['min_markup'],
                                    ss['max_markup'], ss['n_close_orders'],
                                    balance, pos_size, pos_price, highest_bid)
        long_pnl_f = calc_long_pnl_linear
        shrt_pnl_f = calc_shrt_pnl_linear
        cost_f = calc_cost_linear

    liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
        calc_cross_hedge_lig_price(balance, l_psize, l_pprice, s_psize, s_pprice, ss['leverage'])
    
    break_on = {e[0]: eval(e[1]) for e in settings['break_on'] if e[0].startswith('ON:')}

    prev_trade_ts = 0
    prev_long_close_ts, prev_long_entry_ts, prev_long_close_price = 0, 0, 0.0
    prev_shrt_close_ts, prev_shrt_entry_ts, prev_shrt_close_price = 0, 0, 0.0
    min_trade_delay_millis = ss['latency_simulation_ms'] if 'latency_simulation_ms' in ss else 1000

    all_fills = []
    ob = [min(ticks[0][0], ticks[1][0]),
          max(ticks[0][0], ticks[1][0])]
    ema = ticks[0][0]
    ema_alpha = 2 / (ss['ema_span'] + 1)
    ema_alpha_ = 1 - ema_alpha
    # tick tuple: (price, buyer_maker, timestamp)
    for k, tick in enumerate(ticks):
        fills = []
        if tick[1]:
            # maker buy, taker sel
            if ss['do_long']:
                if tick[0] <= liq_price and long_pos_size > -shrt_pos_size:
                    if (liq_diff := calc_diff(liq_price, tick[0])) < 0.1:
                        print('long liq')
                        fills.append({
                            'price': tick[0], 'side': 'sel', 'pos_side': 'long',
                            'type': 'liquidation', 'qty': -long_pos_size,
                            'pnl': long_pnl_f(long_pos_price, tick[0], -long_pos_size),
                            'fee_paid': -cost_f(long_pos_size, tick[0]) * ss['taker_fee'],
                            'long_pos_size': 0.0, 'long_pos_price': np.nan,
                            'shrt_pos_size': 0.0, 'shrt_pos_price': np.nan,
                            'liq_diff': liq_diff
                        })
                if long_pos_size == 0.0:
                    if ss['ema_span'] > 1.0:
                        highest_bid = min(ob[0], round_dn(ema * (1 - ss['ema_spread']), ss['price_step']))
                    else:
                        highest_bid = ob[0]
                elif tick[0] < long_reentry_price:
                    highest_bid = ob[0]
                else:
                    highest_bid = 0.0
                if highest_bid > 0.0 and tick[2] - prev_long_close_ts > min_trade_delay_millis:
                    # create or add to long pos
                    for tpl in iter_long_entries(apparent_balance, long_pos_size, long_pos_price,
                                                 shrt_pos_size, highest_bid):
                        long_reentry_price = tpl[1]
                        if tick[0] < tpl[1]:
                            long_pos_size, long_pos_price, prev_long_entry_ts = tpl[2], tpl[3], tick[2]
                            fills.append({
                                'price': tpl[1], 'side': 'buy', 'pos_side': 'long',
                                'type': 'reentry' if tpl[4] else 'entry', 'qty': tpl[0], 'pnl': 0.0,
                                'fee_paid': -cost_f(tpl[0], tpl[1]) * ss['maker_fee'],
                                'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                                'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                                'liq_diff': calc_diff(liq_price, tick[0])
                            })
                        else:
                            break

            if shrt_pos_size < 0.0 and tick[0] < shrt_pos_price and \
                    (tick[2] - prev_shrt_entry_ts > min_trade_delay_millis):
                # close shrt pos
                for qty, price, new_pos_size in iter_shrt_closes(apparent_balance, shrt_pos_size,
                                                                 shrt_pos_price, ob[0]):
                    if tick[0] < price:
                        if tick[2] - prev_shrt_close_ts < min_trade_delay_millis and \
                                price >= prev_shrt_close_price:
                            break
                        pnl = shrt_pnl_f(shrt_pos_price, price, qty)
                        shrt_pos_size, prev_shrt_close_ts = new_pos_size, tick[2]
                        if shrt_pos_size == 0.0:
                            shrt_pos_price = np.nan
                        fills.append({
                            'price': price, 'side': 'buy', 'pos_side': 'shrt',
                            'type': 'close', 'qty': qty,
                            'pnl': pnl, 'fee_paid': -cost_f(qty, price) * ss['maker_fee'],
                            'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                            'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                            'liq_diff': calc_diff(liq_price, tick[0])
                        })
                        prev_shrt_close_price = price
                        shrt_reentry_price = np.nan
            ob[0] = tick[0]
        else:
            # maker sel, taker buy
            if ss['do_shrt']:
                if tick[0] >= liq_price and -shrt_pos_size > long_pos_size:
                    if (liq_diff := calc_diff(liq_price, tick[0])) < 0.1:
                        print('shrt liq')
                        fills.append({
                            'price': tick[0], 'side': 'buy', 'pos_side': 'shrt',
                            'type': 'liquidation', 'qty': -shrt_pos_size,
                            'pnl': shrt_pnl_f(shrt_pos_price, tick[0], -shrt_pos_size),
                            'fee_paid': -cost_f(shrt_pos_size, tick[0]) * ss['taker_fee']
                        })
                if shrt_pos_size == 0.0:
                    if ss['ema_span'] > 1.0:
                        lowest_ask = max(ob[1], round_up(ema * (1 + ss['ema_spread']), ss['price_step']))
                    else:
                        lowest_ask = ob[1]
                elif tick[0] > shrt_reentry_price:
                    lowest_ask = ob[1]
                else:
                    lowest_ask = 0.0
                if lowest_ask > 0.0 and tick[2] - prev_shrt_close_ts > min_trade_delay_millis:
                    # create or add to shrt pos
                    for tpl in iter_shrt_entries(apparent_balance, long_pos_size, shrt_pos_size,
                                                 shrt_pos_price, lowest_ask):
                        shrt_reentry_price = tpl[1]
                        if tick[0] > tpl[1]:
                            shrt_pos_size, shrt_pos_price = tpl[2], tpl[3]
                            fills.append({
                                'price': tpl[1], 'side': 'sel', 'pos_side': 'shrt',
                                'type': 'reentry' if tpl[4] else 'entry', 'qty': tpl[0], 'pnl': 0.0,
                                'fee_paid': -cost_f(tpl[0], tpl[1]) * ss['maker_fee'],
                                'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                                'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                                'liq_diff': calc_diff(liq_price, tick[0])
                            })
                        else:
                            break
            if long_pos_size > 0.0 and tick[0] > long_pos_price and \
                    (tick[2] - prev_long_entry_ts > min_trade_delay_millis):
                # close long pos
                for qty, price, new_pos_size in iter_long_closes(apparent_balance, long_pos_size,
                                                                 long_pos_price, ob[1]):
                    if tick[0] > price:
                        if tick[2] - prev_long_close_ts < min_trade_delay_millis and \
                                price <= prev_long_close_price:
                            break
                        pnl = long_pnl_f(long_pos_price, price, qty)
                        long_pos_size, prev_long_close_ts = new_pos_size, tick[2]
                        if long_pos_size == 0.0:
                            long_pos_price = np.nan
                        fills.append({
                            'price': price, 'side': 'sel', 'pos_side': 'long',
                            'type': 'close', 'qty': qty,
                            'pnl': pnl, 'fee_paid': -cost_f(qty, price) * ss['maker_fee'],
                            'long_pos_size': long_pos_size, 'long_pos_price': long_pos_price,
                            'shrt_pos_size': shrt_pos_size, 'shrt_pos_price': shrt_pos_price,
                            'liq_diff': calc_diff(liq_price, tick[0])
                        })
                        prev_long_close_price = price
                        long_reentry_price = np.nan
            ob[1] = tick[0]
        ema = calc_ema(ema_alpha, ema_alpha_, ema, tick[0])
        if len(fills) > 0:
            for fill in fills:
                actual_balance += fill['pnl'] + fill['fee_paid']
                apparent_balance = actual_balance * ss['balance_pct']
                liq_price = liq_price_f(actual_balance, long_pos_size, long_pos_price,
                                        shrt_pos_size, shrt_pos_price)
                ms_since_long_pos_change = tick[2] - prev_long_fill_ts \
                    if (prev_long_fill_ts := max(prev_long_close_ts, prev_long_entry_ts)) > 0 else 0
                ms_since_shrt_pos_change = tick[2] - prev_shrt_fill_ts \
                    if (prev_shrt_fill_ts := max(prev_shrt_close_ts, prev_shrt_entry_ts)) > 0 else 0

                if fill['type'].startswith('stop_loss'):
                    loss_cumsum += fill['pnl']
                else:
                    profit_cumsum += fill['pnl']
                fee_paid_cumsum += fill['fee_paid']
                pnl_plus_fees_cumsum += fill['pnl'] + fill['fee_paid']
                upnl_l = x if (x := long_pnl_f(long_pos_price, tick[0], long_pos_size)) == x else 0.0
                upnl_s = y if (y := shrt_pnl_f(shrt_pos_price, tick[0], shrt_pos_size)) == y else 0.0
                fill['equity'] = actual_balance + upnl_l + upnl_s
                fill['pnl_plus_fees_cumsum'] = pnl_plus_fees_cumsum
                fill['loss_cumsum'] = loss_cumsum
                fill['profit_cumsum'] = profit_cumsum
                fill['fee_paid_cumsum'] = fee_paid_cumsum

                fill['actual_balance'] = actual_balance
                fill['apparent_balance'] = apparent_balance
                fill['liq_price'] = liq_price
                fill['timestamp'] = tick[2]
                fill['trade_id'] = k
                fill['progress'] = k / len(ticks)
                fill['drawdown'] = calc_diff(fill['actual_balance'], fill['equity'])
                fill['gain'] = fill['equity'] / settings['starting_balance']
                fill['n_days'] = (tick[2] - ticks[0][2]) / (1000 * 60 * 60 * 24)
                try:
                    fill['average_daily_gain'] = fill['gain'] ** (1 / fill['n_days']) \
                        if (fill['n_days'] > 0.0 and fill['gain'] > 0.0) else 0.0
                except:
                    fill['average_daily_gain'] = 0.0
                fill['hours_since_long_pos_change'] = (lc := ms_since_long_pos_change / (1000 * 60 * 60))
                fill['hours_since_shrt_pos_change'] = (sc := ms_since_shrt_pos_change / (1000 * 60 * 60))
                fill['hours_since_pos_change_max'] = max(lc, sc)
                all_fills.append(fill)
                if actual_balance <= 0.0 or fill['type'] == 'liquidation':
                    return all_fills, False
                for key, condition in break_on.items():
                    if condition(all_fills, ticks, k):
                        print('break on', key)
                        return all_fills, False
            # print(f"\r{k / len(ticks):.2f} ", end=' ')
    return all_fills, True


def calc_new_val(val, range_, m):
    # quick fix of bug where random nums weren't random enuf between processes
    np.random.seed((int(time() * 100000) % (2**31)))
    choice_span = (range_[1] - range_[0]) * m / 2
    biased_mid_point = max(range_[0] + choice_span, min(val, range_[1] - choice_span))
    choice_range = (biased_mid_point - choice_span, biased_mid_point + choice_span)
    new_val = np.random.choice(np.linspace(choice_range[0], choice_range[1], 200))
    #new_val = np.random.random() * (choice_range[1] - choice_range[0])
    return round_(new_val, range_[2])


def get_new_candidate(ranges: dict, best: dict, m=0.2):
    cand = sort_dict_keys({k: calc_new_val(best[k], ranges[k], m) for k in best if k in ranges})
    cand['max_markup'] = max(cand['min_markup'], cand['max_markup'])
    cand['close_qty_pct'] = min(cand['entry_qty_pct'], cand['close_qty_pct'])
    return cand


def get_downloaded_trades(filepath: str, age_limit_millis: float) -> (pd.DataFrame, dict):
    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')],
                           key=lambda x: int(x[:x.find('_')].replace('.cs', '').replace('v', '')))
        chunks = []
        chunk_lengths = {}
        df = pd.DataFrame()
        for f in filenames[::-1]:
            try:
                chunk = pd.read_csv(os.path.join(filepath, f), dtype=np.float64).set_index('trade_id')
            except ValueError as e:
                # old bybit formatting
                chunk = pd.read_csv(os.path.join(filepath, f)).set_index('trade_id')
                chunk = chunk.drop('side', axis=1).join(pd.Series(chunk.side == 'Sell', name='is_buyer_maker', index=chunk.index))
                chunk = chunk.astype(np.float64)
            chunk_lengths[f] = len(chunk)
            chunks.append(chunk)
            if len(chunks) >= 100:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                chunks = []
            print('\rloaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000),
                  end='     ')
            if chunk.timestamp.iloc[0] < age_limit_millis:
                break
        if chunks:
            if df.empty:
                df = pd.concat(chunks, axis=0)
            else:
                chunks.insert(0, df)
                df = pd.concat(chunks, axis=0)
            del chunks
        if not df.empty:
            return df[~df.index.duplicated()], chunk_lengths
        else:
            return None, {}
    else:
        return None, {}


async def load_trades(exchange: str, user: str, symbol: str, n_days: float) -> pd.DataFrame:

    def skip_ids(id_, ids_):
        if id_ in ids_:
            print('skipping from', id_)
            while id_ in ids_:
                id_ -= 1
            print('           to', id_)
        return id_

    def load_cache(index_only=False):
        cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
        if cache_filenames:
            print('loading cached ticks')
            if index_only:
                cache_df = pd.concat(
                    [pd.read_csv(os.path.join(cache_filepath, f), dtype=np.float64, usecols=["trade_id"]) for f in
                     cache_filenames], axis=0)
            else:
                cache_df = pd.concat(
                    [pd.read_csv(os.path.join(cache_filepath, f), dtype=np.float64) for f in cache_filenames], axis=0)
            cache_df = cache_df.set_index('trade_id')
            return cache_df
        return None

    if exchange == 'binance':
        fetch_trades_func = binance_fetch_trades
    elif exchange == 'bybit':
        fetch_trades_func = bybit_fetch_trades
    else:
        print(exchange, 'not found')
        return
    cc = init_ccxt(exchange, user)
    filepath = make_get_filepath(os.path.join('historical_data', exchange, 'agg_trades_futures',
                                              symbol, ''))
    cache_filepath = make_get_filepath(filepath.replace(symbol, symbol + '_cache'))
    age_limit = time() - 60 * 60 * 24 * n_days
    age_limit_millis = age_limit * 1000
    print('age_limit', ts_to_date(age_limit))
    cache_df = load_cache(True)
    trades_df, chunk_lengths = get_downloaded_trades(filepath, age_limit_millis)
    ids = set()
    if trades_df is not None:
        ids.update(trades_df.index)
    if cache_df is not None:
        ids.update(cache_df.index)
        del cache_df
        gc.collect()
    gaps = []
    if trades_df is not None and len(trades_df) > 0:
        # 
        sids = sorted(ids)
        for i in range(1, len(sids)):
            if sids[i-1] + 1 != sids[i]:
                gaps.append((sids[i-1], sids[i]))
        if gaps:
            print('gaps', gaps)
        del sids
        gc.collect()
        # 
    prev_fetch_ts = time()
    new_trades = await fetch_trades_func(cc, symbol)
    k = 0
    while True:
        k += 1
        if (break_ := new_trades[0]['timestamp'] <= age_limit_millis) or k % 20 == 0:
            print('caching trades...')
            new_tdf = pd.DataFrame(new_trades).set_index('trade_id')
            cache_filename = f'{cache_filepath}{new_tdf.index[0]}_{new_tdf.index[-1]}.csv'
            new_tdf.to_csv(cache_filename)
            new_trades = [new_trades[0]]
            if break_:
                break
        from_id = skip_ids(new_trades[0]['trade_id'] - 1, ids) - 999
        # wait at least 0.75 sec between each fetch
        sleep_for = max(0.0, 0.75 - (time() - prev_fetch_ts))
        await asyncio.sleep(sleep_for)
        prev_fetch_ts = time()
        fetched_new_trades = await fetch_trades_func(cc, symbol, from_id=from_id)
        while fetched_new_trades[0]['trade_id'] == new_trades[0]['trade_id']:
            print('gaps in ids', from_id)
            from_id -= 1000
            fetched_new_trades = await fetch_trades_func(cc, symbol, from_id=from_id)
        new_trades = fetched_new_trades + new_trades
        ids.update([e['trade_id'] for e in new_trades])
        if new_trades[0]['trade_id'] <= 1000:
            print('end of the line')
            break
    del ids
    gc.collect()
    tdf = pd.concat([load_cache(), trades_df], axis=0).sort_index()
    tdf = tdf[~tdf.index.duplicated()]
    dump_chunks(filepath, tdf, chunk_lengths)
    cache_filenames = [f for f in os.listdir(cache_filepath) if '.csv' in f]
    print('removing cache...\n')
    for filename in cache_filenames:
        print(f'\rremoving {filename}', end='   ')
        os.remove(cache_filepath + filename)
    await cc.close()
    return tdf[tdf.timestamp >= age_limit_millis]


def dump_chunks(filepath: str, tdf: pd.DataFrame, chunk_lengths: dict, chunk_size=100000):
    chunk_ids = tdf.index // chunk_size * chunk_size
    for g in tdf.groupby(chunk_ids):
        filename = f'{int(g[1].index[0])}_{int(g[1].index[-1])}.csv'
        if filename not in chunk_lengths or chunk_lengths[filename] != chunk_size:
            print('dumping chunk', filename)
            g[1].to_csv(f'{filepath}{filename}')


async def fetch_market_specific_settings(exchange: str, user: str, symbol: str):
    tmp_live_settings = load_live_settings(exchange, do_print=False)
    tmp_live_settings['symbol'] = symbol
    settings_from_exchange = {}
    if exchange == 'binance':
        bot = await create_bot_binance(user, tmp_live_settings)
        settings_from_exchange['inverse'] = False
        settings_from_exchange['maker_fee'] = 0.00018
        settings_from_exchange['taker_fee'] = 0.00036
        settings_from_exchange['exchange'] = 'binance'
    elif exchange == 'bybit':
        bot = await create_bot_bybit(user, tmp_live_settings)
        settings_from_exchange['inverse'] = True
        settings_from_exchange['maker_fee'] = -0.00025
        settings_from_exchange['taker_fee'] = 0.00075
        settings_from_exchange['exchange'] = 'bybit'
    else:
        raise Exception(f'unknown exchange {exchange}')
    settings_from_exchange['max_leverage'] = bot.max_leverage
    settings_from_exchange['min_qty'] = bot.min_qty
    settings_from_exchange['min_cost'] = bot.min_notional
    settings_from_exchange['qty_step'] = bot.qty_step
    settings_from_exchange['price_step'] = bot.price_step
    settings_from_exchange['max_leverage'] = bot.max_leverage
    await bot.cc.close()
    return settings_from_exchange


def live_settings_to_candidate(live_settings: dict, ranges: dict) -> dict:
    return {k: live_settings[k] for k in ranges if k in live_settings}


def candidate_to_live_settings(exchange: str, candidate: dict) -> dict:
    live_settings = load_live_settings(exchange, do_print=False)
    for k in candidate:
        if k in live_settings:
            live_settings[k] = candidate[k]
    live_settings['config_name'] = candidate['session_name']
    live_settings['symbol'] = candidate['symbol']
    live_settings['key'] = candidate['key']
    for k in ['do_long', 'do_shrt']:
        live_settings[k] = bool(candidate[k])
    return live_settings


def calc_candidate_hash_key(candidate: dict, keys: [str]) -> str:
    return sha256(json.dumps({k: candidate[k] for k in sorted(keys)
                              if k in candidate}).encode()).hexdigest()


def load_results(results_filepath: str) -> dict:
    if os.path.exists(results_filepath):
        with open(results_filepath) as f:
            lines = f.readlines()
        results = {(e := json.loads(line))['key']: e for line in lines}
    else:
        results = {}
    return results


async def jackrabbit(ticks: [dict], backtest_config: dict):
    results = load_results(backtest_config['session_dirpath'] + 'results.txt')
    k = backtest_config['starting_k']
    ks = backtest_config['n_jackrabbit_iterations']
    if ks > 1:
        ms = np.array([1 / (i / 2 + 32) for i in range(ks)])
        ms = ((ms - ms.min()) / (ms.max() - ms.min()))
    else:
        ms = np.array([0.0])
    try:
        best_result = json.load(open(backtest_config['session_dirpath'] + 'best_result.json'))
    except Exception as e:
        print('no current best result')
        best_result = {}
    try:
        candidate = live_settings_to_candidate(
            json.load(open(backtest_config['starting_candidate_filepath'])),
            backtest_config['ranges']
        )
        print('using given starting candidate', backtest_config['starting_candidate_filepath'])
    except Exception as e:
        print(e, f"starting candidate {backtest_config['starting_candidate_filepath']} not found.")
        if best_result:
            print('building on current best')
            candidate = get_new_candidate(backtest_config['ranges'], best_result, m=0.0)
            pass
        else:
            print('using random starting candidate')
            candidate = get_new_candidate(
                backtest_config['ranges'],
                {k_: 0.0 for k_ in backtest_config['ranges']},
                m=1.0
            )
    if '--plot' in sys.argv:
        if '--given' in sys.argv:
            best_result = candidate
            print('backtesting and plotting given candidate')

        else:
            _, best_result = load_shared_data(backtest_config['session_dirpath'], Lock())
            if not best_result:
                return
            print('backtesting and plotting best candidate')
        result_, tdf_ = jackrabbit_wrap(ticks, {**backtest_config, **{'break_on': []}, **best_result})
        if tdf_ is None:
            print('no trades')
            return
        tdf_.to_csv(backtest_config['session_dirpath'] + f"backtest_trades_{best_result['key']}.csv")
        print('\nmaking ticks dataframe...')
        df = pd.DataFrame({'price': ticks[:,0], 'buyer_maker': ticks[:,1], 'timestamp': ticks[:,2]})
        dump_plots({**backtest_config, **best_result, **result_}, tdf_, df)
        return
    
    await jackrabbit_multi_core(results,
                                ticks,
                                backtest_config,
                                candidate,
                                k,
                                ks,
                                ms)


async def multiprocess_wrap(func, args=()):
    result = await aiomultiprocess.Worker(target=func, args=args)
    return result


def load_best_result(dirpath: str):
    if os.path.exists((p := dirpath + 'best_result.json')):
        return json.load(open(p))
    return {}


def load_keys(dirpath: str) -> set:
    if os.path.exists((p := dirpath + 'keys.txt')):
        with open(p) as f:
            keys = set([line.strip() for line in f.readlines()])
    else:
        keys = set()
    return keys

def append_key(dirpath: str, key: str):
    with open(dirpath + 'keys.txt', 'a') as f:
        f.write(key + '\n')

def load_shared_data(dirpath: str, lock: Lock) -> (dict, dict):
    lock.acquire()
    try:
        return load_keys(dirpath), load_best_result(dirpath)
    finally:
        lock.release()


def dump_shared_data(dirpath: str, result: dict, best_result: dict, lock: Lock) -> None:
    lock.acquire()
    try:
        with open(dirpath + 'results.txt', 'a') as f:
            f.write(json.dumps(result) + '\n')
        append_key(dirpath, result['key'])
        if load_best_result(dirpath) != best_result:
            json.dump(best_result, open(dirpath + 'best_result.json', 'w'), indent=4)
    finally:
        lock.release()


def get_next_candidate(backtest_config: dict, candidate: dict, ms: [float], k: Value, lock: Lock):
    lock.acquire()
    new_candidate = candidate.copy()
    try:
        dirpath = backtest_config['session_dirpath']
        keys, best_result = load_keys(dirpath), load_best_result(dirpath)
        key = calc_candidate_hash_key(candidate, list(backtest_config['ranges']))
        for _ in range(10):
            if key not in keys:
                break
            new_candidate = get_new_candidate(backtest_config['ranges'],
                                              (best_result if best_result else candidate),
                                              m=ms[k.value])
            key = calc_candidate_hash_key(new_candidate, list(backtest_config['ranges']))
        else:
            return None, key
        append_key(dirpath, key)
        return new_candidate, key
    finally:
        lock.release()


async def jackrabbit_worker(ticks: Array,
                            dim: (),
                            backtest_config: dict,
                            candidate: dict,
                            score_func: Callable,
                            k: Value,
                            ks: int,
                            ms: [float],
                            lock: Lock):
    ticks = np.frombuffer(ticks.get_obj(), dtype='d').reshape(dim)
    start_time = time()
    await asyncio.sleep(0.01)
    while True:

        if k.value >= ks:
            break
        candidate, key = get_next_candidate(backtest_config, candidate, ms, k, lock)
        if candidate is None:
            break
        line = f'running backtest {k.value} of {ks}. m={ms[k.value]:.3f} {key} '
        k.value = k.value + 1
        bpm = k.value / (time() - start_time) * 60
        line += f'backtests per minute: {bpm:.2f}'
        print(line)
        print(candidate, '\n')
        result = await jackrabbit_sliding_window_wrap(ticks, {**backtest_config,
                                                              **candidate,
                                                              **{'key': key}})
        result['key'] = key
        result = {**result, **candidate}
        keys, best_result = load_shared_data(backtest_config['session_dirpath'], lock)
        if score_func(best_result, result):
            print('\n\n### new best ###\n\n')
            best_result = result
            print(json.dumps(best_result, indent=4))
            print('\n\n')
            json.dump(candidate_to_live_settings(backtest_config['exchange'],
                                                 {**backtest_config, **candidate, **best_result}),
                      open(backtest_config['session_dirpath'] + 'live_config.json', 'w'),
                      indent=4)
        dump_shared_data(backtest_config['session_dirpath'], result, best_result, lock)


async def jackrabbit_multi_core(results: dict,
                                ticks: [dict],
                                backtest_config: dict,
                                candidate: dict,
                                k_: int,
                                ks: int,
                                ms: [float]):
    n_cpus = min(cpu_count(), backtest_config['max_n_cpus'])
    print('using', n_cpus, 'cpus')
    score_func = get_score_func(backtest_config['score_metric'])
    lock = Lock()
    k = Value('i', k_)
    workers = []
    ticks_m = Array('d', int(np.prod(ticks.shape)), lock=True)
    ticks_n = np.frombuffer(ticks_m.get_obj(), dtype='d').reshape(ticks.shape)
    ticks_n[:] = ticks
    for _ in range(min(n_cpus, ks)):
        workers.append(asyncio.create_task(multiprocess_wrap(
            jackrabbit_worker, (ticks_m, ticks.shape, backtest_config, candidate, score_func, k, ks,
                                ms, lock)
        )))
        await asyncio.sleep(0.05)
    for w in workers:
        await w


def jackrabbit_wrap(ticks: [dict], backtest_config: dict) -> dict:
    start_ts = time()
    fills, did_finish = backtest(ticks, backtest_config)
    elapsed = time() - start_ts
    if not did_finish or not fills:
        return {}, None
    fdf = pd.DataFrame(fills).set_index('trade_id')
    ms_gap = np.diff([ticks[0][2]] + list(fdf.timestamp) + [ticks[-1][2]]).max()
    hours_since_prev_fill = ticks[-1][2] - fills[-1]['timestamp']
    result = {
        'net_pnl_plus_fees': fills[-1]['pnl_plus_fees_cumsum'],
        'profit_sum': fills[-1]['profit_cumsum'],
        'loss_sum': fills[-1]['loss_cumsum'],
        'fee_sum': fills[-1]['fee_paid_cumsum'],
        'final_equity': fills[-1]['equity'],
        'gain': (gain := fills[-1]['gain']),
        'max_drawdown': fdf.drawdown.max(),
        'n_days': (n_days := (ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': gain ** (1 / n_days) if gain > 0.0 else 0.0,
        'closest_liq': fdf.liq_diff.min(),
        'n_fills': len(fills),
        'n_closes': len(fdf[fdf.type == 'close']),
        'n_reentries': len(fdf[fdf.type == 'reentry']),
        'n_stop_losses': len(fdf[fdf.type.str.startswith('stop_loss')]),
        'biggest_pos_size': fdf[['long_pos_size', 'shrt_pos_size']].abs().max(axis=1).max(),
        'max_n_hours_stuck': max(fdf['hours_since_pos_change_max'].max(),
                                 (ticks[-1][2] - fills[-1]['timestamp']) / (1000 * 60 * 60)),
        'do_long': bool(backtest_config['do_long']),
        'do_shrt': bool(backtest_config['do_shrt']),
        'seconds_elapsed': elapsed
    }
    return result, fdf


def iter_slices(iterable, step: float, size: float):
    i = 0
    n = int(len(iterable) * size)
    s = int(len(iterable) * step)
    while i + n < len(iterable):
        yield iterable[i:i+n]
        i += s
    yield iterable[-n:]
    if size < 1.0:
        # also yield full 
        yield iterable


async def jackrabbit_sliding_window_wrap(ticks: np.ndarray, backtest_config: dict) -> dict:
    sub_runs = []
    start_ts = time()
    for z, slice_ in enumerate(iter_slices(ticks,
                                           backtest_config['sliding_window_step'],
                                           backtest_config['sliding_window_size'])):
        result_, _ = jackrabbit_wrap(slice_, backtest_config)
        if not result_:
            print(f"\n{backtest_config['key']} did not finish backtest")
            return {'key': backtest_config['key']}
        sub_runs.append(result_)
    total_elapsed = time() - start_ts
    result = {}
    skip = ['do_long', 'do_shrt']
    for k in sub_runs[0]:
        if k in skip:
            continue
        try:
            vals = [r[k] for r in sub_runs]
            result[k] = {'avg': np.mean(vals),
                         'std': np.std(vals),
                         'min': min(vals),
                         'max': max(vals),
                         'vals': {**{f'sub_run_{i:03}': v for i, v in enumerate(vals[:-1])},
                                  **{'full_run': vals[-1]}}}
        except:
            continue
    result['n_sub_runs'] = z
    result['total_seconds_elapsed'] = total_elapsed
    result['key'] = backtest_config['key']
    return result


async def load_ticks(backtest_config: dict) -> [dict]:
    ticks_filepath = os.path.join(backtest_config['session_dirpath'], f"ticks_cache.npy")
    if os.path.exists(ticks_filepath):
        print('loading cached trade list', ticks_filepath)
        ticks = np.load(ticks_filepath, allow_pickle=True)
        if ticks.dtype != np.float64:
            print('converting cached trade list')
            np.save(ticks_filepath, ticks.astype("float64"))
            ticks = np.load(ticks_filepath, allow_pickle=True)
            gc.collect()
    else:
        agg_trades = await load_trades(backtest_config['exchange'], backtest_config['user'],
                                       backtest_config['symbol'], backtest_config['n_days'])
        print('preparing ticks...')
        ticks = prep_ticks(agg_trades)
        np.save(ticks_filepath, ticks)
    return ticks


async def prep_backtest_config(config_name: str):
    backtest_config = hjson.load(open(f'backtest_configs/{config_name}.hjson'))

    exchange = backtest_config['exchange']
    user = backtest_config['user']
    symbol = backtest_config['symbol']
    session_name = backtest_config['session_name']

    session_dirpath = make_get_filepath(os.path.join(
        'backtest_results',
        exchange,
        symbol,
        f"{session_name}_{backtest_config['n_days']}_days",
        ''))
    if os.path.exists((mss := session_dirpath + 'market_specific_settings.json')):
        market_specific_settings = json.load(open(mss))
    else:
        market_specific_settings = await fetch_market_specific_settings(exchange, user, symbol)
        json.dump(market_specific_settings, open(mss, 'w'))
    backtest_config.update(market_specific_settings)

    # setting absolute min/max ranges
    for key in ['balance_pct', 'entry_qty_pct', 'close_qty_pct', 'ddown_factor', 'ema_span', 'ema_spread',
                'grid_coefficient', 'grid_spacing',
                'stop_loss_pos_reduction']:
        if key in backtest_config['ranges']:
            backtest_config['ranges'][key][0] = max(0.0, backtest_config['ranges'][key][0])
    for key in ['balance_pct', 'entry_qty_pct', 'close_qty_pct',
                'stop_loss_pos_reduction']:
        if key in backtest_config['ranges']:
            backtest_config['ranges'][key][1] = min(1.0, backtest_config['ranges'][key][1])

    backtest_config['ranges']['leverage'][1] = \
        min(backtest_config['ranges']['leverage'][1],
            backtest_config['max_leverage'])
    backtest_config['ranges']['leverage'][0] = \
        min(backtest_config['ranges']['leverage'][0],
            backtest_config['ranges']['leverage'][1])
    
    backtest_config['session_dirpath'] = session_dirpath

    return backtest_config


async def main():
    config_name = sys.argv[1]
    backtest_config = await prep_backtest_config(config_name)
    ticks = await load_ticks(backtest_config)
    await jackrabbit(ticks, backtest_config)


if __name__ == '__main__':
    asyncio.run(main())
