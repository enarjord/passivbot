from __future__ import annotations
import asyncio
import json
import os
import sys
from copy import deepcopy
from hashlib import sha256
from time import time
from datetime import datetime
from operator import mul
from operator import truediv as div
from typing import Iterable, Optional, Union
from typing import Any, Dict, Iterator
from nptyping import NDArray, Float64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.json._normalize import nested_to_record

from downloader import Downloader, prep_backtest_config
from jitted import calc_diff, round_, iter_entries, iter_long_closes, iter_shrt_closes, calc_available_margin, \
    calc_liq_price_binance, calc_liq_price_bybit, calc_new_psize_pprice, calc_long_pnl, calc_shrt_pnl, calc_cost, \
    iter_indicator_chunks, round_dynamic
from passivbot import make_get_filepath, ts_to_date, get_keys


def dump_plots(result: dict, fdf: pd.DataFrame, df: pd.DataFrame):
    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)

    def gain_conv(x):
        return x * 100 - 100

    lines = []
    lines.append(f"gain percentage {round_dynamic(result['result']['gain'] * 100 - 100, 4)}%")
    lines.append(f"average_daily_gain percentage {round_dynamic((result['result']['average_daily_gain'] - 1) * 100, 3)}%")
    lines.append(f"closest_liq percentage {round_dynamic(result['result']['closest_liq'] * 100, 4)}%")

    for key in [k for k in result['result'] if k not in ['gain', 'average_daily_gain', 'closest_liq', 'do_long', 'do_shrt']]:
        lines.append(f"{key} {round_dynamic(result['result'][key], 6)}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    live_config = candidate_to_live_config(result)
    json.dump(live_config, open(result['plots_dirpath'] + 'live_config.json', 'w'), indent=4)
    json.dump(result, open(result['plots_dirpath'] + 'result.json', 'w'), indent=4)

    ema = df.price.ewm(span=result['ema_span'], adjust=False).mean()
    df.loc[:, 'bid_thr'] = ema * (1 - result['ema_spread'])
    df.loc[:, 'ask_thr'] = ema * (1 + result['ema_spread'])

    print('writing backtest_result.txt...')
    with open(f"{result['plots_dirpath']}backtest_result.txt", 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')

    print('plotting balance and equity...')
    plt.clf()
    fdf.balance.plot()
    fdf.equity.plot()
    plt.savefig(f"{result['plots_dirpath']}balance_and_equity.png")

    print('plotting backtest whole and in chunks...')
    n_parts = 7
    # n_parts = int(round_up(result['n_days'], 1.0))
    for z in range(n_parts):
        start_ = z / n_parts
        end_ = (z + 1) / n_parts
        print(start_, end_)
        fig = plot_fills(df, fdf.iloc[int(len(fdf) * start_):int(len(fdf) * end_)], liq_thr=0.1)
        fig.savefig(f"{result['plots_dirpath']}backtest_{z + 1}of{n_parts}.png")
    fig = plot_fills(df, fdf, liq_thr=0.1)
    fig.savefig(f"{result['plots_dirpath']}whole_backtest.png")

    print('plotting pos sizes...')
    plt.clf()
    fdf.long_psize.plot()
    fdf.shrt_psize.plot()
    plt.savefig(f"{result['plots_dirpath']}psizes_plot.png")

    print('plotting average daily gain...')
    adg_ = fdf.average_daily_gain
    adg_.index = np.linspace(0.0, 1.0, len(fdf))
    plt.clf()
    adg_c = adg_.iloc[int(len(fdf) * 0.1):]  # skipping first 10%
    print('min max', adg_c.min(), adg_c.max())
    adg_c.plot()
    plt.savefig(f"{result['plots_dirpath']}average_daily_gain_plot.png")


def plot_fills(df, fdf, side_: int = 0, liq_thr=0.1):
    plt.clf()

    dfc = df.loc[fdf.index[0]:fdf.index[-1]]
    dfc.price.plot(style='y-')
    if 'bid_thr' in dfc.columns:
        dfc.bid_thr.plot(style='b-')
    if 'ask_thr' in dfc.columns:
        dfc.ask_thr.plot(style='r-')

    if side_ >= 0:
        longs = fdf[fdf.pside == 'long']
        lentry = longs[(longs.type == 'long_entry') | (longs.type == 'long_reentry')]
        lclose = longs[longs.type == 'long_close']
        lstopclose = longs[longs.type == 'stop_loss_long_close']
        lstopentry = longs[longs.type == 'stop_loss_long_entry']
        lentry.price.plot(style='b.')
        lstopentry.price.plot(style='bx')
        lclose.price.plot(style='r.')
        lstopclose.price.plot(style=('rx'))
        longs.long_pprice.fillna(method='ffill').plot(style='b--')
    if side_ <= 0:
        shrts = fdf[fdf.pside == 'shrt']
        sentry = shrts[(shrts.type == 'shrt_entry') | (shrts.type == 'shrt_reentry')]
        sclose = shrts[shrts.type == 'shrt_close']
        sstopclose = shrts[shrts.type == 'stop_loss_shrt_close']
        sstopentry = shrts[shrts.type == 'stop_loss_shrt_entry']
        sentry.price.plot(style='r.')
        sstopentry.price.plot(style='rx')
        sclose.price.plot(style='b.')
        sstopclose.price.plot(style=('bx'))
        shrts.shrt_pprice.fillna(method='ffill').plot(style='r--')

    if 'liq_price' in fdf.columns:
        fdf.liq_price.where(fdf.liq_diff < liq_thr, np.nan).plot(style='k--')
    return plt


def backtest(config: dict, ticks: np.ndarray, do_print=False) -> (list, list, bool):
    if len(ticks) <= config['ema_span']:
        return [], [], False
    long_psize, long_pprice = 0.0, 0.0
    shrt_psize, shrt_pprice = 0.0, 0.0
    liq_price, liq_diff = 0.0, 1.0
    balance = config['starting_balance']

    if all(x in config for x in ['long_pprice', 'long_psize', 'shrt_pprice', 'shrt_psize']):
        long_pprice, long_psize, shrt_pprice, shrt_psize = (
            config["long_pprice"],
            config["long_psize"],
            config["shrt_pprice"],
            config["shrt_psize"],
        )
    else:
        long_pprice, long_psize, shrt_pprice, shrt_psize = 0.0, 0.0, 0.0, 0.0

    pnl_plus_fees_cumsum, loss_cumsum, profit_cumsum, fee_paid_cumsum = 0.0, 0.0, 0.0, 0.0

    xk = {k: float(config[k]) for k in get_keys()}

    if config['exchange'] == 'binance':
        calc_liq_price = calc_liq_price_binance
    elif config['exchange'] == 'bybit':
        calc_liq_price = calc_liq_price_bybit

    prev_long_close_ts, prev_long_entry_ts, prev_long_close_price = 0, 0, 0.0
    prev_shrt_close_ts, prev_shrt_entry_ts, prev_shrt_close_price = 0, 0, 0.0
    latency_simulation_ms = config['latency_simulation_ms'] \
        if 'latency_simulation_ms' in config else 1000

    next_stats_update = 0
    stats = []

    def stats_update():
        upnl_l = x if (x := calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'],
                                          xk['contract_multiplier'])) == x else 0.0
        upnl_s = y if (y := calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'],
                                          xk['contract_multiplier'])) == y else 0.0
        stats.append({'timestamp': tick[2],
                      'balance': balance,  # Redundant with fills, but makes plotting easier
                      'equity': balance + upnl_l + upnl_s})

    all_fills = []
    fills = []
    bids, asks = [], []
    ob = [min(ticks[0][0], ticks[1][0]), max(ticks[0][0], ticks[1][0])]
    ema_span = int(round(config['ema_span']))
    # emas = calc_emas(ticks[:, 0], ema_span)
    # price_stds = calc_stds(ticks[:, 0], ema_span)
    # volatilities = price_stds / emas

    ema_std_iterator = iter_indicator_chunks(ticks[:, 0], ema_span)
    ema_chunk, std_chunk, z = next(ema_std_iterator)
    volatility_chunk = std_chunk / ema_chunk
    zc = 0

    closest_liq = 1.0

    prev_update_plus_delay = ticks[ema_span][2] + latency_simulation_ms
    update_triggered = False
    prev_update_plus_5sec = 0

    tick = ticks[0]
    stats_update()

    # tick tuple: (price, buyer_maker, timestamp)
    for k, tick in enumerate(ticks[ema_span:], start=ema_span):

        chunk_i = k - zc
        if chunk_i >= len(ema_chunk):
            ema_chunk, std_chunk, z = next(ema_std_iterator)
            volatility_chunk = std_chunk / ema_chunk
            zc = z * len(ema_chunk)
            chunk_i = k - zc

        # Update the stats every 1/2 hour
        if tick[2] > next_stats_update:
            closest_liq = min(closest_liq, calc_diff(liq_price, tick[0]))
            stats_update()
            next_stats_update = tick[2] + 1000 * 60 * 30

        fills = []
        if tick[1]:
            if liq_diff < 0.05 and long_psize > -shrt_psize and tick[0] <= liq_price:
                fills.append({'qty': -long_psize, 'price': tick[0], 'pside': 'long',
                              'type': 'long_liquidation', 'side': 'sel',
                              'pnl': calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'],
                                                   xk['contract_multiplier']),
                              'fee_paid': -calc_cost(long_psize, tick[0], xk['inverse'],
                                                     xk['contract_multiplier']) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': 0.0, 'shrt_psize': 0.0,
                              'shrt_pprice': 0.0, 'liq_price': 0.0, 'liq_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
            else:
                if bids:
                    if tick[0] <= bids[0][1]:
                        update_triggered = True
                    while bids:
                        if tick[0] < bids[0][1]:
                            bid = bids.pop(0)
                            fill = {'qty': bid[0], 'price': bid[1], 'side': 'buy', 'type': bid[4],
                                    'fee_paid': -calc_cost(bid[0], bid[1], xk['inverse'],
                                                           xk['contract_multiplier']) * config['maker_fee']}
                            if 'close' in bid[4]:
                                fill['pnl'] = calc_shrt_pnl(shrt_pprice, bid[1], bid[0],
                                                            xk['inverse'],
                                                            xk['contract_multiplier'])
                                shrt_psize = round_(shrt_psize + bid[0], config['qty_step'])
                                fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_shrt_close_ts = tick[2]
                            else:
                                fill['pnl'] = 0.0
                                long_psize, long_pprice = calc_new_psize_pprice(long_psize,
                                                                                long_pprice, bid[0],
                                                                                bid[1],
                                                                                xk['qty_step'])
                                fill.update({'pside': 'long', 'long_psize': bid[2],
                                             'long_pprice': bid[3], 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_long_entry_ts = tick[2]
                            fills.append(fill)
                        else:
                            break
            ob[0] = tick[0]
        else:
            if liq_diff < 0.05 and -shrt_psize > long_psize and tick[0] >= liq_price:
                fills.append({'qty': -shrt_psize, 'price': tick[0], 'pside': 'shrt',
                              'type': 'shrt_liquidation', 'side': 'buy',
                              'pnl': calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'],
                                                   xk['contract_multiplier']),
                              'fee_paid': -calc_cost(shrt_psize, tick[0], xk['inverse'],
                                                     xk['contract_multiplier']) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': 0.0, 'shrt_psize': 0.0,
                              'shrt_pprice': 0.0, 'liq_price': 0.0, 'liq_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
            else:
                if asks:
                    if tick[0] >= asks[0][1]:
                        update_triggered = True
                    while asks:
                        if tick[0] > asks[0][1]:
                            ask = asks.pop(0)
                            fill = {'qty': ask[0], 'price': ask[1], 'side': 'sel', 'type': ask[4],
                                    'fee_paid': -calc_cost(ask[0], ask[1], xk['inverse'],
                                                           xk['contract_multiplier']) * config['maker_fee']}
                            if 'close' in ask[4]:
                                fill['pnl'] = calc_long_pnl(long_pprice, ask[1], ask[0],
                                                            xk['inverse'],
                                                            xk['contract_multiplier'])
                                long_psize = round_(long_psize + ask[0], config['qty_step'])
                                fill.update({'pside': 'long', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_long_close_ts = tick[2]
                            else:
                                fill['pnl'] = 0.0
                                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize,
                                                                                shrt_pprice, ask[0],
                                                                                ask[1],
                                                                                xk['qty_step'])
                                fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_shrt_entry_ts = tick[2]
                            liq_diff = calc_diff(liq_price, tick[0])
                            fill.update({'liq_price': liq_price, 'liq_diff': liq_diff})
                            fills.append(fill)
                        else:
                            break
            ob[1] = tick[0]

        if tick[2] > prev_update_plus_delay and (update_triggered or tick[2] > prev_update_plus_5sec):
            prev_update_plus_delay = tick[2] + latency_simulation_ms
            prev_update_plus_5sec = tick[2] + 5000
            update_triggered = False
            bids, asks = [], []
            liq_diff = calc_diff(liq_price, tick[0])
            closest_liq = min(closest_liq, liq_diff)
            for tpl in iter_entries(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                    liq_price, ob[0], ob[1], ema_chunk[k - zc], tick[0],
                                    volatility_chunk[k - zc], **xk):
                if len(bids) > 2 and len(asks) > 2:
                    break
                if tpl[0] > 0.0:
                    bids.append(tpl)
                elif tpl[0] < 0.0:
                    asks.append(tpl)
                else:
                    break
            if tick[0] <= shrt_pprice and shrt_pprice > 0.0:
                for tpl in iter_shrt_closes(balance, shrt_psize, shrt_pprice, ob[0], **xk):
                    bids.append(list(tpl) + [shrt_pprice, 'shrt_close'])
            if tick[0] >= long_pprice and long_pprice > 0.0:
                for tpl in iter_long_closes(balance, long_psize, long_pprice, ob[1], **xk):
                    asks.append(list(tpl) + [long_pprice, 'long_close'])
            bids = sorted(bids, key=lambda x: x[1], reverse=True)
            asks = sorted(asks, key=lambda x: x[1])

        if len(fills) > 0:
            for fill in fills:
                balance += fill['pnl'] + fill['fee_paid']
                upnl_l = calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'],
                                       xk['contract_multiplier'])
                upnl_s = calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'],
                                       xk['contract_multiplier'])

                liq_price = calc_liq_price(balance, long_psize, long_pprice,
                                           shrt_psize, shrt_pprice, xk['inverse'],
                                           xk['contract_multiplier'], config['max_leverage'])
                liq_diff = calc_diff(liq_price, tick[0])
                fill.update({'liq_price': liq_price, 'liq_diff': liq_diff})

                fill['equity'] = balance + upnl_l + upnl_s
                fill['available_margin'] = calc_available_margin(
                    balance, long_psize, long_pprice, shrt_psize, shrt_pprice, tick[0],
                    xk['inverse'], xk['contract_multiplier'], xk['leverage']
                )
                for side_ in ['long', 'shrt']:
                    if fill[f'{side_}_pprice'] == 0.0:
                        fill[f'{side_}_pprice'] = np.nan
                fill['balance'] = balance
                fill['timestamp'] = tick[2]
                fill['trade_id'] = k
                fill['gain'] = fill['equity'] / config['starting_balance']
                fill['n_days'] = (tick[2] - ticks[ema_span][2]) / (1000 * 60 * 60 * 24)
                fill['closest_liq'] = closest_liq
                try:
                    fill['average_daily_gain'] = fill['gain'] ** (1 / fill['n_days']) \
                        if (fill['n_days'] > 0.5 and fill['gain'] > 0.0) else 0.0
                except:
                    fill['average_daily_gain'] = 0.0
                all_fills.append(fill)
                if balance <= 0.0 or 'liquidation' in fill['type']:
                    return all_fills, stats, False
            if do_print:
                line = f"\r{k / len(ticks):.3f} "
                line += f"adg {all_fills[-1]['average_daily_gain']:.4f} "
                line += f"closest_liq {closest_liq:.4f} "
                print(line, end=' ')

    tick = ticks[-1]
    stats_update()
    return all_fills, stats, True

def candidate_to_live_config(candidate: dict) -> dict:
    live_config = {}
    for k in ["config_name", "logging_level", "ddown_factor", "qty_pct", "leverage",
              "n_close_orders", "grid_spacing", "pos_margin_grid_coeff",
              "volatility_grid_coeff", "volatility_qty_coeff", "min_markup",
              "markup_range", "do_long", "do_shrt", "ema_span", "ema_spread", "stop_loss_liq_diff",
              "stop_loss_pos_pct", "symbol"]:
        if k in candidate:
            live_config[k] = candidate[k]
        else:
            live_config[k] = 0.0
    for k in ['do_long', 'do_shrt']:
        live_config[k] = bool(live_config[k])
    return live_config

# TODO: Make a class Returns?
# Dict of interesting periods and their associated number of seconds
PERIODS = {
    'daily': 60 * 60 * 24,
    'weekly': 60 * 60 * 24 * 7,
    'monthly': 60 * 60 * 24 * 365.25 / 12,
    'yearly': 60 * 60 * 24 * 365.25
}

METRICS_OBJ = ["average_daily_gain", "returns_daily", "sharpe_ratio_daily", "VWR_daily"]

def result_sampled_default() -> dict:
    result = {}
    for period, sec in PERIODS.items():
        result["returns_" + period] = 0.0
        result["sharpe_ratio_" + period] = 0.0
        result["VWR_" + period] = 0.0
    return result

def prepare_result_sampled(stats: list, bc: dict) -> (pd.DataFrame, dict):
    sdf = pd.DataFrame(stats).set_index("timestamp")

    if sdf.empty:
        return sdf, result_sampled_default()

    sample_period = "1H"
    sample_sec = pd.to_timedelta(sample_period).seconds

    equity_start = stats[0]["equity"]
    equity_end = stats[-1]["equity"]

    sdf.index = pd.to_datetime(sdf.index, unit="ms")
    sdf = sdf.resample(sample_period).last()


    returns = sdf.equity.pct_change()
    returns[0] = sdf.equity[0] / equity_start - 1
    returns.fillna(0, inplace=True)
    # returns_diff = (sdf['balance'].pad() / (equity_start * np.exp(returns_log_mean * np.arange(1, N+1)))) - 1

    N = len(returns)
    returns_mean = np.exp(np.mean(np.log(returns + 1))) - 1  # Geometrical mean

    #########################################
    ### Variability-Weighted Return (VWR) ###
    #########################################

    # See https://www.crystalbull.com/sharpe-ratio-better-with-log-returns/
    returns_log = np.log(1 + returns)
    returns_log_mean = np.log(equity_end / equity_start) / N
    # returns_mean = np.exp(returns_log_mean) - 1 # = geometrical mean != returns.mean()

    # Relative difference of the equity E_i and the zero-variability ideal equity E'_i: (E_i / E'i) - 1
    equity_diff = (sdf["equity"].pad() / (equity_start * np.exp(returns_log_mean * np.arange(1, N + 1)))) - 1

    # Standard deviation of equity differentials
    equity_diff_std = np.std(equity_diff, ddof=1)

    tau = bc["tau"]  # Rate at which weighting falls with increasing variability (investor tolerance)
    sdev_max = bc["sdev_max"]  # Maximum acceptable standard deviation (investor limit)

    # Weighting of the expected compounded returns for a given period (daily, ...). Note that
    # - this factor is always less than 1
    # - this factor is negative if equity_diff_std > sdev_max (hence this parameter name)
    # - the smaller (resp. bigger) tau is the quicker this factor tends to zero (resp. 1)
    VWR_weight = 1.0 - (equity_diff_std / sdev_max) ** tau

    result = {}
    for period, sec in PERIODS.items():
        # There are `periods_nb` times `sample_sec` in `period`
        periods_nb = sec / sample_sec

        # Expected compounded returns for `period` (daily returns = adg - 1)
        #  returns_expected_period = np.exp(returns_log_mean * periods_nb) - 1
        returns_expected_period = (returns_mean + 1) ** periods_nb - 1
        volatility_expected_period = returns.std() * np.sqrt(periods_nb)

        SR = returns_expected_period / volatility_expected_period  # Sharpe ratio (risk-free)
        VWR = returns_expected_period * VWR_weight

        result["returns_" + period] = returns_expected_period

        if equity_end > equity_start:
            result["sharpe_ratio_" + period] = SR
            result["VWR_" + period] = VWR if VWR > 0.0 else 0.0
        else:
            result["sharpe_ratio_" + period] = 0.0
            result["VWR_" + period] = result["returns_" + period]

    return sdf, result

def result_filled_default(bc: dict) -> dict:
    return {
        'net_pnl_plus_fees': 0.0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'fee_sum': 0.0,
        'final_equity': 0.0,
        'gain': 0.0,
        'max_drawdown': 0.0,
        'n_days': 0.0,
        'average_daily_gain': 0.0,
        'closest_liq': 0.0,
        'n_fills': 0.0,
        'n_entries': 0.0,
        'n_closes': 0.0,
        'n_reentries': 0.0,
        'n_initial_entries': 0.0,
        'n_normal_closes': 0.0,
        'n_stop_loss_closes': 0.0,
        'n_stop_loss_entries': 0.0,
        'biggest_psize': 0.0,
        'max_hrs_no_fills_same_side': 1000.0,
        'max_hrs_no_fills': 1000.0,
    }


def prepare_result_filled(fills: dict, bc: dict, end: float) -> (pd.DataFrame, dict):
    fdf = pd.DataFrame(fills)

    if fdf.empty:
        return fdf, result_filled_default(bc)

    fdf = fdf.set_index('trade_id')

    if len(longs_ := fdf[fdf.pside == 'long']) > 0:
        long_stuck = np.diff(list(longs_.timestamp) + [end]).max() / (1000 * 60 * 60)
    else:
        long_stuck = 1000.0
    if len(shrts_ := fdf[fdf.pside == 'shrt']) > 0:
        shrt_stuck = np.diff(list(shrts_.timestamp) + [end]).max() / (1000 * 60 * 60)
    else:
        shrt_stuck = 1000.0

    result = {
        'net_pnl_plus_fees': fdf.pnl.sum() + fdf.fee_paid.sum(),
        'profit_sum': fdf[fdf.pnl > 0.0].pnl.sum(),
        'loss_sum': fdf[fdf.pnl < 0.0].pnl.sum(),
        'fee_sum': fdf.fee_paid.sum(),
        'final_equity': fdf.iloc[-1].equity,
        'gain': (gain := fdf.iloc[-1].equity / bc['starting_balance']),
        'max_drawdown': ((fdf.equity - fdf.balance).abs() / fdf.balance).max(),
        'n_days': (n_days := (end - fdf.iloc[0].timestamp) / (1000 * 60 * 60 * 24)),
        'average_daily_gain': gain ** (1 / n_days) if gain > 0.0 and n_days > 0.0 else 0.0,
        'closest_liq': fdf.closest_liq.iloc[-1],
        'n_fills': len(fdf),
        'n_entries': len(fdf[fdf.type.str.contains('entry')]),
        'n_closes': len(fdf[fdf.type.str.contains('close')]),
        'n_reentries': len(fdf[fdf.type.str.contains('reentry')]),
        'n_initial_entries': len(fdf[fdf.type.str.contains('initial')]),
        'n_normal_closes': len(fdf[(fdf.type == 'long_close') | (fdf.type == 'shrt_close')]),
        'n_stop_loss_closes': len(fdf[(fdf.type.str.contains('stop_loss')) &
                                      (fdf.type.str.contains('close'))]),
        'n_stop_loss_entries': len(fdf[(fdf.type.str.contains('stop_loss')) &
                                       (fdf.type.str.contains('entry'))]),
        'biggest_psize': fdf[['long_psize', 'shrt_psize']].abs().max(axis=1).max(),
        'max_hrs_no_fills_long': long_stuck,
        'max_hrs_no_fills_shrt': shrt_stuck,
        'max_hrs_no_fills_same_side': max(long_stuck, shrt_stuck),
        'max_hrs_no_fills': np.diff(list(fdf.timestamp) + [end]).max() / (1000 * 60 * 60),
    }
    return fdf, result

def objective_function(
    result: dict,
    metric: str,
    bc: dict,
) -> float:
    if result['n_fills'] == 0:
        return -1

    try:
        return (
            result[metric]
            * min(1.0, bc["max_hrs_no_fills"] / result["max_hrs_no_fills"])
            * min(1.0, bc["max_hrs_no_fills_same_side"] / result["max_hrs_no_fills_same_side"])
            * min(1.0, result["closest_liq"] / bc["minimum_liquidation_distance"])
        )
    except:
        return -1


def prepare_result(fills: list, stats: list, bc: dict) -> (pd.DataFrame, pd.DataFrame, dict):

    res = {
        "do_long": bool(bc["do_long"]),
        "do_shrt": bool(bc["do_long"]),
        'starting_balance': bc['starting_balance'],
    }

    fdf, res_fill = prepare_result_filled(fills, bc, stats[-1]['timestamp'])
    sdf, res_samp = prepare_result_sampled(stats, bc)

    res.update(res_fill)
    res.update(res_samp)

    # Compute the objective from interesting metrics
    for metric in METRICS_OBJ:
        res[metric + "_obj"] = objective_function(res, metric, bc)

    # Compute the objective from the metric defined in the backtesting config
    if bc["metric"] not in res:
        res[bc["metric"] + "_obj"] = objective_function(res, bc["metric"], bc)

    res["objective"] = res[bc["metric"] + "_obj"]

    return fdf, sdf, res

def plot_wrap(bc, ticks, live_config):
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    config = {**bc, **live_config}
    print('backtesting...')
    fills, stats, did_finish = backtest(config, ticks, do_print=True)
    if not fills:
        print('no fills')
        return
    fdf, sdf, result = prepare_result(fills, stats, config)
    config['result'] = result
    config['plots_dirpath'] = make_get_filepath(os.path.join(
        config['plots_dirpath'], f"{ts_to_date(time())[:19].replace(':', '')}", '')
    )
    fdf.to_csv(config['plots_dirpath'] + "fills.csv")
    df = pd.DataFrame({'price': ticks[:, 0], 'buyer_maker': ticks[:, 1], 'timestamp': ticks[:, 2]})
    dump_plots(config, fdf, df)

# TODO: improve type hinting
class WFO:
    def update_config(self, config: Dict, split: Dict, balance_and_pos: Optional[Dict] = None):
        if balance_and_pos:
            config.update(balance_and_pos)

        config.update(
            {
                "start_date": ts_to_date((split["start_ms"] + self.ts_start) / 1000),
                "end_date": ts_to_date((split["end_ms"] + self.ts_start) / 1000),
                "n_days": split["diff_days"],
            }
        )

    def __init__(
        self,
        ticks: NDArray[(Any, 3), Float64],
        bc: dict,
        P_train: float = 0.2,
        P_test: float = 0.1,
        P_gap: float = 0.0,
        verbose: bool = True,
    ):
        self.step = {"train": P_train, "test": P_test, "gap": P_gap}
        self.ticks = ticks
        self.ts_start = ticks[0][2]
        self.ts_end = ticks[-1][2]
        self.ts_diff = self.ts_end - self.ts_start
        self.timeframe = self.diff(0, ticks.shape[0] - 1)
        self.bc = deepcopy(bc)
        self.verbose = verbose

    def __iter__(self) -> Iterator[Dict[str, Dict[str, float]]]:
        if self.verbose:
            return map(
                lambda x: {
                    "train": self.stats(**x["train"]),
                    "test": self.stats(**x["test"]),
                },
                self.chunks(),
            )
        else:
            return self.chunks()

    def backtest(self, config):
        results = []
        all_daily_returns = []
        all_objectives = []

        for splits in self:
            start, end = splits["train"]["start_idx"], splits["train"]["end_idx"]
            self.update_config(config, splits["train"])  # Update n_days and start/end date
            fills, stats, did_finish = backtest(config, self.ticks[start:end])
            _,_,result_ = prepare_result(fills, stats, self.bc)
            results.append(result_)

            all_daily_returns.append(result_["returns_daily" + "_obj"])  # stats is more accurate than fills
            all_objectives.append(result_[self.bc["metric"] + "_obj"])

        result = {}
        for k in results[0]:
            try:
                result[k] = np.mean([r[k] for r in results])
            except:
                result[k] = results[0][k]

        # Geometrical mean is often used to average returns
        result["daily_gains_gmean"] = np.exp(np.mean(np.log(np.array(all_daily_returns) + 1)))
        result["objective_gmean"] = np.exp(np.mean(np.log(np.array(all_objectives) + 1))) - 1

        return result

    def run(self):
        bc = deepcopy(self.bc)

        balance_and_pos = {
            "starting_balance": bc["starting_balance"],
            "long_pprice": 0.0,
            "long_psize": 0.0,
            "shrt_pprice": 0.0,
            "shrt_psize": 0.0,
        }

        all_daily_returns = []

        for k, split in enumerate(self):
            train = split["train"]
            test = split["test"]

            print("*** STARTIN BALANCE", balance_and_pos["starting_balance"])

            self.update_config(bc, train, balance_and_pos)

            analysis = backtest_tune(self.ticks[train["start_idx"] : train["end_idx"]], bc)
            candidate = clean_result_config(analysis.best_config)

            self.update_config(bc, test)
            fills, stats, did_finish = backtest(candidate, self.ticks[test["start_idx"] : test["end_idx"]])

            _,_,result = prepare_result(
                fills,
                stats,
                bc,
            )

            # Update the balance and positions with the last filled values of the testing run
            balance_and_pos = {key: fills[-1][key] for key in (balance_and_pos.keys() & fills[-1].keys())}
            balance_and_pos["starting_balance"] = stats[-1]["balance"]  # Same as fills

            all_daily_returns.append(result["returns_daily"])  # stats is more accurate than fills
            print("*** EQUITY", stats[-1]["equity"], all_daily_returns, "\n")

            # json.dump: candidate, result, stats, fills
            # we can load these files and generate plots/reports later
            #  with open(f"/tmp/candidate_test_{k}.json", "w") as f:
            #      json.dump(candidate, f)
            #  with open(f"/tmp/res_test_{k}.json", "w") as f:
            #      json.dump(result, f)
            #  with open(f"/tmp/stats_test_{k}.json", "w") as f:
            #      json.dump(stats, f)
            #  with open(f"/tmp/fills_test_{k}.json", "w") as f:
            #      json.dump(fills, f)

        returns_gmean = np.exp(np.mean(np.log(np.array(all_daily_returns) + 1))) - 1
        print("Geometrical mean of all the daily returns", returns_gmean)

    def chunks(self) -> Iterator[Dict[str, Dict[str, int]]]:
        for P_train_cur in np.arange(0.0, 1.0 - self.step["train"], self.step["test"]):
            train_idx, test_idx = {}, {}
            train_idx["start"] = self.find_tick_from_pct(P_train_cur)
            train_idx["end"] = self.find_tick_from_pct(P_train_cur + self.step["train"] - self.step["gap"])
            test_idx["start"] = self.find_tick_from_pct(P_train_cur + self.step["train"])
            test_idx["end"] = self.find_tick_from_pct(P_train_cur + self.step["train"] + self.step["test"])
            yield {
                "train": {
                    "start_idx": self.find_tick_from_pct(P_train_cur),
                    "end_idx": self.find_tick_from_pct(P_train_cur + self.step["train"] - self.step["gap"]),
                },
                "test": {
                    "start_idx": self.find_tick_from_pct(P_train_cur + self.step["train"]),
                    "end_idx": self.find_tick_from_pct(P_train_cur + self.step["train"] + self.step["test"]),
                },
            }

    def set_train_N(self, N: int) -> WFO:
        self.step["test"] = (1.0 - self.step["train"]) / float(N)
        self.step["gap"] = 0.0
        return self

    def set_step(self, key: str, x: float, unit: str) -> WFO:
        ts = self.convert(x, from_ts=False)[unit]
        self.step[key] = self.convert(ts)["pct"]
        return self

    def stats(self, start_idx: int, end_idx: int) -> Dict[str, float]:
        nested = {
            "start": self.diff(0, start_idx),
            "end": self.diff(0, end_idx),
            "diff": self.diff(start_idx, end_idx),
        }
        assert isinstance((ret := nested_to_record(nested, sep="_")), Dict)
        return ret

    def find_tick_from_pct(self, pct: float) -> Any:
        end_ts = self.ts_start + pct * self.timeframe["ms"]
        return min(self.timeframe["idx"], np.searchsorted(self.ticks[:, 2], end_ts))

    def convert(self, diff: float, from_ts: bool = True) -> Dict[str, float]:
        OP = div if from_ts else mul
        return {  # TODO: add pct and idx diff
            "ms": diff,
            "seconds": OP(diff, 1000.0),
            "minutes": OP(diff, (1000.0 * 60.0)),
            "hours": OP(diff, (1000.0 * 3600.0)),
            "days": OP(diff, (1000.0 * 3600.0 * 24.0)),
            "pct": OP(diff, self.ts_diff),
        }

    def diff(self, idx_start: int, idx_end: int) -> Dict[str, float]:
        return {
            "idx": idx_end - idx_start,
            **self.convert(self.ticks[idx_end][2] - self.ticks[idx_start][2]),
        }


async def main(args: list):
    backtest_config = await prep_backtest_config(args[1])
    if backtest_config['exchange'] == 'bybit' and not backtest_config['inverse']:
        print('bybit usdt linear backtesting not supported')
        return
    downloader = Downloader(backtest_config)
    ticks = await downloader.get_ticks(True)
    backtest_config['n_days'] = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    try:
        live_config = json.load(open(args[2]))
        print('backtesting and plotting given candidate')
    except Exception as e:
        print('failed to load live config')
        return
    print(json.dumps(live_config, indent=4))
    plot_wrap(backtest_config, ticks, live_config)

#  # Default: train = 20%, test = 10%, gap = 0%
#  wfo = (
#      WFO(ticks, backtest_config)
#      .set_step("train", 14, "days")
#      .set_step("test", 7, "days")
#      #  .set_step("gap", 6, "hours")
#  )
#
#  print("\n### WFO ###\n")
#  print("start :", ts_pretty(wfo.ts_start))
#  print("end   :", ts_pretty(wfo.ts_end))
#  print(
#      "\n{:.4f} days | {:.4f} hours | {:.0f} ticks\n".format(
#          wfo.timeframe["days"], wfo.timeframe["hours"], wfo.timeframe["idx"]
#      )
#  )
#
#  for key, P in wfo.step.items():
#      print(
#          "{:5s} step : {:7.4f}% | {:7.4f} days | {:9.4f} hours".format(
#              key, P * 100, P * wfo.timeframe["days"], P * wfo.timeframe["hours"]
#          )
#      )
#
#  for split in wfo:
#      train = split["train"]
#      test = split["test"]
#
#      print("")
#      for key, stats in split.items():
#          print(
#              "{:5s} : {} -> {} | {:6.2f}%  -> {:6.2f}% | {:6.2f} days  -> {:6.2f} days  | {:9d}  -> {:9d} [+{:8d} ticks]".format(
#                  key,
#                  ts_pretty(stats["start_ms"] + wfo.ts_start),
#                  ts_pretty(stats["end_ms"] + wfo.ts_start),
#                  stats["start_pct"] * 100,
#                  stats["end_pct"] * 100,
#                  stats["start_days"],
#                  stats["end_days"],
#                  stats["start_idx"],
#                  stats["end_idx"],
#                  stats["diff_idx"],
#              )
#          )
#
#  wfo.run()


if __name__ == '__main__':
    asyncio.run(main(sys.argv))

