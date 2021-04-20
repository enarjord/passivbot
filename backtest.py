import gc
import glob
import logging
import pprint
from hashlib import sha256
from typing import Union

import matplotlib.pyplot as plt
import nevergrad as ng
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch

from downloader import Downloader, prep_backtest_config
from passivbot import *
from reporter import LogReporter

os.environ['TUNE_GLOBAL_CHECKPOINT_S'] = '120'


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
    lines.append(f"n entries {result['n_entries']}")
    lines.append(f"n closes {result['n_closes']}")
    lines.append(f"n reentries {result['n_reentries']}")
    lines.append(f"n initial entries {result['n_initial_entries']}")
    lines.append(f"n normal closes {result['n_normal_closes']}")
    lines.append(f"n stop loss closes {result['n_stop_loss_closes']}")
    lines.append(f"n stop loss entries {result['n_stop_loss_entries']}")
    lines.append(f"biggest_psize {round(result['biggest_psize'], 10)}")
    lines.append(f"closest liq percentage {result['closest_liq'] * 100:.4f}%")
    lines.append(f"max n hours stuck {result['max_n_hours_stuck']:.2f}")
    lines.append(f"starting balance {result['starting_balance']}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    live_config = candidate_to_live_config(result)
    json.dump(live_config, open(result['session_dirpath'] + 'live_config.json', 'w'), indent=4)

    json.dump(result, open(result['session_dirpath'] + 'result.json', 'w'), indent=4)

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
    fdf.balance.plot()
    fdf.equity.plot()
    plt.savefig(f"{result['session_dirpath']}balance_and_equity.png")

    print('plotting backtest whole and in chunks...')
    n_parts = 7
    # n_parts = int(round_up(result['n_days'], 1.0))
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
    fdf.long_psize.plot()
    fdf.shrt_psize.plot()
    plt.savefig(f"{result['session_dirpath']}psizes_plot.png")

    print('plotting average daily gain...')
    adg_ = fdf.average_daily_gain
    adg_.index = np.linspace(0.0, 1.0, len(fdf))
    plt.clf()
    adg_c = adg_.iloc[int(len(fdf) * 0.1):]  # skipping first 10%
    print('min max', adg_c.min(), adg_c.max())
    adg_c.plot()
    plt.savefig(f"{result['session_dirpath']}average_daily_gain_plot.png")


def plot_fills(df, fdf, side_: int = 0, liq_thr=0.1):
    plt.clf()

    df.loc[fdf.index[0]:fdf.index[-1]].price.plot(style='y-')

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


def backtest(config: dict, ticks: np.ndarray, return_fills=False, do_print=False) -> (list, bool):
    long_psize, long_pprice = 0.0, np.nan
    shrt_psize, shrt_pprice = 0.0, np.nan
    liq_price = 0.0
    balance = config['starting_balance']

    pnl_plus_fees_cumsum, loss_cumsum, profit_cumsum, fee_paid_cumsum = 0.0, 0.0, 0.0, 0.0

    if config['inverse']:
        long_pnl_f = calc_long_pnl_inverse
        shrt_pnl_f = calc_shrt_pnl_inverse
        cost_f = calc_cost_inverse
        available_margin_f = lambda balance, long_psize, long_pprice, shrt_psize, \
                                    shrt_pprice, last_price: \
            calc_available_margin_inverse(config['contract_multiplier'], config['leverage'], balance,
                                          long_psize, long_pprice,
                                          shrt_psize, shrt_pprice, last_price)

        iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, liq_price, \
                              highest_bid, lowest_ask, ema, last_price, do_long, do_shrt: \
            iter_entries_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                 config['min_cost'], config['contract_multiplier'],
                                 config['ddown_factor'], config['qty_pct'],
                                 config['leverage'], config['grid_spacing'],
                                 config['grid_coefficient'], config['ema_spread'],
                                 config['stop_loss_liq_diff'], config['stop_loss_pos_pct'],
                                 balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                 liq_price, highest_bid, lowest_ask, ema, last_price, do_long,
                                 do_shrt)

        iter_long_closes = lambda balance, psize, pprice, lowest_ask: \
            iter_long_closes_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                     config['min_cost'], config['contract_multiplier'], config['qty_pct'],
                                     config['leverage'], config['min_markup'], config['markup_range'],
                                     config['n_close_orders'], balance, psize, pprice,
                                     lowest_ask)
        iter_shrt_closes = lambda balance, psize, pprice, highest_bid: \
            iter_shrt_closes_inverse(config['price_step'], config['qty_step'], config['min_qty'],
                                     config['min_cost'], config['contract_multiplier'], config['qty_pct'],
                                     config['leverage'], config['min_markup'], config['markup_range'],
                                     config['n_close_orders'], balance, psize, pprice,
                                     highest_bid)
        if config['exchange'] == 'binance':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_binance_inverse(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                           config['leverage'],
                                                           contract_multiplier=config['contract_multiplier'])
        elif config['exchange'] == 'bybit':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_bybit_inverse(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                         config['leverage'])
    else:
        long_pnl_f = calc_long_pnl_linear
        shrt_pnl_f = calc_shrt_pnl_linear
        cost_f = calc_cost_linear
        available_margin_f = lambda balance, long_psize, long_pprice, shrt_psize, \
                                    shrt_pprice, last_price: \
            calc_available_margin_linear(config['contract_multiplier'], config['leverage'], balance,
                                         long_psize, long_pprice,
                                         shrt_psize, shrt_pprice, last_price)

        iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, \
                              liq_price, highest_bid, lowest_ask, ema, last_price, do_long, do_shrt: \
            iter_entries_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                config['min_cost'], config['contract_multiplier'],
                                config['ddown_factor'], config['qty_pct'],
                                config['leverage'], config['grid_spacing'],
                                config['grid_coefficient'], config['ema_spread'],
                                config['stop_loss_liq_diff'], config['stop_loss_pos_pct'],
                                balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                liq_price, highest_bid, lowest_ask, ema, last_price, do_long,
                                do_shrt)

        iter_long_closes = lambda balance, psize, pprice, lowest_ask: \
            iter_long_closes_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                    config['min_cost'], config['contract_multiplier'],
                                    config['qty_pct'], config['leverage'], config['min_markup'],
                                    config['markup_range'], config['n_close_orders'], balance,
                                    psize, pprice, lowest_ask)
        iter_shrt_closes = lambda balance, psize, pprice, highest_bid: \
            iter_shrt_closes_linear(config['price_step'], config['qty_step'], config['min_qty'],
                                    config['min_cost'], config['contract_multiplier'],
                                    config['qty_pct'], config['leverage'],
                                    config['min_markup'], config['markup_range'],
                                    config['n_close_orders'], balance, psize, pprice,
                                    highest_bid)

        if config['exchange'] == 'binance':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_binance_linear(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                          config['leverage'])
        elif config['exchange'] == 'bybit':
            liq_price_f = lambda balance, l_psize, l_pprice, s_psize, s_pprice: \
                calc_cross_hedge_liq_price_bybit_linear(balance, l_psize, l_pprice, s_psize, s_pprice,
                                                        config['leverage'])

    prev_long_close_ts, prev_long_entry_ts, prev_long_close_price = 0, 0, 0.0
    prev_shrt_close_ts, prev_shrt_entry_ts, prev_shrt_close_price = 0, 0, 0.0
    min_trade_delay_millis = config['latency_simulation_ms'] \
        if 'latency_simulation_ms' in config else 1000

    all_fills = []
    fills = []
    bids, asks = [], []
    stop_loss_order = None
    ob = [min(ticks[0][0], ticks[1][0]), max(ticks[0][0], ticks[1][0])]
    ema_span = int(round(config['ema_span']))
    emas = calc_emas(ticks[:,0], ema_span)
    price_stds = calc_stds(ticks[:,0], ema_span)
    # tick tuple: (price, buyer_maker, timestamp)
    delayed_update = ticks[0][2] + min_trade_delay_millis
    next_update_ts = 0
    for k, tick in enumerate(ticks[ema_span:], start=ema_span):
        liq_diff = calc_diff(liq_price, tick[0])
        if tick[2] > delayed_update:
            # after simulated delay, update open orders
            bids, asks = [], []
            stop_loss_order = (0.0, 0.0, 0.0, 0.0, '')
            tampered_longpsize, tampered_shrtpsize = long_psize, shrt_psize
            next_update_ts = tick[2] + 5000
            for tpl in iter_entries(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                    liq_price, ob[0], ob[1], emas[k], tick[0],
                                    config['do_long'], config['do_shrt']):
                if len(bids) > 2 and len(asks) > 2:
                    break
                if tpl[0] > 0.0:
                    bids.append(tpl)
                    if 'close' in tpl[4]:
                        tampered_shrtpsize = tpl[2]
                elif tpl[0] < 0.0:
                    asks.append(tpl)
                    if 'close' in tpl[4]:
                        tampered_longpsize = tpl[2]
                else:
                    break
            if tick[0] <= shrt_pprice:
                for tpl in iter_shrt_closes(balance, tampered_shrtpsize, shrt_pprice, ob[0]):
                    bids.append(list(tpl) + [shrt_pprice, 'shrt_close'])
            if tick[0] >= long_pprice:
                for tpl in iter_long_closes(balance, tampered_longpsize, long_pprice, ob[1]):
                    asks.append(list(tpl) + [long_pprice, 'long_close'])
            bids = sorted(bids, key=lambda x: x[1], reverse=True)
            asks = sorted(asks, key=lambda x: x[1])
            delayed_update = 9e13
        elif delayed_update == 9e13:
            if fills or \
                    (config['do_long'] and long_psize == 0.0) or \
                    (config['do_shrt'] and shrt_psize == 0.0) or \
                    liq_diff < config['stop_loss_liq_diff'] or \
                    stop_loss_order[0] != 0.0 or \
                    tick[2] > next_update_ts:
                delayed_update = tick[2] + min_trade_delay_millis

        fills = []
        if tick[1]:
            if liq_diff < 0.05 and long_psize > -shrt_psize and tick[0] <= liq_price:
                fills.append({'qty': -long_psize, 'price': tick[0], 'pside': 'long',
                              'type': 'long_liquidation', 'side': 'sel',
                              'pnl': long_pnl_f(long_pprice, tick[0], long_psize),
                              'fee_paid': -cost_f(long_psize, tick[0]) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': np.nan, 'shrt_psize': 0.0,
                              'shrt_pprice': np.nan, 'liq_price': 0.0, 'liq_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, np.nan, 0.0, np.nan
            else:
                while bids:
                    if tick[0] < bids[0][1]:
                        bid = bids.pop(0)
                        fill = {'qty': bid[0], 'price': bid[1], 'side': 'buy', 'type': bid[4],
                                'fee_paid': -cost_f(bid[0], bid[1]) * config['maker_fee']}
                        if 'close' in bid[4]:
                            fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                         'long_pprice': long_pprice, 'shrt_psize': bid[2],
                                         'shrt_pprice': bid[3]})
                            shrt_psize = bid[2]
                            fill['pnl'] = shrt_pnl_f(shrt_pprice, bid[1], bid[0])
                            prev_shrt_close_ts = tick[2]
                        else:
                            fill.update({'pside': 'long', 'long_psize': bid[2],
                                         'long_pprice': bid[3], 'shrt_psize': shrt_psize,
                                         'shrt_pprice': shrt_pprice})
                            long_psize = bid[2]
                            long_pprice = bid[3]
                            fill['pnl'] = 0.0
                            prev_long_entry_ts = tick[2]
                        liq_price = liq_price_f(balance, long_psize, long_pprice,
                                                shrt_psize, shrt_pprice)
                        liq_diff = calc_diff(liq_price, tick[0])
                        fill.update({'liq_price': liq_price, 'liq_diff': liq_diff})
                        fills.append(fill)
                    else:
                        break
            ob[0] = tick[0]
        else:
            if liq_diff < 0.05 and -shrt_psize > long_psize and tick[0] >= liq_price:
                fills.append({'qty': -shrt_psize, 'price': tick[0], 'pside': 'shrt',
                              'type': 'shrt_liquidation', 'side': 'buy',
                              'pnl': shrt_pnl_f(shrt_pprice, tick[0], shrt_psize),
                              'fee_paid': -cost_f(shrt_psize, tick[0]) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': np.nan, 'shrt_psize': 0.0,
                              'shrt_pprice': np.nan, 'liq_price': 0.0, 'liq_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, np.nan, 0.0, np.nan
            else:
                while asks:
                    if tick[0] > asks[0][1]:
                        ask = asks.pop(0)
                        fill = {'qty': ask[0], 'price': ask[1], 'side': 'sel', 'type': ask[4],
                                'fee_paid': -cost_f(ask[0], ask[1]) * config['maker_fee']}
                        if 'close' in ask[4]:
                            fill.update({'pside': 'long', 'long_psize': ask[2],
                                         'long_pprice': ask[3], 'shrt_psize': shrt_psize,
                                         'shrt_pprice': shrt_pprice})
                            long_psize = ask[2]
                            fill['pnl'] = long_pnl_f(long_pprice, ask[1], ask[0])
                            prev_long_close_ts = tick[2]
                        else:
                            fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                         'long_pprice': long_pprice, 'shrt_psize': ask[2],
                                         'shrt_pprice': ask[3]})
                            shrt_psize = ask[2]
                            shrt_pprice = ask[3]
                            fill['pnl'] = 0.0
                            prev_shrt_entry_ts = tick[2]
                        liq_price = liq_price_f(balance, long_psize, long_pprice,
                                                shrt_psize, shrt_pprice)
                        liq_diff = calc_diff(liq_price, tick[0])
                        fill.update({'liq_price': liq_price, 'liq_diff': liq_diff})
                        fills.append(fill)
                    else:
                        break
            ob[1] = tick[0]

        if len(fills) > 0:
            for fill in fills:
                balance += fill['pnl'] + fill['fee_paid']
                ms_since_long_pos_change = tick[2] - prev_long_fill_ts \
                    if (prev_long_fill_ts := max(prev_long_close_ts, prev_long_entry_ts)) > 0 else 0
                ms_since_shrt_pos_change = tick[2] - prev_shrt_fill_ts \
                    if (prev_shrt_fill_ts := max(prev_shrt_close_ts, prev_shrt_entry_ts)) > 0 else 0

                if ('stop_loss' in fill['type'] and 'close' in fill['type']) \
                        or 'liquidation' in fill['type']:
                    loss_cumsum += fill['pnl']
                else:
                    profit_cumsum += fill['pnl']
                fee_paid_cumsum += fill['fee_paid']
                pnl_plus_fees_cumsum += fill['pnl'] + fill['fee_paid']
                upnl_l = x if (x := long_pnl_f(long_pprice, tick[0], long_psize)) == x else 0.0
                upnl_s = y if (y := shrt_pnl_f(shrt_pprice, tick[0], shrt_psize)) == y else 0.0
                fill['equity'] = balance + upnl_l + upnl_s
                fill['pnl_plus_fees_cumsum'] = pnl_plus_fees_cumsum
                fill['loss_cumsum'] = loss_cumsum
                fill['profit_cumsum'] = profit_cumsum
                fill['fee_paid_cumsum'] = fee_paid_cumsum
                fill['available_margin'] = available_margin_f(balance, long_psize, long_pprice,
                                                              shrt_psize, shrt_pprice, tick[0])
                fill['balance'] = balance
                fill['timestamp'] = tick[2]
                fill['trade_id'] = k
                fill['progress'] = k / len(ticks)
                fill['drawdown'] = calc_diff(fill['balance'], fill['equity'])
                fill['balance_starting_balance_ratio'] = fill['balance'] / config['starting_balance']
                fill['equity_starting_balance_ratio'] = fill['equity'] / config['starting_balance']
                fill['equity_balance_ratio'] = fill['equity'] / fill['balance']
                fill['gain'] = fill['equity_starting_balance_ratio']
                fill['n_days'] = (tick[2] - ticks[0][2]) / (1000 * 60 * 60 * 24)
                try:
                    fill['average_daily_gain'] = fill['gain'] ** (1 / fill['n_days']) \
                        if (fill['n_days'] > 0.5 and fill['gain'] > 0.0) else 0.0
                except:
                    fill['average_daily_gain'] = 0.0
                fill['hours_since_long_pos_change'] = (lc := ms_since_long_pos_change / (1000 * 60 * 60))
                fill['hours_since_shrt_pos_change'] = (sc := ms_since_shrt_pos_change / (1000 * 60 * 60))
                fill['hours_since_pos_change_max'] = max(lc, sc)
                all_fills.append(fill)
                if balance <= 0.0 or 'liquidation' in fill['type']:
                    if return_fills:
                        return all_fills, False
                    else:
                        result = prepare_result(all_fills, ticks, config['do_long'], config['do_shrt'])
                        objective = objective_function(result, config['minimum_liquidation_distance'],
                                                       config['minimum_daily_entries'])
                        tune.report(objective=objective, daily_gain=result['average_daily_gain'],
                                    closest_liquidation=result['closest_liq'])
                        del all_fills
                        gc.collect()
                        return objective
            if do_print:
                line = f"\r{all_fills[-1]['progress']:.3f} "
                line += f"adg {all_fills[-1]['average_daily_gain']:.4f} "
                print(line, end=' ')
            # print(f"\r{k / len(ticks):.2f} ", end=' ')
    if return_fills:
        return all_fills, True
    else:
        result = prepare_result(all_fills, ticks, config['do_long'], config['do_shrt'])
        objective = objective_function(result, config['minimum_liquidation_distance'],
                                       config['minimum_daily_entries'])
        tune.report(objective=objective, daily_gain=result['average_daily_gain'],
                    closest_liquidation=result['closest_liq'])
        del all_fills
        gc.collect()
        return objective


def candidate_to_live_config(candidate: dict) -> dict:
    live_config = {}
    for k in ["config_name", "logging_level", "ddown_factor", "qty_pct", "leverage",
              "n_entry_orders", "n_close_orders", "grid_spacing", "grid_coefficient", "min_markup",
              "markup_range", "do_long", "do_shrt", "ema_span", "ema_spread", "stop_loss_liq_diff",
              "stop_loss_pos_pct", "symbol"]:
        if k in candidate:
            live_config[k] = candidate[k]
        else:
            if k == 'n_entry_orders':
                live_config[k] = 8
            else:
                live_config[k] = 0.0
    for k in ['do_long', 'do_shrt']:
        live_config[k] = bool(live_config[k])
    return live_config


def calc_candidate_hash_key(candidate: dict, keys: [str]) -> str:
    return sha256(json.dumps({k: candidate[k] for k in sorted(keys)
                              if k in candidate}).encode()).hexdigest()


def backtest_wrap(ticks: [dict], backtest_config: dict, do_print=False) -> (dict, pd.DataFrame):
    start_ts = time()
    fills, did_finish = backtest(backtest_config, ticks, return_fills=True, do_print=do_print)
    elapsed = time() - start_ts
    if len(fills) == 0:
        return {'average_daily_gain': 0.0, 'closest_liq': 0.0, 'max_n_hours_stuck': 1000.0}, pd.DataFrame()
    fdf = pd.DataFrame(fills).set_index('trade_id')
    result = prepare_result(fills, ticks, bool(backtest_config['do_long']), bool(backtest_config['do_shrt']))
    result['seconds_elapsed'] = elapsed
    if 'key' not in result:
        result['key'] = calc_candidate_hash_key(backtest_config, backtest_config['ranges'])
    return result, fdf


def prepare_result(fills: list, ticks: np.ndarray, do_long: bool, do_shrt: bool) -> dict:
    fdf = pd.DataFrame(fills)
    if fdf.empty:
        result = {
            'net_pnl_plus_fees': 0,
            'profit_sum': 0,
            'loss_sum': 0,
            'fee_sum': 0,
            'final_equity': 0,
            'gain': 0,
            'max_drawdown': 0,
            'n_days': 0,
            'average_daily_gain': 0,
            'closest_liq': 0,
            'n_fills': 0,
            'n_entries': 0,
            'n_closes': 0,
            'n_reentries': 0,
            'n_initial_entries': 0,
            'n_normal_closes': 0,
            'n_stop_loss_closes': 0,
            'n_stop_loss_entries': 0,
            'biggest_psize': 0,
            'max_n_hours_stuck': 0,
            'do_long': do_long,
            'do_shrt': do_shrt
        }
    else:
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
            'max_n_hours_stuck': max(fdf['hours_since_pos_change_max'].max(),
                                     (ticks[-1][2] - fills[-1]['timestamp']) / (1000 * 60 * 60)),
            'do_long': do_long,
            'do_shrt': do_shrt
        }
    return result


def objective_function(result: dict, liq_cap: float, n_daily_entries_cap: int) -> float:
    try:
        return (result['average_daily_gain'] *
                min(1.0, (result['n_entries'] / result['n_days']) / n_daily_entries_cap) *
                min(1.0, result['closest_liq'] / liq_cap))
    except Exception as e:
        print('error with objective function', e, result)
        return 0.0


def create_config(backtest_config: dict) -> dict:
    config = {k: backtest_config[k] for k in backtest_config
              if k not in {'session_name', 'user', 'symbol', 'start_date', 'end_date', 'ranges'}}
    for k in backtest_config['ranges']:
        if backtest_config['ranges'][k][0] == backtest_config['ranges'][k][1]:
            config[k] = backtest_config['ranges'][k][0]
        elif k in ['n_close_orders', 'leverage']:
            config[k] = tune.randint(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1] + 1)
        else:
            config[k] = tune.uniform(backtest_config['ranges'][k][0], backtest_config['ranges'][k][1])
    return config


def clean_start_config(start_config: dict, config: dict, ranges: dict) -> dict:
    clean_start = {}
    for k, v in start_config.items():
        if k in config and k not in ['do_long', 'do_shrt']:
            if type(config[k]) == ray.tune.sample.Float or type(config[k]) == ray.tune.sample.Integer:
                clean_start[k] = min(max(v, ranges[k][0]), ranges[k][1])
    return clean_start


def clean_result_config(config: dict) -> dict:
    for k, v in config.items():
        if type(v) == np.float64:
            config[k] = float(v)
        if type(v) == np.int64 or type(v) == np.int32 or type(v) == np.int16 or type(v) == np.int8:
            config[k] = int(v)
    return config


def backtest_tune(ticks: np.ndarray, backtest_config: dict, current_best: Union[dict, list] = None):
    config = create_config(backtest_config)
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    session_dirpath = make_get_filepath(os.path.join('reports', backtest_config['exchange'], backtest_config['symbol'],
                                                     f"{n_days}_days_{ts_to_date(time())[:19].replace(':', '')}", ''))
    iters = 10
    if 'iters' in backtest_config:
        iters = backtest_config['iters']
    else:
        print('Parameter iters should be defined in the configuration. Defaulting to 10.')
    num_cpus = 2
    if 'num_cpus' in backtest_config:
        num_cpus = backtest_config['num_cpus']
    else:
        print('Parameter num_cpus should be defined in the configuration. Defaulting to 2.')
    n_particles = 10
    if 'n_particles' in backtest_config:
        n_particles = backtest_config['n_particles']
    phi1 = 1.4962
    phi2 = 1.4962
    omega = 0.7298
    if 'options' in backtest_config:
        phi1 = backtest_config['options']['c1']
        phi2 = backtest_config['options']['c2']
        omega = backtest_config['options']['w']
    current_best_params = []
    if current_best:
        if type(current_best) == list:
            for c in current_best:
                c = clean_start_config(c, config, backtest_config['ranges'])
                current_best_params.append(c)
        else:
            current_best = clean_start_config(current_best, config, backtest_config['ranges'])
            current_best_params.append(current_best)

    ray.init(num_cpus=num_cpus, logging_level=logging.FATAL, log_to_driver=False)
    pso = ng.optimizers.ConfiguredPSO(transform='identity', popsize=n_particles, omega=omega, phip=phi1, phig=phi2)
    algo = NevergradSearch(optimizer=pso, points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=num_cpus)
    scheduler = AsyncHyperBandScheduler()

    analysis = tune.run(tune.with_parameters(backtest, ticks=ticks), metric='objective', mode='max', name='search',
                        search_alg=algo, scheduler=scheduler, num_samples=iters, config=config, verbose=1,
                        reuse_actors=True, local_dir=session_dirpath,
                        progress_reporter=LogReporter(metric_columns=['daily_gain', 'closest_liquidation', 'objective'],
                                                      parameter_columns=[k for k in backtest_config['ranges']]))

    ray.shutdown()
    df = analysis.results_df
    df.reset_index(inplace=True)
    df.drop(columns=['trial_id', 'time_this_iter_s', 'done', 'timesteps_total', 'episodes_total', 'training_iteration',
                     'experiment_id', 'date', 'timestamp', 'time_total_s', 'pid', 'hostname', 'node_ip',
                     'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore', 'experiment_tag'],
            inplace=True)
    df.to_csv(os.path.join(backtest_config['session_dirpath'], 'results.csv'), index=False)
    print('Best candidate found:')
    pprint.pprint(analysis.best_config)
    plot_wrap(backtest_config, ticks, clean_result_config(analysis.best_config))
    return analysis


def plot_wrap(bc, ticks, candidate):
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    print('backtesting...')
    result, fdf = backtest_wrap(ticks, {**bc, **{'break_on': {}}, **candidate}, do_print=True)
    if fdf is None or len(fdf) == 0:
        print('no trades')
        return
    backtest_config = {**bc, **candidate, **result}
    backtest_config['session_dirpath'] = make_get_filepath(os.path.join(
        'plots', bc['exchange'], bc['symbol'],
        f"{n_days}_days_{ts_to_date(time())[:19].replace(':', '')}", ''))
    fdf.to_csv(backtest_config['session_dirpath'] + f"backtest_trades_{result['key']}.csv")
    df = pd.DataFrame({'price': ticks[:, 0], 'buyer_maker': ticks[:, 1], 'timestamp': ticks[:, 2]})
    dump_plots(backtest_config, fdf, df)


async def main(args: list):
    config_name = args[1]
    backtest_config = await prep_backtest_config(config_name)
    if backtest_config['exchange'] == 'bybit' and not backtest_config['inverse']:
        print('bybit usdt linear backtesting not supported')
        return
    downloader = Downloader(backtest_config)
    ticks = await downloader.get_ticks(True)
    backtest_config['n_days'] = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    if (p := '--plot') in args:
        try:
            candidate = json.load(open(args[args.index(p) + 1]))
            print('plotting given candidate')
        except Exception as e:
            print(os.listdir(backtest_config['session_dirpath']))
            try:
                candidate = json.load(open(backtest_config['session_dirpath'] + 'live_config.json'))
                print('plotting best candidate')
            except:
                return
        print(json.dumps(candidate, indent=4))
        plot_wrap(backtest_config, ticks, candidate)
        return
    start_candidate = None
    if (s := '--start') in args:
        try:
            if os.path.isdir(args[args.index(s) + 1]):
                start_candidate = [json.load(open(f)) for f in
                                   glob.glob(os.path.join(args[args.index(s) + 1], '*.json'))]
                print('Starting with all configurations in directory.')
            else:
                start_candidate = json.load(open(args[args.index(s) + 1]))
                print('Starting with specified configuration.')
        except:
            print('Could not find specified configuration.')
    if start_candidate:
        backtest_tune(ticks, backtest_config, start_candidate)
    else:
        backtest_tune(ticks, backtest_config)


if __name__ == '__main__':
    asyncio.run(main(sys.argv))

