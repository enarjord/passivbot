import asyncio
import json
import os
import sys
from time import time
from analyze import analyze_fills

import numpy as np
import pandas as pd
import argparse
from plotting import dump_plots

from downloader import Downloader, prep_config, load_live_config
from pure_funcs import create_xk, calc_bankruptcy_price, iter_orders, calc_long_pnl, calc_shrt_pnl, \
    iter_MA_ratios_chunks, calc_spans, calc_available_margin, calc_diff, qty_to_cost, round_, \
    calc_new_psize_pprice
from passivbot import make_get_filepath, ts_to_date, get_keys, add_argparse_args


def backtest(config: dict, ticks: np.ndarray, do_print=False) -> (list, list, bool):
    if len(ticks) <= config['max_span']:
        return [], [], False
    long_psize, long_pprice = 0.0, 0.0
    shrt_psize, shrt_pprice = 0.0, 0.0
    bkr_price, bkr_diff = 0.0, 1.0
    balance = config['starting_balance']

    long_pprice, long_psize, shrt_pprice, shrt_psize = 0.0, 0.0, 0.0, 0.0

    xk = create_xk(config)

    latency_simulation_ms = config['latency_simulation_ms'] \
        if 'latency_simulation_ms' in config else 1000

    next_stats_update = 0
    stats = []

    def stats_update():
        upnl_l = x if (x := calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'],
                                          xk['c_mult'])) == x else 0.0
        upnl_s = y if (y := calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'],
                                          xk['c_mult'])) == y else 0.0
        stats.append({'timestamp': tick[2],
                      'balance': balance,  # Redundant with fills, but makes plotting easier
                      'equity': balance + upnl_l + upnl_s})

    all_fills = []
    fills = []
    bids, asks = [], []
    ob = [min(ticks[0][0], ticks[1][0]), max(ticks[0][0], ticks[1][0])]
    min_span = int(round(config['min_span']))
    max_span = int(round(config['max_span']))
    spans = calc_spans(min_span, max_span, config['n_spans'])

    chunk_iterator = iter_MA_ratios_chunks(ticks[:, 0], spans)
    emas_chunk, ratios_chunk, z = next(chunk_iterator)

    zc = 0

    closest_bkr = 1.0

    prev_update_plus_delay = ticks[max_span][2] + latency_simulation_ms
    update_triggered = False
    prev_update_plus_5sec = 0

    tick = ticks[0]
    stats_update()

    # tick tuple: (price, buyer_maker, timestamp)
    for k, tick in enumerate(ticks[max_span:], start=max_span):

        chunk_i = k - zc
        if chunk_i >= len(emas_chunk):
            emas_chunk, ratios_chunk, z = next(chunk_iterator)
            zc = z * len(emas_chunk)

            chunk_i = k - zc

        # Update the stats every 1/2 hour
        if tick[2] > next_stats_update:
            closest_bkr = min(closest_bkr, calc_diff(bkr_price, tick[0]))
            stats_update()
            next_stats_update = tick[2] + 1000 * 60 * 30

        fills = []
        if tick[1]:
            if bkr_diff < 0.1 and long_psize > -shrt_psize and tick[0] <= bkr_price:
                fills.append({'qty': -long_psize, 'price': tick[0], 'pside': 'long',
                              'type': 'long_liquidation', 'side': 'sel',
                              'pnl': calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'],
                                                   xk['c_mult']),
                              'fee_paid': -qty_to_cost(long_psize, tick[0], xk['inverse'],
                                                       xk['c_mult']) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': 0.0, 'shrt_psize': 0.0,
                              'shrt_pprice': 0.0, 'bkr_price': 0.0, 'bkr_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
            else:
                if bids:
                    if tick[0] <= bids[0][1]:
                        update_triggered = True
                    while bids:
                        if tick[0] < bids[0][1]:
                            bid = bids.pop(0)
                            fill = {'qty': bid[0], 'price': bid[1], 'side': 'buy', 'type': bid[4],
                                    'fee_paid': -qty_to_cost(bid[0], bid[1], xk['inverse'],
                                                             xk['c_mult']) * config['maker_fee']}
                            if 'close' in bid[4]:
                                fill['pnl'] = calc_shrt_pnl(shrt_pprice, bid[1], bid[0],
                                                            xk['inverse'], xk['c_mult'])
                                shrt_psize = min(0.0, round_(shrt_psize + bid[0], config['qty_step']))
                                fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                            else:
                                fill['pnl'] = 0.0
                                long_psize, long_pprice = calc_new_psize_pprice(long_psize,
                                                                                long_pprice, bid[0],
                                                                                bid[1],
                                                                                xk['qty_step'])
                                if long_psize < 0.0:
                                    long_psize, long_pprice = 0.0, 0.0
                                fill.update({'pside': 'long', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                            fills.append(fill)
                        else:
                            break
            ob[0] = tick[0]
        else:
            if bkr_diff < 0.05 and -shrt_psize > long_psize and tick[0] >= bkr_price:
                fills.append({'qty': -shrt_psize, 'price': tick[0], 'pside': 'shrt',
                              'type': 'shrt_liquidation', 'side': 'buy',
                              'pnl': calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'],
                                                   xk['c_mult']),
                              'fee_paid': -qty_to_cost(shrt_psize, tick[0], xk['inverse'],
                                                       xk['c_mult']) * config['taker_fee'],
                              'long_psize': 0.0, 'long_pprice': 0.0, 'shrt_psize': 0.0,
                              'shrt_pprice': 0.0, 'bkr_price': 0.0, 'bkr_diff': 1.0})
                long_psize, long_pprice, shrt_psize, shrt_pprice = 0.0, 0.0, 0.0, 0.0
            else:
                if asks:
                    if tick[0] >= asks[0][1]:
                        update_triggered = True
                    while asks:
                        if tick[0] > asks[0][1]:
                            ask = asks.pop(0)
                            fill = {'qty': ask[0], 'price': ask[1], 'side': 'sel', 'type': ask[4],
                                    'fee_paid': -qty_to_cost(ask[0], ask[1], xk['inverse'],
                                                             xk['c_mult']) * config['maker_fee']}
                            if 'close' in ask[4]:
                                fill['pnl'] = calc_long_pnl(long_pprice, ask[1], ask[0],
                                                            xk['inverse'], xk['c_mult'])
                                long_psize = max(0.0, round_(long_psize + ask[0], config['qty_step']))
                                fill.update({'pside': 'long', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_long_close_ts = tick[2]
                            else:
                                fill['pnl'] = 0.0
                                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize,
                                                                                shrt_pprice, ask[0],
                                                                                ask[1], xk['qty_step'])
                                if shrt_psize > 0.0:
                                    shrt_psize, shrt_pprice = 0.0, 0.0
                                fill.update({'pside': 'shrt', 'long_psize': long_psize,
                                             'long_pprice': long_pprice, 'shrt_psize': shrt_psize,
                                             'shrt_pprice': shrt_pprice})
                                prev_shrt_entry_ts = tick[2]
                            bkr_diff = calc_diff(bkr_price, tick[0])
                            fill.update({'bkr_price': bkr_price, 'bkr_diff': bkr_diff})
                            fills.append(fill)
                        else:
                            break
            ob[1] = tick[0]

        if tick[2] > prev_update_plus_delay and (update_triggered or tick[2] > prev_update_plus_5sec):
            prev_update_plus_delay = tick[2] + latency_simulation_ms
            prev_update_plus_5sec = tick[2] + 5000
            update_triggered = False
            bids, asks = [], []
            bkr_diff = calc_diff(bkr_price, tick[0])
            closest_bkr = min(closest_bkr, bkr_diff)
            MA = (emas_chunk[k - zc][config['long']['MA_idx']], emas_chunk[k - zc][config['shrt']['MA_idx']])
            MA_ratios = ratios_chunk[k - zc]
            for tpl in iter_orders(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                   ob[0], ob[1], MA, tick[0], MA_ratios, **xk):
                if (len(bids) > 2 and len(asks) > 2) or len(bids) > 5 or len(asks) > 5:
                    break
                if tpl[0] > 0.0:
                    bids.append(tpl)
                elif tpl[0] < 0.0:
                    asks.append(tpl)
                else:
                    break
            bids = sorted(bids, key=lambda x: x[1], reverse=True)
            asks = sorted(asks, key=lambda x: x[1])

        if len(fills) > 0:
            for fill in fills:
                balance += fill['pnl'] + fill['fee_paid']
                upnl_l = calc_long_pnl(long_pprice, tick[0], long_psize, xk['inverse'], xk['c_mult'])
                upnl_s = calc_shrt_pnl(shrt_pprice, tick[0], shrt_psize, xk['inverse'], xk['c_mult'])

                bkr_price = calc_bankruptcy_price(balance, long_psize, long_pprice,
                                                  shrt_psize, shrt_pprice, xk['inverse'], xk['c_mult'])
                bkr_diff = calc_diff(bkr_price, tick[0])
                fill.update({'bkr_price': bkr_price, 'bkr_diff': bkr_diff})

                fill['equity'] = balance + upnl_l + upnl_s
                fill['available_margin'] = calc_available_margin(
                    balance, long_psize, long_pprice, shrt_psize, shrt_pprice, tick[0],
                    xk['inverse'], xk['c_mult'], xk['leverage']
                )
                for side_ in ['long', 'shrt']:
                    if fill[f'{side_}_pprice'] == 0.0:
                        fill[f'{side_}_pprice'] = np.nan
                fill['balance'] = balance
                fill['timestamp'] = tick[2]
                fill['trade_id'] = k
                fill['gain'] = fill['equity'] / config['starting_balance']
                fill['n_days'] = (tick[2] - ticks[max_span][2]) / (1000 * 60 * 60 * 24)
                fill['closest_bkr'] = closest_bkr
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
                line += f"closest_bkr {closest_bkr:.4f} "
                print(line, end=' ')

    tick = ticks[-1]
    stats_update()
    return all_fills, stats, True


def plot_wrap(bc, ticks, live_config):
    n_days = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    print('n_days', round_(n_days, 0.1))
    config = {**bc, **live_config}
    print('starting_balance', config['starting_balance'])
    print('backtesting...')
    fills, stats, did_finish = backtest(config, ticks, do_print=True)
    if not fills:
        print('no fills')
        return
    fdf, result = analyze_fills(fills, config, ticks[-1][2])
    config['result'] = result
    config['plots_dirpath'] = make_get_filepath(os.path.join(
        config['plots_dirpath'], f"{ts_to_date(time())[:19].replace(':', '')}", '')
    )
    fdf.to_csv(config['plots_dirpath'] + "fills.csv")
    df = pd.DataFrame({'price': ticks[:, 0], 'buyer_maker': ticks[:, 1], 'timestamp': ticks[:, 2]})
    print('dumping plots...')
    dump_plots(config, fdf, df)


async def main():

    parser = argparse.ArgumentParser(prog='Backtest', description='Backtest given passivbot config.')
    parser.add_argument('live_config_path', type=str, help='path to live config to test')
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    config = await prep_config(args)
    print()
    for k in (keys := ['exchange', 'symbol', 'starting_balance', 'start_date', 'end_date',
                       'latency_simulation_ms', 'do_long', 'do_shrt']):
        if k in config:
            print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
    print()
    if config['exchange'] == 'bybit' and not config['inverse']:
        print('bybit usdt linear backtesting not supported')
        return
    downloader = Downloader(config)
    ticks = await downloader.get_ticks(True)
    config['n_days'] = round_((ticks[-1][2] - ticks[0][2]) / (1000 * 60 * 60 * 24), 0.1)
    live_config = load_live_config(args.live_config_path)
    print(json.dumps(live_config, indent=4))
    plot_wrap(config, ticks, live_config)


if __name__ == '__main__':
    asyncio.run(main())

