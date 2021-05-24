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
from jitted import calc_diff, round_, iter_entries, iter_long_closes, iter_shrt_closes, calc_available_margin, \
    calc_liq_price_binance, calc_liq_price_bybit, calc_new_psize_pprice, calc_long_pnl, calc_shrt_pnl, calc_cost, \
    iter_indicator_chunks, round_dynamic
from passivbot import make_get_filepath, ts_to_date, get_keys, add_argparse_args


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
                                shrt_psize = min(0.0, round_(shrt_psize + bid[0], config['qty_step']))
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
                                if long_psize < 0.0:
                                    long_psize, long_pprice = 0.0, 0.0
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
                                long_psize = max(0.0, round_(long_psize + ask[0], config['qty_step']))
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
                                if shrt_psize > 0.0:
                                    shrt_psize, shrt_pprice = 0.0, 0.0
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

