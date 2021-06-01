import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pure_funcs import round_dynamic, denumpyize, candidate_to_live_config
from njit_funcs import calc_bid_ask_thresholds
from procedures import dump_live_config


def dump_plots(result: dict, fdf: pd.DataFrame, df: pd.DataFrame):
    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)

    def gain_conv(x):
        return x * 100 - 100

    lines = []
    lines.append(f"gain percentage {round_dynamic(result['result']['gain'] * 100 - 100, 4)}%")
    lines.append(f"average_daily_gain percentage {round_dynamic((result['result']['average_daily_gain'] - 1) * 100, 3)}%")
    lines.append(f"closest_bkr percentage {round_dynamic(result['result']['closest_bkr'] * 100, 4)}%")
    lines.append(f"starting balance {round_dynamic(result['starting_balance'], 3)}")

    for key in [k for k in result['result'] if k not in ['gain', 'average_daily_gain', 'closest_bkr', 'do_long', 'do_shrt']]:
        lines.append(f"{key} {round_dynamic(result['result'][key], 6)}")
    lines.append(f"long: {result['do_long']}, short: {result['do_shrt']}")

    dump_live_config(result, result['plots_dirpath'] + 'live_config.json')
    json.dump(denumpyize(result), open(result['plots_dirpath'] + 'result.json', 'w'), indent=4)

    df = df.join(add_bid_ask_thresholds_from_config(df.price.values, result))

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
        fig = plot_fills(df, fdf.iloc[int(len(fdf) * start_):int(len(fdf) * end_)], bkr_thr=0.1)
        fig.savefig(f"{result['plots_dirpath']}backtest_{z + 1}of{n_parts}.png")
    fig = plot_fills(df, fdf, bkr_thr=0.1)
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


def add_bid_ask_thresholds_from_config(prices, config):
    iprc_MAr_coeffs = np.array([config['long']['iprc_MAr_coeffs'], config['shrt']['iprc_MAr_coeffs']])
    iprc_const = config['long']['iprc_const'], config['shrt']['iprc_const']
    MA_idx = np.array([config['long']['MA_idx'], config['shrt']['MA_idx']])
    bids, asks = calc_bid_ask_thresholds(prices, config['spans'], iprc_const, iprc_MAr_coeffs, MA_idx)
    return pd.DataFrame({'bid_thr': bids, 'ask_thr': asks})


def plot_fills(df, fdf, side: int = 0, bkr_thr=0.1):
    plt.clf()

    dfc = df.loc[fdf.index[0]:fdf.index[-1]]
    dfc.price.plot(style='y-')
    if 'bid_thr' in dfc.columns:
        dfc.bid_thr.plot(style='b-.')
    if 'ask_thr' in dfc.columns:
        dfc.ask_thr.plot(style='r-.')

    if side >= 0:
        longs = fdf[fdf.pside == 'long']
        lnentry = longs[(longs.type == 'long_ientry') | (longs.type == 'long_rentry')]
        lhentry = longs[longs.type == 'long_hentry']
        lnclose = longs[longs.type == 'long_nclose']
        lsclose = longs[longs.type == 'long_sclose']
        lnentry.price.plot(style='b.')
        lhentry.price.plot(style='bx')
        lnclose.price.plot(style='r.')
        lsclose.price.plot(style=('rx'))
        longs.long_pprice.fillna(method='ffill').plot(style='b--')
    if side <= 0:
        shrts = fdf[fdf.pside == 'shrt']
        snentry = shrts[(shrts.type == 'shrt_ientry') | (shrts.type == 'shrt_rentry')]
        shentry = shrts[shrts.type == 'shrt_hentry']
        snclose = shrts[shrts.type == 'shrt_nclose']
        ssclose = shrts[shrts.type == 'shrt_sclose']
        snentry.price.plot(style='r.')
        shentry.price.plot(style='rx')
        snclose.price.plot(style='b.')
        ssclose.price.plot(style=('bx'))
        shrts.shrt_pprice.fillna(method='ffill').plot(style='r--')

    if 'bkr_price' in fdf.columns:
        fdf.bkr_price.where(fdf.bkr_diff < bkr_thr, np.nan).plot(style='k--')
    return plt

