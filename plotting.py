import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from jitted import round_dynamic
from analyze import candidate_to_live_config


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

