from backtest import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pprint
from pprint import PrettyPrinter

async def main():

    plt.rcParams['figure.figsize'] = [29, 18]
    pd.set_option('precision', 10)
    pp = PrettyPrinter()

    # plotting method

    def plot_tdf_(df_, tdf_, side_: int = 0, liq_thr=0.1):
        plt.clf()
        df_.loc[tdf_.index[0]:tdf_.index[-1]][0].plot(style='y-')
        if side_ >= 0:
            longs = tdf_[tdf_.side == 'long']
            le = longs[longs.type == 'entry']
            lc = longs[longs.type == 'close']
            ls = longs[longs.type.str.startswith('stop_loss')]
            ls.price.plot(style='gx')
            le.price.plot(style='b.')
            longs.pos_price.plot(style='b--')
            if 'close_price' in longs.columns:
                longs.close_price.plot(style='r--')
            lc.price.plot(style='r.')
        if side_ <= 0:
            shrts = tdf_[tdf_.side == 'shrt']
            se = shrts[shrts.type == 'entry']
            sc = shrts[shrts.type == 'close']
            ss = shrts[shrts.type.str.startswith('stop_loss')]
            ss.price.plot(style='gx')
            se.price.plot(style='r.')
            shrts.pos_price.plot(style='r--')
            if 'close_price' in shrts.columns:
                shrts.close_price.plot(style='b--')
            sc.price.plot(style='b.')
        if 'liq_price' in tdf_.columns:
            tdf_.liq_price.where((tdf_.price - tdf_.liq_price).abs() / tdf_.price < liq_thr, np.nan).plot(style='k--')
        return plt

    backtest_config_name = 'ada'
    backtest_config = await prep_backtest_config(backtest_config_name)
    session_dirpath = backtest_config['session_dirpath']
    session_dirpath

    ticks = await load_ticks(backtest_config)
    df = pd.DataFrame(ticks)

    results = pd.DataFrame(load_results(session_dirpath + 'results.txt')).T.set_index('index').sort_values('fitness', ascending=False)
    print('n completed iterations', len(results))
    results.drop([k for k in backtest_config['ranges']], axis=1).head(40)

    key = results.key.iloc[0]
    #key = '306f245f93a9d3f264f4e24c81c77a7332308783d3ea397aa1f7a05822b07f31'
    print(key)

    result = results.loc[results.key == key].iloc[0]
    backtest_config.update(result)
    result.drop('key')

    tdf = pd.read_csv(f"{session_dirpath}backtest_trades/{key}_full.csv").set_index('trade_id')
    print('price with bid ask entry thresholds')
    ema = df[0].ewm(span=result['ema_span'], adjust=False).mean()
    bids_ = ema * (1 - result['ema_spread'])
    asks_ = ema * (1 + result['ema_spread'])

    plt.clf()
    df[0].iloc[::100].plot()
    bids_.iloc[::100].plot()
    asks_.iloc[::100].plot()
    plt.savefig(f'{session_dirpath}ema_spread_plot.png')

    plot_tdf_(df, tdf)

    tdf.head(60)

    # analyze results
    longs = tdf[tdf.side == 'long']
    shrts = tdf[tdf.side == 'shrt']
    le = longs[longs.type == 'entry']
    lc = longs[longs.type == 'close']
    se = shrts[shrts.type == 'entry']
    sc = shrts[shrts.type == 'close']

    def gain_conv(x):
        return x * 100 - 100

    biggest_pos_size = tdf.pos_size.abs().max()
    net_pnl = tdf.net_pnl_plus_fees.iloc[-1]
    loss_sum = tdf.loss_sum.iloc[-1]
    profit_sum = tdf.profit_sum.iloc[-1]
    fee_sum = tdf.fee_paid.sum()
    gain = (backtest_config['starting_balance'] + net_pnl) / backtest_config['starting_balance']
    closest_liq = tdf.closest_liq.min()
    n_stop_loss = len(tdf[tdf.type == 'stop_loss'])
    n_days = backtest_config['n_days']
    average_daily_gain = gain ** (1 / n_days) if gain > 0.0 else 0.0
    closes = tdf[tdf.type == 'close']
    lines = []
    lines.append(f'net pnl {net_pnl:.6f}')
    lines.append(f'profit sum {profit_sum:.6f}')
    lines.append(f'loss sum {loss_sum:.6f}')
    lines.append(f'fee sum {fee_sum:.6f}')
    lines.append(f'gain {gain * 100 - 100:.2f}%')
    lines.append(f'n_days {n_days}')
    lines.append(f'average daily gain percentage {(average_daily_gain - 1) * 100:.2f}%')
    lines.append(f'n trades {len(tdf)}')
    lines.append(f'n closes {len(closes)}')
    lines.append(f'n stop loss closes {n_stop_loss}')
    lines.append(f'biggest_pos_size {round(biggest_pos_size, 10)}')
    lines.append(f'closest liq {closest_liq * 100:.4f}%')
    lines.append(f"starting balance {backtest_config['starting_balance']}")
    lines.append(f"long: {backtest_config['do_long']}, short: {backtest_config['do_shrt']}")

    with open(f'{session_dirpath}backtest_result.txt', 'w') as f:
        for line in lines:
            print(line)
            f.write(line + '\n')

    # plots are saved in backtesting_results/{exchange}/{symbol}/{session_name}/
    n_parts = 7
    if len(tdf) < n_parts:
        print(f'Bot have made less than {n_parts} trades')
    else:
        for z in range(n_parts):
            start_ = z / n_parts
            end_ = (z + 1) / n_parts
            fig = plot_tdf_(df, tdf.iloc[int(len(tdf) * start_):int(len(tdf) * end_)], liq_thr=0.1)
            fig.savefig(f'{session_dirpath}backtest_{z + 1}of{n_parts}.png')
    fig = plot_tdf_(df, tdf, liq_thr=0.1)
    fig.savefig(f'{session_dirpath}whole_backtest.png')

    counter = 0
    idxs = []
    for row in tdf.itertuples():
        if row.type == 'stop_loss':
            counter += 1
        else:
            if counter > 0:
                idxs.append(row.Index)
            counter = 0
    plt.clf()
    tdf.net_pnl_plus_fees.plot()
    if idxs:
        tdf.net_pnl_plus_fees.loc[idxs].plot(style='ro')
    plt.savefig(f'{session_dirpath}pnlcumsum_plot.png')

    plt.clf()
    tdf.pos_size.plot()
    plt.savefig(f'{session_dirpath}pos_sizes_plot.png')

    dgr_ = tdf.average_daily_gain
    print('min max', dgr_.min(), dgr_.max())
    dgr_.index = tdf.progress
    plt.clf()
    dgr_.iloc[int(len(tdf) * 0.1):].plot()
    plt.savefig(f'{session_dirpath}average_daily_gain_plot.png')

    # visualize behavior
    # execute below cell repeatedly (up arrow, shift enter) to see backtest chunk by chunk
    # adjust step to set zoom level
    step = 120
    i = -step

    i += step
    tdfc = tdf.iloc[i:i+step]
    plot_tdf_(df, tdf.iloc[i:i+step], liq_thr=0.01)

    tdfc.head(60)#.timestamp.diff().values

    tdfc.tail(60)

    closest_liqs = tdf[['closest_long_liq', 'closest_shrt_liq']].min(axis=1).sort_values()
    closest_liqs.head()

    i = 0
    iloc_ = tdf.index.get_loc(closest_liqs.index[i])
    iminus = 10
    iplus = 10
    tdfc = tdf.iloc[max(0, iloc_-iminus):min(iloc_+iplus, len(tdf) - 1)]
    plot_tdf_(df, tdfc, liq_thr=0.1)

    tdfc.head(60)

    tdf[tdf.type.str.startswith('stop')]

    tdfzz = tdf[(tdf.type == 'close') | (tdf.type == 'entry')]
    tdfzz[tdfzz.type == tdfzz.type.shift(1)]

    tdf.timestamp.diff().sort_values()

    tdf.millis_since_prev_trade.sort_values().tail() / (1000 * 60 * 60)

if __name__ == '__main__':
    asyncio.run(main())
