import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import init, Fore
from prettytable import PrettyTable

from definitions.order import BUY, SELL, TP, SL, LIMIT, MARKET, PARTIALLY_FILLED, FILLED
from definitions.order import LONG, SHORT
from helpers.optimized import round_dynamic, round_up


def plot_figure(x_y_label_line_marker_color: List[tuple], save_path: str, x_label: str = None, y_label: str = None,
                title: str = None):
    """
    Plots a figure based on x and y values and optional arguments, such as linestyle, marker, and color. Saves the
    figure to the specified path and adds labels and title if specified.
    :param x_y_label_line_marker_color: Tuple specifying plotting parameters. Needs to define an x and y array/list.
    Linestyle, marker style, and color can be anything or None.
    :param save_path: The path to save the figure to.
    :param x_label: The optional x label.
    :param y_label: The optional y label.
    :param title: The optional title.
    :return:
    """
    sns.set("paper", "white", rc={"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 10, "axes.titlesize": 10,
                                  "xtick.labelsize": 10, "ytick.labelsize": 10})
    pd.set_option('precision', 10)
    fig = plt.figure(figsize=(29, 18))
    ax = fig.add_subplot(111)
    handles = []

    for x, y, label, line, marker, color in x_y_label_line_marker_color:
        ln = ax.plot(x, y, label=label, linestyle=line, marker=marker, color=color)
        handles += ln

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    labels = [l.get_label() for l in handles]
    ax.legend(handles, labels, loc=0, frameon=True, framealpha=1.0, facecolor="white", edgecolor="white")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def dump_plots(result: dict, fill_frame: pd.DataFrame, statistic_frame: pd.DataFrame, candle_frame: pd.DataFrame):
    """
    Prints the results of the run and plots several figures.
    :param result: The configuration including the result.
    :param fill_frame: The frame of fills.
    :param statistic_frame: The frame of statistics.
    :param candle_frame: The frame of candles.
    :return:
    """
    init(autoreset=True)
    pd.set_option('precision', 10)

    statistic_frame['timestamp'] = pd.to_datetime(statistic_frame['timestamp'] * 1000 * 1000)
    fill_frame['timestamp'] = pd.to_datetime(fill_frame['timestamp'] * 1000 * 1000)
    candle_frame['timestamp'] = pd.to_datetime(candle_frame['timestamp'] * 1000 * 1000)

    statistic_frame.set_index('timestamp', inplace=True)
    fill_frame.set_index('timestamp', inplace=True)
    candle_frame.set_index('timestamp', inplace=True)

    daily_statistics = statistic_frame.resample('1D').agg(
        {'balance': 'last', 'equity': 'last', 'profit_and_loss_balance': 'prod', 'profit_and_loss_equity': 'prod',
         'position_balance_ratio': 'max', 'equity_balance_ratio': 'min', 'bankruptcy_distance': 'min'})

    table = PrettyTable(["Metric", "Value"])
    table.align['Metric'] = 'l'
    table.align['Value'] = 'l'
    table.title = 'Summary'

    table.add_row(['Exchange', result['exchange'] if 'exchange' in result else 'unknown'])
    table.add_row(['Market type', result['market_type'] if 'market_type' in result else 'unknown'])
    table.add_row(['Symbol', result['symbol'] if 'symbol' in result else 'unknown'])
    table.add_row(['No. days', round_dynamic(result['result']['number_of_days'], 6)])
    table.add_row(['Starting balance', round_dynamic(result['result']['starting_balance'], 6)])
    profit_color = Fore.RED if result['result']['final_balance'] < result['result']['starting_balance'] else Fore.RESET
    table.add_row(['Final balance', f"{profit_color}{round_dynamic(result['result']['final_balance'], 6)}{Fore.RESET}"])
    table.add_row(['Final equity', f"{profit_color}{round_dynamic(result['result']['final_equity'], 6)}{Fore.RESET}"])
    table.add_row(
        ['Net PNL + fees', f"{profit_color}{round_dynamic(result['result']['net_pnl_plus_fees'], 6)}{Fore.RESET}"])
    table.add_row(['Total gain percentage',
                   f"{profit_color}{round_dynamic(result['result']['gain'] * 100 - 100, 4)}%{Fore.RESET}"])
    table.add_row(['Average daily gain percentage',
                   f"{profit_color}{round_dynamic((result['result']['average_daily_gain'] - 1) * 100, 3)}%{Fore.RESET}"])
    table.add_row(['Adjusted daily gain',
                   f"{profit_color}{round_dynamic(result['result']['adjusted_daily_gain'], 6)}{Fore.RESET}"])
    # table.add_row([f"{result['avg_periodic_gain_key']} percentage",
    #                f"{profit_color}{round_dynamic(result['result']['average_periodic_gain'] * 100, 3)}%{Fore.RESET}"])
    table.add_row(['Median daily gain percentage',
                   f"{profit_color}{round_dynamic((result['result']['median_daily_gain'] - 1) * 100, 3)}%{Fore.RESET}"])
    bankruptcy_color = Fore.RED if result['result']['closest_bankruptcy'] < 0.4 else Fore.YELLOW if result['result'][
                                                                                                        'closest_bankruptcy'] < 0.8 else Fore.RESET
    table.add_row(['Closest bankruptcy percentage',
                   f"{bankruptcy_color}{round_dynamic(result['result']['closest_bankruptcy'] * 100, 4)}%{Fore.RESET}"])
    table.add_row([' ', ' '])
    table.add_row(['Sharpe ratio', round_dynamic(result['result']['sharpe_ratio'], 6)])
    table.add_row(['Profit sum', f"{profit_color}{round_dynamic(result['result']['profit_sum'], 6)}{Fore.RESET}"])
    table.add_row(['Loss sum', f"{Fore.RED}{round_dynamic(result['result']['loss_sum'], 6)}{Fore.RESET}"])
    table.add_row(['Fee sum', round_dynamic(result['result']['fee_sum'], 6)])
    table.add_row(['Lowest equity/balance ratio', round_dynamic(result['result']['lowest_equity_balance_ratio'], 6)])
    table.add_row(['Biggest position size', round_dynamic(result['result']['biggest_position_size'], 6)])
    table.add_row([' ', ' '])
    table.add_row(['No. fills', round_dynamic(result['result']['number_of_fills'], 6)])
    table.add_row(['No. entries', round_dynamic(result['result']['number_of_entries'], 6)])
    table.add_row(['No. closes', round_dynamic(result['result']['number_of_closes'], 6)])
    table.add_row(['No. limit orders', round_dynamic(result['result']['number_of_limit_orders'], 6)])
    table.add_row(['No. market orders', round_dynamic(result['result']['number_of_market_orders'], 6)])
    table.add_row(['No. take profit', round_dynamic(result['result']['number_of_take_profit'], 6)])
    table.add_row(['No. stop loss', round_dynamic(result['result']['number_of_stop_loss'], 6)])
    table.add_row([' ', ' '])
    table.add_row(['Mean hours between fills', round_dynamic(result['result']['mean_hours_between_fills'], 6)])
    table.add_row(
        ['Max hours no fills (same side)', round_dynamic(result['result']['max_hours_no_fills_same_side'], 6)])
    table.add_row(['Max hours no fills', round_dynamic(result['result']['max_hours_no_fills'], 6)])

    longs = fill_frame[fill_frame.position_side.str.contains(LONG)]
    shorts = fill_frame[fill_frame.position_side.str.contains(SHORT)]

    if len(longs) > 0:
        table.add_row([' ', ' '])
        table.add_row(['Long', ' '])
        table.add_row(['No. fills', round_dynamic(len(longs[longs['action'] == FILLED]), 6)])
        table.add_row(['No. partial fills', round_dynamic(len(longs[longs['action'] == PARTIALLY_FILLED]), 6)])
        table.add_row(['No. entries', round_dynamic(len(longs[longs['side'] == BUY]), 6)])
        table.add_row(['No. closes', round_dynamic(len(longs[longs['side'] == SELL]), 6)])
        table.add_row(['No. limit orders', round_dynamic(len(longs[longs['order_type'] == LIMIT]), 6)])
        table.add_row(['No. market orders', round_dynamic(len(longs[longs['order_type'] == MARKET]), 6)])
        table.add_row(['No. take profit', round_dynamic(len(longs[longs['order_type'] == TP]), 6)])
        table.add_row(['No. stop loss', round_dynamic(len(longs[longs['order_type'] == SL]), 6)])
        table.add_row(
            ['Mean hours between fills (long)', round_dynamic(result['result']['mean_hours_between_fills_long'], 6)])
        table.add_row(['Max hours no fills (long)', round_dynamic(result['result']['max_hours_no_fills_long'], 6)])
        profit_color = Fore.RED if longs.profit_and_loss.sum() < 0 else Fore.RESET
        table.add_row(["PNL sum", f"{profit_color}{longs.profit_and_loss.sum()}{Fore.RESET}"])

    if len(shorts) > 0:
        table.add_row([' ', ' '])
        table.add_row(['Short', ' '])
        table.add_row(['No. fills', round_dynamic(len(shorts[shorts['action'] == FILLED]), 6)])
        table.add_row(['No. partial fills', round_dynamic(len(shorts[shorts['action'] == PARTIALLY_FILLED]), 6)])
        table.add_row(['No. entries', round_dynamic(len(shorts[shorts['side'] == BUY]), 6)])
        table.add_row(['No. closes', round_dynamic(len(shorts[shorts['side'] == SELL]), 6)])
        table.add_row(['No. limit orders', round_dynamic(len(shorts[shorts['order_type'] == LIMIT]), 6)])
        table.add_row(['No. market orders', round_dynamic(len(shorts[shorts['order_type'] == MARKET]), 6)])
        table.add_row(['No. take profit', round_dynamic(len(shorts[shorts['order_type'] == TP]), 6)])
        table.add_row(['No. stop loss', round_dynamic(len(shorts[shorts['order_type'] == SL]), 6)])
        table.add_row(['Mean hours between fills (short)',
                       round_dynamic(result['result']['mean_hours_between_of_fills_short'], 6)])
        table.add_row(['Max hours no fills (short)', round_dynamic(result['result']['max_hours_no_fills_short'], 6)])
        profit_color = Fore.RED if shorts.profit_and_loss.sum() < 0 else Fore.RESET
        table.add_row(["PNL sum", f"{profit_color}{shorts.profit_and_loss.sum()}{Fore.RESET}"])

    # dump_live_config(result, result['plots_dirpath'] + 'live_config.json')
    # json.dump(denumpyize(result), open(result['plots_dirpath'] + 'result.json', 'w'), indent=4)

    print('Writing backtest_result.txt...\n')
    with open(f"{result['plots_dirpath']}backtest_result.txt", 'w') as f:
        output = table.get_string(border=True, padding_width=1)
        print(output)
        f.write(re.sub('\033\\[([0-9]+)(;[0-9]+)*m', '', output))

    print('\nPlotting balance and equity...')

    plot_figure([(statistic_frame.index, statistic_frame.balance, 'Balance', None, None, None),
                 (statistic_frame.index, statistic_frame.equity, 'Equity', None, None, None)],
                f"{result['plots_dirpath']}balance_and_equity.png", 'Time', 'Value', 'Balance and equity')

    longs['cum_sum'] = longs.profit_and_loss.cumsum()
    shorts['cum_sum'] = shorts.profit_and_loss.cumsum()

    plot_figure([(longs.index, longs.cum_sum, 'Long', None, None, None),
                 (shorts.index, shorts.cum_sum, 'Short', None, None, None)],
                f"{result['plots_dirpath']}pnl_cum_sum_long_short.png", 'Time', 'Value',
                'Cumulative sum long and short')

    plot_figure([(longs.index, longs.cum_sum, 'Long', None, None, None)],
                f"{result['plots_dirpath']}pnl_cum_sum_long.png", 'Time', 'Value', 'Cumulative sum long')

    plot_figure([(shorts.index, shorts.cum_sum, 'Short', None, None, None)],
                f"{result['plots_dirpath']}pnl_cum_sum_short.png", 'Time', 'Value', 'Cumulative sum short')

    plot_figure([(statistic_frame.index, statistic_frame.profit_and_loss_balance, 'PNL balance', None, None, None),
                 (statistic_frame.index, statistic_frame.profit_and_loss_equity, 'PNL equity', None, None, None)],
                f"{result['plots_dirpath']}pnl_balance_and_equity.png", 'Time', 'Value', 'PNL balance and equity')

    plot_figure([(daily_statistics.index, daily_statistics.profit_and_loss_balance, 'PNL balance', None, None, None),
                 (daily_statistics.index, daily_statistics.profit_and_loss_equity, 'PNL equity', None, None, None)],
                f"{result['plots_dirpath']}adg.png", 'Time', 'Value', 'Daily PNL balance and equity')

    print('Plotting backtest whole and in chunks...')
    number_of_parts = max(3, int(round_up(result['number_of_days'] / 14, 1.0)))
    for part in range(number_of_parts):
        start_ = part / number_of_parts
        end_ = (part + 1) / number_of_parts
        print(f'{part} of {number_of_parts} {start_ * 100:.2f}% to {end_ * 100:.2f}%')
        plot_fills(candle_frame, fill_frame.iloc[int(len(fill_frame) * start_):int(len(fill_frame) * end_)],
                   f"{result['plots_dirpath']}backtest_{part + 1}of{number_of_parts}.png", bankruptcy_threshold=0.1)
    plot_fills(candle_frame, fill_frame, f"{result['plots_dirpath']}whole_backtest.png", bankruptcy_threshold=0.1)

    print('Plotting pos sizes...')
    plot_figure([(longs.index, longs.position_size, 'Long', None, None, None),
                 (shorts.index, shorts.position_size, 'Short', None, None, None)],
                f"{result['plots_dirpath']}psizes_plot.png", 'Time', 'Value', 'Position size long and short')


def plot_fills(candle_frame, fill_frame, save_path: str, side: int = 0, bankruptcy_threshold=0.1):
    """
    Plots type specific fills.
    :param candle_frame: The frame of candles.
    :param fill_frame: The frame of fills.
    :param save_path: The path to save the figure to.
    :param side: The side specifying what should be printed. 0 prints both, larger 0 prints long, smaller 0 prints short.
    :param bankruptcy_threshold: The bankruptcy threshold to print.
    :return:
    """
    if fill_frame.empty:
        print('No fills...')
        return
    tmp_candle_frame = candle_frame[
        (candle_frame.index >= fill_frame.index[0]) & (candle_frame.index <= fill_frame.index[-1])]

    plots = [(tmp_candle_frame.index, tmp_candle_frame.close, 'Close price', None, None, 'black')]

    if side >= 0:
        longs = fill_frame[fill_frame.position_side.str.contains(LONG)]
        long_market_entry = longs[(longs.order_type == MARKET) & longs.side == BUY]
        long_market_close = longs[(longs.order_type == MARKET) & longs.side == SELL]
        long_limit_entry = longs[(longs.order_type == LIMIT) & longs.side == BUY]
        long_limit_close = longs[(longs.order_type == LIMIT) & longs.side == SELL]
        long_take_profit = longs[longs.order_type == TP]
        long_stop_loss = longs[longs.order_type == SL]
        long_position_price = longs.where(longs.position_price != 0.0).position_price.fillna(method='ffill')

        plots.append((long_market_entry.index, long_market_entry.price, 'Long market entry', 'None', '.', 'b'))
        plots.append((long_market_close.index, long_market_close.price, 'Long market close', 'None', '.', 'y'))
        plots.append((long_limit_entry.index, long_limit_entry.price, 'Long limit entry', 'None', '.', 'g'))
        plots.append((long_limit_close.index, long_limit_close.price, 'Long limit close', 'None', '.', 'r'))
        plots.append((long_take_profit.index, long_take_profit.price, 'Long take profit', 'None', '.', 'r'))
        plots.append((long_stop_loss.index, long_stop_loss.price, 'Long stop loss', 'None', 'x', 'r'))
        plots.append((long_position_price.index, long_position_price, 'Long position price', '--', None, 'g'))
    if side <= 0:
        shorts = fill_frame[fill_frame.position_side.str.contains(SHORT)]
        short_market_entry = shorts[(shorts.order_type == MARKET) & shorts.side == BUY]
        short_market_close = shorts[(shorts.order_type == MARKET) & shorts.side == SELL]
        short_limit_entry = shorts[(shorts.order_type == LIMIT) & shorts.side == BUY]
        short_limit_close = shorts[(shorts.order_type == LIMIT) & shorts.side == SELL]
        short_take_profit = shorts[shorts.order_type == TP]
        short_stop_loss = shorts[shorts.order_type == SL]
        shorts_position_price = shorts.where(shorts.position_price != 0.0).position_price.fillna(method='ffill')

        plots.append((short_market_entry.index, short_market_entry.price, 'Short market entry', 'None', '.', 'y'))
        plots.append((short_market_close.index, short_market_close.price, 'Short market close', 'None', '.', 'b'))
        plots.append((short_limit_entry.index, short_limit_entry.price, 'Short limit entry', 'None', '.', 'r'))
        plots.append((short_limit_close.index, short_limit_close.price, 'Short limit close', 'None', '.', 'g'))
        plots.append((short_take_profit.index, short_take_profit.price, 'Short take profit', 'None', '.', 'g'))
        plots.append((short_stop_loss.index, short_stop_loss.price, 'Short stop loss', 'None', 'x', 'g'))
        plots.append((shorts_position_price.index, shorts_position_price, 'Short position price', '--', None, 'r'))
    plot_figure(plots, save_path, 'Time', 'Value', 'Position entries and exits')
    # if 'bkr_price' in fdf.columns:
    #     fdf.bkr_price.where(fdf.bkr_diff < bkr_thr, np.nan).plot(style='k--')
    # return plt
