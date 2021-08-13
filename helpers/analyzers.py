import numpy as np
import pandas as pd

from definitions.order import LONG, SHORT, BUY, SELL, TP, SL, LIMIT, MARKET, PARTIALLY_FILLED


def get_empty_analysis(config: dict) -> dict:
    """
    Returns an empty result dictionary.
    :param config: The config to use.
    :return: Empty result dictionary.
    """
    return {
        'exchange': config['exchange'] if 'exchange' in config else 'unknown',
        'symbol': config['symbol'] if 'symbol' in config else 'unknown',
        'starting_balance': config['starting_balance'],
        'final_balance': 0.0,
        'final_equity': 0.0,
        'net_pnl_plus_fees': 0.0,
        'gain': 1.0,
        'number_of_days': 0.0,
        'average_daily_gain': 0.0,
        'average_periodic_gain': 0.0,
        'median_daily_gain': 0.0,
        'adjusted_daily_gain': 0.0,
        'sharpe_ratio': 0.0,
        'profit_sum': 0.0,
        'loss_sum': 0.0,
        'fee_sum': 0.0,
        'lowest_equity_balance_ratio': 0.0,
        'closest_bankruptcy': 1.0,
        'number_of_fills': 0,
        'number_of_partial_fills': 0,
        'number_of_entries': 0,
        'number_of_closes': 0,
        'number_of_limit_orders': 0,
        'number_of_market_orders': 0,
        'number_of_take_profit': 0,
        'number_of_stop_loss': 0,
        'biggest_position_size': 0.0,
        'mean_hours_between_fills': 10000.0,
        'mean_hours_between_fills_long': 10000.0,
        'mean_hours_between_of_fills_short': 10000.0,
        'max_hours_no_fills_long': 10000.0,
        'max_hours_no_fills_short': 10000.0,
        'max_hours_no_fills_same_side': 10000.0,
        'max_hours_no_fills': 10000.0,
    }


def analyze_fills(fill_frame: pd.DataFrame, statistic_frame: pd.DataFrame, config: dict, first_timestamp: float,
                  last_timestamp: float) -> dict:
    """
    Calculates a result dictionary with several parameters.
    :param fill_frame: The frame of fills.
    :param statistic_frame: The frame of statistics.
    :param config: The config to use.
    :param first_timestamp: The first timestamp.
    :param last_timestamp: The last timestamp.
    #
    :return:
    """
    if fill_frame.empty:
        return get_empty_analysis(config)

    fill_frame = fill_frame.copy(deep=True)
    statistic_frame = statistic_frame.copy(deep=True)

    statistic_frame['timestamp'] = pd.to_datetime(statistic_frame['timestamp'] * 1000 * 1000)
    statistic_frame.set_index('timestamp', inplace=True)
    daily_statistics = statistic_frame.resample('1D').agg(
        {'balance': 'last', 'equity': 'last', 'profit_and_loss_balance': 'prod', 'profit_and_loss_equity': 'prod',
         'position_balance_ratio': 'max', 'equity_balance_ratio': 'min', 'bankruptcy_distance': 'min'})

    adgs = (fill_frame.equity / config['starting_balance']) ** (
            1 / ((fill_frame.timestamp - first_timestamp) / (1000 * 60 * 60 * 24)))
    fill_frame = fill_frame.join(adgs.rename('adg'))  # .set_index('timestamp')

    longs = fill_frame[fill_frame.position_side.str.contains(LONG)]
    shorts = fill_frame[fill_frame.position_side.str.contains(SHORT)]

    long_stuck_mean = 1000.0
    long_stuck = 1000.0
    short_stuck_mean = 1000.0
    short_stuck = 1000.0

    if len(longs) > 0:
        long_fill_ts_diffs = np.diff([first_timestamp] + list(longs.timestamp) + [last_timestamp]) / (1000 * 60 * 60)
        long_stuck_mean = np.mean(long_fill_ts_diffs)
        long_stuck = np.max(long_fill_ts_diffs)

    if len(shorts) > 0:
        short_fill_ts_diffs = np.diff([first_timestamp] + list(shorts.timestamp) + [last_timestamp]) / (1000 * 60 * 60)
        short_stuck_mean = np.mean(short_fill_ts_diffs)
        short_stuck = np.max(short_fill_ts_diffs)

    periodic_statistics = daily_statistics.resample(str(int(config['periodic_gain_n_days'])) + 'D').agg(
        {'balance': 'last', 'equity': 'last', 'profit_and_loss_balance': 'prod', 'profit_and_loss_equity': 'prod',
         'position_balance_ratio': 'max', 'equity_balance_ratio': 'min', 'bankruptcy_distance': 'min'})

    periodic_gains_mean = np.nan_to_num(periodic_statistics.profit_and_loss_balance.mean())
    periodic_gains_std = periodic_statistics.profit_and_loss_balance.std()
    sharpe_ratio = periodic_gains_mean / periodic_gains_std if periodic_gains_std != 0.0 else -20.0
    sharpe_ratio = np.nan_to_num(sharpe_ratio)

    result = {
        'exchange': config['exchange'] if 'exchange' in config else 'unknown',
        'symbol': config['symbol'] if 'symbol' in config else 'unknown',
        'starting_balance': config['starting_balance'],
        'final_balance': statistic_frame.iloc[-1].balance,
        'final_equity': statistic_frame.iloc[-1].equity,
        'net_pnl_plus_fees': fill_frame.profit_and_loss.sum() + fill_frame.fee_paid.sum(),
        'gain': statistic_frame.iloc[-1].equity / config['starting_balance'],
        'number_of_days': len(daily_statistics),
        'average_daily_gain': (adg := daily_statistics.profit_and_loss_equity.mean()),
        'average_periodic_gain': periodic_gains_mean,
        'median_daily_gain': daily_statistics.profit_and_loss_equity.median(),
        'adjusted_daily_gain': np.tanh(10 * (adg - 1)) + 1,
        'sharpe_ratio': sharpe_ratio,
        'profit_sum': fill_frame[fill_frame.profit_and_loss > 0.0].profit_and_loss.sum(),
        'loss_sum': fill_frame[fill_frame.profit_and_loss < 0.0].profit_and_loss.sum(),
        'fee_sum': fill_frame.fee_paid.sum(),
        'lowest_equity_balance_ratio': daily_statistics.equity_balance_ratio.min(),
        'closest_bankruptcy': daily_statistics.bankruptcy_distance.min(),
        'number_of_fills': len(fill_frame[fill_frame['action'] == PARTIALLY_FILLED]),
        'number_of_partial_fills': len(fill_frame[fill_frame['action'] == PARTIALLY_FILLED]),
        'number_of_entries': len(longs[longs['side'] == BUY]) + len(shorts[shorts['side'] == SELL]),
        'number_of_closes': len(longs[longs['side'] == SELL]) + len(shorts[shorts['side'] == BUY]),
        'number_of_limit_orders': len(fill_frame[fill_frame['order_type'] == LIMIT]),
        'number_of_market_orders': len(fill_frame[fill_frame['order_type'] == MARKET]),
        'number_of_take_profit': len(fill_frame[fill_frame['order_type'] == TP]),
        'number_of_stop_loss': len(fill_frame[fill_frame['order_type'] == SL]),
        'biggest_position_size': fill_frame.position_size.abs().max(),
        'mean_hours_between_fills': np.mean(
            np.diff([first_timestamp] + list(fill_frame.timestamp) + [last_timestamp])) / (1000 * 60 * 60),
        'mean_hours_between_fills_long': long_stuck_mean,
        'mean_hours_between_of_fills_short': short_stuck_mean,
        'max_hours_no_fills_long': long_stuck,
        'max_hours_no_fills_short': short_stuck,
        'max_hours_no_fills_same_side': max(long_stuck, short_stuck),
        'max_hours_no_fills': np.max(np.diff([first_timestamp] + list(fill_frame.timestamp) + [last_timestamp])) / (
                    1000 * 60 * 60),
    }
    return result
