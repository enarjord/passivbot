from typing import List

import numpy as np
import pandas as pd
from numba import njit

from definitions.candle import Candle
from definitions.fill import Fill
from definitions.statistic import Statistic
from definitions.tick import Tick


@njit
def convert_array_to_tick_list(tick_list: List[Tick], data: np.ndarray) -> List[Tick]:
    """
    Converts an array into a Tick list so that it can be further processed.
    :param tick_list: The Tick list to use.
    :param data: The data to use in the form: timestamp, price, quantity, is_buyer_maker.
    :return: The tick list with added ticks.
    """
    for row in data:
        tick_list.append(Tick(int(row[0]), int(row[1]), float(row[2]), float(row[3]), bool(row[4])))
    return tick_list


@njit
def candles_to_array(candles: List[Candle]) -> np.ndarray:
    """
    Converts a list of Candles into a numpy array.
    :param candles: The list of Candles.
    :return: A numpy array int he form: timestamp, open, high, low, close, volume.
    """
    array = np.zeros((len(candles), 6))
    for i in range(len(candles)):
        array[i] = np.asarray([candles[i].timestamp, candles[i].open, candles[i].high, candles[i].low, candles[i].close,
                               candles[i].volume], dtype=np.float64)
    return array


def fills_to_frame(fills: List[Fill]) -> pd.DataFrame:
    """
    Converts a list of Fills to a pandas dataframe.
    :param fills: The list of Fills to convert.
    :return: The converted dataframe.
    """
    fill_list = []
    for fill in fills:
        fill_list.append({'order_id': fill.order_id,
                          'timestamp': fill.timestamp,
                          'profit_and_loss': fill.profit_and_loss,
                          'fee_paid': fill.fee_paid,
                          'balance': fill.balance,
                          'equity': fill.equity,
                          'position_balance_ratio': fill.position_balance_ratio,
                          'quantity': fill.quantity,
                          'price': fill.price,
                          'position_size': fill.position_size,
                          'position_price': fill.position_price,
                          'order_type': fill.order_type,
                          'action': fill.action,
                          'side': fill.side,
                          'position_side': fill.position_side})
    fill_frame = pd.DataFrame(fill_list)
    return fill_frame


def statistics_to_frame(statistics: List[Statistic]) -> pd.DataFrame:
    """
    Converts a list of Statistics to a pandas dataframe.
    :param statistics: The list of Statistics to convert.
    :return: The converted dataframe.
    """
    statistic_list = []
    for statistic in statistics:
        statistic_list.append({'timestamp': statistic.timestamp,
                               'balance': statistic.balance,
                               'equity': statistic.equity,
                               'profit_and_loss_balance': statistic.profit_and_loss_balance,
                               'profit_and_loss_equity': statistic.profit_and_loss_equity,
                               'position_balance_ratio': statistic.position_balance_ratio})
    statistic_frame = pd.DataFrame(statistic_list)
    return statistic_frame
