from typing import List

import numpy as np
from numba import njit

from definitions.candle import Candle
from definitions.tick import Tick


@njit
def convert_array_to_tick_list(tick_list: List[Tick], data: np.ndarray) -> List[Tick]:
    """
    Converts an array into a tick list so that it can be further processed.
    :param tick_list: The tick list to use.
    :param data: The data to use in the form: timestamp, price, quantity, is_buyer_maker.
    :return: The tick list with added ticks.
    """
    for row in data:
        tick_list.append(Tick(int(row[0]), int(row[1]), float(row[2]), float(row[3]), bool(row[4])))
    return tick_list


@njit
def candles_to_array(candles: List[Candle]) -> np.ndarray:
    """
    Converts a list of candles into a numpy array.
    :param candles: The list of candles.
    :return: A numpy array int he form: timestamp, open, high, low, close, volume.
    """
    array = np.zeros((len(candles), 6))
    for i in range(len(candles)):
        array[i] = np.asarray([candles[i].timestamp, candles[i].open, candles[i].high, candles[i].low, candles[i].close,
                               candles[i].volume], dtype=np.float64)
    return array
