import datetime
import os
from typing import Union

import numpy as np
import pandas as pd

from definitions.candle import empty_candle, empty_candle_list
from definitions.tick import empty_tick_list
from helpers.optimized import convert_array_to_tick_list, prepare_candles, candles_to_array


def sort_dict_keys(d: Union[dict, list]) -> Union[dict, list]:
    """
    Sort dictionaries and keys.
    :param d: The object to sort.
    :return: A sorted list or dictionary.
    """
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def ts_to_date(timestamp: float) -> str:
    """
    Converts timestamp to human readable time.
    :param timestamp: The epoch timestamp.
    :return: A human readable time.
    """
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def get_filenames(filepath: str) -> list:
    """
    Get filenames of saved CSV files.
    :param filepath: Path to CSV files.
    :return: List of sorted file names.
    """
    return sorted([f for f in os.listdir(filepath) if f.endswith(".csv")],
                  key=lambda x: int(eval(x[:x.find("_")].replace(".cs", "").replace("v", ""))))


def create_test_data(filepath: str, tick_interval: float = 0.25):
    """
    Basic function to create test data for the backtester. Reads CSV files, creates ticks out of it, aggregates ticks
    into candles, and saved them in a numpy array. Can later be used in the downloader to create arrays for the
    backtester.
    :param filepath: Path to CSV files.
    :param tick_interval: Tick interval to use, defaults to 0.25 seconds.
    :return:
    """
    files = get_filenames(filepath)
    last_tick_update = 0
    tick_list = empty_tick_list()
    last_candle = empty_candle()
    candles = empty_candle_list()
    for f in files[:10]:
        df = pd.read_csv(os.path.join(filepath, f))
        if last_tick_update == 0:
            last_tick_update = int(df.iloc[0]['timestamp'] - (df.iloc[0]['timestamp'] % (tick_interval * 1000)))
        next_update = int(df.iloc[-1]['timestamp'] - (df.iloc[-1]['timestamp'] % (tick_interval * 1000))) + int(
            tick_interval * 1000)
        tick_list = convert_array_to_tick_list(tick_list, df[['timestamp', 'price', 'qty', 'is_buyer_maker']].values)
        c, tick_list, last_tick_update = prepare_candles(tick_list, last_tick_update, next_update, last_candle,
                                                         tick_interval)
        candles.extend(c)
    data = candles_to_array(candles)
    np.save('test_data.npy', data)
