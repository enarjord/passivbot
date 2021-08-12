import datetime
import os
from typing import Union

import numpy as np
import pandas as pd

from definitions.candle import empty_candle, empty_candle_list
from definitions.tick import empty_tick_list
from helpers.converters import convert_array_to_tick_list, candles_to_array
from helpers.optimized import prepare_candles
from helpers.print_functions import print_


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


def make_get_filepath(filepath: str) -> str:
    """
    If the filepath is not a path, it creates the directory and sub directories and returns the path.
    :param filepath: The file path to use.
    :return: The actual path.
    """
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    os.makedirs(dirpath, exist_ok=True)
    return filepath


def get_utc_now_timestamp() -> int:
    """
    Creates a millisecond based timestamp of UTC now.
    :return: Millisecond based timestamp of UTC now.
    """
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)


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


def check_dict(d: dict) -> dict:
    """
    Checks a optimization configuration dictionary or sub dictionary and calls appropriate function.
    :param d: Dictionary to check.
    :return: A dictionary with tune types in the same structure as the original dictionary.
    """
    new_d = {}
    for key, value in d.items():
        if type(value) == OrderedDict or type(value) == dict:
            new_d[key] = check_dict(value)
        elif type(value) == list:
            new_d[key] = check_list(value)
        else:
            print_(["Something wrong in checking dictionary"])
    return new_d


def check_list(l: list) -> Union[float, int, list, Float, Integer]:
    """
    Checks a optimization configuration list or sub list and calls appropriate function or creates variable.
    :param l: List to check.
    :return: A list, integer, float, tune float, or tune integer.
    """
    new_l = []
    if type(l[0]) == float:
        if l[0] == l[1]:
            return l[0]
        else:
            return uniform(l[0], l[1])
    elif type(l[0]) == int:
        if l[0] == l[1]:
            return l[0]
        else:
            return randint(l[0], l[1] + 1)
    else:
        for item in l:
            if type(item) == list:
                new_l.append(check_list(item))
            elif type(item) == OrderedDict or type(item) == dict:
                new_l.append(check_dict(item))
            else:
                print_(["Something wrong in checking list"])
    return new_l
