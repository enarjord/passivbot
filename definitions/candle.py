from typing import List

from numba import types, typed, njit
from numba.experimental import jitclass


@jitclass([
    ('timestamp', types.int64),
    ('open', types.float64),
    ('high', types.float64),
    ('low', types.float64),
    ('close', types.float64),
    ('volume', types.float64)
])
class Candle:
    """
    A class representing a candle.
    """

    def __init__(self, timestamp: int, open: float, high: float, low: float, close: float, volume: float):
        """
        Creates a candle.
        :param timestamp: The timestamp of the candle.
        :param open: The open price of the candle.
        :param high: The highest price of the candle.
        :param low: The lowest price of the candle.
        :param close: The close price of the candle.
        :param volume: The quantity of the candle.
        """
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def equal(self, candle):
        """
        Check for equality between two candles.
        :param candle: The candle to check against.
        :return: If equal or not.
        """
        return self.timestamp == candle.timestamp and self.open == candle.open and self.high == candle.high and \
               self.low == candle.low and self.close == candle.close and self.volume == candle.volume

    def empty(self):
        """
        Checks whether a candle is empty.
        :return: Whether the candle is empty or not.
        """
        return self.timestamp == 0 and self.open == 0.0 and self.high == 0.0 and self.low == 0.0 and \
               self.close == 0.0 and self.volume == 0.0

    # def copy(self):
    #     """
    #     Creates a new Candle object with the current values.
    #     :return: New Candle.
    #     """
    #     return Candle(self.timestamp, self.open, self.high, self.low, self.close, self.volume)


@njit
def copy_candle(candle: Candle) -> Candle:
    """
    Creates a new instance of a Candle with the same values as the Candle that the function was called with.
    The order can not directly have a copy method because using the type in the method does not allow to create a typed
    list.
    :param candle: The candle to copy.
    :return: The copied candle.
    """
    return Candle(candle.timestamp, candle.open, candle.high, candle.low, candle.close, candle.volume)


@njit
def empty_candle_list() -> List[Candle]:
    """
    Returns an empty Candle typed list.
    :return: Empty Candle typed list
    """
    l = typed.List()
    l.append(Candle(0, 0.0, 0.0, 0.0, 0.0, 0.0))
    l.clear()
    return l


@njit
def empty_candle() -> Candle:
    """
    Returns an empty Candle with all values set to 0.
    :return: Empty Candle.
    """
    c = Candle(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return c


@njit
def precompile_candle():
    """
    Precompile function for Candle. Executes all methods and functions in script.
    :return:
    """
    c = empty_candle()
    c.equal(c)
    c.empty()
    copy_candle(c)
    empty_candle_list()
