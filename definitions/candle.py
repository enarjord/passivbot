from numba import types
from numba.experimental import jitclass


@jitclass([
    ('open', types.float64),
    ('high', types.float64),
    ('low', types.float64),
    ('close', types.float64),
    ('qty', types.float64)
])
class Candle:
    """
    A class representing a candle.
    """

    def __init__(self, open: float, high: float, low: float, close: float, qty: float):
        """
        Creates a candle.
        :param open: The open price of the candle.
        :param high: The highest price of the candle.
        :param low: The lowest price of the candle.
        :param close: The close price of the candle.
        :param qty: The quantity of the candle.
        """
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.qty = qty
