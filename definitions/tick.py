from typing import List

from numba import types, njit, typed
from numba.experimental import jitclass


@jitclass([
    ('trade_id', types.int64),
    ('timestamp', types.int64),
    ('price', types.float64),
    ('quantity', types.float64),
    ('is_buyer_maker', types.boolean)
])
class Tick:
    """
    A class representing a price tick.
    """

    def __init__(self, trade_id: int, timestamp: int, price: float, quantity: float, is_buyer_maker: bool):
        """
        Create a tick.
        :param trade_id: The trade ID of the tick.
        :param timestamp: The timestamp of the tick.
        :param price: The price of the tick.
        :param quantity: The quantity of the tick.
        :param is_buyer_maker: Whether it is a maker or taker tick.
        """
        self.trade_id = trade_id
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity
        self.is_buyer_maker = is_buyer_maker


@njit
def empty_tick_list() -> List[Tick]:
    l = typed.List()
    l.append(empty_tick())
    l.clear()
    return l


@njit
def empty_tick() -> Tick:
    """
    Returns an empty Tick with all values set to 0 except is_buyer_maker to False.
    :return: Empty Tick.
    """
    return Tick(0, 0, 0.0, 0.0, False)


@njit
def precompile_tick():
    """
    Precompile function for Tick. Executes all methods and functions in script.
    :return:
    """
    t = empty_tick()
    empty_tick_list()
