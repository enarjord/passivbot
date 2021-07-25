from numba import types, njit, typed
from numba.experimental import jitclass


@jitclass([
    ('timestamp', types.int64),
    ('price', types.float64),
    ('qty', types.float64),
    ('is_buyer_maker', types.boolean)
])
class Tick:
    """
    A class representing a price tick.
    """

    def __init__(self, timestamp: int, price: float, qty: float, is_buyer_maker: bool):
        """
        Create a tick.
        :param timestamp: The timestamp of the tick.
        :param price: The price of the tick.
        :param qty: The quantity of the tick.
        :param is_buyer_maker: Whether it is a maker or taker tick.
        """
        self.timestamp = timestamp
        self.price = price
        self.qty = qty
        self.is_buyer_maker = is_buyer_maker


@njit
def empty_tick_list():
    l = typed.List()
    l.append(Tick(0, 0.0, 0.0, False))
    l.clear()
    return l
