from numba import typeof, njit
from numba.experimental import jitclass

from definitions.position import Position, empty_long_position, empty_short_position, copy_position


@jitclass([
    ("long", typeof(empty_long_position())),
    ("short", typeof(empty_short_position()))
])
class PositionList:
    """
    A class representing long and short positions.
    """

    def __init__(self):
        """
        Creates two empty positions, one for SHORT and one for LONG.
        """
        self.long = empty_long_position()
        self.short = empty_short_position()

    def update_long(self, position: Position):
        """
        Updates the values of the long position.
        :param position: New position.
        :return:
        """
        self.long.symbol = position.symbol
        self.long.size = position.size
        self.long.price = position.price
        self.long.liquidation_price = position.liquidation_price
        self.long.upnl = position.upnl
        self.long.leverage = position.leverage
        self.long.position_side = position.position_side

    def update_short(self, position: Position):
        """
        Updates the values of the short position.
        :param position: New position.
        :return:
        """
        self.short.symbol = position.symbol
        self.short.size = position.size
        self.short.price = position.price
        self.short.liquidation_price = position.liquidation_price
        self.short.upnl = position.upnl
        self.short.leverage = position.leverage
        self.short.position_side = position.position_side

    def copy(self):
        """
        Creates a new PositionList object with the current values. Does a deep copy of all positions.
        :return: New PositionList.
        """
        p = PositionList()
        p.update_long(copy_position(self.long))
        p.update_short(copy_position(self.short))
        return p


@njit
def precompile_position_list():
    """
    Precompile function for OrderList. Executes all methods and functions in script.
    :return:
    """
    p = PositionList()
    l = empty_long_position()
    p.update_long(l)
    l = empty_short_position()
    p.update_short(l)
    p.copy()
