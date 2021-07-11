from numba import typeof
from numba.experimental import jitclass

from definitions.position import Position


@jitclass([
    ("long", typeof(Position('', 0.0, 0.0, 0.0, 0.0, 0, ''))),
    ("short", typeof(Position('', 0.0, 0.0, 0.0, 0.0, 0, '')))
])
class PositionList:
    """
    A class representing long and short positions.
    """

    def __init__(self):
        """
        Creates two empty positions, one for SHORT and one for LONG.
        """
        self.long = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        self.short = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')

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
        Creates a new object with the current positions.
        :return: New positions.
        """
        p = PositionList()
        p.update_long(self.long)
        p.update_short(self.short)
        return p
