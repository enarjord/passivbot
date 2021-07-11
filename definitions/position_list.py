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
        self.long = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        self.short = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')

    def update_long(self, symbol: str, size: float, price: float, liquidation_price: float, upnl: float, leverage: int,
                    position_side: str):
        self.long.symbol = symbol
        self.long.size = size
        self.long.price = price
        self.long.liquidation_price = liquidation_price
        self.long.upnl = upnl
        self.long.leverage = leverage
        self.long.position_side = position_side

    def update_short(self, symbol: str, size: float, price: float, liquidation_price: float, upnl: float, leverage: int,
                     position_side: str):
        self.short.symbol = symbol
        self.short.size = size
        self.short.price = price
        self.short.liquidation_price = liquidation_price
        self.short.upnl = upnl
        self.short.leverage = leverage
        self.short.position_side = position_side
