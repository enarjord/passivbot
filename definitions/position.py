from numba import types
from numba.experimental import jitclass


@jitclass([
    ('symbol', types.string),
    ('size', types.float64),
    ('price', types.float64),
    ('liquidation_price', types.float64),
    ('upnl', types.float64),
    ('leverage', types.int64),
    ('position_side', types.string)
])
class Position:
    """
    A class representing a position.
    """

    def __init__(self, symbol: str, size: float, price: float, liquidation_price: float, upnl: float, leverage: int,
                 position_side: str):
        self.symbol = symbol.upper()
        self.size = size
        self.price = price
        self.liquidation_price = liquidation_price
        self.upnl = upnl
        self.leverage = leverage
        self.position_side = position_side.upper()
