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
        """
        Creates a position.
        :param symbol: The symbol of the position, for example BTCUSDT.
        :param size: The quantity or size of the position.
        :param price: The average price of the position.
        :param liquidation_price: The price where the position is liquidated.
        Not necessarily accurate as account updates might not send this information.
        :param upnl: The unrealized profit or loss of the position.
        Not necessarily accurate as account updates might not send this information.
        :param leverage: The leverage that is used for this position.
        :param position_side: The side of the position, LONG or SHORT.
        """
        self.symbol = symbol.upper()
        self.size = size
        self.price = price
        self.liquidation_price = liquidation_price
        self.upnl = upnl
        self.leverage = leverage
        self.position_side = position_side.upper()

    def equal(self, position):
        """
        Check for equality between two positions.
        :param position: The position to check against.
        :return: If equal or not.
        """
        if self.symbol == position.symbol and self.size == position.size and self.price == position.price \
                and self.liquidation_price == position.liquidation_price and self.upnl == position.upnl \
                and self.leverage == position.leverage and self.position_side == position.position_side:
            return True
        else:
            return False
