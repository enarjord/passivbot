from numba import types, njit
from numba.experimental import jitclass

from definitions.order import LONG, SHORT


@jitclass([
    ('symbol', types.string),
    ('size', types.float64),
    ('price', types.float64),
    ('liquidation_price', types.float64),
    ('upnl', types.float64),
    ('leverage', types.float64),
    ('position_side', types.string)
])
class Position:
    """
    A class representing a position.
    """

    def __init__(self, symbol: str, size: float, price: float, liquidation_price: float, upnl: float, leverage: float,
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
        return self.symbol == position.symbol and self.size == position.size and self.price == position.price and \
               self.liquidation_price == position.liquidation_price and self.upnl == position.upnl and \
               self.leverage == position.leverage and self.position_side == position.position_side

    def empty(self):
        """
        Checks whether a position is empty. Ignores the symbol and position side, and checks for leverage = 1.0.
        :return: Whether the position is empty or not.
        """
        return self.size == 0.0 and self.price == 0.0 and self.liquidation_price == 0.0 and self.upnl == 0.0 and \
               self.leverage == 1.0

    # A copy method in the class makes the class unpickable and empty_long_position and empty_short_position will not
    # work anymore.
    # def copy(self):
    #     """
    #     Creates a new Position object with the current values.
    #     :return: New Position.
    #     """
    #     return Position(self.symbol, self.size, self.price, self.liquidation_price, self.upnl, self.leverage,
    #                     self.position_side)


@njit
def copy_position(position: Position) -> Position:
    """
    Creates a new instance of a Position with the same values as the Position that the function was called with.
    The position can not directly have a copy method because using the type in the method does not allow to call it
    indirectly.
    :param position: The position to copy.
    :return: The position order.
    """
    return Position(position.symbol, position.size, position.price, position.liquidation_price, position.upnl,
                    position.leverage, position.position_side)


@njit
def empty_long_position() -> Position:
    """
    Creates an empty long Position.
    :return: Empty long Position.
    """
    return Position('', 0.0, 0.0, 0.0, 0.0, 1.0, LONG)


@njit
def empty_short_position() -> Position:
    """
    Creates an empty short Position.
    :return: Empty short Position.
    """
    return Position('', 0.0, 0.0, 0.0, 0.0, 1.0, SHORT)


@njit
def precompile_position():
    """
    Precompile function for Position. Executes all methods and functions in script.
    :return:
    """
    p = empty_long_position()
    p = empty_short_position()
    p.equal(p)
    p.empty()
    copy_position(p)
