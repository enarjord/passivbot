from numba import types
from numba.experimental import jitclass

TP = 'TAKE_PROFIT'
SL = 'STOP_LOSS'
LIMIT = 'LIMIT'
MARKET = 'MARKET'
LQ = 'LIQUIDATION'

NEW = 'NEW'
PARTIALY_FILLED = 'PARTIALLY_FILLED'
FILLED = 'FILLED'
CANCELED = 'CANCELED'
EXPIRED = 'EXPIRED'
TRADE = 'TRADE'
CALCULATED = 'CALCULATED'
NEW_INSURANCE = 'NEW_INSURANCE'
NEW_ADL = 'NEW_ADL'

BUY = 'BUY'
SELL = 'SELL'

LONG = 'LONG'
SHORT = 'SHORT'
BOTH = 'BOTH'


@jitclass([
    ('symbol', types.string),
    ('order_id', types.int64),
    ('price', types.float64),
    ('stop_price', types.float64),
    ('qty', types.float64),
    ('type', types.string),
    ('side', types.string),
    ('timestamp', types.int64),
    ('action', types.string),
    ('position_side', types.string)
])
class Order:
    """
    A class representing an order.
    """

    def __init__(self, symbol: str, order_id: int, price: float, stop_price: float, qty: float, type: str, side: str,
                 timestamp: int, action: str, position_side: str):
        """
        Creates an order.
        :param symbol: The symbol of the order, for example BTCUSDT.
        :param order_id: The ID of the order. Is created by the exchange and can be set to 0 when creating an order.
        :param price: The price of the order either a limit or at which price it was executed.
        :param stop_price: The stop price in case a stop loss or take profit order is created.
        Can be set to 0 for normal orders.
        :param qty: The quantity or size of the order.
        :param type: The type of the order, for example LIMIT.
        :param side: The side of the order, meaning BUY or SELL.
        :param timestamp: The timestamp when the order was executed. Can be set to 0 when creating an order.
        :param action: The action of an order, for example FILLED. Can be set to an empty string when creating an order.
        :param position_side: The side of the order, LONG or SHORT.
        """
        self.symbol = symbol.upper()
        self.order_id = order_id
        self.price = price
        self.stop_price = stop_price
        self.qty = qty
        self.type = type.upper()
        self.side = side.upper()
        self.timestamp = timestamp
        self.action = action.upper()
        self.position_side = position_side.upper()

    def equal(self, order):
        """
        Check for equality between two orders.
        :param order: The order to check against.
        :return: If equal or not.
        """
        if self.symbol == order.symbol and self.order_id == order.order_id and self.price == order.price \
                and self.stop_price == order.stop_price and self.qty == order.qty and self.type == order.type \
                and self.side == order.side and self.timestamp == order.timestamp and self.action == order.action \
                and self.position_side == order.position_side:
            return True
        else:
            return False
