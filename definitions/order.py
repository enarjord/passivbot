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


@jitclass([
    ('symbol', types.string),
    ('order_id', types.int64),
    ('price', types.float64),
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

    def __init__(self, symbol: str, order_id: int, price: float, qty: float, type: str, side: str, timestamp: int,
                 action: str, position_side: str):
        self.symbol = symbol.upper()
        self.order_id = order_id
        self.price = price
        self.qty = qty
        self.type = type.upper()
        self.side = side.upper()
        self.timestamp = timestamp
        self.action = action.upper()
        self.position_side = position_side.upper()
