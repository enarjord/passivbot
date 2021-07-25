from numba import types, njit, typed
from numba.experimental import jitclass


@jitclass([
    ('order_id', types.int64),
    ('timestamp', types.int64),
    ('profit_and_loss', types.float64),
    ('fee_paid', types.float64),
    ('balance', types.float64),
    ('equity', types.float64),
    ('position_balance_ratio', types.float64),
    ('qty', types.float64),
    ('price', types.float64),
    ('position_size', types.float64),
    ('position_price', types.float64),
    ('type', types.string)
])
class Fill:
    """
    A class representing an order fill.
    """

    def __init__(self, order_id: int, timestamp: int, profit_and_loss: float, fee_paid: float, balance: float,
                 equity: float,
                 position_balance_ratio: float, qty: float, price: float, position_size: float, position_price: float,
                 type: str):
        """
        Create a fill.
        :param order_id: The id of the trade it was filled.
        :param timestamp: The timestamp it was filled.
        :param profit_and_loss: The profit and loss if any.
        :param fee_paid: The fee for the fill.
        :param balance: The current balance.
        :param equity: The current value of the equity.
        :param position_balance_ratio: The ratio of the position vs the wallet.
        :param qty: The quantity that was filled.
        :param price: The price at which it was filled.
        :param position_size: The current position size.
        :param position_price: The current position price.
        :param type: The type of fill that happened. Will use types from Order.
        """
        self.order_id = order_id
        self.timestamp = timestamp
        self.profit_and_loss = profit_and_loss
        self.fee_paid = fee_paid
        self.balance = balance
        self.equity = equity
        self.position_balance_ratio = position_balance_ratio
        self.qty = qty
        self.price = price
        self.position_size = position_size
        self.position_price = position_price
        self.type = type


@njit
def empty_fill_list():
    l = typed.List()
    l.append(Fill(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ''))
    l.clear()
    return l
