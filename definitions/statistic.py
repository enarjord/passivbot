from typing import List

from numba import types, njit, typed
from numba.experimental import jitclass


@jitclass([
    ('timestamp', types.int64),
    ('balance', types.float64),
    ('equity', types.float64),
    ('profit_and_loss', types.float64),
    ('position_balance_ratio', types.float64)
])
class Statistic:
    """
    A class representing an order fill.
    """

    def __init__(self, timestamp: int, balance: float, equity: float, profit_and_loss_balance: float,
                 profit_and_loss_equity: float, position_balance_ratio: float):
        """
        Create a statistic.
        :param timestamp: The timestamp it was filled.
        :param balance: The current balance.
        :param equity: The current value of the equity.
        :param profit_and_loss_balance: The profit and loss of the balance if any.
        :param profit_and_loss_equity: The profit and loss of the equity if any.
        :param position_balance_ratio: The ratio of the position vs the wallet.
        """
        self.timestamp = timestamp
        self.balance = balance
        self.equity = equity
        self.profit_and_loss_balance = profit_and_loss_balance
        self.profit_and_loss_equity = profit_and_loss_equity
        self.position_balance_ratio = position_balance_ratio


@njit
def empty_statistic_list() -> List[Statistic]:
    """
    Returns an empty Fill typed list.
    :return: Empty Fill typed list.
    """
    l = typed.List()
    l.append(empty_statistic())
    l.clear()
    return l


@njit
def empty_statistic() -> Statistic:
    """
    Returns an empty Fill.
    :return: Empty Fill.
    """
    return Statistic(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)


def precompile_fill():
    """
    Precompile function for Statistic. Executes all methods and functions in script.
    :return:
    """
    s = empty_statistic()
    empty_statistic_list()
