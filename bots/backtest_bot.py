from typing import List

import numpy as np
from numba import typeof, types
from numba.experimental import jitclass

from bots.base_bot import Bot, base_bot_spec
from definitions.candle import Candle, empty_candle_list
from definitions.order import Order
from definitions.position import Position


@jitclass([
    ('qty_step', types.float64),
    ('price_step', types.float64),
    ('call_interval', types.float64),
    ('leverage', types.float64)
])
class BacktestConfig:
    """
    A class representing a backtest config.
    """

    def __init__(self, qty_step: float, price_step: float, call_interval: float, leverage: float):
        """
        Creates a backtest config.
        :param qty_step: Quantity step to use in backtesting.
        :param price_step: Price step to use in backtesting.
        :param call_interval: Call interval for strategy to use in backtesting.
        :param leverage: Leverage to use in backtesting.
        """
        self.qty_step = qty_step
        self.price_step = price_step
        self.call_interval = call_interval
        self.leverage = leverage


@jitclass(base_bot_spec +
          [
              ("config", typeof(BacktestConfig(0.0, 0.0, 1.0, 1.0))),
              ("config", typeof(to_be_replaced_strategy)),
          ])
class BacktestBot(Bot):
    __init_Base = Bot.__init__

    def __init__(self, config, strategy, data: np.ndarray):
        self.__init_Base()
        self.config = config
        self.strategy = strategy
        self.data = data
        self.qty_step = config.qty_step
        self.price_step = config.price_step
        self.call_interval = config.call_interval

    def update_account_and_orders(self, price: Candle):
        self.handle_account_update(0.0, Position('', 0.0, 0.0, 0.0, 0.0, 0, ''),
                                   Position('', 0.0, 0.0, 0.0, 0.0, 0, ''))
        self.handle_order_update(Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', ''))

    def prepare_candle(self, row, last_candle):
        candle = Candle(row[1], row[2], row[3], row[4], row[5])
        candle.qty = candle.qty - last_candle.qty
        return candle

    def start_websocket(self) -> None:
        price_list = empty_candle_list()
        last_candle = Candle(0.0, 0.0, 0.0, 0.0, 0.0)
        last_update = self.data[0, 0]
        # Time, open, high, low, close, qty
        for i in self.data:
            candle = self.prepare_candle(i, last_candle)
            self.update_account_and_orders(candle)
            price_list.append(candle)
            last_candle = candle
            current = i[0]
            if current - last_update >= self.strategy.call_interval * 1000:
                last_update = current
                self.decide(price_list)
                price_list = empty_candle_list()

    def create_orders(self, orders_to_create: List[Order]):
        raise NotImplementedError

    def cancel_orders(self, orders_to_cancel: List[Order]):
        raise NotImplementedError
