from numba import types
from numba.experimental import jitclass


@jitclass([
    ('symbol', types.string),
    ('user', types.string),
    ('exchange', types.string),
    ('market_type', types.string),
    ('leverage', types.int64),
    ('call_interval', types.float64),
    ('historic_tick_range', types.float64),
    ('historic_fill_range', types.float64),
    ('tick_interval', types.float64)
])
class LiveConfig:
    """
    A class representing a live config.
    """

    def __init__(self, symbol: str, user: str, exchange: str, market_type: str, leverage: int, call_interval: float,
                 historic_tick_range: float, historic_fill_range: float, tick_interval: float):
        """
        Creates a live config.
        :param symbol: The symbol to use.
        :param user: The user for the API keys.
        :param exchange: The exchange to use.
        :param market_type: The leverage to use.
        :param leverage: The leverage to use.
        :param call_interval: Call interval for strategy to use in live.
        :param historic_tick_range: Range for which to fetch historic ticks in seconds. 0 if nothing to fetch.
        :param historic_tick_range: Range for which to fetch historic fills in seconds. 0 if nothing to fetch.
        :param tick_interval: Interval for which to aggregate ticks, length of candle.
        """
        self.symbol = symbol
        self.user = user
        self.exchange = exchange
        self.market_type = market_type
        self.leverage = leverage
        self.call_interval = call_interval
        self.historic_tick_range = historic_tick_range
        self.historic_fill_range = historic_fill_range
        self.tick_interval = tick_interval


@jitclass([
    ('quantity_step', types.float64),
    ('price_step', types.float64),
    ("minimal_quantity", types.float64),
    ("minimal_cost", types.float64),
    ('call_interval', types.float64),
    ('historic_tick_range', types.float64),
    ('historic_fill_range', types.float64),
    ('tick_interval', types.float64),
    ('statistic_interval', types.int64),
    ('leverage', types.float64),
    ('symbol', types.string),
    ('maker_fee', types.float64),
    ('taker_fee', types.float64),
    ('latency', types.float64),
    ('market_type', types.string),
    ('inverse', types.boolean),
    ('contract_multiplier', types.float64)
])
class BacktestConfig:
    """
    A class representing a backtest config.
    """

    def __init__(self, quantity_step: float, price_step: float, minimal_quantity: float, minimal_cost: float,
                 call_interval: float, historic_tick_range: float, historic_fill_range: float, tick_interval: float,
                 statistic_interval: int, leverage: float, symbol: str, maker_fee: float, taker_fee: float,
                 latency: float, market_type: str, inverse: bool, contract_multiplier: float):
        """
        Creates a backtest config.
        :param quantity_step: Quantity step to use in backtesting.
        :param price_step: Price step to use in backtesting.
        :param minimal_quantity: Minimal quantity to use in backtesting.
        :param minimal_cost: Minimal costto use in backtesting.
        :param call_interval: Call interval for strategy to use in backtesting.
        :param historic_tick_range: Range for which to collect historic ticks in seconds before execution. 0 if nothing
        to fetch.
        :param historic_tick_range: Range for which to collect historic fills in seconds before execution. 0 if nothing
        to fetch.
        :param tick_interval: Interval for which to aggregate ticks, length of candle.
        :param statistic_interval: Interval at which to collect statistics.
        :param leverage: Leverage to use in backtesting.
        :param symbol: The symbol to test.
        :param maker_fee: The maker fee to use.
        :param taker_fee: The taker fee to use.
        :param latency: The latency to use.
        :param market_type: The market type to use.
        :param inverse: Whether it's an inverse market or not.
        :param contract_multiplier: The contract multiplier to use.
        """
        self.quantity_step = quantity_step
        self.price_step = price_step
        self.minimal_quantity = minimal_quantity
        self.minimal_cost = minimal_cost
        self.call_interval = call_interval
        self.historic_tick_range = historic_tick_range
        self.historic_fill_range = historic_fill_range
        self.tick_interval = tick_interval
        self.statistic_interval = statistic_interval
        self.leverage = leverage
        self.symbol = symbol
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency = latency
        self.market_type = market_type
        self.inverse = inverse
        self.contract_multiplier = contract_multiplier
