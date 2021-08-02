from typing import Tuple, List

import numpy as np
from numba import njit

from definitions.candle import Candle, empty_candle_list
from definitions.order import Order
from definitions.tick import Tick, empty_tick_list


@njit
def isclose(x: float, y: float, atol: float = 1.e-60, rtol: float = 1.e-60):
    """
    Test closeness of two floating point numbers. Simplified version of numpy.isclose that only works for single numbers
    and doesn't consider that a number could be NaN.
    :param x: First number to compare.
    :param y: Second number to compare.
    :param atol: Absolute tolerance parameter.
    :param rtol: Relative tolerance parameter.
    :return: Whether or not the numbers are close.
    """
    return abs(x - y) <= atol + rtol * abs(y)


@njit
def round_up_or_down(value: float, step: float):
    """
    Rounds a value up or down to the corresponding step depending on distance to the rounded up or down step.
    :param value: The value to round.
    :param step: The step to use.
    :return: The rounded value.
    """
    if abs(value - round_down(value, step)) < abs(value - round_up(value, step)):
        return round_down(value, step)
    else:
        return round_up(value, step)


@njit
def correct_order_float_precision(order: Order, price_step: float, quantity_step: float) -> Order:
    """
    Correct the precision of the order price and quantity if the rounding went wrong.
    :param order: The order to correct.
    :param price_step: The price step to use.
    :param quantity_step: The quantity step to use.
    :return: The corrected order.
    """
    if not isclose(order.price, round_down(order.price, price_step)) or not isclose(order.price,
                                                                                    round_up(order.price, price_step)):
        order.price = round_up_or_down(order.price, price_step)
    if not isclose(order.stop_price, round_down(order.stop_price, price_step)) or not isclose(order.stop_price,
                                                                                              round_up(order.stop_price,
                                                                                                       price_step)):
        order.stop_price = round_up_or_down(order.stop_price, price_step)
    if not isclose(order.quantity, round_down(order.quantity, quantity_step)) or not isclose(order.quantity,
                                                                                             round_up(order.quantity,
                                                                                                      quantity_step)):
        order.quantity = round_up_or_down(order.quantity, quantity_step)
    return order


@njit
def round_down(n, step, safety_rounding=10) -> float:
    """
    Round a float to the closest lower step size.
    :param n: Float to round.
    :param step: Step size to round to.
    :param safety_rounding: Precision rounding safety.
    :return: Rounded float.
    """
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_up(n, step, safety_rounding=10) -> float:
    """
    Round a float to the closest upper step size.
    :param n: Float to round.
    :param step: Step size to round to.
    :param safety_rounding: Precision rounding safety.
    :return: Rounded float.
    """
    return np.round(np.ceil(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_(n, step, safety_rounding=10) -> float:
    """
    Round a float.
    :param n: Float to round.
    :param step: Step size to round to.
    :param safety_rounding: Precision rounding safety.
    :return: Rounded float.
    """
    return np.round(np.round(n / step) * step, safety_rounding)


@njit
def quantity_to_cost(quantity, price, inverse, c_mult) -> float:
    """
    Calculates the cost of a position with given quantity and price.
    :param qty: Given quantity.
    :param price: Given price.
    :param inverse: Inverse contract.
    :param c_mult: Contract multiplier
    :return: The cost of the position.
    """
    return (abs(quantity / price) if price > 0.0 else 0.0) * c_mult if inverse else abs(quantity * price)


@njit
def nan_to_0(x) -> float:
    """
    Converts value to 0 if it is NaN.
    :param x: Value to convert.
    :return: 0 or value.
    """
    return x if x == x else 0.0


@njit
def calculate_long_pnl(entry_price: float, close_price: float, quantity: float, inverse: float,
                       contract_multiplier: float) -> float:
    """
    Calculates the profit and loss of a long position change.
    :param entry_price: The entry price of the position.
    :param close_price: The close price of the position.
    :param quantity: The quantity of the position.
    :param inverse: Whether it is an inverse contract or not.
    :param contract_multiplier: The multiplier of the contract.
    :return: The profit or loss of the position change.
    """
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(quantity) * contract_multiplier * (1.0 / entry_price - 1.0 / close_price)
    else:
        return abs(quantity) * (close_price - entry_price)


@njit
def calculate_short_pnl(entry_price: float, close_price: float, quantity: float, inverse: float,
                        contract_multiplier: float) -> float:
    """
    Calculates the profit and loss of a short position change.
    :param entry_price: The entry price of the position.
    :param close_price: The close price of the position.
    :param quantity: The quantity of the position.
    :param inverse: Whether it is an inverse contract or not.
    :param contract_multiplier: The multiplier of the contract.
    :return: The profit or loss of the position change.
    """
    if inverse:
        if entry_price == 0.0 or close_price == 0.0:
            return 0.0
        return abs(quantity) * contract_multiplier * (1.0 / close_price - 1.0 / entry_price)
    else:
        return abs(quantity) * (entry_price - close_price)


@njit
def calculate_new_position_size_position_price(position_size: float, position_price: float, quantity: float,
                                               price: float, quantity_step: float) -> (float, float):
    """
    Calculates the new size and price of a position based on the old price and size and the new price and quantity.
    :param position_size: Old position size.
    :param position_price: Old position price.
    :param quantity: Added/Removed quantity.
    :param price: Price of added/removed quantity.
    :param quantity_step: Quantity step of the symbol.
    :return: New position size and price.
    """
    if quantity == 0.0:
        return position_size, position_price
    new_position_size = round_(position_size + quantity, quantity_step)
    if new_position_size == 0.0:
        return 0.0, 0.0
    return new_position_size, nan_to_0(position_price) * (position_size / new_position_size) + price * (
            quantity / new_position_size)


@njit
def calculate_available_margin(balance: float, long_position_size: float, long_position_price: float,
                               shrt_position_size: float, shrt_position_price: float, last_price: float, inverse: bool,
                               contract_multiplier: float, leverage: float) -> float:
    """
    Calculates the currently available margin.
    :param balance: The current balance.
    :param long_position_size: The current long position size.
    :param long_position_price: The current long position price.
    :param shrt_position_size: The current short position size.
    :param shrt_position_price: The current short position price.
    :param last_price: The last price of the symbol.
    :param inverse: Whether it is an inverse contract or not.
    :param contract_multiplier: The multiplier of the contract.
    :param leverage: The leverage used.
    :return: The current available margin.
    """
    used_margin = 0.0
    equity = balance
    if long_position_price and long_position_size:
        equity += calculate_long_pnl(long_position_price, last_price, long_position_size, inverse, contract_multiplier)
        used_margin += quantity_to_cost(long_position_size, long_position_price, inverse, contract_multiplier)
    if shrt_position_price and shrt_position_size:
        equity += calculate_short_pnl(shrt_position_price, last_price, shrt_position_size, inverse, contract_multiplier)
        used_margin += quantity_to_cost(shrt_position_size, shrt_position_price, inverse, contract_multiplier)
    return max(0.0, equity * leverage - used_margin)


@njit
def calculate_bankruptcy_price(balance: float, long_position_size: float, long_position_price: float,
                               shrt_position_size: float, shrt_position_price: float, inverse: bool,
                               contract_multiplier: float):
    """
    Calculates the bankruptcy price.
    :param balance: The current balance.
    :param long_position_size: The current long position size.
    :param long_position_price: The current long position price.
    :param shrt_position_size: The current short position size.
    :param shrt_position_price: The current short position price.
    :param inverse: Whether it is an inverse contract or not.
    :param contract_multiplier: The multiplier of the contract.
    :return:
    """
    long_position_price = nan_to_0(long_position_price)
    shrt_position_price = nan_to_0(shrt_position_price)
    long_position_size *= contract_multiplier
    shrt_position_size = abs(shrt_position_size) * contract_multiplier
    if inverse:
        shrt_cost = shrt_position_size / shrt_position_price if shrt_position_price > 0.0 else 0.0
        long_cost = long_position_size / long_position_price if long_position_price > 0.0 else 0.0
        denominator = (shrt_cost - long_cost - balance)
        if denominator == 0.0:
            return 0.0
        bankruptcy_price = (shrt_position_size - long_position_size) / denominator
    else:
        denominator = long_position_size - shrt_position_size
        if denominator == 0.0:
            return 0.0
        bankruptcy_price = (
                                   -balance + long_position_size * long_position_price - shrt_position_size * shrt_position_price) / denominator
    return max(0.0, bankruptcy_price)


@njit
def calculate_equity(balance: float, long_position_size: float, long_position_price: float, short_position_size: float,
                     short_position_price: float, last_price: float, inverse: bool, contract_multiplier: float):
    """
    Calculates the equity, the current balance plus the current worth of the positions.
    :param balance: The current balance.
    :param long_position_size: The long position size to use.
    :param long_position_price: The long position price to use.
    :param short_position_size: The short position size to use.
    :param short_position_price: The short position price to use.
    :param last_price: The last price to calculate it on.
    :param inverse: Whether it is an inverse contract or not.
    :param contract_multiplier: The contract multiplier for inverse contracts.
    :return: The current equity.
    """
    equity = balance
    if long_position_price and long_position_size:
        equity += calculate_long_pnl(long_position_price, last_price, long_position_size, inverse, contract_multiplier)
    if short_position_price and short_position_size:
        equity += calculate_short_pnl(short_position_price, last_price, short_position_size, inverse,
                                      contract_multiplier)
    return equity


@njit
def aggregate_ticks_to_candle(tick_list: List[Tick], candle_list: List[Candle], candle_start_time: int,
                              last_candle: Candle, tick_interval: float) -> List[Candle]:
    """
    Aggregates ticks to a candle or creates an 'empty' candle that has open = high = low = close price with zero
    quantity.
    :param tick_list: List of ticks to aggregate.
    :param candle_list: List of already previously aggregates candles.
    :param candle_start_time: Start time of the candle.
    :param last_candle: The last candle before the current candle preparation step.
    :param tick_interval: The tick interval to aggregate in seconds.
    :return: A list of candles.
    """
    if tick_list:
        prices = []
        quantity = []
        for t in tick_list:
            prices.append(t.price)
            quantity.append(t.quantity)
        prices = np.asarray(prices)
        quantity = np.asarray(quantity)
        candle = Candle(candle_start_time + int(tick_interval * 1000), prices[0], np.max(prices), np.min(prices),
                        prices[-1], np.sum(quantity))
        candle_list.append(candle)
    else:
        if candle_list:
            last_candle = candle_list[-1]
            candle = Candle(candle_start_time + int(tick_interval * 1000), last_candle.close, last_candle.close,
                            last_candle.close, last_candle.close, 0.0)
            candle_list.append(candle)
        else:
            candle = Candle(candle_start_time + int(tick_interval * 1000), last_candle.close, last_candle.close,
                            last_candle.close, last_candle.close, 0.0)
            candle_list.append(candle)
    return candle_list


@njit
def prepare_candles(tick_list: List[Tick], last_candle_start_time: int, max_candle_start_time: int, last_candle: Candle,
                    tick_interval: float) -> Tuple[List[Candle], List[Tick], int]:
    """
    Prepares a list of ticks into one or more candles. Fills in gap in candles if update time is longer than the tick
    interval.
    :param tick_list: The list of ticks to aggregate.
    :param last_candle_start_time: The last time this function was called.
    :param max_candle_start_time: The stop time to aggregate. This represents the current, not finished, tick interval.
    :param last_candle: The last candle of the last aggregation.
    :param tick_interval: The tick interval to aggregate in seconds.
    :return: A list of candles, a list of not yet aggregated ticks, the timestamp of the current candle.
    """
    tmp_tick_list = empty_tick_list()
    candle_list = empty_candle_list()
    current_lowest_time = last_candle_start_time
    for tick in tick_list:
        if tick.timestamp < (current_lowest_time + tick_interval * 1000):
            tmp_tick_list.append(tick)
        else:
            while (current_lowest_time + tick_interval * 1000 - 1) < int(
                    tick.timestamp - (tick.timestamp % (tick_interval * 1000))) and (
                    current_lowest_time + tick_interval * 1000) < max_candle_start_time:
                candle_list = aggregate_ticks_to_candle(tmp_tick_list, candle_list, current_lowest_time, last_candle,
                                                        tick_interval)
                tmp_tick_list = empty_tick_list()
                current_lowest_time += int(tick_interval * 1000)
            tmp_tick_list.append(tick)
    return candle_list, tmp_tick_list, current_lowest_time


@njit
def convert_array_to_tick_list(tick_list: List[Tick], data: np.ndarray) -> List[Tick]:
    """
    Converts an array into a tick list so that it can be further processed.
    :param tick_list: The tick list to use.
    :param data: The data to use in the form: timestamp, price, quantity, is_buyer_maker.
    :return: The tick list with added ticks.
    """
    for row in data:
        tick_list.append(Tick(row[0], row[1], row[2], bool(row[3])))
    return tick_list


@njit
def candles_to_array(candles: List[Candle]) -> np.ndarray:
    """
    Converts a list of candles into a numpy array.
    :param candles: The list of candles.
    :return: A numpy array int he form: timestamp, open, high, low, close, volume.
    """
    array = np.zeros((len(candles), 6))
    for i in range(len(candles)):
        array[i] = np.asarray([candles[i].timestamp, candles[i].open, candles[i].high, candles[i].low, candles[i].close,
                               candles[i].volume])
    return array


@njit
def average_candle_price(candle: Candle) -> float:
    """
    Calculates the average price of a candle based on open, high, low, and close price.
    :param candle: The candle to calculate the price for.
    :return: The average price.
    """
    return float(np.mean(np.asarray([candle.open, candle.high, candle.low, candle.close])))
