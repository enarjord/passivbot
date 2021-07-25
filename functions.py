import datetime
import json
import sys
from importlib.util import spec_from_loader, module_from_spec
from pathlib import Path
from typing import Union

import hjson

from definitions.candle import Candle
from definitions.order import Order
from definitions.order_list import OrderList
from definitions.position import Position
from definitions.position_list import PositionList
from definitions.tick import Tick


def remove_numba_decorators(text: str) -> str:
    """
    Removes the decorators associated with numba so that a clean file remains.
    :param text: The text (script) to clean.
    :return: Cleaned text.
    """
    match = '@jitclass('
    index = text.find(match)
    while index != -1:
        open_brackets = 1
        start_index = index + len(match)
        end_index = 0
        for i in range(len(text[start_index:])):
            if text[start_index:][i] == '(':
                open_brackets += 1
            if text[start_index:][i] == ')':
                open_brackets -= 1
            if open_brackets == 0:
                end_index = i
                break
        text = text.replace(text[index:index + len(match) + end_index + 1], '')
        index = text.find(match)
    text = text.replace('@njit', '')
    return text


def load_module_from_file(file_name: str, module_name: str, to_be_replaced: tuple = (None, None),
                          insert_at_start: str = ''):
    """
    Loads a module directly from a file. In the case of a script with numba compatibility, it strips it first, imports
    the clean module, and then imports the module with numba enabled.
    :param file_name: The filename to import.
    :param module_name: The name of the new module.
    :return: A module.
    """
    text = Path(file_name).read_text()
    original_text = text
    if insert_at_start:
        original_text = insert_at_start + '\n' + original_text
    if to_be_replaced[0]:
        original_text = original_text.replace(to_be_replaced[0], to_be_replaced[1])
    numba_free_text = remove_numba_decorators(text)
    module_spec = spec_from_loader(module_name, loader=None)
    module = module_from_spec(module_spec)
    exec(numba_free_text, module.__dict__)
    sys.modules[module_name] = module
    exec(original_text, module.__dict__)
    sys.modules[module_name] = module
    return module


def get_strategy_definition(filename: str) -> str:
    """
    Finds the strategy definition inside a strategy script and extracts it so that it can be inserted in the backtest
    bot file before loading it.
    :param filename: The filename to check.
    :return: A string containing the strategy definition.
    """
    text = Path(filename).read_text()
    match = 'strategy_definition'
    index = text.find(match)
    index = index + len(match)
    start_index = 0
    for i in range(len(text[index:])):
        if text[index:][i].isalnum():
            start_index = i
            break
    index = index + start_index
    open_brackets = 0
    first_bracket = False
    end_index = 0
    for i in range(len(text[index:])):
        if text[index:][i] == '(':
            open_brackets += 1
            if not first_bracket:
                first_bracket = True
        if text[index:][i] == ')':
            open_brackets -= 1
        if open_brackets == 0 and first_bracket:
            end_index = i
            break
    return text[index:index + end_index + 1]


def sort_dict_keys(d: Union[dict, list]) -> Union[dict, list]:
    """
    Sort dictionaries and keys.
    :param d: The object to sort.
    :return: A sorted list or dictionary.
    """
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def ts_to_date(timestamp: float) -> str:
    """
    Converts timestamp to human readable time.
    :param timestamp: The epoch timestamp.
    :return: A human readable time.
    """
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def print_(args, r=False, n=False):
    """
    Prints a list of arguments.
    :param args: The list of arguments.
    :param r: If r as newline should be used.
    :param n: If n as newline should be used.
    :return: The arguments as a string.
    """
    line = str(datetime.datetime.now()) + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        print('\n' + line, end=' ')
    elif r:
        print('\r' + line, end=' ')
    else:
        print(line)
    return line


def load_key_secret(exchange: str, user: str) -> (str, str):
    """
    Loads the key and secret from the API key file.
    :param exchange: The exchange to use as a key.
    :param user: The user to use as a key.
    :return: The key and secret.
    """
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:
            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]
            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print_(["Looks like the keys aren't configured yet, or you entered the wrong username!"], n=True)
        raise Exception('API KeyFile Missing!')
    except FileNotFoundError:
        print_(["File not found!"], n=True)
        raise Exception('API KeyFile Missing!')


def load_base_config(path: str) -> dict:
    """
    Loads the base config from an hjson file.
    :param path: The path to the config.
    :return: The config as a dictionary.
    """
    try:
        config = hjson.load(open(path))
        return config
    except Exception as e:
        print_(["Could not read config", e], n=True)
        return {}


def print_candle(candle: Candle):
    """
    Prints a Candle.
    :param candle: The candle to print.
    :return:
    """
    print_(['Timestamp', candle.timestamp, 'Open', candle.open, 'High', candle.high, 'Low', candle.low, 'High',
            candle.high, 'Close', candle.close, 'Quantity', candle.qty], n=True)


def print_tick(tick: Tick):
    """
    Prints a tick.
    :param tick: The tick to print.
    :return:
    """
    print_(['Timestamp', tick.timestamp, 'Price', tick.price, 'Quantity', tick.qty, 'Maker', tick.is_buyer_maker],
           n=True)


def print_order(order: Order):
    """
    Prints an Order.
    :param order: The order to print.
    :return:
    """
    print_(['Symbol', order.symbol, 'Order_id', order.order_id, 'Price', order.price, 'Stop price', order.stop_price,
            'Qty', order.qty, 'Type', order.type, 'Side', order.side, 'Timestamp', order.timestamp, 'Action',
            order.action, 'Position_side', order.position_side], n=True)


def print_position(position: Position):
    """
    Prints a Position.
    :param position: The position to print.
    :return:
    """
    print_(['Symbol', position.symbol, 'Size', position.size, 'Price', position.price, 'Liquidation_price',
            position.liquidation_price, 'Upnl', position.upnl, 'Leverage', position.leverage, 'Position_side',
            position.position_side], n=True)


def print_order_list(order_list: OrderList):
    """
    Prints an OrderList.
    :param order_list: The order list to print.
    :return:
    """
    print_(['Long:'], n=True)
    for order in order_list.long:
        print_order(order)
    print_(['Short:'], n=True)
    for order in order_list.short:
        print_order(order)


def print_position_list(position_list: PositionList):
    """
    Prints a PositionList.
    :param position_list: The position list to print.
    :return:
    """
    print_(['Long:'], n=True)
    print_position(position_list.long)
    print_(['Short:'], n=True)
    print_position(position_list.short)
