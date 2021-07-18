import datetime
import json
from typing import Union

import hjson

from definitions.candle import Candle
from definitions.order import Order
from definitions.order_list import OrderList
from definitions.position import Position
from definitions.position_list import PositionList


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
    print('Open', candle.open, 'High', candle.high, 'Low', candle.low, 'High', candle.high, 'Close', candle.close)


def print_order(order: Order):
    """
    Prints an Order.
    :param order: The order to print.
    :return:
    """
    print('Symbol', order.symbol, 'Order_id', order.order_id, 'Price', order.price, 'Stop price', order.stop_price,
          'Qty', order.qty, 'Type', order.type, 'Side', order.side, 'Timestamp', order.timestamp, 'Action',
          order.action, 'Position_side', order.position_side)


def print_position(position: Position):
    """
    Prints a Position.
    :param position: The position to print.
    :return:
    """
    print('Symbol', position.symbol, 'Size', position.size, 'Price', position.price, 'Liquidation_price',
          position.liquidation_price, 'Upnl', position.upnl, 'Leverage', position.leverage, 'Position_side',
          position.position_side)


def print_order_list(order_list: OrderList):
    """
    Prints an OrderList.
    :param order_list: The order list to print.
    :return:
    """
    print('Long:')
    for order in order_list.long:
        print_order(order)
    print('Short:')
    for order in order_list.short:
        print_order(order)


def print_position_list(position_list: PositionList):
    """
    Prints a PositionList.
    :param position_list: The position list to print.
    :return:
    """
    print('Long:')
    print_position(position_list.long)
    print('Short:')
    print_position(position_list.short)
