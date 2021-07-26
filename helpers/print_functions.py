import datetime

from definitions.candle import Candle
from definitions.order import Order
from definitions.order_list import OrderList
from definitions.position import Position
from definitions.position_list import PositionList
from definitions.tick import Tick


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
