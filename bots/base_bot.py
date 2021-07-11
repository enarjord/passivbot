import asyncio
import json
from threading import Lock
from typing import Union, Tuple, List

import aiohttp
import numpy as np
import websockets
from numba import njit

from definitions.order import NEW, PARTIALY_FILLED, FILLED, CANCELED, EXPIRED, LONG, SHORT, NEW_INSURANCE, NEW_ADL
from definitions.order import Order
from definitions.order_list import OrderList, empty_order_list
from definitions.position import Position
from definitions.position_list import PositionList
from functions import load_key_secret
from functions import print_
from strategies.grid import Grid

ORDER_UPDATE = 'order'
ACCOUNT_UPDATE = 'account'


@njit
def round_dn(n, step, safety_rounding=10) -> float:
    return np.round(np.floor(np.round(n / step, safety_rounding)) * step, safety_rounding)


@njit
def round_up(n, step, safety_rounding=10) -> float:
    return np.round(np.ceil(np.round(n / step, safety_rounding)) * step, safety_rounding)


class Bot:
    def __init__(self, config: dict):
        self.config = config
        self.strategy = Grid(config)

        self.symbol = config['symbol']

        self.user = config['user']

        self.balance = 0
        self.balance_lock = Lock()

        self.session = aiohttp.ClientSession()

        self.position = PositionList()
        self.position_lock = Lock()
        self.open_orders = OrderList()
        self.open_orders_lock = Lock()

        self.long = True
        self.short = False
        self.key, self.secret = load_key_secret(config['exchange'], self.user)

        self.qty_step = None
        self.price_step = None

        self.last_filled_order = None

        self.base_endpoint = ''
        self.endpoints = {
            'listenkey': '',
            'position': '',
            'balance': '',
            'exchange_info': '',
            'leverage_bracket': '',
            'open_orders': '',
            'ticker': '',
            'fills': '',
            'income': '',
            'create_order': '',
            'cancel_order': '',
            'ticks': '',
            'margin_type': '',
            'leverage': '',
            'position_side': '',
            'websocket': '',
            'websocket_user': ''
        }

    async def init(self):
        raise NotImplementedError

    async def fetch_orders(self) -> List[Order]:
        raise NotImplementedError

    async def fetch_position(self) -> Tuple[Position, Position]:
        raise NotImplementedError

    async def fetch_balance(self) -> float:
        raise NotImplementedError

    def prepare_order(self, msg) -> Order:
        raise NotImplementedError

    def prepare_account(self, msg) -> Tuple[float, Position, Position]:
        raise NotImplementedError

    async def public_get(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_(self, type_: str, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_get(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_post(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_put(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def update_heartbeat(self):
        pass

    def determine_update_type(self, msg) -> str:
        raise NotImplementedError

    def precompile(self):
        p = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        p.equal(p)
        o = Order('', 0, 0.0, 0.0, 0.0, '', '', 0, '', '')
        o.equal(o)
        pl = PositionList()
        pl.update_long(p)
        pl.update_short(p)
        pl.copy()
        ol = OrderList()
        ol.add_long(empty_order_list())
        ol.add_short(empty_order_list())
        ol.delete_long(empty_order_list())
        ol.delete_short(empty_order_list())
        ol.update_long(empty_order_list())
        ol.update_short(empty_order_list())
        ol.copy()
        self.strategy.precompile()

    async def reset(self):
        self.precompile()
        self.balance_lock.acquire()
        self.balance = 0
        self.balance_lock.release()
        self.position_lock.acquire()
        self.position = PositionList()
        self.position_lock.release()
        self.open_orders_lock.acquire()
        self.open_orders = OrderList()
        self.open_orders_lock.release()
        await self.init_orders()
        await self.init_position()
        await self.init_balance()
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

    async def init_orders(self):
        add_orders = await self.fetch_orders()
        self.update_orders(add_orders, [])

    async def init_position(self):
        long, short = await self.fetch_position()
        self.update_position(long, short)

    async def init_balance(self):
        bal = await self.fetch_balance()
        self.update_balance(bal)

    def update_orders(self, add_orders: list = [], delete_orders: list = []):
        self.open_orders_lock.acquire()
        add_long = empty_order_list()
        add_short = empty_order_list()
        delete_long = empty_order_list()
        delete_short = empty_order_list()
        for order in delete_orders:
            if order.position_side == LONG:
                delete_long.append(order)
            elif order.position_side == SHORT:
                delete_short.append(order)
        for order in add_orders:
            if order.position_side == LONG:
                add_long.append(order)
            elif order.position_side == SHORT:
                add_short.append(order)
        self.open_orders.delete_long(delete_long)
        self.open_orders.delete_short(delete_short)
        self.open_orders.add_long(add_long)
        self.open_orders.add_short(add_short)
        self.open_orders_lock.release()

    def update_position(self, long: Position, short: Position):
        self.position_lock.acquire()
        self.position.update_long(long)
        self.position.update_short(short)
        self.position_lock.release()

    def update_balance(self, balance: float = None):
        self.balance_lock.acquire()
        if balance:
            self.balance = balance
        self.balance_lock.release()

    def get_orders(self):
        self.open_orders_lock.acquire()
        open_orders = self.open_orders.copy()
        self.open_orders_lock.release()
        return open_orders

    def get_position(self):
        self.position_lock.acquire()
        position = self.position.copy()
        self.position_lock.release()
        return position

    def get_balance(self):
        self.balance_lock.acquire()
        balance = self.balance
        self.balance_lock.release()
        return balance

    async def handle_order_update(self, order: Order):
        add_orders = []
        delete_orders = []
        if order.action in [CANCELED, FILLED, EXPIRED, NEW_INSURANCE, NEW_ADL]:
            delete_orders.append(order)
        if order.action in [NEW]:
            add_orders.append(order)
        if order.action in [PARTIALY_FILLED]:
            delete_orders.append(order)
            add_orders.append(order)
        if order.action == FILLED:
            self.last_filled_order = order
            self.order_fill_change = True
        self.update_orders(add_orders, delete_orders)

    async def handle_account_update(self, balance: float, long: Position, short: Position):
        self.update_balance(balance)
        if not self.position.long.equal(long) or not self.position.short.equal(short):
            self.position_change = True
        self.update_position(long, short)

    async def start_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.update_heartbeat()

    async def start_user_data(self) -> None:
        while True:
            try:
                self.position_change = False
                self.order_fill_change = False
                await self.reset()
                await self.update_heartbeat()
                async with websockets.connect(self.endpoints['websocket_user']) as ws:
                    async for msg in ws:
                        if msg is None:
                            continue
                        try:
                            msg = json.loads(msg)
                            type = self.determine_update_type(msg)
                            if type:
                                # print(msg)
                                if type == ORDER_UPDATE:
                                    await self.handle_order_update(self.prepare_order(msg))
                                elif type == ACCOUNT_UPDATE:
                                    await self.handle_account_update(*self.prepare_account(msg))
                                if self.position_change and self.order_fill_change:
                                    self.strategy.update_balance(self.get_balance())
                                    self.strategy.update_orders(self.get_orders())
                                    add_orders, delete_orders = self.strategy.on_update(self.get_position(),
                                                                                        self.last_filled_order)
                                    self.strategy.update_values(self.get_balance(), self.get_position(),
                                                                self.get_orders())
                                    asyncio.create_task(self.cancel_orders(delete_orders))
                                    asyncio.create_task(self.create_orders(add_orders))
                                    self.position_change = False
                                    self.order_fill_change = False
                        except Exception as e:
                            print_(['User stream error', e], n=True)
            except Exception as e_out:
                print_(['User stream error', e_out], n=True)
                print_(['Retrying to connect in 5 seconds...'], n=True)
                await asyncio.sleep(5)

    async def start_websocket(self) -> None:
        while True:
            async with websockets.connect(self.endpoints['websocket'] + f"{self.symbol.lower()}@kline_1m") as ws:
                async for msg in ws:
                    if msg is None:
                        continue
                    try:
                        msg = json.loads(msg)
                        if msg['k']['x']:
                            pass
                            print_(['Kline closed, do something'], n=True)
                            # asyncio.create_task(self.decide(float(msg['k']['c'])))
                    except Exception as e:
                        if 'success' not in msg:
                            print_(['Error in price stream', e, msg], n=True)

    async def execute_leverage_change(self):
        raise NotImplementedError

    async def execute_order(self, order: Order) -> Union[dict, bool]:
        raise NotImplementedError

    async def execute_cancellation(self, order: Order) -> Union[dict, bool]:
        raise NotImplementedError

    async def create_orders(self, orders_to_create: List[Order]):
        raise NotImplementedError

    async def cancel_orders(self, orders_to_cancel: List[Order]):
        raise NotImplementedError

    def correct_float_precision(self, order):
        if not np.isclose(order.price, round_dn(order.price, self.price_step), rtol=1e-60, atol=1e-60):
            if order.price > round_dn(order.price, self.price_step):
                order.price = round_up(order.price, self.price_step)
            else:
                order.price = round_dn(order.price, self.price_step)
        if not np.isclose(order.stop_price, round_dn(order.stop_price, self.price_step), rtol=1e-60, atol=1e-60):
            if order.stop_price > round_dn(order.stop_price, self.price_step):
                order.stop_price = round_up(order.stop_price, self.price_step)
            else:
                order.stop_price = round_dn(order.stop_price, self.price_step)
        if not np.isclose(order.qty, round_dn(order.qty, self.qty_step), rtol=1e-60, atol=1e-60):
            if order.qty > round_dn(order.qty, self.qty_step):
                order.qty = round_up(order.qty, self.qty_step)
            else:
                order.qty = round_dn(order.qty, self.qty_step)
        return order

    async def decide(self, price):
        add_orders, delete_orders = self.strategy.make_decision(self.get_balance(), self.get_position(),
                                                                self.get_orders(), price)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

        await self.cancel_orders(delete_orders)
        await self.create_orders(add_orders)
