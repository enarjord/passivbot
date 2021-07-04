import asyncio
import json
from threading import Lock
from typing import Union

import aiohttp
import websockets

from functions import load_key_secret, add_or_append
from functions import print_
from strategies.grid import Grid


class Bot:
    def __init__(self, config: dict):
        self.config = config
        self.strategy = Grid(config)

        self.symbol = config['symbol']

        self.user = config['user']

        self.balance = 0
        self.balance_lock = Lock()

        self.session = aiohttp.ClientSession()

        self.position = {'LONG': {}, 'SHORT': {}}
        self.position_lock = Lock()
        self.open_orders = {'LONG': [], 'SHORT': []}
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

    async def fetch_orders(self):
        raise NotImplementedError

    async def fetch_position(self):
        raise NotImplementedError

    async def fetch_balance(self):
        raise NotImplementedError

    def prepare_order(self, msg) -> dict:
        raise NotImplementedError

    def prepare_account(self, msg) -> dict:
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

    async def reset(self):
        self.balance_lock.acquire()
        self.balance = 0
        self.balance_lock.release()
        self.position_lock.acquire()
        self.position = {'LONG': {}, 'SHORT': {}}
        self.position_lock.release()
        self.open_orders_lock.acquire()
        self.open_orders = {'LONG': [], 'SHORT': []}
        self.open_orders_lock.release()
        await self.init_orders()
        await self.init_position()
        await self.init_balance()
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

    async def init_orders(self):
        add_orders = await self.fetch_orders()
        self.update_orders(add_orders, {})

    async def init_position(self):
        long, short = await self.fetch_position()
        self.update_position(long, short)

    async def init_balance(self):
        bal = await self.fetch_balance()
        self.update_balance(bal)

    def update_orders(self, add_orders: dict = {}, delete_orders: dict = {}):
        self.open_orders_lock.acquire()
        for side, orders in delete_orders.items():
            for j in range(len(orders)):
                for i in range(len(self.open_orders[side])):
                    if self.open_orders[side][i]['order_id'] == orders[j]['order_id']:
                        del self.open_orders[side][i]
                        break
        for side, orders in add_orders.items():
            for j in range(len(orders)):
                self.open_orders[side].append(orders[j])
        self.open_orders_lock.release()

    def update_position(self, long: dict = None, short: dict = None):
        self.position_lock.acquire()
        if long is not None:
            self.position['LONG'] = long
        if short is not None:
            self.position['SHORT'] = short
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

    async def handle_order_update(self, order):
        d = {'order_id': order['order_id'],
             'price': order['price'],
             'qty': order['qty'],
             'type': order['type'],
             'side': order['side'],
             'timestamp': order['timestamp']}
        add_orders = {}
        delete_orders = {}
        if order['action'] in ['CANCELED', 'FILLED', 'EXPIRED', 'NEW_INSURANCE', 'NEW_ADL']:
            delete_orders = add_or_append(delete_orders, order['position_side'], d)
        if order['action'] in ['NEW']:
            add_orders = add_or_append(add_orders, order['position_side'], d)
        if order['action'] in ['PARTIALLY_FILLED']:
            delete_orders = add_or_append(delete_orders, order['position_side'], d)
            add_orders = add_or_append(add_orders, order['position_side'], d)
        if order['action'] == 'FILLED':
            self.last_filled_order = order
            self.order_fill_change = True
        self.update_orders(add_orders, delete_orders)

    async def handle_account_update(self, account):
        self.update_balance(account['balance'])
        if 'position' in account:
            position = self.get_position()
            if 'last_long' in account['position']:
                last_long = account['position']['last_long']
                if last_long['price'] == 0.0 and position['LONG']:
                    self.update_position({}, None)
                elif last_long['price'] != 0.0:
                    self.update_position(last_long, None)
            if 'last_short' in account['position']:
                last_short = account['position']['last_short']
                if last_short['price'] == 0.0 and position['SHORT']:
                    self.update_position(None, {})
                elif last_short['price'] != 0.0:
                    self.update_position(None, last_short)
            self.position_change = True

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
                                if type == 'order':
                                    await self.handle_order_update(self.prepare_order(msg))
                                elif type == 'account':
                                    await self.handle_account_update(self.prepare_account(msg))
                                if self.position_change and self.order_fill_change:
                                    self.strategy.update_balance(self.get_balance())
                                    self.strategy.update_orders(self.get_orders())
                                    changed_orders = self.strategy.on_update(self.get_position(),
                                                                             self.last_filled_order)
                                    self.strategy.update_values(self.get_balance(), self.get_position(),
                                                                self.get_orders())
                                    asyncio.create_task(self.cancel_orders(changed_orders['cancel']))
                                    asyncio.create_task(self.create_orders(changed_orders['add']))
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

    async def execute_order(self, order: dict) -> Union[dict, bool]:
        raise NotImplementedError

    async def execute_cancellation(self, order: dict) -> Union[dict, bool]:
        raise NotImplementedError

    async def create_orders(self, orders_to_create: [dict]) -> [dict]:
        raise NotImplementedError

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        raise NotImplementedError

    async def decide(self, price):
        changed_orders = self.strategy.make_decision(self.get_balance(), self.get_position(), self.get_orders(), price)
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())
        await self.cancel_orders(changed_orders['cancel'])
        await self.create_orders(changed_orders['add'])
