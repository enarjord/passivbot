import asyncio
import datetime
import hashlib
import hmac
import json
from threading import Thread, Lock, Timer
from time import time
from urllib.parse import urlencode

import aiohttp
import numpy as np
import websockets


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
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
    try:
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:
            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]
            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
        raise Exception('API KeyFile Missing!')
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


class Bot:
    def __init__(self, config: dict):
        self.config = config
        self.reentry_grid = {'DCA1': (0.6, 1.0),
                             'DCA2': (0.6, 1.0),
                             'DCA3': (1.4, 3.0),
                             'DCA4': (1.7, 2.0),
                             'DCA5': (2.3, 2.0),
                             'DCA6': (3.4, 2.2)}

        self.tp_grid = {'TP1': (0.2, 1.0),
                        'TP2': (0.2, 1.0),
                        'TP3': (0.2, 1.0),
                        'TP4': (0.2, 1.0),
                        'TP5': (0.2, 1.0)}

        self.symbol = config['symbol']
        if 'USDT' in self.symbol:
            self.quote_asset = 'USDT'
        else:
            self.quote_asset = None
        self.user = config['user']

        self.balance = 0
        self.balance_lock = Lock()
        self.percent = self.config['percent']

        self.session = aiohttp.ClientSession()

        self.position = {'LONG': {}, 'SHORT': {}}
        self.position_lock = Lock()
        self.open_orders = {'LONG': [], 'SHORT': []}
        self.open_orders_lock = Lock()

        self.hedge_mode = True
        self.long = True
        self.short = False
        self.key, self.secret = load_key_secret(config['exchange'], self.user)

        self.stop_websocket = False
        self.process_websocket_ticks = True

        self.base_endpoint = 'https://fapi.binance.com'
        self.endpoints = {
            'listenkey': '/fapi/v1/listenKey',
            'position': '/fapi/v2/positionRisk',
            'balance': '/fapi/v2/balance',
            'exchange_info': '/fapi/v1/exchangeInfo',
            'leverage_bracket': '/fapi/v1/leverageBracket',
            'open_orders': '/fapi/v1/openOrders',
            'ticker': '/fapi/v1/ticker/bookTicker',
            'fills': '/fapi/v1/userTrades',
            'income': '/fapi/v1/income',
            'create_order': '/fapi/v1/order',
            'cancel_order': '/fapi/v1/order',
            'ticks': '/fapi/v1/aggTrades',
            'margin_type': '/fapi/v1/marginType',
            'leverage': '/fapi/v1/leverage',
            'position_side': '/fapi/v1/positionSide/dual',
            'websocket': 'wss://fstream.binance.com/ws/'
        }
        self.listen_updater = None

    async def init(self):
        try:
            res = await self.private_post(self.endpoints['position_side'], {'dualSidePosition': 'true'})
            print(res)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print(e)
                print('unable to set hedge mode, aborting')
                raise Exception('failed to set hedge mode')
        try:
            print(await self.private_post(self.endpoints['margin_type'],
                                          {'symbol': self.symbol, 'marginType': 'CROSSED'}))
        except Exception as e:
            print(e)
        try:
            lev = await self.execute_leverage_change()
            print('Set leverage to', lev)
        except Exception as e:
            print(e)

        exchange_info = await self.public_get(self.endpoints['exchange_info'])

        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                for q in e['filters']:
                    if q['filterType'] == 'LOT_SIZE':
                        self.min_qty = self.config['min_qty'] = float(q['minQty'])
                    elif q['filterType'] == 'MARKET_LOT_SIZE':
                        self.qty_step = self.config['qty_step'] = float(q['stepSize'])
                    elif q['filterType'] == 'PRICE_FILTER':
                        self.price_step = self.config['price_step'] = float(q['tickSize'])
                    elif q['filterType'] == 'MIN_NOTIONAL':
                        self.min_cost = self.config['min_cost'] = float(q['notional'])
                try:
                    z = self.min_cost
                except AttributeError:
                    self.min_cost = self.config['min_cost'] = 0.0
                break

        self.user_thread = Thread(target=self.start_user_data)
        self.user_thread.start()

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

    async def init_orders(self):
        ords = await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        add_orders = {'LONG': [], 'SHORT': []}
        for o in ords:
            if o['symbol'] == self.symbol:
                d = {'order_id': int(o['orderId']),
                     'price': float(o['price']),
                     'qty': float(o['origQty']),
                     'type': o['type'].upper(),
                     'side': o['side'].upper(),
                     'timestamp': int(o['time'])}
                if o['positionSide'].upper() == 'LONG':
                    add_orders['LONG'].append(d)
                elif o['positionSide'].upper() == 'SHORT':
                    add_orders['SHORT'].append(d)
                else:
                    print(o)
        self.update_orders(add_orders, {})

    async def init_position(self):
        pos = await self.private_get(self.endpoints['position'], ({'symbol': self.symbol}))
        long = None
        short = None
        for p in pos:
            if p['symbol'] == self.symbol:
                d = {'size': float(p['positionAmt']),
                     'price': float(p['entryPrice']),
                     # 'liquidation_price': float(p['liquidationPrice']),
                     # 'leverage': float(p['leverage']),
                     'upnl': float(p['unRealizedProfit'])}
                if p['positionSide'] == 'LONG' and float(p['positionAmt']) != 0.0:
                    long = d
                elif p['positionSide'] == 'SHORT' and float(p['positionAmt']) != 0.0:
                    short = d
        self.update_position(long, short)

    async def init_balance(self):
        bal = await self.private_get(self.endpoints['balance'], {})
        for b in bal:
            if b['asset'] == self.quote_asset:
                self.update_balance(float(b['balance']))
                break

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

    def get_position_size(self, current_price):
        wallet_balance = self.get_balance()
        available_balance = wallet_balance * self.percent
        reentry_prices = []
        reentry_mults = []
        price_multi = 1.0
        size_multi = 1.0
        # Calculate price levels for reentries and position multiplicators on each level
        for k, v in self.reentry_grid.items():
            price_multi *= (1 - v[0] / 100)
            size_multi *= v[1]
            reentry_prices.append(price_multi * current_price)
            reentry_mults.append(size_multi)
        # Calculate the amount of the initial position based on multiplicators and what is needed for reentries. Plus one for initial position.
        amount = available_balance / (sum(reentry_mults) + 1)
        initial_size = amount / current_price
        reentry_sizes = []
        # Calculate size of reentry in symbol value based on price level and multiplicated amount
        for i in range(len(reentry_mults)):
            reentry_sizes.append(amount * reentry_mults[i] / reentry_prices[i])
        return initial_size, reentry_prices, reentry_sizes

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(self, type_: str, url: str, params: dict = {}) -> dict:
        timestamp = int(time() * 1000)
        params.update({'timestamp': timestamp, 'recvWindow': 5000})
        for k in params:
            if type(params[k]) == bool:
                params[k] = 'true' if params[k] else 'false'
            elif type(params[k]) == float:
                params[k] = str(params[k])
        params = sort_dict_keys(params)
        params['signature'] = hmac.new(self.secret.encode('utf-8'),
                                       urlencode(params).encode('utf-8'),
                                       hashlib.sha256).hexdigest()
        headers = {'X-MBX-APIKEY': self.key}
        async with getattr(self.session, type_)(self.base_endpoint + url, params=params, headers=headers) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}) -> dict:
        return await self.private_('get', url, params)

    async def private_post(self, url: str, params: dict = {}) -> dict:
        return await self.private_('post', url, params)

    async def private_put(self, url: str, params: dict = {}) -> dict:
        return await self.private_('put', url, params)

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        return await self.private_('delete', url, params)

    async def update_listen(self):
        await self.private_put(self.endpoints['listenkey'], {})
        self.listen_updater = Timer(60, self.update_listen)
        self.listen_updater.start()

    async def handle_order_update(self, order):
        d = {'order_id': int(order['i']),
             'price': float(order['p']),
             'qty': float(order['q']),
             'type': order['o'].upper(),
             'side': order['S'].upper(),
             'timestamp': int(order['T'])}
        add_orders = {'LONG': [], 'SHORT': []}
        delete_orders = {'LONG': [], 'SHORT': []}
        if order['X'] in ['CANCELED', 'FILLED', 'EXPIRED', 'NEW_INSURANCE',
                          'NEW_ADL']:
            delete_orders[order['ps'].upper()].append(d)
        if order['X'] in ['NEW']:
            add_orders[order['ps'].upper()].append(d)
        if order['X'] in ['PARTIALLY_FILLED']:
            delete_orders[order['ps'].upper()].append(d)
            add_orders[order['ps'].upper()].append(d)
        self.update_orders(add_orders, delete_orders)

    async def handle_account_update(self, account):
        for b in account['B']:
            if b['a'].upper() == self.quote_asset:
                self.update_balance(float(b['wb']))
                break
        if 'P' in account:
            position = self.get_position()
            last_position = None
            for p in account['P']:
                if p['s'] == self.symbol:
                    d = {'size': float(p['pa']),
                         'price': float(p['ep']),
                         'upnl': float(p['up'])}
                    last_position = (p['ps'].upper(), d)
            if last_position:
                if last_position[1]['price'] == 0.0 and position[last_position[0]]:
                    if last_position[0] == 'LONG':
                        self.update_position({}, None)
                    if last_position[0] == 'SHORT':
                        self.update_position(None, {})
                elif last_position[1]['price'] != 0.0:
                    if last_position[0] == 'LONG':
                        self.update_position(last_position[1], None)
                    if last_position[0] == 'SHORT':
                        self.update_position(None, last_position[1])

    async def start_user_data(self) -> None:
        while True:
            try:
                await self.reset()
                self.listenKey = await self.private_post(self.endpoints['listenkey'], {})
                self.listenKey = self.listenKey['listenKey']
                if self.listen_updater:
                    self.listen_updater.cancel()
                self.listen_updater = Timer(60, self.update_listen)
                self.listen_updater.start()
                async with websockets.connect(self.endpoints['websocket'] + self.listenKey) as ws:
                    async for msg in ws:
                        if msg is None:
                            continue
                        try:
                            if 'e' in msg:
                                if msg['o']['s'].upper() == self.symbol:
                                    if msg['e'] == 'ORDER_TRADE_UPDATE':
                                        await self.handle_order_update(msg['o'])
                                    elif msg['e'] == 'ACCOUNT_UPDATE':
                                        await self.handle_account_update(msg['a'])
                            print(msg)
                        except Exception as e:
                            if 'success' not in msg:
                                print('error in websocket', e, msg)
            except Exception as e_out:
                print(e_out)
                print(datetime.datetime.now(), 'Retrying to connect in 5 seconds...')
                await asyncio.sleep(5)

    async def start_websocket(self) -> None:
        async with websockets.connect(self.endpoints['websocket'] + f"{self.symbol.lower()}@kline_1m") as ws:
            async for msg in ws:
                if msg is None:
                    continue
                try:
                    asyncio.create_task(self.decide())
                except Exception as e:
                    if 'success' not in msg:
                        print('error in websocket', e, msg)

    async def execute_leverage_change(self):
        return await self.private_post(self.endpoints['leverage'],
                                       {'symbol': self.symbol, 'leverage': int(self.config['leverage'])})

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'positionSide': order['position_side'].replace('shrt', 'short').upper(),
                  'type': order['type'].upper(),
                  'quantity': str(order['qty'])}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = str(order['price'])
        if 'custom_id' in order:
            params['newClientOrderId'] = \
                f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        o = await self.private_post(self.endpoints['create_order'], params)
        if 'side' in o:
            return {'symbol': self.symbol,
                    'side': o['side'].lower(),
                    'position_side': o['positionSide'].lower().replace('short', 'shrt'),
                    'type': o['type'].lower(),
                    'qty': float(o['origQty']),
                    'price': float(o['price'])}
        else:
            return o

    async def execute_cancellation(self, order: dict) -> [dict]:
        cancellation = await self.private_delete(self.endpoints['cancel_order'],
                                                 {'symbol': self.symbol, 'orderId': order['order_id']})
        if 'side' in cancellation:
            return {'symbol': self.symbol, 'side': cancellation['side'].lower(),
                    'position_side': cancellation['positionSide'].lower().replace('short', 'shrt'),
                    'qty': float(cancellation['origQty']), 'price': float(cancellation['price'])}
        else:
            return cancellation

    async def create_orders(self, orders_to_create: [dict]) -> [dict]:
        if not orders_to_create:
            return {}
        creations = []
        for oc in orders_to_create:
            try:
                creations.append((oc, asyncio.create_task(self.execute_order(oc))))
            except Exception as e:
                print_(['error creating order a', oc, e], n=True)
        created_orders = []
        for oc, c in creations:
            try:
                o = await c
                created_orders.append(o)
                if 'side' in o:
                    print_([' created order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error creating order b', o, oc], n=True)
            except Exception as e:
                print_(['error creating order c', oc, c.exception(), e], n=True)
        return created_orders

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if not orders_to_cancel:
            return []
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletions.append((oc, asyncio.create_task(self.execute_cancellation(oc))))
            except Exception as e:
                print_(['error cancelling order a', oc, e])
        canceled_orders = []
        for oc, c in deletions:
            try:
                o = await c
                canceled_orders.append(o)
                if 'side' in o:
                    print_(['cancelled order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error cancelling order', o], n=True)
            except Exception as e:
                print_(['error cancelling order b', oc, c.exception(), e], n=True)
        return canceled_orders

    async def decide(self):
        if True:
            pass
