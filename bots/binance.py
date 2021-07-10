import asyncio
import hashlib
import hmac
import json
from time import time
from typing import Union
from urllib.parse import urlencode

from bots.base_bot import Bot
from functions import sort_dict_keys, print_


class BinanceBot(Bot):
    def __init__(self, config: dict):
        super().__init__(config)
        if 'USDT' in self.symbol:
            self.quote_asset = 'USDT'
        else:
            self.quote_asset = None

        self.hedge_mode = True

        self.listenKey = None

        self.leverage = config['leverage']

        self.base_endpoint = 'https://testnet.binancefuture.com'  # 'https://fapi.binance.com'
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
            'websocket': 'wss://stream.binancefuture.com/ws/',  # 'wss://fstream.binance.com/ws/'
            'websocket_user': ''
        }

    async def init(self):
        try:
            res = await self.private_post(self.endpoints['position_side'], {'dualSidePosition': 'true'})
            print_([res], n=True)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print_([e, 'Unable to set hedge mode, aborting'], n=True)
                raise Exception('Failed to set hedge mode')
        try:
            print_([await self.private_post(self.endpoints['margin_type'],
                                            {'symbol': self.symbol, 'marginType': 'CROSSED'})], n=True)
        except Exception as e:
            print_([e], n=True)
        try:
            lev = await self.execute_leverage_change()
            print_(['Set leverage to', lev], n=True)
        except Exception as e:
            print_([e], n=True)

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
        self.strategy.update_steps(self.qty_step, self.price_step)

    async def fetch_orders(self):
        ords = await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        orders = {'LONG': [], 'SHORT': []}
        for o in ords:
            if o['symbol'] == self.symbol:
                d = {'order_id': int(o['orderId']),
                     'price': float(o['price']),
                     'qty': float(o['origQty']),
                     'type': o['type'].upper(),
                     'side': o['side'].upper(),
                     'timestamp': int(o['time'])}
                if o['positionSide'].upper() == 'LONG':
                    orders['LONG'].append(d)
                elif o['positionSide'].upper() == 'SHORT':
                    orders['SHORT'].append(d)
                else:
                    print_([o], n=True)
        return orders

    async def fetch_position(self):
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
        return long, short

    async def fetch_balance(self):
        bal = await self.private_get(self.endpoints['balance'], {})
        for b in bal:
            if b['asset'] == self.quote_asset:
                return float(b['balance'])

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

    def prepare_order(self, msg) -> dict:
        o = {'order_id': int(msg['o']['i']),
             'price': float(msg['o']['p']),
             'qty': float(msg['o']['q']),
             'type': msg['o']['o'].upper(),
             'side': msg['o']['S'].upper(),
             'timestamp': int(msg['o']['T']),
             'action': msg['o']['X'],
             'position_side': msg['o']['ps'].upper()}
        if 'ot' in msg['o']:
            if msg['o']['ot'] == 'MARKET' and o['action'] != 'PARTIALLY_FILLED':
                o['price'] = float(msg['o']['ap'])
        if o['action'] == 'PARTIALLY_FILLED':
            o['qty'] = o['qty'] - float(msg['o']['z'])
        return o

    def prepare_account(self, msg) -> dict:
        a = {}
        for b in msg['a']['B']:
            if b['a'].upper() == self.quote_asset:
                a['balance'] = float(b['wb'])
                break
        if 'P' in msg['a']:
            a['position'] = {}
            a['position']['last_long'] = None
            a['position']['last_short'] = None
            for p in msg['a']['P']:
                if p['s'] == self.symbol:
                    d = {'size': float(p['pa']),
                         'price': float(p['ep']),
                         'upnl': float(p['up']),
                         'leverage': self.leverage}
                    if p['ps'].upper() == 'LONG':
                        a['position']['last_long'] = d
                    if p['ps'].upper() == 'SHORT':
                        a['position']['last_short'] = d
        return a

    async def update_heartbeat(self):
        if self.listenKey:
            try:
                await self.private_put(self.endpoints['listenkey'], {})
            except Exception as e_listen:
                print_(['Could not refresh listen key', e_listen], n=True)
        else:
            try:
                tmp = await self.private_post(self.endpoints['listenkey'], {})
                self.listenKey = tmp['listenKey']
                self.endpoints['websocket_user'] = self.endpoints['websocket'] + self.listenKey
            except Exception as e_listen:
                print_(['Could not initialize listen key', e_listen], n=True)

    def determine_update_type(self, msg) -> str:
        type = None
        if 'e' in msg:
            if msg['e'] == 'ORDER_TRADE_UPDATE':
                if msg['o']['s'].upper() == self.symbol:
                    type = 'order'
            elif msg['e'] == 'ACCOUNT_UPDATE':
                type = 'account'
        return type

    async def execute_leverage_change(self):
        return await self.private_post(self.endpoints['leverage'],
                                       {'symbol': self.symbol, 'leverage': int(self.config['leverage'])})

    async def execute_order(self, order: dict) -> Union[dict, bool]:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'positionSide': order['position_side'].upper(),
                  'type': order['type'].upper(),
                  'quantity': str(order['qty'])}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = str(order['price'])
        if params['type'] == 'TAKE_PROFIT':
            params['price'] = str(order['price'])
            params['stopPrice'] = str(order['stop_price'])
        o = await self.private_post(self.endpoints['create_order'], params)
        if 'code' in o:
            return o
        else:
            return True

    async def execute_cancellation(self, order: dict) -> Union[dict, bool]:
        c = await self.private_delete(self.endpoints['cancel_order'],
                                      {'symbol': self.symbol, 'orderId': order['order_id']})
        if 'code' in c:
            return c
        else:
            return True

    async def create_orders(self, orders_to_create: [dict]):
        if not orders_to_create:
            return
        creations = []
        for oc in orders_to_create:
            try:
                creations.append((oc, asyncio.create_task(self.execute_order(oc))))
            except Exception as e:
                print_(['Error creating order', oc, e], n=True)
        for oc, c in creations:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error creating order'], n=True)
                else:
                    print_(['Error creating order', o], n=True)
            except Exception as e:
                print_(['Error creating order', oc, c.exception(), e], n=True)
        return

    async def cancel_orders(self, orders_to_cancel: [dict]):
        if not orders_to_cancel:
            return
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletions.append((oc, asyncio.create_task(self.execute_cancellation(oc))))
            except Exception as e:
                print_(['Error cancelling order a', oc, e], n=True)
        for oc, c in deletions:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error cancelling order'], n=True)
                else:
                    print_(['Error cancelling order', o], n=True)
            except Exception as e:
                print_(['Error cancelling order', oc, c.exception(), e], n=True)
        return
