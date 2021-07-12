import asyncio
import hashlib
import hmac
import json
from time import time
from typing import Union, Tuple, List
from urllib.parse import urlencode

from bots.base_bot import Bot, ORDER_UPDATE, ACCOUNT_UPDATE
from definitions.order import Order
from definitions.order import TP, SL, LIMIT, MARKET, LQ, NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, TRADE, \
    CALCULATED, BUY, SELL, LONG, SHORT, BOTH, NEW_INSURANCE, NEW_ADL
from definitions.position import Position
from functions import sort_dict_keys, print_, print_order

order_mapping = {'BUY': BUY, 'SELL': SELL, 'MARKET': MARKET, 'LIMIT': LIMIT, 'STOP': SL, 'TAKE_PROFIT': TP,
                 'LIQUIDATYION': LQ, 'NEW': NEW, 'CANCELED': CANCELED, 'CALCULATED': CALCULATED, 'EXPIRED': EXPIRED,
                 'TRADE': TRADE, 'PARTIALLY_FILLED': PARTIALLY_FILLED, 'FILLED': FILLED, 'LONG': LONG, 'SHORT': SHORT,
                 'BOTH': BOTH, 'NEW_INSURANCE': NEW_INSURANCE, 'NEW_ADL': NEW_ADL}

reverse_order_mapping = {TP: 'TAKE_PROFIT', SL: 'STOP_LOSS', LIMIT: 'LIMIT', MARKET: 'MARKET', BUY: 'BUY', SELL: 'SELL',
                         LONG: 'LONG', SHORT: 'SHORT'}


def mapping(item):
    try:
        return order_mapping[item.upper()]
    except Exception as e:
        print('Could not map', e)
        return ''


def reverse_mapping(item):
    try:
        return reverse_order_mapping[item]
    except Exception as e:
        print('Could not map', e)
        return ''


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

    async def fetch_orders(self) -> List[Order]:
        ords = await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        # orders = {'LONG': [], 'SHORT': []}
        orders = []
        for o in ords:
            if o['symbol'] == self.symbol:
                order = Order(o['symbol'].upper(),
                              int(o['orderId']),
                              float(o['price']),
                              float(o['stopPrice']),
                              float(o['origQty']),
                              mapping(o['type']),
                              mapping(o['side']),
                              int(o['time']),
                              mapping(o['status']),
                              mapping(o['positionSide']))
                if order.position_side == LONG or order.position_side == SHORT:
                    orders.append(order)
                else:
                    print_([o], n=True)
        return orders

    async def fetch_position(self) -> Tuple[Position, Position]:
        pos = await self.private_get(self.endpoints['position'], ({'symbol': self.symbol}))
        long = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        short = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        for p in pos:
            if p['symbol'] == self.symbol:
                position = Position(p['symbol'].upper(),
                                    float(p['positionAmt']),
                                    float(p['entryPrice']),
                                    float(p['liquidationPrice']),
                                    float(p['unRealizedProfit']),
                                    int(p['leverage']),
                                    mapping(p['positionSide']))
                if position.position_side == LONG and position.size != 0.0:
                    long = position
                elif position.position_side == SHORT and position.size != 0.0:
                    short = position
        return long, short

    async def fetch_balance(self) -> float:
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

    def prepare_order(self, msg) -> Order:
        order = Order(msg['o']['s'].upper(),
                      int(msg['o']['i']),
                      float(msg['o']['p']),
                      0.0,
                      float(msg['o']['q']),
                      mapping(msg['o']['o']),
                      mapping(msg['o']['S']),
                      int(msg['o']['T']),
                      mapping(msg['o']['X']),
                      mapping(msg['o']['ps']))
        if 'ot' in msg['o']:
            if mapping(msg['o']['ot']) == MARKET and order.action != PARTIALLY_FILLED:
                order.price = float(msg['o']['ap'])
        if order.action == PARTIALLY_FILLED:
            order.qty = order.qty - float(msg['o']['z'])
        return order

    def prepare_account(self, msg) -> Tuple[float, Position, Position]:
        balance = None
        last_long = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        last_short = Position('', 0.0, 0.0, 0.0, 0.0, 0, '')
        for b in msg['a']['B']:
            if b['a'].upper() == self.quote_asset:
                balance = float(b['wb'])
                break
        if 'P' in msg['a']:
            for p in msg['a']['P']:
                if p['s'] == self.symbol:
                    position = Position(p['s'].upper(),
                                        float(p['pa']),
                                        float(p['ep']),
                                        0.0,
                                        float(p['up']),
                                        self.leverage,
                                        mapping(p['ps']))
                    if position.position_side == LONG:
                        last_long = position
                    if position.position_side == SHORT:
                        last_short = position
        return balance, last_long, last_short

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
                    type = ORDER_UPDATE
            elif msg['e'] == 'ACCOUNT_UPDATE':
                type = ACCOUNT_UPDATE
        return type

    async def execute_leverage_change(self):
        return await self.private_post(self.endpoints['leverage'],
                                       {'symbol': self.symbol, 'leverage': int(self.config['leverage'])})

    async def execute_order(self, order: Order) -> Union[dict, bool]:
        params = {'symbol': order.symbol,
                  'side': reverse_mapping(order.side),
                  'positionSide': reverse_mapping(order.position_side),
                  'type': reverse_mapping(order.type),
                  'quantity': str(order.qty)}
        if params['type'] == LIMIT:
            params['timeInForce'] = 'GTX'
            params['price'] = str(order.price)
        if params['type'] == TP:
            params['price'] = str(order.price)
            params['stopPrice'] = str(order.stop_price)
        o = await self.private_post(self.endpoints['create_order'], params)
        if 'code' in o:
            return o
        else:
            return True

    async def execute_cancellation(self, order: Order) -> Union[dict, bool]:
        c = await self.private_delete(self.endpoints['cancel_order'],
                                      {'symbol': order.symbol, 'orderId': order.order_id})
        if 'code' in c:
            return c
        else:
            return True

    async def create_orders(self, orders_to_create: List[Order]):
        if not orders_to_create:
            return
        creations = []
        for order in orders_to_create:
            try:
                order = self.correct_float_precision(order)
                creations.append((order, asyncio.create_task(self.execute_order(order))))
            except Exception as e:
                print_(['Error creating order', print_order(order), e], n=True)
        for order, c in creations:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error creating order', print_order(order)], n=True)
                else:
                    print_(['Error creating order', print_order(order), o], n=True)
            except Exception as e:
                print_(['Error creating order', print_order(order), c.exception(), e], n=True)
        return

    async def cancel_orders(self, orders_to_cancel: List[Order]):
        if not orders_to_cancel:
            return
        deletions = []
        for order in orders_to_cancel:
            try:
                order = self.correct_float_precision(order)
                deletions.append((order, asyncio.create_task(self.execute_cancellation(order))))
            except Exception as e:
                print_(['Error cancelling order a', print_order(order), e], n=True)
        for order, c in deletions:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error cancelling order', print_order(order)], n=True)
                else:
                    print_(['Error cancelling order', print_order(order), o], n=True)
            except Exception as e:
                print_(['Error cancelling order', print_order(order), c.exception(), e], n=True)
        return
