import asyncio
import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

import aiohttp
import numpy as np
import traceback

from pure_funcs import ts_to_date, sort_dict_keys, calc_long_pprice, format_float, get_position_fills, spotify_config
from njit_funcs import round_dn, round_up
from passivbot import Bot
from procedures import print_


class BinanceBotSpot(Bot):
    def __init__(self, config: dict):
        self.exchange = 'binance_spot'
        self.balance = {}
        super().__init__(spotify_config(config))
        self.spot = self.config['spot'] = True
        self.inverse = self.config['inverse'] = False
        self.hedge_mode = self.config['hedge_mode'] = False
        self.do_long = self.config['do_long'] = self.config['long']['enabled'] = True
        self.do_shrt = self.config['do_shrt'] = self.config['shrt']['enabled'] = False
        self.session = aiohttp.ClientSession()
        self.base_endpoint = ''

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(self, type_: str, base_endpoint: str, url: str, params: dict = {}) -> dict:
        timestamp = int(time() * 1000)
        params.update({'timestamp': timestamp, 'recvWindow': 5000})
        for k in params:
            if type(params[k]) == bool:
                params[k] = 'true' if params[k] else 'false'
            elif type(params[k]) == float:
                params[k] = format_float(params[k])
        params = sort_dict_keys(params)
        params['signature'] = hmac.new(self.secret.encode('utf-8'),
                                       urlencode(params).encode('utf-8'),
                                       hashlib.sha256).hexdigest()
        headers = {'X-MBX-APIKEY': self.key}
        async with getattr(self.session, type_)(base_endpoint + url, params=params,
                                                headers=headers) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        if base_endpoint is not None:
            return await self.private_('get', base_endpoint, url, params)
        else:
            return await self.private_('get', self.base_endpoint, url, params)

    async def private_post(self, base_endpoint: str, url: str, params: dict = {}) -> dict:
        return await self.private_('post', base_endpoint, url, params)

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        return await self.private_('delete', self.base_endpoint, url, params)

    def init_market_type(self):
        print('spot market')
        if 'spot' not in self.market_type:
            self.market_type += '_spot'
        self.inverse = self.config['inverse'] = False
        self.spot = True
        self.hedge_mode = False
        self.base_endpoint = 'https://api.binance.com'
        self.pair = self.symbol
        self.endpoints = {
            'balance': '/api/v3/account',
            'exchange_info': '/api/v3/exchangeInfo',
            'open_orders': '/api/v3/openOrders',
            'ticker': '/api/v3/ticker/bookTicker',
            'fills': '/api/v3/myTrades',
            'create_order': '/api/v3/order',
            'cancel_order': '/api/v3/order',
            'ticks': '/api/v3/aggTrades',
            'ohlcvs': '/api/v3/klines',
            'websocket': f"wss://stream.binance.com/ws/{self.symbol.lower()}@aggTrade"
        }
        self.endpoints['transfer'] = '/sapi/v1/asset/transfer'
        self.endpoints['account'] = '/api/v3/account'

    async def _init(self):
        self.init_market_type()
        exchange_info = await self.public_get(self.endpoints['exchange_info'])
        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.coin = e['baseAsset']
                self.quot = self.margin_coin = e['quoteAsset']
                for q in e['filters']:
                    if q['filterType'] == 'LOT_SIZE':
                        self.min_qty = self.config['min_qty'] = float(q['minQty'])
                        self.qty_step = self.config['qty_step'] = float(q['stepSize'])
                    elif q['filterType'] == 'PRICE_FILTER':
                        self.price_step = self.config['price_step'] = float(q['tickSize'])
                        self.min_price = float(q['minPrice'])
                        self.max_price = float(q['maxPrice'])
                    elif q['filterType'] == 'PERCENT_PRICE':
                        self.price_multiplier_up = float(q['multiplierUp'])
                        self.price_multiplier_dn = float(q['multiplierDown'])
                    elif q['filterType'] == 'MIN_NOTIONAL':
                        self.min_cost = self.config['min_cost'] = float(q['minNotional'])
                try:
                    z = self.min_cost
                except AttributeError:
                    self.min_cost = self.config['min_cost'] = 0.0
                break

        await super()._init()
        await self.init_order_book()
        await self.update_position()

    def calc_orders(self):
        default_orders = super().calc_orders()
        orders = []
        for order in sorted(default_orders, key=lambda x: x['price']):
            if order['price'] > min(self.max_price, round_dn(self.price * self.price_multiplier_up, self.price_step)):
                print(f'price {order["price"]} too high')
                continue
            if order['price'] < max(self.min_price, round_up(self.price * self.price_multiplier_dn, self.price_step)):
                print(f'price {order["price"]} too low')
                continue
            orders.append(order)
        sum_buy_cost = sum([o['qty'] * o['price'] for o in orders if o['side'] == 'buy'])
        excess_cost = max(0.0, sum_buy_cost - self.balance[self.quot]['onhand'])
        if excess_cost:
            orders[0]['qty'] = round_dn((orders[0]['qty'] * orders[0]['price'] - excess_cost) / orders[0]['price'], self.qty_step)
            if orders[0]['qty'] < max(self.min_qty, self.min_cost / orders[0]['price']):
                orders = orders[1:]
        return orders


    async def check_if_other_positions(self, abort=True):
        pass
        # todo...

    async def execute_leverage_change(self):
        pass

    async def init_exchange_config(self):
        await self.check_if_other_positions()

    async def init_order_book(self):
        ticker = await self.public_get(self.endpoints['ticker'], {'symbol': self.symbol})
        self.ob = [float(ticker['bidPrice']), float(ticker['askPrice'])]
        self.price = np.random.choice(self.ob)

    async def fetch_open_orders(self) -> [dict]:
        return [
            {'order_id': int(e['orderId']),
             'symbol': e['symbol'],
             'price': float(e['price']),
             'qty': float(e['origQty']),
             'type': e['type'].lower(),
             'side': e['side'].lower(),
             'position_side': 'long',
             'timestamp': int(e['time'])}
            for e in await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        ]

    async def fetch_position(self) -> dict:
        balances, new_fills = await asyncio.gather(self.private_get(self.endpoints['balance']),
                                                   self.update_fills())
        balance = {}
        for elm in balances['balances']:
            for k in [self.quot, self.coin]:
                if elm['asset'] == k:
                    balance[k] = {'free': float(elm['free'])}
                    balance[k]['locked'] = float(elm['locked'])
                    balance[k]['onhand'] = balance[k]['free'] + balance[k]['locked']
                    break
            if self.quot in balance and self.coin in balance:
                break
        long_psize = round_dn(balance[self.coin]['onhand'], self.qty_step)
        self.long_pfills, self.shrt_pfills = get_position_fills(long_psize, 0.0, self.fills)
        long_pprice = calc_long_pprice(long_psize, self.long_pfills) if long_psize else 0.0
        if long_psize * long_pprice < self.min_cost:
            long_psize, long_pprice, self.long_pfills = 0.0, 0.0, []
        position = {'long': {'size': long_psize,
                             'price': long_pprice,
                             'liquidation_price': 0.0,
                             'upnl': long_psize * (self.price - long_pprice),
                             'leverage': 1.0},
                    'shrt': {'size': 0.0,
                             'price': 0.0,
                             'liquidation_price': 0.0,
                             'upnl': 0.0,
                             'leverage': 0.0},
                    'wallet_balance': balance[self.quot]['onhand'] + balance[self.coin]['onhand'] * long_pprice,
                    'equity': balance[self.quot]['onhand'] + balance[self.coin]['onhand'] * self.price}
        self.balance = balance
        return position

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'type': order['type'].upper(),
                  'quantity': format_float(order['qty'])}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTC'
            params['price'] = format_float(order['price'])
        if 'custom_id' in order:
            params['newClientOrderId'] = \
                f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        o = await self.private_post(self.base_endpoint, self.endpoints['create_order'], params)
        if 'side' in o:
            return {'symbol': self.symbol,
                    'side': o['side'].lower(),
                    'position_side': 'long',
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
                    'position_side': 'long',
                    'qty': float(cancellation['origQty']), 'price': float(cancellation['price'])}
        else:
            return cancellation

    async def fetch_fills(self, limit: int = 1000, from_id: int = None, start_time: int = None, end_time: int = None):
        params = {'symbol': self.symbol, 'limit': min(1000, max(500, limit))}
        if from_id is not None:
            params['fromId'] = max(0, from_id)
        if start_time is not None:
            params['startTime'] = int(start_time)
        if end_time is not None:
            params['endTime'] = int(min(end_time, start_time + 1000 * 60 * 60 * 23.99))
        try:
            fetched = await self.private_get(self.endpoints['fills'], params)
            fills = [{'symbol': x['symbol'],
                      'id': int(x['id']),
                      'order_id': int(x['orderId']),
                      'side': 'buy' if x['isBuyer'] else 'sell',
                      'price': float(x['price']),
                      'qty': float(x['qty']),
                      'realized_pnl': 0.0,
                      'cost': float(x['quoteQty']),
                      'fee_paid': float(x['commission']),
                      'fee_token': x['commissionAsset'],
                      'timestamp': int(x['time']),
                      'position_side': 'long',
                      'is_maker': x['isMaker']} for x in fetched]
        except Exception as e:
            print('error fetching fills a', e)
            traceback.print_exc()
            return []
        return fills

    async def fetch_income(self, limit: int = 1000, start_time: int = None, end_time: int = None):
        print('fetch income not implemented in spot')
        return []

    async def fetch_account(self):
        try:
            return await self.private_get(self.endpoints['balance'])
        except Exception as e:
            print('error fetching account: ', e)
            return {'balances': []}

    async def fetch_ticks(self, from_id: int = None, start_time: int = None, end_time: int = None,
                          do_print: bool = True):
        params = {'symbol': self.symbol, 'limit': 1000}
        if from_id is not None:
            params['fromId'] = max(0, from_id)
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        try:
            fetched = await self.public_get(self.endpoints['ticks'], params)
        except Exception as e:
            print('error fetching ticks a', e)
            return []
        try:
            ticks = [{'trade_id': int(t['a']), 'price': float(t['p']), 'qty': float(t['q']),
                      'timestamp': int(t['T']), 'is_buyer_maker': t['m']}
                     for t in fetched]
            if do_print:
                print_(['fetched ticks', self.symbol, ticks[0]['trade_id'],
                        ts_to_date(float(ticks[0]['timestamp']) / 1000)])
        except Exception as e:
            print('error fetching ticks b', e, fetched)
            ticks = []
            if do_print:
                print_(['fetched no new ticks', self.symbol])
        return ticks

    async def fetch_ticks_time(self, start_time: int, end_time: int = None, do_print: bool = True):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    async def fetch_ohlcvs(self, start_time: int = None, interval='1m', limit=1000):
        # m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
        interval_map = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360,
                        '12h': 720, '1d': 60 * 60 * 24, '1w': 60 * 60 * 24 * 7, '1M': 60 * 60 * 24 * 30}
        assert interval in interval_map
        params = {'symbol': self.symbol, 'interval': interval, 'limit': limit}
        if start_time is not None:
            params['startTime'] = int(start_time)
            params['endTime'] = params['startTime'] + interval_map[interval] * 60 * 1000 * limit
        try:
            fetched = await self.public_get(self.endpoints['ohlcvs'], params)
            return [{**{'timestamp': int(e[0])},
                     **{k: float(e[i + 1]) for i, k in enumerate(['open', 'high', 'low', 'close', 'volume'])}}
                    for e in fetched]
        except Exception as e:
            print('error fetching ohlcvs', fetched, e)
            traceback.print_exc()

    async def transfer(self, type_: str, amount: float, asset: str = 'USDT'):
        print('transfer not implemented in spot')
        return

    def standardize_websocket_ticks(self, data: dict) -> [dict]:
        try:
            return [{'timestamp': int(data['T']), 'price': float(data['p']), 'qty': float(data['q']),
                     'is_buyer_maker': data['m']}]
        except Exception as e:
            print('error in websocket tick', e)
        return []

    async def subscribe_ws(self, ws):
        pass
