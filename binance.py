import asyncio
import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

import aiohttp
import numpy as np

from passivbot import load_key_secret, print_, ts_to_date, Bot, sort_dict_keys, config_to_xk


async def create_bot(user: str, config: str):
    bot = BinanceBot(user, config)
    await bot._init()
    return bot


class BinanceBot(Bot):
    def __init__(self, user: str, config: dict):
        self.exchange = 'binance'
        super().__init__(user, config)
        self.max_pos_size_ito_usdt = 0.0
        self.max_pos_size_ito_coin = 0.0
        self.session = aiohttp.ClientSession()
        self.base_endpoint = ''
        self.key, self.secret = load_key_secret('binance', user)

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
        async with getattr(self.session, type_)(self.base_endpoint + url, params=params,
                                                headers=headers) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}) -> dict:
        return await self.private_('get', url, params)

    async def private_post(self, url: str, params: dict = {}) -> dict:
        return await self.private_('post', url, params)

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        return await self.private_('delete', url, params)

    def init_market_type(self):
        if self.symbol.endswith('USDT'):
            print('linear perpetual')
            self.market_type = 'linear_perpetual'
            self.inverse = self.config['inverse'] = False
            self.base_endpoint = 'https://fapi.binance.com'
            self.endpoints = {
                'position': '/fapi/v2/positionRisk',
                'balance': '/fapi/v2/balance',
                'exchange_info': '/fapi/v1/exchangeInfo',
                'leverage_bracket': '/fapi/v1/leverageBracket',
                'open_orders': '/fapi/v1/openOrders',
                'ticker': '/fapi/v1/ticker/bookTicker',
                'create_order': '/fapi/v1/order',
                'cancel_order': '/fapi/v1/order',
                'ticks': '/fapi/v1/aggTrades',
                'margin_type': '/fapi/v1/marginType',
                'leverage': '/fapi/v1/leverage',
                'position_side': '/fapi/v1/positionSide/dual',
                'websocket': f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
            }

        else:
            print('inverse coin margined')
            self.base_endpoint = 'https://dapi.binance.com'
            self.market_type = 'inverse_coin_margined'
            self.inverse = self.config['inverse'] = True
            self.endpoints = {
                'position': '/dapi/v1/positionRisk',
                'balance': '/dapi/v1/balance',
                'exchange_info': '/dapi/v1/exchangeInfo',
                'leverage_bracket': '/dapi/v1/leverageBracket',
                'open_orders': '/dapi/v1/openOrders',
                'ticker': '/dapi/v1/ticker/bookTicker',
                'create_order': '/dapi/v1/order',
                'cancel_order': '/dapi/v1/order',
                'ticks': '/dapi/v1/aggTrades',
                'margin_type': '/dapi/v1/marginType',
                'leverage': '/dapi/v1/leverage',
                'position_side': '/dapi/v1/positionSide/dual',
                'websocket': f"wss://dstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
            }

    async def _init(self):
        self.init_market_type()
        exchange_info, leverage_bracket = await asyncio.gather(
            self.public_get(self.endpoints['exchange_info']),
            self.private_get(self.endpoints['leverage_bracket']),
        )
        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.coin = e['baseAsset']
                self.quot = e['quoteAsset']
                self.margin_coin = e['marginAsset']
                self.pair = e['pair']
                if self.market_type == 'inverse_coin_margined':
                    self.contract_multiplier = self.config['contract_multiplier'] = \
                        float(e['contractSize'])
                price_precision = e['pricePrecision']
                qty_precision = e['quantityPrecision']
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
                    self.min_cost = 0.0
                break
        max_lev = 10
        for e in leverage_bracket:
            if ('pair' in e and e['pair'] == self.pair) or \
                    ('symbol' in e and e['symbol'] == self.symbol):
                for br in e['brackets']:
                    max_lev = max(max_lev, int(br['initialLeverage']))
                break
        self.max_leverage = max_lev
        self.xk = config_to_xk(self.config)
        await self.init_order_book()
        await self.update_position()

    async def check_if_other_positions(self, abort=True):
        positions, open_orders = await asyncio.gather(
            self.private_get(self.endpoints['position']),
            self.private_get(self.endpoints['open_orders'])
        )
        do_abort = False
        for e in positions:
            if float(e['positionAmt']) != 0.0:
                if e['symbol'] != self.symbol:
                    print('\n\nWARNING\n\n')
                    print('account has position in other symbol:', e)
                    print('\n\n')
                    do_abort = True
        for e in open_orders:
            if e['symbol'] != self.symbol:
                print('\n\nWARNING\n\n')
                print('account has open orders in other symbol:', e)
                print('\n\n')
                do_abort = True
        if do_abort:
            if abort:
                raise Exception('please close other positions and cancel other open orders')
        else:
            print('no positions or open orders in other symbols sharing margin wallet')

    async def init_exchange_config(self):
        try:
            print(await self.private_post(self.endpoints['margin_type'],
                                          {'symbol': self.symbol, 'marginType': 'CROSSED'}))
        except Exception as e:
            print(e)
        try:
            lev = await self.private_post(self.endpoints['leverage'],
                                          {'symbol': self.symbol, 'leverage': int(round(self.leverage))})
            print(lev)
            if self.market_type == 'linear_perpetual':
                self.max_pos_size_ito_usdt = float(lev['maxNotionalValue'])
                print('max pos size in terms of usdt', self.max_pos_size_ito_usdt)
            elif self.market_type == 'inverse_coin_margined':
                self.max_pos_size_ito_coin = float(lev['maxQty'])
                print('max pos size in terms of coin', self.max_pos_size_ito_coin)

        except Exception as e:
            print(e)
        try:
            res = await self.private_post(self.endpoints['position_side'],
                                          {'dualSidePosition': 'true'})
            print(res)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print(e)
                print('unable to set hedge mode, aborting')
                raise Exception('failed to set hedge mode')
        await self.check_if_other_positions()
        await self.init_ema()

    async def init_order_book(self):
        ticker = await self.public_get(self.endpoints['ticker'], {'symbol': self.symbol})
        if self.market_type == 'inverse_coin_margined':
            ticker = ticker[0]
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
             'position_side': e['positionSide'].lower().replace('short', 'shrt'),
             'timestamp': int(e['time'])}
            for e in await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        ]

    async def fetch_position(self) -> dict:
        positions, balance = await asyncio.gather(
            self.private_get(self.endpoints['position'], ({'symbol': self.symbol}
                                                          if self.market_type == 'linear_perpetual'
                                                          else {'pair': self.pair})),
            self.private_get(self.endpoints['balance'], {})
        )
        position = {}
        if positions:
            for p in positions:
                if p['positionSide'] == 'LONG':
                    position['long'] = {'size': float(p['positionAmt']),
                                        'price': float(p['entryPrice']),
                                        'liquidation_price': float(p['liquidationPrice']),
                                        'upnl': float(p['unRealizedProfit']),
                                        'leverage': float(p['leverage'])}
                elif p['positionSide'] == 'SHORT':
                    position['shrt'] = {'size': float(p['positionAmt']),
                                        'price': float(p['entryPrice']),
                                        'liquidation_price': float(p['liquidationPrice']),
                                        'upnl': float(p['unRealizedProfit']),
                                        'leverage': float(p['leverage'])}
        for e in balance:
            if e['asset'] == (self.quot if self.market_type == 'linear_perpetual' else self.coin):
                position['wallet_balance'] = float(e['balance'])
                position['equity'] = position['wallet_balance'] + float(e['crossUnPnl'])
                break
        return position

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
            fetched = await self.private_get(self.endpoints['ticks'], params)
        except Exception as e:
            print('error fetching ticks a', e)
            return []
        try:
            ticks = [{'trade_id': int(t['a']), 'price': float(t['p']), 'qty': float(t['q']),
                      'timestamp': int(t['T']), 'is_buyer_maker': t['m']}
                     for t in fetched]
            if do_print:
                print_(['fetched trades', self.symbol, ticks[0]['trade_id'],
                        ts_to_date(float(ticks[0]['timestamp']) / 1000)])
        except Exception as e:
            print('errer fetching ticks b', e, fetched)
            ticks = []
            if do_print:
                print_(['fetched no new ticks', self.symbol])
        return ticks

    async def fetch_ticks_time(self, start_time: int, end_time: int = None, do_print: bool = True):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    def calc_max_pos_size(self, balance: float, price: float):
        if self.market_type == 'linear_perpetual':
            return min((balance / price) * self.leverage, self.max_pos_size_ito_usdt / price) * 0.92
        elif self.market_type == 'inverse_coin_margined':
            return min((balance * price) * self.leverage, self.max_pos_size_ito_coin * price) * 0.92

    def standardize_websocket_ticks(self, data: dict) -> [dict]:
        try:
            ticks = [{'price': float(data['p']), 'is_buyer_maker': data['m']}]
            if ticks[0]['price'] != self.price:
                return ticks
        except Exception as e:
            print('errer in websocket tick', e)
        return []

    async def subscribe_ws(self, ws):
        pass
