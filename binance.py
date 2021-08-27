import asyncio
import hashlib
import hmac
import json
from time import time
from urllib.parse import urlencode

import aiohttp
import numpy as np
import traceback

from pure_funcs import ts_to_date, sort_dict_keys
from passivbot import Bot
from procedures import print_


class BinanceBot(Bot):
    def __init__(self, config: dict):
        self.exchange = 'binance'
        super().__init__(config)
        self.max_pos_size_ito_usdt = 0.0
        self.max_pos_size_ito_coin = 0.0
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
                params[k] = str(params[k])
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

    async def init_market_type(self):
        fapi_endpoint = 'https://fapi.binance.com'
        dapi_endpoint = 'https://dapi.binance.com'
        fapi_info = await self.private_get('/fapi/v1/exchangeInfo', base_endpoint=fapi_endpoint)
        if self.symbol in {e['symbol'] for e in fapi_info['symbols']}:
            print('linear perpetual')
            self.market_type += '_linear_perpetual'
            self.inverse = self.config['inverse'] = False
            self.base_endpoint = fapi_endpoint
            self.endpoints = {
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
                'ohlcvs': '/fapi/v1/klines',
                'margin_type': '/fapi/v1/marginType',
                'leverage': '/fapi/v1/leverage',
                'position_side': '/fapi/v1/positionSide/dual',
                'websocket': f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
            }
        else:
            dapi_info = await self.private_get('/dapi/v1/exchangeInfo', base_endpoint=dapi_endpoint)
            if self.symbol in {e['symbol'] for e in dapi_info['symbols']}:
                print('inverse coin margined')
                self.base_endpoint = dapi_endpoint
                self.market_type += '_inverse_coin_margined'
                self.inverse = self.config['inverse'] = True
                self.endpoints = {
                    'position': '/dapi/v1/positionRisk',
                    'balance': '/dapi/v1/balance',
                    'exchange_info': '/dapi/v1/exchangeInfo',
                    'leverage_bracket': '/dapi/v1/leverageBracket',
                    'open_orders': '/dapi/v1/openOrders',
                    'ticker': '/dapi/v1/ticker/bookTicker',
                    'fills': '/dapi/v1/userTrades',
                    'income': '/dapi/v1/income',
                    'create_order': '/dapi/v1/order',
                    'cancel_order': '/dapi/v1/order',
                    'ticks': '/dapi/v1/aggTrades',
                    'ohlcvs': '/dapi/v1/klines',
                    'margin_type': '/dapi/v1/marginType',
                    'leverage': '/dapi/v1/leverage',
                    'position_side': '/dapi/v1/positionSide/dual',
                    'websocket': f"wss://dstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
                }
            else:
                raise Exception(f'unknown symbol {self.symbol}')

        self.spot_base_endpoint = 'https://api.binance.com'
        self.endpoints['transfer'] = '/sapi/v1/asset/transfer'
        self.endpoints['account'] = '/api/v3/account'

    async def _init(self):
        await self.init_market_type()
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
                if 'inverse_coin_margined' in self.market_type:
                    self.c_mult = self.config['c_mult'] = \
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
                    self.min_cost = self.config['min_cost'] = 0.0
                break
        max_lev = 25 # lowest max lev for any binance futures symbol, as per 2021-06-12
        for e in leverage_bracket:
            if ('pair' in e and e['pair'] == self.pair) or \
                    ('symbol' in e and e['symbol'] == self.symbol):
                for br in e['brackets']:
                    max_lev = max(max_lev, int(br['initialLeverage']))
                break
        self.max_leverage = self.config['max_leverage'] = max_lev
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def check_if_other_positions(self):
        positions, open_orders = await asyncio.gather(
            self.private_get(self.endpoints['position']),
            self.private_get(self.endpoints['open_orders'])
        )
        do_abort = False
        for e in positions:
            if float(e['positionAmt']) != 0.0:
                if e['symbol'] != self.symbol and self.margin_coin in e['symbol']:
                    print('\n\nWARNING\n\n')
                    print('account has position in other symbol:', e)
                    print('\n\n')
                    do_abort = True
        for e in open_orders:
            if e['symbol'] != self.symbol and self.margin_coin in e['symbol']:
                print('\n\nWARNING\n\n')
                print('account has open orders in other symbol:', e)
                print('\n\n')
                do_abort = True
        if do_abort:
            if not ('allow_sharing_wallet' in self.config and self.config['allow_sharing_wallet']):
                print('please close other positions and cancel other open orders '
                      'or add "allow_sharing_wallet": True to config')
                self.stop()
                return True
        else:
            print('no positions or open orders in other symbols sharing margin wallet')
        return False

    async def execute_leverage_change(self):
        lev = int(min(self.max_leverage, max(3.0, np.ceil(max(self.xk['pbr_limit']) * 3))))
        return await self.private_post(self.base_endpoint,
                                       self.endpoints['leverage'],
                                       {'symbol': self.symbol, 'leverage': lev})

    async def init_exchange_config(self):
        try:
            print(await self.private_post(self.base_endpoint,
                                          self.endpoints['margin_type'],
                                          {'symbol': self.symbol, 'marginType': 'CROSSED'}))
        except Exception as e:
            print(e)
        try:
            lev = await self.execute_leverage_change()
            print_([lev])
            if 'linear_perpetual' in self.market_type:
                self.max_pos_size_ito_usdt = float(lev['maxNotionalValue'])
                print('max pos size in terms of usdt', self.max_pos_size_ito_usdt)
            elif 'inverse_coin_margined' in self.market_type:
                self.max_pos_size_ito_coin = float(lev['maxQty'])
                print('max pos size in terms of coin', self.max_pos_size_ito_coin)

        except Exception as e:
            print(e)
        try:
            res = await self.private_post(self.base_endpoint,
                                          self.endpoints['position_side'],
                                          {'dualSidePosition': 'true'})
            print(res)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print(e)
                print('unable to set hedge mode, aborting')
                raise Exception('failed to set hedge mode')
        return await self.check_if_other_positions()

    async def init_order_book(self):
        ticker = await self.public_get(self.endpoints['ticker'], {'symbol': self.symbol})
        if 'inverse_coin_margined' in self.market_type:
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
                                                          if 'linear_perpetual' in self.market_type
                                                          else {'pair': self.pair})),
            self.private_get(self.endpoints['balance'], {})
        )
        positions = [e for e in positions if e['symbol'] == self.symbol]
        position = {'long': {'size': 0.0, 'price': 0.0, 'liquidation_price': 0.0, 'upnl': 0.0, 'leverage': 0.0},
                    'shrt': {'size': 0.0, 'price': 0.0, 'liquidation_price': 0.0, 'upnl': 0.0, 'leverage': 0.0},
                    'wallet_balance': 0.0, 'equity': 0.0}
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
            if e['asset'] == (self.quot if 'linear_perpetual' in self.market_type else self.coin):
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
        o = await self.private_post(self.base_endpoint, self.endpoints['create_order'], params)
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

    async def fetch_fills(self, limit: int = 1000, from_id: int = None, start_time: int = None, end_time: int = None):
        params = {'symbol': self.symbol, 'limit': min(100, limit) if self.inverse else limit}
        if from_id is not None:
            params['fromId'] = max(0, from_id)
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        try:
            fetched = await self.private_get(self.endpoints['fills'], params)
            fills = [{'symbol': x['symbol'],
                      'id': int(x['id']),
                      'order_id': int(x['orderId']),
                      'side': x['side'].lower(),
                      'price': float(x['price']),
                      'qty': float(x['qty']),
                      'realized_pnl': float(x['realizedPnl']),
                      'cost': float(x['baseQty']) if self.inverse else float(x['quoteQty']),
                      'fee_paid': float(x['commission']),
                      'fee_token': x['commissionAsset'],
                      'timestamp': int(x['time']),
                      'position_side': x['positionSide'].lower().replace('short', 'shrt'),
                      'is_maker': x['maker']} for x in fetched]
        except Exception as e:
            print('error fetching fills a', e)
            return []
        return fills

    async def fetch_income(self, limit: int = 1000, start_time: int = None, end_time: int = None):
        params = {'symbol': self.symbol, 'limit': limit}
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        try:
            fetched = await self.private_get(self.endpoints['income'], params)
            income = [{'symbol': x['symbol'],
                      'incomeType': x['incomeType'],
                      'income': float(x['income']),
                      'asset': x['asset'],
                      'info': x['info'],
                      'timestamp': int(x['time']),
                      'tranId': x['tranId'],
                      'tradeId': x['tradeId']} for x in fetched]
        except Exception as e:
            print('error fetching incoming: ', e)
            return []
        return income

    async def fetch_account(self):
        try:
            return await self.private_get(base_endpoint=self.spot_base_endpoint, url=self.endpoints['account'])
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
            fetched = await self.private_get(self.endpoints['ticks'], params)
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

    async def fetch_ohlcvs(self, start_time: int = None, interval='1m', limit=1500):
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
        params = {'type': type_.upper(), 'amount': amount, 'asset': asset}
        return await self.private_post(self.spot_base_endpoint, self.endpoints['transfer'],  params)

    def standardize_websocket_ticks(self, data: dict) -> [dict]:
        try:
            return [{'timestamp': int(data['T']), 'price': float(data['p']), 'qty': float(data['q']),
                     'is_buyer_maker': data['m']}]
        except Exception as e:
            print('error in websocket tick', e, data)
        return []

    async def subscribe_ws(self, ws):
        pass
