import asyncio
import json
import websockets
import os
import sys
import numpy as np
import pandas as pd
import pprint
import datetime
import aiohttp
import hmac
import hashlib
from urllib.parse import urlencode
from math import ceil
from math import floor
from time import time, sleep
from typing import Callable, Iterator
from passivbot import load_key_secret, load_live_settings, make_get_filepath, print_, \
    ts_to_date, flatten, filter_orders, Bot, start_bot, round_up, round_dn, \
    calc_min_order_qty, sort_dict_keys, \
    iter_long_closes_linear, iter_shrt_closes_linear, calc_ema, iter_entries_linear, \
    iter_long_closes_inverse, iter_shrt_closes_inverse, calc_ema, iter_entries_inverse


async def create_bot(user: str, settings: str):
    bot = BinanceBot(user, settings)
    await bot._init()
    return bot


class BinanceBot(Bot):
    def __init__(self, user: str, settings: dict):
        self.exchange = 'binance'
        super().__init__(user, settings)
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
            self.base_endpoint = 'https://fapi.binance.com'
            self.endpoints = {'position': '/fapi/v2/positionRisk',
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
                              'websocket': 'wss://fstream.binance.com'}
            self.iter_long_closes = lambda balance, pos_size, pos_price, lowest_ask: \
                iter_long_closes_linear(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                        self.qty_pct, self.leverage, self.min_markup,
                                        self.markup_range, self.n_close_orders, balance, pos_size,
                                        pos_price, lowest_ask)
            self.iter_shrt_closes = lambda balance, pos_size, pos_price, highest_bid: \
                iter_shrt_closes_linear(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                        self.qty_pct, self.leverage, self.min_markup,
                                        self.markup_range, self.n_close_orders, balance, pos_size,
                                        pos_price, highest_bid)
            self.iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, \
                highest_bid, lowest_ask, last_price, do_long, do_shrt: \
                iter_entries_linear(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                    self.ddown_factor, self.qty_pct, self.leverage,
                                    self.grid_spacing, self.grid_coefficient, balance, long_psize,
                                    long_pprice, shrt_psize, shrt_pprice, highest_bid, lowest_ask,
                                    last_price, do_long, do_shrt)

        else:
            print('inverse perpetual')
            self.base_endpoint = 'https://dapi.binance.com'
            self.market_type = 'inverse_perpetual'
            self.endpoints = {'position': '/dapi/v1/positionRisk',
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
                              'websocket': 'wss://dstream.binance.com'}
            self.iter_long_closes = lambda balance, pos_size, pos_price, lowest_ask: \
                iter_long_closes_inverse(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                         self.qty_pct, self.leverage, self.min_markup,
                                         self.markup_range, self.n_close_orders, balance, pos_size,
                                         pos_price, lowest_ask)
            self.iter_shrt_closes = lambda balance, pos_size, pos_price, highest_bid: \
                iter_shrt_closes_inverse(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                         self.qty_pct, self.leverage, self.min_markup,
                                         self.markup_range, self.n_close_orders, balance, pos_size,
                                         pos_price, highest_bid)
            self.iter_entries = lambda balance, long_psize, long_pprice, shrt_psize, shrt_pprice, \
                highest_bid, lowest_ask, last_price, do_long, do_shrt: \
                iter_entries_inverse(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                     self.ddown_factor, self.qty_pct, self.leverage,
                                     self.grid_spacing, self.grid_coefficient, balance, long_psize,
                                     long_pprice, shrt_psize, shrt_pprice, highest_bid, lowest_ask,
                                     last_price, do_long, do_shrt)

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
                self.contract_size = float(e['contractSize']) \
                    if self.market_type == 'inverse_perpetual' else 1.0
                price_precision = e['pricePrecision']
                qty_precision = e['quantityPrecision']
                for q in e['filters']:
                    if q['filterType'] == 'LOT_SIZE':
                        self.min_qty = float(q['minQty'])
                    elif q['filterType'] == 'MARKET_LOT_SIZE':
                        self.qty_step = float(q['stepSize'])
                    elif q['filterType'] == 'PRICE_FILTER':
                        self.price_step = float(q['tickSize'])
                    elif q['filterType'] == 'MIN_NOTIONAL':
                        self.min_cost = float(q['notional'])
                try:
                    z = self.min_cost
                except AttributeError:
                    self.min_cost = 0.0
                self.calc_min_qty = lambda price_: \
                    max(self.min_qty, round_up(self.cost / price_, self.qty_step))
                self.calc_min_order_qty = lambda balance_, last_price: \
                    calc_min_order_qty(self.calc_min_qty(last_price),
                                       self.qty_step,
                                       (balance_ / last_price) * self.leverage,
                                       self.qty_pct)
                break
        max_lev = 0
        for e in leverage_bracket:
            if ('pair' in e and e['pair'] == self.pair) or \
                    ('symbol' in e and e['symbol'] ==self.symbol):
                for br in e['brackets']:
                    max_lev = max(max_lev, int(br['initialLeverage']))
                break
        self.max_leverage = max_lev
        await self.update_position()
        await self.init_order_book()
        await self.init_ema()

    async def init_ema(self):
        # fetch 10 tick chunks to initiate ema
        ticks = await self.fetch_ticks(do_print=False)
        additional_ticks = flatten(await asyncio.gather(
            *[self.fetch_ticks(from_id=ticks[0]['trade_id'] - len(ticks) * i, do_print=False)
              for i in range(1, 10)]
        ))
        ticks = sorted(ticks + additional_ticks, key=lambda x: x['trade_id'])
        ema = ticks[0]['price']
        for i in range(1, len(ticks)):
            if ticks[i]['price'] != ticks[i-1]['price']:
                ema = ema * self.ema_alpha_ + ticks[i]['price'] * self.ema_alpha
        self.ema = ema

    async def init_exchange_settings(self):
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
            elif self.market_type == 'inverse_perpetual':
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

    async def init_order_book(self):
        ticker = await self.public_get(self.endpoints['ticker'], {'symbol': self.symbol})
        if self.market_type == 'inverse_perpetual':
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
                                        'leverage': float(p['leverage'])}
                elif p['positionSide'] == 'SHORT':
                    position['shrt'] = {'size': float(p['positionAmt']),
                                        'price': float(p['entryPrice']),
                                        'liquidation_price': float(p['liquidationPrice']),
                                        'leverage': float(p['leverage'])}
        for e in balance:
            if e['asset'] == (self.quot if self.market_type == 'linear_perpetual' else self.coin):
                position['wallet_balance'] = float(e['balance']) / self.contract_size
                position['equity'] = position['wallet_balance'] + float(e['crossUnPnl']) / self.contract_size
                break
        return position

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'positionSide': order['position_side'].replace('shrt', 'short').upper(),
                  'type': order['type'].upper(),
                  'quantity': order['qty']}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = order['price']
        if 'custom_id' in order:
            params['newClientOrderId'] = \
                f"{order['custom_id']}_{int(time() * 1000)}_{int(np.random.random() * 1000)}"
        o = await self.private_post(self.endpoints['create_order'], params)
        return {'symbol': self.symbol,
                'side': o['side'].lower(),
                'position_side': o['positionSide'].lower().replace('short', 'shrt'),
                'type': o['type'].lower(),
                'qty': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_cancellation(self, order: dict) -> [dict]:
        cancellation = await self.private_delete(self.endpoints['cancel_order'],
                                                 {'symbol': self.symbol, 'orderId': order['order_id']})
        return {'symbol': self.symbol, 'side': cancellation['side'].lower(),
                'position_side': cancellation['positionSide'].lower().replace('short', 'shrt'),
                'qty': float(cancellation['origQty']), 'price': float(cancellation['price'])}

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
            print('error fetching ticks', e)
            return []
        try:
            ticks = [{'trade_id': int(t['a']), 'price': float(t['p']), 'qty': float(t['q']),
                      'timestamp': int(t['T']), 'is_buyer_maker': t['m']}
                     for t in fetched]
            if do_print:
                print_(['fetched trades', self.symbol, ticks[0]['trade_id'],
                        ts_to_date(float(ticks[0]['timestamp']) / 1000)])
        except Exception as e:
            print(e)
            ticks = []
            if do_print:
                print_(['fetched no new ticks', self.symbol])
        return ticks

    async def fetch_ticks_time(self, start_time: int, end_time: int = None, do_print: bool = True):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    def calc_max_pos_size(self, balance: float, price: float):
        if self.market_type == 'linear_perpetual':
            return min((balance / price) * self.leverage, self.max_pos_size_ito_usdt / price) * 0.92
        elif self.market_type == 'inverse_perpetual':
            return min((balance * price) * self.leverage, self.max_pos_size_ito_coin * price) * 0.92

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"{self.endpoints['websocket']}/ws/{self.symbol.lower()}@aggTrade"
        print_([uri])
        await self.update_position()
        await self.init_exchange_settings()
        k = 1
        async with websockets.connect(uri) as ws:
            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                price = float(data['p'])
                trade_id = data['a']
                if data['m']:
                    self.ob[0] = price
                else:
                    self.ob[1] = price
                if price != self.price:
                    self.ema = calc_ema(self.ema_alpha, self.ema_alpha_, self.ema, price)
                self.price = price
                if self.ts_locked['decide'] < self.ts_released['decide']:
                    asyncio.create_task(self.decide())
                if k % 10 == 0:
                    self.flush_stuck_locks()
                    k = 1
                if self.stop_websocket:
                    break
                k += 1


async def main() -> None:
    bot = await create_bot(sys.argv[1], load_live_settings('binance', sys.argv[1]))
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())

