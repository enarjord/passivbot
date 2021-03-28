import asyncio
import json
import websockets
import os
import sys
import numpy as np
import pandas as pd
import pprint
import hashlib
import hmac
from datetime import datetime
from math import ceil
from math import floor
from time import time, sleep
from typing import Callable, Iterator
from passivbot import init_ccxt, load_key_secret, load_live_settings, make_get_filepath, print_, \
    ts_to_date, flatten, Bot, start_bot, round_up, round_dn, \
    calc_min_order_qty_inverse, sort_dict_keys, calc_ema, iter_long_entries_inverse, \
    iter_shrt_entries_inverse, iter_long_closes_inverse, iter_shrt_closes_inverse, calc_diff
import aiohttp
from urllib.parse import urlencode


def first_capitalized(s: str):
    return s[0].upper() + s[1:].lower()


def calc_isolated_long_liq_price(balance,
                                 pos_size,
                                 pos_price,
                                 leverage,
                                 mm=0.005) -> float:
    return (pos_price * leverage) / (leverage + 1 - mm * leverage)


def calc_isolated_shrt_liq_price(balance,
                                 pos_size,
                                 pos_price,
                                 leverage,
                                 mm=0.005) -> float:
    return (pos_price * leverage) / (leverage - 1 + mm * leverage)


def determine_pos_side(o: dict) -> str:
    side = o['side'].lower()
    if side == 'buy':
        if 'entry' in o['order_link_id']:
            position_side = 'long'
        elif 'close' in o['order_link_id']:
            position_side = 'shrt'
        else:
            position_side = 'unknown'
    else:
        if 'entry' in o['order_link_id']:
            position_side = 'shrt'
        elif 'close' in o['order_link_id']:
            position_side = 'long'
        else:
            position_side = 'unknown'
    return position_side


def format_tick(tick: dict) -> dict:
    return {'trade_id': int(tick['id']),
            'price': float(tick['price']),
            'qty': float(tick['qty']),
            'timestamp': date_to_ts(tick['time']),
            'is_buyer_maker': tick['side'] == 'Sell'}


async def fetch_ticks(cc, symbol: str, from_id: int = None, do_print=True) -> [dict]:

    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['from'] = max(0, from_id)
    try:
        fetched_trades = await cc.v2_public_get_trading_records(params=params)
    except Exception as e:
        print(e)
        return []
    trades = [format_tick(t) for t in fetched_trades['result']]
    if do_print:
        print_(['fetched trades', symbol, trades[0]['trade_id'],
                ts_to_date(trades[0]['timestamp'] / 1000)])
    return trades

def date_to_ts(date: str):
    try:
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp() * 1000
    except ValueError:
        formats = ["%Y-%m-%dT%H:%M:%S%z"]
        for f in formats:
            try:
                return datetime.strptime(date, f).timestamp() * 1000
            except ValueError:
                continue
    raise Exception(f'unable to convert date {date} to timestamp')

async def create_bot(user: str, settings: str):
    bot = Bybit(user, settings)
    await bot._init()
    return bot


class Bybit(Bot):
    def __init__(self, user: str, settings: dict):
        self.exchange = 'bybit'
        self.min_notional = 0.0
        super().__init__(user, settings)
        self.key, self.secret = load_key_secret('bybit', user)
        self.base_endpoint = 'https://api.bybit.com'
        self.session = aiohttp.ClientSession()

    async def _init(self):
        info = await self.public_get('/v2/public/symbols')
        for e in info['result']:
            if e['name'] == self.symbol:
                break
        else:
            raise Exception('symbol missing')
        self.max_leverage = e['leverage_filter']['max_leverage']
        self.coin = e['base_currency']
        self.quot = e['quote_currency']
        self.price_step = float(e['price_filter']['tick_size'])
        self.qty_step = float(e['lot_size_filter']['qty_step'])
        self.min_qty = float(e['lot_size_filter']['min_trading_qty'])
        self.min_cost = 0.0
        self.calc_min_qty = lambda price_: self.min_qty
        self.calc_min_order_qty = lambda balance_, last_price: \
            calc_min_order_qty_inverse(self.qyt_step, self.min_qty, self.min_cost,
                                       self.qty_pct, self.leverage, balance_, last_price)
        await asyncio.gather(
            self.update_position(),
            self.init_order_book(),
            self.init_ema(),
        )
        self.iter_long_entries = lambda balance, long_psize, long_pprice, shrt_psize, highest_bid: \
            iter_long_entries_inverse(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                      self.ddown_factor, self.qty_pct, self.leverage,
                                      self.grid_spacing, self.grid_coefficient, balance, long_psize,
                                      long_pprice, shrt_psize, highest_bid)
        self.iter_shrt_entries = lambda balance, long_psize, shrt_psize, shrt_pprice, lowest_ask: \
            iter_shrt_entries_inverse(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                      self.ddown_factor, self.qty_pct, self.leverage,
                                      self.grid_spacing, self.grid_coefficient, balance, long_psize,
                                      shrt_psize, shrt_pprice, lowest_ask)
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

    async def init_ema(self):
        # fetch 10000 ticks to initiate ema
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

    async def init_order_book(self):
        ticker = await self.private_get('/v2/public/tickers', {'symbol': self.symbol})
        self.ob = [float(ticker['result'][0]['bid_price']), float(ticker['result'][0]['ask_price'])]
        self.price = float(ticker['result'][0]['last_price'])


    async def fetch_open_orders(self) -> [dict]:
        fetched = await self.private_get('/futures/private/order', {'symbol': self.symbol})
        oos = []
        for elm in fetched['result']:
            if elm['order_status'] == 'New':
                position_side = determine_pos_side(elm)
                oos.append({'order_id': elm['order_id'],
                            'custom_id': elm['order_link_id'],
                            'symbol': elm['symbol'],
                            'price': float(elm['price']),
                            'qty': float(elm['qty']),
                            'side': elm['side'].lower(),
                            'position_side': position_side,
                            'timestamp': date_to_ts(elm['created_at'])})
        return oos

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(self, type_: str, url: str, params: dict = {}) -> dict:
        timestamp = int(time() * 1000)
        params.update({'api_key': self.key, 'timestamp': timestamp})
        for k in params:
            if type(params[k]) == bool:
                params[k] = 'true' if params[k] else 'false'
            elif type(params[k]) == float:
                params[k] = str(params[k])
        params['sign'] = hmac.new(self.secret.encode('utf-8'),
                                  urlencode(sort_dict_keys(params)).encode('utf-8'),
                                  hashlib.sha256).hexdigest()
        async with getattr(self.session, type_)(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}) -> dict:
        return await self.private_('get', url, params)

    async def private_post(self, url: str, params: dict = {}) -> dict:
        return await self.private_('post', url, params)

    async def fetch_position(self) -> dict:
        fetched = await self.private_get('/futures/private/position/list', {'symbol': self.symbol})
        position = {}
        for e in fetched['result']:
            if e['data']['position_idx'] == 1:
                position['long'] = {'size': float(e['data']['size']),
                                    'price': float(e['data']['entry_price']),
                                    'leverage': float(e['data']['leverage']),
                                    'liquidation_price': float(e['data']['liq_price']),
                                    'equity': (b := float(e['data']['wallet_balance'])),
                                    'wallet_balance': b}
            elif e['data']['position_idx'] == 2:
                position['shrt'] = {'size': -abs(float(e['data']['size'])),
                                    'price': float(e['data']['entry_price']),
                                    'leverage': float(e['data']['leverage']),
                                    'liquidation_price': float(e['data']['liq_price']),
                                    'equity': (b := float(e['data']['wallet_balance'])),
                                    'wallet_balance': b}
        position['wallet_balance'] = position['long']['wallet_balance']
        return position

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side':  first_capitalized(order['side']),
                  'position_idx': 1 if order['position_side'] == 'long' else 2,
                  'order_type': first_capitalized(order['type']),
                  'qty': int(order['qty'])}
        if params['order_type'] == 'Limit':
            params['time_in_force'] = 'PostOnly'
            params['price'] = str(order['price'])
        else:
            params['time_in_force'] = 'GoodTillCancel'
        if 'custom_id' in order:
            params['order_link_id'] = \
                f"{order['custom_id']}_{int(time() * 1000)}_{int(np.random.random() * 1000)}"
        o = await self.private_post('/futures/private/order/create', params)
        if o['result']:
            return {'symbol': o['result']['symbol'],
                    'side': o['result']['side'].lower(),
                    'position_side': order['position_side'],
                    'type': o['result']['order_type'].lower(),
                    'qty': o['result']['qty'],
                    'price': o['result']['price']}
        else:
            return {}

    async def execute_cancellation(self, id_: str) -> [dict]:
        o = await self.private_post('/futures/private/order/cancel',
                                    {'symbol': self.symbol, 'order_id': id_})
        return {'symbol': o['result']['symbol'], 'side': o['result']['side'].lower(),
                'position_side': determine_pos_side(o['result']),
                'qty': o['result']['qty'], 'price': o['result']['price']}

    async def init_my_trades(self, age_limit_days: float = 7.0) -> [dict]:
        age_limit = self.cc.milliseconds() - 1000 * 60 * 60 * 24 * age_limit_days
        mtl = await self.fetch_my_trades()
        print('loading my trades cache...')
        mtl += self.load_cached_my_trades()
        mtd = {t['order_id']: t for t in mtl}
        mt = sorted(mtd.values(), key=lambda x: x['timestamp'])
        page = 2
        while mt[0]['timestamp'] > age_limit:
            print('fetching my trades', ts_to_date(mt[0]['timestamp'] / 1000))
            new_mt = await self.fetch_my_trades(page)
            if len(new_mt) == 0 or new_mt[0]['order_id'] in mtd:
                break
            page += 1
            mtd = {t['order_id']: t for t in mt + new_mt}
            mt = sorted(mtd.values(), key=lambda x: x['timestamp'])
        my_trades = [t for t in mt if t['timestamp'] > age_limit]
        print('dumping trades to cache...')
        with open(self.my_trades_cache_filepath, 'w') as f:
            for t in my_trades:
                f.write(json.dumps(t) + '\n')
        self.my_trades = my_trades

    async def fetch_my_trades(self, page: int = 1):
        params = {'symbol': self.symbol, 'limit': 200, 'order': 'desc', 'page': page}
        fetched = await self.cc.v2_private_get_execution_list(params=params)
        mt = {t['exec_id']: {'custom_id': t['order_link_id'],
                             'order_id': t['order_id'],
                             'symbol': t['symbol'],
                             'side': t['side'].lower(),
                             'type': t['order_type'].lower(),
                             'price': float(t['order_price']),
                             'qty': float(t['order_qty']),
                             'timestamp': t['trade_time_ms']}
              for t in fetched['result']['trade_list']}
        return sorted(mt.values(), key=lambda t: t['timestamp'])

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        params = {'symbol': self.symbol, 'limit': 1000}
        if from_id is not None:
            params['from'] = max(0, from_id)
        try:
            ticks = await self.public_get('/v2/public/trading-records', params)
        except Exception as e:
            print('error fetching ticks', e)
            return []
        return list(map(format_tick, ticks['result']))

    def calc_margin_cost(self, qty: float, price: float) -> float:
        return qty / price / self.leverage

    def calc_max_pos_size(self, balance: float, price: float):
        return balance * price * self.leverage * 0.95

    async def init_exchange_settings(self):
        try:
            res = await self.private_post('/futures/private/position/switch-isolated',
                                          {'symbol': self.symbol, 'is_isolated': False,
                                           'buy_leverage': int(self.leverage),
                                           'sell_leverage': int(self.leverage)})
            print(res)
        except Exception as e:
            print(e)
        try:
            lev = await self.private_post('/futures/private/position/leverage/save',
                                          {'symbol': self.symbol,
                                           'buy_leverage': int(self.leverage),
                                           'sell_leverage': int(self.leverage)})
            print(lev)
        except Exception as e:
            print(e)
        try:
            res = await self.private_post('/futures/private/position/switch-mode',
                                          {'symbol': self.symbol, 'mode': 3})
            print(res)
        except Exception as e:
            print(e)

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://stream.bybit.com/realtime"
        print_([uri])
        await self.init_exchange_settings()
        param = {'op': 'subscribe', 'args': ['trade.' + self.symbol]}
        k = 1
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps(param))
            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                price_changed = False
                try:
                    for e in data['data']:
                        if e['price'] != self.price:
                            if e['side'] == 'Buy':
                                self.ob[1] = e['price']
                            elif e['side'] == 'Sell':
                                self.ob[0] = e['price']
                            self.price = e['price']
                            price_changed = True
                            self.ema = calc_ema(self.ema_alpha, self.ema_alpha_, self.ema, e['price'])
                except Exception as e:
                    if 'success' not in data:
                        print('error in websocket streamed data', e)
                if price_changed:
                    if self.ts_locked['decide'] < self.ts_released['decide']:
                        asyncio.create_task(self.decide())
                    if k % 10 == 0:
                        self.flush_stuck_locks()
                        k = 1
                    k += 1


async def main() -> None:
    bot = await create_bot(sys.argv[1], load_live_settings('bybit', sys.argv[1]))
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())


