import asyncio
import json
import websockets
import os
import sys
import numpy as np
import pandas as pd
import pprint
from datetime import datetime
from math import ceil
from math import floor
from time import time, sleep
from typing import Callable, Iterator
from passivbot import init_ccxt, load_key_secret, load_live_settings, make_get_filepath, print_, \
    ts_to_date, flatten, filter_orders, Bot, start_bot, round_up, round_dn, calc_min_entry_qty


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


def calc_cross_long_liq_price(balance,
                              pos_size,
                              pos_price,
                              leverage,
                              mm=0.005) -> float:
    order_cost = pos_size / pos_price
    order_margin = order_cost / leverage
    bankruptcy_price = calc_cross_long_bankruptcy_price(pos_size, order_cost, balance, order_margin)
    if bankruptcy_price == 0.0:
        return 0.0
    rhs = -(balance - order_margin - (pos_size / pos_price) * mm - \
        (pos_size * 0.00075) / bankruptcy_price)
    return (pos_price * pos_size) / (pos_size - pos_price * rhs)


def calc_cross_long_bankruptcy_price(pos_size, order_cost, balance, order_margin) -> float:
    return (1.00075 * pos_size) / (order_cost + (balance - order_margin))


def calc_cross_shrt_liq_price(balance,
                              pos_size,
                              pos_price,
                              leverage,
                              mm=0.005) -> float:
    _pos_size = abs(pos_size)
    order_cost = _pos_size / pos_price
    order_margin = order_cost / leverage
    bankruptcy_price = calc_cross_shrt_bankruptcy_price(_pos_size, order_cost, balance, order_margin)
    if bankruptcy_price == 0.0:
        return 0.0
    rhs = -(balance - order_margin - (_pos_size / pos_price) * mm - \
        (_pos_size * 0.00075) / bankruptcy_price)
    shrt_liq_price = (pos_price * _pos_size) / (pos_price * rhs + _pos_size)
    if shrt_liq_price <= 0.0:
        return 0.0
    return shrt_liq_price


def calc_cross_shrt_bankruptcy_price(pos_size, order_cost, balance, order_margin) -> float:
    return (0.99925 * pos_size) / (order_cost - (balance - order_margin))


async def fetch_trades(cc, symbol: str, from_id: int = None) -> [dict]:

    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['from'] = from_id
    fetched_trades = await cc.v2_public_get_trading_records(params=params)
    trades = [{'trade_id': int(t['id']),
               'side': t['side'],
               'price': t['price'],
               'qty': t['qty'],
               'timestamp': date_to_ts(t['time'][:-1])} for t in fetched_trades['result']]
    print_(['fetched trades', symbol, trades[0]['trade_id'],
            ts_to_date(trades[0]['timestamp'] / 1000)])
    return trades

def date_to_ts(date: str):
    date = date[:23].replace('Z', '')
    try:
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f").timestamp() * 1000
    except ValueError:
        formats = ["%Y-%m-%dT%H:%M:%S"]
        for f in formats:
            try:
                return datetime.strptime(date, f).timestamp() * 1000
            except ValueError:
                continue
    raise Exception(f'unable to convert date {date} to timestamp')

async def create_bot(user: str, settings: str):
    bot = BybitBot(user, settings)
    await bot._init()
    return bot


class BybitBot(Bot):
    def __init__(self, user: str, settings: dict):
        self.exchange = 'bybit'
        self.min_notional = 0.0
        super().__init__(user, settings)
        self.cc = init_ccxt(self.exchange, user)

    async def _init(self):
        info = await self.cc.v2_public_get_symbols()
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
        self.calc_min_qty = lambda price_: self.min_qty
        self.calc_min_entry_qty = lambda balance_, last_price: \
            calc_min_entry_qty(self.min_qty, self.qty_step,
                                   balance_ * last_price * self.leverage,
                                   self.entry_qty_pct)
        await self.update_position()
        await self.init_order_book()

    async def init_order_book(self):
        ticker = await self.cc.v2_public_get_tickers(params={'symbol': self.symbol})
        self.ob = [float(ticker['result'][0]['bid_price']), float(ticker['result'][0]['ask_price'])]
        self.price = float(ticker['result'][0]['last_price'])

    async def fetch_open_orders(self) -> [dict]:
        fetched = await self.cc.v2_private_get_order(params={'symbol': self.symbol})
        return [
            {'order_id': e['order_id'],
             'symbol': e['symbol'],
             'price': float(e['price']),
             'qty': float(e['qty']),
             'side': e['side'].lower(),
             'timestamp': date_to_ts(e['created_at'])}
            for e in fetched['result']
        ]

    async def fetch_position(self) -> None:

        position, balance, funding = await asyncio.gather(
            self.cc.v2_private_get_position_list(params={'symbol': self.symbol}),
            self.cc.v2_private_get_wallet_balance(),
            self.cc.v2_private_get_funding_predicted_funding(params={'symbol': self.symbol})
        )
        pos = position['result']
        result = {'size': pos['size'] * (-1.0 if pos['side'] == 'Sell' else 1.0),
                  'price': float(pos['entry_price']),
                  'leverage': float(pos['leverage']),
                  'liquidation_price': float(pos['liq_price']),
                  'equity': balance['result'][self.coin]['equity'],
                  'wallet_balance': balance['result'][self.coin]['wallet_balance']}
        result['cost'] = abs(result['size']) / result['price'] if result['price'] else 0.0
        result['margin_cost'] = result['cost'] / self.leverage
        result['predicted_funding_rate'] = funding['result']['predicted_funding_rate']
        return result

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side':  first_capitalized(order['side']),
                  'reduce_only': order['reduce_only'],
                  'order_type': first_capitalized(order['type']),
                  'qty': order['qty']}
        if params['order_type'] == 'Limit':
            params['time_in_force'] = 'PostOnly'
            params['price'] = order['price']
        else:
            params['time_in_force'] = 'GoodTillCancel'
        if 'custom_id' in order:
            params['order_link_id'] = \
                f"{order['custom_id']}_{int(time() * 1000)}_{int(np.random.random() * 1000)}"
        o = await self.cc.v2_private_post_order_create(params=params)
        return {'symbol': o['result']['symbol'],
                'side': o['result']['side'].lower(),
                'type': o['result']['order_type'].lower(),
                'qty': o['result']['qty'],
                'price': o['result']['price']}

    async def execute_cancellation(self, id_: [dict]) -> [dict]:
        o = await self.cc.v2_private_post_order_cancel(
            params={'symbol': self.symbol, 'order_id': id_}
        )
        return {'symbol': o['result']['symbol'], 'side': o['result']['side'].lower(),
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

    async def fetch_trades(self, from_id: int = None):
        return await fetch_trades(self.cc, self.symbol, from_id)

    def calc_margin_cost(self, qty: float, price: float) -> float:
        return qty / price / self.leverage

    def calc_max_pos_size(self, balance: float, price: float):
        return balance * price * self.leverage * 0.95

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://stream.bybit.com/realtime"
        print_([uri])
        await self.init_indicators()
        await self.update_position()
        try:
            leverage_ = 0 if self.settings['cross_mode'] else self.leverage
            print(await self.cc.v2_private_post_position_leverage_save(
                params={'symbol': self.symbol, 'leverage': leverage_}
            ))
        except Exception as e:
            print('error starting websocket', e)
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
                            self.update_indicators({'timestamp': e['trade_time_ms'],
                                                    'price': e['price'],
                                                    'side': e['side'].lower(),
                                                    'qty': e['size']})
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


