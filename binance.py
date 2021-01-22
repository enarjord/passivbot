from __future__ import annotations
import asyncio
import json
import websockets
import os
import sys
import numpy as np
import pandas as pd
import pprint
import datetime
from math import ceil
from math import floor
from time import time, sleep
from typing import Callable, Iterator
from passivbot import init_ccxt, load_key_secret, load_settings, make_get_filepath, print_, \
    ts_to_date, flatten, filter_orders, Bot, start_bot, round_up, round_dn, calc_default_qty


def get_maintenance_margin_rate(pos_size_ito_usdt: float) -> float:
    kvs = [(50000, 0.004), (250000, 0.005), (5000000, 0.025), (20000000, 0.05), (50000000, 0.1),
           (100000000, 0.125), (200000000, 0.15), (-1, 0.25)]
    for kv in kvs:
        if pos_size_ito_usdt < kv[0]:
            return kv[1]
    return kvs[-1][1]


def calc_cross_long_liq_price(balance,
                              pos_size,
                              pos_price,
                              mm=0.004,
                              leverage=100):
    pos_margin = pos_size * pos_price / leverage
    d = (pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance + pos_margin - pos_size * pos_price) / d


def calc_cross_shrt_liq_price(balance,
                              pos_size,
                              pos_price,
                              mm=0.004,
                              leverage=100):
    abs_pos_size = abs(pos_size)
    pos_margin = abs_pos_size * pos_price / leverage
    d = (abs_pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance + pos_margin - pos_size * pos_price) / d


async def fetch_trades(cc, symbol: str, from_id: int = None) -> [dict]:
    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['fromId'] = from_id
    fetched_trades = await cc.fapiPublic_get_aggtrades(params=params)
    trades = [{'trade_id': int(t['a']),
               'price': float(t['p']),
               'qty': float(t['q']),
               'timestamp': t['T'],
               'is_buyer_maker': t['m']} for t in fetched_trades]
    print_(['fetched trades', symbol, trades[0]['trade_id'],
            ts_to_date(trades[0]['timestamp'] / 1000)])
    return trades

async def create_bot(user: str, settings: str):
    bot = BinanceBot(user, settings)
    await bot._init()
    return bot


class BinanceBot(Bot):
    def __init__(self, user: str, settings: dict):
        self.exchange = 'binance'
        super().__init__(user, settings)
        self.cc = init_ccxt(self.exchange, user)
        self.trade_id = 0

    async def _init(self):
        exchange_info = await self.cc.fapiPublic_get_exchangeinfo()
        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.coin = e['baseAsset']
                self.quot = e['quoteAsset']
                self.margin_coin = e['marginAsset']
                price_precision = e['pricePrecision']
                qty_precision = e['quantityPrecision']
                for q in e['filters']:
                    if q['filterType'] == 'LOT_SIZE':
                        self.min_qty = float(q['minQty'])
                    elif q['filterType'] == 'MARKET_LOT_SIZE':
                        self.qty_step = float(q['stepSize'])
                    elif q['filterType'] == 'PRICE_FILTER':
                        self.price_step = float(q['tickSize'])
                self.ardn = lambda n: round_dn(n, self.qty_step)
                self.arup = lambda n: round_up(n, self.qty_step)
                self.prdn = lambda n: round_dn(n, self.price_step)
                self.prup = lambda n: round_up(n, self.price_step)
                self.calc_default_qty = lambda balance_, last_price: \
                    calc_default_qty(self.min_qty,
                                     self.qty_step,
                                     balance_ / last_price,
                                     self.default_qty)
                break
        await self.update_position()
        await self.init_order_book()

    async def init_order_book(self):
        ticker = await self.cc.fapiPublic_get_ticker_bookticker(params={'symbol': self.symbol})
        self.ob = [float(ticker['bidPrice']), float(ticker['askPrice'])]
        self.price = np.random.choice(self.ob)

    def calc_entry_qty(self, balance_, pos_size_, pos_price_):
        return calc_entry_qty(self.qty_step,
                              self.min_qty,
                              self.ddown_factor,
                              self.leverage,
                              balance_,
                              pos_size_,
                              1 / pos_price_)

    def calc_long_entry_price(self, balance_, pos_size_, pos_price_):
        return calc_long_entry_price(self.price_step,
                                     self.leverage,
                                     self.grid_spacing,
                                     self.grid_spacing_coefficient,
                                     balance_,
                                     pos_size_ * pos_price_**2,
                                     pos_price_)

    def calc_shrt_entry_price(self, balance_, pos_size_, pos_price_):
        return calc_shrt_entry_price(self.price_step,
                                     self.leverage,
                                     self.grid_spacing,
                                     self.grid_spacing_coefficient,
                                     balance_,
                                     pos_size_ * pos_price_**2,
                                     pos_price_)

    async def fetch_open_orders(self) -> [dict]:
        return [
            {'order_id': int(e['orderId']),
             'symbol': e['symbol'],
             'price': float(e['price']),
             'qty': float(e['origQty']),
             'type': e['type'].lower(),
             'side': e['side'].lower(),
             'timestamp': int(e['time'])}
            for e in await self.cc.fapiPrivate_get_openorders(params={'symbol': self.symbol})
        ]

    async def fetch_position(self) -> None:
        positions, balance = await asyncio.gather(
            self.cc.fapiPrivate_get_positionrisk(params={'symbol': self.symbol}),
            self.cc.fapiPrivate_get_balance(),
        )
        if positions:
            position = {'size': float(positions[0]['positionAmt']),
                        'price': float(positions[0]['entryPrice']),
                        'liquidation_price': float(positions[0]['liquidationPrice']),
                        'leverage': float(positions[0]['leverage'])}
        else:
            position = {'size': 0.0,
                        'price': 0.0,
                        'liquidation_price': 0.0,
                        'leverage': 1.0}
        position['cost'] = abs(position['size']) * position['price']
        position['margin_cost'] = position['cost'] / self.leverage
        for e in balance:
            if e['asset'] == 'USDT':
                position['balance'] = float(e['balance'])
                break
        return position

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'type': order['type'].upper(),
                  'quantity': order['qty'],
                  'reduceOnly': order['reduce_only']}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = order['price']
        o = await self.cc.fapiPrivate_post_order(params=params)
        return {'symbol': self.symbol,
                'side': o['side'].lower(),
                'type': o['type'].lower(),
                'qty': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_cancellation(self, id_: [dict]) -> [dict]:
        cancellation = await self.cc.fapiPrivate_delete_order(params={
            'symbol': self.symbol, 'orderId': id_
        })
        return {'symbol': self.symbol, 'side': cancellation['side'].lower(),
                'qty': float(cancellation['origQty']), 'price': float(cancellation['price'])}

    async def fetch_trades(self, from_id: int = None):
        return await fetch_trades(self.cc, self.symbol, from_id)

    def calc_margin_cost(self, qty: float, price: float) -> float:
        return qty * price / self.leverage

    def calc_max_pos_size(self, balance: float, price: float):
        return (balance / price) * self.leverage

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
        print_([uri])
        try:
            print(await self.cc.fapiPrivate_post_margintype(params={'symbol': self.symbol,
                                                                    'marginType': 'CROSSED'}))
        except Exception as e:
            print(e)
        try:
            print(await self.cc.fapiPrivate_post_leverage(params={'symbol': self.symbol,
                                                                  'leverage': int(self.leverage)}))
        except Exception as e:
            print(e)
        await self.init_indicators()
        await self.update_position()
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
                self.price = price
                self.update_indicators({'timestamp': data['T'],
                                        'price': price,
                                        'side': 'sell' if data['m'] else 'buy',
                                        'qty': float(data['q'])})
                if self.ts_locked['decide'] < self.ts_released['decide']:
                    asyncio.create_task(self.decide())
                elif k % 10 == 0:
                    self.flush_stuck_locks()
                    k = 1
                if self.stop_websocket:
                    break
                k += 1


async def main() -> None:
    bot = await create_bot(sys.argv[1], load_settings('binance', sys.argv[1]))
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())


