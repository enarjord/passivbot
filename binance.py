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
from passivbot import init_ccxt, load_key_secret, load_live_settings, make_get_filepath, print_, \
    ts_to_date, flatten, filter_orders, Bot, start_bot, round_up, round_dn, \
    calc_min_entry_qty


def get_maintenance_margin_rate(pos_size_ito_usdt: float) -> float:
    kvs = [(50000, 0.004), (250000, 0.005), (5000000, 0.025), (20000000, 0.05), (50000000, 0.1),
           (100000000, 0.125), (200000000, 0.15), (-1, 0.25)]
    for kv in kvs:
        if pos_size_ito_usdt < kv[0]:
            return kv[1]
    return kvs[-1][1]


def get_max_pos_size_ito_usdt(symbol: str, leverage: int) -> float:
    if symbol == 'BTCUSDT':
        kvs = [(100, 50000), (50, 250000), (20, 1000000), (10, 5000000), (5, 20000000),
               (4, 50000000), (3, 100000000), (2, 200000000)]
    elif symbol == 'ETHUSDT':
        kvs = [(75, 10000), (50, 100000), (25, 500000), (10, 1000000), (5, 2000000),
               (4, 5000000), (3, 10000000), (2, 20000000)]
    elif symbol in ['ADAUSDT', 'BNBUSDT', 'DOTUSDT', 'EOSUSDT', 'ETCUSDT', 'LINKUSDT', 'LTCUSDT',
                    'TRXUSDT', 'XLMUSDT', 'XMRUSDT', 'XRPUSDT', 'XTZUSDT', 'BCHUSDT']:
        kvs = [(50, 10000), (25, 50000), (10, 250000), (5, 1000000), (4, 2000000),
               (3, 5000000), (2, 10000000)]
    elif symbol in ['AAVEUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'AXSUSDT',
                    'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BELUSDT', 'BLZUSDT', 'BZRXUSDT', 'COMPUSDT',
                    'CRVUSDT', 'CVCUSDT', 'DASHUSDT', 'DEFIUSDT', 'DOGEUSDT', 'EGLDUSDT', 'ENJUSDT',
                    'FILUSDT', 'FLMUSDT', 'FTMUSDT', 'HNTUSDT', 'ICXUSDT', 'IOSTUSDT', 'IOTAUSDT',
                    'KAVAUSDT', 'KNCUSDT', 'KSMUSDT', 'LRCUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
                    'NEOUSDT', 'OCEANUSDT', 'OMGUSDT', 'ONTUSDT', 'QTUMUSDT', 'RENUSDT', 'RLCUSDT',
                    'RSRUSDT', 'RUNEUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STORJUSDT',
                    'SUSHIUSDT', 'SXPUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'UNIUSDT',
                    'VETUSDT', 'WAVESUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT', 'ZILUSDT', 'ZRXUSDT',
                    'ZENUSDT', 'SKLUSDT', 'GRTUSDT', '1INCHUSDT']:
        kvs = [(20, 5000), (10, 25000), (5, 100000), (2, 250000), (1, 1000000)]
    elif symbol in ['CTKUSDT', 'LITUSDT']:
        kvs = [(10, 5000), (5, 25000), (4, 100000), (2, 250000), (1, 1000000)]
    else:
        print(f'{symbol} unknown symbol')
        kvs = [(0, 5000)]
    for kv in kvs:
        if leverage > kv[0]:
            return kv[1]
    return 9e12


def calc_isolated_long_liq_price(balance,
                                 pos_size,
                                 pos_price,
                                 leverage,
                                 mm=0.004) -> float:
    return calc_cross_long_liq_price(pos_size * pos_price / leverage, pos_size, pos_price, leverage, mm)


def calc_isolated_shrt_liq_price(balance,
                                 pos_size,
                                 pos_price,
                                 leverage,
                                 mm=0.004) -> float:
    return calc_cross_shrt_liq_price(abs(pos_size) * pos_price / leverage, pos_size, pos_price, leverage, mm)


def calc_cross_long_liq_price(balance,
                              pos_size,
                              pos_price,
                              leverage,
                              mm=0.004) -> float:
    pos_margin = pos_size * pos_price / leverage
    d = (pos_size * mm - pos_size)
    if d == 0.0:
        return 0.0
    return (balance + pos_margin - pos_size * pos_price) / d


def calc_cross_shrt_liq_price(balance,
                              pos_size,
                              pos_price,
                              leverage,
                              mm=0.004) -> float:
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
        self.max_pos_size_ito_usdt = get_max_pos_size_ito_usdt(settings['symbol'],
                                                               settings['leverage'])

        self.cc = init_ccxt(self.exchange, user)
        self.trade_id = 0

    async def _init(self):
        exchange_info, leverage_bracket = await asyncio.gather(
            self.cc.fapiPublic_get_exchangeinfo(),
            self.cc.fapiPrivate_get_leveragebracket()
        )
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
                    elif q['filterType'] == 'MIN_NOTIONAL':
                        self.min_notional = float(q['notional'])
                self.calc_min_qty = lambda price_: \
                    max(self.min_qty, round_up(self.min_notional / price_, self.qty_step))
                self.calc_min_entry_qty = lambda balance_, last_price: \
                    calc_min_entry_qty(self.calc_min_qty(last_price),
                                       self.qty_step,
                                       (balance_ / last_price) * self.leverage,
                                       self.entry_qty_pct)
                break
        max_lev = 0
        for e in leverage_bracket:
            if e['symbol'] == self.symbol:
                for br in e['brackets']:
                    max_lev = max(max_lev, br['initialLeverage'])
                break
        self.max_leverage = max_lev
        await self.update_position()
        await self.init_order_book()

    async def init_order_book(self):
        ticker = await self.cc.fapiPublic_get_ticker_bookticker(params={'symbol': self.symbol})
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
             'timestamp': int(e['time'])}
            for e in await self.cc.fapiPrivate_get_openorders(params={'symbol': self.symbol})
        ]

    async def fetch_position(self) -> dict:
        positions, account, funding = await asyncio.gather(
            self.cc.fapiPrivate_get_positionrisk(params={'symbol': self.symbol}),
            self.cc.fapiPrivate_get_account(),
            self.cc.fapiPublic_get_fundingrate()
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
        for e in funding:
            if e['symbol'] == self.symbol:
                position['predicted_funding_rate'] = float(e['fundingRate'])
                break
        for e in account['assets']:
            if e['asset'] == 'USDT':
                position['equity'] = float(e['marginBalance'])
                position['wallet_balance'] = float(e['walletBalance'])
                break
        return position

    async def init_my_trades(self, age_limit_days: float = 7.0) -> [dict]:
        age_limit = self.cc.milliseconds() - 1000 * 60 * 60 * 24 * age_limit_days
        mtl = self.load_cached_my_trades()
        print(f'loaded {len(mtl)} cached my trades')
        if not mtl:
            mtl = await self.fetch_my_trades(start_time_ms=age_limit)
        else:
            mtl += await self.fetch_my_trades(start_time_ms=mtl[-1]['timestamp'])
        mtd = {t['order_id']: t for t in mtl}
        mt = sorted(mtd.values(), key=lambda x: x['timestamp'])
        if len(mt) == 0:
            return
        while True:
            print('fetching my trades', ts_to_date(mt[-1]['timestamp'] / 1000))
            new_mt = await self.fetch_my_trades(order_id=mt[-1]['order_id'] + 1)
            if len(new_mt) == 0:
                break
            mt += new_mt
        mtd = {t['order_id']: t for t in mt}
        my_trades = sorted(mtd.values(), key=lambda x: x['order_id'])
        print('dumping trades to cache...')
        with open(self.my_trades_cache_filepath, 'w') as f:
            for t in my_trades:
                f.write(json.dumps(t) + '\n')
        self.my_trades = my_trades

    async def fetch_my_trades(self,
                              start_time_ms: int = -1,
                              order_id: int = -1,
                              limit: int = 1000) -> [dict]:
        params = {'symbol': self.symbol, 'limit': limit}
        if order_id != -1:
            params['orderId'] = order_id
        elif start_time_ms != -1:
            params['startTime'] = int(start_time_ms)
        mt = await self.cc.fapiPrivate_get_allorders(params=params)
        return sorted([{'custom_id': t['clientOrderId'],
                        'order_id': int(t['orderId']),
                        'symbol': t['symbol'],
                        'side': t['side'].lower(),
                        'type': t['type'].lower(),
                        'price': float(t['avgPrice']),
                        'qty': float(t['executedQty']),
                        'timestamp': t['time']}
                       for t in mt if t['status'] == 'FILLED'], key=lambda x: x['timestamp'])

    async def execute_order(self, order: dict) -> dict:
        params = {'symbol': self.symbol,
                  'side': order['side'].upper(),
                  'type': order['type'].upper(),
                  'quantity': order['qty'],
                  'reduceOnly': order['reduce_only']}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = order['price']
        if 'custom_id' in order:
            params['newClientOrderId'] = \
                f"{order['custom_id']}_{int(time() * 1000)}_{int(np.random.random() * 1000)}"
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
        return min((balance / price) * self.leverage, self.max_pos_size_ito_usdt / price) * 0.92

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
        print_([uri])
        try:
            mode = 'CROSSED' if self.settings['cross_mode'] else 'ISOLATED'
            print(await self.cc.fapiPrivate_post_margintype(params={'symbol': self.symbol,
                                                                    'marginType': mode}))
        except Exception as e:
            print(e)
        try:
            lev = await self.cc.fapiPrivate_post_leverage(params={'symbol': self.symbol,
                                                                  'leverage': int(self.leverage)})
            self.max_pos_size_ito_usdt = float(lev['maxNotionalValue'])
            print('max pos size in terms of usdt', self.max_pos_size_ito_usdt)
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
                if price != self.price:
                    self.update_indicators({'timestamp': data['T'],
                                            'price': price,
                                            'side': 'sell' if data['m'] else 'buy',
                                            'qty': float(data['q'])})
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

