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
    calc_min_order_qty, \
    iter_long_closes_linear, iter_shrt_closes_linear, calc_ema, iter_entries_linear


def get_maintenance_margin_rate(pos_size_ito_usdt: float) -> float:
    kvs = [(50000, 0.004), (250000, 0.005), (5000000, 0.025), (20000000, 0.05), (50000000, 0.1),
           (100000000, 0.125), (200000000, 0.15), (-1, 0.25)]
    for kv in kvs:
        if pos_size_ito_usdt < kv[0]:
            return kv[1]
    return kvs[-1][1]


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


async def fetch_ticks(cc, symbol: str, from_id: int = None, do_print=True) -> [dict]:
    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['fromId'] = from_id
    fetched_trades = await cc.fapiPublic_get_aggtrades(params=params)
    trades = [{'trade_id': int(t['a']),
               'price': float(t['p']),
               'qty': float(t['q']),
               'timestamp': t['T'],
               'is_buyer_maker': t['m']} for t in fetched_trades]
    if do_print:
        print_(['fetched trades', symbol, trades[0]['trade_id'],
                ts_to_date(float(trades[0]['timestamp']) / 1000)])
    return trades


async def fetch_ticks_time(cc, symbol: str, start_time: int, end_time: int = None, do_print=True) -> [dict]:
    params = {'symbol': symbol, 'limit': 1000, 'startTime': start_time}
    if end_time:
        params['endTime'] = end_time
    fetched_trades = await cc.fapiPublic_get_aggtrades(params=params)
    trades = [{'trade_id': int(t['a']),
               'price': float(t['p']),
               'qty': float(t['q']),
               'timestamp': t['T'],
               'is_buyer_maker': t['m']} for t in fetched_trades]
    try:
        if do_print:
            print_(['fetched trades', symbol, trades[0]['trade_id'],
                    ts_to_date(float(trades[0]['timestamp']) / 1000)])
    except:
        if do_print:
            print_(['fetched no new trades', symbol])
    return trades


async def create_bot(user: str, settings: str):
    bot = BinanceBot(user, settings)
    await bot._init()
    return bot


class BinanceBot(Bot):
    def __init__(self, user: str, settings: dict):
        self.exchange = 'binance'
        super().__init__(user, settings)
        self.max_pos_size_ito_usdt = 0.0
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
                        self.min_notional = self.min_cost = float(q['notional'])
                self.calc_min_qty = lambda price_: \
                    max(self.min_qty, round_up(self.min_notional / price_, self.qty_step))
                self.calc_min_order_qty = lambda balance_, last_price: \
                    calc_min_order_qty(self.calc_min_qty(last_price),
                                       self.qty_step,
                                       (balance_ / last_price) * self.leverage,
                                       self.qty_pct)
                break
        max_lev = 0
        for e in leverage_bracket:
            if e['symbol'] == self.symbol:
                for br in e['brackets']:
                    max_lev = max(max_lev, int(br['initialLeverage']))
                break
        self.max_leverage = max_lev
        await self.update_position()
        await self.init_order_book()
        await self.init_ema()
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
            highest_bid, lowest_ask, last_price: \
            iter_entries_linear(self.price_step, self.qty_step, self.min_qty, self.min_cost,
                                self.ddown_factor, self.qty_pct, self.leverage,
                                self.grid_spacing, self.grid_coefficient, balance, long_psize,
                                long_pprice, shrt_psize, shrt_pprice, highest_bid, lowest_ask,
                                last_price, self.do_long, self.do_shrt)

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
            mode = 'CROSSED'
            print(await self.cc.fapiPrivate_post_margintype(params={'symbol': self.symbol,
                                                                    'marginType': mode}))
        except Exception as e:
            print(e)
        try:
            lev = await self.cc.fapiPrivate_post_leverage(params={'symbol': self.symbol,
                                                                  'leverage': int(round(self.leverage))})
            self.max_pos_size_ito_usdt = float(lev['maxNotionalValue'])
            print('max pos size in terms of usdt', self.max_pos_size_ito_usdt)
        except Exception as e:
            print(e)
        try:
            res = await self.cc.fapiPrivate_post_positionside_dual(params={'dualSidePosition': 'true'})
            print(res)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print(e)
                print('unable to set hedge mode, aborting')
                raise Exception('failed to set hedge mode')

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
             'position_side': e['positionSide'].lower().replace('short', 'shrt'),
             'timestamp': int(e['time'])}
            for e in await self.cc.fapiPrivate_get_openorders(params={'symbol': self.symbol})
        ]

    async def fetch_position(self) -> dict:
        positions, account = await asyncio.gather(
            self.cc.fapiPrivate_get_positionrisk(params={'symbol': self.symbol}),
            self.cc.fapiPrivate_get_account()
        )
        position = {}
        if positions:
            for p in positions:
                if p['positionSide'] == 'LONG':
                    position['long'] = {'size': (lsize:= float(p['positionAmt'])),
                                        'price': (lprice := float(p['entryPrice'])),
                                        'liquidation_price': float(p['liquidationPrice']),
                                        'leverage': float(p['leverage']),
                                        'cost': (lcost := abs(lsize) * lprice),
                                        'margin_cost': lcost / self.leverage}
                elif p['positionSide'] == 'SHORT':
                    position['shrt'] = {'size': (ssize:= float(p['positionAmt'])),
                                        'price': (sprice := float(p['entryPrice'])),
                                        'liquidation_price': float(p['liquidationPrice']),
                                        'leverage': float(p['leverage']),
                                        'cost': (scost := abs(ssize) * sprice),
                                        'margin_cost': scost / self.leverage}
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
                  'positionSide': order['position_side'].replace('shrt', 'short').upper(),
                  'type': order['type'].upper(),
                  'quantity': order['qty']}
        if params['type'] == 'LIMIT':
            params['timeInForce'] = 'GTX'
            params['price'] = order['price']
        if 'custom_id' in order:
            params['newClientOrderId'] = \
                f"{order['custom_id']}_{int(time() * 1000)}_{int(np.random.random() * 1000)}"
        o = await self.cc.fapiPrivate_post_order(params=params)
        return {'symbol': self.symbol,
                'side': o['side'].lower(),
                'position_side': o['positionSide'].lower().replace('short', 'shrt'),
                'type': o['type'].lower(),
                'qty': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_cancellation(self, id_: str) -> [dict]:
        cancellation = await self.cc.fapiPrivate_delete_order(params={
            'symbol': self.symbol, 'orderId': id_
        })
        return {'symbol': self.symbol, 'side': cancellation['side'].lower(),
                'position_side': cancellation['positionSide'].lower().replace('short', 'shrt'),
                'qty': float(cancellation['origQty']), 'price': float(cancellation['price'])}

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        return await fetch_ticks(self.cc, self.symbol, from_id, do_print=do_print)

    async def fetch_ticks_time(self, start_time: int, end_time: int = None, do_print: bool = True):
        return await fetch_ticks_time(self.cc, self.symbol, start_time, end_time, do_print=do_print)

    def calc_margin_cost(self, qty: float, price: float) -> float:
        return qty * price / self.leverage

    def calc_max_pos_size(self, balance: float, price: float):
        return min((balance / price) * self.leverage, self.max_pos_size_ito_usdt / price) * 0.92

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
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

