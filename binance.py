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
    ts_to_date, flatten, calc_new_ema, filter_orders, Bot, start_bot


async def fetch_trades(cc, symbol: str, from_id: int = None) -> [dict]:
    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['fromId'] = from_id
    fetched_trades = await cc.fapiPublic_get_aggtrades(params=params)
    trades = [{'trade_id': t['a'],
               'price': float(t['p']),
               'amount': float(t['q']),
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
        super().__init__(user, settings)
        self.cc = init_ccxt('binance', user)
        self.trade_id = 0

    async def _init(self):
        exchange_info = await self.cc.fapiPublic_get_exchangeinfo()
        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.coin = e['baseAsset']
                self.quot = e['quoteAsset']
                self.margin_coin = e['marginAsset']
                price_precision = e['pricePrecision']
                self.round_up = lambda n: ceil(n * (dexp := 10**price_precision)) / dexp
                self.round_dn = lambda n: floor(n * (dexp := 10**price_precision)) / dexp
                break
        await self.update_position()

    async def fetch_open_orders(self) -> [dict]:
        return [
            {'order_id': int(e['orderId']),
             'symbol': e['symbol'],
             'price': float(e['price']),
             'amount': float(e['origQty']),
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
                        'entry_price': float(positions[0]['entryPrice']),
                        'liquidation_price': float(positions[0]['liquidationPrice']),
                        'leverage': float(positions[0]['leverage'])}
        else:
            position = {'size': 0.0,
                        'entry_price': 0.0,
                        'liquidation_price': 0.0,
                        'leverage': 1.0}
        for e in balance:
            if e['asset'] == 'USDT':
                position['equity'] = float(e['balance'])
                break
        return position

    async def execute_bid(self, amount: float, price: float) -> dict:
        o = await self.cc.fapiPrivate_post_order(params={
            'symbol': self.symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'timeInForce': 'GTC'
        })
        return {'symbol': self.symbol,
                'side': 'buy',
                'type': 'limit',
                'amount': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_ask(self, amount: float, price: float) -> dict:
        o = await self.cc.fapiPrivate_post_order(params={
            'symbol': self.symbol,
            'side': 'SELL',
            'type': 'LIMIT',
            'quantity': amount,
            'price': price,
            'timeInForce': 'GTC'
        })
        return {'symbol': self.symbol,
                'side': 'sell',
                'type': 'limit',
                'amount': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_cancellation(self, id_: [dict]) -> [dict]:
        cancellation = await self.cc.fapiPrivate_delete_order(params={
            'symbol': self.symbol, 'orderId': id_
        })
        return {'symbol': self.symbol, 'side': cancellation['side'].lower(),
                'amount': float(cancellation['origQty']), 'price': float(cancellation['price'])}

    async def fetch_trades(self, from_id: int = None):
        return await fetch_trades(self.cc, self.symbol, from_id)

    def get_margin_cost(self, amount: float, price: float) -> float:
        return amount * price / self.leverage

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
        print_([uri])
        print(await self.cc.fapiPrivate_post_leverage(params={'symbol': self.symbol,
                                                              'leverage': self.leverage}))
        await self.update_position()
        await self.init_emas()
        async with websockets.connect(uri) as ws:
            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                price = float(data['p'])
                trade_id = data['a']
                for span in self.ema_spans:
                    self.emas[span] = calc_new_ema(self.price,
                                                   price,
                                                   self.emas[span],
                                                   alpha=self.ema_alphas[span],
                                                   n_steps=trade_id - self.trade_id)
                self.trade_id = trade_id
                self.price = price
                if self.ts_locked['decide'] < self.ts_released['decide']:
                    asyncio.create_task(self.decide())
                elif self.trade_id % 10 == 0:
                    self.flush_stuck_locks()
                if self.stop_websocket:
                    break




def iter_chunks(symbol: str) -> Iterator[pd.DataFrame]:
    chunk_size = 100000
    filepath = f'historical_data/binance/agg_trades_futures/{symbol}/'
    if os.path.isdir(filepath):
        filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')])
        for f in filenames[::-1]:
            chunk = pd.read_csv(filepath + f).set_index('trade_id')
            if chunk is not None:
                print('loaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000))
                yield chunk
            else:
                yield None
        yield None
    else:
        yield None


async def load_trades(symbol: str, n_days: float) -> pd.DataFrame:
    cc = init_ccxt('binance', 'example_user')
    filepath = make_get_filepath(f'historical_data/binance/agg_trades_futures/{symbol}/')
    cache_filepath = make_get_filepath(
        f'historical_data/binance/agg_trades_futures/{symbol}_cache/'
    )
    cache_filenames = [f for f in os.listdir(cache_filepath) if f.endswith('.csv')]
    ids = set()
    if cache_filenames:
        print('loaded cached trades')
        cached_trades = pd.concat([pd.read_csv(cache_filepath + f) for f in cache_filenames],
                                  axis=0)
        cached_trades = cached_trades.set_index('trade_id').sort_index()
        cached_trades = cached_trades[~cached_trades.index.duplicated()]
        ids.update(cached_trades.index)
    else:
        cached_trades = None
    age_limit = time() - 60 * 60 * 24 * n_days
    age_limit_millis = age_limit * 1000
    print('age_limit', ts_to_date(age_limit))
    chunk_iterator = iter_chunks(symbol)
    chunk = next(chunk_iterator)
    chunks = {} if chunk is None else {int(chunk.index[0]): chunk}
    if chunk is not None:
        ids.update(chunk.index)
    min_id = min(ids) if ids else 0
    new_trades = await fetch_trades(cc, symbol)
    cached_ids = set()
    k = 0
    while True:
        if new_trades[0]['timestamp'] <= age_limit_millis:
            break
        from_id = new_trades[0]['trade_id'] - 1
        while True:
            if chunk is None:
                min_id = 0
                break
            if from_id in ids:
                print('skipping from', from_id)
                while from_id in ids:
                    from_id -= 1
                print('           to', from_id)
            if from_id < min_id:
                chunk = next(chunk_iterator)
                if chunk is None:
                    min_id = 0
                    break
                else:
                    chunks[int(chunk.index[0])] = chunk
                    ids.update(chunk.index)
                    min_id = min(ids)
                    if chunk.timestamp.max() < age_limit_millis:
                        break
            else:
                break
        from_id -= 999
        new_trades = await fetch_trades(cc, symbol, from_id=from_id) + new_trades
        k += 1
        if k % 20 == 0:
            print('dumping cache')
            cache_df = pd.DataFrame([t for t in new_trades
                                     if t['trade_id'] not in cached_ids]).set_index('trade_id')
            cache_df.to_csv(cache_filepath + str(int(time() * 1000)) + '.csv')
            cached_ids.update(cache_df.index)
    new_trades_df = pd.DataFrame(new_trades).set_index('trade_id')
    trades_updated = pd.concat(list(chunks.values()) + [new_trades_df, cached_trades], axis=0)
    no_dup = trades_updated[~trades_updated.index.duplicated()]
    no_dup_sorted = no_dup.sort_index()
    chunk_size = 100000
    chunk_ids = no_dup_sorted.index // chunk_size * chunk_size
    for g in no_dup_sorted.groupby(chunk_ids):
        if g[0] not in chunks or len(chunks[g[0]]) != chunk_size:
            print('dumping chunk', g[0])
            g[1].to_csv(f'{filepath}{str(g[0])}.csv')
    for f in [f_ for f_ in os.listdir(cache_filepath) if f_.endswith('.csv')]:
        os.remove(cache_filepath + f)
    await cc.close()
    return no_dup_sorted[no_dup_sorted.timestamp >= age_limit_millis]


async def main() -> None:
    bot = await create_bot(sys.argv[1], load_settings('binance_futures', sys.argv[1]))
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())


