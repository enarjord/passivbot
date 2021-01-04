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
    ts_to_date, flatten, calc_new_ema, filter_orders, Bot, start_bot, round_dn, round_up, \
    calc_initial_long_entry_qty, calc_initial_shrt_entry_qty, calc_long_entry_qty, \
    calc_shrt_entry_qty, calc_long_entry_price, calc_shrt_entry_price


async def fetch_trades(cc, symbol: str, from_id: int = None) -> [dict]:
    params = {'symbol': symbol, 'limit': 1000}
    if from_id:
        params['fromId'] = from_id
    fetched_trades = await cc.fapiPublic_get_aggtrades(params=params)
    trades = [{'trade_id': t['a'],
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
                break
        self.calc_initial_long_entry_qty = lambda equity_, price_: calc_initial_long_entry_qty(
            self.min_qty, self.qty_step, self.entry_qty_equity_multiplier, equity_, 1 / price_
        )
        self.calc_initial_shrt_entry_qty = lambda equity_, price_: calc_initial_shrt_entry_qty(
            self.min_qty, self.qty_step, self.entry_qty_equity_multiplier, equity_, 1 / price_
        )

        self.calc_long_entry_qty = lambda equity_, pos_size_, pos_price_: calc_long_entry_qty(
            self.min_qty,
            self.qty_step,
            self.leverage,
            self.ddown_factor,
            self.entry_qty_equity_multiplier,
            equity_,
            pos_size_,
            1 / pos_price_,
        )

        self.calc_shrt_entry_qty = lambda equity_, pos_size_, pos_price_: calc_shrt_entry_qty(
            self.min_qty,
            self.qty_step,
            self.leverage,
            self.ddown_factor,
            self.entry_qty_equity_multiplier,
            equity_,
            pos_size_,
            1 / pos_price_,
        )

        self.calc_long_entry_price = lambda pos_price_: calc_long_entry_price(
            self.price_step,
            self.grid_spacing,
            pos_price_,
        )

        self.calc_shrt_entry_price = lambda pos_price_: calc_shrt_entry_price(
            self.price_step,
            self.grid_spacing,
            pos_price_,
        )

        await self.update_position()

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

    async def execute_bid(self, qty: float, price: float) -> dict:
        o = await self.cc.fapiPrivate_post_order(params={
            'symbol': self.symbol,
            'side': 'BUY',
            'type': 'LIMIT',
            'quantity': qty,
            'price': price,
            'timeInForce': 'GTC'
        })
        return {'symbol': self.symbol,
                'side': 'buy',
                'type': 'limit',
                'qty': float(o['origQty']),
                'price': float(o['price'])}

    async def execute_ask(self, qty: float, price: float) -> dict:
        o = await self.cc.fapiPrivate_post_order(params={
            'symbol': self.symbol,
            'side': 'SELL',
            'type': 'LIMIT',
            'quantity': qty,
            'price': price,
            'timeInForce': 'GTC'
        })
        return {'symbol': self.symbol,
                'side': 'sell',
                'type': 'limit',
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
                                                                  'leverage': self.leverage}))
        except Exception as e:
            print(e)
        await self.update_position()
        await self.init_emas()
        async with websockets.connect(uri) as ws:
            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                price = float(data['p'])
                trade_id = data['a']
                if price != self.price:
                    for span in self.ema_spans:
                        self.emas[span] = calc_new_ema(self.price,
                                                       price,
                                                       self.emas[span],
                                                       alpha=self.ema_alphas[span],
                                                       n_steps=trade_id - self.trade_id)
                if data['m']:
                    self.ob[0] = price
                else:
                    self.ob[1] = price
                self.trade_id = trade_id
                self.price = price
                if self.ts_locked['decide'] < self.ts_released['decide']:
                    asyncio.create_task(self.decide())
                elif self.trade_id % 10 == 0:
                    self.flush_stuck_locks()
                if self.stop_websocket:
                    break


async def main() -> None:
    bot = await create_bot(sys.argv[1], load_settings('binance_futures', sys.argv[1]))
    await start_bot(bot)


if __name__ == '__main__':
    asyncio.run(main())


