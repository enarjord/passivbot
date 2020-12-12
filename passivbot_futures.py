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

import ccxt.async_support as ccxt_async


def filter_orders(actual_orders: [dict],
                  ideal_orders: [dict],
                  keys: [str] = ['symbol', 'side', 'amount', 'price']) -> ([dict], [dict]):
    # returns (orders_to_delete, orders_to_create)

    if not actual_orders:
        return [], ideal_orders
    if not ideal_orders:
        return actual_orders, []
    actual_orders = actual_orders.copy()
    orders_to_create = []
    ideal_orders_cropped = [{k: o[k] for k in keys} for o in ideal_orders]
    actual_orders_cropped = [{k: o[k] for k in keys} for o in actual_orders]
    for ioc, io in zip(ideal_orders_cropped, ideal_orders):
        matches = [(aoc, ao) for aoc, ao in zip(actual_orders_cropped, actual_orders) if aoc == ioc]
        if matches:
            actual_orders.remove(matches[0][1])
            actual_orders_cropped.remove(matches[0][0])
        else:
            orders_to_create.append(io)
    return actual_orders, orders_to_create


def calc_new_ema(prev_val: float,
                 new_val: float,
                 prev_ema: float,
                 span: float = None,
                 alpha: float = None,
                 n_steps: int = 1) -> float:
    if alpha is None:
        if span is None:
            raise Exception('please specify alpha or span')
        alpha = 2 / (span + 1)
    if n_steps == 1:
        return prev_ema * (1 - alpha) + new_val * alpha
    elif n_steps <= 0:
        return prev_ema
    else:
        return calc_new_ema(prev_val,
                            new_val,
                            prev_ema * (1 - alpha) + prev_val * alpha,
                            alpha=alpha,
                            n_steps=n_steps - 1)



def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


def round_up(n: float, d: int = 0):
    return ceil(n * (dexp := 10**d)) / dexp


def round_dn(n: float, d: int = 0):
    return floor(n * (dexp := 10**d)) / dexp


def make_get_filepath(filepath: str) -> str:
    '''
    if not is path, creates dir and subdirs for path, returns path
    '''
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        line = '\n' + line
    if r:
        sys.stdout.write('\r' + line + '   ')
    else:
        print(line)
    sys.stdout.flush()
    return line


def load_key_secret(exchange: str, user: str) -> (str, str):
    try:
        return json.load(open(f'api_key_secrets/{exchange}/{user}.json'))
    except(FileNotFoundError):
        print(f'\n\nPlease specify {exchange} API key/secret in file\n\napi_key_secre' + \
              f'ts/{exchange}/{user}.json\n\nformatted thus:\n["Ktnks95U...", "yDKRQqA6..."]\n\n')
        raise Exception('api key secret missing')


def load_settings(user: str = 'default') -> dict:
    fpath = 'settings/binance_futures/'
    try:
        settings = json.load(open(f'{fpath}{user}.json'))
    except FileNotFoundError:
        print(f'settings for user {user} not found, using default settings')
        settings = json.load(open(f'{fpath}default.json'))
    print('\nloaded settings:')
    pprint.pprint(settings)
    return settings


async def create_bot(user: str, settings: str):
    bot = Bot(user, settings)
    await bot._init()
    return bot


class Bot:
    def __init__(self, user: str, settings: dict):
        self.settings = settings
        self.user = user
        self.symbol = settings['symbol']
        self.ema_span = settings['ema_span']
        self.markup = settings['markup']
        self.entry_amount = settings['entry_amount']
        self.flashcrash_factor = settings['flashcrash_factor']
        self.leverage = settings['leverage']

        self.cc = ccxt_async.binance({'apiKey': (ks := load_key_secret('binance', user))[0],
                                      'secret': ks[1]})

        self.ts_locked = {'create_bid': 0, 'create_ask': 0, 'cancel_all_open_orders': 0,
                          'verify_orders': 0, 'decide': 0, 'update_state': 0, 'order_taken': 0}
        self.ts_released = {k: 1 for k in self.ts_locked}

        self.positions = {}
        self.open_orders = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.ema = 0
        self.agg_id = 0
        self.price = 0
        self.ema_alpha = 2 / (self.ema_span + 1)
        self.ema_alpha_ = 1 - self.ema_alpha
        self.bid_ema_multiplier = 1 - self.flashcrash_factor
        self.ask_ema_multiplier = 1 + self.flashcrash_factor
        self.bid_trigger_ema_multiplier = 1 - self.flashcrash_factor * 0.9
        self.ask_trigger_ema_multiplier = 1 + self.flashcrash_factor * 0.9

        self.exit_price = 0.0
        self.double_down_price = 0.0

        self.stop_websocket = False

    async def _init(self):
        print(await self.cc.fapiPrivate_post_leverage(params={'symbol': self.symbol,
                                                              'leverage': self.leverage}))
        exchange_info = await self.cc.fapiPublic_get_exchangeinfo()
        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.price_precision = e['pricePrecision']
                self.amount_precision = e['quantityPrecision']
                break
        await self.update_state()

    async def update_open_orders(self) -> None:
        open_orders = await self.cc.fapiPrivate_get_openorders()
        self.open_orders = []
        self.highest_bid, self.lowest_ask = 0.0, 9.9e9
        for e in open_orders:
            if e['symbol'] != self.symbol:
                continue
            self.open_orders.append({
                'orderId': int(e['orderId']),
                'symbol': e['symbol'],
                'status': e['status'],
                'clientOrderId': e['clientOrderId'],
                'price': float(e['price']),
                'avgPrice': float(e['avgPrice']),
                'origQty': float(e['origQty']),
                'executedQty': float(e['executedQty']),
                'cumQuote': float(e['cumQuote']),
                'timeInForce': e['timeInForce'],
                'type': e['type'],
                'reduceOnly': e['reduceOnly'],
                'closePosition': e['closePosition'],
                'side': e['side'],
                'positionSide': e['positionSide'],
                'stopPrice': float(e['stopPrice']),
                'workingType': e['workingType'],
                'priceProtect': e['priceProtect'],
                'origType': e['origType'],
                'time': int(e['time']),
                'updateTime': int(e['updateTime'])
            })
            if self.open_orders[-1]['side'] == 'BUY':
                self.highest_bid = max(self.highest_bid, self.open_orders[-1]['price'])
            else:
                self.lowest_ask = min(self.lowest_ask, self.open_orders[-1]['price'])


    async def update_state(self) -> None:
        if self.ts_locked['update_state'] > self.ts_released['update_state']:
            return
        self.ts_locked['update_state'] = time()
        positions, account, open_orders = await asyncio.gather(
            self.cc.fapiPrivate_get_positionrisk(),
            self.cc.fapiPrivate_get_account(),
            self.update_open_orders()
        )

        self.positions = {e['symbol']: {
            'symbol': e['symbol'],
            'positionAmt': float(e['positionAmt']),
            'entryPrice': entry_price,
            'markPrice': float(e['markPrice']),
            'unRealizedProfit': float(e['unRealizedProfit']),
            'liquidationPrice': float(e['liquidationPrice']),
            'leverage': int(e['leverage']),
            'maxNotionalValue': int(e['maxNotionalValue']),
            'marginType': e['marginType'],
            'isolatedMargin': float(e['isolatedMargin']),
            'isAutoAddMargin': e['isAutoAddMargin'] == 'true',
            'positionSide': e['positionSide']
        } for e in positions if (entry_price := float(e['entryPrice'])) != 0.0}

        self.account = {'available_long_balance': float(account['availableBalance'])}
        self.account['available_shrt_balance'] = self.account['available_long_balance']
        for e in account['positions']:
            if e['symbol'] == self.symbol:
                self.account['leverage'] = float(e['leverage'])
                if self.symbol in self.positions:
                    if self.positions[self.symbol]['positionAmt'] > 0.0:
                        self.account['available_shrt_balance'] += float(e['positionInitialMargin'])
                    else:
                        self.account['available_long_balance'] += float(e['positionInitialMargin'])
                break
        self.ts_released['update_state'] = time()

    async def create_bid(self, symbol: str, amount: float, price: float) -> dict:
        if self.ts_locked['create_bid'] > self.ts_released['create_bid']:
            return
        self.ts_locked['create_bid'] = time()
        try:
            assert self.account['available_long_balance'] * self.account['leverage'] / price > amount
            o = await self.cc.fapiPrivate_post_order(params={
                'symbol': symbol,
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': amount,
                'price': price,
                'timeInForce': 'GTC'})
            print_([' created order', symbol, o['side'], o['origQty'], o['price'], '\n'], r=True)
            await self.update_state()
        except Exception as e:
            if e.args:
                print(e)
            o = {}
        self.ts_released['create_bid'] = time()
        return o

    async def create_ask(self, symbol: str, amount: float, price: float) -> dict:
        if self.ts_locked['create_ask'] > self.ts_released['create_ask']:
            return
        self.ts_locked['create_ask'] = time()
        try:
            print(self.account['available_shrt_balance'] * self.account['leverage'] / price)
            assert self.account['available_shrt_balance'] * self.account['leverage'] / price > amount
            o = await self.cc.fapiPrivate_post_order(params={
                'symbol': symbol,
                'side': 'SELL',
                'type': 'LIMIT',
                'quantity': amount,
                'price': price,
                'timeInForce': 'GTC'})
            print_([' created order', symbol, o['side'], o['origQty'], o['price'], '\n'], r=True)
            await self.update_state()
        except Exception as e:
            if e.args:
                print(e)
            o = {}
        self.ts_released['create_ask'] = time()
        return o

    async def cancel_open_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if self.ts_locked['cancel_all_open_orders'] > self.ts_released['cancel_all_open_orders']:
            return
        self.ts_locked['cancel_all_open_orders'] = time()
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletion = self.cc.fapiPrivate_delete_order(params={
                    'symbol': oc['symbol'], 'orderId': oc['orderId']
                })
                deletions.append(deletion)
            except Exception as e:
                print(e)
        canceled_orders = await asyncio.gather(*deletions)
        for o in canceled_orders:
            try:
                print_(['canceled order', o['symbol'], o['side'], o['origQty'], o['price'], '\n'],
                       r=True)
            except Exception as e:
                print(e)
                continue
        await self.update_open_orders()
        self.ts_released['cancel_all_open_orders'] = time()
        return canceled_orders

    async def init_ema(self) -> None:
        agg_trades = await self.fetch_trades(self.symbol)
        additional_agg_trades = await asyncio.gather(
            *[self.fetch_trades(self.symbol, from_id=agg_trades[0]['agg_id'] - 1000 * i)
              for i in range(1, 10)])
        agg_trades = sorted(agg_trades + flatten(additional_agg_trades), key=lambda x: x['agg_id'])
        ema = agg_trades[0]['price']
        for t in agg_trades:
            ema = ema * self.ema_alpha_ + t['price'] * self.ema_alpha
            r = t['price'] / ema
        self.agg_id = t['agg_id']
        self.price = t['price']
        self.ema = ema

    def stop(self) -> None:
        self.stop_websocket = True

    def flush_stuck_locks(self, timeout: float = 3.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now

    def calc_exit_double_down(self) -> [dict]:
        try:
            pos = self.positions[self.symbol]
        except:
            return []
        if pos['positionAmt'] > 0.0:
            ideal_bid_price = round_up(max(pos['liquidationPrice'],
                                           pos['entryPrice'] * (1 - (1 / pos['leverage']) / 2)),
                                       self.price_precision)
            ideal_ask_price = round(pos['entryPrice'] * (1 + self.markup), self.price_precision)
            self.exit_price = ideal_ask_price
            self.double_down_price = ideal_bid_price
        else:
            ideal_ask_price = round_dn(min(pos['liquidationPrice'],
                                           pos['entryPrice'] * (1 + (1 / pos['leverage']) / 2)),
                                       self.price_precision)
            ideal_bid_price = round(pos['entryPrice'] * (1 - self.markup), self.price_precision)
            self.exit_price = ideal_bid_price
            self.double_down_price = ideal_ask_price
        amount = abs(pos['positionAmt'])
        return [{'side': 'BUY', 'origQty': amount, 'price': ideal_bid_price},
                {'side': 'SELL', 'origQty': amount, 'price': ideal_ask_price}]

    async def create_exits(self) -> list:
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_exit_double_down(),
                                             keys=['side', 'origQty', 'price'])
        tasks = []
        if to_cancel:
            tasks.append(self.cancel_open_orders(to_cancel))
        for o in to_create:
            if o['side'] == 'BUY':
                tasks.append(self.create_bid(self.symbol, o['origQty'], o['price']))
            elif o['side'] == 'SELL':
                tasks.append(self.create_ask(self.symbol, o['origQty'], o['price']))
        results = await asyncio.gather(*tasks)
        return results

    def check_if_order_taken(self):
        if self.price <= self.highest_bid:
            self.ts_released['order_taken'] = time()
            print('\nbid maybe taken')
        elif self.price >= self.lowest_ask:
            self.ts_released['order_taken'] = time()
            print('\nask maybe taken')

    async def decide(self) -> None:
        self.check_if_order_taken()
        if 0 < self.ts_locked['decide'] - self.ts_released['decide'] < 2:
            return
        self.ts_locked['decide'] = time()
        if time() - self.ts_released['order_taken'] < 2.0:
            await asyncio.sleep(0.1)
            await self.update_state()
            await self.create_exits()
            self.ts_released['decide'] = time()
            return
        elif self.symbol not in self.positions:
            if self.price <= self.ema * self.bid_trigger_ema_multiplier:
                await self.create_bid(self.symbol,
                                      self.entry_amount,
                                      round_dn(self.ema * self.bid_ema_multiplier,
                                               self.price_precision))
                await asyncio.sleep(0.25)
                await self.update_state()
                await self.create_exits()
                self.ts_released['decide'] = time()
                return
            elif self.price >= self.ema * self.ask_trigger_ema_multiplier:
                await self.create_ask(self.symbol,
                                      self.entry_amount,
                                      round_up(self.ema * self.ask_ema_multiplier,
                                               self.price_precision))
                await asyncio.sleep(0.25)
                await self.update_state()
                await self.create_exits()
                self.ts_released['decide'] = time()
                return
        if time() - self.ts_released['verify_orders'] > 1:
            self.ts_released['verify_orders'] = time()
            self.flush_stuck_locks()
            await self.create_exits()
            if time() - self.ts_released['update_state'] > 5:
                await self.update_state()
            line = f"{self.symbol} "
            if self.symbol in self.positions:
                if self.positions[self.symbol]['positionAmt'] > 0.0:
                    line += f"long {self.positions[self.symbol]['positionAmt']} @ "
                else:
                    line += f"shrt {abs(self.positions[self.symbol]['positionAmt'])} @ "
                line += f"{self.positions[self.symbol]['entryPrice']} "
                line += f"exit {self.exit_price} ddown {self.double_down_price} "
            else:
                line += f'no pos '
                line += f"bid {self.ema * self.bid_ema_multiplier:.{self.price_precision}f} "
                line += f"ask {self.ema * self.ask_ema_multiplier:.{self.price_precision}f} "
            line += f'last {self.price:.{self.price_precision}f} '
            print_([line], r=True)
        self.ts_released['decide'] = time()

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
        print_([uri])
        await self.update_state()
        await self.init_ema()
        async with websockets.connect(uri) as ws:
            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                price = float(data['p'])
                agg_id = data['a']
                self.ema = calc_new_ema(self.price,
                                        price,
                                        self.ema,
                                        alpha=self.ema_alpha,
                                        n_steps=agg_id - self.agg_id)
                self.agg_id = agg_id
                self.price = price
                self.ratio = price / self.ema
                await self.decide()
                if self.stop_websocket:
                    break

    async def fetch_trades(self, symbol: str, from_id: int = None) -> [dict]:
        params = {'symbol': symbol, 'limit': 1000}
        if from_id:
            params['fromId'] = from_id
        fetched_trades = await self.cc.fapiPublic_get_aggtrades(params=params)
        trades = [{'agg_id': t['a'],
                   'price': float(t['p']),
                   'amount': float(t['q']),
                   'timestamp': t['T'],
                   'is_buyer_maker': t['m']} for t in fetched_trades]
        print_(['fetched trades', symbol, trades[0]['agg_id'],
                ts_to_date(trades[0]['timestamp'] / 1000)])
        return trades

    def iter_chunks(self, symbol: str) -> Iterator[pd.DataFrame]:
        chunk_size = 100000
        filepath = f'historical_data/agg_trades_futures/{symbol}/'
        if os.path.isdir(filepath):
            filenames = sorted([f for f in os.listdir(filepath) if f.endswith('.csv')])
            for f in filenames[::-1]:
                chunk = pd.read_csv(filepath + f).set_index('agg_id')
                if chunk is not None:
                    print('loaded chunk of trades', f, ts_to_date(chunk.timestamp.iloc[0] / 1000))
                    yield chunk
                else:
                    yield None
            yield None
        else:
            yield None

    async def load_trades(self, symbol: str, n_days: float) -> pd.DataFrame:
        filepath = make_get_filepath(f'historical_data/agg_trades_futures/{symbol}/')
        age_limit = time() - 60 * 60 * 24 * n_days
        age_limit_millis = age_limit * 1000
        print('age_limit', ts_to_date(age_limit))
        chunk_iterator = self.iter_chunks(symbol)
        chunk = next(chunk_iterator)
        chunks = {} if chunk is None else {int(chunk.index[0]): chunk}
        ids = set() if chunk is None else set(chunk.index)
        min_id = min(ids) if ids else 0
        new_trades = await self.fetch_trades(symbol)
        while True:
            if new_trades[0]['timestamp'] <= age_limit_millis:
                break
            from_id = new_trades[0]['agg_id'] - 1
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
            new_trades = await self.fetch_trades(symbol, from_id=from_id) + new_trades
        new_trades_df = pd.DataFrame(new_trades).set_index('agg_id')
        trades_updated = pd.concat(list(chunks.values()) + [new_trades_df], axis=0)
        no_dup = trades_updated[~trades_updated.index.duplicated()]
        no_dup_sorted = no_dup.sort_index()
        chunk_size = 100000
        chunk_ids = no_dup_sorted.index // chunk_size * chunk_size
        for g in no_dup_sorted.groupby(chunk_ids):
            if g[0] not in chunks or len(chunks[g[0]]) != chunk_size:
                print('dumping chunk', g[0])
                g[1].to_csv(f'{filepath}{str(g[0])}.csv')
        return no_dup_sorted[no_dup_sorted.timestamp >= age_limit_millis]

    async def fetch_my_trades(self, symbol: str) -> [dict]:
        my_trades = await self.cc.fapiPrivate_get_usertrades(params={'symbol': symbol})
        return [{'symbol': mt['symbol'],
                 'id': mt['id'],
                 'orderId': mt['orderId'],
                 'side': mt['side'],
                 'price': float(mt['price']),
                 'amount': float(mt['qty']),
                 'realizedPnl': float(mt['realizedPnl']),
                 'marginAsset': mt['marginAsset'],
                 'quoteQty': float(mt['quoteQty']),
                 'commission': float(mt['commission']),
                 'commissionAsset': mt['commissionAsset'],
                 'timestamp': mt['time'],
                 'positionSide': mt['positionSide'],
                 'maker': mt['maker'],
                 'buyer': mt['maker']} for mt in my_trades]


def backtest(adf: pd.DataFrame, settings: dict) -> ([dict], [dict], pd.DataFrame):
    flashcrash_factor = settings['flashcrash_factor']
    ema_span = settings['ema_span']
    markup = settings['markup']
    leverage = settings['leverage']
    entry_amount = settings['entry_amount']

    thp = 1 + flashcrash_factor
    thm = 1 - flashcrash_factor
    maker_fee = round(0.018 * 0.01, 8)
    taker_fee = round(0.036 * 0.01, 8)
    print('maker_fee, taker_fee', maker_fee, taker_fee)
    roe = markup * leverage

    enter_long = True
    enter_shrt = False


    max_n_double_downs = 20
    max_margin = 10000

    liq_multiplier = (1 / leverage) / 2
    #reentry_markup = liq_multiplier / 2
    reentry_markup = liq_multiplier * 2
    print('roe', roe)
    print('liq_multiplier', liq_multiplier)
    print('max n double downs', max_n_double_downs)
    print('max_margin', max_margin)

    pos_amount = 0.0
    entry_price = 0.0
    liq_price = 0.0
    exit_price = 0.0
    double_down_price = 0.0

    realized_pnl_sum = 0.0
    n_double_downs = 0
    initial_margin_max = 0

    trades = []
    logs = []

    if 'ratio' not in adf.columns:
        ema = adf.price.ewm(span=ema_span, adjust=False).mean()
        ema.name = 'ema'
        ratio = adf.price / ema
        ratio.name = 'ratio'
        adf_ = adf.join(ema).join(ratio)
    else:
        adf_ = adf

    idxrange = adf.index[-1] - adf.index[0]

    for row in adf_.itertuples():
        if pos_amount == 0.0:
            # no position
            if enter_shrt and row.ratio > thp:
                pos_amount = -entry_amount
                entry_price = row.price
                liq_price = entry_price * (1 + liq_multiplier)
                exit_price = entry_price * (1 - markup)
                reentry_price = entry_price * (1 + reentry_markup)
                double_down_price = liq_price
                realized_pnl_sum -= entry_amount * row.price * taker_fee
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'entry',
                               'agg_id': row.Index, 'price': row.price, 'amount': -entry_amount})
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price})
            elif enter_long and row.ratio < thm:
                pos_amount = entry_amount
                entry_price = row.price
                liq_price = entry_price * (1 - liq_multiplier)
                exit_price = entry_price * (1 + markup)
                reentry_price = entry_price * (1 - reentry_markup)
                double_down_price = liq_price
                realized_pnl_sum -= entry_amount * row.price * taker_fee
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'entry',
                               'agg_id': row.Index, 'price': row.price, 'amount': entry_amount})
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price})
        elif pos_amount > 0.0:
            # long position
            if row.price >= exit_price:
                initial_margin = exit_price * pos_amount / leverage
                initial_margin_max = max(initial_margin_max, initial_margin)
                realized_pnl = initial_margin * roe
                realized_pnl_sum += realized_pnl - pos_amount * exit_price * maker_fee
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'exit',
                               'agg_id': row.Index, 'price': row.price, 'amount': pos_amount,
                               'realized_pnl': realized_pnl})
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price,
                             'realized_pnl_sum': realized_pnl_sum,
                             'initial_margin': initial_margin})

                (pos_amount, entry_price, liq_price, exit_price,
                 double_down_price, n_double_downs, reentry_price) = \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
            elif row.price <= double_down_price:
                if n_double_downs > max_n_double_downs or \
                        pos_amount * double_down_price / leverage > max_margin:
                    print('liquidation')
                    trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'liquidation',
                                   'agg_id': row.Index, 'price': double_down_price,
                                   'amount': pos_amount})
                    realized_pnl = pos_amount * double_down_price / leverage
                    realized_pnl_sum -= realized_pnl
                    (pos_amount, entry_price, liq_price, exit_price,
                     double_down_price, n_double_downs, reentry_price) = \
                        0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
                else:
                    n_double_downs += 1
                    trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'entry',
                                   'agg_id': row.Index, 'price': row.price, 'amount': pos_amount})
                    realized_pnl_sum -= pos_amount * double_down_price * maker_fee
                    pos_amount *= 2
                    entry_price = (entry_price + double_down_price) / 2
                    liq_price = entry_price * (1 - liq_multiplier)
                    exit_price = entry_price * (1 + markup)
                    reentry_price = entry_price * (1 - reentry_markup)
                    double_down_price = liq_price
                    line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                    line += f'{initial_margin_max:.2f} '
                    line += f'pos_amount {pos_amount}   '
                    sys.stdout.write(line)
                    logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                                 'initial_margin_max': initial_margin_max,
                                 'pos_amount': pos_amount, 'entry_price': entry_price,
                                 'liq_price': liq_price, 'exit_price': exit_price})
            elif row.price <= reentry_price and \
                    pos_amount * double_down_price / leverage < max_margin:
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'reentry',
                               'agg_id': row.Index, 'price': reentry_price,
                               'amount': entry_amount})
                realized_pnl_sum -= entry_amount * reentry_price * maker_fee
                new_pos_amount = pos_amount + entry_amount
                entry_price = (entry_price * (pos_amount / new_pos_amount) +
                               reentry_price * (entry_amount / new_pos_amount))
                pos_amount = new_pos_amount
                liq_price = entry_price * (1 - liq_multiplier)
                exit_price = entry_price * (1 + markup)
                reentry_price = entry_price * (1 - reentry_markup)
                double_down_price = liq_price
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price})

        else:
            # shrt position
            if row.price <= exit_price:
                initial_margin = exit_price * (-pos_amount) / leverage
                initial_margin_max = max(initial_margin_max, initial_margin)
                realized_pnl = initial_margin * roe
                realized_pnl_sum += realized_pnl - abs(pos_amount) * exit_price * maker_fee
                trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'exit',
                               'agg_id': row.Index, 'price': row.price, 'amount': pos_amount,
                               'realized_pnl': realized_pnl})
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price,
                             'realized_pnl_sum': realized_pnl_sum,
                             'initial_margin': initial_margin})

                (pos_amount, entry_price, liq_price, exit_price,
                 double_down_price, n_double_downs, reentry_price) = \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
            elif row.price >= double_down_price:
                if n_double_downs > max_n_double_downs or \
                        abs(pos_amount) * double_down_price / leverage > max_margin:
                    trades.append({'timestamp': row.timestamp, 'side': 'buy', 'type': 'liquidation',
                                   'agg_id': row.Index, 'price': double_down_price,
                                   'amount': pos_amount})
                    print('liquidation')
                    realized_pnl = abs(pos_amount) * double_down_price / leverage
                    realized_pnl_sum -= realized_pnl
                    (pos_amount, entry_price, liq_price, exit_price,
                     double_down_price, n_double_downs, reentry_price) = \
                        0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0
                else:
                    n_double_downs += 1
                    trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'entry',
                                   'agg_id': row.Index, 'price': row.price, 'amount': pos_amount})
                    realized_pnl_sum -= abs(pos_amount) * double_down_price * maker_fee
                    pos_amount *= 2
                    entry_price = (entry_price + double_down_price) / 2
                    liq_price = entry_price * (1 + liq_multiplier)
                    exit_price = entry_price * (1 - markup)
                    reentry_price = entry_price * (1 + reentry_markup)
                    double_down_price = liq_price
                    line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                    line += f'{initial_margin_max:.2f} '
                    line += f'pos_amount {pos_amount}   '
                    sys.stdout.write(line)
                    logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                                 'initial_margin_max': initial_margin_max,
                                 'pos_amount': pos_amount, 'entry_price': entry_price,
                                 'liq_price': liq_price, 'exit_price': exit_price})
            elif row.price >= reentry_price:
                trades.append({'timestamp': row.timestamp, 'side': 'sel', 'type': 'reentry',
                               'agg_id': row.Index, 'price': reentry_price,
                               'amount': -entry_amount})
                abs_pos_amount = abs(pos_amount)
                realized_pnl_sum -= entry_amount * reentry_price * maker_fee
                new_pos_amount = abs_pos_amount + entry_amount
                entry_price = (entry_price * (abs_pos_amount / new_pos_amount) +
                               reentry_price * (entry_amount / new_pos_amount))
                pos_amount = -new_pos_amount
                liq_price = entry_price * (1 + liq_multiplier)
                exit_price = entry_price * (1 - markup)
                reentry_price = entry_price * (1 + reentry_markup)
                double_down_price = liq_price
                line = f'\r{(row.Index - adf.index[0]) / idxrange:.4f} {realized_pnl_sum:.2f} '
                line += f'{initial_margin_max:.2f} '
                line += f'pos_amount {pos_amount}   '
                sys.stdout.write(line)
                logs.append({'timestamp': row.timestamp, 'agg_id': row.Index,
                             'initial_margin_max': initial_margin_max,
                             'pos_amount': pos_amount, 'entry_price': entry_price,
                             'liq_price': liq_price, 'exit_price': exit_price})

    return logs, trades, adf_


async def main() -> None:
    await start_bot()


async def start_bot(n_tries: int = 0) -> None:
    user = sys.argv[1]
    settings = load_settings(user)
    max_n_tries = 10
    try:
        bot = await create_bot(user, settings)
        await bot.start_websocket()
    except KeyboardInterrupt:
        await bot.cc.close()
    except Exception as e:
        await bot.cc.close()
        print(e)
        if n_tries >= max_n_tries:
            return
        n_tries += 1
        for k in range(60, -1, -1):
            sys.stdout.write(f'\rrestarting bot in {k} seconds   ')
            sleep(1)
        await start_bot(n_tries + 1)


if __name__ == '__main__':
    asyncio.run(main())


