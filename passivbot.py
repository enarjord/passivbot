import ccxt.async_support as ccxt_async
import json
import os
import datetime
import numpy as np
import pandas as pd
import pprint
import asyncio
from time import time, sleep
from typing import Iterator


def calc_liq_price(amount: float, entry_price: float, leverage: float):
    if not entry_price:
        return 0.0
    cost = amount / entry_price
    margin = abs(cost / leverage)
    margin_plus_cost = margin + cost
    if not margin_plus_cost:
        return 0.0
    return amount / margin_plus_cost


def make_get_filepath(filepath: str) -> str:
    '''
    if not is path, creates dir and subdirs for path, returns path
    '''
    dirpath = os.path.dirname(filepath) if filepath[-1] != '/' else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_key_secret(exchange: str, user: str) -> (str, str):
    try:
        return json.load(open(f'api_key_secrets/{exchange}/{user}.json'))
    except(FileNotFoundError):
        print(f'\n\nPlease specify {exchange} API key/secret in file\n\napi_key_secre' + \
              f'ts/{exchange}/{user}.json\n\nformatted thus:\n["Ktnks95U...", "yDKRQqA6..."]\n\n')
        raise Exception('api key secret missing')


def init_ccxt(exchange: str = None, user: str = None):
    if user is None:
        return getattr(ccxt_async, exchange)
    try:
        return getattr(ccxt_async, exchange)({'apiKey': (ks := load_key_secret(exchange, user))[0],
                                              'secret': ks[1]})
    except Exception as e:
        print('error init ccxt', e)
        return getattr(ccxt_async, exchange)


def print_(args, r=False, n=False):
    line = ts_to_date(time())[:19] + '  '
    str_args = '{} ' * len(args)
    line += str_args.format(*args)
    if n:
        print('\n' + line, end=' ')
    elif r:
        print('\r' + line, end=' ')
    else:
        print(line)
    return line


def load_settings(exchange: str, user: str = 'default') -> dict:
    fpath = f'settings/{exchange}/'
    try:
        settings = json.load(open(f'{fpath}{user}.json'))
    except FileNotFoundError:
        print(f'settings for user {user} not found, using default settings')
        settings = json.load(open(f'{fpath}default.json'))
    print('\nloaded settings:')
    pprint.pprint(settings)
    return settings


def ts_to_date(timestamp: float) -> str:
    return str(datetime.datetime.fromtimestamp(timestamp)).replace(' ', 'T')


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


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


class Bot:
    def __init__(self, user: str, settings: dict):
        self.settings = settings
        self.user = user
        self.symbol = settings['symbol']
        self.ema_spans = settings['ema_spans']
        self.leverage = settings['leverage']
        self.markup = settings['markup']
        self.spread_minus = 1 - settings['spread']
        self.spread_plus = 1 + settings['spread']
        self.entry_amount = settings['entry_amount']
        self.enter_long = settings['enter_long']
        self.enter_shrt = settings['enter_shrt']
        self.ts_locked = {'cancel_orders': 0, 'decide': 0, 'update_open_orders': 0,
                          'update_position': 0, 'print': 0, 'create_orders': 0}
        self.ts_released = {k: 1 for k in self.ts_locked}

        self.position = {}
        self.open_orders = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0
        self.ema_alphas = {span: 2 / (span + 1) for span in self.ema_spans}
        self.ob = [0.0, 0.0]

        self.stop_websocket = False

    async def update_open_orders(self) -> None:
        if self.ts_locked['update_open_orders'] > self.ts_released['update_open_orders']:
            return
        self.open_orders = await self.fetch_open_orders()
        self.highest_bid, self.lowest_ask = 0.0, 9.9e9
        for o in self.open_orders:
            if o['side'] == 'buy':
                self.highest_bid = max(self.highest_bid, o['price'])
            elif o['side'] == 'sell':
                self.lowest_ask = min(self.lowest_ask, o['price'])
        self.ts_released['update_open_orders'] = time()

    async def update_position(self) -> None:
        # also updates open orders
        if self.ts_locked['update_position'] > self.ts_released['update_position']:
            return
        self.ts_locked['update_position'] = time()
        self.position, _ = await asyncio.gather(self.fetch_position(), self.update_open_orders())
        self.ts_released['update_position'] = time()

    async def create_orders(self, orders_to_create: [dict]) -> dict:
        if self.ts_locked['create_orders'] > self.ts_released['create_orders']:
            return
        self.ts_locked['create_orders'] = time()
        creations = []
        for oc in sorted(orders_to_create, key=lambda x: x['amount']):
            try:
                if oc['side'] == 'buy':
                    creations.append(self.execute_bid(oc['amount'], oc['price']))
                else:
                    creations.append(self.execute_ask(oc['amount'], oc['price']))
            except Exception as e:
                print('error creating orders a', orders_to_create, e)
        try:
            created_orders = await asyncio.gather(*creations)
        except Exception as e:
            print('error creating orders b', orders_to_create, e)
            created_orders = []
        for o in created_orders:
            try:
                print_([' created order', o['symbol'], o['side'], o['amount'], o['price']], n=True)
            except Exception as e:
                print('error creating orders c', orders_to_create, e)
        await self.update_position()
        self.ts_released['create_orders'] = time()
        return created_orders

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if self.ts_locked['cancel_orders'] > self.ts_released['cancel_orders']:
            return
        self.ts_locked['cancel_orders'] = time()
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletions.append(self.execute_cancellation(oc['order_id']))
            except Exception as e:
                print('error cancelling orders a', orders_to_cancel, e)
        try:
            canceled_orders = await asyncio.gather(*deletions)
        except Exception as e:
            print('error cancelling orders b', orders_to_cancel, e)
            canceled_orders = []
        for o in canceled_orders:
            try:
                print_(['canceled order', o['symbol'], o['side'], o['amount'], o['price']], n=True)
            except Exception as e:
                print('error cancelling orders c', orders_to_cancel, e)
        await self.update_open_orders()
        self.ts_released['cancel_orders'] = time()
        return canceled_orders

    async def init_emas(self) -> None:
        trades = await self.fetch_trades()
        additional_trades = await asyncio.gather(
            *[self.fetch_trades(from_id=trades[0]['trade_id'] - 1000 * i)
              for i in range(1, min(50, max(self.ema_spans) // 1000))])
        trades = sorted(trades + flatten(additional_trades), key=lambda x: x['trade_id'])
        emas = {span: trades[0]['price'] for span in self.ema_spans}
        for t in trades:
            for span in self.ema_spans:
                emas[span] = emas[span] * (1 - self.ema_alphas[span]) + \
                    t['price'] * self.ema_alphas[span]
        self.price = t['price']
        self.trade_id = t['trade_id']
        self.emas = emas
        self.ob = [self.price, self.price]

    def stop(self) -> None:
        self.stop_websocket = True

    def calc_orders(self):
        orders = []
        if self.position['size'] == 0:
            bid_price = min(self.round_dn(min(self.emas.values()) * self.spread_minus), self.ob[0])
            ask_price = max(self.round_up(max(self.emas.values()) * self.spread_plus), self.ob[1])
            bid_diff = self.price / bid_price
            ask_diff = ask_price / self.price
            threshold = 1.00007
            if bid_diff < ask_diff:
                if bid_diff < threshold:
                    orders.append({'symbol': self.symbol, 'side': 'buy',
                                   'amount': self.entry_amount, 'price': bid_price})
            else:
                if ask_diff < threshold:
                    orders.append({'symbol': self.symbol, 'side': 'sell',
                                   'amount': self.entry_amount, 'price': ask_price})
        else:
            available_balance = self.position['equity']
            if self.position['size'] > 0.0:
                entry_price = self.position['entry_price']
                ddown_amount = self.position['size']
                ddown_price = self.round_up(max(
                    entry_price * (1 - (1 / self.leverage) / 2),
                    self.position['liquidation_price'] + 0.000001
                ))

                for k in range(4):
                    margin_cost = self.calc_margin_cost(ddown_amount, ddown_price)
                    if margin_cost < available_balance:
                        orders.append({'side': 'buy', 'amount': ddown_amount,
                                       'price': ddown_price})
                        available_balance -= margin_cost
                    else:
                        break
                    entry_price = (ddown_price + entry_price) / 2
                    ddown_price = self.round_up(
                        entry_price * (1 - (1 / self.leverage) / 2)
                    )
                    ddown_amount *= 2
                orders.append({
                    'symbol': self.symbol, 'side': 'sell', 'amount': self.position['size'],
                    'price': self.round_up(self.position['entry_price'] * (1 + self.markup))
                })
            else:
                pos_size = abs(self.position['size'])
                ddown_amount = pos_size
                entry_price = self.position['entry_price']
                ddown_price = self.round_dn(min(
                    entry_price * (1 + (1 / self.leverage) / 2),
                    self.position['liquidation_price'] - 0.000001
                ))

                for k in range(4):
                    margin_cost = self.calc_margin_cost(ddown_amount, ddown_price)

                    if margin_cost < available_balance:
                        orders.append({'side': 'sell', 'amount': ddown_amount,
                                       'price': ddown_price})
                        available_balance -= margin_cost
                    else:
                        break
                    entry_price = (ddown_price + entry_price) / 2
                    ddown_price = self.round_dn(
                        entry_price * (1 + (1 / self.leverage) / 2)
                    )
                    ddown_amount *= 2
                orders.append({
                    'side': 'buy', 'amount': pos_size,
                    'price': self.round_dn(self.position['entry_price'] * (1 - self.markup))
                })
        return orders

    async def cancel_and_create(self):
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_orders(),
                                             keys=['side', 'amount', 'price'])
        tasks = []
        if to_cancel:
            tasks.append(self.cancel_orders(to_cancel))
        tasks.append(self.create_orders(to_create))
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
        await self.update_position()
        if any(results):
            print()
        return results


    async def decide(self):
        if self.price <= self.highest_bid:
            self.ts_locked['decide'] = time()
            print_(['bid maybe taken'], n=True)
            await self.update_position()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if self.price >= self.lowest_ask:
            self.ts_locked['decide'] = time()
            print_(['ask maybe taken'], n=True)
            await self.update_position()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_locked['decide'] > 5:
            self.ts_locked['decide'] = time()
            await self.update_position()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_released['print'] >= 0.5:
            self.ts_released['print'] = time()
            line = f"{self.symbol} "
            if self.position['size'] == 0:
                bid_price = self.round_dn(min(self.emas.values()) * self.spread_minus)
                ask_price = self.round_up(max(self.emas.values()) * self.spread_plus)
                line += f"no position bid {bid_price} ask {ask_price} "
                ratio = 0.0
            elif self.position['size'] > 0.0:
                line += f"long {self.position['size']} @ {self.position['entry_price']:.2f} "
                line += f"exit {self.lowest_ask} ddown {self.highest_bid} "
                ratio = (self.price - self.highest_bid) / (self.lowest_ask - self.highest_bid)
            else:
                line += f"shrt {self.position['size']} @ {self.position['entry_price']:.2f} "
                ratio = 1 - (self.price - self.highest_bid) / (self.lowest_ask - self.highest_bid)
                line += f"exit {self.highest_bid} ddown {self.lowest_ask } "

            line += f"pct {ratio:.2f} last {self.price}   "
            print_([line], r=True)


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

    def flush_stuck_locks(self, timeout: float = 4.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now


async def start_bot(bot, n_tries: int = 0) -> None:
    max_n_tries = 30
    try:
        await bot.start_websocket()
    except KeyboardInterrupt:
        await bot.cc.close()
    except Exception as e:
        await bot.cc.close()
        print(e)
        if n_tries >= max_n_tries:
            return
        n_tries += 1
        for k in range(10, -1, -1):
            print(f'\rrestarting bot in {k} seconds   ', end=' ')
            sleep(1)
        await start_bot(bot, n_tries + 1)

