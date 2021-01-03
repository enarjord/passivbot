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


def round_up(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.ceil(n / step) * step, safety_rounding)


def round_dn(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.floor(n / step) * step, safety_rounding)


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def calc_long_liq_price(price, leverage):
    return (price * leverage) / (leverage + 1)


def calc_shrt_liq_price(price, leverage):
    return (price * leverage) / (leverage - 1)


def calc_long_entry(min_qty: float,
                    qty_step: float,
                    price_step: float,
                    leverage: float,
                    entry_qty_scaling_factor: float,
                    grid_spacing: float,
                    entry_qty_equity_multiplier: float,
                    equity: float,
                    pos_size: float,
                    pos_price: float):
    qty = round_dn(min(equity * pos_price * leverage - pos_size,
                       max([pos_size * entry_qty_scaling_factor,
                            min_qty,
                            equity * entry_qty_equity_multiplier * pos_price])), qty_step)
    price = round_dn(pos_price * (1 - grid_spacing),
                     round_up(pos_price * grid_spacing / 4, price_step))
    return [qty, price]


def calc_shrt_entry(min_qty: float,
                    qty_step: float,
                    price_step: float,
                    leverage: float,
                    entry_qty_scaling_factor: float,
                    grid_spacing: float,
                    entry_qty_equity_multiplier: float,
                    equity: float,
                    pos_size: float,
                    pos_price: float):
    qty = -round_dn(min(equity * pos_price * leverage + pos_size,
                        max([-pos_size * entry_qty_scaling_factor,
                             min_qty,
                             equity * entry_qty_equity_multiplier * pos_price])), qty_step)
    price = round_up(pos_price * (1 + grid_spacing),
                     round_up(pos_price * grid_spacing / 4, price_step))
    return [qty, price]


def calc_initial_long_entry_qty(min_qty: float,
                                qty_step: float,
                                entry_qty_equity_multiplier: float,
                                equity: float,
                                price: float):
    return round_dn(max(min_qty, equity * entry_qty_equity_multiplier * price), qty_step)


def calc_initial_shrt_entry_qty(min_qty: float,
                                qty_step: float,
                                entry_qty_equity_multiplier: float,
                                equity: float,
                                price: float):
    return -round_dn(max(min_qty, equity * entry_qty_equity_multiplier * price), qty_step)


def calc_liq_price(qty: float, entry_price: float, leverage: float):
    if not entry_price:
        return 0.0
    cost = qty / entry_price
    margin = abs(cost / leverage)
    margin_plus_cost = margin + cost
    if not margin_plus_cost:
        return 0.0
    return qty / margin_plus_cost


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
                  keys: [str] = ['symbol', 'side', 'qty', 'price']) -> ([dict], [dict]):
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
        self.leverage = settings['leverage']
        self.ema_spans = settings['ema_spans']
        self.spread_minus = 1 - settings['ema_spread'] / 2
        self.spread_plus = 1 + settings['ema_spread'] / 2
        self.entry_qty_equity_multiplier = settings['entry_qty_equity_multiplier']
        self.entry_qty_scaling_factor = settings['entry_qty_scaling_factor']
        self.grid_spacing = settings['grid_spacing']
        self.markup = settings['markup']
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
        try:
            self.open_orders = await self.fetch_open_orders()
        except Exception as e:
            print('error with update open orders', e)
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
        try:
            self.position, _ = await asyncio.gather(self.fetch_position(),
                                                    self.update_open_orders())
        except Exception as e:
            print('error with update position', e)
        self.ts_released['update_position'] = time()

    async def create_orders(self, orders_to_create: [dict]) -> dict:
        if self.ts_locked['create_orders'] > self.ts_released['create_orders']:
            return
        self.ts_locked['create_orders'] = time()
        creations = []
        for oc in sorted(orders_to_create, key=lambda x: x['qty']):
            try:
                if oc['side'] == 'buy':
                    creations.append(self.execute_bid(oc['qty'], oc['price']))
                else:
                    creations.append(self.execute_ask(oc['qty'], oc['price']))
            except Exception as e:
                print('error creating orders a', orders_to_create, e)
        try:
            created_orders = await asyncio.gather(*creations)
        except Exception as e:
            print('error creating orders b', orders_to_create, e)
            created_orders = []
        for o in created_orders:
            try:
                print_([' created order', o['symbol'], o['side'], o['qty'], o['price']], n=True)
            except Exception as e:
                print('error creating orders c', orders_to_create, e)
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
                print_(['canceled order', o['symbol'], o['side'], o['qty'], o['price']], n=True)
            except Exception as e:
                print('error cancelling orders c', orders_to_cancel, e)
        await self.update_open_orders()
        self.ts_released['cancel_orders'] = time()
        return canceled_orders

    async def init_emas(self) -> None:
        trades = await self.fetch_trades()
        additional_trades = await asyncio.gather(
            *[self.fetch_trades(from_id=trades[0]['trade_id'] - 1000 * i)
              for i in range(1, min(50, (max(self.ema_spans) // 1000) * 2))])
        trades = sorted(trades + flatten(additional_trades), key=lambda x: x['trade_id'])
        emas = {span: trades[0]['price'] for span in self.ema_spans}
        for i in range(1, len(trades)):
            if trades[i]['price'] == trades[i-1]['price']:
                continue
            for span in self.ema_spans:
                emas[span] = emas[span] * (1 - self.ema_alphas[span]) + \
                    trades[i]['price'] * self.ema_alphas[span]
        self.price = trades[-1]['price']
        self.trade_id = trades[-1]['trade_id']
        self.emas = emas
        self.ob = [self.price, self.price]

    def stop(self) -> None:
        self.stop_websocket = True

    def calc_orders(self):
        n_orders = 11
        orders = []
        if self.position['size'] == 0:
            bid_price = min(self.prdn(min(self.emas.values()) * self.spread_minus), self.ob[0])
            ask_price = max(self.prup(max(self.emas.values()) * self.spread_plus), self.ob[1])
            bid_diff = self.price / bid_price
            ask_diff = ask_price / self.price
            threshold = 1.00007
            if bid_diff < ask_diff:
                if bid_diff < threshold:
                    bid_qty = calc_initial_long_entry_qty(self.min_qty,
                                                          self.qty_step,
                                                          self.entry_qty_equity_multiplier,
                                                          self.position['equity'],
                                                          self.ob[0])
                    if bid_qty >= self.min_qty:
                        orders.append({'side': 'buy', 'qty': bid_qty, 'price': bid_price})
            else:
                if ask_diff < threshold:
                    ask_qty = -calc_initial_shrt_entry_qty(self.min_qty,
                                                           self.qty_step,
                                                           self.entry_qty_equity_multiplier,
                                                           self.position['equity'],
                                                           self.ob[1])
                    if ask_qty >= self.min_qty:
                        orders.append({'side': 'sell', 'qty': ask_qty, 'price': ask_price})
        else:
            pos_size = self.position['size']
            pos_price = self.position['entry_price']
            if self.position['size'] > 0.0:
                for k in range(n_orders):
                    bid_qty, bid_price = calc_long_entry(
                        self.min_qty,
                        self.qty_step,
                        self.price_step,
                        self.leverage,
                        self.entry_qty_scaling_factor,
                        self.grid_spacing,
                        self.entry_qty_equity_multiplier,
                        self.position['equity'] * 0.95,
                        pos_size,
                        pos_price
                    )
                    if bid_qty < self.min_qty:
                        break
                    if bid_price <= self.ob[0]:
                        orders.append({'side': 'buy', 'qty': bid_qty, 'price': bid_price})
                    new_pos_size = pos_size + bid_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        bid_price * (bid_qty / new_pos_size)
                    pos_size = new_pos_size
                orders.append({
                    'side': 'sell', 'qty': self.position['size'],
                    'price': max(self.prup(self.position['entry_price'] * (1 + self.markup)),
                                 self.ob[1])
                })
            else:
                for k in range(n_orders):
                    ask_qty, ask_price = calc_shrt_entry(
                        self.min_qty,
                        self.qty_step,
                        self.price_step,
                        self.leverage,
                        self.entry_qty_scaling_factor,
                        self.grid_spacing,
                        self.entry_qty_equity_multiplier,
                        self.position['equity'] * 0.95,
                        pos_size,
                        pos_price
                    )
                    if -ask_qty < self.min_qty:
                        break
                    if ask_price >= self.ob[1]:
                        orders.append({'side': 'sell', 'qty': -ask_qty, 'price': ask_price})
                    new_pos_size = pos_size + ask_qty
                    pos_price = pos_price * (pos_size / new_pos_size) + \
                        ask_price * (ask_qty / new_pos_size)
                    pos_size = new_pos_size
                orders.append({
                    'side': 'buy', 'qty': -self.position['size'],
                    'price': min(self.prdn(self.position['entry_price'] * (1 - self.markup)),
                                 self.ob[0])
                })
        return orders

    async def cancel_and_create(self):
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_orders(),
                                             keys=['side', 'qty', 'price'])
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
                bid_price = self.prdn(min(self.emas.values()) * self.spread_minus)
                ask_price = self.prup(max(self.emas.values()) * self.spread_plus)
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
                 'qty': float(mt['qty']),
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

