import asyncio
import datetime
import json
import logging
import os
import signal
import sys
from collections import deque
from pathlib import Path
from time import time

import numpy as np

import telegram_bot
import websockets

from jitted import round_, calc_diff, calc_ema, calc_cost, iter_entries, iter_long_closes, \
    iter_shrt_closes, round_dynamic, compress_float

logging.getLogger("telegram").setLevel(logging.CRITICAL)

def get_keys():
    return ['inverse', 'do_long', 'do_shrt', 'qty_step', 'price_step', 'min_qty', 'min_cost',
            'contract_multiplier', 'ddown_factor', 'qty_pct', 'leverage', 'n_close_orders',
            'grid_spacing', 'pos_margin_grid_coeff', 'volatility_grid_coeff',
            'volatility_qty_coeff', 'min_markup', 'markup_range', 'ema_span', 'ema_spread',
            'stop_loss_liq_diff', 'stop_loss_pos_pct']


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if type(v) == dict:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
        keyfile = json.load(open('api-keys.json'))
        # Checks that the user exists, and it is for the correct exchange
        if user in keyfile and keyfile[user]["exchange"] == exchange:

            # If we need to get the `market` key:
            # market = keyfile[user]["market"]
            # print("The Market Type is " + str(market))

            keyList = [str(keyfile[user]["key"]), str(keyfile[user]["secret"])]

            return keyList
        elif user not in keyfile or keyfile[user]["exchange"] != exchange:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception('API KeyFile Missing!')


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


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


class LockNotAvailableException(Exception):
    pass

class Bot:
    def __init__(self, user: str, config: dict):
        self.config = config
        self.user = user
        self.telegram = None

        for key in config:
            setattr(self, key, config[key])

        self.ema_span = int(round(self.ema_span))
        self.ema_alpha = 2 / (self.ema_span + 1)
        self.ema_alpha_ = 1 - self.ema_alpha

        self.ts_locked = {'cancel_orders': 0, 'decide': 0, 'update_open_orders': 0,
                          'update_position': 0, 'print': 0, 'create_orders': 0}
        self.ts_released = {k: 1 for k in self.ts_locked}

        self.position = {}
        self.open_orders = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0
        self.ob = [0.0, 0.0]
        self.ema = 0.0

        self.n_open_orders_limit = 8
        self.last_price_diff_limit = 0.15
        self.n_orders_per_execution = 4

        self.hedge_mode = True
        self.contract_multiplier = self.config['contract_multiplier'] = 1.0

        self.log_filepath = make_get_filepath(f"logs/{self.exchange}/{config['config_name']}.log")

        self.key, self.secret = load_key_secret(config['exchange'], user)

        self.log_level = 0

        self.stop_websocket = False
        self.process_websocket_ticks = True
        self.lock_file = f"{str(Path.home())}/.passivbotlock"
        self.stop_mode = self.config['stop_mode'] = None

    def set_config(self, config):
        config['ema_span'] = int(round(config['ema_span']))
        self.config = config
        for key in config:
            setattr(self, key, config[key])
            if key in self.xk:
                self.xk[key] = config[key]

    def set_config_value(self, key, value):
        self.config[key] = value
        setattr(self, key, self.config[key])

    async def _init(self):
        self.xk = {k: float(self.config[k]) for k in get_keys()}

    def dump_log(self, data) -> None:
        if self.config['logging_level'] > 0:
            with open(self.log_filepath, 'a') as f:
                f.write(json.dumps({**{'log_timestamp': time()}, **data}) + '\n')

    async def update_open_orders(self) -> None:
        if self.ts_locked['update_open_orders'] > self.ts_released['update_open_orders']:
            return
        try:
            open_orders = await self.fetch_open_orders()
            self.highest_bid, self.lowest_ask = 0.0, 9.9e9
            for o in open_orders:
                if o['side'] == 'buy':
                    self.highest_bid = max(self.highest_bid, o['price'])
                elif o['side'] == 'sell':
                    self.lowest_ask = min(self.lowest_ask, o['price'])
            if self.open_orders != open_orders:
                self.dump_log({'log_type': 'open_orders', 'data': open_orders})
            self.open_orders = open_orders
            self.ts_released['update_open_orders'] = time()
        except Exception as e:
            print('error with update open orders', e)

    async def update_position(self) -> None:
        # also updates open orders
        if self.ts_locked['update_position'] > self.ts_released['update_position']:
            return
        self.ts_locked['update_position'] = time()
        try:
            position, _ = await asyncio.gather(self.fetch_position(),
                                               self.update_open_orders())
            position['used_margin'] = \
                ((calc_cost(position['long']['size'], position['long']['price'],
                            self.xk['inverse'], self.xk['contract_multiplier'])
                  if position['long']['price'] else 0.0) +
                 (calc_cost(position['shrt']['size'], position['shrt']['price'],
                            self.xk['inverse'], self.xk['contract_multiplier'])
                  if position['shrt']['price'] else 0.0)) / self.leverage
            position['available_margin'] = (position['equity'] - position['used_margin']) * 0.9
            position['long']['liq_diff'] = calc_diff(position['long']['liquidation_price'], self.price)
            position['shrt']['liq_diff'] = calc_diff(position['shrt']['liquidation_price'], self.price)
            if self.position != position:
                self.dump_log({'log_type': 'position', 'data': position})
            self.position = position
            self.ts_released['update_position'] = time()
        except Exception as e:
            print('error with update position', e)

    async def create_orders(self, orders_to_create: [dict]) -> dict:
        if not orders_to_create:
            return
        if self.ts_locked['create_orders'] > self.ts_released['create_orders']:
            return
        self.ts_locked['create_orders'] = time()
        creations = []
        for oc in sorted(orders_to_create, key=lambda x: x['qty']):
            try:
                creations.append((oc, asyncio.create_task(self.execute_order(oc))))
            except Exception as e:
                print_(['error creating order a', oc, e], n=True)
        created_orders = []
        for oc, c in creations:
            try:
                o = await c
                created_orders.append(o)
                if 'side' in o:
                    print_([' created order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error creating order b', o, oc], n=True)
                self.dump_log({'log_type': 'create_order', 'data': o})
            except Exception as e:
                print_(['error creating order c', oc, c.exception(), e], n=True)
                self.dump_log({'log_type': 'create_order', 'data': {'result': str(c.exception()),
                                                                    'error': repr(e), 'data': oc}})
        self.ts_released['create_orders'] = time()
        return created_orders

    async def cancel_orders(self, orders_to_cancel: [dict]) -> [dict]:
        if not orders_to_cancel:
            return
        if self.ts_locked['cancel_orders'] > self.ts_released['cancel_orders']:
            return
        self.ts_locked['cancel_orders'] = time()
        deletions = []
        for oc in orders_to_cancel:
            try:
                deletions.append((oc,
                                  asyncio.create_task(self.execute_cancellation(oc))))
            except Exception as e:
                print_(['error cancelling order a', oc, e])
        canceled_orders = []
        for oc, c in deletions:
            try:
                o = await c
                canceled_orders.append(o)
                if 'side' in o:
                    print_(['cancelled order', o['symbol'], o['side'], o['position_side'], o['qty'],
                            o['price']], n=True)
                else:
                    print_(['error cancelling order', o], n=True)
                self.dump_log({'log_type': 'cancel_order', 'data': o})
            except Exception as e:
                print_(['error cancelling order b', oc, c.exception(), e], n=True)
                self.dump_log({'log_type': 'cancel_order', 'data': {'result': str(c.exception()),
                                                                    'error': repr(e), 'data': oc}})
        self.ts_released['cancel_orders'] = time()
        return canceled_orders

    def stop(self, signum=None, frame=None) -> None:
        print("\nStopping passivbot, please wait...")
        try:
            self.stop_websocket = True
            if self.telegram is not None:
                self.telegram.exit()
            else:
                print("No telegram active")
        except Exception as e:
            print(f"An error occurred during shutdown: {e}")

    def pause(self) -> None:
        self.process_websocket_ticks = False

    def resume(self) -> None:
        self.process_websocket_ticks = True

    def calc_orders(self):
        balance = self.position['wallet_balance'] * 0.9
        long_psize = self.position['long']['size']
        long_pprice = self.position['long']['price']
        shrt_psize = self.position['shrt']['size']
        shrt_pprice = self.position['shrt']['price']

        if self.hedge_mode:
            do_long = self.do_long or long_psize != 0.0
            do_shrt = self.do_shrt or shrt_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and shrt_psize == 0.0
            do_long = (no_pos and self.do_long) or long_psize != 0.0
            do_shrt = (no_pos and self.do_shrt) or shrt_psize != 0.0
                                              
        self.xk['do_long'] = do_long
        self.xk['do_shrt'] = do_shrt

        liq_price = self.position['long']['liquidation_price'] if long_psize > abs(shrt_psize) \
            else self.position['shrt']['liquidation_price']

        long_entry_orders, shrt_entry_orders, long_close_orders, shrt_close_orders = [], [], [], []
        stop_loss_close = False

        for tpl in iter_entries(balance, long_psize, long_pprice, shrt_psize, shrt_pprice,
                                liq_price, self.ob[0], self.ob[1], self.ema, self.price,
                                self.volatility, **self.xk):
            if (len(long_entry_orders) >= self.n_open_orders_limit and
                len(shrt_entry_orders) >= self.n_open_orders_limit) or \
                    calc_diff(tpl[1], self.price) > self.last_price_diff_limit:
                break
            if tpl[4] == 'stop_loss_shrt_close':
                shrt_close_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(tpl[0]),
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': True,
                                          'custom_id': tpl[4]})
                shrt_psize = tpl[2]
                stop_loss_close = True
            elif tpl[4] == 'stop_loss_long_close':
                long_close_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(tpl[0]),
                                          'price': tpl[1], 'type': 'limit', 'reduce_only': True,
                                          'custom_id': tpl[4]})
                long_psize = tpl[2]
                stop_loss_close = True
            else:
                if self.stop_mode not in ['freeze']:
                    if tpl[0] > 0.0:
                        long_entry_orders.append({'side': 'buy', 'position_side': 'long', 'qty': tpl[0],
                                                  'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                                  'custom_id': tpl[4]})
                    else:
                        shrt_entry_orders.append({'side': 'sell', 'position_side': 'shrt', 'qty': abs(tpl[0]),
                                                  'price': tpl[1], 'type': 'limit', 'reduce_only': False,
                                                  'custom_id': tpl[4]})

        for ask_qty, ask_price, _ in iter_long_closes(balance, long_psize, long_pprice, self.ob[1],
                                                      **self.xk):
            if len(long_close_orders) >= self.n_open_orders_limit or \
                    calc_diff(ask_price, self.price) > self.last_price_diff_limit or \
                    stop_loss_close:
                break
            long_close_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(ask_qty),
                                      'price': float(ask_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})

        for bid_qty, bid_price, _ in iter_shrt_closes(balance, shrt_psize, shrt_pprice, self.ob[0],
                                                      **self.xk):
            if len(shrt_close_orders) >= self.n_open_orders_limit or \
                    calc_diff(bid_price, self.price) > self.last_price_diff_limit or \
                    stop_loss_close:
                break
            shrt_close_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(bid_qty),
                                      'price': float(bid_price), 'type': 'limit',
                                      'reduce_only': True, 'custom_id': 'close'})
        return long_entry_orders + shrt_entry_orders + long_close_orders + shrt_close_orders

    async def cancel_and_create(self):
        await asyncio.sleep(0.005)
        await self.update_position()
        await asyncio.sleep(0.005)
        if any([self.ts_locked[k_] > self.ts_released[k_]
                for k_ in [x for x in self.ts_locked if x != 'decide']]):
            return
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_orders(),
                                             keys=['side', 'position_side', 'qty', 'price'])
        to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x['price'], self.price))
        to_create = sorted(to_create, key=lambda x: calc_diff(x['price'], self.price))
        results = []
        if to_cancel:
            results.append(asyncio.create_task(self.cancel_orders(to_cancel[:self.n_orders_per_execution])))
            await asyncio.sleep(0.005)  # sleep 5 ms between sending cancellations and creations
        if to_create:
            results.append(await self.create_orders(to_create[:self.n_orders_per_execution]))
        await asyncio.sleep(0.005)
        await self.update_position()
        if any(results):
            print()
        return results

    async def decide(self):
        if self.stop_mode is not None:
            print(f'Effectuating stop mode {self.stop_mode}')
        if self.price <= self.highest_bid:
            self.ts_locked['decide'] = time()
            print_(['bid maybe taken'], n=True)
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if self.price >= self.lowest_ask:
            self.ts_locked['decide'] = time()
            print_(['ask maybe taken'], n=True)
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_locked['decide'] > 5:
            self.ts_locked['decide'] = time()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_released['print'] >= 0.5:
            await self.update_output_information()

    async def update_output_information(self):
        self.ts_released['print'] = time()
        line = f"{self.symbol} "
        line += f"l {self.position['long']['size']} @ "
        line += f"{round_(self.position['long']['price'], self.price_step)} "
        long_closes = sorted([o for o in self.open_orders if o['side'] == 'sell'
                              and o['position_side'] == 'long'], key=lambda x: x['price'])
        long_entries = sorted([o for o in self.open_orders if o['side'] == 'buy'
                               and o['position_side'] == 'long'], key=lambda x: x['price'])
        line += f"c@ {long_closes[0]['price'] if long_closes else 0.0} "
        line += f"e@ {long_entries[-1]['price'] if long_entries else 0.0} "
        line += f"|| s {self.position['shrt']['size']} @ "
        line += f"{round_(self.position['shrt']['price'], self.price_step)} "
        shrt_closes = sorted([o for o in self.open_orders if o['side'] == 'buy'
                              and (o['position_side'] == 'shrt' or
                                   (o['position_side'] == 'both' and
                                    self.position['shrt']['size'] != 0.0))],
                             key=lambda x: x['price'])
        shrt_entries = sorted([o for o in self.open_orders if o['side'] == 'sell'
                               and (o['position_side'] == 'shrt' or
                                    (o['position_side'] == 'both' and
                                     self.position['shrt']['size'] != 0.0))],
                              key=lambda x: x['price'])
        line += f"c@ {shrt_closes[-1]['price'] if shrt_closes else 0.0} "
        line += f"e@ {shrt_entries[0]['price'] if shrt_entries else 0.0} "
        if self.position['long']['size'] > abs(self.position['shrt']['size']):
            liq_price = self.position['long']['liquidation_price']
            sl_trigger_price = liq_price / (1 - self.stop_loss_liq_diff)
        else:
            liq_price = self.position['shrt']['liquidation_price']
            sl_trigger_price = liq_price / (1 + self.stop_loss_liq_diff)
        line += f"|| last {self.price} liq {compress_float(liq_price, 5)} "
        line += f"sl trig {compress_float(sl_trigger_price, 5)} "
        line += f"ema {compress_float(self.ema, 5)} "
        line += f"bal {compress_float(self.position['wallet_balance'], 3)} "
        line += f"eq {compress_float(self.position['equity'], 3)} "
        line += f"v. {compress_float(self.volatility, 5)} "
        print_([line], r=True)

    def flush_stuck_locks(self, timeout: float = 4.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now

    async def fetch_compressed_ticks(self):

        def drop_consecutive_same_prices(ticks_):
            compressed_ = [ticks_[0]]
            for i in range(1, len(ticks_)):
                if ticks_[i]['price'] != compressed_[-1]['price']:
                    compressed_.append(ticks_[i])
            return compressed_

        ticks_unabridged = await self.fetch_ticks(do_print=False)
        ticks_per_fetch = len(ticks_unabridged)
        ticks = drop_consecutive_same_prices(ticks_unabridged)
        if self.exchange == 'bybit' and self.market_type == 'linear_perpetual':
            print('\nwarning:  bybit linear usdt symbols only allows fetching most recent 1000 ticks')
            return ticks
        delay_between_fetches = 0.55
        print()
        while True:
            print(f'\rfetching ticks... {len(ticks)} of {self.ema_span} ', end= ' ')
            sts = time()
            new_ticks = await self.fetch_ticks(from_id=ticks[0]['trade_id'] - ticks_per_fetch,
                                               do_print=False)
            wait_for = max(0.0, delay_between_fetches - (time() - sts))
            ticks = drop_consecutive_same_prices(sorted(new_ticks + ticks,
                                                        key=lambda x: x['trade_id']))
            if len(ticks) > self.ema_span:
                break
            await asyncio.sleep(wait_for)
        new_ticks = await self.fetch_ticks(do_print=False)
        return drop_consecutive_same_prices(sorted(ticks + new_ticks, key=lambda x: x['trade_id']))

    async def init_indicators(self):
        ticks = await self.fetch_compressed_ticks()
        ema = ticks[0]['price']
        self.tick_prices_deque = deque(maxlen=self.ema_span)
        for tick in ticks:
            self.tick_prices_deque.append(tick['price'])
            ema = ema * self.ema_alpha_ + tick['price'] * self.ema_alpha
        if len(self.tick_prices_deque) < self.ema_span:
            print('\nwarning: insufficient ticks fetched, filling deque with duplicate ticks...')
            print('ema and volatility will be inaccurate until deque is filled with websocket ticks')
            while len(self.tick_prices_deque) < self.ema_span:
                self.tick_prices_deque.extend([t['price'] for t in ticks])
        self.ema = ema
        self.sum_prices = sum(self.tick_prices_deque)
        self.sum_prices_squared = sum([e ** 2 for e in self.tick_prices_deque])
        self.price_std = np.sqrt((self.sum_prices_squared / len(self.tick_prices_deque) -
                                 ((self.sum_prices / len(self.tick_prices_deque)) ** 2)))
        self.volatility = self.price_std / self.ema
        print('\ndebug len ticks, prices deque, ema_span')
        print(len(ticks), len(self.tick_prices_deque), self.ema_span)

    def update_indicators(self, ticks: dict):
        for tick in ticks:
            if tick['is_buyer_maker']:
                self.ob[0] = tick['price']
            else:
                self.ob[1] = tick['price']
            self.ema = calc_ema(self.ema_alpha, self.ema_alpha_, self.ema, tick['price'])
            self.sum_prices -= self.tick_prices_deque[0]
            self.sum_prices_squared -= self.tick_prices_deque[0] ** 2
            self.tick_prices_deque.append(tick['price'])
            self.sum_prices += self.tick_prices_deque[-1]
            self.sum_prices_squared += self.tick_prices_deque[-1] ** 2
        self.price_std = np.sqrt((self.sum_prices_squared / len(self.tick_prices_deque) -
                                 ((self.sum_prices / len(self.tick_prices_deque)) ** 2)))
        self.price = ticks[-1]['price']
        self.volatility = self.price_std / self.ema

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        self.process_websocket_ticks = True
        print_([self.endpoints['websocket']])
        await self.update_position()
        await self.init_exchange_config()
        await self.init_indicators()
        self.remove_lock_file()
        k = 1
        async with websockets.connect(self.endpoints['websocket']) as ws:
            await self.subscribe_ws(ws)
            async for msg in ws:
                if msg is None:
                    continue
                try:
                    ticks = self.standardize_websocket_ticks(json.loads(msg))
                    if self.process_websocket_ticks:
                        if ticks:
                            self.update_indicators(ticks)
                        if self.ts_locked['decide'] < self.ts_released['decide']:
                            asyncio.create_task(self.decide())
                    if k % 10 == 0:
                        self.flush_stuck_locks()
                        k = 1
                    if self.stop_websocket:
                        if self.telegram is not None:
                            self.telegram.send_msg("<pre>Bot stopped</pre>")
                        break
                    k += 1

                except Exception as e:
                    if 'success' not in msg:
                        print('error in websocket', e, msg)

    def remove_lock_file(self):
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                print('The lock file has been removed succesfully')
            else:
                print("The lock file doesn't exists, no need to do anything, but it should've been there!")
        except:
            print(f"Failed to remove the lock file! Please remove the file {self.lock_file} manually to ensure fast restarts")

    async def acquire_interprocess_lock(self):
        if os.path.exists(self.lock_file):
            #if the file was last modified over 15 minutes ago, assume that something went wrong
            if time() - os.path.getmtime(self.lock_file) > 15 * 60:
                print(f'File {self.lock_file} last modified more than 15 minutes ago. Assuming something went wrong'
                      f' trying to delete it, attempting removal of file')
                self.remove_lock_file()
            else:
                raise LockNotAvailableException("Another bot has the lock")

        pid = str(os.getpid())
        with open(self.lock_file, 'w') as f:
            f.write(pid)
            f.flush()
            f.close()

        await asyncio.sleep(5)

        pid_in_file = open(self.lock_file).read()
        if pid_in_file != pid:
            raise LockNotAvailableException("Lock is stolen by another bot")
        return

async def start_bot(bot):
    while not bot.stop_websocket:
        try:
            await bot.acquire_interprocess_lock()
            await bot.start_websocket()
        except Exception as e:
            print('Websocket connection has been lost or unable to acquire lock to start, attempting to reinitialize the bot...', e)
            await asyncio.sleep(10)

async def create_binance_bot(user: str, config: str):
    from binance import BinanceBot
    bot = BinanceBot(user, config)
    await bot._init()
    return bot


async def create_bybit_bot(user: str, config: str):
    from bybit import Bybit
    bot = Bybit(user, config)
    await bot._init()
    return bot


async def _start_telegram(account: dict, bot: Bot):
    telegram = telegram_bot.Telegram(token=account['telegram']['token'],
                                     chat_id=account['telegram']['chat_id'],
                                     bot=bot,
                                     loop=asyncio.get_event_loop())
    telegram.log_start()
    return telegram

async def main() -> None:
    try:
        accounts = json.load(open('api-keys.json'))
    except Exception as e:
        print(e, 'failed to load api-keys.json file')
        return
    if sys.argv[1] in accounts:
        account = accounts[sys.argv[1]]
    else:
        print('unrecognized account name', sys.argv[1])
        return
    try:
        config = json.load(open(sys.argv[3]))
    except Exception as e:
        print(e, 'failed to load config', sys.argv[3])
        return
    config['exchange'] = account['exchange']
    config['symbol'] = sys.argv[2]

    if account['exchange'] == 'binance':
        bot = await create_binance_bot(sys.argv[1], config)
    elif account['exchange'] == 'bybit':
        bot = await create_bybit_bot(sys.argv[1], config)
    else:
        raise Exception('unknown exchange', account['exchange'])
    print('using config')
    print(json.dumps(config, indent=4))

    if 'telegram' in account and account['telegram']['enabled']:
        telegram = await _start_telegram(account=account, bot=bot)
        bot.telegram = telegram
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    await start_bot(bot)
    await bot.session.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        print('\nPassivbot was stopped succesfully')
        os._exit(0)
