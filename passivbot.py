import os
if 'NOJIT' not in os.environ:
    os.environ['NOJIT'] = 'true'

import traceback
import argparse
import asyncio
import json
import logging
import signal
import pprint
from pathlib import Path
from time import time
from procedures import load_live_config, make_get_filepath, load_exchange_key_secret, print_, add_argparse_args, \
    utc_ms
from pure_funcs import get_xk_keys, get_ids_to_fetch, flatten, calc_indicators_from_ticks_with_gaps, \
    drop_consecutive_same_prices, filter_orders, compress_float, create_xk, round_dynamic, denumpyize, \
    calc_spans, spotify_config, get_position_fills, determine_config_type
from njit_funcs import calc_new_psize_pprice, qty_to_cost, calc_diff, round_, calc_orders, calc_emas, \
    calc_samples, calc_emas_last, calc_long_scalp_entry, calc_shrt_scalp_entry, calc_long_close_grid, \
    calc_shrt_close_grid

import numpy as np
import websockets
import telegram_bot

logging.getLogger("telegram").setLevel(logging.CRITICAL)


class LockNotAvailableException(Exception):
    pass

class Bot:
    def __init__(self, config: dict):
        self.spot = False
        self.config = config
        self.config['do_long'] = config['long']['enabled']
        self.config['do_shrt'] = config['shrt']['enabled']
        self.config['max_leverage'] = 25
        self.telegram = None
        self.xk = {}

        self.hedge_mode = self.config['hedge_mode'] = True
        self.set_config(self.config)

        self.ts_locked = {'cancel_orders': 0.0, 'decide': 0.0, 'update_open_orders': 0.0,
                          'update_position': 0.0, 'print': 0.0, 'create_orders': 0.0,
                          'check_fills': 0.0, 'update_fills': 0.0}
        self.ts_released = {k: 1.0 for k in self.ts_locked}
        self.heartbeat_ts = 0

        self.position = {}
        self.open_orders = []
        self.fills = []
        self.long_pfills = []
        self.shrt_pfills = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0.0
        self.is_buyer_maker = True
        self.agg_qty = 0.0
        self.qty = 0.0
        self.ob = [0.0, 0.0]

        self.n_open_orders_limit = 8
        self.n_orders_per_execution = 4

        self.c_mult = self.config['c_mult'] = 1.0

        self.log_filepath = make_get_filepath(f"logs/{self.exchange}/{config['config_name']}.log")

        _, self.key, self.secret = load_exchange_key_secret(self.user)

        self.log_level = 0

        self.stop_websocket = False
        self.process_websocket_ticks = True
        self.lock_file = f"{str(Path.home())}/.{self.exchange}_passivbotlock"

    def set_config(self, config):
        if 'min_span' in config:
            config['spans'] = calc_spans(config['min_span'], config['max_span'], config['n_spans'])
        if 'stop_mode' not in config:
            config['stop_mode'] = None
        if 'last_price_diff_limit' not in config:
            config['last_price_diff_limit'] = 0.3
        if 'profit_trans_pct' not in config:
            config['profit_trans_pct'] = 0.0
        if 'assigned_balance' not in config:
            config['assigned_balance'] = None
        if 'cross_wallet_pct' not in config:
            config['cross_wallet_pct'] = 1.0
        if config['cross_wallet_pct'] > 1.0 or config['cross_wallet_pct'] <= 0.0:
            print(f'An invalid value is provided for `cross_wallet_pct` ({config["cross_wallet_pct"]}). The value must be bigger than 0.0 and less than or equal to 1.0. The'
                  f'bot will start with the default value of 1.0, meaning it will utilize the full wallet balance available.')
            config['cross_wallet_pct'] = 1.0
        if config['assigned_balance'] is not None and config['cross_wallet_pct'] < 1.0:
            print(f'A value of {config["cross_wallet_pct"]} for the cross_wallet_pct parameter was provided while a value of {config["assigned_balance"]} for assigned_balance '
                  f'was also provided. The cross_wallet_pct parameter is ignored, and the assigned_balance parameter takes effect. Please make sure that either cross_wallet_pct '
                  f'is removed, or the assigned_balance parameter.')
        self.config = config
        for key in config:
            setattr(self, key, config[key])
            if key in self.xk:
                self.xk[key] = config[key]
        self.config_type = self.config['config_type'] = determine_config_type(config)
        if self.config_type == 'vanilla':
            self.ema_alpha = 2.0 / (self.spans + 1.0) if hasattr(self, 'spans') else 0.0
            self.ema_alpha_ = 1.0 - self.ema_alpha
            self.spans_secs = self.spans * 60 if hasattr(self, 'spans') else 0.0  # spans are in minutes
            self.ema_alpha_secs = 2.0 / (self.spans_secs + 1.0)
            self.ema_alpha_secs_ = 1.0 - self.ema_alpha_secs
            self.ema_sec = 0
            self.emas = np.zeros(len(self.spans))
            self.ratios = np.zeros(len(self.spans))

    def set_config_value(self, key, value):
        self.config[key] = value
        setattr(self, key, self.config[key])

    async def _init(self):
        self.xk = create_xk(self.config)
        await self.init_fills()

    def dump_log(self, data) -> None:
        if self.config['logging_level'] > 0:
            with open(self.log_filepath, 'a') as f:
                f.write(json.dumps({**{'log_timestamp': time()}, **data}) + '\n')

    async def update_open_orders(self) -> None:
        if self.ts_locked['update_open_orders'] > self.ts_released['update_open_orders']:
            return
        try:
            open_orders = await self.fetch_open_orders()
            open_orders = [x for x in open_orders if x['symbol'] == self.symbol]
            self.highest_bid, self.lowest_ask = 0.0, 9.9e9
            for o in open_orders:
                if o['side'] == 'buy':
                    self.highest_bid = max(self.highest_bid, o['price'])
                elif o['side'] == 'sell':
                    self.lowest_ask = min(self.lowest_ask, o['price'])
            if self.open_orders != open_orders:
                self.dump_log({'log_type': 'open_orders', 'data': open_orders})
            self.open_orders = open_orders
        except Exception as e:
            print('error with update open orders', e)
        finally:
            self.ts_released['update_open_orders'] = time()

    async def update_position(self) -> None:
        # also updates open orders
        if self.ts_locked['update_position'] > self.ts_released['update_position']:
            return
        self.ts_locked['update_position'] = time()
        try:
            position, _ = await asyncio.gather(self.fetch_position(),
                                               self.update_open_orders())
            position['used_margin'] = \
                ((qty_to_cost(position['long']['size'], position['long']['price'],
                              self.xk['inverse'], self.xk['c_mult'])
                  if position['long']['price'] else 0.0) +
                 (qty_to_cost(position['shrt']['size'], position['shrt']['price'],
                              self.xk['inverse'], self.xk['c_mult'])
                  if position['shrt']['price'] else 0.0)) / self.max_leverage

            if self.assigned_balance is not None:
                used_balance = self.assigned_balance
                if position['wallet_balance'] < self.assigned_balance:
                    print(f'The assigned balance {self.assigned_balance} is greater than the total wallet balance of {position["wallet_balance"]} available. '
                          f'The total wallet available will be used.')
                    used_balance = position['wallet_balance']
                position['equity'] -= used_balance
                position['wallet_balance'] = used_balance
            else:
                position['equity'] -= position['wallet_balance'] * (1 - self.cross_wallet_pct)
                position['wallet_balance'] *= self.cross_wallet_pct

            position['available_margin'] = (position['equity'] - position['used_margin']) * 0.9
            position['long']['liq_diff'] = calc_diff(position['long']['liquidation_price'], self.price)
            position['shrt']['liq_diff'] = calc_diff(position['shrt']['liquidation_price'], self.price)
            position['long']['pbr'] = (qty_to_cost(position['long']['size'], position['long']['price'],
                                                   self.xk['inverse'], self.xk['c_mult']) /
                                       position['wallet_balance']) if position['wallet_balance'] else 0.0
            position['shrt']['pbr'] = (qty_to_cost(position['shrt']['size'], position['shrt']['price'],
                                                   self.xk['inverse'], self.xk['c_mult']) /
                                       position['wallet_balance']) if position['wallet_balance'] else 0.0
            if self.position != position:
                if self.position and not 'spot' in self.market_type and \
                        (self.position['long']['size'] != position['long']['size'] or
                         self.position['shrt']['size'] != position['shrt']['size']):
                    # update fills if position size changed
                    await self.update_fills()
                self.dump_log({'log_type': 'position', 'data': position})
            self.position = position
            self.long_pfills, self.shrt_pfills = get_position_fills(self.position['long']['size'],
                                                                    abs(self.position['shrt']['size']),
                                                                    self.fills)
        except Exception as e:
            print('error with update position', e)
        finally:
            self.ts_released['update_position'] = time()

    async def init_fills(self, n_days_limit=60):
        self.fills = await self.fetch_fills()
        #self.fills = await self.fetch_all_fills(n_days_limit)

    async def fetch_all_fills(self, n_days_limit=60):
        try:
            from pure_funcs import ts_to_date
            now = utc_ms()
            day = 1000 * 60 * 60 * 23.99
            week = day * 6.99
            fetch_timespan_limit = day if self.spot else week
            fetch_time = now - day * n_days_limit
            recent_fills = await self.fetch_fills()
            if not recent_fills:
                return []
            if recent_fills[0]['timestamp'] <= fetch_time:
                print('debug returing recent fills')
                return recent_fills
            oldest_fills = await self.fetch_fills(start_time=fetch_time, end_time=fetch_time + fetch_timespan_limit)
            while oldest_fills == [] and fetch_time < now:
                print('debug no fills, fetching ahead', ts_to_date(fetch_time / 1000))
                fetch_time += fetch_timespan_limit
                oldest_fills = await self.fetch_fills(start_time=fetch_time, end_time=fetch_time + fetch_timespan_limit)
            if oldest_fills:
                additional_fills = await self.fetch_fills(from_id=oldest_fills[-1]['id'])
                while additional_fills[-1]['timestamp'] < recent_fills[0]['timestamp']:
                    print('debug fetching additional_fills')
                    new_additional_fills = await self.fetch_fills(from_id=additional_fills[-1]['id'])
                    if new_additional_fills == [] or new_additional_fills[-1] == additional_fills[-1]:
                        break
                    additional_fills += new_additional_fills
            fills = {x['id']: x for x in oldest_fills + recent_fills + additional_fills}
            return sorted(fills.values(), key=lambda x: x['id'])
        except Exception as e:
            print('error with init fills', e)
            return []

    async def update_fills(self, max_n_fills=10000) -> [dict]:
        '''
        fetches recent fills
        updates self.fills, drops older fills max_n_fills
        returns list of new fills
        '''
        if self.ts_locked['update_fills'] > self.ts_released['update_fills']:
            return
        self.ts_locked['update_fills'] = time()
        try:
            ids_set = set([x['order_id'] for x in self.fills])
            fetched = await self.fetch_fills()
            new_fills = [x for x in fetched if x['order_id'] not in ids_set]
            if new_fills:
                self.fills = sorted([x for x in self.fills + new_fills], key=lambda x: x['order_id'])[-1000:]
                self.long_pfills, self.shrt_pfills = get_position_fills(self.position['long']['size'],
                                                                        abs(self.position['shrt']['size']),
                                                                        self.fills)
            return new_fills
        except Exception as e:
            print('error with update fills', e)
        finally:
            self.ts_released['update_fills'] = time()

    async def create_orders(self, orders_to_create: [dict]) -> dict:
        if not orders_to_create:
            return {}
        if self.ts_locked['create_orders'] > self.ts_released['create_orders']:
            return {}
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
                    print_(['  created order', o['symbol'], o['side'], o['position_side'], o['qty'],
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
        balance = self.position['wallet_balance']
        long_psize = self.position['long']['size']
        long_pprice = self.position['long']['price']
        shrt_psize = self.position['shrt']['size']
        shrt_pprice = self.position['shrt']['price']

        if self.stop_mode in ['panic']:
            if self.exchange == 'bybit':
                print('\n\npanic mode temporarily disabled for bybit\n\n')
                return []
            panic_orders = []
            if long_psize != 0.0:
                panic_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(long_psize), 'price': self.ob[1],
                                     'type': 'market', 'reduce_only': True, 'custom_id': 'long_panic'})
            if shrt_psize != 0.0:
                panic_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(shrt_psize), 'price': self.ob[0],
                                     'type': 'market', 'reduce_only': True, 'custom_id': 'shrt_panic'})
            return panic_orders

        if self.hedge_mode:
            do_long = self.do_long or long_psize != 0.0
            do_shrt = self.do_shrt or shrt_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and shrt_psize == 0.0
            do_long = (no_pos and self.do_long) or long_psize != 0.0
            do_shrt = (no_pos and self.do_shrt) or shrt_psize != 0.0
                                              
        self.xk['do_long'] = do_long
        self.xk['do_shrt'] = do_shrt
        if self.config_type == 'vanilla':
            return self.calc_orders_vanilla(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, do_long, do_shrt)
        elif self.config_type == 'scalp':
            return self.calc_orders_scalp(balance, long_psize, long_pprice, shrt_psize, shrt_pprice, do_long, do_shrt)
        else:
            raise Exception('unknown config type')

    def calc_orders_vanilla(self, balance, long_psize, long_pprice, shrt_psize, shrt_pprice, do_long, do_shrt):
        orders = []
        long_done, shrt_done = False, False

        inf_loop_prevention = 100
        i = 0

        while True:
            i += 1
            if i >= inf_loop_prevention:
                raise Exception('warning -- infinite loop in calc_orders')
            long_entry, shrt_entry, long_close, shrt_close, bkr_price, available_margin = calc_orders(
                balance,
                long_psize,
                long_pprice,
                shrt_psize,
                shrt_pprice,
                self.ob[0],
                self.ob[1],
                self.price,
                self.emas,
                **self.xk)
            if i == 1 and long_close[0] != 0.0 and \
                    calc_diff(long_close[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(float(long_close[0])),
                               'price': float(long_close[1]), 'type': 'limit', 'reduce_only': True,
                               'custom_id': long_close[2]})
            if i == 1 and shrt_close[0] != 0.0 and \
                    calc_diff(shrt_close[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(float(shrt_close[0])),
                               'price': float(shrt_close[1]), 'type': 'limit', 'reduce_only': True,
                               'custom_id': shrt_close[2]})
            if not long_done and self.stop_mode not in ['freeze'] and long_entry[0] != 0.0 and \
                    calc_diff(long_entry[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'buy', 'position_side': 'long', 'qty': float(long_entry[0]),
                               'price': float(long_entry[1]), 'type': 'limit', 'reduce_only': False,
                               'custom_id': long_entry[2]})
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                long_entry[0], long_entry[1], self.qty_step)
            else:
                long_done = True
            if not shrt_done and self.stop_mode not in ['freeze'] and shrt_entry[0] != 0.0 and \
                    calc_diff(shrt_entry[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'sell', 'position_side': 'shrt', 'qty': abs(float(shrt_entry[0])),
                               'price': float(shrt_entry[1]), 'type': 'limit', 'reduce_only': False,
                               'custom_id': shrt_entry[2]})
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice,
                                                                shrt_entry[0], shrt_entry[1], self.qty_step)
            else:
                shrt_done = True
            if len(orders) >= self.n_open_orders_limit or (long_done and shrt_done):
                break
        return orders

    def calc_orders_scalp(self, balance, long_psize, long_pprice, shrt_psize, shrt_pprice, do_long, do_shrt):
        
        orders = []
        if do_long:
            long_closes = calc_long_close_grid(
                balance, long_psize, long_pprice, self.ob[1], self.spot, self.inverse, self.qty_step,
                self.price_step, self.min_qty, self.min_cost, self.c_mult, self.max_leverage,
                self.xk['primary_initial_qty_pct'][0], self.xk['min_markup'][0], self.xk['markup_range'][0],
                self.xk['n_close_orders'][0]
            )
            long_closes = [{'side': 'sell', 'position_side': 'long', 'qty': abs(x[0]),
                            'price': x[1], 'type': 'limit', 'custom_id': x[2]}
                           for x in long_closes if x[0] != 0.0]
            i = 0
            long_entries = []
            while i < self.n_open_orders_limit:
                i += 1
                long_entry = calc_long_scalp_entry(
                    balance, long_psize, long_pprice, ((0.0, 0.0),), self.ob[0], self.spot, self.inverse, do_long,
                    self.qty_step, self.price_step, self.min_qty, self.min_cost, self.c_mult, self.max_leverage,
                    self.xk['primary_initial_qty_pct'][0], self.xk['primary_ddown_factor'][0],
                    self.xk['primary_grid_spacing'][0], self.xk['primary_grid_spacing_pbr_weighting'][0],
                    self.xk['primary_pbr_limit'][0], self.xk['secondary_ddown_factor'][0],
                    self.xk['secondary_grid_spacing'][0], self.xk['secondary_pbr_limit_added'][0]
                )
                if long_entry[0] == 0.0:
                    break
                long_entries.append(long_entry)
                if long_psize == 0.0:
                    break
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                long_entry[0], long_entry[1], self.qty_step)
            long_entries = [{'side': 'buy', 'position_side': 'long', 'qty': abs(x[0]),
                             'price': x[1], 'type': 'limit', 'custom_id': x[2]}
                            for x in long_entries]
            orders.extend(long_entries + long_closes)
        if do_shrt:
            shrt_entries = []
            shrt_closes = calc_shrt_close_grid(
                balance, shrt_psize, shrt_pprice, self.ob[0], self.spot, self.inverse, self.qty_step,
                self.price_step, self.min_qty, self.min_cost, self.c_mult, self.max_leverage,
                self.xk['primary_initial_qty_pct'][1], self.xk['min_markup'][1], self.xk['markup_range'][1],
                self.xk['n_close_orders'][1]
            )
            shrt_closes = [{'side': 'buy', 'position_side': 'shrt', 'qty': abs(x[0]),
                            'price': x[1], 'type': 'limit', 'custom_id': x[2]}
                           for x in shrt_closes if x[0] != 0.0]

            i = 0
            while i < self.n_open_orders_limit:
                i += 1
                shrt_entry = calc_shrt_scalp_entry(
                    balance, shrt_psize, shrt_pprice, ((0.0, 0.0),), self.ob[1], self.spot, self.inverse, do_shrt,
                    self.qty_step, self.price_step, self.min_qty, self.min_cost, self.c_mult, self.max_leverage,
                    self.xk['primary_initial_qty_pct'][1], self.xk['primary_ddown_factor'][1],
                    self.xk['primary_grid_spacing'][1], self.xk['primary_grid_spacing_pbr_weighting'][1],
                    self.xk['primary_pbr_limit'][1], self.xk['secondary_ddown_factor'][1],
                    self.xk['secondary_grid_spacing'][1], self.xk['secondary_pbr_limit_added'][1]
                )
                if shrt_entry[0] == 0.0:
                    break
                shrt_entries.append(shrt_entry)
                if shrt_psize == 0.0:
                    break
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice,
                                                                shrt_entry[0], shrt_entry[1], self.qty_step)
            shrt_entries = [{'side': 'sell', 'position_side': 'shrt', 'qty': abs(x[0]),
                             'price': x[1], 'type': 'limit', 'custom_id': x[2]}
                            for x in shrt_entries]
            orders.extend(shrt_entries + shrt_closes)
        return [o for o in orders if o['qty'] != 0.0]


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
        if self.stop_mode not in ['manual']:
            if to_cancel:
                results.append(asyncio.create_task(self.cancel_orders(to_cancel[:self.n_orders_per_execution + 1])))
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
            print(f'{self.stop_mode} stop mode is active')

        if self.price <= self.highest_bid:
            self.ts_locked['decide'] = time()
            print_(['bid maybe taken'], n=True)
            await self.cancel_and_create()
            asyncio.create_task(self.check_fills())
            self.ts_released['decide'] = time()
            return
        if self.price >= self.lowest_ask:
            self.ts_locked['decide'] = time()
            print_(['ask maybe taken'], n=True)
            await self.cancel_and_create()
            asyncio.create_task(self.check_fills())
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_locked['decide'] > 5:
            self.ts_locked['decide'] = time()
            await self.cancel_and_create()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_released['print'] >= 0.5:
            await self.update_output_information()
        if time() - self.ts_released['check_fills'] > 120:
            asyncio.create_task(self.check_fills())
        if time() - self.heartbeat_ts > 60 * 60:
            print_(['heartbeat\n'], n=True)
            self.heartbeat_ts = time()

    async def check_fills(self):
        if self.ts_locked['check_fills'] > self.ts_released['check_fills']:
            # return if another call is in progress
            return
        if self.exchange == 'bybit':
            # bybit not supported
            return
        try:
            now = time()
            if now - self.ts_released['check_fills'] < 5.0:
                # minimum 5 sec between consecutive check fills
                return
            self.ts_locked['check_fills'] = now
            # check fills if two mins since prev check has passed
            new_fills = await self.update_fills()
            if new_fills:
                await self.check_long_fills(new_fills)
                await self.check_shrt_fills(new_fills)
        finally:
            self.ts_released['check_fills'] = time()

    async def check_shrt_fills(self, new_fills):
        # closing orders
        new_shrt_closes = [item for item in new_fills if item['side'] == 'buy' and item['position_side'] == 'shrt']
        if len(new_shrt_closes) > 0:
            realized_pnl_shrt = sum(fill['realized_pnl'] for fill in new_shrt_closes)
            if self.telegram is not None:
                qty_sum = sum([fill['qty'] for fill in new_shrt_closes])
                cost = sum(fill['qty'] / fill['price'] if self.inverse else fill['qty'] * fill['price']
                           for fill in new_shrt_closes)
                # volume weighted average price
                vwap = qty_sum / cost if self.inverse else cost / qty_sum
                fee = sum([fill['fee_paid'] for fill in new_shrt_closes])
                total_size = self.position['shrt']['size']
                self.telegram.notify_close_order_filled(realized_pnl=realized_pnl_shrt, position_side='short',
                                                        qty=qty_sum, fee=fee,
                                                        wallet_balance=self.position['wallet_balance'],
                                                        remaining_size=total_size, price=vwap)
            if realized_pnl_shrt >= 0 and self.profit_trans_pct > 0.0:
                amount = realized_pnl_shrt * self.profit_trans_pct
                self.telegram.send_msg(f'Transferring {round_(amount, 0.001)} USDT ({self.profit_trans_pct * 100 }%) of profit {round_(realized_pnl_shrt, self.price_step)} to Spot wallet')
                transfer_result = await self.transfer(type_='UMFUTURE_MAIN', amount=amount)
                if 'code' in transfer_result:
                    self.telegram.send_msg(f'Error transferring to Spot wallet: {transfer_result["msg"]}')
                else:
                    self.telegram.send_msg(f'Transferred {round_(amount, 0.001)} USDT to Spot wallet')

        # entry orders
        new_shrt_entries = [item for item in new_fills if item['side'] == 'sell' and item['position_side'] == 'shrt']
        if len(new_shrt_entries) > 0:
            if self.telegram is not None:
                qty_sum = sum(fill['qty'] for fill in new_shrt_entries)
                cost = sum(fill['qty'] / fill['price'] if self.inverse else fill['qty'] * fill['price']
                           for fill in new_shrt_entries)
                # volume weighted average price
                vwap = qty_sum / cost if self.inverse else cost / qty_sum
                fee = sum([fill['fee_paid'] for fill in new_shrt_entries])
                total_size = self.position['shrt']['size']
                self.telegram.notify_entry_order_filled(position_side='short', qty=qty_sum, fee=fee, price=vwap, total_size=total_size)

    async def check_long_fills(self, new_fills):
        #closing orders
        new_long_closes = [item for item in new_fills if item['side'] == 'sell' and item['position_side'] == 'long']
        if len(new_long_closes) > 0:
            realized_pnl_long = sum(fill['realized_pnl'] for fill in new_long_closes)
            if self.telegram is not None:
                qty_sum = sum([fill['qty'] for fill in new_long_closes])
                cost = sum(fill['qty'] / fill['price'] if self.inverse else fill['qty'] * fill['price']
                           for fill in new_long_closes)
                # volume weighted average price
                vwap = qty_sum / cost if self.inverse else cost / qty_sum
                fee = sum([fill['fee_paid'] for fill in new_long_closes])
                total_size = self.position['long']['size']
                self.telegram.notify_close_order_filled(realized_pnl=realized_pnl_long, position_side='long',
                                                        qty=qty_sum, fee=fee,
                                                        wallet_balance=self.position['wallet_balance'],
                                                        remaining_size=total_size, price=vwap)
            if realized_pnl_long >= 0 and self.profit_trans_pct > 0.0:
                amount = realized_pnl_long * self.profit_trans_pct
                self.telegram.send_msg(f'Transferring {round_(amount, 0.001)} USDT ({self.profit_trans_pct * 100 }%) of profit {round_(realized_pnl_long, self.price_step)} to Spot wallet')
                transfer_result = await self.transfer(type_='UMFUTURE_MAIN', amount=amount)
                if 'code' in transfer_result:
                    self.telegram.send_msg(f'Error transferring to Spot wallet: {transfer_result["msg"]}')
                else:
                    self.telegram.send_msg(f'Transferred {round_(amount, 0.001)} USDT to Spot wallet')

        # entry orders
        new_long_entries = [item for item in new_fills if item['side'] == 'buy' and item['position_side'] == 'long']
        if len(new_long_entries) > 0:
            if self.telegram is not None:
                qty_sum = sum(fill['qty'] for fill in new_long_entries)
                cost = sum(fill['qty'] / fill['price'] if self.inverse else fill['qty'] * fill['price']
                           for fill in new_long_entries)
                # volume weighted average price
                vwap = qty_sum / cost if self.inverse else cost / qty_sum
                fee = sum([fill['fee_paid'] for fill in new_long_entries])
                total_size = self.position['long']['size']
                self.telegram.notify_entry_order_filled(position_side='long', qty=qty_sum, fee=fee, price=vwap, total_size=total_size)

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
        else:
            liq_price = self.position['shrt']['liquidation_price']
        line += f"|| last {self.price} liq {round_dynamic(liq_price, 5)} "

        line += f"lpbr {self.position['long']['pbr']:.3f} spbr {self.position['shrt']['pbr']:.3f} "
        if self.config_type == 'vanilla':
            line += f"EMAr {[round_dynamic(r, 4) for r in self.ratios]} "
        line += f"bal {compress_float(self.position['wallet_balance'], 3)} "
        line += f"eq {compress_float(self.position['equity'], 3)} "
        print_([line], r=True)

    def flush_stuck_locks(self, timeout: float = 4.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now

    async def init_indicators(self, max_n_samples: int = 60):
        if self.config_type == 'scalp':
            return
        ticks = await self.fetch_ticks(do_print=False)
        if self.exchange == 'binance':
            ohlcvs_per_fetch = 1000 if self.spot else 1500
            additional_ticks = await asyncio.gather(*[self.fetch_ticks(from_id=ticks[0]['trade_id'] - 1000 * i, do_print=False)
                                                      for i in range(1, 11)])
        else:
            ohlcvs_per_fetch = 200
            if 'linear' in self.market_type:
                additional_ticks = []
            else:
                additional_ticks = await asyncio.gather(*[self.fetch_ticks(from_id=ticks[0]['trade_id'] - 1000 * i, do_print=False)
                                                          for i in range(1, 11)])
        ticksd = {e['trade_id']: e for e in ticks + flatten(additional_ticks)}
        ticks = sorted(ticksd.values(), key=lambda x: x['trade_id'])
        millis_per_fetch = 1000 * 60 * ohlcvs_per_fetch
        first_fetch_ts = ticks[0]['timestamp'] // 1000 * 1000 - millis_per_fetch
        last_fetch_ts = first_fetch_ts - max(self.spans) * 60 * 1000
        if last_fetch_ts + millis_per_fetch * (max_n_samples - 10) > first_fetch_ts:
            timestamps_to_fetch = np.arange(first_fetch_ts, last_fetch_ts - millis_per_fetch, -millis_per_fetch)
        else:
            timestamps_to_fetch = np.linspace(first_fetch_ts, last_fetch_ts - millis_per_fetch, max_n_samples - 10)
        ohlcvs = flatten(await asyncio.gather(*[self.fetch_ohlcvs(start_time=ts) for ts in timestamps_to_fetch]))
        combined = np.array(sorted([[e['timestamp'], e['qty'], e['price']] for e in ticks] +
                                   [[e['timestamp'], e['volume'], e['open']] for e in ohlcvs]))
        samples = calc_samples(combined)
        self.emas = calc_emas_last(samples[:, 2], self.spans_secs)
        self.ratios = np.append(self.price, self.emas[:-1]) / self.emas
        self.ema_sec = int(combined[-1][0] // 1000 * 1000)

    def update_indicators(self, ticks):
        if self.config_type == 'scalp':
            if ticks[-1]['is_buyer_maker']:
                self.ob[0] = ticks[-1]['price']
            else:
                self.ob[1] = ticks[-1]['price']
            self.price = ticks[-1]['price']
            return
        for tick in ticks:
            self.agg_qty += tick['qty']
            if tick['is_buyer_maker']:
                self.ob[0] = tick['price']
            else:
                self.ob[1] = tick['price']
            ts_sec = int(tick['timestamp'] // 1000 * 1000)
            if ts_sec <= self.ema_sec:
                self.ema_sec = ts_sec
                self.price = tick['price']
                continue
            self.qty = self.agg_qty
            self.agg_qty = 0.0
            while self.ema_sec < ts_sec - 1000:
                self.emas = self.emas * self.ema_alpha_secs_ + tick['price'] * self.ema_alpha_secs
                self.ema_sec += 1000
            self.emas = self.emas * self.ema_alpha_secs_ + self.price * self.ema_alpha_secs
            self.ema_sec += 1000
            self.price = tick['price']
            self.ratios = np.append(self.price, self.emas[:-1]) / self.emas

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        self.process_websocket_ticks = True
        print_([self.endpoints['websocket']])
        await self.update_position()
        abort = await self.init_exchange_config()
        if abort:
            return
        await self.init_indicators()
        await self.init_order_book()
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

async def start_bot(bot):
    while not bot.stop_websocket:
        try:
            await bot.start_websocket()
        except Exception as e:
            print('Websocket connection has been lost, attempting to reinitialize the bot...', e)
            traceback.print_exc()
            await asyncio.sleep(10)


async def _start_telegram(account: dict, bot: Bot):
    telegram = telegram_bot.Telegram(config=account['telegram'],
                                     bot=bot,
                                     loop=asyncio.get_event_loop())
    telegram.log_start()
    return telegram


async def main() -> None:
    parser = argparse.ArgumentParser(prog='passivbot', description='run passivbot')
    parser.add_argument('user', type=str, help='user/account_name defined in api-keys.json')
    parser.add_argument('symbol', type=str, help='symbol to trade')
    parser.add_argument('live_config_path', type=str, help='live config to use')
    parser.add_argument('-m', '--market_type', type=str, required=False, dest='market_type', default=None,
                        help='specify whether spot or futures (default), overriding value from backtest config')
    parser.add_argument('-gs', '--graceful_stop', action='store_true',
                        help='if true, disable long and short')

    args = parser.parse_args()
    try:
        accounts = json.load(open('api-keys.json'))
    except Exception as e:
        print(e, 'failed to load api-keys.json file')
        return
    try:
        account = accounts[args.user]
    except Exception as e:
        print('unrecognized account name', args.user, e)
        return
    try:
        config = load_live_config(args.live_config_path)
    except Exception as e:
        print(e, 'failed to load config', args.live_config_path)
        return
    config['user'] = args.user
    config['exchange'] = account['exchange']
    config['symbol'] = args.symbol
    config['live_config_path'] = args.live_config_path
    config['market_type'] = args.market_type if args.market_type is not None else 'futures'

    if args.graceful_stop:
        print('\n\ngraceful stop enabled, will not make new entries once existing positions are closed\n')
        config['long']['enabled'] = config['do_long'] = False
        config['shrt']['enabled'] = config['do_shrt'] = False

    if 'spot' in config['market_type']:
        config = spotify_config(config)

    if account['exchange'] == 'binance':
        if 'spot' in config['market_type']:
            from procedures import create_binance_bot_spot
            bot = await create_binance_bot_spot(config)
        else:
            from procedures import create_binance_bot
            bot = await create_binance_bot(config)
    elif account['exchange'] == 'bybit':
        from procedures import create_bybit_bot
        bot = await create_bybit_bot(config)
    else:
        raise Exception('unknown exchange', account['exchange'])

    print('using config')
    pprint.pprint(denumpyize(config))

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
    except Exception as e:
        print(f'\nThere was an error starting the bot: {e}')
    finally:
        print('\nPassivbot was stopped succesfully')
        os._exit(0)
