import argparse
import asyncio
import json
import logging
import os
import signal
import pprint
from pathlib import Path
from time import time
from procedures import load_live_config, make_get_filepath, load_key_secret, print_
from pure_funcs import get_xk_keys, get_ids_to_fetch, flatten, calc_indicators_from_ticks_with_gaps, \
    drop_consecutive_same_prices, filter_orders, compress_float, create_xk, round_dynamic, denumpyize, \
    calc_spans
from njit_funcs import calc_orders, calc_new_psize_pprice, qty_to_cost, calc_diff, round_, calc_emas
import numpy as np
import websockets
import telegram_bot

logging.getLogger("telegram").setLevel(logging.CRITICAL)


class LockNotAvailableException(Exception):
    pass

class Bot:
    def __init__(self, config: dict):
        self.config = config
        self.config['do_long'] = config['long']['enabled']
        self.config['do_shrt'] = config['shrt']['enabled']
        self.config['max_leverage'] = 25
        self.telegram = None
        self.xk = {}

        self.set_config(self.config)

        self.ema_alpha = 2.0 / (self.spans + 1.0)
        self.ema_alpha_ = 1.0 - self.ema_alpha

        self.ts_locked = {'cancel_orders': 0.0, 'decide': 0.0, 'update_open_orders': 0.0,
                          'update_position': 0.0, 'print': 0.0, 'create_orders': 0.0,
                          'check_fills': 0.0}
        self.ts_released = {k: 1.0 for k in self.ts_locked}

        self.position = {}
        self.open_orders = []
        self.fills = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0
        self.is_buyer_maker = True
        self.agg_qty = 0.0
        self.qty = 0.0
        self.ob = [0.0, 0.0]

        self.emas = np.zeros(len(self.spans))
        self.ratios = np.zeros(len(self.spans))

        self.n_open_orders_limit = 8
        self.n_orders_per_execution = 4

        self.hedge_mode = True
        self.c_mult = self.config['c_mult'] = 1.0

        self.log_filepath = make_get_filepath(f"logs/{self.exchange}/{config['config_name']}.log")

        self.key, self.secret = load_key_secret(config['exchange'], self.user)

        self.log_level = 0

        self.stop_websocket = False
        self.process_websocket_ticks = True
        self.lock_file = f"{str(Path.home())}/.{self.exchange}_passivbotlock"

    def set_config(self, config):
        config['spans'] = calc_spans(config['min_span'], config['max_span'], config['n_spans'])
        if 'stop_mode' not in config:
            config['stop_mode'] = None
        if 'last_price_diff_limit' not in config:
            config['last_price_diff_limit'] = 0.3
        if 'profit_trans_pct' not in config:
            config['profit_trans_pct'] = 0.0
        self.config = config
        for key in config:
            setattr(self, key, config[key])
            if key in self.xk:
                self.xk[key] = config[key]

    def set_config_value(self, key, value):
        self.config[key] = value
        setattr(self, key, self.config[key])

    async def _init(self):
        self.xk = create_xk(self.config)
        self.fills = await self.fetch_fills()

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
                ((qty_to_cost(position['long']['size'], position['long']['price'],
                              self.xk['inverse'], self.xk['c_mult'])
                  if position['long']['price'] else 0.0) +
                 (qty_to_cost(position['shrt']['size'], position['shrt']['price'],
                              self.xk['inverse'], self.xk['c_mult'])
                  if position['shrt']['price'] else 0.0)) / self.max_leverage
            position['available_margin'] = (position['equity'] - position['used_margin']) * 0.9
            position['long']['liq_diff'] = calc_diff(position['long']['liquidation_price'], self.price)
            position['shrt']['liq_diff'] = calc_diff(position['shrt']['liquidation_price'], self.price)
            position['long']['pbr'] = qty_to_cost(position['long']['size'], position['long']['price'],
                                                  self.xk['inverse'], self.xk['c_mult']) / position['wallet_balance']
            position['shrt']['pbr'] = qty_to_cost(position['shrt']['size'], position['shrt']['price'],
                                                  self.xk['inverse'], self.xk['c_mult']) / position['wallet_balance']
            if self.position != position:
                self.dump_log({'log_type': 'position', 'data': position})
            self.position = position
            self.ts_released['update_position'] = time()
        except Exception as e:
            print('error with update position', e)

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

        if self.hedge_mode:
            do_long = self.do_long or long_psize != 0.0
            do_shrt = self.do_shrt or shrt_psize != 0.0
        else:
            no_pos = long_psize == 0.0 and shrt_psize == 0.0
            do_long = (no_pos and self.do_long) or long_psize != 0.0
            do_shrt = (no_pos and self.do_shrt) or shrt_psize != 0.0
                                              
        self.xk['do_long'] = do_long
        self.xk['do_shrt'] = do_shrt

        if self.stop_mode in ['panic']:
            panic_orders = []
            if long_psize != 0.0:
                panic_orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(long_psize), 'price': self.ob[1],
                                     'type': 'market', 'reduce_only': True, 'custom_id': 'long_panic'})
            if shrt_psize != 0.0:
                panic_orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(shrt_psize), 'price': self.ob[0],
                                     'type': 'market', 'reduce_only': True, 'custom_id': 'shrt_panic'})
            return panic_orders


        orders = []
        long_closed, shrt_closed = long_psize == 0.0, shrt_psize == 0.0
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
            if not long_closed and long_close[0] != 0.0 and \
                    calc_diff(long_close[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'sell', 'position_side': 'long', 'qty': abs(float(long_close[0])),
                               'price': float(long_close[1]), 'type': 'limit', 'reduce_only': True,
                               'custom_id': long_close[4]})
                long_closed = True
            if not shrt_closed and shrt_close[0] != 0.0 and \
                    calc_diff(shrt_close[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'buy', 'position_side': 'shrt', 'qty': abs(float(shrt_close[0])),
                               'price': float(shrt_close[1]), 'type': 'limit', 'reduce_only': True,
                               'custom_id': shrt_close[4]})
                shrt_closed = True
            if self.stop_mode not in ['freeze'] and long_entry[0] != 0.0 and \
                    calc_diff(long_entry[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'buy', 'position_side': 'long', 'qty': float(long_entry[0]),
                               'price': float(long_entry[1]), 'type': 'limit', 'reduce_only': False,
                               'custom_id': long_entry[4]})
                long_psize, long_pprice = calc_new_psize_pprice(long_psize, long_pprice,
                                                                long_entry[0], long_entry[1], self.qty_step)
            else:
                long_done = True
            if self.stop_mode not in ['freeze'] and shrt_entry[0] != 0.0 and \
                    calc_diff(shrt_entry[1], self.price) < self.last_price_diff_limit:
                orders.append({'side': 'sell', 'position_side': 'shrt', 'qty': abs(float(shrt_entry[0])),
                               'price': float(shrt_entry[1]), 'type': 'limit', 'reduce_only': False,
                               'custom_id': shrt_entry[4]})
                shrt_psize, shrt_pprice = calc_new_psize_pprice(shrt_psize, shrt_pprice,
                                                                shrt_entry[0], shrt_entry[1], self.qty_step)
            else:
                shrt_done = True
            if len(orders) >= self.n_open_orders_limit or (long_done and shrt_done):
                break
        return orders


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

    async def check_fills(self):
        if self.ts_locked['check_fills'] > self.ts_released['check_fills']:
            # return if another call is in progress
            return
        now = time()
        if now - self.ts_released['check_fills'] < 5.0:
            # minimum 5 sec between consecutive check fills
            return
        self.ts_locked['check_fills'] = now
        print_(['checking if new fills...\n'], n=True)
        # check fills if two mins since prev check has passed
        fills = await self.fetch_fills()
        if self.fills != fills:
            await self.check_long_fills(fills)
            await self.check_shrt_fills(fills)

        self.fills = fills
        self.ts_released['check_fills'] = time()

    async def check_shrt_fills(self, fills):
        # closing orders
        new_shrt_closes = [item for item in fills if item not in self.fills and
                           item['side'] == 'buy' and item['position_side'] == 'shrt']
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
        new_shrt_entries = [item for item in fills if item not in self.fills and
                            item['side'] == 'sell' and item['position_side'] == 'shrt']
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

    async def check_long_fills(self, fills):
        #closing orders
        new_long_closes = [item for item in fills if item not in self.fills and
                          item['side'] == 'sell' and item['position_side'] == 'long']
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
        new_long_entries = [item for item in fills if item not in self.fills and
                            item['side'] == 'buy' and item['position_side'] == 'long']
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

    async def init_indicators(self):
        ticks = await self.fetch_ticks()
        if self.exchange == 'bybit' and 'linear' in self.market_type:
            print('\nwarning: insufficient ticks fetched')
            print('emas and ema ratios will be inaccurate until websocket catches up')
            self.emas = calc_emas(np.array([e['price'] for e in ticks]), self.spans)[-1]
        else:
            idxs = get_ids_to_fetch(self.spans, ticks[-1]['trade_id'])
            fetched_ticks = await asyncio.gather(*[self.fetch_ticks(from_id=int(i)) for i in idxs])
            compressed = drop_consecutive_same_prices(sorted(flatten(fetched_ticks) + ticks, key=lambda x: x['trade_id']))
            self.emas = calc_indicators_from_ticks_with_gaps(self.spans, compressed)
        self.ratios = np.append(self.price, self.emas[:-1]) / self.emas

    def update_indicators(self, ticks):
        for tick in ticks:
            self.agg_qty += tick['qty']
            if tick['price'] == self.price and tick['is_buyer_maker'] == self.is_buyer_maker:
                continue
            self.qty = self.agg_qty
            self.agg_qty = 0.0
            self.price = tick['price']
            self.is_buyer_maker = tick['is_buyer_maker']
            if tick['is_buyer_maker']:
                self.ob[0] = tick['price']
            else:
                self.ob[1] = tick['price']
            self.emas = self.emas * self.ema_alpha_ + tick['price'] * self.ema_alpha
            self.ratios = np.append(self.price, self.emas[:-1]) / self.emas

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        self.process_websocket_ticks = True
        print_([self.endpoints['websocket']])
        await self.update_position()
        await self.init_exchange_config()
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
            await asyncio.sleep(10)


async def _start_telegram(account: dict, bot: Bot):
    telegram = telegram_bot.Telegram(config=account['telegram'],
                                     bot=bot,
                                     loop=asyncio.get_event_loop())
    telegram.log_start()
    return telegram


def add_argparse_args(parser):
    parser.add_argument('--nojit', help='disable numba', action='store_true')
    parser.add_argument('-b', '--backtest_config', type=str, required=False, dest='backtest_config_path',
                        default='configs/backtest/default.hjson', help='backtest config hjson file')
    parser.add_argument('-o', '--optimize_config', type=str, required=False, dest='optimize_config_path',
                        default='configs/optimize/default.hjson', help='optimize config hjson file')
    parser.add_argument('-d', '--download-only', help='download only, do not dump ticks caches', action='store_true')
    parser.add_argument('-s', '--symbol', type=str, required=False, dest='symbol',
                        default=None, help='specify symbol, overriding symbol from backtest config')
    parser.add_argument('-u', '--user', type=str, required=False, dest='user',
                        default=None,
                        help='specify user, a.k.a. account_name, overriding user from backtest config')
    parser.add_argument('--start_date', type=str, required=False, dest='start_date',
                        default=None,
                        help='specify start date, overriding value from backtest config')
    parser.add_argument('--end_date', type=str, required=False, dest='end_date',
                        default=None,
                        help='specify end date, overriding value from backtest config')
    return parser


def get_passivbot_argparser():
    parser = argparse.ArgumentParser(prog='passivbot', description='run passivbot')
    parser.add_argument('user', type=str, help='user/account_name defined in api-keys.json')
    parser.add_argument('symbol', type=str, help='symbol to trade')
    parser.add_argument('live_config_path', type=str, help='live config to use')
    return parser


async def main() -> None:
    args = add_argparse_args(get_passivbot_argparser()).parse_args()
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
        print('using config')
        pprint.pprint(denumpyize(config))
    except Exception as e:
        print(e, 'failed to load config', args.live_config_path)
        return
    config['user'] = args.user
    config['exchange'] = account['exchange']
    config['symbol'] = args.symbol
    config['live_config_path'] = args.live_config_path

    if account['exchange'] == 'binance':
        from procedures import create_binance_bot
        bot = await create_binance_bot(config)
    elif account['exchange'] == 'bybit':
        from procedures import create_bybit_bot
        bot = await create_bybit_bot(config)
    else:
        raise Exception('unknown exchange', account['exchange'])

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
