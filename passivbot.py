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


def round_(n: float, step: float, safety_rounding=10) -> float:
    return np.round(np.round(n / step) * step, safety_rounding)


def calc_diff(x, y):
    return abs(x - y) / abs(y)


def sort_dict_keys(d):
    if type(d) == list:
        return [sort_dict_keys(e) for e in d]
    if type(d) != dict:
        return d
    return {key: sort_dict_keys(d[key]) for key in sorted(d)}


def calc_long_reentry_price(price_step: float,
                            grid_spacing: float,
                            grid_coefficient: float,
                            balance: float,
                            pos_margin: float,
                            pos_price: float):
    modified_grid_spacing = grid_spacing * (1 + pos_margin / balance * grid_coefficient)
    return round_dn(pos_price * (1 - modified_grid_spacing),
                    round_up(pos_price * grid_spacing / 4, price_step))


def calc_shrt_reentry_price(price_step: float,
                            grid_spacing: float,
                            grid_coefficient: float,
                            balance: float,
                            pos_margin: float,
                            pos_price: float):
    modified_grid_spacing = grid_spacing * (1 + pos_margin / balance * grid_coefficient)
    return round_up(pos_price * (1 + modified_grid_spacing),
                    round_up(pos_price * grid_spacing / 4, price_step))


def calc_long_closes(price_step: float,
                     qty_step: float,
                     min_qty: float,
                     min_markup: float,
                     max_markup: float,
                     pos_size: float,
                     pos_price: float,
                     lowest_ask: float,
                     n_orders: int = 10,
                     single_order_price_diff_threshold: float = 0.003):
    n_orders = int(round(min(n_orders, pos_size / min_qty)))
    prices = round_up(np.linspace(pos_price * (1 + min_markup), pos_price * (1 + max_markup),
                                  n_orders),
                      price_step)
    prices = np.unique(prices)
    prices = prices[np.where(prices >= lowest_ask)]
    if len(prices) == 0:
        return np.array([-pos_size]), np.array([lowest_ask])
    elif len(prices) == 1:
        return np.array([-pos_size]), prices
    elif calc_diff(prices[1], prices[0]) > single_order_price_diff_threshold:
        # too great spacing between prices, return single order
        return (np.array([-pos_size]), 
                np.array([max(lowest_ask, round_up(pos_price * (1 + min_markup), price_step))]))
    qtys = round_up(np.repeat(pos_size / len(prices), len(prices)), qty_step)
    qtys_sum = qtys.sum()
    while qtys_sum > pos_size:
        for i in range(len(qtys)):
            qtys[i] = round_(qtys[i] - min_qty, qty_step)
            qtys_sum = round_(qtys_sum - min_qty, qty_step)
            if qtys_sum <= pos_size:
                break
    return qtys * -1, prices


def calc_default_qty(min_qty: float,
                     qty_step: float,
                     balance_ito_contracts: float,
                     qty_balance_pct: float) -> float:
    return max(min_qty, round_dn(balance_ito_contracts * abs(qty_balance_pct), qty_step))


def calc_entry_qty(qty_step: float,
                   ddown_factor: float,
                   default_qty: float,
                   max_pos_size: float,
                   pos_size: float):
    abs_pos_size = abs(pos_size)
    qty_available = max(0.0, round_dn(max_pos_size - abs_pos_size, qty_step))
    return min(qty_available, max(default_qty, round_dn(abs_pos_size * ddown_factor, qty_step)))


def calc_shrt_closes(price_step: float,
                     qty_step: float,
                     min_qty: float,
                     min_markup: float,
                     max_markup: float,
                     pos_size: float,
                     pos_price: float,
                     highest_bid: float,
                     n_orders: int = 10,
                     single_order_price_diff_threshold: float = 0.003):
    abs_pos_size = abs(pos_size)
    n_orders = int(round(min(n_orders, abs_pos_size / min_qty)))
    prices = round_dn(np.linspace(pos_price * (1 - min_markup), pos_price * (1 - max_markup),
                                  n_orders),
                      price_step)
    prices = np.unique(prices)
    prices = -np.sort(-prices[np.where(prices <= highest_bid)])
    if len(prices) == 0:
        return np.array([-pos_size]), np.array([highest_bid])
    elif len(prices) == 1:
        return np.array([-pos_size]), prices
    elif calc_diff(prices[0], prices[1]) > single_order_price_diff_threshold:
        # too great spacing between prices, return single order
        return (np.array([-pos_size]),
                np.array([min(highest_bid, round_dn(pos_price * (1 - min_markup), price_step))]))
    qtys = round_up(np.repeat(abs_pos_size / len(prices), len(prices)), qty_step)
    qtys_sum = qtys.sum()
    while qtys_sum > abs_pos_size:
        for i in range(len(qtys) - 1, -1, -1):
            qtys[i] = round_(qtys[i] - min_qty, qty_step)
            qtys_sum = round_(qtys_sum - min_qty, qty_step)
            if qtys_sum <= abs_pos_size:
                break
    return qtys, prices


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
        cc = getattr(ccxt_async, exchange)
    try:
        cc = getattr(ccxt_async, exchange)({'apiKey': (ks := load_key_secret(exchange, user))[0],
                                            'secret': ks[1]})
    except Exception as e:
        print('error init ccxt', e)
        cc = getattr(ccxt_async, exchange)
    #print('ccxt enableRateLimit true')
    #cc.enableRateLimit = True
    return cc


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


def load_settings(exchange: str, user: str = 'default', print_=True) -> dict:
    fpath = f'live_settings/{exchange}/'
    try:
        settings = json.load(open(f'{fpath}{user}.json'))
    except FileNotFoundError:
        print_(f'settings for user {user} not found, using default settings')
        settings = json.load(open(f'{fpath}default.json'))
    if print_:
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


def flatten(lst: list) -> list:
    return [y for x in lst for y in x]


class Bot:
    def __init__(self, user: str, settings: dict):
        self.settings = settings
        self.indicator_settings = settings['indicator_settings']
        self.user = user
        self.symbol = settings['symbol']
        self.leverage = settings['leverage']
        self.liq_diff_threshold = settings['liq_diff_threshold']
        self.stop_loss_pos_reduction = settings['stop_loss_pos_reduction']
        self.grid_coefficient = settings['grid_coefficient']
        self.grid_spacing = settings['grid_spacing']
        self.max_markup = settings['max_markup']
        self.min_markup = settings['min_markup'] if self.max_markup >= settings['min_markup'] \
            else settings['max_markup']
        self.balance = settings['balance']
        self.n_entry_orders = settings['n_entry_orders']
        self.n_close_orders = settings['n_close_orders']
        self.default_qty = settings['default_qty']
        self.ddown_factor = settings['ddown_factor']

        self.market_stop_loss = settings['market_stop_loss']

        self.ts_locked = {'cancel_orders': 0, 'decide': 0, 'update_open_orders': 0,
                          'update_position': 0, 'print': 0, 'create_orders': 0}
        self.ts_released = {k: 1 for k in self.ts_locked}

        self.position = {}
        self.open_orders = []
        self.highest_bid = 0.0
        self.lowest_ask = 9.9e9
        self.price = 0
        self.ob = [0.0, 0.0]

        self.indicators = {'tick': {}, 'ohlcv': {}}
        self.ohlcvs = {}

        self.logs_base_filepath = make_get_filepath(
            f"logs/{self.exchange}/{ts_to_date(time())[:19].replace(':', '_')}.txt"
        )
        self.log_level = 0

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
                creations.append(self.execute_order(oc))
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
        self.ts_released['cancel_orders'] = time()
        return canceled_orders

    def stop(self) -> None:
        self.stop_websocket = True

    def determine_entry_side(self):
        # using indicators
        pass

    def calc_initial_bid_ask(self):
        bid_price = min(self.ob[0], round_dn(self.indicators['tick_ema'], self.price_step)) \
            if self.indicator_settings['do_long'] else 0.0
        ask_price = max(self.ob[1], round_up(self.indicators['tick_ema'], self.price_step)) \
            if self.indicator_settings['do_shrt'] else 9e9
        return bid_price, ask_price

    def calc_orders(self):
        last_price_diff_limit = 0.05
        balance = self.position['wallet_balance'] * min(1.0, abs(self.balance)) \
            if self.balance <= 0 else self.balance
        default_qty = self.default_qty if self.default_qty > 0.0 else \
            self.calc_default_qty(balance, self.price)
        orders = []
        if calc_diff(self.position['liquidation_price'], self.price) < self.liq_diff_threshold:
            if self.position['size'] > 0.0:
                # controlled long loss
                orders.append(
                    {'side': 'sell', 'type': 'market' if self.market_stop_loss else 'limit',
                     'qty': round_up(self.position['size'] * self.stop_loss_pos_reduction,
                                     self.qty_step),
                     'price': self.ob[1], 'reduce_only': True, 'custom_id': 'long_close_stop_loss'}
                )
            else:
                # controlled shrt loss
                orders.append(
                    {'side': 'buy', 'type': 'market' if self.market_stop_loss else 'limit',
                     'qty': round_up(-self.position['size'] * self.stop_loss_pos_reduction,
                                     self.qty_step),
                     'price': self.ob[0], 'reduce_only': True, 'custom_id': 'shrt_close_stop_loss'}
                )
            stop_loss_qty = orders[-1]['qty']
        else:
            stop_loss_qty = 0.0
        if self.position['size'] == 0: # no pos
            bid_price, ask_price = self.calc_initial_bid_ask()
            orders.append({'side': 'buy', 'qty': default_qty, 'price': bid_price,
                           'type': 'limit', 'reduce_only': False, 'custom_id': 'entry'})
            orders.append({'side': 'sell', 'qty': default_qty, 'price': ask_price,
                           'type': 'limit', 'reduce_only': False, 'custom_id': 'entry'})
        elif self.position['size'] > 0.0: # long pos
            pos_size = self.position['size']
            pos_price = self.position['price']
            pos_margin = self.calc_margin_cost(pos_size, pos_price)
            bid_price = min(self.ob[0], calc_long_reentry_price(self.price_step,
                                                                self.grid_spacing,
                                                                self.grid_coefficient,
                                                                balance,
                                                                pos_margin,
                                                                pos_price))
            for k in range(self.n_entry_orders):
                max_pos_size = self.calc_max_pos_size(min(balance, self.position['equity']),
                                                      bid_price)
                bid_qty = calc_entry_qty(self.qty_step, self.ddown_factor, default_qty,
                                         max_pos_size, pos_size)
                if bid_qty < default_qty:
                    break
                new_pos_size = pos_size + bid_qty
                if new_pos_size >= max_pos_size:
                    break
                pos_price = pos_price * (bid_qty / new_pos_size) + \
                    bid_price * (pos_size / new_pos_size)
                pos_size = new_pos_size
                pos_margin = self.calc_margin_cost(pos_size, pos_price)
                if calc_diff(bid_price, self.price) > last_price_diff_limit:
                    break
                orders.append({'side': 'buy', 'qty': bid_qty, 'price': bid_price,
                               'type': 'limit', 'reduce_only': False, 'custom_id': 'entry'})
                bid_price = min(self.ob[0], calc_long_reentry_price(self.price_step,
                                                                    self.grid_spacing,
                                                                    self.grid_coefficient,
                                                                    balance,
                                                                    pos_margin,
                                                                    pos_price))
        else: # shrt pos
            pos_size = self.position['size']
            pos_price = self.position['price']
            pos_margin = self.calc_margin_cost(-pos_size, pos_price)
            ask_price = max(self.ob[1], calc_shrt_reentry_price(self.price_step,
                                                                self.grid_spacing,
                                                                self.grid_coefficient,
                                                                balance,
                                                                pos_margin,
                                                                pos_price))
            for k in range(self.n_entry_orders):
                max_pos_size = self.calc_max_pos_size(min(balance, self.position['equity']),
                                                      ask_price)
                ask_qty = calc_entry_qty(self.qty_step, self.ddown_factor, default_qty,
                                         max_pos_size, pos_size)
                if ask_qty < default_qty:
                    break
                new_pos_size = pos_size - ask_qty
                if abs(new_pos_size) >= max_pos_size:
                    break
                pos_price = pos_price * (-ask_qty / new_pos_size) + \
                    ask_price * (pos_size / new_pos_size)
                pos_size = new_pos_size
                pos_margin = self.calc_margin_cost(-pos_size, pos_price)
                if calc_diff(ask_price, self.price) > last_price_diff_limit:
                    break
                orders.append({'side': 'sell', 'qty': ask_qty, 'price': ask_price,
                    'type': 'limit', 'reduce_only': False, 'custom_id': 'entry'})
                ask_price = max(self.ob[1], calc_shrt_reentry_price(self.price_step,
                                                                    self.grid_spacing,
                                                                    self.grid_coefficient,
                                                                    balance,
                                                                    pos_margin,
                                                                    pos_price))
        if self.position['size'] > 0.0:
            ask_qtys, ask_prices = calc_long_closes(self.price_step,
                                                    self.qty_step,
                                                    self.min_qty,
                                                    self.min_markup,
                                                    self.max_markup,
                                                    self.position['size'] - stop_loss_qty,
                                                    self.position['price'],
                                                    self.ob[1],
                                                    self.n_close_orders)
            close_orders = sorted([{'side': 'sell', 'qty': abs_qty, 'price': float(price_),
                                    'type': 'limit', 'reduce_only': True, 'custom_id': 'close'}
                                   for qty_, price_ in zip(ask_qtys, ask_prices)
                                   if (abs_qty := abs(float(qty_))) > 0.0
                                   and calc_diff(price_, self.price) < last_price_diff_limit],
                                  key=lambda x: x['price'])[:self.n_entry_orders]
            orders += close_orders
        elif self.position['size'] < 0.0:
            bid_qtys, bid_prices = calc_shrt_closes(self.price_step,
                                                    self.qty_step,
                                                    self.min_qty,
                                                    self.min_markup,
                                                    self.max_markup,
                                                    self.position['size'] + stop_loss_qty,
                                                    self.position['price'],
                                                    self.ob[0],
                                                    self.n_close_orders)
            close_orders = sorted([{'side': 'buy', 'qty': float(qty_), 'price': float(price_),
                                    'type': 'limit', 'reduce_only': True, 'custom_id': 'close'}
                                   for qty_, price_ in zip(bid_qtys, bid_prices) if qty_ > 0.0],
                                  key=lambda x: x['price'], reverse=True)[:self.n_entry_orders]
            orders += close_orders
        return orders

    async def cancel_and_create(self):
        await asyncio.sleep(0.1)
        await self.update_position()
        await asyncio.sleep(0.1)
        n_orders_limit = 4
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_orders(),
                                             keys=['side', 'qty', 'price'])
        to_cancel = sorted(to_cancel, key=lambda x: calc_diff(x['price'], self.price))
        to_create = sorted(to_create, key=lambda x: calc_diff(x['price'], self.price))
        tasks = []
        if to_cancel:
            tasks.append(self.cancel_orders(to_cancel[:n_orders_limit]))
        tasks.append(self.create_orders(to_create[:n_orders_limit]))
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
            self.ts_released['print'] = time()
            line = f"{self.symbol} "
            if self.position['size'] == 0:
                line += f"no position bid {self.highest_bid} ask {self.lowest_ask} "
                ratio = (self.price - self.highest_bid) / (self.lowest_ask - self.highest_bid)
            elif self.position['size'] > 0.0:
                line += f"long {self.position['size']} @ {self.position['price']:.2f} "
                line += f"exit {self.lowest_ask} ddown {self.highest_bid} "
                ratio = (self.price - self.highest_bid) / (self.lowest_ask - self.highest_bid)
            else:
                line += f"shrt {self.position['size']} @ {self.position['price']:.2f} "
                ratio = 1 - (self.price - self.highest_bid) / (self.lowest_ask - self.highest_bid)
                line += f"exit {self.highest_bid} ddown {self.lowest_ask } "

            liq_diff = calc_diff(self.position['liquidation_price'], self.price)
            line += f"pct {ratio:.2f} liq_diff {liq_diff:.3f} last {self.price}   "
            print_([line], r=True)

    def init_tick_ema(self, trades: [dict]):
        print_(['initiating tick ema...'])
        ema_span = self.indicator_settings['tick_ema']['span']
        ema = trades[0]['price']
        alpha = 2 / (ema_span + 1)
        for t in trades:
            ema = ema * (1 - alpha) + t['price'] * alpha
        self.indicators['tick_ema'] = ema
        self.indicator_settings['tick_ema']['alpha'] = alpha
        self.indicator_settings['tick_ema']['alpha_'] = 1 - alpha

    def update_tick_ema(self, websocket_tick):
        self.indicators['tick_ema'] = \
            self.indicators['tick_ema'] * self.indicator_settings['tick_ema']['alpha_'] + \
            websocket_tick['price'] * self.indicator_settings['tick_ema']['alpha']

    def init_fancy_indicator_001(self, trades: [dict]):
        pass

    def update_fancy_indicator_001(self, websocket_tick: dict):
        pass

    def init_fancy_indicator_002(self, trades: [dict]):
        pass

    def update_fancy_indicator_002(self, websocket_tick: dict):
        pass

    async def init_indicators(self):
        # called upon websocket start
        n_trades_to_fetch = 20000 # each fetch contains 1000 trades
        trades = await self.fetch_trades()
        additional_trades = await asyncio.gather(
            *[self.fetch_trades(from_id=trades[0]['trade_id'] - 1000 * i)
              for i in range(1, min(50, n_trades_to_fetch // 1000))])
        trades = sorted(trades + flatten(additional_trades), key=lambda x: x['trade_id'])
        self.init_tick_ema(trades)

    def update_indicators(self, websocket_tick: dict):
        # called each websocket tick
        # {'price': float, 'qty': float, 'timestamp': int, 'side': 'buy'|'sell'}
        self.update_tick_ema(websocket_tick)

    def init_ohlcv(self, period_ms: int, trades: [dict]):
        print_([f'initiating ohlcvs {period_ms}...'])
        self.ohlcvs[period_ms] = [{
            'timestamp': trades[0]['timestamp'] - trades[0]['timestamp'] % 10000,
            'open': trades[0]['price'],
            'high': trades[0]['price'],
            'low': trades[0]['price'],
            'close': trades[0]['price'],
            'volume': trades[0]['qty']
        }]
        for t in trades[1:]:
            self.update_ohlcv(period_ms, t)

    def update_ohlcv(self, period_ms, websocket_tick):

        if websocket_tick['timestamp'] > round(self.ohlcvs[period_ms][-1]['timestamp'] + period_ms):
            while websocket_tick['timestamp'] > \
                    round(self.ohlcvs[period_ms][-1]['timestamp'] + period_ms * 2):
                # fill empty ohlcvs
                self.ohlcvs[period_ms].append({
                    'timestamp': int(round(self.ohlcvs[period_ms][-1]['timestamp'] + period_ms)),
                    'open': self.ohlcvs[period_ms][-1]['close'],
                    'high': self.ohlcvs[period_ms][-1]['close'],
                    'low': self.ohlcvs[period_ms][-1]['close'],
                    'close': self.ohlcvs[period_ms][-1]['close'],
                    'volume': 0.0
                })
            # new ohlcv
            self.ohlcvs[period_ms].append({
                'timestamp': int(round(self.ohlcvs[period_ms][-1]['timestamp'] + period_ms)),
                'open': websocket_tick['price'],
                'high': websocket_tick['price'],
                'low': websocket_tick['price'],
                'close': websocket_tick['price'],
                'volume': websocket_tick['qty']
            })
        else:
            # update current ohlcv
            self.ohlcvs[period_ms][-1]['high'] = \
                max(self.ohlcvs[period_ms][-1]['high'], websocket_tick['price'])
            self.ohlcvs[period_ms][-1]['low'] = \
                min(self.ohlcvs[period_ms][-1]['low'], websocket_tick['price'])
            self.ohlcvs[period_ms][-1]['close'] = \
                websocket_tick['price']
            self.ohlcvs[period_ms][-1]['volume'] = \
                round(self.ohlcvs[period_ms][-1]['volume'] + websocket_tick['qty'], 10)
        if len(self.ohlcvs[period_ms]) > self.indicator_settings['max_periods_in_memory'] + 20:
            self.ohlcvs[period_ms] = \
                self.ohlcvs[period_ms][-self.indicator_settings['max_periods_in_memory']:]

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


async def start_bot(bot):
    await bot.start_websocket()

