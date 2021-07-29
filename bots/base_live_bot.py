import asyncio
import datetime
import json
from typing import Tuple, List

import aiohttp
import websockets
from numba import types
from numba.experimental import jitclass

from bots.base_bot import Bot, ORDER_UPDATE, ACCOUNT_UPDATE
from definitions.candle import Candle, empty_candle_list
from definitions.order import Order, empty_order_list
from definitions.position import Position
from definitions.tick import empty_tick_list
from helpers.loaders import load_key_secret
from helpers.print_functions import print_


@jitclass([
    ('symbol', types.string),
    ('user', types.string),
    ('exchange', types.string),
    ('leverage', types.int64),
    ('call_interval', types.float64)
])
class LiveConfig:
    """
    A class representing a live config.
    """

    def __init__(self, symbol: str, user: str, exchange: str, leverage: int, call_interval: float):
        """
        Creates a live config.
        :param symbol: The symbol to use.
        :param user: The user for the API keys.
        :param exchange: The exchange to use.
        :param leverage: The leverage to use.
        :param call_interval: Call interval for strategy to use in live.
        """
        self.symbol = symbol
        self.user = user
        self.exchange = exchange
        self.leverage = leverage
        self.call_interval = call_interval


class LiveBot(Bot):
    """
    Live implementation of the base bot class using async functions and websockets.
    """

    def __init__(self, config: LiveConfig, strategy):
        """

        :param config: A live configuration class.
        :param strategy: A strategy implementing the logic.
        """
        super(LiveBot, self).__init__()
        self.config = config
        self.strategy = strategy

        self.symbol = config.symbol

        self.user = config.user

        self.session = aiohttp.ClientSession()

        self.key, self.secret = load_key_secret(config.exchange, self.user)

        self.call_interval = config.call_interval

        self.base_endpoint = ''
        self.endpoints = {
            'listenkey': '',
            'position': '',
            'balance': '',
            'exchange_info': '',
            'leverage_bracket': '',
            'open_orders': '',
            'ticker': '',
            'fills': '',
            'income': '',
            'create_order': '',
            'cancel_order': '',
            'ticks': '',
            'margin_type': '',
            'leverage': '',
            'position_side': '',
            'websocket': '',
            'websocket_user': '',
            'websocket_data': ''
        }

    async def async_init(self):
        self.init()
        pass

    async def fetch_orders(self) -> List[Order]:
        raise NotImplementedError

    async def fetch_position(self) -> Tuple[Position, Position]:
        raise NotImplementedError

    async def fetch_balance(self) -> float:
        raise NotImplementedError

    async def public_get(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_(self, type_: str, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_get(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_post(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_put(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        raise NotImplementedError

    async def update_heartbeat(self):
        pass

    async def async_reset(self):
        self.reset()
        await self.async_init_orders()
        await self.async_init_position()
        await self.async_init_balance()
        self.strategy.update_values(self.get_balance(), self.get_position(), self.get_orders())

    async def async_init_orders(self):
        self.init_orders()
        a = await self.fetch_orders()
        add_orders = empty_order_list()
        delete_orders = empty_order_list()
        for order in a:
            add_orders.append(order)
        self.update_orders(add_orders, delete_orders)

    async def async_init_position(self):
        self.init_orders()
        long, short = await self.fetch_position()
        self.update_position(long, short)

    async def async_init_balance(self):
        self.init_balance()
        bal = await self.fetch_balance()
        self.update_balance(bal)

    async def async_handle_order_update(self, msg):
        self.handle_order_update(self.prepare_order(msg))
        if self.position_change and self.order_fill_change:
            asyncio.create_task(self.async_execute_strategy_update())

    async def async_handle_account_update(self, msg):
        self.handle_account_update(*self.prepare_account(msg))
        if self.position_change and self.order_fill_change:
            asyncio.create_task(self.async_execute_strategy_update())

    async def start_heartbeat(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.update_heartbeat()

    async def start_user_data(self) -> None:
        while True:
            try:
                self.position_change = False
                self.order_fill_change = False
                await self.async_reset()
                await self.update_heartbeat()
                async with websockets.connect(self.endpoints['websocket_user']) as ws:
                    async for msg in ws:
                        if msg is None:
                            continue
                        try:
                            msg = json.loads(msg)
                            type = self.determine_update_type(msg)
                            if type:
                                # print(msg)
                                if type == ORDER_UPDATE:
                                    asyncio.create_task(self.async_handle_order_update(msg))
                                elif type == ACCOUNT_UPDATE:
                                    asyncio.create_task(self.async_handle_account_update(msg))
                        except Exception as e:
                            print_(['User stream error inner', e], n=True)
            except Exception as e_out:
                print_(['User stream error outer', e_out], n=True)
                print_(['Retrying to connect in 5 seconds...'], n=True)
                await asyncio.sleep(5)

    async def start_websocket(self) -> None:
        while True:
            price_list = empty_candle_list()
            tick_list = empty_tick_list()
            last_tick_update = 0
            last_update = datetime.datetime.now()
            last_candle = Candle(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            async with websockets.connect(self.endpoints['websocket_data']) as ws:
                async for msg in ws:
                    if msg is None:
                        continue
                    try:
                        msg = json.loads(msg)
                        # print_([msg], n=True)
                        tick = self.prepare_tick(msg)
                        if last_tick_update == 0:
                            # Make sure it starts at a base unit
                            # If tick interval is 250ms the base unit is either 0.0, 0.25, 0.5, or 0.75 seconds
                            last_tick_update = int(tick.timestamp - (tick.timestamp % (self.tick_interval * 1000)))
                        # print_tick(tick)
                        if tick.timestamp - last_tick_update < self.tick_interval * 1000:
                            tick_list.append(tick)
                        else:
                            tick_list.append(tick)
                            # Calculate the time when the candle of the current tick ends
                            next_update = int(tick.timestamp - (tick.timestamp % (self.tick_interval * 1000))) + int(
                                self.tick_interval * 1000)
                            # Calculate a list of candles based on the given ticks, gaps are filled
                            # The tick list and last update are already updated
                            candles, tick_list, last_tick_update = self.prepare_candles(tick_list, last_tick_update,
                                                                                        next_update, last_candle)
                            if candles:
                                # Update last candle
                                last_candle = candles[-1]
                            # print_candle(last_candle)
                            # Extend candle list with new candles
                            price_list.extend(candles)
                        current = datetime.datetime.now()
                        if current - last_update >= datetime.timedelta(seconds=self.strategy.call_interval):
                            last_update = current
                            print_(['Do something'], n=True)
                            # asyncio.create_task(self.async_decide(price_list))
                            price_list = empty_candle_list()
                    except Exception as e:
                        if 'success' not in msg:
                            print_(['Error in price stream', e, msg], n=True)

    async def execute_leverage_change(self):
        raise NotImplementedError

    async def async_create_orders(self, orders_to_create: List[Order]):
        raise NotImplementedError

    async def async_cancel_orders(self, orders_to_cancel: List[Order]):
        raise NotImplementedError

    async def async_execute_strategy_update(self):
        add_orders, delete_orders = self.execute_strategy_update()

        asyncio.create_task(self.async_cancel_orders(delete_orders))
        asyncio.create_task(self.async_create_orders(add_orders))

    async def async_decide(self, prices: List[Candle]):
        add_orders, delete_orders = self.decide(prices)

        await self.async_cancel_orders(delete_orders)
        await self.async_create_orders(add_orders)
