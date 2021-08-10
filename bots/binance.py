import asyncio
import hashlib
import hmac
import json
from io import BytesIO
from time import time
from typing import Tuple, List
from typing import Union
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bots.base_bot import ORDER_UPDATE, ACCOUNT_UPDATE
from bots.base_live_bot import LiveBot, LiveConfig
from definitions.fill import Fill, empty_fill_list
from definitions.order import Order, TP, SL, LIMIT, MARKET, LQ, NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, TRADE, \
    CALCULATED, BUY, SELL, LONG, SHORT, BOTH, NEW_INSURANCE, NEW_ADL
from definitions.position import Position, empty_long_position, empty_short_position
from definitions.tick import Tick, empty_tick_list
from helpers.misc import sort_dict_keys, ts_to_date
from helpers.print_functions import print_, print_order

order_mapping = {'BUY': BUY, 'SELL': SELL, 'MARKET': MARKET, 'LIMIT': LIMIT, 'STOP': SL, 'TAKE_PROFIT': TP,
                 'LIQUIDATYION': LQ, 'NEW': NEW, 'CANCELED': CANCELED, 'CALCULATED': CALCULATED, 'EXPIRED': EXPIRED,
                 'TRADE': TRADE, 'PARTIALLY_FILLED': PARTIALLY_FILLED, 'FILLED': FILLED, 'LONG': LONG, 'SHORT': SHORT,
                 'BOTH': BOTH, 'NEW_INSURANCE': NEW_INSURANCE, 'NEW_ADL': NEW_ADL}

reverse_order_mapping = {TP: 'TAKE_PROFIT', SL: 'STOP_LOSS', LIMIT: 'LIMIT', MARKET: 'MARKET', BUY: 'BUY', SELL: 'SELL',
                         LONG: 'LONG', SHORT: 'SHORT'}


def mapping(item: str) -> str:
    """
    Function to map Binance specific order terminology to a neutral format.
    :param item: The item to map.
    :return: The neutral format.
    """
    try:
        return order_mapping[item.upper()]
    except Exception as e:
        print('Could not map', e)
        return ''


def reverse_mapping(item: str) -> str:
    """
    Function to map a neutral format to Binance specific order terminology.
    :param item: The item to map.
    :return: The Binance specific format.
    """
    try:
        return reverse_order_mapping[item]
    except Exception as e:
        print('Could not map', e)
        return ''


class BinanceBot(LiveBot):
    """
    Binance specific implementation of the live bot.
    """

    def __init__(self, config: LiveConfig, strategy):
        """
        Creates an instance of the Binance live bot with configuration and strategy. Sets endpoints to the Binance API
        endpoints.
        :param config: A live configuration class.
        :param strategy: A strategy implementing the logic.
        """
        super(BinanceBot, self).__init__(config, strategy)

        self.hedge_mode = True

        self.listenKey = None

    async def exchange_init(self):
        """
        Binance specific initialization. Sets it to hedge mode, gets exchange specific information, and sets the
        leverage.
        :return:
        """
        await self.market_type_init()
        try:
            res = await self.private_post(self.endpoints['position_side'], {'dualSidePosition': 'true'})
            print_([res], n=True)
        except Exception as e:
            if '"code":-4059' not in e.args[0]:
                print_([e, 'Unable to set hedge mode, aborting'], n=True)
                raise Exception('Failed to set hedge mode')
        try:
            print_([await self.private_post(self.endpoints['margin_type'],
                                            {'symbol': self.symbol, 'marginType': 'CROSSED'})], n=True)
        except Exception as e:
            print_([e], n=True)
        try:
            lev = await self.execute_leverage_change()
            print_(['Set leverage to', lev], n=True)
        except Exception as e:
            print_([e], n=True)

        await self.fetch_exchange_info()

    async def fetch_exchange_info(self):
        """
        Exchange specific information fetching. Gets values from the exchange.
        :return:
        """
        exchange_info = await self.public_get(self.endpoints['exchange_info'])

        for e in exchange_info['symbols']:
            if e['symbol'] == self.symbol:
                self.base_asset = e['baseAsset']
                self.quote_asset = e['quoteAsset']
                self.margin_asset = e['marginAsset']
                if 'inverse_coin_margined' in self.market_type:
                    self.contract_multiplier = float(e['contractSize'])
                for q in e['filters']:
                    if q['filterType'] == 'LOT_SIZE':
                        self.minimal_quantity = float(q['minQty'])
                    elif q['filterType'] == 'MARKET_LOT_SIZE':
                        self.quantity_step = float(q['stepSize'])
                    elif q['filterType'] == 'PRICE_FILTER':
                        self.price_step = float(q['tickSize'])
                    elif q['filterType'] == 'MIN_NOTIONAL':
                        self.minimal_cost = float(q['notional'])
                try:
                    z = self.minimal_cost
                except AttributeError:
                    self.minimal_cost = 0.0
                break

    async def market_type_init(self):
        """
        Exchange specific market initialization. Sets endpoints.
        :return:
        """
        fapi_endpoint = 'https://testnet.binancefuture.com'  # 'https://fapi.binance.com'
        dapi_endpoint = 'https://testnet.binancefuture.com'  # 'https://dapi.binance.com'
        fapi_info = await self.public_get('/fapi/v1/exchangeInfo', base_endpoint=fapi_endpoint)
        if self.symbol in {e['symbol'] for e in fapi_info['symbols']}:
            print('Identified as linear perpetual')
            self.market_type += '_linear_perpetual'
            self.inverse = False
            self.base_endpoint = fapi_endpoint
            self.endpoints = {
                'listenkey': '/fapi/v1/listenKey',
                'position': '/fapi/v2/positionRisk',
                'balance': '/fapi/v2/balance',
                'exchange_info': '/fapi/v1/exchangeInfo',
                'leverage_bracket': '/fapi/v1/leverageBracket',
                'open_orders': '/fapi/v1/openOrders',
                'ticker': '/fapi/v1/ticker/bookTicker',
                'fills': '/fapi/v1/userTrades',
                'income': '/fapi/v1/income',
                'create_order': '/fapi/v1/order',
                'cancel_order': '/fapi/v1/order',
                'ticks': '/fapi/v1/aggTrades',
                'ohlcvs': '/fapi/v1/klines',
                'margin_type': '/fapi/v1/marginType',
                'leverage': '/fapi/v1/leverage',
                'position_side': '/fapi/v1/positionSide/dual',
                'websocket': 'wss://stream.binancefuture.com/ws/',  # 'wss://fstream.binance.com/ws/'
                'websocket_user': '',
                'websocket_data': f"wss://stream.binancefuture.com/ws/{self.symbol.lower()}@aggTrade"
                # f"wss://fstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
            }
        else:
            dapi_info = await self.public_get('/fapi/v1/exchangeInfo', base_endpoint=dapi_endpoint)
            if self.symbol in {e['symbol'] for e in dapi_info['symbols']}:
                print('Identified as inverse coin margined')
                self.base_endpoint = dapi_endpoint
                self.market_type += '_inverse_coin_margined'
                self.inverse = True
                self.endpoints = {
                    'listenkey': '/dapi/v1/listenKey',
                    'position': '/dapi/v1/positionRisk',
                    'balance': '/dapi/v1/balance',
                    'exchange_info': '/dapi/v1/exchangeInfo',
                    'leverage_bracket': '/dapi/v1/leverageBracket',
                    'open_orders': '/dapi/v1/openOrders',
                    'ticker': '/dapi/v1/ticker/bookTicker',
                    'fills': '/dapi/v1/userTrades',
                    'income': '/dapi/v1/income',
                    'create_order': '/dapi/v1/order',
                    'cancel_order': '/dapi/v1/order',
                    'ticks': '/dapi/v1/aggTrades',
                    'ohlcvs': '/dapi/v1/klines',
                    'margin_type': '/dapi/v1/marginType',
                    'leverage': '/dapi/v1/leverage',
                    'position_side': '/dapi/v1/positionSide/dual',
                    'websocket': 'wss://stream.binancefuture.com/ws/',  # 'wss://dstream.binance.com/ws/'
                    'websocket_user': '',
                    'websocket_data': f"wss://stream.binancefuture.com/ws/{self.symbol.lower()}@aggTrade",
                    # f"wss://dstream.binance.com/ws/{self.symbol.lower()}@aggTrade"
                }
            else:
                raise Exception(f'Unknown symbol {self.symbol}')

        # self.spot_base_endpoint = 'https://api.binance.com'
        self.endpoints['transfer'] = '/sapi/v1/asset/transfer'
        self.endpoints['account'] = '/api/v3/account'

    async def fetch_orders(self) -> List[Order]:
        """
        Function to fetch current open orders. Fetches and converts all open orders.
        :return: A list of current open orders.
        """
        ords = await self.private_get(self.endpoints['open_orders'], {'symbol': self.symbol})
        orders = []
        for o in ords:
            if o['symbol'] == self.symbol:
                order = Order(o['symbol'].upper(),
                              int(o['orderId']),
                              float(o['price']),
                              float(o['stopPrice']),
                              float(o['origQty']),
                              mapping(o['type']),
                              mapping(o['side']),
                              int(o['time']),
                              mapping(o['status']),
                              mapping(o['positionSide']))
                if order.position_side == LONG or order.position_side == SHORT:
                    orders.append(order)
                else:
                    print_([o], n=True)
        return orders

    async def fetch_position(self) -> Tuple[Position, Position]:
        """
        Function to fetch current position. Fetches and converts long and short position.
        :return: The current long and short position.
        """
        pos = await self.private_get(self.endpoints['position'], ({'symbol': self.symbol}))
        long = empty_long_position()
        short = empty_short_position()
        for p in pos:
            if p['symbol'] == self.symbol:
                position = Position(p['symbol'].upper(),
                                    float(p['positionAmt']),
                                    float(p['entryPrice']),
                                    float(p['liquidationPrice']),
                                    float(p['unRealizedProfit']),
                                    float(p['leverage']),
                                    mapping(p['positionSide']))
                if position.position_side == LONG and position.size != 0.0:
                    long = position
                elif position.position_side == SHORT and position.size != 0.0:
                    short = position
        return long, short

    async def fetch_balance(self) -> float:
        """
        Function to fetch current balance. Fetches the balance for the base asset.
        :return: The current balance.
        """
        bal = await self.private_get(self.endpoints['balance'], {})
        for b in bal:
            if b['asset'] == self.quote_asset:
                return float(b['balance'])

    async def fetch_ticks(self, from_id: int = None, start_time: int = None, end_time: int = None,
                          do_print: bool = True) -> List[Tick]:
        """
        Function to fetch ticks, either based on ID or based on time.
        :param from_id: The ID from which to fetch.
        :param start_time: The start time from which to fetch.
        :param end_time: The end time to which to fetch.
        :param do_print: Whether to print output or not.
        :return: A list of Ticks.
        """
        params = {'symbol': self.symbol, 'limit': 1000}
        tick_list = empty_tick_list()
        if from_id is not None:
            params['fromId'] = max(0, from_id)
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        try:
            fetched = await self.private_get(self.endpoints['ticks'], params)
        except Exception as e:
            print_(['Error fetching ticks', e], n=True)
            return tick_list
        try:
            if len(fetched) > 0:
                for t in fetched:
                    tick_list.append(Tick(int(t['a']), int(t['T']), float(t['p']), float(t['q']), t['m']))
                if do_print:
                    print_(['fetched ticks', self.symbol, tick_list[0].trade_id,
                            ts_to_date(float(tick_list[0].timestamp) / 1000)])
        except Exception as e:
            print_(['Error fetching ticks', e, fetched], n=True)
            if do_print:
                print_(['Fetched no new ticks', self.symbol], n=True)
        return tick_list

    async def fetch_fills(self, from_id: int = None, start_time: int = None, end_time: int = None, limit: int = 1000) -> \
            List[Fill]:
        """
        Function to fetch fills, either based on ID or based on time.
        :param from_id: The ID from which to fetch.
        :param start_time: The start time from which to fetch.
        :param end_time: The end time to which to fetch.
        :param limit: Maximum fills to fetch.
        :return: A list of Fills.
        """
        params = {'symbol': self.symbol, 'limit': min(100, limit) if self.inverse else limit}
        fill_list = empty_fill_list()
        if from_id is not None:
            params['fromId'] = max(0, from_id)
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        try:
            fetched = await self.private_get(self.endpoints['fills'], params)
        except Exception as e:
            print_(['Error fetching fills', e], n=True)
            return fill_list
        try:
            if len(fetched) > 0:
                for f in fetched:
                    fill_list.append(
                        Fill(int(f['orderId']), int(f['time']), float(f['realizedPnl']), float(f['commission']), 0.0,
                             0.0, 0.0, float(f['qty']), float(f['price']), 0.0, 0.0,
                             LIMIT if bool(f['maker']) else MARKET, FILLED, mapping(f['side']),
                             mapping(f['positionSide'])))
                # fills = [{'symbol': x['symbol'],
                #           # 'id': int(x['id']),
                #           # 'order_id': int(x['orderId']),
                #           # 'side': x['side'].lower(),
                #           # 'price': float(x['price']),
                #           # 'qty': float(x['qty']),
                #           # 'realized_pnl': float(x['realizedPnl']),
                #           'cost': float(x['baseQty']) if self.inverse else float(x['quoteQty']),
                #           # 'fee_paid': float(x['commission']),
                #           'fee_token': x['commissionAsset'],
                #           # 'timestamp': int(x['time']),
                #           # 'position_side': x['positionSide'].lower().replace('short', 'shrt'),
                #           # 'is_maker': x['maker']
                #           } for x in fetched]
        except Exception as e:
            print_(['Error fetching fills', e, fetched], n=True)
            return fill_list
        return fill_list

    async def public_get(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        """
        Function for public API endpoints. Uses the underlying session to execute the call.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :param base_endpoint: Alternative base URL to use.
        :return: The answer decoded into json.
        """
        if base_endpoint is None:
            base_endpoint = self.base_endpoint
        async with self.session.get(base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(self, type_: str, base_endpoint: str, url: str, params: dict = {}) -> dict:
        """
        Base function for private API endpoints. Calculates signature, encoding, and headers. Uses the underlying
        session to execute the call.
        :param type_: The type of call to call specific function.
        :param base_endpoint: The base URL to use.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :return: The answer decoded into json.
        """
        timestamp = int(time() * 1000)
        params.update({'timestamp': timestamp, 'recvWindow': 5000})
        for k in params:
            if type(params[k]) == bool:
                params[k] = 'true' if params[k] else 'false'
            elif type(params[k]) == float:
                params[k] = str(params[k])
        params = sort_dict_keys(params)
        params['signature'] = hmac.new(self.secret.encode('utf-8'),
                                       urlencode(params).encode('utf-8'),
                                       hashlib.sha256).hexdigest()
        headers = {'X-MBX-APIKEY': self.key}
        async with getattr(self.session, type_)(base_endpoint + url, params=params, headers=headers) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        """
        Function for private GET API endpoints. Calls the base private function with correct type.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :param base_endpoint: Alternative base URL to use.
        :return: The answer string.
        """
        if base_endpoint is not None:
            return await self.private_('get', base_endpoint, url, params)
        else:
            return await self.private_('get', self.base_endpoint, url, params)

    async def private_post(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        """
        Function for private POST API endpoints. Calls the base private function with correct type.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :param base_endpoint: Alternative base URL to use.
        :return: The answer string.
        """
        if base_endpoint is not None:
            return await self.private_('post', base_endpoint, url, params)
        else:
            return await self.private_('post', self.base_endpoint, url, params)

    async def private_put(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        """
        Function for private PUT API endpoints. Calls the base private function with correct type.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :param base_endpoint: Alternative base URL to use.
        :return: The answer string.
        """
        if base_endpoint is not None:
            return await self.private_('put', base_endpoint, url, params)
        else:
            return await self.private_('put', self.base_endpoint, url, params)

    async def private_delete(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        """
        Function for private DELETE API endpoints. Calls the base private function with correct type.
        :param url: The URL to use in accordance with the base URL.
        :param params: The parameters to pass to the call.
        :param base_endpoint: Alternative base URL to use.
        :return: The answer string.
        """
        if base_endpoint is not None:
            return await self.private_('delete', base_endpoint, url, params)
        else:
            return await self.private_('delete', self.base_endpoint, url, params)

    def prepare_order(self, msg) -> Order:
        """
        Function to get an order in the correct format.
        :param msg: Message that needs to be translated.
        :return: An order object.
        """
        order = Order(msg['o']['s'].upper(),
                      int(msg['o']['i']),
                      float(msg['o']['p']),
                      0.0,
                      float(msg['o']['q']),
                      mapping(msg['o']['o']),
                      mapping(msg['o']['S']),
                      int(msg['o']['T']),
                      mapping(msg['o']['X']),
                      mapping(msg['o']['ps']))
        if 'ot' in msg['o']:
            if mapping(msg['o']['ot']) == MARKET and order.action != PARTIALLY_FILLED:
                order.price = float(msg['o']['ap'])
        if order.action == PARTIALLY_FILLED:
            order.quantity = order.quantity - float(msg['o']['z'])
        return order

    def prepare_account(self, msg) -> Tuple[float, Position, Position]:
        """
        Function to get an account update in the correct format.
        :param msg: Message that needs to be translated.
        :return: A tuple of balance, long position, and short position.
        """
        balance = None
        last_long = Position('', 0.0, 0.0, 0.0, 0.0, 0.0, '')
        last_short = Position('', 0.0, 0.0, 0.0, 0.0, 0.0, '')
        for b in msg['a']['B']:
            if b['a'].upper() == self.quote_asset:
                balance = float(b['wb'])
                break
        if 'P' in msg['a']:
            for p in msg['a']['P']:
                if p['s'] == self.symbol:
                    position = Position(p['s'].upper(),
                                        float(p['pa']),
                                        float(p['ep']),
                                        0.0,
                                        float(p['up']),
                                        self.leverage,
                                        mapping(p['ps']))
                    if position.position_side == LONG:
                        last_long = position
                    if position.position_side == SHORT:
                        last_short = position
        return balance, last_long, last_short

    def prepare_tick(self, msg) -> Tick:
        """
        Function to get a tick update in the correct format.
        :param msg: Message that needs to be translated.
        :return: A tick object.
        """
        tick = Tick(int(msg['a']), int(msg['T']), float(msg['p']), float(msg['q']), bool(msg['m']))
        return tick

    async def update_heartbeat(self):
        """
        Function that triggers an update of the websocket or initializes it. Uses an empty POST request to get a listen
        key. If a key exists, it uses an empty PUT request to the key endpoint to keep the key valid.
        :return:
        """
        if self.listenKey:
            try:
                await self.private_put(self.endpoints['listenkey'], {})
            except Exception as e_listen:
                print_(['Could not refresh listen key', e_listen], n=True)
        else:
            try:
                tmp = await self.private_post(self.endpoints['listenkey'], {})
                self.listenKey = tmp['listenKey']
                self.endpoints['websocket_user'] = self.endpoints['websocket'] + self.listenKey
            except Exception as e_listen:
                print_(['Could not initialize listen key', e_listen], n=True)

    def determine_update_type(self, msg) -> str:
        """
        Function that determines whether the message is an order or account update.
        :param msg: Message that needs to be identified.
        :return: ORDER_UPDATE or ACCOUNT_UPDATE.
        """
        type = None
        if 'e' in msg:
            if msg['e'] == 'ORDER_TRADE_UPDATE':
                if msg['o']['s'].upper() == self.symbol:
                    type = ORDER_UPDATE
            elif msg['e'] == 'ACCOUNT_UPDATE':
                type = ACCOUNT_UPDATE
        return type

    async def execute_leverage_change(self):
        """
        Function to execute the leverage change and ensure that the leverage is set correctly on the exchange.
        :return:
        """
        return await self.private_post(self.endpoints['leverage'],
                                       {'symbol': self.symbol, 'leverage': int(self.leverage)})

    async def execute_order(self, order: Order) -> Union[dict, bool]:
        """
        Executes an order creation request. Adds the necessary attributes to the parameters depending on the execution
        type.
        :param order: The order to create.
        :return: True if it was successful or the error message.
        """
        params = {'symbol': order.symbol,
                  'side': reverse_mapping(order.side),
                  'positionSide': reverse_mapping(order.position_side),
                  'type': reverse_mapping(order.order_type),
                  'quantity': str(order.quantity)}
        if params['type'] == LIMIT:
            params['timeInForce'] = 'GTX'
            params['price'] = str(order.price)
        if params['type'] == TP:
            params['price'] = str(order.price)
            params['stopPrice'] = str(order.stop_price)
        o = await self.private_post(self.endpoints['create_order'], params)
        if 'code' in o:
            return o
        else:
            return True

    async def execute_cancellation(self, order: Order) -> Union[dict, bool]:
        """
        Executes an order cancellation request.
        :param order: The order to cancel.
        :return: True if it was successful or the error message.
        """
        c = await self.private_delete(self.endpoints['cancel_order'],
                                      {'symbol': order.symbol, 'orderId': order.order_id})
        if 'code' in c:
            return c
        else:
            return True

    async def async_create_orders(self, orders_to_create: List[Order]):
        """
        Creates and schedules execution for each creation order. Checks whether the orders were received correctly.
        :param orders_to_create: Orders to create/send to the exchange.
        :return:
        """
        if not orders_to_create:
            return
        creations = []
        for order in orders_to_create:
            try:
                creations.append((order, asyncio.create_task(self.execute_order(order))))
            except Exception as e:
                print_(['Error creating order', print_order(order), e], n=True)
        for order, c in creations:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error creating order', print_order(order)], n=True)
                else:
                    print_(['Error creating order', print_order(order), o], n=True)
            except Exception as e:
                print_(['Error creating order', print_order(order), c.exception(), e], n=True)
        return

    async def async_cancel_orders(self, orders_to_cancel: List[Order]):
        """
        Creates and schedules execution for each cancellation order. Checks whether the orders were received correctly.
        :param orders_to_cancel: Orders to cancel/send to the exchange.
        :return:
        """
        if not orders_to_cancel:
            return
        deletions = []
        for order in orders_to_cancel:
            try:
                deletions.append((order, asyncio.create_task(self.execute_cancellation(order))))
            except Exception as e:
                print_(['Error cancelling order a', print_order(order), e], n=True)
        for order, c in deletions:
            try:
                o = await c
                if type(o) == bool:
                    if not o:
                        print_(['Error cancelling order', print_order(order)], n=True)
                else:
                    print_(['Error cancelling order', print_order(order), o], n=True)
            except Exception as e:
                print_(['Error cancelling order', print_order(order), c.exception(), e], n=True)
        return

    def fetch_from_repo(self, date: Union[Tuple[str, str], Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Function to allow fetching trade data from the Binance repository.
        :param date: The date, a tuple representing either a year and a month, or a year, a month, and a day. The order
        is year, month, day.
        :return: A dataframe with following columns: trade_id (int64), price (float64), qty (float64),
        timestamp (int64), is_buyer_maker (int8)
        """
        print_(['Fetching', self.symbol, date])
        # if self.spot:
        #     daily_base_url = "https://data.binance.vision/data/spot/daily/aggTrades/"
        #     monthly_base_url = "https://data.binance.vision/data/spot/monthly/aggTrades/"
        market_type = 'cm' if self.inverse else 'um'

        daily_base_url = f"https://data.binance.vision/data/futures/{market_type}/daily/aggTrades/"
        monthly_base_url = f"https://data.binance.vision/data/futures/{market_type}/monthly/aggTrades/"
        if len(date) == 2:
            url = "{}{}/{}-aggTrades-{}.zip".format(monthly_base_url, self.symbol.upper(), self.symbol.upper(),
                                                    date[0] + "-" + date[1])
        else:
            url = "{}{}/{}-aggTrades-{}.zip".format(daily_base_url, self.symbol.upper(), self.symbol.upper(),
                                                    date[0] + "-" + date[1] + "-" + date[2])
        df = pd.DataFrame(columns=['trade_id', 'price', 'qty', 'timestamp', 'is_buyer_maker'])
        column_names = ['trade_id', 'price', 'qty', 'first', 'last', 'timestamp', 'is_buyer_maker']
        # if self.spot:
        #     column_names.append('best_match')
        try:
            resp = urlopen(url)
            with ZipFile(BytesIO(resp.read())) as my_zip_file:
                for contained_file in my_zip_file.namelist():
                    tf = pd.read_csv(my_zip_file.open(contained_file), names=column_names)
                    tf.drop(errors='ignore', columns=['first', 'last', 'best_match'], inplace=True)
                    tf["trade_id"] = tf["trade_id"].astype(np.int64)
                    tf["price"] = tf["price"].astype(np.float64)
                    tf["qty"] = tf["qty"].astype(np.float64)
                    tf["timestamp"] = tf["timestamp"].astype(np.int64)
                    tf["is_buyer_maker"] = tf["is_buyer_maker"].astype(np.int8)
                    tf.sort_values("trade_id", inplace=True)
                    tf.drop_duplicates("trade_id", inplace=True)
                    tf.reset_index(drop=True, inplace=True)
                    if df.empty:
                        df = tf
                    else:
                        df = pd.concat([df, tf])
        except Exception as e:
            print('Failed to fetch', date, e)
            df = pd.DataFrame()
        return df
