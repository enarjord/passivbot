from passivbot_multi import Passivbot, logging
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
from pure_funcs import multi_replace
from procedures import print_async_exception


class BybitBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )

    async def init_bot(self):
        # require symbols to be formatted to ccxt standard COIN/USDT:USDT
        self.markets = await self.cca.fetch_markets()
        self.markets_dict = {elm["symbol"]: elm for elm in self.markets}
        approved_symbols = []
        for symbol in sorted(set(self.config["symbols"])):
            if not symbol.endswith("/USDT:USDT"):
                coin_extracted = multi_replace(symbol, [("/", ""), (":", ""), ("USDT", ""), ("BUSD", ""), ("USDC", "")])
                symbol_reformatted = coin_extracted + "/USDT:USDT"
                logging.info(f"symbol {symbol} is wrongly formatted. Trying to reformat to {symbol_reformatted}")
                symbol = symbol_reformatted
            if symbol not in self.markets_dict:
                logging.info(f"{symbol} missing from {self.exchange}")
            else:
                elm = self.markets_dict[symbol]
                if elm["type"] != "swap":
                    logging.info(f"wrong market type for {symbol}: {elm['type']}")
                elif not elm["active"]:
                    logging.info(f"{symbol} not active")
                elif not elm["linear"]:
                    logging.info(f"{symbol} is not a linear market")
                else:
                    approved_symbols.append(symbol)
        logging.info(f"approved symbols: {approved_symbols}")
        self.symbols = sorted(set(approved_symbols))
        for symbol in approved_symbols:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = 0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.tickers[symbol] = {"bid": 0.0, "ask": 0.0, "timestamp": 0.0}
            self.open_orders[symbol] = []
            self.positions[symbol] = {}
            self.upd_timestamps["open_orders"][symbol] = 0.0
            self.upd_timestamps["tickers"][symbol] = 0.0

    async def start_webstockets(self):
        await asyncio.gather(
            self.watch_balance(),
            self.watch_orders(),
            self.watch_tickers(),
        )

    async def watch_balance(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
                await self.handle_balance_update(res)
            except Exception as e:
                print(f"exception watch_balance", e)
                traceback.print_exc()

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                await self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()

    async def watch_tickers(self, symbols=None):
        if symbols is None:
            symbols = self.symbols
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_tickers(symbols)
                await self.handle_ticker_update(res)
            except Exception as e:
                print(f"exception watch_tickers {symbols}", e)
                traceback.print_exc()

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = {}
        limit = 50
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol, limit=limit)
            while True:
                if all([elm["id"] in open_orders for elm in fetched]):
                    break
                next_page_cursor = None
                for elm in fetched:
                    open_orders[elm["id"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                if len(fetched) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched = await self.cca.fetch_open_orders(
                    symbol=symbol, limit=limit, params={"cursor": next_page_cursor}
                )
            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self, symbol: str = None):
        fetched = None
        positions = {}
        limit = 200
        try:
            fetched = await self.cca.fetch_positions(symbol=symbol, limit=limit)
            while True:
                if all([elm["symbol"] + elm["side"] in positions for elm in fetched]):
                    break
                next_page_cursor = None
                for elm in fetched:
                    positions[elm["symbol"] + elm["side"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                    positions[elm["symbol"] + elm["side"]] = elm
                if len(fetched) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched = await self.cca.fetch_positions(
                    symbol=symbol, limit=limit, params={"cursor": next_page_cursor}
                )
            return sorted(positions.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False
