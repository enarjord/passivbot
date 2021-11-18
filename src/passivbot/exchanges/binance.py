import asyncio
import logging
import time
import traceback
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from passivbot.bot import Bot
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.httpclient import BinanceHTTPClient
from passivbot.utils.httpclient import HTTPRequestError
from passivbot.utils.procedures import print_
from passivbot.utils.procedures import print_async_exception

log = logging.getLogger(__name__)


class BinanceBot(Bot):
    def __init__(self, config: Dict[str, Any]):
        self.exchange = "binance"
        super().__init__(config)

    async def init_market_type(self):
        fapi_endpoint = "https://fapi.binance.com"
        dapi_endpoint = "https://dapi.binance.com"
        self.exchange_info = await BinanceHTTPClient.onetime_get(
            f"{fapi_endpoint}/fapi/v1/exchangeInfo"
        )
        if self.symbol in {e["symbol"] for e in self.exchange_info["symbols"]}:
            print("linear perpetual")
            self.market_type += "_linear_perpetual"
            self.inverse = self.config["inverse"] = False
            websocket_url = "wss://fstream.binance.com/ws"
            self.httpclient = BinanceHTTPClient(
                fapi_endpoint,
                self.key,
                self.secret,
                endpoints={
                    "position": "/fapi/v2/positionRisk",
                    "balance": "/fapi/v2/balance",
                    "exchange_info": "/fapi/v1/exchangeInfo",
                    "leverage_bracket": "/fapi/v1/leverageBracket",
                    "open_orders": "/fapi/v1/openOrders",
                    "ticker": "/fapi/v1/ticker/bookTicker",
                    "fills": "/fapi/v1/userTrades",
                    "income": "/fapi/v1/income",
                    "create_order": "/fapi/v1/order",
                    "cancel_order": "/fapi/v1/order",
                    "ticks": "/fapi/v1/aggTrades",
                    "ohlcvs": "/fapi/v1/klines",
                    "margin_type": "/fapi/v1/marginType",
                    "leverage": "/fapi/v1/leverage",
                    "position_side": "/fapi/v1/positionSide/dual",
                    "websocket": websocket_url,
                    "websocket_market": f"{websocket_url}/{self.symbol.lower()}@aggTrade",
                    "websocket_user": websocket_url,
                    "listen_key": "/fapi/v1/listenKey",
                },
            )
        else:
            self.exchange_info = await BinanceHTTPClient.onetime_get(
                f"{dapi_endpoint}/dapi/v1/exchangeInfo"
            )
            if self.symbol in {e["symbol"] for e in self.exchange_info["symbols"]}:
                print("inverse coin margined")
                self.market_type += "_inverse_coin_margined"
                self.inverse = self.config["inverse"] = True
                websocket_url = "wss://dstream.binance.com/ws"
                self.httpclient = BinanceHTTPClient(
                    dapi_endpoint,
                    self.key,
                    self.secret,
                    endpoints={
                        "position": "/dapi/v1/positionRisk",
                        "balance": "/dapi/v1/balance",
                        "exchange_info": "/dapi/v1/exchangeInfo",
                        "leverage_bracket": "/dapi/v1/leverageBracket",
                        "open_orders": "/dapi/v1/openOrders",
                        "ticker": "/dapi/v1/ticker/bookTicker",
                        "fills": "/dapi/v1/userTrades",
                        "income": "/dapi/v1/income",
                        "create_order": "/dapi/v1/order",
                        "cancel_order": "/dapi/v1/order",
                        "ticks": "/dapi/v1/aggTrades",
                        "ohlcvs": "/dapi/v1/klines",
                        "margin_type": "/dapi/v1/marginType",
                        "leverage": "/dapi/v1/leverage",
                        "position_side": "/dapi/v1/positionSide/dual",
                        "websocket": websocket_url,
                        "websocket_market": f"{websocket_url}/{self.symbol.lower()}@aggTrade",
                        "websocket_user": websocket_url,
                        "listen_key": "/dapi/v1/listenKey",
                    },
                )
            else:
                raise Exception(f"unknown symbol {self.symbol}")

        self.spot_base_endpoint = "https://api.binance.com"
        self.httpclient.endpoints["transfer"] = "https://api.binance.com/sapi/v1/asset/transfer"
        self.httpclient.endpoints["account"] = "https://api.binance.com/api/v3/account"

    async def _init(self):
        await self.init_market_type()
        for e in self.exchange_info["symbols"]:
            if e["symbol"] == self.symbol:
                self.coin = e["baseAsset"]
                self.quot = e["quoteAsset"]
                self.margin_coin = e["marginAsset"]
                self.pair = e["pair"]
                if "inverse_coin_margined" in self.market_type:
                    self.c_mult = self.config["c_mult"] = float(e["contractSize"])
                for q in e["filters"]:
                    if q["filterType"] == "LOT_SIZE":
                        self.min_qty = self.config["min_qty"] = float(q["minQty"])
                    elif q["filterType"] == "MARKET_LOT_SIZE":
                        self.qty_step = self.config["qty_step"] = float(q["stepSize"])
                    elif q["filterType"] == "PRICE_FILTER":
                        self.price_step = self.config["price_step"] = float(q["tickSize"])
                    elif q["filterType"] == "MIN_NOTIONAL":
                        self.min_cost = self.config["min_cost"] = float(q["notional"])
                try:
                    self.min_cost
                except AttributeError:
                    self.min_cost = self.config["min_cost"] = 0.0
                break

        self.max_leverage = self.config["max_leverage"] = 25
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def execute_leverage_change(self):
        lev = 7  # arbitrary
        return await self.httpclient.post(
            "leverage", params={"symbol": self.symbol, "leverage": lev}
        )

    async def init_exchange_config(self) -> bool:
        try:
            print_(
                [
                    await self.httpclient.post(
                        "margin_type",
                        params={"symbol": self.symbol, "marginType": "CROSSED"},
                    )
                ]
            )
        except HTTPRequestError as exc:
            if exc.code not in (-4046, -4059):
                raise
            log.info(exc.msg)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
        try:
            print_([await self.execute_leverage_change()])
        except Exception as e:
            print(e)
        try:
            print_(
                [await self.httpclient.post("position_side", params={"dualSidePosition": "true"})]
            )
        except HTTPRequestError as exc:
            if exc.code != -4059:
                raise
            log.info(exc.msg)
        except Exception as e:
            log.error("Unable to set hedge mode, aborting. Error: %s", e, exc_info=True)
            raise Exception("failed to set hedge mode")

    async def init_order_book(self):
        ticker = await self.httpclient.get("ticker", params={"symbol": self.symbol})
        if "inverse_coin_margined" in self.market_type:
            ticker = ticker[0]
        self.ob = [float(ticker["bidPrice"]), float(ticker["askPrice"])]
        self.price = np.random.choice(self.ob)

    async def fetch_open_orders(self) -> List[Dict[str, Any]]:
        return [
            {
                "order_id": int(e["orderId"]),
                "symbol": e["symbol"],
                "price": float(e["price"]),
                "qty": float(e["origQty"]),
                "type": e["type"].lower(),
                "side": e["side"].lower(),
                "position_side": e["positionSide"].lower(),
                "timestamp": int(e["time"]),
            }
            for e in await self.httpclient.get(
                "open_orders", signed=True, params={"symbol": self.symbol}
            )
        ]

    async def fetch_position(self) -> Dict[str, Any]:
        positions, balance = await asyncio.gather(
            self.httpclient.get(
                "position",
                signed=True,
                params=(
                    {"symbol": self.symbol}
                    if "linear_perpetual" in self.market_type
                    else {"pair": self.pair}
                ),
            ),
            self.httpclient.get("balance", signed=True),
        )
        positions = [e for e in positions if e["symbol"] == self.symbol]
        position = {
            "long": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "wallet_balance": 0.0,
            "equity": 0.0,
        }
        if positions:
            for p in positions:
                if p["positionSide"] == "LONG":
                    position["long"] = {
                        "size": float(p["positionAmt"]),
                        "price": float(p["entryPrice"]),
                        "liquidation_price": float(p["liquidationPrice"]),
                    }
                elif p["positionSide"] == "SHORT":
                    position["short"] = {
                        "size": float(p["positionAmt"]),
                        "price": float(p["entryPrice"]),
                        "liquidation_price": float(p["liquidationPrice"]),
                    }
        for e in balance:
            if e["asset"] == (self.quot if "linear_perpetual" in self.market_type else self.coin):
                position["wallet_balance"] = float(e["balance"])
                position["equity"] = position["wallet_balance"] + float(e["crossUnPnl"])
                break
        return position

    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        o = None
        try:
            params = {
                "symbol": self.symbol,
                "side": order["side"].upper(),
                "positionSide": order["position_side"].upper(),
                "type": order["type"].upper(),
                "quantity": str(order["qty"]),
            }
            if params["type"] == "LIMIT":
                params["timeInForce"] = "GTX"
                params["price"] = order["price"]
            if "custom_id" in order:
                params[
                    "newClientOrderId"
                ] = f"{order['custom_id']}_{str(int(time.time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
            o = await self.httpclient.post("create_order", params=params)
            return {
                "symbol": self.symbol,
                "side": o["side"].lower(),
                "position_side": o["positionSide"].lower(),
                "type": o["type"].lower(),
                "qty": float(o["origQty"]),
                "order_id": int(o["orderId"]),
                "price": float(o["price"]),
            }
        except Exception as e:
            print(f"error executing order {order} {e}")
            print_async_exception(o)
            traceback.print_exc()
            return {}

    async def execute_cancellation(self, order: Dict[str, Any]) -> Dict[str, Any]:
        cancellation = None
        try:
            cancellation = await self.httpclient.delete(
                "cancel_order",
                params={"symbol": self.symbol, "orderId": order["order_id"]},
            )

            return {
                "symbol": self.symbol,
                "side": cancellation["side"].lower(),
                "order_id": int(cancellation["orderId"]),
                "position_side": cancellation["positionSide"].lower(),
                "qty": float(cancellation["origQty"]),
                "price": float(cancellation["price"]),
            }
        except Exception as e:
            print(f"error cancelling order {order} {e}")
            print_async_exception(cancellation)
            traceback.print_exc()
            self.ts_released["force_update"] = 0.0
            return {}

    async def fetch_fills(
        self,
        symbol=None,
        limit: int = 1000,
        from_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ):
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "limit": min(100, limit) if self.inverse else limit,
        }
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(min(end_time, start_time + 1000 * 60 * 60 * 24 * 6.99))
        try:
            fetched = await self.httpclient.get("fills", signed=True, params=params)
            fills = [
                {
                    "symbol": x["symbol"],
                    "id": int(x["id"]),
                    "order_id": int(x["orderId"]),
                    "side": x["side"].lower(),
                    "price": float(x["price"]),
                    "qty": float(x["qty"]),
                    "realized_pnl": float(x["realizedPnl"]),
                    "cost": float(x["baseQty"]) if self.inverse else float(x["quoteQty"]),
                    "fee_paid": float(x["commission"]),
                    "fee_token": x["commissionAsset"],
                    "timestamp": int(x["time"]),
                    "position_side": x["positionSide"].lower(),
                    "is_maker": x["maker"],
                }
                for x in fetched
            ]
        except Exception as e:
            print("error fetching fills", e)
            traceback.print_exc()
            return []
        return fills

    async def get_all_income(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        income_type: str = "realized_pnl",
        end_time: Optional[int] = None,
    ):
        income: List[Dict[str, Any]] = []
        while True:
            fetched = await self.fetch_income(
                symbol=symbol, start_time=start_time, income_type=income_type, limit=1000
            )
            print_(["fetched income", ts_to_date(fetched[0]["timestamp"])])
            if fetched == income[-len(fetched) :]:
                break
            income += fetched
            if len(fetched) < 1000:
                break
            start_time = income[-1]["timestamp"]
        income_d = {e["transaction_id"]: e for e in income}
        return sorted(income_d.values(), key=lambda x: x["timestamp"])

    async def fetch_income(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ):
        params = {"limit": limit}
        if symbol is not None:
            params["symbol"] = symbol
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if income_type is not None:
            params["incomeType"] = income_type.upper()
        try:
            fetched = await self.httpclient.get("income", signed=True, params=params)
            return [
                {
                    "symbol": e["symbol"],
                    "income_type": e["incomeType"].lower(),
                    "income": float(e["income"]),
                    "token": e["asset"],
                    "timestamp": float(e["time"]),
                    "info": e["info"],
                    "transaction_id": float(e["tranId"]),
                    "trade_id": float(e["tradeId"]),
                }
                for e in fetched
            ]
        except Exception as e:
            print("error fetching income: ", e)
            traceback.print_exc()
            return []

    async def fetch_account(self):
        try:
            return await self.httpclient.get("account", signed=True)
        except Exception as e:
            print("error fetching account: ", e)
            return {"balances": []}

    async def fetch_ticks(
        self,
        from_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        do_print: bool = True,
    ):
        params = {"symbol": self.symbol, "limit": 1000}
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        try:
            fetched = await self.httpclient.get("ticks", params=params)
        except Exception as e:
            print("error fetching ticks a", e)
            return []
        try:
            ticks = [
                {
                    "trade_id": int(t["a"]),
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "timestamp": int(t["T"]),
                    "is_buyer_maker": t["m"],
                }
                for t in fetched
            ]
            if do_print:
                print_(
                    [
                        "fetched ticks",
                        self.symbol,
                        ticks[0]["trade_id"],
                        ts_to_date(float(ticks[0]["timestamp"]) / 1000),
                    ]
                )
        except Exception as e:
            print("error fetching ticks b", e, fetched)
            ticks = []
            if do_print:
                print_(["fetched no new ticks", self.symbol])
        return ticks

    async def fetch_ticks_time(
        self, start_time: int, end_time: Optional[int] = None, do_print: bool = True
    ):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    async def fetch_ohlcvs(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[int] = None,
        interval="1m",
        limit=1500,
    ):
        # m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 60 * 60 * 24,
            "1w": 60 * 60 * 24 * 7,
            "1M": 60 * 60 * 24 * 30,
        }
        assert interval in interval_map
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
            params["endTime"] = params["startTime"] + interval_map[interval] * 60 * 1000 * limit
        try:
            fetched = await self.httpclient.get("ohlcvs", params=params)
            return [
                {
                    **{"timestamp": int(e[0])},
                    **{
                        k: float(e[i + 1])
                        for i, k in enumerate(["open", "high", "low", "close", "volume"])
                    },
                }
                for e in fetched
            ]
        except Exception as e:
            print("error fetching ohlcvs", fetched, e)
            traceback.print_exc()

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        params = {"type": type_.upper(), "amount": amount, "asset": asset}
        return await self.httpclient.post("transfer", params=params)

    def standardize_market_stream_event(self, data: dict) -> List[Dict[str, Any]]:
        try:
            return [
                {
                    "timestamp": int(data["T"]),
                    "price": float(data["p"]),
                    "qty": float(data["q"]),
                    "is_buyer_maker": data["m"],
                }
            ]
        except Exception as e:
            print("error in websocket tick", e, data)
        return []

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        try:
            response = await self.httpclient.post("listen_key")
            self.listen_key = response["listenKey"]
            self.httpclient.endpoints[
                "websocket_user"
            ] = f'{self.httpclient.endpoints["websocket"]}/{self.listen_key}'
        except Exception as e:
            traceback.print_exc()
            print_(["error fetching listen key", e])

    def standardize_user_stream_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        standardized = {}
        if "e" in event:
            if event["e"] == "ACCOUNT_UPDATE":
                if "a" in event and "B" in event["a"]:
                    for x in event["a"]["B"]:
                        if x["a"] == self.margin_coin:
                            standardized["wallet_balance"] = float(x["cw"])
                if event["a"]["m"] == "ORDER":
                    for x in event["a"]["P"]:
                        if x["s"] != self.symbol:
                            standardized["other_symbol"] = x["s"]
                            standardized["other_type"] = "account_update"
                            continue
                        if x["ps"] == "LONG":
                            standardized["long_psize"] = float(x["pa"])
                            standardized["long_pprice"] = float(x["ep"])
                        elif x["ps"] == "SHORT":
                            standardized["short_psize"] = float(x["pa"])
                            standardized["short_pprice"] = float(x["ep"])
            elif event["e"] == "ORDER_TRADE_UPDATE":
                if event["o"]["s"] == self.symbol:
                    if event["o"]["X"] == "NEW":
                        standardized["new_open_order"] = {
                            "order_id": int(event["o"]["i"]),
                            "symbol": event["o"]["s"],
                            "price": float(event["o"]["p"]),
                            "qty": float(event["o"]["q"]),
                            "type": event["o"]["o"].lower(),
                            "side": event["o"]["S"].lower(),
                            "position_side": event["o"]["ps"].lower(),
                            "timestamp": int(event["o"]["T"]),
                        }
                    elif event["o"]["X"] in ["CANCELED", "EXPIRED"]:
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                    elif event["o"]["X"] == "FILLED":
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                        standardized["filled"] = True
                    elif event["o"]["X"] == "PARTIALLY_FILLED":
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                        standardized["partially_filled"] = True
                else:
                    standardized["other_symbol"] = event["o"]["s"]
                    standardized["other_type"] = "order_update"
        return standardized
