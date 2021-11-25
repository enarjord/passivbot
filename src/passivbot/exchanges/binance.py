from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from passivbot.bot import Bot
from passivbot.datastructures import Fill
from passivbot.datastructures import Order
from passivbot.datastructures import Position
from passivbot.datastructures import Tick
from passivbot.datastructures.runtime import RuntimeFuturesConfig
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.httpclient import BinanceHTTPClient
from passivbot.utils.httpclient import HTTPRequestError
from passivbot.utils.procedures import print_async_exception

log = logging.getLogger(__name__)


class BinanceBot(Bot):
    def __bot_init__(self):
        """
        Subclass initialization routines
        """
        self.exchange = "binance"
        self.rtc: RuntimeFuturesConfig = RuntimeFuturesConfig(
            market_type=self.config.market_type, short=self.config.short, long=self.config.long
        )

    async def init_market_type(self):
        fapi_endpoint = "https://fapi.binance.com"
        dapi_endpoint = "https://dapi.binance.com"
        self.exchange_info = await BinanceHTTPClient.onetime_get(
            f"{fapi_endpoint}/fapi/v1/exchangeInfo"
        )
        if self.config.symbol in {e["symbol"] for e in self.exchange_info["symbols"]}:
            log.info("linear perpetual")
            self.rtc.market_type += "_linear_perpetual"
            self.rtc.inverse = False
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
                    "websocket_market": f"{websocket_url}/{self.config.symbol.lower()}@aggTrade",
                    "websocket_user": websocket_url,
                    "listen_key": "/fapi/v1/listenKey",
                },
            )
        else:
            self.exchange_info = await BinanceHTTPClient.onetime_get(
                f"{dapi_endpoint}/dapi/v1/exchangeInfo"
            )
            if self.config.symbol in {e["symbol"] for e in self.exchange_info["symbols"]}:
                log.info("inverse coin margined")
                self.rtc.market_type += "_inverse_coin_margined"
                self.rtc.inverse = True
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
                        "websocket_market": f"{websocket_url}/{self.config.symbol.lower()}@aggTrade",
                        "websocket_user": websocket_url,
                        "listen_key": "/dapi/v1/listenKey",
                    },
                )
            else:
                raise Exception(f"unknown symbol {self.config.symbol}")

        self.spot_base_endpoint = "https://api.binance.com"
        self.httpclient.endpoints["transfer"] = "https://api.binance.com/sapi/v1/asset/transfer"
        self.httpclient.endpoints["account"] = "https://api.binance.com/api/v3/account"

    async def _init(self):
        await self.init_market_type()
        for e in self.exchange_info["symbols"]:
            if e["symbol"] == self.config.symbol:
                self.rtc.coin = e["baseAsset"]
                self.rtc.quote = e["quoteAsset"]
                self.rtc.margin_coin = e["marginAsset"]
                self.rtc.pair = e["pair"]
                if "inverse_coin_margined" in self.rtc.market_type:
                    self.rtc.c_mult = float(e["contractSize"])
                for q in e["filters"]:
                    if q["filterType"] == "LOT_SIZE":
                        self.rtc.min_qty = float(q["minQty"])
                    elif q["filterType"] == "MARKET_LOT_SIZE":
                        self.rtc.qty_step = float(q["stepSize"])
                    elif q["filterType"] == "PRICE_FILTER":
                        self.rtc.price_step = float(q["tickSize"])
                    elif q["filterType"] == "MIN_NOTIONAL":
                        self.rtc.min_cost = float(q["notional"])
                break

        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def execute_leverage_change(self):
        lev = 7  # arbitrary
        return await self.httpclient.post(
            "leverage", params={"symbol": self.config.symbol, "leverage": lev}
        )

    async def init_exchange_config(self) -> None:
        try:
            ret = await self.httpclient.post(
                "margin_type",
                params={"symbol": self.config.symbol, "marginType": "CROSSED"},
            )
            log.info("Init Exchange Config: %s", ret)
        except HTTPRequestError as exc:
            if exc.code not in (-4046, -4059):
                raise
            log.info(exc.msg)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
        try:
            ret = await self.execute_leverage_change()
            log.info("Leverage Change: %s", ret)
        except Exception as e:
            log.error("Error: %s", e, exc_info=True)
        try:
            ret = await self.httpclient.post("position_side", params={"dualSidePosition": "true"})
            log.info("Position side: %s", ret)
        except HTTPRequestError as exc:
            if exc.code != -4059:
                raise
            log.info(exc.msg)
        except Exception as e:
            log.error("Unable to set hedge mode, aborting. Error: %s", e, exc_info=True)
            raise Exception("failed to set hedge mode")

    async def init_order_book(self):
        ticker: dict[str, Any] | list[dict[str, Any]]
        ticker = await self.httpclient.get("ticker", params={"symbol": self.config.symbol})
        if "inverse_coin_margined" in self.rtc.market_type:
            ticker = ticker[0]  # type: ignore[index]
        self.ob = [float(ticker["bidPrice"]), float(ticker["askPrice"])]  # type: ignore[call-overload]
        self.rtc.price = np.random.choice(self.ob)

    async def fetch_open_orders(self) -> list[Order]:
        return [
            Order.from_binance_payload(e, futures=True)
            for e in await self.httpclient.get(
                "open_orders", signed=True, params={"symbol": self.config.symbol}
            )
        ]

    async def fetch_position(self) -> Position:
        if "linear_perpetual" in self.rtc.market_type:
            params = {"symbol": self.config.symbol}
            assert self.rtc.quote
            asset = self.rtc.quote
        else:
            assert self.rtc.coin
            assert self.rtc.pair
            params = {"pair": self.rtc.pair}
            asset = self.rtc.coin

        positions: list[dict[str, Any]] = await self.httpclient.get("position", signed=True, params=params)  # type: ignore[assignment]
        balance: list[dict[str, Any]] = await self.httpclient.get("balance", signed=True)  # type: ignore[assignment]
        positions = [e for e in positions if e["symbol"] == self.config.symbol]
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
            if e["asset"] == asset:
                wallet_balance = float(e["balance"])
                position["wallet_balance"] = wallet_balance
                position["equity"] = wallet_balance + float(e["crossUnPnl"])
                break
        return Position.parse_obj(position)

    async def execute_order(self, order: Order) -> Order | None:
        o = None
        try:
            params = order.to_binance_payload(futures=True)
            o = await self.httpclient.post("create_order", params=params)
            log.debug("Create Order Returned Payload: %s", o)
            o["symbol"] = self.config.symbol
            return Order.from_binance_payload(o, futures=True)
        except Exception as e:
            log.info("error executing order %s: %s", order, e, exc_info=True)
            print_async_exception(o)
        return None

    async def execute_cancellation(self, order: Order) -> Order | None:
        cancellation = None
        try:
            cancellation = await self.httpclient.delete(
                "cancel_order",
                params={"symbol": self.config.symbol, "orderId": order.order_id},
            )

            cancellation["symbol"] = order.symbol
            return Order.from_binance_payload(cancellation, futures=True)
        except Exception as e:
            log.info("error cancelling order %s: %s", order, e, exc_info=True)
            print_async_exception(cancellation)
            self.ts_released["force_update"] = 0.0
        return None

    async def fetch_fills(
        self,
        symbol=None,
        limit: int = 1000,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Fill]:
        params = {
            "symbol": self.config.symbol if symbol is None else symbol,
            "limit": min(100, limit) if self.rtc.inverse else limit,
        }
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None and start_time is not None:
            params["endTime"] = int(min(end_time, start_time + 1000 * 60 * 60 * 24 * 6.99))
        try:
            fetched: list[dict[str, Any]] = await self.httpclient.get("fills", signed=True, params=params)  # type: ignore[assignment]
            fills = [
                Fill.from_binance_payload(x, futures=True, inverse=self.rtc.inverse)
                for x in fetched
            ]
        except Exception as e:
            log.error("error fetching fills: %s", e, exc_info=True)
            return []
        return fills

    async def get_all_income(
        self,
        symbol: str | None = None,
        start_time: int | None = None,
        income_type: str = "realized_pnl",
        end_time: int | None = None,
    ):
        income: list[dict[str, Any]] = []
        while True:
            fetched = await self.fetch_income(
                symbol=symbol, start_time=start_time, income_type=income_type, limit=1000
            )
            log.info("Fetched income: %s", ts_to_date(fetched[0]["timestamp"]))
            if fetched == income[-len(fetched) :]:
                break
            income += fetched
            if len(fetched) < 1000:
                break
            start_time = income[-1]["timestamp"]
        income_d = {e["transaction_id"]: e for e in income}
        return sorted(income_d.values(), key=lambda x: x["timestamp"])  # type: ignore[no-any-return]

    async def fetch_income(
        self,
        symbol: str | None = None,
        income_type: str | None = None,
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ):
        params: dict[str, Any] = {"limit": limit}
        if symbol is not None:
            params["symbol"] = symbol
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if income_type is not None:
            params["incomeType"] = income_type.upper()
        try:
            fetched: list[dict[str, Any]] = await self.httpclient.get("income", signed=True, params=params)  # type: ignore[assignment]
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
            log.error("error fetching income: %s", e, exc_info=True)
            return []

    async def fetch_account(self):
        try:
            return await self.httpclient.get("account", signed=True)
        except Exception as e:
            log.error("error fetching account: %s", e, exc_info=True)
            return {"balances": []}

    async def fetch_ticks(
        self,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        do_print: bool = True,
    ):
        params = {"symbol": self.config.symbol, "limit": 1000}
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        try:
            fetched: list[dict[str, Any]] = await self.httpclient.get("ticks", params=params)  # type: ignore[assignment]
        except Exception as e:
            log.error("error fetching ticks a: %s", e)
            return []
        try:
            ticks = [Tick.from_binance_payload(t) for t in fetched]
            if do_print:
                log.info(
                    "Fetched ticks for symbol %r %s %s",
                    self.config.symbol,
                    ticks[0].trade_id,
                    ts_to_date(float(ticks[0].timestamp) / 1000),
                )
        except Exception as e:
            log.error("error fetching ticks b: %s  %s", e, fetched)
            ticks = []
            if do_print:
                log.info("fetched no new ticks %s", self.config.symbol)
        return ticks

    async def fetch_ticks_time(
        self, start_time: int, end_time: int | None = None, do_print: bool = True
    ):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    async def fetch_ohlcvs(
        self,
        symbol: str | None = None,
        start_time: int | None = None,
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
            "symbol": self.config.symbol if symbol is None else symbol,
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
            log.error("error fetching ohlcvs: %s - %s", fetched, e, exc_info=True)

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        params = {"type": type_.upper(), "amount": amount, "asset": asset}
        return await self.httpclient.post("transfer", params=params)

    def standardize_market_stream_event(self, data: dict[str, Any]) -> list[Tick]:
        try:
            return [Tick.from_binance_payload(data)]
        except Exception as e:
            log.error("error in websocket tick: %s data: %s", e, data)
        return []

    async def beat_heart_user_stream(self, ws) -> None:
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        try:
            response = await self.httpclient.post("listen_key", signed=False)
            self.listen_key = response["listenKey"]
            self.httpclient.endpoints[
                "websocket_user"
            ] = f'{self.httpclient.endpoints["websocket"]}/{self.listen_key}'
        except Exception as e:
            log.error("error fetching listen key: %s", e, exc_info=True)

    def standardize_user_stream_event(self, event: dict[str, Any]) -> dict[str, Any]:
        standardized: dict[str, Any] = {}
        if "e" in event:
            if event["e"] == "ACCOUNT_UPDATE":
                if "a" in event and "B" in event["a"]:
                    for x in event["a"]["B"]:
                        if x["a"] == self.rtc.margin_coin:
                            standardized["wallet_balance"] = float(x["cw"])
                if event["a"]["m"] == "ORDER":
                    for x in event["a"]["P"]:
                        if x["s"] != self.config.symbol:
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
                if event["o"]["s"] == self.config.symbol:
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
