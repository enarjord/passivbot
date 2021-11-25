from __future__ import annotations

import datetime
import enum
import logging
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
from pydantic import BaseModel
from pydantic import root_validator
from pydantic import validator

log = logging.getLogger(__name__)


class StopMode(enum.Enum):
    NORMAL = "normal"
    GRACEFUL = "graceful"
    MANUAL = "manual"
    PANIC = "panic"


class Fill(BaseModel):
    symbol: str
    id: str
    order_id: int
    side: str
    price: float
    qty: float
    cost: float
    realized_pnl: float
    fee_paid: float
    fee_token: str
    timestamp: int
    position_side: str
    is_maker: bool
    dt: datetime.datetime

    @validator("dt")
    @classmethod
    def _validate_dt(cls, value):
        if value and isinstance(value, int):
            # Late import to avoid circular imports issue
            from passivbot.utils.funcs.pure import ts_to_date

            value = ts_to_date(value, cast_to_str=False)
        return value

    @classmethod
    def from_binance_payload(
        cls, payload: Dict[str, Any], futures: bool = False, inverse: bool = False
    ) -> Fill:
        # Late import to avoid circular imports issue
        from passivbot.utils.funcs.pure import ts_to_date

        if inverse:
            payload["cost"] = payload["baseQty"]
        else:
            payload["cost"] = payload["quoteQty"]
        if futures is False:
            payload["side"] = "buy" if payload["isBuyer"] else "sell"
            payload["realized_pnl"] = 0.0
            payload["maker"] = payload["isMaker"]
            payload["position_side"] = "long"
        else:
            payload["position_side"] = payload["positionSide"].lower()
            payload["realized_pnl"] = float(payload["realizedPnl"])

        payload["timestamp"] = int(payload["time"])
        payload["dt"] = ts_to_date(payload["timestamp"], cast_to_str=False)

        return cls.parse_obj(
            {
                "symbol": payload["symbol"],
                "id": int(payload["id"]),
                "order_id": int(payload["orderId"]),
                "side": payload["side"].lower(),
                "price": float(payload["price"]),
                "qty": float(payload["qty"]),
                "cost": payload["cost"],
                "realized_pnl": payload["realized_pnl"],
                "fee_paid": float(payload["commission"]),
                "fee_token": payload["commissionAsset"],
                "timestamp": payload["timestamp"],
                "position_side": payload["position_side"],
                "is_maker": payload["maker"],
                "dt": payload["dt"],
            }
        )

    @classmethod
    def from_bybit_payload(cls, payload: Dict[str, Any]) -> Fill:
        raise NotADirectoryError


class Tick(BaseModel):
    trade_id: Optional[Union[int, str]]
    price: float
    qty: float
    timestamp: int
    is_buyer_maker: bool

    @classmethod
    def from_binance_payload(cls, payload: Dict[str, Any]) -> Tick:
        return cls.parse_obj(
            {
                "trade_id": payload["a"],
                "price": payload["p"],
                "qty": payload["q"],
                "timestamp": payload["T"],
                "is_buyer_maker": payload["m"],
            }
        )

    @classmethod
    def from_bybit_payload(cls, payload: Dict[str, Any]) -> Tick:
        # Late import to avoid circular imports issue
        from passivbot.utils.funcs.pure import date_to_ts

        if "time" in payload:
            payload["time"] = date_to_ts(int(payload["time"]))
        elif "trade_time_ms" in payload:
            payload["time"] = int(payload.pop("trade_time_ms"))

        renames = [
            ("id", "trade_id"),
            ("size", "qty"),
        ]
        for payload_key, target_key in renames:
            if payload_key in payload:
                payload[target_key] = payload.pop(payload_key)
        return cls.parse_obj(
            {
                "trade_id": payload["trade_id"],
                "price": payload["price"],
                "qty": payload["qty"],
                "timestamp": payload["time"],
                "is_buyer_maker": payload["side"] == "Sell",
            }
        )


class Order(BaseModel):
    order_id: Optional[int]
    custom_id: Optional[str]
    symbol: str
    price: float
    qty: float
    type: str
    side: str
    position_side: str
    timestamp: Optional[int]
    reduce_only: Optional[bool]
    close_position: Optional[bool]

    @classmethod
    def from_binance_payload(cls, payload, futures: bool = False) -> Order:
        renames = [
            ("orderId", "order_id"),
            ("origQty", "qty"),
        ]
        if futures:
            renames.extend(
                [
                    ("positionSide", "position_side"),
                    ("clientOrderId", "custom_id"),
                    ("updateTime", "timestamp"),
                    ("reduceOnly", "reduce_only"),
                    ("closePosiion", "close_position"),
                ]
            )
        else:
            payload["position_side"] = "long"
        for payload_key, target_key in renames:
            if payload_key in payload:
                payload[target_key] = payload.pop(payload_key)
        for key in ("type", "side", "position_side"):
            payload[key] = payload[key].lower()
        return cls.parse_obj(payload)

    def to_binance_payload(self, futures: bool = True) -> Dict[str, Any]:
        # Late import to avoid circular imports issue
        from passivbot.utils.funcs.pure import format_float

        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side.upper(),
            "type": self.type.upper(),
        }
        if futures:
            payload["positionSide"] = self.position_side.upper()
            payload["quantity"] = str(self.qty)
        else:
            payload["quantity"] = format_float(self.qty)
        if payload["type"] == "LIMIT":
            if futures:
                payload["timeInForce"] = "GTX"
                payload["price"] = str(self.price)
            else:
                payload["timeInForce"] = "GTC"
                payload["price"] = format_float(self.price)
        if self.custom_id:
            payload[
                "newClientOrderId"
            ] = f"{self.custom_id}_{str(int(time.time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        log.debug("Generated Order Payload: %s", payload)
        return payload

    @classmethod
    def from_bybit_payload(cls, payload, created_at_key: str = "created_at") -> Order:
        # Late import to avoid circular imports issue
        from passivbot.utils.funcs.pure import date_to_ts

        def determine_pos_side(o: dict[str, Any]) -> str | None:
            if o["side"].lower() == "buy":
                if "entry" in o["order_link_id"]:
                    return "long"
                elif "close" in o["order_link_id"]:
                    return "short"
                else:
                    return "both"
            else:
                if "entry" in o["order_link_id"]:
                    return "short"
                elif "close" in o["order_link_id"]:
                    return "long"
                else:
                    return "both"

        renames = (("order_link_id", "custom_id"),)
        for payload_key, target_key in renames:
            payload[target_key] = payload.pop(payload_key)
        payload["position_side"] = determine_pos_side(payload)
        payload["timestamp"] = date_to_ts(payload[created_at_key])
        for key in ("side",):
            payload[key] = payload[key].lower()
        return cls.parse_obj(payload)

    def to_bybit_payload(self, hedge_mode: bool, market_type: str) -> Dict[str, Any]:
        payload = {
            "symbol": self.symbol,
            "side": self.side.capitalize(),
            "order_type": self.type.capitalize(),
            "close_on_trigger": False,
        }
        if "linear_perpetual" in market_type:
            payload["qty"] = float(self.qty)
        else:
            payload["qty"] = int(self.qty)
        if hedge_mode:
            if self.position_side == "long":
                payload["position_idx"] = 1
            else:
                payload["position_idx"] = 2
            if "linear_perpetual" in market_type:
                payload["reduce_only"] = self.custom_id and "close" in self.custom_id
        else:
            payload["position_idx"] = 0
            payload["reduce_only"] = self.custom_id and "close" in self.custom_id
        if payload["order_type"] == "Limit":
            payload["time_in_force"] = "PostOnly"
            payload["price"] = str(self.price)
        else:
            payload["time_in_force"] = "GoodTillCancel"
        if self.custom_id is None:
            payload[
                "order_link_id"
            ] = f"{self.custom_id}_{str(int(time.time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        log.debug("Generated Order Payload: %s", payload)
        return payload


class LongPosition(BaseModel):
    size: float = 0.0
    price: float = 0.0
    liquidation_price: float = 0.0
    wallet_exposure: float = 0.0
    wallet_exposure_limit: float = 0.0


class ShortPosition(BaseModel):
    size: float = 0.0
    price: float = 0.0
    liquidation_price: float = 0.0
    wallet_exposure: float = 0.0
    wallet_exposure_limit: float = 0.0


class Position(BaseModel):
    equity: float = 0.0
    wallet_balance: float = 0.0
    long: LongPosition = LongPosition()
    short: ShortPosition = ShortPosition()


class Asset(BaseModel):
    free: float
    locked: float
    onhand: float


class Candle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
