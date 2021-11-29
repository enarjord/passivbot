from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from passivbot.datastructures.config import LongConfig
from passivbot.datastructures.config import ShortConfig


class RuntimeExchangeConfig(BaseModel):
    market_type: str
    coin: Optional[str] = None
    quote: Optional[str] = None
    margin_coin: Optional[str] = None
    pair: Optional[str] = None
    min_qty: float = 0.0
    qty_step: float = 0.0
    price_step: float = 0.0
    min_cost: float = 0.0
    price: float = 0.0
    c_mult: float = 1.0
    hedge_mode: bool = True
    do_long: bool = True
    do_short: bool = True
    inverse: bool = True
    spot: bool = False


class RuntimeFuturesConfig(RuntimeExchangeConfig):
    market_type: str = "futures"
    max_leverage: int = 25
    short: ShortConfig
    long: LongConfig


class RuntimeSpotConfig(RuntimeExchangeConfig):
    spot: bool = True
    market_type: str = "spot"
    hedge_mode: bool = False
    do_long: bool = True
    do_short: bool = False
    inverse: bool = False
    min_price: float = 0.0
    max_price: float = 0.0
    price_multiplier_up: float = 1.0
    price_multiplier_dn: float = 1.0
    long: LongConfig
