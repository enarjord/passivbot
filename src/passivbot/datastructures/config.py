from __future__ import annotations

import pathlib
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from passivbot.datastructures import StopMode


class PassivbotBaseModel(BaseModel):
    """
    Base class for configurations
    """


class LongConfig(PassivbotBaseModel):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: int
    min_markup: float
    n_close_orders: int
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float


class ShortConfig(PassivbotBaseModel):
    enabled: bool
    eprice_exp_base: float
    eprice_pprice_diff: float
    grid_span: float
    initial_qty_pct: float
    markup_range: float
    max_n_entry_orders: int
    min_markup: float
    n_close_orders: int
    wallet_exposure_limit: float
    secondary_allocation: float
    secondary_pprice_diff: float


class NamedConfig(PassivbotBaseModel):
    exchange: str
    api_key_name: str
    symbol: str
    live_config_path: pathlib.Path
    market_type: str = "futures"
    assigned_balance: Optional[float] = None
    cross_wallet_pct: float = Field(1.0, ge=0.0, le=1.0)
    last_price_diff_limit: float = 0.3
    max_leverage: int = Field(25, ge=1)
    profit_trans_pct: float = 0.0
    stop_mode: StopMode = StopMode.NORMAL
    long: LongConfig
    short: ShortConfig
