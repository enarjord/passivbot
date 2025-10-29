"""Domain models used by the risk management dashboard and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence


@dataclass
class Position:
    """A lightweight representation of a trading position."""

    symbol: str
    side: str
    notional: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    wallet_exposure_pct: Optional[float]
    unrealized_pnl: float
    max_drawdown_pct: Optional[float]
    take_profit_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    size: Optional[float] = None
    signed_notional: Optional[float] = None
    volatility: Optional[Mapping[str, float]] = None
    funding_rates: Optional[Mapping[str, float]] = None
    daily_realized_pnl: float = 0.0

    def exposure_relative_to(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return abs(self.notional) / balance

    def pnl_pct(self, balance: float) -> float:
        if balance == 0:
            return 0.0
        return self.unrealized_pnl / balance


@dataclass
class Order:
    """Representation of an open order."""

    symbol: str
    side: str
    order_type: str
    price: Optional[float]
    amount: Optional[float]
    remaining: Optional[float]
    status: str
    reduce_only: bool
    stop_price: Optional[float] = None
    notional: Optional[float] = None
    order_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Account:
    """Account level snapshot."""

    name: str
    balance: float
    positions: Sequence[Position]
    orders: Sequence[Order] = ()
    daily_realized_pnl: float = 0.0

    def total_abs_notional(self) -> float:
        return sum(abs(p.notional) for p in self.positions)

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)

    def total_daily_realized(self) -> float:
        return sum(p.daily_realized_pnl for p in self.positions)

    def exposure_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return self.total_abs_notional() / self.balance

    def net_notional(self) -> float:
        total = 0.0
        for position in self.positions:
            if position.signed_notional is not None:
                total += position.signed_notional
            else:
                total += position.notional if position.side.lower() == "long" else -position.notional
        return total

    def gross_exposure_pct(self) -> float:
        return self.exposure_pct()

    def net_exposure_pct(self) -> float:
        if self.balance == 0:
            return 0.0
        return self.net_notional() / self.balance

    def exposures_by_symbol(self) -> Dict[str, Dict[str, float]]:
        exposures: Dict[str, Dict[str, float]] = {}
        for position in self.positions:
            signed = (
                position.signed_notional
                if position.signed_notional is not None
                else position.notional if position.side.lower() == "long" else -position.notional
            )
            data = exposures.setdefault(position.symbol, {"gross": 0.0, "net": 0.0})
            data["gross"] += abs(signed)
            data["net"] += signed
        return exposures


@dataclass
class AlertThresholds:
    wallet_exposure_pct: float = 0.6
    position_wallet_exposure_pct: float = 0.25
    max_drawdown_pct: float = 0.3
    loss_threshold_pct: float = -0.12


__all__ = [
    "Position",
    "Order",
    "Account",
    "AlertThresholds",
]
