from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from freshness_ledger import FreshnessLedger
from market_snapshot import MarketSnapshot


@dataclass(frozen=True)
class PlanningSurfaceStamp:
    name: str
    epoch: int
    updated_ms: int
    signature: Any
    min_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "epoch": self.epoch,
            "updated_ms": self.updated_ms,
            "signature": self.signature,
            "min_epoch": self.min_epoch,
        }


@dataclass(frozen=True)
class PlanningMarketSnapshot:
    symbol: str
    bid: float
    ask: float
    last: float
    fetched_ms: int
    source: str

    @classmethod
    def from_snapshot(cls, snap: MarketSnapshot) -> "PlanningMarketSnapshot":
        return cls(
            symbol=str(snap.symbol),
            bid=float(snap.bid),
            ask=float(snap.ask),
            last=float(snap.last),
            fetched_ms=int(snap.fetched_ms),
            source=str(snap.source),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "fetched_ms": self.fetched_ms,
            "source": self.source,
        }


@dataclass(frozen=True)
class PlanningSnapshot:
    """Immutable contract for the exact live data set handed to Rust planning."""

    ts_ms: int
    exchange: str
    user: str
    epoch: int
    symbols: tuple[str, ...]
    required_surfaces: tuple[str, ...]
    surfaces: tuple[PlanningSurfaceStamp, ...]
    market_snapshots: tuple[PlanningMarketSnapshot, ...]
    market_snapshot_max_age_ms: int
    completed_candle_signature: Any

    @classmethod
    def capture(
        cls,
        *,
        ts_ms: int,
        exchange: str,
        user: str,
        ledger: FreshnessLedger,
        required_surfaces: Iterable[str],
        min_epochs: Mapping[str, int],
        symbols: Iterable[str],
        market_snapshots: Mapping[str, MarketSnapshot],
        market_snapshot_max_age_ms: int,
    ) -> "PlanningSnapshot":
        ordered_symbols = tuple(sorted(dict.fromkeys(str(symbol) for symbol in symbols if symbol)))
        required = tuple(sorted(dict.fromkeys(str(surface) for surface in required_surfaces)))
        surfaces = tuple(
            PlanningSurfaceStamp(
                name=surface,
                epoch=ledger.surface_epoch(surface),
                updated_ms=ledger.surface_updated_ms(surface),
                signature=ledger.surface_signature(surface),
                min_epoch=int(min_epochs[surface]),
            )
            for surface in required
        )
        rows = tuple(
            PlanningMarketSnapshot.from_snapshot(market_snapshots[symbol])
            for symbol in ordered_symbols
            if symbol in market_snapshots and market_snapshots[symbol].is_valid()
        )
        return cls(
            ts_ms=int(ts_ms),
            exchange=str(exchange),
            user=str(user),
            epoch=int(ledger.epoch),
            symbols=ordered_symbols,
            required_surfaces=required,
            surfaces=surfaces,
            market_snapshots=rows,
            market_snapshot_max_age_ms=int(market_snapshot_max_age_ms),
            completed_candle_signature=ledger.surface_signature("completed_candles"),
        )

    def invalid_details(self, *, now_ms: int) -> list[dict[str, Any]]:
        invalid: list[dict[str, Any]] = []
        for surface in self.surfaces:
            if surface.epoch < surface.min_epoch:
                invalid.append(
                    {
                        "surface": surface.name,
                        "reason": "surface_epoch_too_old",
                        "epoch": surface.epoch,
                        "min_epoch": surface.min_epoch,
                    }
                )
        market_by_symbol = {row.symbol: row for row in self.market_snapshots}
        missing_market = [symbol for symbol in self.symbols if symbol not in market_by_symbol]
        if missing_market:
            invalid.append(
                {
                    "surface": "market_snapshot",
                    "reason": "missing_symbols",
                    "symbols": missing_market,
                }
            )
        for symbol, row in market_by_symbol.items():
            age_ms = int(now_ms) - int(row.fetched_ms)
            if age_ms > self.market_snapshot_max_age_ms:
                invalid.append(
                    {
                        "surface": "market_snapshot",
                        "reason": "snapshot_too_old",
                        "symbol": symbol,
                        "age_ms": age_ms,
                        "max_age_ms": self.market_snapshot_max_age_ms,
                    }
                )
        return invalid

    def raise_if_invalid(self, *, now_ms: int, context: str) -> None:
        invalid = self.invalid_details(now_ms=now_ms)
        if invalid:
            raise RuntimeError(
                f"planning snapshot invalid before {context}: {invalid[:8]}"
            )

    def last_prices(self) -> dict[str, float]:
        return {row.symbol: row.last for row in self.market_snapshots}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "exchange": self.exchange,
            "user": self.user,
            "epoch": self.epoch,
            "symbols": list(self.symbols),
            "required_surfaces": list(self.required_surfaces),
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "market_snapshots": [row.to_dict() for row in self.market_snapshots],
            "market_snapshot_max_age_ms": self.market_snapshot_max_age_ms,
            "completed_candle_signature": self.completed_candle_signature,
        }
