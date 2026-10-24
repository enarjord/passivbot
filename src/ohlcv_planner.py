from __future__ import annotations

from dataclasses import dataclass

from ohlcv_catalog import GapRecord, OhlcvCatalog


@dataclass(frozen=True)
class SymbolRangePlan:
    exchange: str
    timeframe: str
    symbol: str
    start_ts: int
    end_ts: int
    status: str
    bounds: tuple[int | None, int | None]
    persistent_gaps: tuple[GapRecord, ...]

    @property
    def local_store_complete(self) -> bool:
        return self.status == "store_complete"

    @property
    def blocked_by_persistent_gap(self) -> bool:
        return self.status == "blocked_by_persistent_gap"

    @property
    def requires_remote_fetch(self) -> bool:
        return self.status == "missing_local"


def plan_local_symbol_range(
    *,
    catalog: OhlcvCatalog,
    exchange: str,
    timeframe: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> SymbolRangePlan:
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")
    bounds = catalog.get_symbol_bounds(exchange, timeframe, symbol)
    store_complete = (
        bounds[0] is not None
        and bounds[1] is not None
        and int(bounds[0]) <= int(start_ts)
        and int(bounds[1]) >= int(end_ts)
    )
    persistent_gaps = tuple(
        catalog.get_persistent_gaps(exchange, timeframe, symbol, start_ts, end_ts)
    )
    if persistent_gaps:
        status = "blocked_by_persistent_gap"
    elif store_complete:
        status = "store_complete"
    else:
        status = "missing_local"
    return SymbolRangePlan(
        exchange=str(exchange),
        timeframe=str(timeframe),
        symbol=str(symbol),
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        status=status,
        bounds=bounds,
        persistent_gaps=persistent_gaps,
    )
