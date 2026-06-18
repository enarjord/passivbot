from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from live.planning_snapshot import PlanningSnapshot


ACCOUNT_REQUIRED_SURFACES: tuple[str, ...] = ("balance", "positions", "open_orders")
STRATEGY_REQUIRED_SURFACES: tuple[str, ...] = ("completed_candles",)
FILL_REQUIRED_SURFACES: tuple[str, ...] = ("fills",)
MARKET_REQUIRED_SURFACES: tuple[str, ...] = ("market_snapshot",)

ORDER_CLASSES: tuple[str, ...] = (
    "initial_entry",
    "risk_increasing_entry",
    "take_profit_close",
    "trailing_close",
    "unstuck_close",
    "wel_twel_reduce_close",
    "hsl_panic_close",
    "entry_cancel",
    "protective_close_cancel",
)

POSITION_SIDES: tuple[str, ...] = ("long", "short")

ORDER_CLASS_REQUIRED_SURFACES: dict[str, tuple[str, ...]] = {
    "initial_entry": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *STRATEGY_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "risk_increasing_entry": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *STRATEGY_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "take_profit_close": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "trailing_close": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "unstuck_close": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "wel_twel_reduce_close": (
        *ACCOUNT_REQUIRED_SURFACES,
        *MARKET_REQUIRED_SURFACES,
        *FILL_REQUIRED_SURFACES,
    ),
    "hsl_panic_close": (*ACCOUNT_REQUIRED_SURFACES, *MARKET_REQUIRED_SURFACES),
    "entry_cancel": ACCOUNT_REQUIRED_SURFACES,
    "protective_close_cancel": ACCOUNT_REQUIRED_SURFACES,
}


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items if item))


def _surface_stamps(snapshot: PlanningSnapshot) -> dict[str, Any]:
    return {surface.name: surface for surface in snapshot.surfaces}


def _surface_epochs(snapshot: PlanningSnapshot) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted(
            (
                str(surface.name),
                int(surface.epoch),
                int(surface.min_epoch),
            )
            for surface in snapshot.surfaces
        )
    )


def _packet_revisions(snapshot: PlanningSnapshot) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted((packet.kind, int(packet.revision)) for packet in snapshot.data_packets)
    )


def _completed_candle_symbols(signature: Any) -> set[str]:
    if not isinstance(signature, (list, tuple)):
        return set()
    symbols: set[str] = set()
    for item in signature:
        if isinstance(item, (list, tuple)) and item:
            symbol = str(item[0] or "")
            if symbol:
                symbols.add(symbol)
    return symbols


def _surface_stamp_failure(
    stamps: dict[str, Any], surface: str
) -> tuple[str, str] | None:
    stamp = stamps.get(surface)
    if stamp is None:
        return surface, "missing_surface"
    if int(stamp.epoch) < int(stamp.min_epoch):
        return surface, "surface_epoch_too_old"
    return None


def _market_snapshot_failure(
    snapshot: PlanningSnapshot,
    *,
    symbol: str,
    now_ms: int,
) -> tuple[str, str] | None:
    rows = {row.symbol: row for row in snapshot.market_snapshots}
    row = rows.get(symbol)
    if row is None:
        return "market_snapshot", "missing_market_snapshot"
    age_ms = int(now_ms) - int(row.fetched_ms)
    if age_ms > int(snapshot.market_snapshot_max_age_ms):
        return "market_snapshot", "market_snapshot_too_old"
    return None


def _completed_candle_failure(
    snapshot: PlanningSnapshot, *, symbol: str
) -> tuple[str, str] | None:
    candle_symbols = _completed_candle_symbols(snapshot.completed_candle_signature)
    if symbol not in candle_symbols:
        return "completed_candles", "missing_completed_candles"
    return None


def _availability_failures(
    snapshot: PlanningSnapshot,
    *,
    symbol: str,
    required_surfaces: tuple[str, ...],
    now_ms: int,
) -> tuple[tuple[str, str], ...]:
    stamps = _surface_stamps(snapshot)
    failures: list[tuple[str, str]] = []
    for surface in required_surfaces:
        stamp_failure = _surface_stamp_failure(stamps, surface)
        if stamp_failure is not None:
            failures.append(stamp_failure)
            continue
        if surface == "market_snapshot":
            failure = _market_snapshot_failure(snapshot, symbol=symbol, now_ms=now_ms)
        elif surface == "completed_candles":
            failure = _completed_candle_failure(snapshot, symbol=symbol)
        else:
            failure = None
        if failure is not None:
            failures.append(failure)
    return tuple(failures)


@dataclass(frozen=True)
class PlanningAvailabilityRecord:
    cycle_id: int
    snapshot_id: str
    symbol: str
    position_side: str
    order_class: str
    status: str
    reason_code: str | None
    required_surfaces: tuple[str, ...]
    unavailable_surfaces: tuple[str, ...]
    packet_revisions: tuple[tuple[str, int], ...]
    surface_epochs: tuple[tuple[str, int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "position_side": self.position_side,
            "order_class": self.order_class,
            "status": self.status,
            "required_surfaces": list(self.required_surfaces),
            "unavailable_surfaces": list(self.unavailable_surfaces),
            "packet_revisions": {
                kind: revision for kind, revision in self.packet_revisions
            },
            "surface_epochs": {
                kind: {"epoch": epoch, "min_epoch": min_epoch}
                for kind, epoch, min_epoch in self.surface_epochs
            },
        }
        if self.reason_code is not None:
            out["reason_code"] = self.reason_code
        return out


@dataclass(frozen=True)
class PlanningAvailability:
    """Passive live-surface availability report for a frozen planning snapshot."""

    cycle_id: int
    snapshot_id: str
    records: tuple[PlanningAvailabilityRecord, ...]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PlanningSnapshot,
        *,
        now_ms: int,
        order_classes: Iterable[str] = ORDER_CLASSES,
        position_sides: Iterable[str] = POSITION_SIDES,
    ) -> "PlanningAvailability":
        packet_revisions = _packet_revisions(snapshot)
        surface_epochs = _surface_epochs(snapshot)
        records: list[PlanningAvailabilityRecord] = []
        for symbol in snapshot.symbols:
            for pside in position_sides:
                for order_class in order_classes:
                    required_surfaces = _dedupe(
                        ORDER_CLASS_REQUIRED_SURFACES.get(
                            str(order_class), snapshot.required_surfaces
                        )
                    )
                    failures = _availability_failures(
                        snapshot,
                        symbol=str(symbol),
                        required_surfaces=required_surfaces,
                        now_ms=int(now_ms),
                    )
                    unavailable_surfaces = _dedupe(
                        surface for surface, _reason in failures
                    )
                    reason_code = failures[0][1] if failures else None
                    records.append(
                        PlanningAvailabilityRecord(
                            cycle_id=int(snapshot.epoch),
                            snapshot_id=str(snapshot.snapshot_id),
                            symbol=str(symbol),
                            position_side=str(pside),
                            order_class=str(order_class),
                            status="available" if not failures else "unavailable",
                            reason_code=reason_code,
                            required_surfaces=required_surfaces,
                            unavailable_surfaces=unavailable_surfaces,
                            packet_revisions=packet_revisions,
                            surface_epochs=surface_epochs,
                        )
                    )
        return cls(
            cycle_id=int(snapshot.epoch),
            snapshot_id=str(snapshot.snapshot_id),
            records=tuple(records),
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status] = (
                counts[record.status] + 1 if record.status in counts else 1
            )
        return {
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "record_count": len(self.records),
            "status_counts": counts,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "records": [record.to_dict() for record in self.records],
        }
