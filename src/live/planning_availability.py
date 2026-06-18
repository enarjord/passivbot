from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from live.planning_snapshot import PlanningSnapshot


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
    packet_revisions: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "position_side": self.position_side,
            "order_class": self.order_class,
            "status": self.status,
            "required_surfaces": list(self.required_surfaces),
            "packet_revisions": {
                kind: revision for kind, revision in self.packet_revisions
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
        invalid = snapshot.invalid_details(now_ms=now_ms)
        status = "available" if not invalid else "unavailable"
        reason_code = (
            None
            if not invalid
            else str(invalid[0].get("reason") or "snapshot_invalid")
        )
        required_surfaces = tuple(snapshot.required_surfaces)
        packet_revisions = tuple(
            sorted(
                (packet.kind, int(packet.revision))
                for packet in snapshot.data_packets
            )
        )
        records = tuple(
            PlanningAvailabilityRecord(
                cycle_id=int(snapshot.epoch),
                snapshot_id=str(snapshot.snapshot_id),
                symbol=str(symbol),
                position_side=str(pside),
                order_class=str(order_class),
                status=status,
                reason_code=reason_code,
                required_surfaces=required_surfaces,
                packet_revisions=packet_revisions,
            )
            for symbol in snapshot.symbols
            for pside in position_sides
            for order_class in order_classes
        )
        return cls(
            cycle_id=int(snapshot.epoch),
            snapshot_id=str(snapshot.snapshot_id),
            records=records,
        )

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.status] = counts.get(record.status, 0) + 1
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
