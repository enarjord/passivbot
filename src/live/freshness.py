from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACCOUNT_SURFACES = frozenset({"balance", "positions", "open_orders", "fills"})
LIVE_STATE_SURFACES = ACCOUNT_SURFACES | frozenset({"completed_candles", "market_snapshot"})


@dataclass
class SurfaceState:
    name: str
    updated_ms: int = 0
    epoch: int = -1
    signature: Any = None
    changed_epoch: int = -1


class FreshnessLedger:
    """Track live data surface freshness."""

    def __init__(self, *, now_ms: int = 0) -> None:
        self.epoch = 0
        self.surfaces: dict[str, SurfaceState] = {
            surface: SurfaceState(name=surface) for surface in LIVE_STATE_SURFACES
        }
        self.created_ms = int(now_ms or 0)

    def begin_epoch(self, *, now_ms: int | None = None) -> int:
        self.epoch += 1
        return self.epoch

    def stamp(
        self,
        surface: str,
        signature: Any = None,
        *,
        now_ms: int,
        epoch: int | None = None,
    ) -> bool:
        if surface not in self.surfaces:
            self.surfaces[surface] = SurfaceState(name=surface)
        state = self.surfaces[surface]
        changed = state.signature != signature
        state.signature = signature
        state.updated_ms = int(now_ms)
        state.epoch = int(self.epoch if epoch is None else epoch)
        if changed:
            state.changed_epoch = state.epoch
        return changed

    def surface_epoch(self, surface: str) -> int:
        return max(
            0, int(self.surfaces.get(surface, SurfaceState(surface)).epoch)
        )

    def surface_signature(self, surface: str) -> Any:
        return self.surfaces.get(surface, SurfaceState(surface)).signature

    def surface_updated_ms(self, surface: str) -> int:
        return int(self.surfaces.get(surface, SurfaceState(surface)).updated_ms or 0)

    def surfaces_at_epoch(self, epoch: int | None = None) -> frozenset[str]:
        """Return surfaces stamped in the requested refresh cohort."""
        target_epoch = int(self.epoch if epoch is None else epoch)
        return frozenset(
            name
            for name, state in self.surfaces.items()
            if state.epoch == target_epoch
        )

    def changed_surfaces_at_epoch(self, epoch: int | None = None) -> frozenset[str]:
        """Return surfaces whose signature changed in the requested refresh cohort."""
        target_epoch = int(self.epoch if epoch is None else epoch)
        return frozenset(
            name
            for name, state in self.surfaces.items()
            if state.changed_epoch == target_epoch
        )
