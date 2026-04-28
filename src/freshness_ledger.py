from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ACCOUNT_SURFACES = frozenset({"balance", "positions", "open_orders", "fills"})
LIVE_STATE_SURFACES = ACCOUNT_SURFACES | frozenset({"completed_candles", "market_snapshot"})


@dataclass
class SurfaceState:
    name: str
    updated_ms: int = 0
    epoch: int = 0
    generation: int = 0
    signature: Any = None
    changed: bool = False


@dataclass
class SymbolBlock:
    symbol: str
    reason: str
    required_surfaces: frozenset[str]
    min_epoch: int
    detected_ms: int
    details: dict[str, Any] = field(default_factory=dict)


class FreshnessLedger:
    """Track live data surface freshness and symbol-level execution safety blocks."""

    def __init__(self, *, now_ms: int = 0) -> None:
        self.epoch = 0
        self.surfaces: dict[str, SurfaceState] = {
            surface: SurfaceState(name=surface) for surface in LIVE_STATE_SURFACES
        }
        self.symbol_blocks: dict[str, SymbolBlock] = {}
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
        state.changed = changed
        if changed:
            state.generation += 1
        self._clear_satisfied_symbol_blocks()
        return changed

    def surface_epoch(self, surface: str) -> int:
        return int(self.surfaces.get(surface, SurfaceState(surface)).epoch or 0)

    def surface_signature(self, surface: str) -> Any:
        return self.surfaces.get(surface, SurfaceState(surface)).signature

    def surface_updated_ms(self, surface: str) -> int:
        return int(self.surfaces.get(surface, SurfaceState(surface)).updated_ms or 0)

    def surfaces_missing_after(self, surfaces: set[str] | frozenset[str], min_epoch: int) -> list[str]:
        return sorted(surface for surface in surfaces if self.surface_epoch(surface) < int(min_epoch))

    def flag_symbol_block(
        self,
        symbol: str,
        *,
        reason: str,
        required_surfaces: set[str] | frozenset[str],
        min_epoch: int,
        detected_ms: int,
        details: dict[str, Any] | None = None,
    ) -> SymbolBlock:
        block = SymbolBlock(
            symbol=str(symbol),
            reason=str(reason),
            required_surfaces=frozenset(required_surfaces),
            min_epoch=int(min_epoch),
            detected_ms=int(detected_ms),
            details=dict(details or {}),
        )
        self.symbol_blocks[block.symbol] = block
        self._clear_satisfied_symbol_blocks()
        return block

    def blocked_symbols(self) -> dict[str, SymbolBlock]:
        self._clear_satisfied_symbol_blocks()
        return dict(self.symbol_blocks)

    def clear_symbol(self, symbol: str) -> None:
        self.symbol_blocks.pop(str(symbol), None)

    def _clear_satisfied_symbol_blocks(self) -> None:
        if not self.symbol_blocks:
            return
        for symbol, block in list(self.symbol_blocks.items()):
            if not self.surfaces_missing_after(block.required_surfaces, block.min_epoch):
                self.symbol_blocks.pop(symbol, None)
