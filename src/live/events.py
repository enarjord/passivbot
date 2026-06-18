from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any, Iterable, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class DiagnosticEvent:
    """Typed diagnostic event envelope before monitor serialization."""

    kind: str
    tags: tuple[str, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)
    ts_ms: int | None = None
    symbol: str | None = None
    pside: str | None = None

    @classmethod
    def build(
        cls,
        kind: str,
        tags: Iterable[str],
        payload: Mapping[str, Any] | None = None,
        *,
        ts_ms: int | None = None,
        symbol: str | None = None,
        pside: str | None = None,
    ) -> "DiagnosticEvent":
        return cls(
            kind=str(kind),
            tags=tuple(str(tag) for tag in tags),
            payload=dict(payload or {}),
            ts_ms=None if ts_ms is None else int(ts_ms),
            symbol=None if symbol is None else str(symbol),
            pside=None if pside is None else str(pside),
        )

    def emit(self, bot) -> Any:
        recorder = getattr(bot, "_monitor_record_event", None)
        if not callable(recorder):
            return None
        try:
            return recorder(
                self.kind,
                self.tags,
                dict(self.payload),
                ts=self.ts_ms,
                symbol=self.symbol,
                pside=self.pside,
            )
        except Exception as exc:
            logging.debug(
                "[diagnostic] failed to emit %s event: %s",
                self.kind,
                exc,
            )
            return None


def emit_diagnostic_event(bot, event: DiagnosticEvent) -> Any:
    return event.emit(bot)


def run_diagnostic_step(
    label: str, func: Callable[[], T], *, default: T | None = None
) -> T | None:
    """Run best-effort diagnostic work without letting it affect live behavior."""
    try:
        return func()
    except Exception as exc:
        logging.debug(
            "[diagnostic] %s failed: %s",
            str(label),
            exc,
        )
        return default
