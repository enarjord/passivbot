from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from typing import Any, Iterable, Mapping, TypeVar

from live.event_bus import LiveEvent, emit_event


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
        emitted = emit_event(bot, self.to_live_event(bot), require_enqueue=True)
        if emitted is not None:
            return emitted
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

    def to_live_event(self, bot=None) -> LiveEvent:
        exchange = None
        user = None
        bot_id = None
        if bot is not None:
            exchange = getattr(bot, "exchange", None)
            try:
                user = bot.config_get(["live", "user"])
            except Exception:
                user = getattr(bot, "user", None)
            bot_id = getattr(bot, "bot_id", None)
        kwargs = {
            "event_type": self.kind,
            "level": "debug",
            "source": "live",
            "component": "diagnostic",
            "tags": self.tags,
            "exchange": None if exchange is None else str(exchange),
            "user": None if user is None else str(user),
            "bot_id": None if bot_id is None else str(bot_id),
            "symbol": self.symbol,
            "pside": self.pside,
            "data": dict(self.payload),
        }
        if self.ts_ms is not None:
            kwargs["ts_ms"] = self.ts_ms
        return LiveEvent(**kwargs)


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
