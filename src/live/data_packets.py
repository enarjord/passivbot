from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


ACCOUNT_PACKET_KINDS = frozenset({"balance", "positions", "open_orders"})


def _json_default(value: Any) -> str:
    return repr(value)


def stable_hash(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            default=_json_default,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FreshnessStatus:
    status: str = "unknown"
    age_ms: int | None = None
    max_age_ms: int | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"status": self.status}
        if self.age_ms is not None:
            out["age_ms"] = int(self.age_ms)
        if self.max_age_ms is not None:
            out["max_age_ms"] = int(self.max_age_ms)
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class DataPacketMetadata:
    kind: str
    scope: str
    revision: int
    cycle_hint: int | None = None
    call_started_ts_ms: int | None = None
    response_received_ts_ms: int | None = None
    exchange_server_ts_ms: int | None = None
    source: str | None = None
    raw_ref: str | None = None
    raw_hash: str | None = None
    coverage: Mapping[str, Any] = field(default_factory=dict)
    freshness: FreshnessStatus = field(default_factory=FreshnessStatus)
    quality: str = "unknown"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def with_revision(
        self, revision: int, *, cycle_hint: int | None = None
    ) -> "DataPacketMetadata":
        return replace(
            self,
            revision=int(revision),
            cycle_hint=self.cycle_hint if cycle_hint is None else int(cycle_hint),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "scope": self.scope,
            "revision": int(self.revision),
            "freshness": self.freshness.to_dict(),
            "quality": self.quality,
        }
        if self.cycle_hint is not None:
            out["cycle_hint"] = int(self.cycle_hint)
        if self.call_started_ts_ms is not None:
            out["call_started_ts_ms"] = int(self.call_started_ts_ms)
        if self.response_received_ts_ms is not None:
            out["response_received_ts_ms"] = int(self.response_received_ts_ms)
        if self.exchange_server_ts_ms is not None:
            out["exchange_server_ts_ms"] = int(self.exchange_server_ts_ms)
        if self.source is not None:
            out["source"] = self.source
        if self.raw_ref is not None:
            out["raw_ref"] = self.raw_ref
        if self.raw_hash is not None:
            out["raw_hash"] = self.raw_hash
        if self.coverage:
            out["coverage"] = dict(self.coverage)
        if self.warnings:
            out["warnings"] = list(self.warnings)
        if self.errors:
            out["errors"] = list(self.errors)
        return out


def data_packet_scope(kind: str) -> str:
    if kind in {"balance", "open_orders", "positions"}:
        return "global"
    return "global"


def data_packet_coverage(kind: str, value: Any) -> dict[str, Any]:
    if kind == "balance":
        return {"value_present": value is not None}
    if kind == "positions":
        try:
            row_count = len(value) if value is not None else 0
        except TypeError:
            row_count = 0
        return {"row_count": row_count}
    if kind == "open_orders":
        try:
            row_count = len(value) if value is not None else 0
        except TypeError:
            row_count = 0
        return {"row_count": row_count}
    return {}


def build_data_packet_metadata(
    *,
    kind: str,
    value: Any,
    raw_payload: Any = None,
    revision: int = 0,
    cycle_hint: int | None = None,
    call_started_ts_ms: int | None = None,
    response_received_ts_ms: int | None = None,
    exchange_server_ts_ms: int | None = None,
    source: str | None = None,
    quality: str = "ok",
    freshness_status: str = "fresh",
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> DataPacketMetadata:
    hash_source = raw_payload if raw_payload is not None else value
    raw_hash = stable_hash(hash_source) if hash_source is not None else None
    return DataPacketMetadata(
        kind=str(kind),
        scope=data_packet_scope(str(kind)),
        revision=int(revision),
        cycle_hint=cycle_hint,
        call_started_ts_ms=call_started_ts_ms,
        response_received_ts_ms=response_received_ts_ms,
        exchange_server_ts_ms=exchange_server_ts_ms,
        source=source,
        raw_ref=None if raw_hash is None else f"{kind}:{raw_hash[:16]}",
        raw_hash=raw_hash,
        coverage=data_packet_coverage(str(kind), value),
        freshness=FreshnessStatus(status=freshness_status),
        quality=quality,
        warnings=tuple(str(item) for item in warnings),
        errors=tuple(str(item) for item in errors),
    )


def packet_revision_signature(packets: Mapping[str, DataPacketMetadata]) -> tuple:
    return tuple(
        sorted(
            (str(kind), int(packet.revision), str(packet.raw_hash or ""))
            for kind, packet in packets.items()
        )
    )
