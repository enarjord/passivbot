"""Read-only attribution of local fills to observed Passivbot runtime windows."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
DEFAULT_MAX_FILES = 20_000
DEFAULT_MAX_BYTES_PER_FILE = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_FILLS = 250_000
MAX_REPORTED_WARNINGS = 200
MAX_SOURCES_PER_RECORD = 32
MAX_RUNTIME_LOG_START_SKEW_MS = 2_000

_RUNTIME_KEYS = (
    "schema_version",
    "run_id",
    "started_at_ms",
    "passivbot_version",
    "python_git_commit",
    "python_git_dirty",
    "config_sha256",
    "rust_crate_version",
    "rust_source_sha256",
    "rust_artifact_sha256",
)
_LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+")
_STARTUP_BANNER_RE = re.compile(
    r"PASSIVBOT\s+│\s+(?P<exchange>[^:│\s]+):(?P<user>[^│\s]+)\s+│"
)
_RUNTIME_LINE_RE = re.compile(
    r"\[runtime\]\s+run=(?P<run_id>\S+)\s+pb=(?P<version>\S+)\s+"
    r"python=(?P<python>\S+)\s+config=(?P<config>\S+)\s+"
    r"rust_source=(?P<rust_source>\S+)\s+rust_artifact=(?P<rust_artifact>\S+)"
)
_RUNTIME_LOG_RUN_ID_PREFIX_RE = re.compile(r"^[0-9a-f]{12}$")
_FULL_RUNTIME_IDENTITY_SOURCE_KINDS = frozenset(
    {"runtime_manifest", "monitor_manifest", "monitor_event"}
)


class AttributionScanLimitError(ValueError):
    pass


@dataclass
class ScanBudget:
    max_files: int = DEFAULT_MAX_FILES
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES
    max_fills: int = DEFAULT_MAX_FILLS
    files_scanned: int = 0
    bytes_scanned: int = 0
    fills_scanned: int = 0
    warnings: list[dict[str, str]] = field(default_factory=list)
    warnings_suppressed: int = 0
    _seen: set[Path] = field(default_factory=set)

    def read_bytes(self, path: Path) -> bytes:
        resolved = path.resolve()
        if resolved not in self._seen:
            if self.files_scanned >= self.max_files:
                raise AttributionScanLimitError(
                    f"scan exceeds max_files={self.max_files}; narrow the input roots"
                )
            self._seen.add(resolved)
            self.files_scanned += 1
        size = path.stat().st_size
        if size > self.max_bytes_per_file:
            raise AttributionScanLimitError(
                f"file exceeds max_bytes_per_file={self.max_bytes_per_file}: {path}"
            )
        if path.suffix == ".gz":
            with gzip.open(path, "rb") as handle:
                data = handle.read(self.max_bytes_per_file + 1)
        else:
            data = path.read_bytes()
        charged_bytes = max(size, len(data))
        if self.bytes_scanned + charged_bytes > self.max_total_bytes:
            raise AttributionScanLimitError(
                f"scan exceeds max_total_bytes={self.max_total_bytes}; narrow the input roots"
            )
        self.bytes_scanned += charged_bytes
        return data

    def warn(self, path: Path, reason: str) -> None:
        if len(self.warnings) < MAX_REPORTED_WARNINGS:
            self.warnings.append({"path": str(path), "reason": str(reason)})
        else:
            self.warnings_suppressed += 1

    def count_fill(self) -> None:
        self.fills_scanned += 1
        if self.fills_scanned > self.max_fills:
            raise AttributionScanLimitError(
                f"scan exceeds max_fills={self.max_fills}; narrow the input roots or filters"
            )


def _iter_files(roots: Sequence[Path], predicate) -> Iterable[tuple[Path, Path]]:
    seen: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in candidates:
            if not path.is_file() or not predicate(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield root, path


def _scope_from_path(root: Path, path: Path) -> tuple[str, str]:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return "", ""
    if len(parts) >= 3:
        return str(parts[0]), str(parts[1])
    if root.parent.parent.name in {"fill_events", "runtime"}:
        return root.parent.name, root.name
    if root.parent.name in {"fill_events", "runtime"} and parts:
        return root.name, str(parts[0])
    return "", ""


def _append_source(record: dict[str, Any], source: dict[str, str]) -> None:
    sources = record.setdefault("sources", [])
    if source in sources:
        return
    if len(sources) < MAX_SOURCES_PER_RECORD:
        sources.append(source)
    else:
        record["sources_suppressed"] = int(record.get("sources_suppressed") or 0) + 1


def _resolve_filtered_scope(
    exchange: str,
    user: str,
    accepted_exchanges: set[str],
    accepted_users: set[str],
) -> Optional[tuple[str, str]]:
    if not exchange and len(accepted_exchanges) == 1:
        exchange = next(iter(accepted_exchanges))
    if not user and len(accepted_users) == 1:
        user = next(iter(accepted_users))
    if (exchange and not _matches(exchange, accepted_exchanges)) or (
        user and not _matches(user, accepted_users)
    ):
        return None
    return exchange, user


def _safe_json_bytes(path: Path, budget: ScanBudget) -> Any:
    try:
        raw = budget.read_bytes(path)
        if len(raw) > budget.max_bytes_per_file:
            raise AttributionScanLimitError(
                f"decompressed file exceeds max_bytes_per_file={budget.max_bytes_per_file}: {path}"
            )
        return json.loads(raw.decode("utf-8"))
    except AttributionScanLimitError:
        raise
    except Exception as exc:
        budget.warn(path, type(exc).__name__)
        return None


def _safe_ndjson(path: Path, budget: ScanBudget) -> Iterable[dict[str, Any]]:
    try:
        raw = budget.read_bytes(path)
        if len(raw) > budget.max_bytes_per_file:
            raise AttributionScanLimitError(
                f"decompressed file exceeds max_bytes_per_file={budget.max_bytes_per_file}: {path}"
            )
    except AttributionScanLimitError:
        raise
    except Exception as exc:
        budget.warn(path, type(exc).__name__)
        return
    for line_number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception:
            budget.warn(path, f"invalid_json_line:{line_number}")
            continue
        if isinstance(value, dict):
            yield value


def _normalize_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    identity = {key: value[key] for key in _RUNTIME_KEYS if key in value}
    if "run_id" in identity:
        identity["run_id"] = str(identity["run_id"])
    if "started_at_ms" in identity:
        try:
            identity["started_at_ms"] = int(identity["started_at_ms"])
        except (TypeError, ValueError):
            identity.pop("started_at_ms", None)
    return identity


def _runtime_record(
    identity: Mapping[str, Any],
    *,
    exchange: str,
    user: str,
    source: str,
    source_kind: str,
) -> Optional[dict[str, Any]]:
    normalized = _normalize_identity(identity)
    started_at_ms = int(normalized.get("started_at_ms") or 0)
    if started_at_ms <= 0:
        return None
    run_id = str(normalized.get("run_id") or "")
    if not run_id:
        digest = hashlib.sha256(
            f"{exchange}\0{user}\0{started_at_ms}".encode("utf-8")
        ).hexdigest()[:20]
        run_id = f"legacy-{digest}"
        normalized["run_id"] = run_id
    return {
        "run_id": run_id,
        "exchange": str(exchange or ""),
        "user": str(user or ""),
        "started_at_ms": started_at_ms,
        "ended_before_ms": None,
        "identity": normalized,
        "sources": [{"kind": source_kind, "path": source}],
    }


def _collect_runtime_manifests(
    roots: Sequence[Path],
    budget: ScanBudget,
    accepted_exchanges: set[str],
    accepted_users: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root, path in _iter_files(roots, lambda item: item.suffix == ".json"):
        exchange, user = _scope_from_path(root, path)
        scope = _resolve_filtered_scope(exchange, user, accepted_exchanges, accepted_users)
        if scope is None:
            continue
        exchange, user = scope
        value = _safe_json_bytes(path, budget)
        if not isinstance(value, dict):
            continue
        record = _runtime_record(
            value,
            exchange=str(value.get("exchange") or exchange),
            user=str(value.get("user") or user),
            source=str(path),
            source_kind="runtime_manifest",
        )
        if record is not None:
            records.append(record)
    return records


def _runtime_from_monitor_record(
    value: Mapping[str, Any],
    path: Path,
    *,
    fallback_exchange: str = "",
    fallback_user: str = "",
) -> Optional[dict[str, Any]]:
    kind = str(value.get("kind") or "")
    payload = value.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    live_event = payload.get("live_event")
    live_event = live_event if isinstance(live_event, Mapping) else {}
    data = live_event.get("data")
    data = data if isinstance(data, Mapping) else {}
    identity: Any = None
    if kind == "runtime.started":
        identity = data
    elif kind in {"bot.started", "bot.start"}:
        identity = data.get("runtime") or payload.get("runtime")
    if not isinstance(identity, Mapping):
        return None
    exchange = str(
        live_event.get("exchange")
        or value.get("exchange")
        or payload.get("exchange")
        or fallback_exchange
    )
    user = str(
        live_event.get("user")
        or value.get("user")
        or payload.get("user")
        or fallback_user
    )
    return _runtime_record(
        identity,
        exchange=exchange,
        user=user,
        source=str(path),
        source_kind="monitor_event",
    )


def _compact_fill(
    value: Mapping[str, Any],
    *,
    exchange: str,
    user: str,
    source: str,
    source_kind: str,
) -> Optional[dict[str, Any]]:
    try:
        timestamp = int(value.get("timestamp") or value.get("ts") or 0)
    except (TypeError, ValueError):
        return None
    fill_id = str(value.get("id") or "")
    if timestamp <= 0 or not fill_id:
        return None
    provenance = value.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, Mapping) else {}
    return {
        "id": fill_id,
        "timestamp": timestamp,
        "exchange": str(exchange or value.get("exchange") or ""),
        "user": str(user or value.get("user") or ""),
        "symbol": str(value.get("symbol") or ""),
        "position_side": str(value.get("position_side") or value.get("pside") or ""),
        "side": str(value.get("side") or ""),
        "pb_order_type": str(value.get("pb_order_type") or ""),
        "client_order_id": str(value.get("client_order_id") or ""),
        "source_ids": [str(item) for item in (value.get("source_ids") or []) if item],
        "provenance": provenance,
        "sources": [{"kind": source_kind, "path": source}],
    }


def _collect_fill_cache(
    roots: Sequence[Path],
    budget: ScanBudget,
    accepted_exchanges: set[str],
    accepted_users: set[str],
) -> list[dict[str, Any]]:
    fills: list[dict[str, Any]] = []
    for root, path in _iter_files(
        roots,
        lambda item: item.suffix == ".json" and item.name != "metadata.json",
    ):
        exchange, user = _scope_from_path(root, path)
        scope = _resolve_filtered_scope(exchange, user, accepted_exchanges, accepted_users)
        if scope is None:
            continue
        exchange, user = scope
        value = _safe_json_bytes(path, budget)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            fill = _compact_fill(
                item,
                exchange=exchange,
                user=user,
                source=str(path),
                source_kind="fill_cache",
            )
            if fill is not None:
                budget.count_fill()
                fills.append(fill)
    return fills


def _collect_monitor(
    roots: Sequence[Path],
    budget: ScanBudget,
    accepted_exchanges: set[str],
    accepted_users: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtimes: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    for root, path in _iter_files(
        roots,
        lambda item: item.name == "manifest.json"
        or item.name.endswith(".ndjson")
        or item.name.endswith(".ndjson.gz"),
    ):
        exchange, user = _scope_from_path(root, path)
        scope = _resolve_filtered_scope(exchange, user, accepted_exchanges, accepted_users)
        if scope is None:
            continue
        exchange, user = scope
        if path.name == "manifest.json":
            value = _safe_json_bytes(path, budget)
            if not isinstance(value, Mapping):
                continue
            identity = value.get("runtime")
            record = _runtime_record(
                identity if isinstance(identity, Mapping) else {},
                exchange=str(value.get("exchange") or exchange),
                user=str(value.get("user") or user),
                source=str(path),
                source_kind="monitor_manifest",
            )
            if record is not None:
                runtimes.append(record)
            continue
        for value in _safe_ndjson(path, budget):
            runtime = _runtime_from_monitor_record(
                value,
                path,
                fallback_exchange=exchange,
                fallback_user=user,
            )
            if runtime is not None:
                runtimes.append(runtime)
            if str(value.get("stream") or "") != "fills" or str(value.get("kind") or "") != "fill":
                continue
            payload = value.get("payload")
            if not isinstance(payload, Mapping):
                continue
            fill = _compact_fill(
                payload,
                exchange=str(value.get("exchange") or exchange),
                user=str(value.get("user") or user),
                source=str(path),
                source_kind="monitor_fill_history",
            )
            if fill is not None:
                budget.count_fill()
                fills.append(fill)
    return runtimes, fills


def _parse_log_ts(line: str) -> int:
    match = _LOG_TS_RE.match(line)
    if match is None:
        return 0
    try:
        parsed = datetime.strptime(match.group("ts"), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1000)


def _collect_logs(roots: Sequence[Path], budget: ScanBudget) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _root, path in _iter_files(
        roots,
        lambda item: item.suffix in {".log", ".gz"} or ".log." in item.name,
    ):
        try:
            raw = budget.read_bytes(path)
            if len(raw) > budget.max_bytes_per_file:
                raise AttributionScanLimitError(
                    "decompressed file exceeds "
                    f"max_bytes_per_file={budget.max_bytes_per_file}: {path}"
                )
        except AttributionScanLimitError:
            raise
        except Exception as exc:
            budget.warn(path, type(exc).__name__)
            continue
        latest: Optional[dict[str, Any]] = None
        for line in raw.decode("utf-8", errors="replace").splitlines():
            timestamp = _parse_log_ts(line)
            banner = _STARTUP_BANNER_RE.search(line)
            if banner is not None and timestamp > 0:
                latest = _runtime_record(
                    {"started_at_ms": timestamp},
                    exchange=banner.group("exchange"),
                    user=banner.group("user"),
                    source=str(path),
                    source_kind="legacy_startup_log",
                )
                if latest is not None:
                    records.append(latest)
                continue
            runtime_match = _RUNTIME_LINE_RE.search(line)
            if runtime_match is None or latest is None:
                continue
            if timestamp > 0 and timestamp - int(latest["started_at_ms"]) > 60_000:
                continue
            dirty_python = runtime_match.group("python")
            latest["run_id"] = runtime_match.group("run_id")
            latest["identity"].update(
                {
                    "run_id": runtime_match.group("run_id"),
                    "passivbot_version": runtime_match.group("version"),
                    "python_git_commit": dirty_python.removesuffix("+dirty"),
                    "python_git_dirty": dirty_python.endswith("+dirty"),
                    "config_sha256": runtime_match.group("config"),
                    "rust_source_sha256": runtime_match.group("rust_source"),
                    "rust_artifact_sha256": runtime_match.group("rust_artifact"),
                }
            )
            _append_source(latest, {"kind": "runtime_log", "path": str(path)})
    return records


def _is_full_manifest_or_event_identity(record: Mapping[str, Any]) -> bool:
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or any(
        identity.get(key) in (None, "", "unknown") for key in _RUNTIME_KEYS
    ):
        return False
    sources = record.get("sources")
    return isinstance(sources, list) and any(
        isinstance(source, Mapping)
        and source.get("kind") in _FULL_RUNTIME_IDENTITY_SOURCE_KINDS
        for source in sources
    )


def _has_bounded_runtime_log_identity(record: Mapping[str, Any]) -> bool:
    if _RUNTIME_LOG_RUN_ID_PREFIX_RE.fullmatch(str(record.get("run_id") or "")) is None:
        return False
    sources = record.get("sources")
    return isinstance(sources, list) and any(
        isinstance(source, Mapping) and source.get("kind") == "runtime_log"
        for source in sources
    )


def _merge_runtimes(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: (
            str(item.get("exchange") or ""),
            str(item.get("user") or ""),
            int(item.get("started_at_ms") or 0),
            str(item.get("run_id") or ""),
        ),
    )
    merged: list[dict[str, Any]] = []
    by_run: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in ordered:
        key = (record["exchange"], record["user"], record["run_id"])
        existing = by_run.get(key)
        if existing is None:
            merged.append(record)
            by_run[key] = record
            continue
        existing["started_at_ms"] = min(existing["started_at_ms"], record["started_at_ms"])
        existing["identity"].update(
            {key: value for key, value in record["identity"].items() if value not in (None, "")}
        )
        for source in record["sources"]:
            _append_source(existing, source)

    # A legacy banner and a manifest/event within one second describe the same start.
    for legacy in list(merged):
        if not str(legacy["run_id"]).startswith("legacy-"):
            continue
        candidates = [
            item
            for item in merged
            if item is not legacy
            and item["exchange"] == legacy["exchange"]
            and item["user"] == legacy["user"]
            and abs(item["started_at_ms"] - legacy["started_at_ms"]) <= 1_000
            and not str(item["run_id"]).startswith("legacy-")
        ]
        if len(candidates) == 1:
            target = candidates[0]
            for source in legacy["sources"]:
                _append_source(target, source)
            merged.remove(legacy)

    # Producer logs uuid4().hex[:12] at second resolution. Join that prefix only
    # to one complete manifest/event identity in the same scope.
    for partial in list(merged):
        partial_id = str(partial["run_id"])
        if not _has_bounded_runtime_log_identity(partial):
            continue
        candidates = [
            item
            for item in merged
            if item is not partial
            and item["exchange"] == partial["exchange"]
            and item["user"] == partial["user"]
            and abs(item["started_at_ms"] - partial["started_at_ms"])
            <= MAX_RUNTIME_LOG_START_SKEW_MS
            and str(item["run_id"]).startswith(partial_id)
            and len(str(item["run_id"])) > len(partial_id)
            and _is_full_manifest_or_event_identity(item)
        ]
        if len(candidates) != 1:
            continue
        target = candidates[0]
        for key, value in partial["identity"].items():
            if key not in target["identity"] or target["identity"][key] in (None, "", "unknown"):
                target["identity"][key] = value
        for source in partial["sources"]:
            _append_source(target, source)
        merged.remove(partial)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in merged:
        groups.setdefault((record["exchange"], record["user"]), []).append(record)
    for group in groups.values():
        group.sort(key=lambda item: (item["started_at_ms"], item["run_id"]))
        starts = sorted({item["started_at_ms"] for item in group})
        for record in group:
            next_starts = [start for start in starts if start > record["started_at_ms"]]
            record["ended_before_ms"] = min(next_starts) if next_starts else None
            identity = record["identity"]
            known = sum(
                1
                for key in _RUNTIME_KEYS
                if key in identity and identity[key] not in (None, "", "unknown")
            )
            if known <= 2:
                record["identity_status"] = "legacy_start_only"
            elif known == len(_RUNTIME_KEYS):
                record["identity_status"] = "full"
            else:
                record["identity_status"] = "partial"
            record["sources"] = sorted(
                record["sources"], key=lambda item: (item["kind"], item["path"])
            )
    return sorted(
        merged,
        key=lambda item: (item["exchange"], item["user"], item["started_at_ms"], item["run_id"]),
    )


def _merge_fills(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record["exchange"],
            record["user"],
            record["id"],
            record["timestamp"],
            record["symbol"],
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        if record["provenance"] and not existing["provenance"]:
            existing["provenance"] = record["provenance"]
        for field in ("position_side", "side", "pb_order_type", "client_order_id"):
            if not existing[field] and record[field]:
                existing[field] = record[field]
        existing["source_ids"] = sorted(set(existing["source_ids"]) | set(record["source_ids"]))
        for source in record["sources"]:
            _append_source(existing, source)
    return sorted(
        merged.values(),
        key=lambda item: (item["timestamp"], item["exchange"], item["user"], item["id"]),
    )


def _matches(value: str, accepted: set[str]) -> bool:
    return not accepted or str(value).lower() in accepted


def _attribute_fill(fill: dict[str, Any], runtimes: Sequence[dict[str, Any]]) -> None:
    provenance = fill.pop("provenance", {})
    provenance_runtime = provenance.get("runtime") if isinstance(provenance, Mapping) else None
    provenance_runtime = provenance_runtime if isinstance(provenance_runtime, Mapping) else {}
    provenance_run_id = str(provenance_runtime.get("run_id") or "")
    if (
        isinstance(provenance, Mapping)
        and provenance.get("attribution") == "first_ingested_by_runtime"
        and provenance_run_id
    ):
        fill["first_ingestion"] = {
            "status": "recorded",
            "run_id": provenance_run_id,
            "first_ingested_at_ms": provenance.get("first_ingested_at_ms"),
            "runtime_identity": _normalize_identity(provenance_runtime),
        }
    else:
        fill["first_ingestion"] = {
            "status": "unattributed",
            "reason": "legacy_or_missing_fill_provenance",
        }

    candidates = []
    for runtime in runtimes:
        if runtime["exchange"] != fill["exchange"] or runtime["user"] != fill["user"]:
            continue
        if fill["timestamp"] < runtime["started_at_ms"]:
            continue
        ended_before_ms = runtime.get("ended_before_ms")
        if ended_before_ms is not None and fill["timestamp"] >= ended_before_ms:
            continue
        candidates.append(runtime["run_id"])
    if len(candidates) == 1:
        status = "single_runtime_window_candidate"
        reason = "fill_timestamp_within_one_observed_runtime_window"
    elif candidates:
        status = "ambiguous_runtime_window"
        reason = "overlapping_runtime_observations"
    else:
        status = "unattributed"
        reason = "no_observed_runtime_window_at_fill_timestamp"
    fill["producer_attribution"] = {
        "status": status,
        "candidate_run_ids": sorted(candidates),
        "reason": reason,
        "proven": False,
        "caveat": (
            "Runtime-window correlation does not prove which binary submitted the order; "
            "use client-order IDs and contemporaneous order/execution logs when available."
        ),
    }
    fill["is_trailing"] = "trailing" in fill["pb_order_type"].lower()
    fill["sources"] = sorted(fill["sources"], key=lambda item: (item["kind"], item["path"]))


def build_runtime_attribution_report(
    *,
    fill_roots: Sequence[str | Path] = ("caches/fill_events",),
    runtime_roots: Sequence[str | Path] = ("caches/runtime",),
    monitor_roots: Sequence[str | Path] = ("monitor",),
    log_roots: Sequence[str | Path] = ("logs",),
    exchanges: Sequence[str] = (),
    users: Sequence[str] = (),
    symbols: Sequence[str] = (),
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
    trailing_only: bool = False,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_fills: int = DEFAULT_MAX_FILLS,
) -> dict[str, Any]:
    if max_files <= 0 or max_bytes_per_file <= 0 or max_total_bytes <= 0 or max_fills <= 0:
        raise ValueError("scan limits must be positive")
    if since_ms is not None and until_ms is not None and since_ms > until_ms:
        raise ValueError("since_ms must be less than or equal to until_ms")
    accepted_exchanges = {str(item).lower() for item in exchanges if str(item)}
    accepted_users = {str(item).lower() for item in users if str(item)}
    accepted_symbols = {str(item).lower() for item in symbols if str(item)}
    budget = ScanBudget(
        max_files=max_files,
        max_bytes_per_file=max_bytes_per_file,
        max_total_bytes=max_total_bytes,
        max_fills=max_fills,
    )
    fill_paths = [Path(item) for item in fill_roots]
    runtime_paths = [Path(item) for item in runtime_roots]
    monitor_paths = [Path(item) for item in monitor_roots]
    log_paths = [Path(item) for item in log_roots]

    runtimes = _collect_runtime_manifests(
        runtime_paths,
        budget,
        accepted_exchanges,
        accepted_users,
    )
    monitor_runtimes, monitor_fills = _collect_monitor(
        monitor_paths,
        budget,
        accepted_exchanges,
        accepted_users,
    )
    runtimes.extend(monitor_runtimes)
    runtimes.extend(_collect_logs(log_paths, budget))
    runtimes = _merge_runtimes(runtimes)

    fills = _merge_fills(
        _collect_fill_cache(
            fill_paths,
            budget,
            accepted_exchanges,
            accepted_users,
        )
        + monitor_fills
    )
    filtered_fills: list[dict[str, Any]] = []
    for fill in fills:
        if not _matches(fill["exchange"], accepted_exchanges):
            continue
        if not _matches(fill["user"], accepted_users):
            continue
        if not _matches(fill["symbol"], accepted_symbols):
            continue
        if since_ms is not None and fill["timestamp"] < since_ms:
            continue
        if until_ms is not None and fill["timestamp"] > until_ms:
            continue
        _attribute_fill(fill, runtimes)
        if trailing_only and not fill["is_trailing"]:
            continue
        filtered_fills.append(fill)

    filtered_runtimes = [
        runtime
        for runtime in runtimes
        if _matches(runtime["exchange"], accepted_exchanges)
        and _matches(runtime["user"], accepted_users)
    ]
    first_ingestion_counts: dict[str, int] = {}
    producer_counts: dict[str, int] = {}
    trailing_count = 0
    for fill in filtered_fills:
        first_status = fill["first_ingestion"]["status"]
        producer_status = fill["producer_attribution"]["status"]
        first_ingestion_counts[first_status] = first_ingestion_counts.get(first_status, 0) + 1
        producer_counts[producer_status] = producer_counts.get(producer_status, 0) + 1
        trailing_count += int(fill["is_trailing"])

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "first_ingestion": (
                "Recorded provenance identifies the runtime that first persisted a fill locally."
            ),
            "producer_attribution": (
                "Runtime windows are candidates only and never prove which binary "
                "submitted an order."
            ),
        },
        "inputs": {
            "fill_roots": [str(item) for item in fill_paths],
            "runtime_roots": [str(item) for item in runtime_paths],
            "monitor_roots": [str(item) for item in monitor_paths],
            "log_roots": [str(item) for item in log_paths],
            "filters": {
                "exchanges": sorted(accepted_exchanges),
                "users": sorted(accepted_users),
                "symbols": sorted(accepted_symbols),
                "since_ms": since_ms,
                "until_ms": until_ms,
                "trailing_only": bool(trailing_only),
            },
            "limits": {
                "max_files": max_files,
                "max_fills": max_fills,
                "max_bytes_per_file": max_bytes_per_file,
                "max_total_bytes": max_total_bytes,
            },
        },
        "scan": {
            "files_scanned": budget.files_scanned,
            "bytes_scanned": budget.bytes_scanned,
            "fills_scanned": budget.fills_scanned,
            "warnings": budget.warnings,
            "warnings_suppressed": budget.warnings_suppressed,
        },
        "summary": {
            "runtime_count": len(filtered_runtimes),
            "fill_count": len(filtered_fills),
            "trailing_fill_count": trailing_count,
            "first_ingestion_status_counts": first_ingestion_counts,
            "producer_attribution_status_counts": producer_counts,
        },
        "runtimes": filtered_runtimes,
        "fills": filtered_fills,
    }
