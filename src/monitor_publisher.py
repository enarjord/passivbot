from __future__ import annotations

import gzip
import hashlib
import errno
import json
import logging
import os
import re
import stat
import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


_MONITOR_EVENT_PHASE_TIMING_KEYS = (
    "lock_wait_ns",
    "rotation_ns",
    "persist_ns",
    "maintenance_ns",
    "manifest_checkpoint_count",
    "manifest_checkpoint_ns_total",
    "manifest_checkpoint_ns_max",
    "retention_run_count",
    "retention_ns_total",
    "retention_ns_max",
    "retention_thread_cpu_ns_total",
    "retention_thread_cpu_ns_max",
    "retention_non_cpu_ns_total",
    "retention_non_cpu_ns_max",
    "retention_inventory_ns_total",
    "retention_inventory_ns_max",
    "retention_age_filter_ns_total",
    "retention_age_filter_ns_max",
    "retention_cap_prune_ns_total",
    "retention_cap_prune_ns_max",
    "retention_age_unlink_ns_total",
    "retention_age_unlink_ns_max",
    "retention_cap_unlink_ns_total",
    "retention_cap_unlink_ns_max",
    "retention_inventory_entries_visited",
    "retention_inventory_candidates",
    "retention_age_deleted",
    "retention_cap_deleted",
)
_RETENTION_TIMING_KEYS = tuple(
    key for key in _MONITOR_EVENT_PHASE_TIMING_KEYS if key.startswith("retention_")
)
_CURRENT_EVENTS_REVERSE_READ_BYTES = 64 * 1024
_EVENT_RECOVERY_CHECKSUM_BYTES = 16
_EVENT_RECOVERY_MARKER = b',"_recovery":{"checksum":"'
_EVENT_RECOVERY_SEQ_SEPARATOR = b'","seq":'
_EVENT_RECOVERY_TRAILER_MAX_BYTES = 160
_MONITOR_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_MONITOR_ERROR_CONTEXT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,159}")
_MONITOR_ERROR_SENSITIVE_RE = re.compile(
    r"(?:api[_-]?key|secret|token|signature|authorization|auth|password|cookie|"
    r"credential|bearer|basic|sk[_-]?live)",
    re.IGNORECASE,
)
_MONITOR_ERROR_CONTEXT_MAX_INT = (1 << 63) - 1
_MONITOR_ERROR_CONTEXT_TOKEN_KEYS = (
    "source",
    "stage",
    "operation",
    "action",
    "status",
    "code",
    "cycle_id",
)
_MONITOR_ERROR_CONTEXT_COUNT_KEYS = (
    "attempt",
    "count",
)


def _empty_monitor_event_phase_timing() -> dict[str, int]:
    return {key: 0 for key in _MONITOR_EVENT_PHASE_TIMING_KEYS}


def _safe_monitor_error_context(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    context: dict[str, Any] = {}
    for key in _MONITOR_ERROR_CONTEXT_TOKEN_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        if _MONITOR_ERROR_SENSITIVE_RE.search(value):
            continue
        if _MONITOR_ERROR_CONTEXT_RE.fullmatch(value):
            context[key] = value
    for key in _MONITOR_ERROR_CONTEXT_COUNT_KEYS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 0 <= value <= _MONITOR_ERROR_CONTEXT_MAX_INT:
            context[key] = value
    return context


def _merge_retention_timing(target: dict[str, int], source: dict[str, int]) -> None:
    for key in _RETENTION_TIMING_KEYS:
        value = max(0, int(source.get(key, 0)))
        if key.endswith("_max"):
            target[key] = max(int(target.get(key, 0)), value)
        else:
            target[key] = int(target.get(key, 0)) + value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                separators=(",", ":"),
                sort_keys=True,
                default=_json_default,
            )
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except FileNotFoundError:
            pass


def _is_disk_full_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return True
    return "No space left on device" in str(exc)


class MonitorPublisher:
    schema_version = 1

    def __init__(
        self,
        *,
        exchange: str,
        user: str,
        root_dir: str,
        snapshot_interval_seconds: float,
        checkpoint_interval_minutes: float,
        event_rotation_mb: float,
        event_rotation_minutes: float,
        retain_days: float,
        max_total_bytes: int,
        compress_rotated_segments: bool,
        retain_price_ticks: bool,
        retain_candles: bool,
        retain_fills: bool,
        price_tick_min_interval_ms: int,
        emit_completed_candles: bool,
        include_raw_fill_payloads: bool,
        runtime_identity: Optional[dict] = None,
    ):
        self.exchange = str(exchange)
        self.user = str(user)
        self.base_root = Path(root_dir).expanduser()
        self.root = self.base_root / self.exchange / self.user
        self.events_dir = self.root / "events"
        self.history_dir = self.root / "history"
        self.checkpoints_dir = self.root / "checkpoints"
        self.manifest_path = self.root / "manifest.json"
        self.state_latest_path = self.root / "state.latest.json"
        self.current_events_path = self.events_dir / "current.ndjson"
        self.snapshot_interval_ms = max(1, int(float(snapshot_interval_seconds) * 1000.0))
        self.checkpoint_interval_ms = max(0, int(float(checkpoint_interval_minutes) * 60_000.0))
        self.event_rotation_bytes = max(1, int(float(event_rotation_mb) * 1024.0 * 1024.0))
        self.event_rotation_interval_ms = max(1, int(float(event_rotation_minutes) * 60_000.0))
        self.retain_days = float(retain_days)
        self.max_total_bytes = int(max_total_bytes)
        self.compress_rotated_segments = bool(compress_rotated_segments)
        self.retain_price_ticks = bool(retain_price_ticks)
        self.retain_candles = bool(retain_candles)
        self.retain_fills = bool(retain_fills)
        self.price_tick_min_interval_ms = max(0, int(price_tick_min_interval_ms))
        self.emit_completed_candles = bool(emit_completed_candles)
        self.include_raw_fill_payloads = bool(include_raw_fill_payloads)
        self.runtime_identity = deepcopy(runtime_identity) if runtime_identity else {}
        self.pid = os.getpid()
        self.created_ts_ms = self._now_ms()
        self.last_snapshot_ms = 0
        self.last_checkpoint_ms = 0
        self.last_retention_ms = 0
        self.current_segment_started_ms = self.created_ts_ms
        self.history_segment_started_ms: dict[str, int] = {}
        self._current_history_paths: dict[str, Path] = {}
        self._last_price_tick_emitted_ms: dict[str, int] = {}
        self._last_candle_ts_by_key: dict[tuple[str, str], int] = {}
        self._disk_full_last_log_ms = 0
        self._disk_full_suppressed = 0
        self._lock = threading.RLock()
        self._pending_retention_timing = {
            key: 0 for key in _RETENTION_TIMING_KEYS
        }
        self._thread_cpu_clock_warning_logged = False
        self.seq = 0
        self._manifest_dirty = False
        self._manifest_retry_needed = False
        self._last_manifest_write_monotonic_ms: Optional[int] = None
        self._ensure_layout()
        self._load_manifest_state()
        self._recover_seq_from_current_events()
        self._write_manifest()

    @classmethod
    def from_config(
        cls,
        *,
        exchange: str,
        user: str,
        config: dict,
        runtime_identity: Optional[dict] = None,
    ) -> "MonitorPublisher":
        return cls(
            exchange=exchange,
            user=user,
            root_dir=str(config["root_dir"]),
            snapshot_interval_seconds=float(config["snapshot_interval_seconds"]),
            checkpoint_interval_minutes=float(config["checkpoint_interval_minutes"]),
            event_rotation_mb=float(config["event_rotation_mb"]),
            event_rotation_minutes=float(config["event_rotation_minutes"]),
            retain_days=float(config["retain_days"]),
            max_total_bytes=int(config["max_total_bytes"]),
            compress_rotated_segments=bool(config["compress_rotated_segments"]),
            retain_price_ticks=bool(config["retain_price_ticks"]),
            retain_candles=bool(config["retain_candles"]),
            retain_fills=bool(config["retain_fills"]),
            price_tick_min_interval_ms=int(config["price_tick_min_interval_ms"]),
            emit_completed_candles=bool(config["emit_completed_candles"]),
            include_raw_fill_payloads=bool(config["include_raw_fill_payloads"]),
            runtime_identity=runtime_identity,
        )

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000)

    def _ensure_layout(self) -> None:
        for path in (self.root, self.events_dir, self.history_dir, self.checkpoints_dir):
            path.mkdir(parents=True, exist_ok=True)
        if not self.current_events_path.exists():
            self.current_events_path.touch()

    def _load_manifest_state(self) -> None:
        if not self.manifest_path.exists():
            return
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("[monitor] unable to load manifest %s: %s", self.manifest_path, exc)
            return
        try:
            self.seq = max(0, int(data.get("last_seq", 0) or 0))
        except Exception:
            self.seq = 0
        try:
            self.created_ts_ms = int(data.get("created_ts_ms", self.created_ts_ms) or self.created_ts_ms)
        except Exception:
            pass
        try:
            self.current_segment_started_ms = int(
                data.get("current_segment_started_ms", self.current_segment_started_ms)
                or self.current_segment_started_ms
            )
        except Exception:
            pass
        try:
            raw_history_started = data.get("history_current_segment_started_ms", {})
            if isinstance(raw_history_started, dict):
                self.history_segment_started_ms = {
                    str(stream): int(started_ms)
                    for stream, started_ms in raw_history_started.items()
                    if int(started_ms) > 0
                }
        except Exception:
            self.history_segment_started_ms = {}

    def _recover_seq_from_current_events(self) -> None:
        recovered_seq = self._max_recoverable_event_seq_in_current_segment()
        if recovered_seq is not None:
            self.seq = max(self.seq, recovered_seq)

    def _max_recoverable_event_seq_in_current_segment(self) -> Optional[int]:
        """Return the maximum recoverable seq without buffering whole rows."""

        def iter_line_ranges_reverse(f, file_size: int):
            line_end = file_size
            scan_end = file_size
            while scan_end > 0:
                read_size = min(_CURRENT_EVENTS_REVERSE_READ_BYTES, scan_end)
                scan_start = scan_end - read_size
                f.seek(scan_start)
                chunk = f.read(read_size)
                search_end = len(chunk)
                while True:
                    newline_idx = chunk.rfind(b"\n", 0, search_end)
                    if newline_idx < 0:
                        break
                    line_start = scan_start + newline_idx + 1
                    yield line_start, line_end
                    line_end = scan_start + newline_idx
                    search_end = newline_idx
                scan_end = scan_start
            yield 0, line_end

        def seq_from_line_range(f, line_start: int, line_end: int) -> Optional[int]:
            if line_end > line_start:
                f.seek(line_end - 1)
                if f.read(1) == b"\r":
                    line_end -= 1
            suffix_size = min(_EVENT_RECOVERY_TRAILER_MAX_BYTES, line_end - line_start)
            if suffix_size <= 0:
                return None
            f.seek(line_end - suffix_size)
            suffix = f.read(suffix_size)
            marker_idx = suffix.rfind(_EVENT_RECOVERY_MARKER)
            if marker_idx < 0:
                return None
            marker_offset = line_end - suffix_size + marker_idx
            trailer = suffix[marker_idx + len(_EVENT_RECOVERY_MARKER) :]
            checksum, separator, seq_tail = trailer.partition(_EVENT_RECOVERY_SEQ_SEPARATOR)
            if separator != _EVENT_RECOVERY_SEQ_SEPARATOR:
                return None
            if len(checksum) != _EVENT_RECOVERY_CHECKSUM_BYTES * 2:
                return None
            if not seq_tail.endswith(b"}}"):
                return None
            seq_raw = seq_tail[:-2]
            if not seq_raw or len(seq_raw) > 32 or not seq_raw.isdigit():
                return None

            digest = hashlib.blake2b(digest_size=_EVENT_RECOVERY_CHECKSUM_BYTES)
            scan_offset = line_start
            while scan_offset < marker_offset:
                f.seek(scan_offset)
                chunk = f.read(
                    min(_CURRENT_EVENTS_REVERSE_READ_BYTES, marker_offset - scan_offset)
                )
                if not chunk:
                    return None
                digest.update(chunk)
                scan_offset += len(chunk)
            digest.update(b"}")
            digest.update(b"\0seq:")
            digest.update(seq_raw)
            if digest.hexdigest().encode("ascii") != checksum:
                return None
            return int(seq_raw)

        try:
            with open(self.current_events_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                max_seq = None
                for line_start, line_end in iter_line_ranges_reverse(f, file_size):
                    seq = seq_from_line_range(f, line_start, line_end)
                    if seq is not None and (max_seq is None or seq > max_seq):
                        max_seq = seq
                return max_seq
        except OSError as exc:
            logging.warning(
                "[monitor] unable to read current event segment %s: %s",
                self.current_events_path,
                exc,
            )
            return None

    def _serialize_event_line(self, envelope: dict) -> str:
        base_line = json.dumps(
            envelope,
            separators=(",", ":"),
            sort_keys=True,
            default=_json_default,
        )
        seq_text = str(int(envelope["seq"]))
        digest = hashlib.blake2b(digest_size=_EVENT_RECOVERY_CHECKSUM_BYTES)
        digest.update(base_line.encode("utf-8"))
        digest.update(b"\0seq:")
        digest.update(seq_text.encode("ascii"))
        checksum = digest.hexdigest()
        trailer = (
            ',"_recovery":{"checksum":"'
            + checksum
            + '","seq":'
            + seq_text
            + "}}"
        )
        return base_line[:-1] + trailer

    def _build_manifest(self, now_ms: Optional[int] = None) -> dict:
        now_ms = self._now_ms() if now_ms is None else int(now_ms)
        history_streams = {
            "fills": self.retain_fills,
            "price_ticks": self.retain_price_ticks,
            "candles_1m": self.retain_candles and self.emit_completed_candles,
            "candles_1h": self.retain_candles and self.emit_completed_candles,
        }
        return {
            "schema_version": self.schema_version,
            "exchange": self.exchange,
            "user": self.user,
            "pid": self.pid,
            "created_ts_ms": self.created_ts_ms,
            "updated_ts_ms": now_ms,
            "last_seq": self.seq,
            "current_segment_started_ms": self.current_segment_started_ms,
            "runtime": deepcopy(self.runtime_identity),
            "history_current_segment_started_ms": dict(self.history_segment_started_ms),
            "paths": {
                "root": str(self.root),
                "state_latest": str(self.state_latest_path),
                "events_current": str(self.current_events_path),
                "history_dir": str(self.history_dir),
                "history_current": {
                    stream: str(self._history_current_path(stream)) for stream in sorted(history_streams)
                },
                "checkpoints_dir": str(self.checkpoints_dir),
            },
            "config": {
                "snapshot_interval_seconds": self.snapshot_interval_ms / 1000.0,
                "checkpoint_interval_minutes": self.checkpoint_interval_ms / 60_000.0,
                "event_rotation_mb": self.event_rotation_bytes / (1024.0 * 1024.0),
                "event_rotation_minutes": self.event_rotation_interval_ms / 60_000.0,
                "retain_days": self.retain_days,
                "max_total_bytes": self.max_total_bytes,
                "compress_rotated_segments": self.compress_rotated_segments,
                "retain_price_ticks": self.retain_price_ticks,
                "retain_candles": self.retain_candles,
                "retain_fills": self.retain_fills,
                "price_tick_min_interval_ms": self.price_tick_min_interval_ms,
                "emit_completed_candles": self.emit_completed_candles,
                "include_raw_fill_payloads": self.include_raw_fill_payloads,
            },
            "capabilities": {
                "snapshot": True,
                "events": True,
                "history": any(history_streams.values()),
                "history_streams": history_streams,
                "checkpoints": True,
            },
        }

    def _mark_manifest_dirty(self) -> None:
        self._manifest_dirty = True

    def _write_manifest(self, now_ms: Optional[int] = None) -> bool:
        with self._lock:
            try:
                _atomic_write_json(self.manifest_path, self._build_manifest(now_ms=now_ms))
            except Exception as exc:
                self._manifest_dirty = True
                self._manifest_retry_needed = True
                self._log_write_failure("writing manifest", exc)
                return False
            self._manifest_dirty = False
            self._manifest_retry_needed = False
            self._last_manifest_write_monotonic_ms = self._monotonic_ms()
            return True

    def _manifest_checkpoint_due(self) -> bool:
        if not self._manifest_dirty:
            return False
        cadence_now_ms = self._monotonic_ms()
        return (
            self._manifest_retry_needed
            or self._last_manifest_write_monotonic_ms is None
            or cadence_now_ms < self._last_manifest_write_monotonic_ms
            or cadence_now_ms - self._last_manifest_write_monotonic_ms
            >= self.snapshot_interval_ms
        )

    def _write_manifest_if_due(
        self,
        now_ms: Optional[int] = None,
        *,
        on_attempt: Optional[Callable[[], None]] = None,
    ) -> bool:
        with self._lock:
            if not self._manifest_checkpoint_due():
                return False
            if on_attempt is not None:
                on_attempt()
            return self._write_manifest(now_ms=now_ms)

    def _log_write_failure(self, action: str, exc: BaseException) -> None:
        if not _is_disk_full_error(exc):
            logging.error("[monitor] %s: %s", action, exc)
            return
        now_ms = self._now_ms()
        if now_ms - self._disk_full_last_log_ms >= 60_000:
            suffix = ""
            if self._disk_full_suppressed:
                suffix = f" | suppressed={self._disk_full_suppressed}"
            self._disk_full_suppressed = 0
            self._disk_full_last_log_ms = now_ms
            logging.error(
                "[monitor] disk full while %s: %s%s | suppressing repeat disk-full monitor errors for 60s",
                action,
                exc,
                suffix,
            )
        else:
            self._disk_full_suppressed += 1
            logging.debug("[monitor] disk full while %s: %s", action, exc)

    def _segment_label(self, now_ms: int) -> str:
        dt = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%S")

    def _gzip_file(self, path: Path) -> Path:
        gz_path = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as src:
            with gzip.open(gz_path, "wb") as dst:
                dst.write(src.read())
        path.unlink()
        return gz_path

    def _retention_inventory(
        self,
        *,
        on_entry: Optional[Callable[[], None]] = None,
        on_candidate: Optional[Callable[[], None]] = None,
    ) -> tuple[int, list[tuple[Path, int, float]]]:
        """Return total retained bytes and direct rotated-file deletion candidates."""

        protected = {
            self.manifest_path,
            self.state_latest_path,
            self.current_events_path,
            *self._current_history_paths.values(),
        }
        candidate_dirs = {self.events_dir, self.history_dir, self.checkpoints_dir}
        total_bytes = 0
        candidates: list[tuple[Path, int, float]] = []
        pending_dirs = deque([self.root])
        while pending_dirs:
            directory = pending_dirs.popleft()
            entries = None
            for _attempt in range(2):
                try:
                    entries = os.scandir(directory)
                    break
                except OSError:
                    continue
            if entries is None:
                continue
            child_dirs: list[Path] = []
            with entries:
                for entry in entries:
                    path = Path(entry.path)
                    if on_entry is not None:
                        on_entry()
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            child_dirs.append(path)
                    except OSError:
                        pass
                    try:
                        file_stat = entry.stat()
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(file_stat.st_mode):
                        continue
                    total_bytes += file_stat.st_size
                    if path.parent in candidate_dirs and path not in protected:
                        candidates.append((path, file_stat.st_size, file_stat.st_mtime))
                        if on_candidate is not None:
                            on_candidate()
            pending_dirs.extend(child_dirs)
        return total_bytes, sorted(candidates, key=lambda candidate: candidate[2])

    def _retention_due(self, now_ms: int) -> bool:
        return not (
            self.last_retention_ms and now_ms - self.last_retention_ms < 60_000
        )

    def _thread_cpu_time_ns(self) -> int | None:
        try:
            return time.thread_time_ns()
        except Exception as exc:
            # CPU attribution is diagnostic-only and must not block persistence.
            if not self._thread_cpu_clock_warning_logged:
                logging.warning(
                    "[monitor] thread CPU clock unavailable; retention CPU attribution disabled: %s",
                    exc,
                )
                self._thread_cpu_clock_warning_logged = True
            return None

    def _record_pending_retention_timing(self, now_ms: Optional[int]) -> None:
        timing = {key: 0 for key in _RETENTION_TIMING_KEYS}
        retention_started_ns: int | None = None
        retention_thread_cpu_started_ns: int | None = None

        def on_run() -> None:
            nonlocal retention_started_ns, retention_thread_cpu_started_ns
            retention_started_ns = time.monotonic_ns()
            retention_thread_cpu_started_ns = self._thread_cpu_time_ns()
            timing["retention_run_count"] += 1

        def on_phase(phase: str, duration_ns: int) -> None:
            total_key = f"retention_{phase}_ns_total"
            max_key = f"retention_{phase}_ns_max"
            duration_ns = max(0, int(duration_ns))
            timing[total_key] += duration_ns
            timing[max_key] = max(timing[max_key], duration_ns)

        def on_counter(counter: str) -> None:
            timing[f"retention_{counter}"] += 1

        try:
            self._prune_retention(
                now_ms=now_ms,
                on_run=on_run,
                on_phase=on_phase,
                on_counter=on_counter,
            )
        finally:
            if retention_started_ns is not None:
                retention_thread_cpu_finished_ns = (
                    self._thread_cpu_time_ns()
                    if retention_thread_cpu_started_ns is not None
                    else None
                )
                retention_ns = max(
                    0, time.monotonic_ns() - retention_started_ns
                )
                timing["retention_ns_total"] += retention_ns
                timing["retention_ns_max"] = max(
                    timing["retention_ns_max"], retention_ns
                )
                if retention_thread_cpu_finished_ns is not None:
                    retention_thread_cpu_ns = max(
                        0,
                        retention_thread_cpu_finished_ns
                        - int(retention_thread_cpu_started_ns),
                    )
                    retention_non_cpu_ns = max(
                        0, retention_ns - retention_thread_cpu_ns
                    )
                    timing["retention_thread_cpu_ns_total"] += (
                        retention_thread_cpu_ns
                    )
                    timing["retention_thread_cpu_ns_max"] = max(
                        timing["retention_thread_cpu_ns_max"],
                        retention_thread_cpu_ns,
                    )
                    timing["retention_non_cpu_ns_total"] += retention_non_cpu_ns
                    timing["retention_non_cpu_ns_max"] = max(
                        timing["retention_non_cpu_ns_max"],
                        retention_non_cpu_ns,
                    )
                with self._lock:
                    _merge_retention_timing(self._pending_retention_timing, timing)

    def _consume_pending_retention_timing(
        self, timing: dict[str, int]
    ) -> None:
        with self._lock:
            _merge_retention_timing(timing, self._pending_retention_timing)
            self._pending_retention_timing = {
                key: 0 for key in _RETENTION_TIMING_KEYS
            }

    def _prune_retention(
        self,
        now_ms: Optional[int] = None,
        *,
        on_run: Optional[Callable[[], None]] = None,
        on_phase: Optional[Callable[[str, int], None]] = None,
        on_counter: Optional[Callable[[str], None]] = None,
    ) -> None:
        if on_run is None and on_phase is None and on_counter is None:
            self._record_pending_retention_timing(now_ms)
            return
        with self._lock:
            now_ms = self._now_ms() if now_ms is None else int(now_ms)
            if not self._retention_due(now_ms):
                return
            self.last_retention_ms = now_ms
            if on_run is not None:
                on_run()
            try:
                cutoff_ms = now_ms - int(
                    self.retain_days * 24.0 * 60.0 * 60.0 * 1000.0
                )
                inventory_started_ns = (
                    time.monotonic_ns() if on_phase is not None else None
                )
                try:
                    if on_counter is None:
                        total_bytes, candidates = self._retention_inventory()
                    else:
                        total_bytes, candidates = self._retention_inventory(
                            on_entry=lambda: on_counter("inventory_entries_visited"),
                            on_candidate=lambda: on_counter("inventory_candidates"),
                        )
                finally:
                    if inventory_started_ns is not None:
                        on_phase(
                            "inventory",
                            max(0, time.monotonic_ns() - inventory_started_ns),
                        )
                age_filter_started_ns = (
                    time.monotonic_ns() if on_phase is not None else None
                )
                try:
                    if self.retain_days >= 0.0:
                        survivors = []
                        for path, size, mtime in candidates:
                            if int(mtime * 1000.0) < cutoff_ms:
                                age_unlink_started_ns = (
                                    time.monotonic_ns() if on_phase is not None else None
                                )
                                try:
                                    path.unlink()
                                except FileNotFoundError:
                                    total_bytes -= size
                                    continue
                                finally:
                                    if age_unlink_started_ns is not None:
                                        on_phase(
                                            "age_unlink",
                                            max(0, time.monotonic_ns() - age_unlink_started_ns),
                                        )
                                total_bytes -= size
                                if on_counter is not None:
                                    on_counter("age_deleted")
                            else:
                                survivors.append((path, size, mtime))
                    else:
                        survivors = candidates
                finally:
                    if age_filter_started_ns is not None:
                        on_phase(
                            "age_filter",
                            max(0, time.monotonic_ns() - age_filter_started_ns),
                        )

                cap_prune_started_ns = (
                    time.monotonic_ns() if on_phase is not None else None
                )
                try:
                    if total_bytes <= self.max_total_bytes:
                        return

                    for path, size, _mtime in survivors:
                        if total_bytes <= self.max_total_bytes:
                            break
                        cap_unlink_started_ns = (
                            time.monotonic_ns() if on_phase is not None else None
                        )
                        try:
                            path.unlink()
                        finally:
                            if cap_unlink_started_ns is not None:
                                on_phase(
                                    "cap_unlink",
                                    max(0, time.monotonic_ns() - cap_unlink_started_ns),
                                )
                        total_bytes -= size
                        if on_counter is not None:
                            on_counter("cap_deleted")
                finally:
                    if cap_prune_started_ns is not None:
                        on_phase(
                            "cap_prune",
                            max(0, time.monotonic_ns() - cap_prune_started_ns),
                        )
            except Exception as exc:
                logging.error("[monitor] retention pruning failed: %s", exc)

    def _rotated_path(self, directory: Path, stem: str, suffix: str) -> Path:
        path = directory / f"{stem}{suffix}"
        if not path.exists() and not path.with_suffix(path.suffix + ".gz").exists():
            return path
        for _ in range(1000):
            candidate = directory / f"{stem}.{uuid.uuid4().hex[:8]}{suffix}"
            if not candidate.exists() and not candidate.with_suffix(
                candidate.suffix + ".gz"
            ).exists():
                return candidate
        return directory / f"{stem}.{uuid.uuid4().hex}{suffix}"

    def _rotate_events_if_needed(self, now_ms: Optional[int] = None) -> None:
        with self._lock:
            now_ms = self._now_ms() if now_ms is None else int(now_ms)
            try:
                if not self.current_events_path.exists():
                    self.current_events_path.touch()
                    self.current_segment_started_ms = now_ms
                    return
                size_bytes = self.current_events_path.stat().st_size
                age_ms = now_ms - self.current_segment_started_ms
                if size_bytes <= 0:
                    return
                if size_bytes < self.event_rotation_bytes and age_ms < self.event_rotation_interval_ms:
                    return
                rotated = self._rotated_path(
                    self.events_dir, self._segment_label(now_ms), ".ndjson"
                )
                # Preserve the current sequence before the active segment moves.
                if not self._write_manifest(now_ms=now_ms):
                    return
                os.replace(self.current_events_path, rotated)
                if self.compress_rotated_segments:
                    rotated = self._gzip_file(rotated)
                self.current_events_path.touch()
                self.current_segment_started_ms = now_ms
                self._mark_manifest_dirty()
                self._write_manifest(now_ms=now_ms)
                self._prune_retention(now_ms=now_ms)
            except Exception as exc:
                logging.error("[monitor] event rotation failed: %s", exc)

    def _history_current_path(self, stream: str) -> Path:
        path = self._current_history_paths.get(stream)
        if path is None:
            path = self.history_dir / f"{stream}.current.ndjson"
            self._current_history_paths[stream] = path
        return path

    def _rotate_history_if_needed(self, stream: str, *, now_ms: int) -> None:
        with self._lock:
            try:
                current_path = self._history_current_path(stream)
                started_ms = int(self.history_segment_started_ms.get(stream, self.created_ts_ms))
                if not current_path.exists():
                    current_path.touch()
                    self.history_segment_started_ms[stream] = now_ms
                    return
                size_bytes = current_path.stat().st_size
                age_ms = now_ms - started_ms
                if size_bytes <= 0:
                    return
                if size_bytes < self.event_rotation_bytes and age_ms < self.event_rotation_interval_ms:
                    return
                rotated = self._rotated_path(
                    self.history_dir, f"{stream}.{self._segment_label(now_ms)}", ".ndjson"
                )
                os.replace(current_path, rotated)
                if self.compress_rotated_segments:
                    rotated = self._gzip_file(rotated)
                current_path.touch()
                self.history_segment_started_ms[stream] = now_ms
                self._mark_manifest_dirty()
                self._write_manifest(now_ms=now_ms)
                self._prune_retention(now_ms=now_ms)
            except Exception as exc:
                logging.error("[monitor] history rotation failed for %s: %s", stream, exc)

    def record_history_entry(
        self,
        stream: str,
        kind: str,
        payload: dict,
        *,
        ts: Optional[int] = None,
        symbol: Optional[str] = None,
        pside: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Optional[dict]:
        with self._lock:
            now_ms = self._now_ms() if ts is None else int(ts)
            try:
                self._rotate_history_if_needed(stream, now_ms=now_ms)
                envelope = {
                    "ts": now_ms,
                    "kind": str(kind),
                    "stream": str(stream),
                    "exchange": self.exchange,
                    "user": self.user,
                    "payload": payload or {},
                }
                if symbol is not None:
                    envelope["symbol"] = str(symbol)
                if pside is not None:
                    envelope["pside"] = str(pside)
                if timeframe is not None:
                    envelope["timeframe"] = str(timeframe)
                current_path = self._history_current_path(stream)
                if not current_path.exists():
                    current_path.touch()
                    self.history_segment_started_ms[stream] = now_ms
                line = json.dumps(
                    envelope,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=_json_default,
                )
                with open(current_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                self._mark_manifest_dirty()
                self._write_manifest_if_due(now_ms=now_ms)
                self._prune_retention(now_ms=now_ms)
                return envelope
            except Exception as exc:
                self._log_write_failure(f"recording history entry {stream}/{kind}", exc)
                return None

    def record_fill(
        self,
        payload: dict,
        *,
        ts: Optional[int] = None,
        symbol: Optional[str] = None,
        pside: Optional[str] = None,
        raw_payload: Any = None,
    ) -> Optional[dict]:
        if not self.retain_fills:
            return None
        entry_payload = dict(payload or {})
        if self.include_raw_fill_payloads and raw_payload is not None:
            entry_payload["raw"] = raw_payload
        return self.record_history_entry(
            "fills",
            "fill",
            entry_payload,
            ts=ts,
            symbol=symbol,
            pside=pside,
        )

    def record_price_tick(
        self,
        symbol: str,
        last: float,
        *,
        ts: Optional[int] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        source: Optional[str] = None,
    ) -> Optional[dict]:
        with self._lock:
            if not self.retain_price_ticks:
                return None
            now_ms = self._now_ms() if ts is None else int(ts)
            symbol = str(symbol)
            last_emitted_ms = int(self._last_price_tick_emitted_ms.get(symbol, 0))
            if (
                self.price_tick_min_interval_ms > 0
                and last_emitted_ms
                and now_ms - last_emitted_ms < self.price_tick_min_interval_ms
            ):
                return None
            entry_payload = {"last": float(last)}
            if bid is not None:
                entry_payload["bid"] = float(bid)
            if ask is not None:
                entry_payload["ask"] = float(ask)
            if source is not None:
                entry_payload["source"] = str(source)
            entry = self.record_history_entry(
                "price_ticks",
                "price_tick",
                entry_payload,
                ts=now_ms,
                symbol=symbol,
            )
            if entry is not None:
                self._last_price_tick_emitted_ms[symbol] = now_ms
            return entry

    def record_completed_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: Iterable[dict],
    ) -> list[dict]:
        with self._lock:
            if not (self.retain_candles and self.emit_completed_candles):
                return []
            timeframe = str(timeframe)
            stream = f"candles_{timeframe}"
            candles_list = list(candles)
            if not candles_list:
                return []
            key = (str(symbol), timeframe)
            sorted_candles = sorted(candles_list, key=lambda candle: int(candle["ts"]))
            last_known_ts = self._last_candle_ts_by_key.get(key)
            if last_known_ts is None:
                to_emit = [sorted_candles[-1]]
            else:
                to_emit = [
                    candle for candle in sorted_candles if int(candle["ts"]) > last_known_ts
                ]
            if not to_emit:
                self._last_candle_ts_by_key[key] = int(sorted_candles[-1]["ts"])
                return []
            emitted: list[dict] = []
            for candle in to_emit:
                entry = self.record_history_entry(
                    stream,
                    "completed_candle",
                    dict(candle),
                    ts=int(candle["ts"]),
                    symbol=symbol,
                    timeframe=timeframe,
                )
                if entry is not None:
                    emitted.append(entry)
            self._last_candle_ts_by_key[key] = int(sorted_candles[-1]["ts"])
            return emitted

    def record_event(
        self,
        kind: str,
        tags: Iterable[str],
        payload: Optional[dict] = None,
        *,
        ts: Optional[int] = None,
        symbol: Optional[str] = None,
        pside: Optional[str] = None,
    ) -> Optional[dict]:
        result, _timing = self._record_event_timed(
            kind,
            tags,
            payload,
            ts=ts,
            symbol=symbol,
            pside=pside,
            consume_pending_retention_timing=False,
        )
        return result

    def _record_event_timed(
        self,
        kind: str,
        tags: Iterable[str],
        payload: Optional[dict] = None,
        *,
        ts: Optional[int] = None,
        symbol: Optional[str] = None,
        pside: Optional[str] = None,
        consume_pending_retention_timing: bool = True,
    ) -> tuple[Optional[dict], dict[str, int]]:
        timing = _empty_monitor_event_phase_timing()
        lock_wait_started_ns = time.monotonic_ns()
        with self._lock:
            timing["lock_wait_ns"] = max(0, time.monotonic_ns() - lock_wait_started_ns)
            now_ms = self._now_ms() if ts is None else int(ts)
            try:
                rotation_started_ns = time.monotonic_ns()
                try:
                    self._rotate_events_if_needed(now_ms=now_ms)
                finally:
                    timing["rotation_ns"] = max(
                        0, time.monotonic_ns() - rotation_started_ns
                    )

                persist_started_ns = time.monotonic_ns()
                try:
                    self.seq += 1
                    envelope = {
                        "ts": now_ms,
                        "seq": self.seq,
                        "kind": str(kind),
                        "tags": [str(tag) for tag in tags],
                        "exchange": self.exchange,
                        "user": self.user,
                        "payload": payload or {},
                    }
                    if symbol is not None:
                        envelope["symbol"] = str(symbol)
                    if pside is not None:
                        envelope["pside"] = str(pside)
                    line = self._serialize_event_line(envelope)
                    with open(self.current_events_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                finally:
                    timing["persist_ns"] = max(0, time.monotonic_ns() - persist_started_ns)

                maintenance_started_ns = time.monotonic_ns()
                try:
                    self._mark_manifest_dirty()
                    manifest_checkpoint_started_ns: int | None = None

                    def on_manifest_checkpoint_attempt() -> None:
                        nonlocal manifest_checkpoint_started_ns
                        manifest_checkpoint_started_ns = time.monotonic_ns()
                        timing["manifest_checkpoint_count"] += 1

                    try:
                        self._write_manifest_if_due(
                            now_ms=now_ms, on_attempt=on_manifest_checkpoint_attempt
                        )
                    finally:
                        if manifest_checkpoint_started_ns is not None:
                            manifest_checkpoint_ns = max(
                                0, time.monotonic_ns() - manifest_checkpoint_started_ns
                            )
                            timing["manifest_checkpoint_ns_total"] += manifest_checkpoint_ns
                            timing["manifest_checkpoint_ns_max"] = max(
                                timing["manifest_checkpoint_ns_max"], manifest_checkpoint_ns
                            )
                    retention_started_ns: int | None = None
                    retention_thread_cpu_started_ns: int | None = None

                    def on_retention_run() -> None:
                        nonlocal retention_started_ns, retention_thread_cpu_started_ns
                        retention_started_ns = time.monotonic_ns()
                        retention_thread_cpu_started_ns = self._thread_cpu_time_ns()
                        timing["retention_run_count"] += 1

                    def on_retention_phase(phase: str, duration_ns: int) -> None:
                        total_key = f"retention_{phase}_ns_total"
                        max_key = f"retention_{phase}_ns_max"
                        duration_ns = max(0, int(duration_ns))
                        timing[total_key] += duration_ns
                        timing[max_key] = max(timing[max_key], duration_ns)

                    def on_retention_counter(counter: str) -> None:
                        timing[f"retention_{counter}"] += 1

                    if consume_pending_retention_timing:
                        self._prune_retention(
                            now_ms=now_ms,
                            on_run=on_retention_run,
                            on_phase=on_retention_phase,
                            on_counter=on_retention_counter,
                        )
                    else:
                        self._prune_retention(now_ms=now_ms)
                    if consume_pending_retention_timing:
                        if retention_started_ns is not None:
                            retention_thread_cpu_finished_ns = (
                                self._thread_cpu_time_ns()
                                if retention_thread_cpu_started_ns is not None
                                else None
                            )
                            retention_ns = max(
                                0, time.monotonic_ns() - retention_started_ns
                            )
                            timing["retention_ns_total"] += retention_ns
                            timing["retention_ns_max"] = max(
                                timing["retention_ns_max"], retention_ns
                            )
                            if retention_thread_cpu_finished_ns is not None:
                                retention_thread_cpu_ns = max(
                                    0,
                                    retention_thread_cpu_finished_ns
                                    - int(retention_thread_cpu_started_ns),
                                )
                                retention_non_cpu_ns = max(
                                    0, retention_ns - retention_thread_cpu_ns
                                )
                                timing["retention_thread_cpu_ns_total"] += (
                                    retention_thread_cpu_ns
                                )
                                timing["retention_thread_cpu_ns_max"] = max(
                                    timing["retention_thread_cpu_ns_max"],
                                    retention_thread_cpu_ns,
                                )
                                timing["retention_non_cpu_ns_total"] += (
                                    retention_non_cpu_ns
                                )
                                timing["retention_non_cpu_ns_max"] = max(
                                    timing["retention_non_cpu_ns_max"],
                                    retention_non_cpu_ns,
                                )
                finally:
                    timing["maintenance_ns"] = max(
                        0, time.monotonic_ns() - maintenance_started_ns
                    )
                return envelope, timing
            except Exception as exc:
                self._log_write_failure(f"recording event {kind}", exc)
                return None, timing
            finally:
                if consume_pending_retention_timing:
                    self._consume_pending_retention_timing(timing)

    def record_error(
        self,
        kind: str,
        error: Exception,
        *,
        tags: Optional[Iterable[str]] = None,
        payload: Optional[dict] = None,
        ts: Optional[int] = None,
        symbol: Optional[str] = None,
        pside: Optional[str] = None,
    ) -> Optional[dict]:
        error_payload = _safe_monitor_error_context(payload)
        error_type = type(error).__name__
        error_payload["error_type"] = (
            error_type if _MONITOR_ERROR_TYPE_RE.fullmatch(error_type) else "unknown"
        )
        return self.record_event(
            kind,
            tags or ("error",),
            error_payload,
            ts=ts,
            symbol=symbol,
            pside=pside,
        )

    def write_snapshot(
        self, snapshot: dict, *, ts: Optional[int] = None, force: bool = False
    ) -> bool:
        with self._lock:
            now_ms = self._now_ms() if ts is None else int(ts)
            if not force and self.last_snapshot_ms and (
                now_ms - self.last_snapshot_ms < self.snapshot_interval_ms
            ):
                return False
            try:
                payload = dict(snapshot)
                payload["schema_version"] = self.schema_version
                meta = dict(payload.get("meta") or {})
                meta.setdefault("exchange", self.exchange)
                meta.setdefault("user", self.user)
                meta.setdefault("runtime", deepcopy(self.runtime_identity))
                meta["snapshot_ts_ms"] = now_ms
                meta["seq"] = self.seq
                payload["meta"] = meta
                _atomic_write_json(self.state_latest_path, payload)
                self.last_snapshot_ms = now_ms
                self._write_manifest(now_ms=now_ms)
                if self.checkpoint_interval_ms > 0 and (
                    force or now_ms - self.last_checkpoint_ms >= self.checkpoint_interval_ms
                ):
                    checkpoint = self._rotated_path(
                        self.checkpoints_dir,
                        f"state.{self._segment_label(now_ms)}",
                        ".json",
                    )
                    _atomic_write_json(checkpoint, payload)
                    if self.compress_rotated_segments:
                        self._gzip_file(checkpoint)
                    self.last_checkpoint_ms = now_ms
                self._prune_retention(now_ms=now_ms)
                return True
            except Exception as exc:
                self._log_write_failure("writing snapshot", exc)
                return False

    def close(self) -> None:
        with self._lock:
            self._write_manifest()
