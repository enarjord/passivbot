from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True, default=_json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


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
        include_raw_fill_payloads: bool,
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
        self.include_raw_fill_payloads = bool(include_raw_fill_payloads)
        self.pid = os.getpid()
        self.created_ts_ms = self._now_ms()
        self.last_snapshot_ms = 0
        self.last_checkpoint_ms = 0
        self.last_retention_ms = 0
        self.current_segment_started_ms = self.created_ts_ms
        self.seq = 0
        self._ensure_layout()
        self._load_manifest_state()
        self._write_manifest()

    @classmethod
    def from_config(cls, *, exchange: str, user: str, config: dict) -> "MonitorPublisher":
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
            include_raw_fill_payloads=bool(config["include_raw_fill_payloads"]),
        )

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

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

    def _build_manifest(self, now_ms: Optional[int] = None) -> dict:
        now_ms = self._now_ms() if now_ms is None else int(now_ms)
        return {
            "schema_version": self.schema_version,
            "exchange": self.exchange,
            "user": self.user,
            "pid": self.pid,
            "created_ts_ms": self.created_ts_ms,
            "updated_ts_ms": now_ms,
            "last_seq": self.seq,
            "current_segment_started_ms": self.current_segment_started_ms,
            "paths": {
                "root": str(self.root),
                "state_latest": str(self.state_latest_path),
                "events_current": str(self.current_events_path),
                "history_dir": str(self.history_dir),
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
                "include_raw_fill_payloads": self.include_raw_fill_payloads,
            },
            "capabilities": {
                "snapshot": True,
                "events": True,
                "history": False,
                "checkpoints": True,
            },
        }

    def _write_manifest(self, now_ms: Optional[int] = None) -> None:
        try:
            _atomic_write_json(self.manifest_path, self._build_manifest(now_ms=now_ms))
        except Exception as exc:
            logging.error("[monitor] failed writing manifest: %s", exc)

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

    def _rotatable_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in (self.events_dir, self.history_dir, self.checkpoints_dir):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                if path == self.current_events_path:
                    continue
                files.append(path)
        return sorted(files, key=lambda path: path.stat().st_mtime)

    def _prune_retention(self, now_ms: Optional[int] = None) -> None:
        now_ms = self._now_ms() if now_ms is None else int(now_ms)
        if self.last_retention_ms and now_ms - self.last_retention_ms < 60_000:
            return
        self.last_retention_ms = now_ms
        try:
            cutoff_ms = now_ms - int(self.retain_days * 24.0 * 60.0 * 60.0 * 1000.0)
            if self.retain_days >= 0.0:
                for path in list(self._rotatable_files()):
                    try:
                        if int(path.stat().st_mtime * 1000.0) < cutoff_ms:
                            path.unlink()
                    except FileNotFoundError:
                        continue

            total_bytes = 0
            all_files: list[Path] = []
            for path in self.root.rglob("*"):
                if not path.is_file():
                    continue
                total_bytes += path.stat().st_size
                all_files.append(path)
            if total_bytes <= self.max_total_bytes:
                return

            protected = {self.manifest_path, self.state_latest_path, self.current_events_path}
            candidates = [path for path in self._rotatable_files() if path not in protected]
            for path in candidates:
                if total_bytes <= self.max_total_bytes:
                    break
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    continue
                path.unlink()
                total_bytes -= size
        except Exception as exc:
            logging.error("[monitor] retention pruning failed: %s", exc)

    def _rotate_events_if_needed(self, now_ms: Optional[int] = None) -> None:
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
            rotated = self.events_dir / f"{self._segment_label(now_ms)}.ndjson"
            os.replace(self.current_events_path, rotated)
            if self.compress_rotated_segments:
                rotated = self._gzip_file(rotated)
            self.current_events_path.touch()
            self.current_segment_started_ms = now_ms
            self._write_manifest(now_ms=now_ms)
            self._prune_retention(now_ms=now_ms)
        except Exception as exc:
            logging.error("[monitor] event rotation failed: %s", exc)

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
        now_ms = self._now_ms() if ts is None else int(ts)
        try:
            self._rotate_events_if_needed(now_ms=now_ms)
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
            line = json.dumps(envelope, separators=(",", ":"), sort_keys=True, default=_json_default)
            with open(self.current_events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._write_manifest(now_ms=now_ms)
            self._prune_retention(now_ms=now_ms)
            return envelope
        except Exception as exc:
            logging.error("[monitor] failed recording event %s: %s", kind, exc)
            return None

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
        error_payload = dict(payload or {})
        error_payload["error_type"] = type(error).__name__
        error_payload["message"] = str(error)
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
            meta["snapshot_ts_ms"] = now_ms
            meta["seq"] = self.seq
            payload["meta"] = meta
            _atomic_write_json(self.state_latest_path, payload)
            self.last_snapshot_ms = now_ms
            if self.checkpoint_interval_ms > 0 and (
                force or now_ms - self.last_checkpoint_ms >= self.checkpoint_interval_ms
            ):
                checkpoint = self.checkpoints_dir / f"state.{self._segment_label(now_ms)}.json"
                _atomic_write_json(checkpoint, payload)
                if self.compress_rotated_segments:
                    self._gzip_file(checkpoint)
                self.last_checkpoint_ms = now_ms
            self._write_manifest(now_ms=now_ms)
            self._prune_retention(now_ms=now_ms)
            return True
        except Exception as exc:
            logging.error("[monitor] failed writing snapshot: %s", exc)
            return False

    def close(self) -> None:
        self._write_manifest()
