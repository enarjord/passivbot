"""Immutable, non-secret identity for one live Passivbot runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import passivbot_rust as pbr

from passivbot_version import __version__
from rust_utils import collect_runtime_provenance


RUNTIME_IDENTITY_SCHEMA_VERSION = 1
FILL_PROVENANCE_SCHEMA_VERSION = 1


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonical_value(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _canonical_value(tolist())
        except Exception:
            pass
    return str(value)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a canonical config without retaining or emitting its contents."""
    payload = json.dumps(
        _canonical_value(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_value(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _rust_build_info() -> dict[str, str]:
    getter = getattr(pbr, "runtime_build_info", None)
    if not callable(getter):
        return {}
    try:
        result = getter()
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return {str(key): str(value) for key, value in result.items() if value not in (None, "")}


@dataclass(frozen=True)
class RuntimeIdentity:
    schema_version: int
    run_id: str
    started_at_ms: int
    passivbot_version: str
    python_git_commit: str
    python_git_dirty: Optional[bool]
    config_sha256: str
    rust_crate_version: str
    rust_source_sha256: str
    rust_artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fill_provenance(self) -> dict[str, object]:
        return {
            "schema_version": FILL_PROVENANCE_SCHEMA_VERSION,
            "attribution": "first_ingested_by_runtime",
            "runtime": self.to_dict(),
        }


def build_runtime_identity(
    config: Mapping[str, Any],
    *,
    started_at_ms: int,
    repo_root: Optional[Path] = None,
) -> RuntimeIdentity:
    repo_root = (
        Path(__file__).resolve().parents[1] if repo_root is None else Path(repo_root).resolve()
    )
    commit = _git_value(repo_root, "rev-parse", "HEAD") or "unknown"
    status = _git_value(repo_root, "status", "--porcelain", "--untracked-files=no")
    try:
        config_digest = config_sha256(config)
    except Exception:
        config_digest = "unknown"
    try:
        rust_runtime = collect_runtime_provenance()
    except Exception:
        rust_runtime = {}
    rust_build = _rust_build_info()
    return RuntimeIdentity(
        schema_version=RUNTIME_IDENTITY_SCHEMA_VERSION,
        run_id=uuid.uuid4().hex,
        started_at_ms=int(started_at_ms),
        passivbot_version=str(__version__),
        python_git_commit=commit,
        python_git_dirty=None if status is None else bool(status),
        config_sha256=config_digest,
        rust_crate_version=str(rust_build.get("crate_version") or "unknown"),
        rust_source_sha256=str(
            rust_build.get("source_fingerprint")
            or rust_runtime.get("runtime_module_source_stamp")
            or "unknown"
        ),
        rust_artifact_sha256=str(rust_runtime.get("runtime_module_sha256") or "unknown"),
    )


def write_runtime_manifest(
    identity: RuntimeIdentity,
    *,
    root: Path = Path("caches/runtime"),
) -> Path:
    """Persist one immutable manifest; an existing run id is never overwritten."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{identity.run_id}.json"
    payload = json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path
