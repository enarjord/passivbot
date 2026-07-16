"""Bounded, value-free inventory of secret-like material in retained logs."""

from __future__ import annotations

import gzip
import hashlib
import re
import time
from collections import Counter
from pathlib import Path
from typing import BinaryIO


DEFAULT_MAX_FILES = 200
DEFAULT_MAX_BYTES_PER_FILE = 1_000_000

SECRET_CLASSES = (
    "private_websocket_query_credentials",
    "authorization_cookie_headers",
    "labeled_secret_values",
    "bearer_basic_schemes",
    "pem_private_keys",
    "secret_url_query_params",
    "raw_http_body_html_markers",
)

_PRIVATE_WEBSOCKET_QUERY_RE = re.compile(
    r"\bwss?://[^\s\"'<>]*[?&](?:api[_-]?key|api[_-]?secret|token|signature|"
    r"password|passphrase|secret)=(?!\[redacted\](?:[&#\s]|$))[^\s\"'<>&#]+",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(
    r'^\s*[\"\']?(?:authorization|proxy-authorization|x-mbx-apikey|cookie|set-cookie)'
    r'[\"\']?\s*:\s*(?P<value>[^\r\n]+)',
    re.IGNORECASE | re.MULTILINE,
)
_LABELED_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_?&-])[\"']?"
    r"(?P<label>api[_ -]?key|api[_ -]?secret|token|signature|password|"
    r"passphrase|private[_ -]?key)[\"']?\s*[:=]\s*"
    r"(?P<value>\[redacted\]|\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&}\]\r\n]+)",
    re.IGNORECASE,
)
_BEARER_BASIC_RE = re.compile(
    r"\b(?:bearer|basic)\s+(?!\[redacted\](?:\s|$))[^\s,;\"']+", re.IGNORECASE
)
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----", re.IGNORECASE
)
_SECRET_QUERY_RE = re.compile(
    r"[?&](?:api[_-]?key|api[_-]?secret|token|"
    r"signature|password|passphrase|secret)="
    r"(?!\[redacted\](?:\W|$))[^\s\"'<>&#]+",
    re.IGNORECASE,
)
_RAW_HTTP_BODY_HTML_RE = re.compile(
    r"\b(?:request|response)\s+body\s*[:=]\s*(?=\S)|"
    r"\braw[_ -]?body\s*[:=]\s*(?=\S)|<!doctype\s+html|<html\b|<body\b",
    re.IGNORECASE,
)


def _is_redacted(value: str) -> bool:
    return value.lstrip(" \t\"'").lower().startswith("[redacted]")


def _is_fully_redacted(value: str) -> bool:
    if _is_redacted(value):
        return True
    remainder = re.sub(r"\[redacted\]", "", value, flags=re.IGNORECASE)
    remainder = re.sub(
        r"(?i)\b(?:bearer|basic|api[_ -]?key|apikey)\b", "", remainder
    )
    remainder = re.sub(r"[A-Za-z0-9_.-]+\s*=", "", remainder)
    return re.search(r"[A-Za-z0-9]", remainder) is None


def _is_benign_label(label: str, value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", label.lower())
    if normalized.endswith(("count", "counts", "length", "len", "present", "enabled")):
        return True
    return _is_fully_redacted(value)


def classify_secret_like_text(text: str) -> dict[str, int]:
    """Return classification counts without retaining any matched source values."""
    counts = Counter({name: 0 for name in SECRET_CLASSES})
    counts["private_websocket_query_credentials"] = len(_PRIVATE_WEBSOCKET_QUERY_RE.findall(text))
    counts["authorization_cookie_headers"] = sum(
        1
        for match in _HEADER_RE.finditer(text)
        if not _is_fully_redacted(match.group("value"))
    )
    counts["bearer_basic_schemes"] = len(_BEARER_BASIC_RE.findall(text))
    counts["pem_private_keys"] = len(_PEM_PRIVATE_KEY_RE.findall(text))
    counts["secret_url_query_params"] = len(_SECRET_QUERY_RE.findall(text))
    counts["raw_http_body_html_markers"] = len(_RAW_HTTP_BODY_HTML_RE.findall(text))
    counts["labeled_secret_values"] = sum(
        1
        for match in _LABELED_SECRET_RE.finditer(text)
        if not _is_benign_label(match.group("label"), match.group("value"))
    )
    return dict(counts)


def _normalized_error_type(exc: BaseException) -> str:
    """Expose only a bounded type, never exception text or a source path."""
    if isinstance(exc, PermissionError):
        return "PermissionError"
    if isinstance(exc, FileNotFoundError):
        return "FileNotFoundError"
    if isinstance(exc, gzip.BadGzipFile):
        return "BadGzipFile"
    if isinstance(exc, EOFError):
        return "EOFError"
    if isinstance(exc, UnicodeError):
        return "UnicodeError"
    return "OSError"


def _open_log(path: Path) -> BinaryIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _scan_file(path: Path, relative_path: str, *, max_bytes_per_file: int, now_s: int) -> dict:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "path": relative_path,
            "status": "unreadable",
            "error_type": _normalized_error_type(exc),
            "class_counts": {name: 0 for name in SECRET_CLASSES},
        }

    summary = {
        "path": relative_path,
        "status": "scanned",
        "size_bytes": int(stat.st_size),
        "mtime_epoch_s": int(stat.st_mtime),
        "age_seconds": max(0, int(now_s - stat.st_mtime)),
        "compressed": path.name.lower().endswith(".gz"),
        "bytes_scanned": 0,
        "truncated": False,
        "sha256": None,
        "sha256_scope": None,
        "class_counts": {name: 0 for name in SECRET_CLASSES},
    }
    try:
        with _open_log(path) as stream:
            data = stream.read(max_bytes_per_file + 1)
    except (OSError, EOFError, UnicodeError) as exc:
        summary.update(
            {
                "status": "unreadable",
                "error_type": _normalized_error_type(exc),
                "class_counts": {name: 0 for name in SECRET_CLASSES},
            }
        )
        return summary

    summary["truncated"] = len(data) > max_bytes_per_file
    data = data[:max_bytes_per_file]
    summary["bytes_scanned"] = len(data)
    summary["sha256"] = hashlib.sha256(data).hexdigest()
    if summary["compressed"]:
        summary["sha256_scope"] = (
            "decompressed_content_prefix"
            if summary["truncated"]
            else "decompressed_content"
        )
    else:
        summary["sha256_scope"] = "content_prefix" if summary["truncated"] else "full_content"
    summary["class_counts"] = classify_secret_like_text(data.decode("utf-8", errors="replace"))
    return summary


def _iter_regular_files(root: Path) -> tuple[list[tuple[str, Path]], int, int]:
    """Discover regular files in deterministic root-relative order."""
    entries: list[tuple[str, Path]] = []
    discovery_errors = 0
    symlinks_skipped = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_symlink():
                    symlinks_skipped += 1
                    continue
                if path.is_file():
                    entries.append((path.relative_to(root).as_posix(), path))
            except OSError:
                discovery_errors += 1
    except OSError:
        discovery_errors += 1
    entries.sort(key=lambda item: item[0])
    return entries, discovery_errors, symlinks_skipped


def build_log_secret_inventory(
    logs_root: str | Path,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
    now_s: int | None = None,
) -> dict:
    """Build an offline, bounded report without exposing source text or values."""
    if max_files < 0:
        raise ValueError("max_files must be non-negative")
    if max_bytes_per_file <= 0:
        raise ValueError("max_bytes_per_file must be positive")

    root = Path(logs_root)
    if not root.is_dir():
        raise ValueError("logs_root must be an existing directory")
    now = int(time.time() if now_s is None else now_s)
    candidates, discovery_errors, symlinks_skipped = _iter_regular_files(root)
    selected = candidates[:max_files]
    files = [
        _scan_file(path, relative_path, max_bytes_per_file=max_bytes_per_file, now_s=now)
        for relative_path, path in selected
    ]
    class_counts = Counter({name: 0 for name in SECRET_CLASSES})
    for file_summary in files:
        class_counts.update(file_summary["class_counts"])
    unreadable_files = sum(file_summary["status"] == "unreadable" for file_summary in files)
    return {
        "report_type": "log_secret_inventory",
        "read_only": True,
        "root": ".",
        "limits": {"max_files": max_files, "max_bytes_per_file": max_bytes_per_file},
        "summary": {
            "files_discovered": len(candidates),
            "files_scanned": len(files),
            "files_skipped_max_files": max(0, len(candidates) - len(files)),
            "files_unreadable": unreadable_files,
            "discovery_errors": discovery_errors,
            "symlinks_skipped": symlinks_skipped,
        },
        "class_counts": dict(class_counts),
        "files": files,
    }
