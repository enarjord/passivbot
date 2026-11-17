"""Shared external OHLCV path resolution and cheap source revision checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


class ExternalSourceChangedError(RuntimeError):
    """The selected source changed while its prepared snapshot was being consumed."""


def source_directory_candidates(coin: str, symbol: str | None) -> list[str]:
    """Match the direct loader's supported coin/symbol and Windows folder names."""
    names = []
    windows_compat = os.name == 'nt' or os.getenv('WINDOWS_COMPATIBILITY') == '1'
    for name in (coin, symbol):
        if not name:
            continue
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid external OHLCV coin or symbol path')
        names.append(name)
        if '/' in name:
            names.append(name.replace('/', '_'))
        if windows_compat and ('/' in name or ':' in name):
            names.append(name.replace('/', '_').replace(':', '_'))
    return list(dict.fromkeys(names))


def _directory_revision(directory: Path) -> list[tuple]:
    """Inspect only daily candle file metadata, never read or duplicate raw arrays."""
    try:
        before = directory.stat()
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError('External OHLCV symbol path is not a directory')
    rows = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}\.(npz|npy)', entry.name):
                    continue
                info = entry.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('External OHLCV candle path is not a regular file')
                rows.append((entry.name, info.st_dev, info.st_ino, info.st_size,
                             info.st_mtime_ns, info.st_ctime_ns))
        after = directory.stat()
    except FileNotFoundError as exc:
        raise ExternalSourceChangedError('External OHLCV source changed during inspection; retry') from exc
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ExternalSourceChangedError('External OHLCV source changed during inspection; retry')
    return sorted(rows)


def external_source_revision(root: str | Path, identities: list[tuple]) -> str:
    """Invalidate selected-source caches on daily-file additions, edits or replacements."""
    root = Path(root)
    directories = set()
    for venue, coin, symbol in identities:
        for name in source_directory_candidates(coin, symbol):
            directories.add(Path(venue) / '1m' / name)
    digest = hashlib.sha256(b'external-ohlcv-source-revision-v1\0')
    for relative in sorted(directories, key=lambda p: p.as_posix()):
        rows = _directory_revision(root / relative)
        digest.update(json.dumps((relative.as_posix(), rows), separators=(',', ':')).encode())
    return digest.hexdigest()
