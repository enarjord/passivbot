"""Storage backend for Optuna optimization.

Uses JournalStorage with append-only file backend for safe concurrent
multi-process access. Each write operation is atomic, eliminating the
partial-read corruption that can occur with SQLite.
"""
from __future__ import annotations

from pathlib import Path

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def create_journal_storage(journal_path: Path) -> JournalStorage:
    """Create JournalStorage for multi-process optimization.

    JournalStorage uses an append-only log format where each operation
    is written atomically. This eliminates the partial-read corruption
    (JSONDecodeError) that can occur with SQLite when multiple workers
    write simultaneously, even with WAL mode enabled.

    Args:
        journal_path: Path to the journal log file

    Returns:
        Configured JournalStorage instance
    """
    return JournalStorage(JournalFileBackend(str(journal_path)))
