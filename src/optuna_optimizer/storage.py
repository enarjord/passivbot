"""Storage backend for Optuna optimization.

Provides InMemoryJournalBackend for in-memory storage during optimization,
with SQLite dump/load functions for persistence and resume capability.

The in-memory backend eliminates all disk I/O during optimization,
providing optimal throughput with single-process Rust parallelism.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

from optuna.storages import JournalStorage
from optuna.storages.journal._base import BaseJournalBackend


class InMemoryJournalBackend(BaseJournalBackend):
    """Simple in-memory journal backend for single-process optimization.

    Stores all journal logs in a regular Python list. No disk I/O occurs
    during optimization, providing optimal performance. No multiprocessing
    overhead since Rust Rayon handles parallelism.

    Usage:
        backend = InMemoryJournalBackend()
        storage = JournalStorage(backend)

    Note:
        Data is lost if the process crashes. Use dump_to_sqlite()
        to persist the study before shutdown.
    """

    def __init__(self) -> None:
        """Initialize backend with empty log list."""
        self._logs: list[str] = []

    def read_logs(self, log_number_from: int) -> Generator[dict[str, Any], None, None]:
        """Read logs starting from the given log number.

        Args:
            log_number_from: Starting log number (0-indexed)

        Yields:
            Log entries as dictionaries
        """
        for log_json in self._logs[log_number_from:]:
            yield json.loads(log_json)

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        """Append logs to the in-memory storage.

        Args:
            logs: List of log entries to append
        """
        for log in logs:
            self._logs.append(json.dumps(log))

    def __len__(self) -> int:
        """Return the number of log entries."""
        return len(self._logs)


def dump_to_sqlite(
    backend: InMemoryJournalBackend,
    study_name: str,
    sqlite_path: Path,
) -> None:
    """Copy study from in-memory backend to SQLite file using bulk insert.

    Directly inserts trial data into SQLite using executemany for ~200x speedup
    over optuna.copy_study() which commits per-trial.

    Args:
        backend: In-memory backend containing the study data
        study_name: Name of the study to copy
        sqlite_path: Path to write the SQLite database
    """
    import sqlite3

    import optuna

    if sqlite_path.exists():
        sqlite_path.unlink()

    storage = JournalStorage(backend)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # Create SQLite database with Optuna schema
    conn = sqlite3.connect(str(sqlite_path))
    try:
        _create_optuna_schema(conn)
        _bulk_insert_study(conn, study)
        conn.commit()
    finally:
        conn.close()


def _create_optuna_schema(conn: "sqlite3.Connection") -> None:
    """Create Optuna 4.x SQLite schema."""
    conn.executescript("""
        -- Alembic version tracking (required by Optuna)
        CREATE TABLE IF NOT EXISTS alembic_version (
            version_num VARCHAR(32) NOT NULL PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS version_info (
            version_info_id INTEGER PRIMARY KEY,
            schema_version INTEGER,
            library_version VARCHAR(256)
        );

        CREATE TABLE IF NOT EXISTS studies (
            study_id INTEGER PRIMARY KEY,
            study_name VARCHAR(512) NOT NULL UNIQUE
        );

        CREATE TABLE IF NOT EXISTS study_directions (
            study_direction_id INTEGER PRIMARY KEY,
            direction VARCHAR(8) NOT NULL,
            study_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            FOREIGN KEY (study_id) REFERENCES studies(study_id)
        );

        CREATE TABLE IF NOT EXISTS study_user_attributes (
            study_user_attribute_id INTEGER PRIMARY KEY,
            study_id INTEGER,
            key VARCHAR(512),
            value_json TEXT,
            FOREIGN KEY (study_id) REFERENCES studies(study_id),
            UNIQUE (study_id, key)
        );

        CREATE TABLE IF NOT EXISTS study_system_attributes (
            study_system_attribute_id INTEGER PRIMARY KEY,
            study_id INTEGER,
            key VARCHAR(512),
            value_json TEXT,
            FOREIGN KEY (study_id) REFERENCES studies(study_id),
            UNIQUE (study_id, key)
        );

        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY,
            number INTEGER,
            study_id INTEGER,
            state VARCHAR(8) NOT NULL,
            datetime_start DATETIME,
            datetime_complete DATETIME,
            FOREIGN KEY (study_id) REFERENCES studies(study_id),
            UNIQUE (study_id, number)
        );

        CREATE TABLE IF NOT EXISTS trial_values (
            trial_value_id INTEGER PRIMARY KEY,
            trial_id INTEGER NOT NULL,
            objective INTEGER NOT NULL,
            value FLOAT,
            value_type VARCHAR(7) NOT NULL,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
            UNIQUE (trial_id, objective)
        );

        CREATE TABLE IF NOT EXISTS trial_params (
            param_id INTEGER PRIMARY KEY,
            trial_id INTEGER,
            param_name VARCHAR(512),
            param_value FLOAT,
            distribution_json TEXT,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
            UNIQUE (trial_id, param_name)
        );

        CREATE TABLE IF NOT EXISTS trial_user_attributes (
            trial_user_attribute_id INTEGER PRIMARY KEY,
            trial_id INTEGER,
            key VARCHAR(512),
            value_json TEXT,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
            UNIQUE (trial_id, key)
        );

        CREATE TABLE IF NOT EXISTS trial_system_attributes (
            trial_system_attribute_id INTEGER PRIMARY KEY,
            trial_id INTEGER,
            key VARCHAR(512),
            value_json TEXT,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
            UNIQUE (trial_id, key)
        );

        CREATE TABLE IF NOT EXISTS trial_intermediate_values (
            trial_intermediate_value_id INTEGER PRIMARY KEY,
            trial_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            intermediate_value FLOAT,
            intermediate_value_type VARCHAR(7) NOT NULL,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
            UNIQUE (trial_id, step)
        );

        CREATE TABLE IF NOT EXISTS trial_heartbeats (
            trial_heartbeat_id INTEGER PRIMARY KEY,
            trial_id INTEGER NOT NULL,
            heartbeat DATETIME NOT NULL,
            FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
        );

        CREATE INDEX IF NOT EXISTS ix_trials_study_id ON trials(study_id);
        CREATE INDEX IF NOT EXISTS ix_trial_values_trial_id ON trial_values(trial_id);
        CREATE INDEX IF NOT EXISTS ix_trial_params_trial_id ON trial_params(trial_id);
    """)


def _bulk_insert_study(conn: "sqlite3.Connection", study: "optuna.Study") -> None:
    """Bulk insert study and all trials into SQLite."""
    import optuna

    # Insert alembic version (current Optuna 4.x schema - v3.2.0.a is the latest migration)
    conn.execute("INSERT INTO alembic_version (version_num) VALUES (?)", ("v3.2.0.a",))

    # Insert version info
    conn.execute(
        "INSERT INTO version_info (schema_version, library_version) VALUES (?, ?)",
        (12, optuna.__version__),
    )

    # Insert study
    conn.execute("INSERT INTO studies (study_id, study_name) VALUES (?, ?)", (0, study.study_name))

    # Insert study directions
    direction_rows = []
    for i, direction in enumerate(study.directions):
        direction_str = "MINIMIZE" if direction == optuna.study.StudyDirection.MINIMIZE else "MAXIMIZE"
        direction_rows.append((direction_str, 0, i))
    conn.executemany(
        "INSERT INTO study_directions (direction, study_id, objective) VALUES (?, ?, ?)",
        direction_rows,
    )

    # Insert study user attributes
    study_attrs = []
    for key, value in study.user_attrs.items():
        study_attrs.append((0, key, json.dumps(value)))
    if study_attrs:
        conn.executemany(
            "INSERT INTO study_user_attributes (study_id, key, value_json) VALUES (?, ?, ?)",
            study_attrs,
        )

    # Bulk insert all trials
    trial_rows = []
    value_rows = []
    param_rows = []
    user_attr_rows = []
    system_attr_rows = []
    intermediate_rows = []

    for trial in study.trials:
        tid = trial.number
        dt_start = trial.datetime_start.isoformat() if trial.datetime_start else None
        dt_complete = trial.datetime_complete.isoformat() if trial.datetime_complete else None
        trial_rows.append((tid, tid, 0, trial.state.name, dt_start, dt_complete))

        if trial.values is not None:
            for i, val in enumerate(trial.values):
                vtype = _get_value_type(val)
                value_rows.append((tid, i, val if vtype == "FINITE" else None, vtype))

        for name, val in trial.params.items():
            dist = trial.distributions.get(name)
            dist_json = json.dumps(_distribution_to_json(dist)) if dist else "{}"
            internal = dist.to_internal_repr(val) if dist else val
            param_rows.append((tid, name, internal, dist_json))

        for key, val in trial.user_attrs.items():
            user_attr_rows.append((tid, key, json.dumps(val)))

        for key, val in trial.system_attrs.items():
            system_attr_rows.append((tid, key, json.dumps(val)))

        for step, val in trial.intermediate_values.items():
            vtype = _get_value_type(val)
            intermediate_rows.append((tid, step, val if vtype == "FINITE" else None, vtype))

    # Execute bulk inserts
    conn.executemany(
        "INSERT INTO trials (trial_id, number, study_id, state, datetime_start, datetime_complete) VALUES (?,?,?,?,?,?)",
        trial_rows,
    )
    if value_rows:
        conn.executemany("INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (?,?,?,?)", value_rows)
    if param_rows:
        conn.executemany("INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) VALUES (?,?,?,?)", param_rows)
    if user_attr_rows:
        conn.executemany("INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?,?,?)", user_attr_rows)
    if system_attr_rows:
        conn.executemany("INSERT INTO trial_system_attributes (trial_id, key, value_json) VALUES (?,?,?)", system_attr_rows)
    if intermediate_rows:
        conn.executemany("INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value, intermediate_value_type) VALUES (?,?,?,?)", intermediate_rows)


def _get_value_type(value: float) -> str:
    """Get Optuna value_type string for a float value."""
    from math import isinf, isnan

    if value is None or not (isnan(value) or isinf(value)):
        return "FINITE"
    if isnan(value):
        return "NAN"
    return "INF_POS" if value > 0 else "INF_NEG"


def _distribution_to_json(dist) -> dict:
    """Convert Optuna distribution to JSON-serializable dict (only FloatDistribution used)."""
    return {"name": "FloatDistribution", "attributes": {"low": dist.low, "high": dist.high, "log": dist.log, "step": dist.step}}


def load_from_sqlite(
    sqlite_path: Path,
    study_name: str,
    sampler: "optuna.samplers.BaseSampler | None" = None,
) -> tuple[InMemoryJournalBackend, "optuna.Study"]:
    """Load existing SQLite study into memory for resume.

    Creates fresh InMemoryJournalBackend, copies study into it,
    returns backend and loaded study ready for optimization.

    Args:
        sqlite_path: Path to the SQLite database
        study_name: Name of the study to load
        sampler: Optional sampler to use (recreated in caller)

    Returns:
        Tuple of (backend, study) ready for continued optimization
    """
    import optuna

    # Create fresh in-memory backend
    backend = InMemoryJournalBackend()
    to_storage = JournalStorage(backend)

    from_storage = f"sqlite:///{sqlite_path}"

    optuna.copy_study(
        from_study_name=study_name,
        from_storage=from_storage,
        to_storage=to_storage,
    )

    # Load study from the in-memory backend
    study = optuna.load_study(
        study_name=study_name,
        storage=to_storage,
        sampler=sampler,
    )

    return backend, study
