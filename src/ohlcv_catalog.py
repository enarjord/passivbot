from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _utc_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class ChunkRecord:
    exchange: str
    timeframe: str
    symbol: str
    year: int
    month: int
    body_path: str
    valid_path: str
    start_ts: int
    end_ts: int
    rows: int
    status: str
    schema_version: int
    checksum: Optional[str]
    updated_at: int


@dataclass(frozen=True)
class GapRecord:
    exchange: str
    timeframe: str
    symbol: str
    start_ts: int
    end_ts: int
    reason: str
    persistent: bool
    retry_count: int
    last_attempt_at: Optional[int]
    next_retry_at: Optional[int]
    note: Optional[str]


@dataclass(frozen=True)
class FetchLogRecord:
    exchange: str
    timeframe: str
    symbol: str
    start_ts: int
    end_ts: int
    attempt: int
    outcome: str
    latency_ms: Optional[int]
    created_at: int
    note: Optional[str]


class OhlcvCatalog:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS symbols (
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    first_ts INTEGER,
                    last_ts INTEGER,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(exchange, timeframe, symbol)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    month INTEGER NOT NULL,
                    body_path TEXT NOT NULL,
                    valid_path TEXT NOT NULL,
                    start_ts INTEGER NOT NULL,
                    end_ts INTEGER NOT NULL,
                    rows INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    checksum TEXT,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(exchange, timeframe, symbol, year, month)
                );
                CREATE TABLE IF NOT EXISTS gaps (
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_ts INTEGER NOT NULL,
                    end_ts INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    persistent INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    last_attempt_at INTEGER,
                    next_retry_at INTEGER,
                    note TEXT,
                    PRIMARY KEY(exchange, timeframe, symbol, start_ts, end_ts)
                );
                CREATE TABLE IF NOT EXISTS fetch_log (
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_ts INTEGER NOT NULL,
                    end_ts INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    latency_ms INTEGER,
                    created_at INTEGER NOT NULL,
                    note TEXT,
                    PRIMARY KEY(exchange, timeframe, symbol, start_ts, end_ts, attempt)
                );
                """
            )

    def get_symbol_bounds(
        self, exchange: str, timeframe: str, symbol: str
    ) -> tuple[int | None, int | None]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT first_ts, last_ts
                FROM symbols
                WHERE exchange = ? AND timeframe = ? AND symbol = ?
                """,
                (exchange, timeframe, symbol),
            ).fetchone()
        if row is None:
            return None, None
        return row["first_ts"], row["last_ts"]

    def upsert_symbol_bounds(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> None:
        now = _utc_ms()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO symbols(exchange, timeframe, symbol, first_ts, last_ts, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, timeframe, symbol) DO UPDATE SET
                    first_ts = CASE
                        WHEN symbols.first_ts IS NULL THEN excluded.first_ts
                        ELSE MIN(symbols.first_ts, excluded.first_ts)
                    END,
                    last_ts = CASE
                        WHEN symbols.last_ts IS NULL THEN excluded.last_ts
                        ELSE MAX(symbols.last_ts, excluded.last_ts)
                    END,
                    updated_at = excluded.updated_at
                """,
                (exchange, timeframe, symbol, int(start_ts), int(end_ts), now),
            )

    def register_chunk(
        self,
        *,
        exchange: str,
        timeframe: str,
        symbol: str,
        year: int,
        month: int,
        body_path: str,
        valid_path: str,
        start_ts: int,
        end_ts: int,
        rows: int,
        status: str,
        schema_version: int = 1,
        checksum: str | None = None,
    ) -> None:
        now = _utc_ms()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chunks(
                    exchange, timeframe, symbol, year, month, body_path, valid_path,
                    start_ts, end_ts, rows, status, schema_version, checksum, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, timeframe, symbol, year, month) DO UPDATE SET
                    body_path = excluded.body_path,
                    valid_path = excluded.valid_path,
                    start_ts = excluded.start_ts,
                    end_ts = excluded.end_ts,
                    rows = excluded.rows,
                    status = excluded.status,
                    schema_version = excluded.schema_version,
                    checksum = excluded.checksum,
                    updated_at = excluded.updated_at
                """,
                (
                    exchange,
                    timeframe,
                    symbol,
                    int(year),
                    int(month),
                    body_path,
                    valid_path,
                    int(start_ts),
                    int(end_ts),
                    int(rows),
                    status,
                    int(schema_version),
                    checksum,
                    now,
                ),
            )

    def list_chunks(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> list[ChunkRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE exchange = ? AND timeframe = ? AND symbol = ?
                  AND end_ts >= ? AND start_ts <= ?
                ORDER BY year, month
                """,
                (exchange, timeframe, symbol, int(start_ts), int(end_ts)),
            ).fetchall()
        return [ChunkRecord(**dict(row)) for row in rows]

    def mark_gap(
        self,
        *,
        exchange: str,
        timeframe: str,
        symbol: str,
        start_ts: int,
        end_ts: int,
        reason: str,
        persistent: bool,
        retry_count: int = 0,
        last_attempt_at: int | None = None,
        next_retry_at: int | None = None,
        note: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gaps(
                    exchange, timeframe, symbol, start_ts, end_ts, reason, persistent,
                    retry_count, last_attempt_at, next_retry_at, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, timeframe, symbol, start_ts, end_ts) DO UPDATE SET
                    reason = excluded.reason,
                    persistent = excluded.persistent,
                    retry_count = excluded.retry_count,
                    last_attempt_at = excluded.last_attempt_at,
                    next_retry_at = excluded.next_retry_at,
                    note = excluded.note
                """,
                (
                    exchange,
                    timeframe,
                    symbol,
                    int(start_ts),
                    int(end_ts),
                    reason,
                    int(bool(persistent)),
                    int(retry_count),
                    last_attempt_at,
                    next_retry_at,
                    note,
                ),
            )

    def get_gaps(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> list[GapRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM gaps
                WHERE exchange = ? AND timeframe = ? AND symbol = ?
                  AND end_ts >= ? AND start_ts <= ?
                ORDER BY start_ts, end_ts
                """,
                (exchange, timeframe, symbol, int(start_ts), int(end_ts)),
            ).fetchall()
        return [
            GapRecord(
                exchange=row["exchange"],
                timeframe=row["timeframe"],
                symbol=row["symbol"],
                start_ts=row["start_ts"],
                end_ts=row["end_ts"],
                reason=row["reason"],
                persistent=bool(row["persistent"]),
                retry_count=row["retry_count"],
                last_attempt_at=row["last_attempt_at"],
                next_retry_at=row["next_retry_at"],
                note=row["note"],
            )
            for row in rows
        ]

    def get_persistent_gaps(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> list[GapRecord]:
        return [
            gap
            for gap in self.get_gaps(exchange, timeframe, symbol, start_ts, end_ts)
            if gap.persistent
        ]

    def record_fetch_attempt(
        self,
        *,
        exchange: str,
        timeframe: str,
        symbol: str,
        start_ts: int,
        end_ts: int,
        attempt: int,
        outcome: str,
        latency_ms: int | None = None,
        note: str | None = None,
    ) -> None:
        now = _utc_ms()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fetch_log(
                    exchange, timeframe, symbol, start_ts, end_ts, attempt, outcome,
                    latency_ms, created_at, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exchange, timeframe, symbol, start_ts, end_ts, attempt) DO UPDATE SET
                    outcome = excluded.outcome,
                    latency_ms = excluded.latency_ms,
                    created_at = excluded.created_at,
                    note = excluded.note
                """,
                (
                    exchange,
                    timeframe,
                    symbol,
                    int(start_ts),
                    int(end_ts),
                    int(attempt),
                    str(outcome),
                    latency_ms if latency_ms is None else int(latency_ms),
                    now,
                    note,
                ),
            )

    def list_fetch_attempts(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> list[FetchLogRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM fetch_log
                WHERE exchange = ? AND timeframe = ? AND symbol = ?
                  AND end_ts >= ? AND start_ts <= ?
                ORDER BY created_at, attempt
                """,
                (exchange, timeframe, symbol, int(start_ts), int(end_ts)),
            ).fetchall()
        return [FetchLogRecord(**dict(row)) for row in rows]
