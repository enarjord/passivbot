from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ohlcv_catalog import OhlcvCatalog


DEFAULT_ROOT = Path("caches/ohlcvs")


@dataclass(frozen=True)
class SymbolSummary:
    exchange: str
    timeframe: str
    symbol: str
    first_ts: int | None
    last_ts: int | None
    chunk_count: int
    persistent_gap_count: int


@dataclass(frozen=True)
class ChunkSummary:
    year: int
    month: int
    status: str
    start_ts: int
    end_ts: int
    rows: int
    valid_rows: int
    first_valid_ts: int | None
    last_valid_ts: int | None
    body_path: str
    valid_path: str


def _ts_to_iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _db_path(root: Path) -> Path:
    return root / "catalog.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _require_db(root: Path) -> Path:
    db_path = _db_path(root)
    if not db_path.exists():
        raise FileNotFoundError(f"v2 OHLCV catalog not found: {db_path}")
    return db_path


def _fetch_overview_counts(conn: sqlite3.Connection, exchange: str | None, timeframe: str) -> dict[str, int]:
    filters = ["timeframe = ?"]
    params: list[Any] = [timeframe]
    if exchange:
        filters.append("exchange = ?")
        params.append(exchange)
    where = " AND ".join(filters)
    counts = {
        "symbols": conn.execute(f"SELECT COUNT(*) FROM symbols WHERE {where}", params).fetchone()[0],
        "chunks": conn.execute(f"SELECT COUNT(*) FROM chunks WHERE {where}", params).fetchone()[0],
        "gaps": conn.execute(f"SELECT COUNT(*) FROM gaps WHERE {where}", params).fetchone()[0],
        "persistent_gaps": conn.execute(
            f"SELECT COUNT(*) FROM gaps WHERE {where} AND persistent = 1", params
        ).fetchone()[0],
        "fetch_attempts": conn.execute(f"SELECT COUNT(*) FROM fetch_log WHERE {where}", params).fetchone()[0],
    }
    return {key: int(value) for key, value in counts.items()}


def _fetch_symbol_summaries(
    conn: sqlite3.Connection, exchange: str | None, timeframe: str, limit: int
) -> list[SymbolSummary]:
    filters = ["s.timeframe = ?"]
    params: list[Any] = [timeframe]
    if exchange:
        filters.append("s.exchange = ?")
        params.append(exchange)
    where = " AND ".join(filters)
    limit_sql = "" if limit == 0 else f"LIMIT {int(limit)}"
    rows = conn.execute(
        f"""
        SELECT
            s.exchange,
            s.timeframe,
            s.symbol,
            s.first_ts,
            s.last_ts,
            COALESCE((
                SELECT COUNT(*)
                FROM chunks c
                WHERE c.exchange = s.exchange
                  AND c.timeframe = s.timeframe
                  AND c.symbol = s.symbol
            ), 0) AS chunk_count,
            COALESCE((
                SELECT COUNT(*)
                FROM gaps g
                WHERE g.exchange = s.exchange
                  AND g.timeframe = s.timeframe
                  AND g.symbol = s.symbol
                  AND g.persistent = 1
            ), 0) AS persistent_gap_count
        FROM symbols s
        WHERE {where}
        ORDER BY s.exchange, s.symbol
        {limit_sql}
        """,
        params,
    ).fetchall()
    return [SymbolSummary(**dict(row)) for row in rows]


def _summarize_chunk(chunk) -> ChunkSummary:
    valid = np.load(chunk.valid_path, mmap_mode="r")
    valid_rows = int(valid.sum())
    true_indices = np.flatnonzero(valid)
    first_valid_ts = None
    last_valid_ts = None
    if true_indices.size:
        interval_ms = 60_000 if chunk.timeframe == "1m" else 60 * 60_000
        first_valid_ts = int(chunk.start_ts + int(true_indices[0]) * interval_ms)
        last_valid_ts = int(chunk.start_ts + int(true_indices[-1]) * interval_ms)
    return ChunkSummary(
        year=int(chunk.year),
        month=int(chunk.month),
        status=str(chunk.status),
        start_ts=int(chunk.start_ts),
        end_ts=int(chunk.end_ts),
        rows=int(chunk.rows),
        valid_rows=valid_rows,
        first_valid_ts=first_valid_ts,
        last_valid_ts=last_valid_ts,
        body_path=str(chunk.body_path),
        valid_path=str(chunk.valid_path),
    )


def build_overview_payload(root: Path, exchange: str | None, timeframe: str, limit: int) -> dict[str, Any]:
    db_path = _require_db(root)
    with _connect(db_path) as conn:
        counts = _fetch_overview_counts(conn, exchange=exchange, timeframe=timeframe)
        symbol_summaries = _fetch_symbol_summaries(conn, exchange=exchange, timeframe=timeframe, limit=limit)
    return {
        "root": str(root),
        "db_path": str(db_path),
        "filters": {"exchange": exchange, "timeframe": timeframe},
        "counts": counts,
        "symbols": [
            {
                **asdict(item),
                "first_iso": _ts_to_iso(item.first_ts),
                "last_iso": _ts_to_iso(item.last_ts),
            }
            for item in symbol_summaries
        ],
    }


def build_symbol_payload(
    root: Path, exchange: str, timeframe: str, symbol: str, fetch_log_limit: int
) -> dict[str, Any]:
    db_path = _require_db(root)
    catalog = OhlcvCatalog(db_path)
    first_ts, last_ts = catalog.get_symbol_bounds(exchange, timeframe, symbol)
    if first_ts is None or last_ts is None:
        raise FileNotFoundError(f"no v2 OHLCV symbol found for {exchange}:{timeframe}:{symbol}")
    chunks = catalog.list_chunks(exchange, timeframe, symbol, first_ts, last_ts)
    all_start_ts = -1
    all_end_ts = 9_223_372_036_854_775_807
    gaps = catalog.get_gaps(exchange, timeframe, symbol, all_start_ts, all_end_ts)
    attempts = catalog.list_fetch_attempts(exchange, timeframe, symbol, all_start_ts, all_end_ts)
    attempts = attempts[-fetch_log_limit:] if fetch_log_limit > 0 else attempts
    chunk_summaries = [_summarize_chunk(chunk) for chunk in chunks]
    return {
        "root": str(root),
        "db_path": str(db_path),
        "exchange": exchange,
        "timeframe": timeframe,
        "symbol": symbol,
        "bounds": {
            "first_ts": int(first_ts),
            "last_ts": int(last_ts),
            "first_iso": _ts_to_iso(first_ts),
            "last_iso": _ts_to_iso(last_ts),
        },
        "chunks": [
            {
                **asdict(item),
                "start_iso": _ts_to_iso(item.start_ts),
                "end_iso": _ts_to_iso(item.end_ts),
                "first_valid_iso": _ts_to_iso(item.first_valid_ts),
                "last_valid_iso": _ts_to_iso(item.last_valid_ts),
            }
            for item in chunk_summaries
        ],
        "gaps": [
            {
                **asdict(gap),
                "start_iso": _ts_to_iso(gap.start_ts),
                "end_iso": _ts_to_iso(gap.end_ts),
                "last_attempt_iso": _ts_to_iso(gap.last_attempt_at),
                "next_retry_iso": _ts_to_iso(gap.next_retry_at),
            }
            for gap in gaps
        ],
        "fetch_attempts": [
            {
                **asdict(attempt),
                "start_iso": _ts_to_iso(attempt.start_ts),
                "end_iso": _ts_to_iso(attempt.end_ts),
                "created_iso": _ts_to_iso(attempt.created_at),
            }
            for attempt in attempts
        ],
    }


def print_overview(payload: dict[str, Any]) -> None:
    print(f"Root: {payload['root']}")
    print(f"Catalog: {payload['db_path']}")
    print(
        "Counts: "
        f"symbols={payload['counts']['symbols']} "
        f"chunks={payload['counts']['chunks']} "
        f"gaps={payload['counts']['gaps']} "
        f"persistent_gaps={payload['counts']['persistent_gaps']} "
        f"fetch_attempts={payload['counts']['fetch_attempts']}"
    )
    symbols = payload["symbols"]
    if not symbols:
        print("No matching symbols.")
        return
    print("Symbols:")
    for item in symbols:
        print(
            "  "
            f"{item['exchange']} {item['timeframe']} {item['symbol']} "
            f"first={item['first_iso'] or '-'} last={item['last_iso'] or '-'} "
            f"chunks={item['chunk_count']} persistent_gaps={item['persistent_gap_count']}"
        )


def print_symbol_details(payload: dict[str, Any]) -> None:
    print(
        f"Symbol: {payload['exchange']} {payload['timeframe']} {payload['symbol']}\n"
        f"Bounds: {payload['bounds']['first_iso']} -> {payload['bounds']['last_iso']}"
    )
    print(f"Chunks: {len(payload['chunks'])}")
    for chunk in payload["chunks"]:
        print(
            "  "
            f"{chunk['year']:04d}-{chunk['month']:02d} status={chunk['status']} "
            f"rows={chunk['rows']} valid_rows={chunk['valid_rows']} "
            f"first_valid={chunk['first_valid_iso'] or '-'} "
            f"last_valid={chunk['last_valid_iso'] or '-'}"
        )
    print(f"Gaps: {len(payload['gaps'])}")
    for gap in payload["gaps"]:
        print(
            "  "
            f"{gap['start_iso']} -> {gap['end_iso']} "
            f"reason={gap['reason']} persistent={'y' if gap['persistent'] else 'n'} "
            f"retry_count={gap['retry_count']} note={gap['note'] or '-'}"
        )
    print(f"Recent fetch attempts: {len(payload['fetch_attempts'])}")
    for attempt in payload["fetch_attempts"]:
        print(
            "  "
            f"{attempt['created_iso']} outcome={attempt['outcome']} "
            f"range={attempt['start_iso']} -> {attempt['end_iso']} "
            f"attempt={attempt['attempt']} latency_ms={attempt['latency_ms'] or '-'} "
            f"note={attempt['note'] or '-'}"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool inspect-ohlcvs",
        description="Inspect v2 OHLCV cache metadata, chunk coverage, persistent gaps, and fetch attempts",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="v2 OHLCV cache root")
    parser.add_argument("--exchange", help="exchange id to inspect")
    parser.add_argument("--timeframe", default="1m", help="timeframe to inspect")
    parser.add_argument("--symbol", help="symbol to inspect in detail")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="max number of symbols to list in overview mode; 0 means all",
    )
    parser.add_argument(
        "--fetch-log-limit",
        type=int,
        default=10,
        help="max number of recent fetch attempts to show in detail mode; 0 means all",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.symbol:
        if not args.exchange:
            parser.error("--exchange is required when --symbol is provided")
        payload = build_symbol_payload(
            root=root,
            exchange=str(args.exchange),
            timeframe=str(args.timeframe),
            symbol=str(args.symbol),
            fetch_log_limit=int(args.fetch_log_limit),
        )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print_symbol_details(payload)
        return 0

    payload = build_overview_payload(
        root=root,
        exchange=args.exchange,
        timeframe=str(args.timeframe),
        limit=int(args.limit),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_overview(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
