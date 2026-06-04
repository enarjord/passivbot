from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import BACKTEST_OHLCV_DTYPE, month_end_ts, month_start_ts, rows_in_month


DEFAULT_ROOT = Path("caches/ohlcvs")
SUPPORTED_TIMEFRAMES = ("1m",)


@dataclass(frozen=True)
class DoctorIssue:
    severity: str
    check: str
    path: str
    message: str
    fixable: bool = False
    fixed: bool = False


@dataclass(frozen=True)
class ChunkScan:
    exchange: str
    timeframe: str
    symbol: str
    symbol_dir: str
    year: int
    month: int
    body_path: str
    valid_path: str
    rows: int
    valid_rows: int
    first_valid_ts: int | None
    last_valid_ts: int | None
    checksum: str


@dataclass
class DoctorReport:
    root: str
    repair_catalog: bool
    prune_missing_catalog: bool
    exchanges_scanned: int = 0
    symbols_scanned: int = 0
    chunks_scanned: int = 0
    chunks_indexed: int = 0
    catalog_rows_pruned: int = 0
    issues: list[DoctorIssue] = field(default_factory=list)
    by_check: dict[str, int] = field(default_factory=dict)
    by_severity: dict[str, int] = field(default_factory=dict)

    def add_issue(self, issue: DoctorIssue) -> None:
        self.issues.append(issue)
        self.by_check[issue.check] = self.by_check.get(issue.check, 0) + 1
        self.by_severity[issue.severity] = self.by_severity.get(issue.severity, 0) + 1


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, DoctorReport):
        payload = asdict(obj)
        payload["issues"] = [_jsonable(issue) for issue in obj.issues]
        return payload
    if isinstance(obj, DoctorIssue):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _symbol_from_dir_name(symbol_dir: str) -> str:
    parts = str(symbol_dir).split("_")
    if len(parts) >= 3:
        base = "_".join(parts[:-2])
        quote = parts[-2]
        settle = parts[-1]
        if base and quote and settle:
            return f"{base}/{quote}:{settle}"
    if len(parts) == 2 and all(parts):
        return f"{parts[0]}/{parts[1]}"
    return str(symbol_dir)


def _symbol_matches(symbol_filter: str | None, symbol: str, symbol_dir: str) -> bool:
    if not symbol_filter:
        return True
    return symbol_filter in {symbol, symbol_dir}


def _iter_exchange_dirs(root: Path, exchanges: set[str] | None) -> Iterable[Path]:
    data_root = root / "data"
    if not data_root.exists():
        return []
    dirs = [path for path in data_root.iterdir() if path.is_dir()]
    if exchanges:
        dirs = [path for path in dirs if path.name in exchanges]
    return sorted(dirs, key=lambda path: path.name)


def _compute_chunk_checksum(body: np.ndarray, valid: np.ndarray) -> str:
    body_arr = np.ascontiguousarray(body)
    valid_arr = np.ascontiguousarray(valid)
    hasher = hashlib.sha256()
    hasher.update(str(body_arr.dtype).encode("utf-8"))
    hasher.update(str(body_arr.shape).encode("utf-8"))
    hasher.update(body_arr.tobytes(order="C"))
    hasher.update(str(valid_arr.dtype).encode("utf-8"))
    hasher.update(str(valid_arr.shape).encode("utf-8"))
    hasher.update(valid_arr.tobytes(order="C"))
    return hasher.hexdigest()


def _load_memmap(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"failed loading {path}: {exc}") from exc


def _scan_chunk(
    *,
    exchange: str,
    timeframe: str,
    symbol_dir: str,
    body_path: Path,
    report: DoctorReport,
) -> ChunkScan | None:
    month_text = body_path.stem
    if not month_text.isdigit():
        report.add_issue(
            DoctorIssue(
                severity="warning",
                check="unexpected_file",
                path=str(body_path),
                message="skipping non-month .npy file",
            )
        )
        return None
    year_text = body_path.parent.name
    if not year_text.isdigit():
        report.add_issue(
            DoctorIssue(
                severity="warning",
                check="unexpected_file",
                path=str(body_path),
                message="skipping chunk under non-year directory",
            )
        )
        return None
    year = int(year_text)
    month = int(month_text)
    if not 1 <= month <= 12:
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="invalid_month",
                path=str(body_path),
                message=f"month must be 01..12, got {month_text}",
            )
        )
        return None

    valid_path = body_path.with_name(f"{month_text}.valid.npy")
    if not valid_path.exists():
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="missing_valid_mask",
                path=str(body_path),
                message=f"missing matching valid mask {valid_path}",
            )
        )
        return None

    try:
        body = _load_memmap(body_path)
        valid = _load_memmap(valid_path)
    except ValueError as exc:
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="unreadable_chunk",
                path=str(body_path),
                message=str(exc),
            )
        )
        return None

    rows = rows_in_month(year, month, timeframe)
    expected_body_shape = (rows, 4)
    if tuple(body.shape) != expected_body_shape:
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="body_shape",
                path=str(body_path),
                message=f"body shape {tuple(body.shape)} != expected {expected_body_shape}",
            )
        )
        return None
    if body.dtype != BACKTEST_OHLCV_DTYPE:
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="body_dtype",
                path=str(body_path),
                message=f"body dtype {body.dtype} != expected {BACKTEST_OHLCV_DTYPE}",
            )
        )
        return None
    if tuple(valid.shape) != (rows,):
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="valid_shape",
                path=str(valid_path),
                message=f"valid shape {tuple(valid.shape)} != expected ({rows},)",
            )
        )
        return None
    if valid.dtype != np.bool_:
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="valid_dtype",
                path=str(valid_path),
                message=f"valid dtype {valid.dtype} != expected bool",
            )
        )
        return None

    true_indices = np.flatnonzero(valid)
    first_valid_ts = None
    last_valid_ts = None
    if true_indices.size:
        start_ts = month_start_ts(year, month)
        first_valid_ts = int(start_ts + int(true_indices[0]) * 60_000)
        last_valid_ts = int(start_ts + int(true_indices[-1]) * 60_000)
    checksum = _compute_chunk_checksum(body, valid)
    return ChunkScan(
        exchange=exchange,
        timeframe=timeframe,
        symbol=_symbol_from_dir_name(symbol_dir),
        symbol_dir=symbol_dir,
        year=year,
        month=month,
        body_path=str(body_path.resolve()),
        valid_path=str(valid_path.resolve()),
        rows=rows,
        valid_rows=int(valid.sum()),
        first_valid_ts=first_valid_ts,
        last_valid_ts=last_valid_ts,
        checksum=checksum,
    )


def _scan_filesystem(
    *,
    root: Path,
    exchanges: set[str] | None,
    timeframe: str,
    symbol_filter: str | None,
    report: DoctorReport,
) -> list[ChunkScan]:
    chunks: list[ChunkScan] = []
    exchange_dirs = list(_iter_exchange_dirs(root, exchanges))
    report.exchanges_scanned = len(exchange_dirs)
    seen_symbols: set[tuple[str, str, str]] = set()
    data_root = root / "data"
    if not data_root.exists():
        report.add_issue(
            DoctorIssue(
                severity="error",
                check="missing_data_root",
                path=str(data_root),
                message="v2 OHLCV data root does not exist",
            )
        )
        return chunks
    for exchange_dir in exchange_dirs:
        tf_dir = exchange_dir / timeframe
        if not tf_dir.exists():
            report.add_issue(
                DoctorIssue(
                    severity="warning",
                    check="missing_timeframe_dir",
                    path=str(tf_dir),
                    message="exchange has no requested timeframe directory",
                )
            )
            continue
        for symbol_dir in sorted([path for path in tf_dir.iterdir() if path.is_dir()]):
            symbol = _symbol_from_dir_name(symbol_dir.name)
            if not _symbol_matches(symbol_filter, symbol, symbol_dir.name):
                continue
            seen_symbols.add((exchange_dir.name, timeframe, symbol))
            for body_path in sorted(symbol_dir.glob("*/*.npy")):
                if body_path.name.endswith(".valid.npy"):
                    continue
                scan = _scan_chunk(
                    exchange=exchange_dir.name,
                    timeframe=timeframe,
                    symbol_dir=symbol_dir.name,
                    body_path=body_path,
                    report=report,
                )
                if scan is not None:
                    chunks.append(scan)
    report.symbols_scanned = len(seen_symbols)
    report.chunks_scanned = len(chunks)
    return chunks


def _set_symbol_bounds_exact(catalog: OhlcvCatalog, bounds: dict[tuple[str, str, str], tuple[int, int]]) -> None:
    with catalog._connect() as conn:
        for (exchange, timeframe, symbol), (first_ts, last_ts) in sorted(bounds.items()):
            conn.execute(
                """
                INSERT INTO symbols(exchange, timeframe, symbol, first_ts, last_ts, updated_at)
                VALUES (?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER) * 1000)
                ON CONFLICT(exchange, timeframe, symbol) DO UPDATE SET
                    first_ts = excluded.first_ts,
                    last_ts = excluded.last_ts,
                    updated_at = excluded.updated_at
                """,
                (exchange, timeframe, symbol, int(first_ts), int(last_ts)),
            )


def _repair_catalog(catalog: OhlcvCatalog, chunks: list[ChunkScan]) -> int:
    bounds: dict[tuple[str, str, str], tuple[int, int]] = {}
    for chunk in chunks:
        catalog.register_chunk(
            exchange=chunk.exchange,
            timeframe=chunk.timeframe,
            symbol=chunk.symbol,
            year=chunk.year,
            month=chunk.month,
            body_path=chunk.body_path,
            valid_path=chunk.valid_path,
            start_ts=month_start_ts(chunk.year, chunk.month),
            end_ts=month_end_ts(chunk.year, chunk.month, chunk.timeframe),
            rows=chunk.rows,
            status="open",
            checksum=chunk.checksum,
        )
        if chunk.first_valid_ts is not None and chunk.last_valid_ts is not None:
            key = (chunk.exchange, chunk.timeframe, chunk.symbol)
            current = bounds.get(key)
            if current is None:
                bounds[key] = (chunk.first_valid_ts, chunk.last_valid_ts)
            else:
                bounds[key] = (
                    min(current[0], chunk.first_valid_ts),
                    max(current[1], chunk.last_valid_ts),
                )
    _set_symbol_bounds_exact(catalog, bounds)
    return len(chunks)


def _catalog_filters(exchanges: set[str] | None, timeframe: str, symbol_filter: str | None) -> tuple[str, list[Any]]:
    clauses = ["timeframe = ?"]
    params: list[Any] = [timeframe]
    if exchanges:
        placeholders = ",".join("?" for _ in sorted(exchanges))
        clauses.append(f"exchange IN ({placeholders})")
        params.extend(sorted(exchanges))
    if symbol_filter:
        canonical_symbol = _symbol_from_dir_name(symbol_filter)
        clauses.append("symbol IN (?, ?)")
        params.extend([symbol_filter, canonical_symbol])
    return " AND ".join(clauses), params


def _rebuild_issue_counts(report: DoctorReport) -> None:
    report.by_check = {}
    report.by_severity = {}
    for issue in report.issues:
        report.by_check[issue.check] = report.by_check.get(issue.check, 0) + 1
        report.by_severity[issue.severity] = report.by_severity.get(issue.severity, 0) + 1


def _audit_missing_catalog_paths(
    catalog: OhlcvCatalog,
    *,
    exchanges: set[str] | None,
    timeframe: str,
    symbol_filter: str | None,
    report: DoctorReport,
) -> list[tuple[str, str, str, int, int]]:
    where, params = _catalog_filters(exchanges, timeframe, symbol_filter)
    missing: list[tuple[str, str, str, int, int]] = []
    with catalog._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT exchange, timeframe, symbol, year, month, body_path, valid_path
            FROM chunks
            WHERE {where}
            ORDER BY exchange, timeframe, symbol, year, month
            """,
            params,
        ).fetchall()
    for row in rows:
        body_exists = Path(row["body_path"]).exists()
        valid_exists = Path(row["valid_path"]).exists()
        if body_exists and valid_exists:
            continue
        missing.append(
            (
                str(row["exchange"]),
                str(row["timeframe"]),
                str(row["symbol"]),
                int(row["year"]),
                int(row["month"]),
            )
        )
        report.add_issue(
            DoctorIssue(
                severity="warning",
                check="missing_catalog_path",
                path=str(row["body_path"]),
                message=(
                    f"catalog path missing for {row['exchange']} {row['timeframe']} "
                    f"{row['symbol']} {int(row['year']):04d}-{int(row['month']):02d}"
                ),
                fixable=True,
            )
        )
    return missing


def _prune_catalog_rows(catalog: OhlcvCatalog, rows: list[tuple[str, str, str, int, int]]) -> int:
    if not rows:
        return 0
    affected_symbols = {(exchange, timeframe, symbol) for exchange, timeframe, symbol, _year, _month in rows}
    with catalog._connect() as conn:
        for exchange, timeframe, symbol, year, month in rows:
            conn.execute(
                """
                DELETE FROM chunks
                WHERE exchange = ? AND timeframe = ? AND symbol = ? AND year = ? AND month = ?
                """,
                (exchange, timeframe, symbol, int(year), int(month)),
            )
    _rebuild_catalog_bounds_for_symbols(catalog, affected_symbols)
    return len(rows)


def _rebuild_catalog_bounds_for_symbols(
    catalog: OhlcvCatalog, symbols: set[tuple[str, str, str]]
) -> None:
    if not symbols:
        return
    with catalog._connect() as conn:
        for exchange, timeframe, symbol in sorted(symbols):
            rows = conn.execute(
                """
                SELECT start_ts, body_path, valid_path
                FROM chunks
                WHERE exchange = ? AND timeframe = ? AND symbol = ?
                ORDER BY year, month
                """,
                (exchange, timeframe, symbol),
            ).fetchall()
            first_ts = None
            last_ts = None
            for row in rows:
                valid_path = Path(row["valid_path"])
                if not valid_path.exists():
                    continue
                try:
                    valid = _load_memmap(valid_path)
                except ValueError:
                    continue
                true_indices = np.flatnonzero(valid)
                if not true_indices.size:
                    continue
                chunk_first = int(row["start_ts"]) + int(true_indices[0]) * 60_000
                chunk_last = int(row["start_ts"]) + int(true_indices[-1]) * 60_000
                first_ts = chunk_first if first_ts is None else min(first_ts, chunk_first)
                last_ts = chunk_last if last_ts is None else max(last_ts, chunk_last)
            if first_ts is None or last_ts is None:
                conn.execute(
                    """
                    DELETE FROM symbols
                    WHERE exchange = ? AND timeframe = ? AND symbol = ?
                    """,
                    (exchange, timeframe, symbol),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO symbols(exchange, timeframe, symbol, first_ts, last_ts, updated_at)
                    VALUES (?, ?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER) * 1000)
                    ON CONFLICT(exchange, timeframe, symbol) DO UPDATE SET
                        first_ts = excluded.first_ts,
                        last_ts = excluded.last_ts,
                        updated_at = excluded.updated_at
                    """,
                    (exchange, timeframe, symbol, int(first_ts), int(last_ts)),
                )


def run_doctor(
    *,
    root: Path,
    exchanges: set[str] | None = None,
    timeframe: str = "1m",
    symbol: str | None = None,
    repair_catalog: bool = False,
    prune_missing_catalog: bool = False,
) -> DoctorReport:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"unsupported timeframe {timeframe!r}; supported: {', '.join(SUPPORTED_TIMEFRAMES)}")
    root = Path(root)
    report = DoctorReport(
        root=str(root),
        repair_catalog=bool(repair_catalog),
        prune_missing_catalog=bool(prune_missing_catalog),
    )
    chunks = _scan_filesystem(
        root=root,
        exchanges=exchanges,
        timeframe=timeframe,
        symbol_filter=symbol,
        report=report,
    )
    catalog_path = root / "catalog.sqlite"
    if not catalog_path.exists() and not (repair_catalog or prune_missing_catalog):
        report.add_issue(
            DoctorIssue(
                severity="warning",
                check="missing_catalog",
                path=str(catalog_path),
                message="catalog.sqlite does not exist; run with --repair-catalog to build it",
                fixable=True,
            )
        )
        return report

    catalog = OhlcvCatalog(catalog_path)
    if repair_catalog:
        report.chunks_indexed = _repair_catalog(catalog, chunks)
    missing_catalog_rows = _audit_missing_catalog_paths(
        catalog,
        exchanges=exchanges,
        timeframe=timeframe,
        symbol_filter=symbol,
        report=report,
    )
    if prune_missing_catalog:
        pruned = _prune_catalog_rows(catalog, missing_catalog_rows)
        report.catalog_rows_pruned = pruned
        if pruned:
            report.issues = [
                issue
                for issue in report.issues
                if issue.check != "missing_catalog_path"
            ]
            _rebuild_issue_counts(report)
    return report


def print_text_report(report: DoctorReport) -> None:
    print(f"Root: {report.root}")
    print(
        "Scanned: "
        f"exchanges={report.exchanges_scanned} "
        f"symbols={report.symbols_scanned} "
        f"chunks={report.chunks_scanned}"
    )
    if report.repair_catalog:
        print(f"Catalog chunks indexed: {report.chunks_indexed}")
    if report.prune_missing_catalog:
        print(f"Catalog rows pruned: {report.catalog_rows_pruned}")
    if not report.issues:
        print("Issues: none")
        return
    print(
        "Issues: "
        f"total={len(report.issues)} "
        + " ".join(f"{key}={value}" for key, value in sorted(report.by_severity.items()))
    )
    for issue in report.issues[:50]:
        print(
            f"  [{issue.severity}] {issue.check}: {issue.path} - {issue.message}"
            + (" (fixable)" if issue.fixable else "")
        )
    if len(report.issues) > 50:
        print(f"  ... {len(report.issues) - 50} more issues omitted; use --json for full output")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool ohlcvs-doctor",
        description="Audit and repair v2 OHLCV catalog metadata from caches/ohlcvs/data chunks.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="v2 OHLCV root")
    parser.add_argument(
        "--exchange",
        action="append",
        help="exchange directory to scan; may be provided multiple times",
    )
    parser.add_argument("--timeframe", default="1m", help="timeframe to scan")
    parser.add_argument(
        "--symbol",
        help="symbol to scan, either catalog form like BTC/USDT:USDT or directory form BTC_USDT_USDT",
    )
    parser.add_argument(
        "--repair-catalog",
        action="store_true",
        help="create/update catalog.sqlite chunk rows and exact symbol bounds from data chunks",
    )
    parser.add_argument(
        "--prune-missing-catalog",
        action="store_true",
        help="delete catalog chunk rows whose body or valid file path does not exist",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    exchanges = set(args.exchange) if args.exchange else None
    report = run_doctor(
        root=Path(args.root),
        exchanges=exchanges,
        timeframe=str(args.timeframe),
        symbol=args.symbol,
        repair_catalog=bool(args.repair_catalog),
        prune_missing_catalog=bool(args.prune_missing_catalog),
    )
    if args.json:
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    else:
        print_text_report(report)
    return 1 if report.by_severity.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
