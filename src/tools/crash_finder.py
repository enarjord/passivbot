from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import (
    BACKTEST_OHLCV_DTYPE,
    BACKTEST_OHLCV_FIELDS,
    month_offset,
    timeframe_to_interval_ms,
)
from utils import symbol_to_coin


DEFAULT_ROOT = Path("caches/ohlcvs")
HOUR_MS = 60 * 60_000


@dataclass(frozen=True)
class SymbolRange:
    exchange: str
    timeframe: str
    symbol: str
    first_ts: int
    last_ts: int


@dataclass(frozen=True)
class CachedRange:
    timestamps: np.ndarray
    values: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class ScannedRange:
    exchange: str
    timeframe: str
    symbol: str
    first_ts: int
    last_ts: int
    first_iso: str
    last_iso: str
    valid_rows: int
    hours_scanned: int
    events_found: int


@dataclass(frozen=True)
class ScanError:
    exchange: str
    timeframe: str
    symbol: str
    first_ts: int
    last_ts: int
    first_iso: str
    last_iso: str
    error_type: str
    message: str


@dataclass(frozen=True)
class CrashEvent:
    exchange: str
    symbol: str
    coin: str
    timestamp: int
    timestamp_iso: str
    valid_minutes: int
    range_log: float
    ordered_high_to_later_low_log: float
    prev_close_low_log: float | None
    hour_high: float
    hour_low: float
    previous_close: float | None


@dataclass(frozen=True)
class CrashCluster:
    label: str
    timestamp: int
    timestamp_iso: str
    start_ts: int
    end_ts: int
    start_iso: str
    end_iso: str
    severity: float
    event_count: int
    affected_coin_count: int
    affected_coins: list[str]
    exchanges: list[str]
    market_wide: bool


def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _db_path(root: Path) -> Path:
    return root / "catalog.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_symbol_ranges(
    root: Path,
    *,
    exchange: str | None,
    timeframe: str,
    symbols: Sequence[str] | None,
) -> list[SymbolRange]:
    db_path = _db_path(root)
    if not db_path.exists():
        raise FileNotFoundError(f"v2 OHLCV catalog not found: {db_path}")
    filters = ["timeframe = ?", "first_ts IS NOT NULL", "last_ts IS NOT NULL"]
    params: list[Any] = [timeframe]
    if exchange:
        filters.append("exchange = ?")
        params.append(exchange)
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        filters.append(f"symbol IN ({placeholders})")
        params.extend(symbols)
    where = " AND ".join(filters)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT exchange, timeframe, symbol, first_ts, last_ts
            FROM symbols
            WHERE {where}
            ORDER BY exchange, symbol
            """,
            params,
        ).fetchall()
    return [
        SymbolRange(
            exchange=str(row["exchange"]),
            timeframe=str(row["timeframe"]),
            symbol=str(row["symbol"]),
            first_ts=int(row["first_ts"]),
            last_ts=int(row["last_ts"]),
        )
        for row in rows
    ]


def _floor_ts(ts_ms: int, interval_ms: int) -> int:
    return int(ts_ms) - (int(ts_ms) % int(interval_ms))


def _finite_positive_mask(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    high = values[:, 0]
    low = values[:, 1]
    close = values[:, 2]
    return (
        valid
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & (high > 0.0)
        & (low > 0.0)
        & (close > 0.0)
        & (high >= low)
    )


def _worst_ordered_high_to_later_low_log(high: np.ndarray, low: np.ndarray) -> float:
    prefix_high = np.maximum.accumulate(high)
    ratios = low / prefix_high
    return float(np.log(np.min(ratios)))


def _compute_chunk_checksum_readonly(body_path: str, valid_path: str) -> str:
    hasher = hashlib.sha256()
    body = np.load(body_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    try:
        body_arr = np.ascontiguousarray(body)
        valid_arr = np.ascontiguousarray(valid)
        hasher.update(str(body_arr.dtype).encode("utf-8"))
        hasher.update(str(body_arr.shape).encode("utf-8"))
        hasher.update(body_arr.tobytes(order="C"))
        hasher.update(str(valid_arr.dtype).encode("utf-8"))
        hasher.update(str(valid_arr.shape).encode("utf-8"))
        hasher.update(valid_arr.tobytes(order="C"))
    finally:
        del body
        del valid
    return hasher.hexdigest()


def _verify_chunk_readonly(chunk) -> None:
    if not chunk.checksum:
        raise ValueError(
            f"OHLCV chunk checksum missing for {chunk.exchange} {chunk.symbol} "
            f"{chunk.year:04d}-{chunk.month:02d}"
        )
    body = np.load(chunk.body_path, mmap_mode="r")
    valid = np.load(chunk.valid_path, mmap_mode="r")
    try:
        expected_rows = int(chunk.rows)
        expected_body_shape = (expected_rows, len(BACKTEST_OHLCV_FIELDS))
        expected_valid_shape = (expected_rows,)
        if tuple(body.shape) != expected_body_shape:
            raise ValueError(f"body shape {tuple(body.shape)} != expected {expected_body_shape}")
        if tuple(valid.shape) != expected_valid_shape:
            raise ValueError(f"valid shape {tuple(valid.shape)} != expected {expected_valid_shape}")
        if body.dtype != BACKTEST_OHLCV_DTYPE:
            raise ValueError(f"body dtype {body.dtype} != expected {BACKTEST_OHLCV_DTYPE}")
        if valid.dtype != np.bool_:
            raise ValueError(f"valid dtype {valid.dtype} != expected bool")
    finally:
        del body
        del valid
    actual = _compute_chunk_checksum_readonly(chunk.body_path, chunk.valid_path)
    if actual != chunk.checksum:
        raise ValueError(
            f"OHLCV chunk checksum mismatch for {chunk.exchange} {chunk.symbol} "
            f"{chunk.year:04d}-{chunk.month:02d}: expected {chunk.checksum} got {actual}"
        )


def read_range_readonly(
    catalog: OhlcvCatalog,
    exchange: str,
    timeframe: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> CachedRange:
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")
    interval_ms = timeframe_to_interval_ms(timeframe)
    if start_ts % interval_ms != 0 or end_ts % interval_ms != 0:
        raise ValueError(f"range must align to {timeframe}")
    timestamps = np.arange(start_ts, end_ts + interval_ms, interval_ms, dtype=np.int64)
    values = np.full((len(timestamps), len(BACKTEST_OHLCV_FIELDS)), np.nan, dtype=np.float32)
    valid_out = np.zeros(len(timestamps), dtype=np.bool_)
    for chunk in catalog.list_chunks(exchange, timeframe, symbol, start_ts, end_ts):
        _verify_chunk_readonly(chunk)
        overlap_start = max(int(chunk.start_ts), int(start_ts))
        overlap_end = min(int(chunk.end_ts), int(end_ts))
        if overlap_end < overlap_start:
            continue
        src_start = month_offset(overlap_start, int(chunk.year), int(chunk.month), timeframe)
        src_end = month_offset(overlap_end, int(chunk.year), int(chunk.month), timeframe) + 1
        dest_start = int((overlap_start - start_ts) // interval_ms)
        dest_end = dest_start + (src_end - src_start)
        body = np.load(chunk.body_path, mmap_mode="r")
        valid = np.load(chunk.valid_path, mmap_mode="r")
        try:
            values[dest_start:dest_end] = body[src_start:src_end]
            valid_out[dest_start:dest_end] = valid[src_start:src_end]
        finally:
            del body
            del valid
    return CachedRange(timestamps=timestamps, values=values, valid=valid_out)


def scan_symbol(
    catalog: OhlcvCatalog,
    symbol_range: SymbolRange,
    *,
    interval_ms: int,
    min_valid_minutes: int,
    threshold: float,
    rank_metric: str,
) -> tuple[list[CrashEvent], ScannedRange]:
    start_ts = _floor_ts(symbol_range.first_ts, interval_ms)
    end_ts = _floor_ts(symbol_range.last_ts, interval_ms)
    data = read_range_readonly(
        catalog,
        symbol_range.exchange,
        symbol_range.timeframe,
        symbol_range.symbol,
        start_ts,
        end_ts,
    )
    usable = _finite_positive_mask(data.values, data.valid)
    valid_count = int(usable.sum())
    if valid_count == 0:
        scanned = ScannedRange(
            exchange=symbol_range.exchange,
            timeframe=symbol_range.timeframe,
            symbol=symbol_range.symbol,
            first_ts=start_ts,
            last_ts=end_ts,
            first_iso=_ts_to_iso(start_ts),
            last_iso=_ts_to_iso(end_ts),
            valid_rows=0,
            hours_scanned=0,
            events_found=0,
        )
        return [], scanned

    hour_ids = data.timestamps // HOUR_MS
    unique_hours = np.unique(hour_ids[usable])
    events: list[CrashEvent] = []
    prev_close: float | None = None
    coin = symbol_to_coin(symbol_range.symbol, verbose=False)

    for hour_id in unique_hours:
        hour_mask = usable & (hour_ids == hour_id)
        indices = np.flatnonzero(hour_mask)
        if len(indices) < min_valid_minutes:
            if len(indices):
                prev_close = float(data.values[indices[-1], 2])
            continue
        hour_values = data.values[indices]
        high = np.asarray(hour_values[:, 0], dtype=np.float64)
        low = np.asarray(hour_values[:, 1], dtype=np.float64)
        close = np.asarray(hour_values[:, 2], dtype=np.float64)
        hour_high = float(np.max(high))
        hour_low = float(np.min(low))
        range_log = float(np.log(hour_low / hour_high))
        ordered_log = _worst_ordered_high_to_later_low_log(high, low)
        prev_close_low_log = None
        if prev_close is not None and np.isfinite(prev_close) and prev_close > 0.0:
            prev_close_low_log = float(np.log(hour_low / prev_close))
        if rank_metric == "range":
            threshold_value = range_log
        elif rank_metric == "prev-close":
            threshold_value = prev_close_low_log if prev_close_low_log is not None else ordered_log
        else:
            threshold_value = ordered_log
        if threshold_value <= threshold:
            event_ts = int(hour_id * HOUR_MS)
            events.append(
                CrashEvent(
                    exchange=symbol_range.exchange,
                    symbol=symbol_range.symbol,
                    coin=coin,
                    timestamp=event_ts,
                    timestamp_iso=_ts_to_iso(event_ts),
                    valid_minutes=int(len(indices)),
                    range_log=range_log,
                    ordered_high_to_later_low_log=ordered_log,
                    prev_close_low_log=prev_close_low_log,
                    hour_high=hour_high,
                    hour_low=hour_low,
                    previous_close=prev_close,
                )
            )
        prev_close = float(close[-1])

    scanned = ScannedRange(
        exchange=symbol_range.exchange,
        timeframe=symbol_range.timeframe,
        symbol=symbol_range.symbol,
        first_ts=start_ts,
        last_ts=end_ts,
        first_iso=_ts_to_iso(start_ts),
        last_iso=_ts_to_iso(end_ts),
        valid_rows=valid_count,
        hours_scanned=int(len(unique_hours)),
        events_found=len(events),
    )
    return events, scanned


def _event_severity(event: CrashEvent, rank_metric: str) -> float:
    if rank_metric == "range":
        return event.range_log
    if rank_metric == "prev-close":
        if event.prev_close_low_log is None:
            return event.ordered_high_to_later_low_log
        return event.prev_close_low_log
    return event.ordered_high_to_later_low_log


def select_top_events(
    events: Iterable[CrashEvent],
    *,
    top_per_coin: int,
    dedupe_window_ms: int,
    rank_metric: str,
) -> list[CrashEvent]:
    grouped: dict[tuple[str, str], list[CrashEvent]] = {}
    for event in events:
        grouped.setdefault((event.exchange, event.symbol), []).append(event)

    selected: list[CrashEvent] = []
    for key_events in grouped.values():
        kept: list[CrashEvent] = []
        for event in sorted(key_events, key=lambda item: (_event_severity(item, rank_metric), item.timestamp)):
            if any(abs(event.timestamp - existing.timestamp) <= dedupe_window_ms for existing in kept):
                continue
            kept.append(event)
            if top_per_coin > 0 and len(kept) >= top_per_coin:
                break
        selected.extend(kept)
    return sorted(selected, key=lambda item: (_event_severity(item, rank_metric), item.timestamp, item.symbol))


def _sanitize_label_piece(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {"_", "-"}:
            out.append("_")
    return "".join(out).strip("_") or "event"


def _cluster_label(timestamp: int, market_wide: bool, coins: Sequence[str]) -> str:
    base = datetime.fromtimestamp(int(timestamp) / 1000, tz=UTC).strftime("%Y_%m_%d_%H")
    if market_wide:
        return f"crash_{base}_market_wide"
    suffix = "_".join(_sanitize_label_piece(coin) for coin in list(coins)[:3])
    return f"crash_{base}_{suffix}"


def cluster_events(
    events: Sequence[CrashEvent],
    *,
    cluster_window_ms: int,
    min_market_wide_coins: int,
    rank_metric: str,
) -> list[CrashCluster]:
    if not events:
        return []
    clusters_raw: list[list[CrashEvent]] = []
    current: list[CrashEvent] = []
    current_last_ts: int | None = None
    for event in sorted(events, key=lambda item: (item.timestamp, item.symbol)):
        if current_last_ts is None or event.timestamp - current_last_ts <= cluster_window_ms:
            current.append(event)
        else:
            clusters_raw.append(current)
            current = [event]
        current_last_ts = event.timestamp
    if current:
        clusters_raw.append(current)

    clusters: list[CrashCluster] = []
    for raw in clusters_raw:
        worst = min(raw, key=lambda item: (_event_severity(item, rank_metric), item.timestamp))
        coins = sorted({event.coin for event in raw if event.coin})
        exchanges = sorted({event.exchange for event in raw})
        market_wide = len(coins) >= min_market_wide_coins
        start_ts = min(event.timestamp for event in raw)
        end_ts = max(event.timestamp for event in raw)
        clusters.append(
            CrashCluster(
                label=_cluster_label(worst.timestamp, market_wide, coins),
                timestamp=worst.timestamp,
                timestamp_iso=_ts_to_iso(worst.timestamp),
                start_ts=start_ts,
                end_ts=end_ts,
                start_iso=_ts_to_iso(start_ts),
                end_iso=_ts_to_iso(end_ts),
                severity=float(_event_severity(worst, rank_metric)),
                event_count=len(raw),
                affected_coin_count=len(coins),
                affected_coins=coins,
                exchanges=exchanges,
                market_wide=market_wide,
            )
        )
    return sorted(clusters, key=lambda item: (item.severity, item.timestamp))


def build_suite_payload(
    clusters: Sequence[CrashCluster],
    *,
    pre_days: int,
    post_days: int,
    top_clusters: int,
    coin_mode: str,
    all_scanned_coins: Sequence[str],
) -> dict[str, Any]:
    selected = list(clusters[:top_clusters] if top_clusters > 0 else clusters)
    scenarios = []
    all_coins = sorted(dict.fromkeys(all_scanned_coins))
    for cluster in selected:
        event_dt = datetime.fromtimestamp(cluster.timestamp / 1000, tz=UTC)
        start_dt = event_dt - timedelta(days=pre_days)
        end_dt = event_dt + timedelta(days=post_days)
        if coin_mode == "all-scanned":
            coins = all_coins
        elif coin_mode == "none":
            coins = []
        else:
            coins = list(cluster.affected_coins)
        scenario: dict[str, Any] = {
            "label": cluster.label,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
        }
        if coins:
            scenario["coins"] = coins
        if cluster.exchanges:
            scenario["exchanges"] = list(cluster.exchanges)
        scenarios.append(scenario)
    return {
        "backtest": {
            "suite_enabled": True,
            "scenarios": scenarios,
            "aggregate": {"default": "mean"},
        }
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(
    output_dir: Path,
    *,
    events: Sequence[CrashEvent],
    clusters: Sequence[CrashCluster],
    scanned: Sequence[ScannedRange],
    scan_errors: Sequence[ScanError],
    suite_payload: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_rows = [asdict(event) for event in events]
    cluster_rows = [
        {
            **asdict(cluster),
            "affected_coins": ",".join(cluster.affected_coins),
            "exchanges": ",".join(cluster.exchanges),
        }
        for cluster in clusters
    ]
    scanned_rows = [asdict(item) for item in scanned]
    scan_error_rows = [asdict(item) for item in scan_errors]
    events_path = output_dir / "crash_events.csv"
    clusters_path = output_dir / "crash_clusters.csv"
    scanned_path = output_dir / "scanned_ranges.csv"
    scan_errors_path = output_dir / "scan_errors.csv"
    suite_path = output_dir / "crash_scenarios.hjson"
    _write_csv(events_path, event_rows, list(CrashEvent.__dataclass_fields__.keys()))
    _write_csv(
        clusters_path,
        cluster_rows,
        [
            "label",
            "timestamp",
            "timestamp_iso",
            "start_ts",
            "end_ts",
            "start_iso",
            "end_iso",
            "severity",
            "event_count",
            "affected_coin_count",
            "affected_coins",
            "exchanges",
            "market_wide",
        ],
    )
    _write_csv(scanned_path, scanned_rows, list(ScannedRange.__dataclass_fields__.keys()))
    _write_csv(scan_errors_path, scan_error_rows, list(ScanError.__dataclass_fields__.keys()))
    suite_path.write_text(json.dumps(suite_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return {
        "events_csv": str(events_path),
        "clusters_csv": str(clusters_path),
        "scanned_ranges_csv": str(scanned_path),
        "scan_errors_csv": str(scan_errors_path),
        "suite_hjson": str(suite_path),
    }


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser()
    interval_ms = timeframe_to_interval_ms(args.timeframe)
    if interval_ms != 60_000:
        raise ValueError("first iteration supports scanning 1m cache data only")
    symbols = args.symbol or None
    symbol_ranges = load_symbol_ranges(
        root,
        exchange=args.exchange,
        timeframe=args.timeframe,
        symbols=symbols,
    )
    if not symbol_ranges:
        raise ValueError("no matching cached OHLCV symbols found")

    logging.info(
        "[crash-finder] scanning %d cached symbol range(s) root=%s timeframe=%s exchange=%s",
        len(symbol_ranges),
        root,
        args.timeframe,
        args.exchange or "all",
    )
    catalog = OhlcvCatalog(_db_path(root))
    all_events: list[CrashEvent] = []
    scanned_ranges: list[ScannedRange] = []
    scan_errors: list[ScanError] = []
    for symbol_range in symbol_ranges:
        logging.info(
            "[crash-finder] scan %s %s %s %s -> %s",
            symbol_range.exchange,
            symbol_range.timeframe,
            symbol_range.symbol,
            _ts_to_iso(symbol_range.first_ts),
            _ts_to_iso(symbol_range.last_ts),
        )
        try:
            events, scanned = scan_symbol(
                catalog,
                symbol_range,
                interval_ms=interval_ms,
                min_valid_minutes=args.min_valid_minutes,
                threshold=args.threshold,
                rank_metric=args.rank_metric,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            if args.on_error == "raise":
                raise
            scan_error = ScanError(
                exchange=symbol_range.exchange,
                timeframe=symbol_range.timeframe,
                symbol=symbol_range.symbol,
                first_ts=symbol_range.first_ts,
                last_ts=symbol_range.last_ts,
                first_iso=_ts_to_iso(symbol_range.first_ts),
                last_iso=_ts_to_iso(symbol_range.last_ts),
                error_type=exc.__class__.__name__,
                message=str(exc),
            )
            logging.error(
                "[crash-finder] skipped %s %s %s %s -> %s | %s: %s",
                scan_error.exchange,
                scan_error.timeframe,
                scan_error.symbol,
                scan_error.first_iso,
                scan_error.last_iso,
                scan_error.error_type,
                scan_error.message,
            )
            scan_errors.append(scan_error)
            continue
        logging.info(
            "[crash-finder] scanned %s %s valid_rows=%d hours=%d events=%d",
            scanned.exchange,
            scanned.symbol,
            scanned.valid_rows,
            scanned.hours_scanned,
            scanned.events_found,
        )
        all_events.extend(events)
        scanned_ranges.append(scanned)

    selected_events = select_top_events(
        all_events,
        top_per_coin=args.top_per_coin,
        dedupe_window_ms=int(args.dedupe_coin_window_hours * HOUR_MS),
        rank_metric=args.rank_metric,
    )
    clusters = cluster_events(
        selected_events,
        cluster_window_ms=int(args.cluster_window_hours * HOUR_MS),
        min_market_wide_coins=args.min_market_wide_coins,
        rank_metric=args.rank_metric,
    )
    if args.top_clusters > 0:
        clusters = clusters[: args.top_clusters]
    scanned_coins = sorted({symbol_to_coin(item.symbol, verbose=False) for item in scanned_ranges})
    suite_payload = build_suite_payload(
        clusters,
        pre_days=args.pre_days,
        post_days=args.post_days,
        top_clusters=0,
        coin_mode=args.scenario_coin_mode,
        all_scanned_coins=scanned_coins,
    )

    output_paths = None
    if args.output_dir:
        output_paths = write_outputs(
            Path(args.output_dir).expanduser(),
            events=selected_events,
            clusters=clusters,
            scanned=scanned_ranges,
            scan_errors=scan_errors,
            suite_payload=suite_payload,
        )
        logging.info("[crash-finder] wrote outputs to %s", args.output_dir)

    return {
        "root": str(root),
        "timeframe": args.timeframe,
        "rank_metric": args.rank_metric,
        "threshold": args.threshold,
        "symbols_attempted": len(symbol_ranges),
        "symbols_scanned": len(scanned_ranges),
        "symbols_failed": len(scan_errors),
        "events_found": len(all_events),
        "events_selected": len(selected_events),
        "clusters_selected": len(clusters),
        "scanned_ranges": [asdict(item) for item in scanned_ranges],
        "scan_errors": [asdict(item) for item in scan_errors],
        "events": [asdict(item) for item in selected_events],
        "clusters": [asdict(item) for item in clusters],
        "suite": suite_payload,
        "output_paths": output_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="passivbot tool crash-finder",
        description="Scan local v2 OHLCV cache data for the worst historical crash windows.",
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="v2 OHLCV cache root.")
    parser.add_argument("--exchange", help="Restrict scan to one exchange.")
    parser.add_argument("--symbol", action="append", help="Restrict scan to one symbol; repeatable.")
    parser.add_argument("--timeframe", default="1m", help="Cache timeframe to scan (default: 1m).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-0.10,
        help="Only keep hourly crash candidates at or below this log-return severity.",
    )
    parser.add_argument("--top-per-coin", type=int, default=10)
    parser.add_argument("--top-clusters", type=int, default=30)
    parser.add_argument("--min-valid-minutes", type=int, default=2)
    parser.add_argument("--dedupe-coin-window-hours", type=float, default=6.0)
    parser.add_argument("--cluster-window-hours", type=float, default=6.0)
    parser.add_argument("--min-market-wide-coins", type=int, default=3)
    parser.add_argument("--pre-days", type=int, default=30)
    parser.add_argument("--post-days", type=int, default=150)
    parser.add_argument(
        "--rank-metric",
        choices=("ordered", "range", "prev-close"),
        default="ordered",
        help="Metric used for event ranking and cluster severity.",
    )
    parser.add_argument(
        "--scenario-coin-mode",
        choices=("affected", "all-scanned", "none"),
        default="affected",
        help="Which coins to place in generated backtest scenarios.",
    )
    parser.add_argument("--output-dir", help="Write CSV outputs and crash_scenarios.hjson here.")
    parser.add_argument(
        "--on-error",
        choices=("skip", "raise"),
        default="skip",
        help="How to handle unreadable local cache ranges (default: skip and report).",
    )
    parser.add_argument("--json", action="store_true", help="Print the full result payload as JSON.")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
        help="Logging verbosity.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _print_text_summary(payload: dict[str, Any]) -> None:
    print(
        "Scanned "
        f"{payload['symbols_scanned']}/{payload['symbols_attempted']} symbol range(s); "
        f"skipped {payload['symbols_failed']}; "
        f"selected {payload['events_selected']} event(s), "
        f"{payload['clusters_selected']} cluster(s)."
    )
    if payload.get("output_paths"):
        print("Outputs:")
        for key, value in payload["output_paths"].items():
            print(f"  {key}: {value}")
    clusters = payload.get("clusters") or []
    if clusters:
        print("Worst clusters:")
        for cluster in clusters[:10]:
            coins = ",".join(cluster["affected_coins"][:8])
            if len(cluster["affected_coins"]) > 8:
                coins += ",..."
            print(
                f"  {cluster['label']} severity={cluster['severity']:.6f} "
                f"time={cluster['timestamp_iso']} coins={coins}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    payload = run_scan(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        _print_text_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
