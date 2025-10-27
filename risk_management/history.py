"""Persistent storage for portfolio NAV, equity history, and cash flows."""

from __future__ import annotations

import asyncio
import csv
import io
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class PortfolioHistoryStore:
    """Record portfolio metrics and expose aggregated history windows."""

    _WINDOWS: Dict[str, timedelta] = {
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }

    def __init__(self, directory: Path, *, min_interval_seconds: int = 60) -> None:
        self.base_dir = Path(directory)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "portfolio_history.sqlite3"
        self.min_interval = timedelta(seconds=max(1, int(min_interval_seconds)))
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # public API

    def record(self, snapshot: Mapping[str, Any]) -> None:
        """Persist metrics extracted from ``snapshot``."""

        portfolio = snapshot.get("portfolio") if isinstance(snapshot, Mapping) else None
        if not isinstance(portfolio, Mapping):
            return

        generated_at = self._parse_datetime(snapshot.get("generated_at"))
        nav = self._to_float(portfolio.get("balance"))
        gross = self._to_float(portfolio.get("gross_exposure"))
        net = self._to_float(portfolio.get("net_exposure"))
        realized = self._to_float(portfolio.get("daily_realized_pnl"))

        accounts = snapshot.get("accounts") if isinstance(snapshot, Mapping) else None
        unrealized_total = 0.0
        if isinstance(accounts, Iterable):
            for entry in accounts:
                if isinstance(entry, Mapping):
                    unrealized_total += self._to_float(entry.get("unrealized_pnl"))

        equity = nav + unrealized_total

        with self._lock:
            with self._connect() as conn:
                last_row = conn.execute(
                    "SELECT timestamp, high_water_mark FROM nav_history ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if last_row is not None:
                    last_ts = self._parse_datetime(last_row[0])
                    if generated_at - last_ts < self.min_interval:
                        # Avoid spamming the history store with near-duplicate points.
                        return
                    high_water_mark = float(last_row[1])
                else:
                    high_water_mark = nav

                high_water_mark = max(high_water_mark, nav)
                drawdown = 0.0
                if high_water_mark > 0:
                    drawdown = max(0.0, (high_water_mark - nav) / high_water_mark)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO nav_history (
                        timestamp, nav, equity, realized, unrealized, gross_exposure,
                        net_exposure, drawdown, high_water_mark
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generated_at.isoformat(),
                        nav,
                        equity,
                        realized,
                        unrealized_total,
                        gross,
                        net,
                        drawdown,
                        high_water_mark,
                    ),
                )

    async def record_async(self, snapshot: Mapping[str, Any]) -> None:
        await asyncio.to_thread(self.record, snapshot)

    def fetch_range(self, window: str) -> Dict[str, Any]:
        """Return NAV and equity history for ``window``."""

        delta = self._parse_window(window)
        now = datetime.now(timezone.utc)
        since = now - delta
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM nav_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (since.isoformat(),),
                ).fetchall()

        series: List[Dict[str, Any]] = []
        for row in rows:
            series.append(
                {
                    "timestamp": row["timestamp"],
                    "nav": float(row["nav"]),
                    "equity": float(row["equity"]),
                    "realized": float(row["realized"]),
                    "unrealized": float(row["unrealized"]),
                    "gross_exposure": float(row["gross_exposure"]),
                    "net_exposure": float(row["net_exposure"]),
                    "drawdown": float(row["drawdown"]),
                }
            )

        summary = self._summarise_series(series)
        cashflow_summary = self._summarise_cashflows(since)
        summary.update(cashflow_summary)

        return {
            "range": window,
            "series": series,
            "summary": summary,
            "updated_at": series[-1]["timestamp"] if series else None,
        }

    async def fetch_range_async(self, window: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.fetch_range, window)

    def add_cashflow(
        self,
        *,
        flow_type: str,
        amount: float,
        currency: str,
        timestamp: Optional[datetime] = None,
        account: Optional[str] = None,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a deposit or withdrawal entry."""

        flow_type_normalized = flow_type.strip().lower()
        if flow_type_normalized not in {"deposit", "withdrawal"}:
            raise ValueError("flow_type must be 'deposit' or 'withdrawal'")
        numeric_amount = float(amount)
        if numeric_amount <= 0:
            raise ValueError("amount must be greater than zero")
        currency_value = currency.strip().upper() or "USDT"
        timestamp_value = self._parse_datetime(timestamp) if timestamp else datetime.now(timezone.utc)

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO cashflows (timestamp, type, amount, currency, account, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp_value.isoformat(),
                        flow_type_normalized,
                        numeric_amount,
                        currency_value,
                        account.strip() if account else None,
                        note.strip() if note else None,
                    ),
                )
                row_id = cursor.lastrowid

        return {
            "id": row_id,
            "timestamp": timestamp_value.isoformat(),
            "type": flow_type_normalized,
            "amount": numeric_amount,
            "currency": currency_value,
            "account": account.strip() if account else None,
            "note": note.strip() if note else None,
            "signed_amount": numeric_amount if flow_type_normalized == "deposit" else -numeric_amount,
        }

    async def add_cashflow_async(self, **kwargs: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(self.add_cashflow, **kwargs)

    def list_cashflows(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent cash flow events."""

        limit_value = max(1, int(limit))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, type, amount, currency, account, note
                    FROM cashflows
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit_value,),
                ).fetchall()

        records: List[Dict[str, Any]] = []
        for row in rows:
            flow_type = str(row["type"])
            amount = float(row["amount"])
            signed = amount if flow_type == "deposit" else -amount
            records.append(
                {
                    "id": int(row["id"]),
                    "timestamp": str(row["timestamp"]),
                    "type": flow_type,
                    "amount": amount,
                    "currency": str(row["currency"]),
                    "account": row["account"],
                    "note": row["note"],
                    "signed_amount": signed,
                }
            )
        return records

    async def list_cashflows_async(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_cashflows, limit=limit)

    def build_portfolio_report(self, window: str) -> tuple[str, bytes]:
        """Return a CSV report containing NAV history for ``window``."""

        snapshot = self.fetch_range(window)
        series: List[Mapping[str, Any]] = snapshot.get("series", [])  # type: ignore[assignment]
        summary = snapshot.get("summary", {})
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"portfolio_report_{window}_{now}.csv"

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Generated at", datetime.now(timezone.utc).isoformat()])
        writer.writerow(["Window", window])
        writer.writerow([])
        writer.writerow(["Summary"])
        for key in (
            "nav_start",
            "nav_end",
            "nav_change",
            "nav_change_pct",
            "equity_start",
            "equity_end",
            "equity_change",
            "equity_change_pct",
            "realized_start",
            "realized_end",
            "realized_change",
            "max_drawdown",
            "cashflow_deposits",
            "cashflow_withdrawals",
            "cashflow_net",
        ):
            writer.writerow([key, summary.get(key)])

        writer.writerow([])
        writer.writerow([
            "timestamp",
            "nav",
            "equity",
            "realized",
            "unrealized",
            "gross_exposure",
            "net_exposure",
            "drawdown",
        ])
        for entry in series:
            writer.writerow(
                [
                    entry.get("timestamp"),
                    entry.get("nav"),
                    entry.get("equity"),
                    entry.get("realized"),
                    entry.get("unrealized"),
                    entry.get("gross_exposure"),
                    entry.get("net_exposure"),
                    entry.get("drawdown"),
                ]
            )

        contents = buffer.getvalue().encode("utf-8")
        return filename, contents

    async def build_portfolio_report_async(self, window: str) -> tuple[str, bytes]:
        return await asyncio.to_thread(self.build_portfolio_report, window)

    # ------------------------------------------------------------------
    # internal helpers

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nav_history (
                    timestamp TEXT PRIMARY KEY,
                    nav REAL NOT NULL,
                    equity REAL NOT NULL,
                    realized REAL NOT NULL,
                    unrealized REAL NOT NULL,
                    gross_exposure REAL NOT NULL,
                    net_exposure REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    high_water_mark REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cashflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL CHECK (type IN ('deposit', 'withdrawal')),
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    account TEXT,
                    note TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cashflows_timestamp ON cashflows(timestamp)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return datetime.now(timezone.utc)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    def _parse_window(self, window: str) -> timedelta:
        key = window.strip().lower()
        if key not in self._WINDOWS:
            raise ValueError(f"Unsupported range '{window}'. Valid options: {', '.join(self._WINDOWS)}")
        return self._WINDOWS[key]

    @staticmethod
    def _summarise_series(series: List[Mapping[str, Any]]) -> Dict[str, Any]:
        if not series:
            return {
                "nav_start": 0.0,
                "nav_end": 0.0,
                "nav_change": 0.0,
                "nav_change_pct": 0.0,
                "equity_start": 0.0,
                "equity_end": 0.0,
                "equity_change": 0.0,
                "equity_change_pct": 0.0,
                "realized_start": 0.0,
                "realized_end": 0.0,
                "realized_change": 0.0,
                "max_drawdown": 0.0,
                "range_start": None,
                "range_end": None,
            }

        first = series[0]
        last = series[-1]
        nav_start = float(first.get("nav", 0.0))
        nav_end = float(last.get("nav", 0.0))
        equity_start = float(first.get("equity", 0.0))
        equity_end = float(last.get("equity", 0.0))
        realized_start = float(first.get("realized", 0.0))
        realized_end = float(last.get("realized", 0.0))
        nav_change = nav_end - nav_start
        equity_change = equity_end - equity_start
        nav_change_pct = (nav_change / nav_start) if nav_start else 0.0
        equity_change_pct = (equity_change / equity_start) if equity_start else 0.0
        max_drawdown = max(float(entry.get("drawdown", 0.0)) for entry in series)

        return {
            "nav_start": nav_start,
            "nav_end": nav_end,
            "nav_change": nav_change,
            "nav_change_pct": nav_change_pct,
            "equity_start": equity_start,
            "equity_end": equity_end,
            "equity_change": equity_change,
            "equity_change_pct": equity_change_pct,
            "realized_start": realized_start,
            "realized_end": realized_end,
            "realized_change": realized_end - realized_start,
            "max_drawdown": max_drawdown,
            "range_start": first.get("timestamp"),
            "range_end": last.get("timestamp"),
        }

    def _summarise_cashflows(self, since: datetime) -> Dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT type, SUM(amount) FROM cashflows WHERE timestamp >= ? GROUP BY type",
                    (since.isoformat(),),
                ).fetchall()

        deposits = 0.0
        withdrawals = 0.0
        for row in rows:
            flow_type = str(row[0])
            total = self._to_float(row[1])
            if flow_type == "deposit":
                deposits = total
            elif flow_type == "withdrawal":
                withdrawals = total

        return {
            "cashflow_deposits": deposits,
            "cashflow_withdrawals": withdrawals,
            "cashflow_net": deposits - withdrawals,
        }

