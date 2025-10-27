"""Utilities for storing and retrieving generated account reports."""

from __future__ import annotations

import asyncio
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional



@dataclass
class StoredReport:
    """Metadata describing a generated report on disk."""

    account: str
    report_id: str
    path: Path
    created_at: datetime
    size: int

    def to_view(self) -> dict[str, Any]:
        """Return a JSON serialisable representation."""

        return {
            "account": self.account,
            "report_id": self.report_id,
            "filename": self.path.name,
            "created_at": self.created_at.replace(tzinfo=timezone.utc).isoformat(),
            "size": self.size,
        }


class ReportManager:
    """Persist generated reports to disk."""

    _FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")

    def __init__(self, base_directory: Path) -> None:
        self.base_directory = base_directory
        self.base_directory.mkdir(parents=True, exist_ok=True)

    async def create_account_report(
        self,
        account_name: str,
        snapshot: Mapping[str, Any],
        *,
        analytics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> StoredReport:
        """Generate and store a CSV report for ``account_name``."""

        return await asyncio.to_thread(
            self._create_account_report_sync,
            account_name,
            snapshot,
            analytics,
        )

    async def list_reports(self, account_name: str) -> list[StoredReport]:
        """Return stored reports for ``account_name`` sorted by newest first."""

        return await asyncio.to_thread(self._list_reports_sync, account_name)

    async def get_report_path(self, account_name: str, report_id: str) -> Optional[Path]:
        """Return the path to a stored report, if it exists."""

        return await asyncio.to_thread(self._get_report_path_sync, account_name, report_id)

    # --- internal helpers -------------------------------------------------

    def _create_account_report_sync(
        self,
        account_name: str,
        snapshot: Mapping[str, Any],
        analytics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> StoredReport:
        account = self._extract_account(snapshot.get("accounts"), account_name)
        if account is None:
            raise ValueError(
                f"Account '{account_name}' is not available in the latest snapshot."
            )

        generated_at = snapshot.get("generated_at")
        generated_at_dt: Optional[datetime] = None
        if isinstance(generated_at, str):
            try:
                generated_at_dt = datetime.fromisoformat(generated_at)
            except ValueError:
                generated_at_dt = None

        timestamp = datetime.now(timezone.utc)
        report_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        account_dir = self._account_directory(account_name)
        account_dir.mkdir(parents=True, exist_ok=True)
        file_path = account_dir / f"{report_id}.csv"

        portfolio = snapshot.get("portfolio") if isinstance(snapshot, Mapping) else None
        alerts = snapshot.get("alerts") if isinstance(snapshot, Mapping) else None

        rows: list[list[str]] = []
        rows.extend(
            self._build_summary_rows(
                account_name,
                account,
                portfolio,
                alerts,
                analytics=analytics,
            )
        )
        rows.extend(self._build_exposure_rows(account.get("symbol_exposures")))
        rows.extend(self._build_positions_rows(account.get("positions")))
        rows.extend(self._build_orders_rows(account.get("orders")))

        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if generated_at_dt is not None:
                writer.writerow(["Snapshot generated at", generated_at_dt.isoformat()])
                writer.writerow([])
            for row in rows:
                writer.writerow(row)

        stat = file_path.stat()
        return StoredReport(
            account=account_name,
            report_id=report_id,
            path=file_path,
            created_at=timestamp,
            size=stat.st_size,
        )

    def _list_reports_sync(self, account_name: str) -> list[StoredReport]:
        account_dir = self._account_directory(account_name)
        if not account_dir.exists():
            return []
        reports: list[StoredReport] = []
        for path in account_dir.glob("*.csv"):
            report_id = path.stem
            created_at = self._parse_report_timestamp(report_id) or datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )
            reports.append(
                StoredReport(
                    account=account_name,
                    report_id=report_id,
                    path=path,
                    created_at=created_at,
                    size=path.stat().st_size,
                )
            )
        reports.sort(key=lambda item: item.created_at, reverse=True)
        return reports

    def _get_report_path_sync(self, account_name: str, report_id: str) -> Optional[Path]:
        account_dir = self._account_directory(account_name)
        if not account_dir.exists():
            return None
        filename = f"{report_id}.csv"
        candidate = account_dir / filename
        return candidate if candidate.exists() else None

    def _account_directory(self, account_name: str) -> Path:
        safe_name = self._FILENAME_PATTERN.sub("_", account_name.strip() or "account")
        return self.base_directory / safe_name

    @staticmethod
    def _extract_account(
        accounts: Any, account_name: str
    ) -> Optional[Mapping[str, Any]]:
        if not isinstance(accounts, Iterable):
            return None
        for entry in accounts:
            if isinstance(entry, Mapping) and entry.get("name") == account_name:
                return entry
        return None

    @staticmethod
    def _parse_report_timestamp(report_id: str) -> Optional[datetime]:
        for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ"):
            try:
                return datetime.strptime(report_id, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    # --- CSV builders -----------------------------------------------------

    def _build_summary_rows(
        self,
        account_name: str,
        account: Mapping[str, Any],
        portfolio: Optional[Mapping[str, Any]],
        alerts: Optional[Iterable[str]],
        *,
        analytics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> list[list[str]]:
        balance = account.get("balance", 0.0)
        gross_notional = account.get("gross_exposure_notional", 0.0)
        gross_pct = account.get("gross_exposure", 0.0)
        net_notional = account.get("net_exposure_notional", 0.0)
        net_pct = account.get("net_exposure", 0.0)
        unrealized = account.get("unrealized_pnl", 0.0)
        positions_count = len(account.get("positions", []) or [])
        orders_count = len(account.get("orders", []) or [])
        alerts_summary = (
            ", ".join(str(alert) for alert in alerts) if alerts else "None"
        )
        portfolio_balance = (
            portfolio.get("balance") if isinstance(portfolio, Mapping) else None
        )
        balance_share = (
            balance / portfolio_balance if portfolio_balance else None
        )
        rows: list[list[str]] = [
            [
                "Account",
                "Balance",
                "Gross Exposure",
                "Gross %",
                "Net Exposure",
                "Net %",
                "Unrealized PnL",
                "Positions",
                "Orders",
                "Alerts",
                "Portfolio Share",
            ],
            [
                account_name,
                self._format_currency(balance),
                self._format_currency(gross_notional),
                self._format_pct(gross_pct),
                self._format_currency(net_notional),
                self._format_pct(net_pct),
                self._format_currency(unrealized),
                str(positions_count),
                str(orders_count),
                alerts_summary,
                self._format_pct(balance_share) if balance_share is not None else "-",
            ],
            [],
        ]
        if analytics:
            rows.append(
                [
                    "Performance period",
                    "PnL",
                    "PnL %",
                    "Realized",
                    "Unrealized",
                    "Drawdown",
                ]
            )
            for identifier, metrics in analytics.items():
                if not isinstance(metrics, Mapping):
                    continue
                summary = metrics.get("summary")
                summary_data = summary if isinstance(summary, Mapping) else metrics
                label = str(metrics.get("label", identifier))
                pnl = self._format_currency(summary_data.get("nav_change", 0.0))
                pnl_pct = self._format_pct(summary_data.get("nav_change_pct", 0.0))
                realized = self._format_currency(summary_data.get("realized_change", 0.0))
                unrealized_change = self._format_currency(summary_data.get("unrealized_change", 0.0))
                drawdown = self._format_pct(summary_data.get("drawdown_pct", 0.0))
                rows.append([label, pnl, pnl_pct, realized, unrealized_change, drawdown])
            rows.append([])
        return rows

    def _build_exposure_rows(self, exposures: Any) -> list[list[str]]:
        items = []
        if isinstance(exposures, Iterable):
            for entry in exposures:
                if not isinstance(entry, Mapping):
                    continue
                items.append(
                    [
                        entry.get("symbol", "-"),
                        self._format_currency(entry.get("gross_notional", 0.0)),
                        self._format_pct(entry.get("gross_pct", 0.0)),
                        self._format_currency(entry.get("net_notional", 0.0)),
                        self._format_pct(entry.get("net_pct", 0.0)),
                    ]
                )
        if not items:
            return [["Symbol exposures"], ["No symbol exposure data available"], []]
        header = [
            "Symbol",
            "Gross Exposure",
            "Gross %",
            "Net Exposure",
            "Net %",
        ]
        return [["Symbol exposures"], header, *items, []]

    def _build_positions_rows(self, positions: Any) -> list[list[str]]:
        items = []
        if isinstance(positions, Iterable):
            for position in positions:
                if not isinstance(position, Mapping):
                    continue
                items.append(
                    [
                        position.get("symbol", "-"),
                        position.get("side", "-"),
                        self._format_currency(position.get("notional", 0.0)),
                        self._format_pct(position.get("exposure", 0.0)),
                        self._format_currency(position.get("unrealized_pnl", 0.0)),
                        self._format_pct(position.get("pnl_pct", 0.0)),
                        self._format_price(position.get("entry_price")),
                        self._format_price(position.get("mark_price")),
                        self._format_price(position.get("liquidation_price")),
                        self._format_price(position.get("take_profit_price")),
                        self._format_price(position.get("stop_loss_price")),
                    ]
                )
        if not items:
            return [["Open positions"], ["No open positions"], []]
        header = [
            "Symbol",
            "Side",
            "Notional",
            "Exposure %",
            "Unrealized PnL",
            "PnL %",
            "Entry",
            "Mark",
            "Liquidation",
            "Take profit",
            "Stop loss",
        ]
        return [["Open positions"], header, *items, []]

    def _build_orders_rows(self, orders: Any) -> list[list[str]]:
        items = []
        if isinstance(orders, Iterable):
            for order in orders:
                if not isinstance(order, Mapping):
                    continue
                items.append(
                    [
                        order.get("order_id") or "-",
                        order.get("symbol", "-"),
                        order.get("side", "-"),
                        order.get("type", "-"),
                        self._format_price(order.get("price")),
                        str(order.get("amount", "-")),
                        str(order.get("remaining", "-")),
                        order.get("status", "-"),
                        "Yes" if order.get("reduce_only") else "No",
                        self._format_currency(order.get("notional")),
                        self._format_price(order.get("stop_price")),
                        order.get("created_at", "-"),
                    ]
                )
        if not items:
            return [["Open orders"], ["No open orders"], []]
        header = [
            "Order ID",
            "Symbol",
            "Side",
            "Type",
            "Price",
            "Amount",
            "Remaining",
            "Status",
            "Reduce only",
            "Notional",
            "Stop price",
            "Created",
        ]
        return [["Open orders"], header, *items, []]

    @staticmethod
    def _format_currency(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"${number:,.2f}"

    @staticmethod
    def _format_pct(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "0.00%"
        return f"{number * 100:.2f}%"

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"{number:,.4f}"


__all__ = ["ReportManager", "StoredReport"]

