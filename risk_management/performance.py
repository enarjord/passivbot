"""Helpers for tracking daily balance snapshots and performance summaries."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

from zoneinfo import ZoneInfo


@dataclass
class PerformanceSnapshot:
    """Description of the latest balance snapshot for an entity."""

    date: str
    balance: float
    timestamp: Optional[str]

    def to_dict(self) -> dict[str, Optional[float | str]]:
        return {"date": self.date, "balance": float(self.balance), "timestamp": self.timestamp}


class PerformanceTracker:
    """Persist daily balance snapshots and derive performance statistics."""

    def __init__(
        self,
        base_directory: Path,
        *,
        target_hour: int = 16,
        timezone_name: str = "America/New_York",
    ) -> None:
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self._data_path = self.base_directory / "daily_balances.json"
        self._lock = threading.Lock()
        self._target_time = time(hour=max(0, min(23, int(target_hour))), minute=0)
        self._tz = ZoneInfo(timezone_name)

    # ------------------------------------------------------------------
    # public API

    def record(
        self,
        *,
        generated_at: Optional[datetime],
        portfolio_balance: float,
        account_balances: Mapping[str, float],
    ) -> Dict[str, Mapping[str, object]]:
        """Persist daily balances when the 4pm ET window has been reached."""

        timestamp = self._normalise_timestamp(generated_at)
        with self._lock:
            data = self._load()
            changed = False

            if self._should_record(timestamp):
                record_date = self._to_local(timestamp).date()
                changed |= self._record_portfolio(data, record_date, portfolio_balance, timestamp)
                for name, balance in account_balances.items():
                    changed |= self._record_account(data, str(name), record_date, balance, timestamp)
                if changed:
                    data["updated_at"] = timestamp.isoformat()
                    self._save(data)

            summary = self._build_summary(data, portfolio_balance, account_balances)

        return summary

    # ------------------------------------------------------------------
    # internal helpers

    def _normalise_timestamp(self, value: Optional[datetime]) -> datetime:
        if value is None:
            value = datetime.now(timezone.utc)
        elif value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _to_local(self, timestamp: datetime) -> datetime:
        return timestamp.astimezone(self._tz)

    def _should_record(self, timestamp: datetime) -> bool:
        local = self._to_local(timestamp)
        local_time = local.timetz()
        return local_time >= self._target_time.replace(tzinfo=self._tz)

    def _load(self) -> Dict[str, object]:
        if not self._data_path.exists():
            return {"portfolio": [], "accounts": {}}
        try:
            payload = json.loads(self._data_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"portfolio": [], "accounts": {}}
        portfolio = payload.get("portfolio")
        accounts = payload.get("accounts")
        if not isinstance(portfolio, list):
            portfolio = []
        if not isinstance(accounts, MutableMapping):
            accounts = {}
        normalised_accounts: Dict[str, list[dict[str, object]]] = {}
        for name, history in accounts.items():
            if isinstance(history, list):
                normalised_accounts[str(name)] = [
                    entry
                    for entry in history
                    if isinstance(entry, Mapping) and "date" in entry and "balance" in entry
                ]
        return {
            "portfolio": [
                entry
                for entry in portfolio
                if isinstance(entry, Mapping) and "date" in entry and "balance" in entry
            ],
            "accounts": normalised_accounts,
        }

    def _save(self, data: Mapping[str, object]) -> None:
        try:
            self._data_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            # I/O errors should not prevent the realtime loop from continuing.
            pass

    def _record_portfolio(
        self,
        data: MutableMapping[str, object],
        record_date: date,
        balance: float,
        timestamp: datetime,
    ) -> bool:
        history: list[dict[str, object]] = data.setdefault("portfolio", [])  # type: ignore[assignment]
        return self._upsert_history_entry(history, record_date, balance, timestamp)

    def _record_account(
        self,
        data: MutableMapping[str, object],
        account_name: str,
        record_date: date,
        balance: float,
        timestamp: datetime,
    ) -> bool:
        accounts: MutableMapping[str, list[dict[str, object]]] = data.setdefault("accounts", {})  # type: ignore[assignment]
        history = accounts.setdefault(account_name, [])
        return self._upsert_history_entry(history, record_date, balance, timestamp)

    @staticmethod
    def _upsert_history_entry(
        history: list[dict[str, object]],
        record_date: date,
        balance: float,
        timestamp: datetime,
    ) -> bool:
        date_str = record_date.isoformat()
        iso_timestamp = timestamp.isoformat()
        for entry in history:
            if str(entry.get("date")) == date_str:
                entry["balance"] = float(balance)
                entry["timestamp"] = iso_timestamp
                return True
        history.append({"date": date_str, "balance": float(balance), "timestamp": iso_timestamp})
        history.sort(key=lambda item: item.get("date", ""))
        return True

    def _build_summary(
        self,
        data: Mapping[str, object],
        portfolio_balance: float,
        account_balances: Mapping[str, float],
    ) -> Dict[str, Mapping[str, object]]:
        portfolio_history = data.get("portfolio")
        summary = {
            "portfolio": self._summarise_history(
                portfolio_history if isinstance(portfolio_history, list) else [],
                portfolio_balance,
            ),
            "accounts": {},
        }
        accounts_history = data.get("accounts")
        if isinstance(accounts_history, Mapping):
            for name, balance in account_balances.items():
                history = accounts_history.get(str(name))
                if not isinstance(history, Iterable):
                    history_list: list[dict[str, object]] = []
                else:
                    history_list = [
                        entry
                        for entry in history
                        if isinstance(entry, Mapping) and "date" in entry and "balance" in entry
                    ]
                summary["accounts"][str(name)] = self._summarise_history(history_list, balance)
        return summary

    def _summarise_history(
        self, history: Iterable[Mapping[str, object]], current_balance: float
    ) -> Dict[str, object]:
        entries = [
            {
                "date": str(entry["date"]),
                "balance": float(entry.get("balance", 0.0)),
                "timestamp": entry.get("timestamp"),
            }
            for entry in history
        ]
        entries.sort(key=lambda item: item["date"])
        summary: Dict[str, object] = {
            "current_balance": float(current_balance),
            "latest_snapshot": None,
            "daily": None,
            "weekly": None,
            "monthly": None,
        }
        if not entries:
            return summary

        latest_entry = entries[-1]
        latest_date = date.fromisoformat(latest_entry["date"])
        summary["latest_snapshot"] = PerformanceSnapshot(
            date=latest_entry["date"],
            balance=float(latest_entry["balance"]),
            timestamp=latest_entry.get("timestamp"),
        ).to_dict()

        for label, days in ("daily", 1), ("weekly", 7), ("monthly", 30):
            reference = self._find_reference(entries, latest_date, days)
            if reference is None:
                continue
            reference_balance = float(reference["balance"])
            summary[label] = {
                "pnl": float(current_balance) - reference_balance,
                "since": reference["date"],
                "reference_balance": reference_balance,
            }

        return summary

    @staticmethod
    def _find_reference(
        entries: Iterable[Mapping[str, object]], anchor_date: date, days: int
    ) -> Optional[Mapping[str, object]]:
        target = anchor_date - timedelta(days=days)
        candidate: Optional[Mapping[str, object]] = None
        for entry in entries:
            entry_date = date.fromisoformat(str(entry["date"]))
            if entry_date <= target:
                if candidate is None or entry_date > date.fromisoformat(str(candidate["date"])):
                    candidate = entry
        return candidate

