"""Repository helpers for loading performance history data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class PerformancePoint:
    """Representation of a historical balance entry."""

    date: str
    balance: float
    timestamp: Optional[str]

    def to_dict(self) -> dict[str, Optional[float | str]]:
        return {"date": self.date, "balance": self.balance, "timestamp": self.timestamp}


class PerformanceRepository:
    """Load performance history stored by :class:`PerformanceTracker`."""

    def __init__(self, base_directory: Path) -> None:
        self._base_directory = Path(base_directory)
        self._data_path = self._base_directory / "daily_balances.json"

    # ------------------------------------------------------------------
    # public API

    def get_portfolio_series(
        self, *, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[dict[str, Optional[float | str]]]:
        data = self._load()
        raw_series = data.get("portfolio", [])
        normalised = self._normalise_series(raw_series)
        return self._filter_series(normalised, start=start, end=end)

    def get_account_series(
        self, account_name: str, *, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[dict[str, Optional[float | str]]]:
        if not account_name:
            raise ValueError("account_name is required")
        data = self._load()
        accounts = data.get("accounts", {})
        if not isinstance(accounts, Mapping):
            accounts = {}
        history = accounts.get(account_name)
        if history is None:
            raise KeyError(account_name)
        normalised = self._normalise_series(history)
        return self._filter_series(normalised, start=start, end=end)

    # ------------------------------------------------------------------
    # internal helpers

    def _load(self) -> Mapping[str, object]:
        if not self._data_path.exists():
            return {"portfolio": [], "accounts": {}}
        try:
            payload = json.loads(self._data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"portfolio": [], "accounts": {}}
        portfolio = payload.get("portfolio")
        accounts = payload.get("accounts")
        if not isinstance(portfolio, list):
            portfolio = []
        if not isinstance(accounts, MutableMapping):
            accounts = {}
        return {"portfolio": list(portfolio), "accounts": dict(accounts)}

    def _normalise_series(self, entries: Iterable[object]) -> List[dict[str, Optional[float | str]]]:
        series: List[dict[str, Optional[float | str]]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            date_value = entry.get("date")
            balance_value = entry.get("balance")
            if date_value is None or balance_value is None:
                continue
            try:
                balance = float(balance_value)
            except (TypeError, ValueError):
                continue
            date_str = self._normalise_date_value(date_value)
            if date_str is None:
                continue
            timestamp_value = entry.get("timestamp")
            timestamp = self._normalise_timestamp(timestamp_value)
            point = PerformancePoint(date=date_str, balance=balance, timestamp=timestamp)
            series.append(point.to_dict())
        series.sort(key=lambda item: item["date"])
        return series

    def _filter_series(
        self,
        series: List[dict[str, Optional[float | str]]],
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[dict[str, Optional[float | str]]]:
        start_date = self._parse_date(start)
        end_date = self._parse_date(end)
        if start_date and end_date and start_date > end_date:
            raise ValueError("start date cannot be after end date")
        if not start_date and not end_date:
            return list(series)
        filtered: List[dict[str, Optional[float | str]]] = []
        for entry in series:
            date_str = entry.get("date")
            if not isinstance(date_str, str):
                continue
            try:
                entry_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            if start_date and entry_date < start_date:
                continue
            if end_date and entry_date > end_date:
                continue
            filtered.append(dict(entry))
        return filtered

    @staticmethod
    def _normalise_date_value(value: object) -> Optional[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                # validate the format
                date.fromisoformat(value)
            except ValueError:
                return None
            return value
        return None

    @staticmethod
    def _normalise_timestamp(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            trimmed = value.strip()
            return trimmed or None
        return str(value)

    @staticmethod
    def _parse_date(raw: Optional[str]) -> Optional[date]:
        if raw is None:
            return None
        candidate = raw.strip()
        if not candidate:
            return None
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            try:
                normalised = candidate.replace("Z", "+00:00")
                dt_value = datetime.fromisoformat(normalised)
            except ValueError as exc:
                raise ValueError(f"Invalid date value '{raw}'") from exc
            return dt_value.date()
