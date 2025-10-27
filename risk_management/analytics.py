"""Utilities for tracking and summarising portfolio time series analytics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _to_datetime(value: str) -> Optional[datetime]:
    """Return a timezone aware ``datetime`` parsed from ``value``."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: object) -> float:
    """Convert ``value`` to ``float`` returning ``0.0`` on failure."""

    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class AccountPoint:
    """Time series datapoint for a single account."""

    balance: float
    unrealized: float
    gross: float
    net: float


@dataclass
class PortfolioPoint:
    """Aggregated portfolio metrics captured at ``timestamp``."""

    timestamp: datetime
    nav: float
    unrealized: float
    gross: float
    net: float
    accounts: Dict[str, AccountPoint]


TimeframeDefinition = Tuple[str, timedelta, str]


TIMEFRAMES: Tuple[TimeframeDefinition, ...] = (
    ("1h", timedelta(hours=1), "Last 1 hour"),
    ("6h", timedelta(hours=6), "Last 6 hours"),
    ("24h", timedelta(hours=24), "Last 24 hours"),
    ("7d", timedelta(days=7), "Last 7 days"),
    ("30d", timedelta(days=30), "Last 30 days"),
)


_TIMEFRAME_LOOKUP: Dict[str, TimeframeDefinition] = {definition[0]: definition for definition in TIMEFRAMES}


class PortfolioHistory:
    """Maintain a rolling history of portfolio snapshots for analytics."""

    def __init__(
        self,
        *,
        max_age: timedelta = timedelta(days=90),
        max_points: int = 50_000,
        min_interval: timedelta = timedelta(seconds=60),
    ) -> None:
        self._points: Deque[PortfolioPoint] = deque()
        self._max_age = max_age
        self._max_points = max_points
        self._min_interval = min_interval

    # ------------------------------------------------------------------
    # Recording snapshots
    # ------------------------------------------------------------------
    def record_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Store ``snapshot`` (presentable view) in the rolling history."""

        timestamp_raw = snapshot.get("generated_at")
        if not isinstance(timestamp_raw, str):
            return
        timestamp = _to_datetime(timestamp_raw)
        if timestamp is None:
            return

        portfolio = snapshot.get("portfolio") if isinstance(snapshot, Mapping) else None
        if not isinstance(portfolio, Mapping):
            return

        nav = _safe_float(portfolio.get("balance"))
        gross = _safe_float(portfolio.get("gross_exposure"))
        net = _safe_float(portfolio.get("net_exposure"))

        accounts_raw = snapshot.get("accounts") if isinstance(snapshot, Mapping) else None
        accounts: Dict[str, AccountPoint] = {}
        total_unrealized = 0.0
        if isinstance(accounts_raw, Iterable):
            for entry in accounts_raw:
                if not isinstance(entry, Mapping):
                    continue
                name_raw = entry.get("name")
                if not isinstance(name_raw, str) or not name_raw.strip():
                    continue
                name = name_raw.strip()
                balance = _safe_float(entry.get("balance"))
                unrealized = _safe_float(entry.get("unrealized_pnl"))
                gross_notional = _safe_float(entry.get("gross_exposure_notional"))
                net_notional = _safe_float(entry.get("net_exposure_notional"))
                total_unrealized += unrealized
                accounts[name] = AccountPoint(
                    balance=balance,
                    unrealized=unrealized,
                    gross=gross_notional,
                    net=net_notional,
                )

        point = PortfolioPoint(
            timestamp=timestamp,
            nav=nav,
            unrealized=total_unrealized,
            gross=gross,
            net=net,
            accounts=accounts,
        )

        if self._points:
            last = self._points[-1]
            if timestamp <= last.timestamp:
                if timestamp == last.timestamp:
                    self._points[-1] = point
                return
            if timestamp - last.timestamp < self._min_interval:
                self._points[-1] = point
            else:
                self._points.append(point)
        else:
            self._points.append(point)

        self._prune(timestamp)

    def _prune(self, now: datetime) -> None:
        cutoff = now - self._max_age
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()
        while len(self._points) > self._max_points:
            self._points.popleft()

    # ------------------------------------------------------------------
    # Timeframe helpers
    # ------------------------------------------------------------------
    def available_timeframes(self) -> List[Dict[str, object]]:
        return [
            {"id": identifier, "label": label}
            for identifier, _duration, label in TIMEFRAMES
        ]

    def _resolve_timeframe(self, timeframe: str) -> TimeframeDefinition:
        try:
            return _TIMEFRAME_LOOKUP[timeframe]
        except KeyError as exc:
            raise ValueError(f"Unsupported timeframe '{timeframe}'.") from exc

    def _points_for_timeframe(self, duration: timedelta) -> List[PortfolioPoint]:
        if not self._points:
            return []
        end = self._points[-1].timestamp
        cutoff = end - duration
        relevant: List[PortfolioPoint] = []
        for point in reversed(self._points):
            if point.timestamp < cutoff:
                break
            relevant.append(point)
        if not relevant:
            return [self._points[-1]]
        return list(reversed(relevant))

    # ------------------------------------------------------------------
    # Summary builders
    # ------------------------------------------------------------------
    def portfolio_summary(
        self, timeframe: str, *, include_series: bool = True
    ) -> Dict[str, object]:
        identifier, duration, label = self._resolve_timeframe(timeframe)
        points = self._points_for_timeframe(duration)
        summary, series = self._summarise_points(points)
        payload: Dict[str, object] = {
            "timeframe": identifier,
            "label": label,
            "summary": summary,
        }
        if include_series:
            payload["series"] = series
        return payload

    def account_summary(
        self, account_name: str, timeframe: str, *, include_series: bool = False
    ) -> Optional[Dict[str, object]]:
        identifier, duration, label = self._resolve_timeframe(timeframe)
        points = self._points_for_timeframe(duration)
        account_points = [
            (point.timestamp, point.accounts.get(account_name)) for point in points
        ]
        filtered = [entry for entry in account_points if entry[1] is not None]
        if not filtered:
            return None
        summary, series = self._summarise_account_points(filtered)
        payload: Dict[str, object] = {
            "timeframe": identifier,
            "label": label,
            "summary": summary,
        }
        if include_series:
            payload["series"] = series
        return payload

    def portfolio_overview(self, timeframes: Sequence[str]) -> Dict[str, Dict[str, object]]:
        overview: Dict[str, Dict[str, object]] = {}
        for timeframe in timeframes:
            payload = self.portfolio_summary(timeframe, include_series=False)
            overview[timeframe] = {
                "label": payload.get("label", timeframe),
                "summary": payload.get("summary", self._empty_summary()),
            }
        return overview

    def account_overview(
        self, account_name: str, timeframes: Sequence[str]
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        for timeframe in timeframes:
            summary = self.account_summary(account_name, timeframe, include_series=False)
            if summary is not None:
                results[timeframe] = {
                    "label": summary.get("label", timeframe),
                    "summary": summary.get("summary", self._empty_summary()),
                }
        return results

    # ------------------------------------------------------------------
    # Internal summarisation helpers
    # ------------------------------------------------------------------
    def _summarise_points(
        self, points: Sequence[PortfolioPoint]
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        if not points:
            return self._empty_summary(), []
        start = points[0]
        end = points[-1]
        nav_change = end.nav - start.nav
        unrealized_change = end.unrealized - start.unrealized
        realized_change = nav_change - unrealized_change
        nav_high = max(point.nav for point in points)
        nav_low = min(point.nav for point in points)
        drawdown = 0.0
        if nav_high > 0:
            drawdown = max(0.0, (nav_high - end.nav) / nav_high)
        summary = {
            "start": start.timestamp.isoformat(),
            "end": end.timestamp.isoformat(),
            "nav": end.nav,
            "baseline": start.nav,
            "nav_change": nav_change,
            "nav_change_pct": nav_change / start.nav if start.nav else 0.0,
            "realized_change": realized_change,
            "unrealized_change": unrealized_change,
            "drawdown_pct": drawdown,
            "nav_high": nav_high,
            "nav_low": nav_low,
            "points": len(points),
        }
        series: List[Dict[str, object]] = []
        baseline_nav = start.nav or end.nav or 1.0
        baseline_unrealized = start.unrealized
        for point in points:
            pnl = point.nav - baseline_nav
            unrealized_delta = point.unrealized - baseline_unrealized
            realized_delta = pnl - unrealized_delta
            series.append(
                {
                    "timestamp": point.timestamp.isoformat(),
                    "nav": point.nav,
                    "pnl": pnl,
                    "realized": realized_delta,
                    "unrealized": unrealized_delta,
                }
            )
        return summary, series

    def _summarise_account_points(
        self, points: Sequence[Tuple[datetime, AccountPoint]]
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        if not points:
            return self._empty_summary(), []
        start_ts, start_point = points[0]
        end_ts, end_point = points[-1]
        nav_change = end_point.balance - start_point.balance
        unrealized_change = end_point.unrealized - start_point.unrealized
        realized_change = nav_change - unrealized_change
        high = max(point.balance for _, point in points)
        low = min(point.balance for _, point in points)
        drawdown = 0.0
        if high > 0:
            drawdown = max(0.0, (high - end_point.balance) / high)
        summary = {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat(),
            "nav": end_point.balance,
            "baseline": start_point.balance,
            "nav_change": nav_change,
            "nav_change_pct": nav_change / start_point.balance if start_point.balance else 0.0,
            "realized_change": realized_change,
            "unrealized_change": unrealized_change,
            "drawdown_pct": drawdown,
            "nav_high": high,
            "nav_low": low,
            "points": len(points),
        }
        series: List[Dict[str, object]] = []
        baseline_nav = start_point.balance or end_point.balance or 1.0
        baseline_unrealized = start_point.unrealized
        for timestamp, point in points:
            pnl = point.balance - baseline_nav
            unrealized_delta = point.unrealized - baseline_unrealized
            realized_delta = pnl - unrealized_delta
            series.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "nav": point.balance,
                    "pnl": pnl,
                    "realized": realized_delta,
                    "unrealized": unrealized_delta,
                }
            )
        return summary, series

    @staticmethod
    def _empty_summary() -> Dict[str, object]:
        return {
            "start": None,
            "end": None,
            "nav": 0.0,
            "baseline": 0.0,
            "nav_change": 0.0,
            "nav_change_pct": 0.0,
            "realized_change": 0.0,
            "unrealized_change": 0.0,
            "drawdown_pct": 0.0,
            "nav_high": 0.0,
            "nav_low": 0.0,
            "points": 0,
        }


__all__ = [
    "PortfolioHistory",
    "TIMEFRAMES",
]

