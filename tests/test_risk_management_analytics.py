from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from passivbot.risk_management.analytics import PortfolioHistory


def _snapshot(at: datetime, balance: float, unrealized: float) -> dict:
    return {
        "generated_at": at.isoformat(),
        "portfolio": {
            "balance": balance,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        },
        "accounts": [
            {
                "name": "Primary",
                "balance": balance,
                "unrealized_pnl": unrealized,
                "gross_exposure_notional": 0.0,
                "net_exposure_notional": 0.0,
            }
        ],
    }


def test_portfolio_history_summaries() -> None:
    history = PortfolioHistory(min_interval=timedelta(seconds=0))
    base = datetime(2025, 5, 1, tzinfo=timezone.utc)
    history.record_snapshot(_snapshot(base - timedelta(hours=6), 1_000.0, 50.0))
    history.record_snapshot(_snapshot(base - timedelta(hours=3), 1_100.0, 40.0))
    history.record_snapshot(_snapshot(base, 1_200.0, 70.0))

    summary_payload = history.portfolio_summary("6h", include_series=False)
    summary = summary_payload["summary"]
    assert pytest.approx(summary["nav_change"], rel=1e-9) == 200.0
    assert pytest.approx(summary["unrealized_change"], rel=1e-9) == 20.0
    assert pytest.approx(summary["realized_change"], rel=1e-9) == 180.0
    assert summary["points"] == 3

    account_payload = history.account_summary("Primary", "6h", include_series=False)
    assert account_payload is not None
    account_summary = account_payload["summary"]
    assert pytest.approx(account_summary["nav_change"], rel=1e-9) == 200.0
    assert pytest.approx(account_summary["unrealized_change"], rel=1e-9) == 20.0

    overview = history.portfolio_overview(["1h", "6h"])
    assert overview["6h"]["label"] == "Last 6 hours"
    assert "summary" in overview["6h"]

    account_overview = history.account_overview("Primary", ["1h", "6h"])
    assert account_overview["1h"]["label"] == "Last 1 hour"
    assert "summary" in account_overview["6h"]
