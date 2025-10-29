import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.performance import PerformanceTracker


def test_performance_tracker_records_daily_snapshots(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)

    first_day = datetime(2024, 3, 1, 21, 0, tzinfo=timezone.utc)
    summary = tracker.record(
        generated_at=first_day,
        portfolio_balance=10_000.0,
        account_balances={"Demo": 10_000.0},
    )
    assert summary["portfolio"]["latest_snapshot"]["balance"] == 10_000.0
    assert summary["accounts"]["Demo"]["latest_snapshot"]["balance"] == 10_000.0

    # A call before the next recording window should not add a new snapshot
    midday = datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc)
    summary_mid = tracker.record(
        generated_at=midday,
        portfolio_balance=10_500.0,
        account_balances={"Demo": 10_500.0},
    )
    assert summary_mid["portfolio"]["daily"] is None

    second_day = datetime(2024, 3, 2, 21, 5, tzinfo=timezone.utc)
    summary_second = tracker.record(
        generated_at=second_day,
        portfolio_balance=11_200.0,
        account_balances={"Demo": 11_200.0},
    )
    portfolio_daily = summary_second["portfolio"]["daily"]
    assert portfolio_daily is not None
    assert pytest.approx(portfolio_daily["pnl"]) == 1_200.0
    assert pytest.approx(portfolio_daily["pct_change"]) == 12.0
    assert portfolio_daily["since"] == "2024-03-01"
    assert pytest.approx(portfolio_daily["reference_balance"]) == 10_000.0

    daily_change = summary_second["accounts"]["Demo"]["daily"]
    assert daily_change is not None
    assert pytest.approx(daily_change["pnl"]) == 1_200.0
    assert pytest.approx(daily_change["pct_change"]) == 12.0
    assert daily_change["since"] == "2024-03-01"
    assert pytest.approx(daily_change["reference_balance"]) == 10_000.0


def test_performance_tracker_handles_missing_history(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)
    summary = tracker.record(
        generated_at=datetime(2024, 3, 3, 21, 0, tzinfo=timezone.utc),
        portfolio_balance=5_000.0,
        account_balances={"Demo": 5_000.0},
    )
    assert summary["portfolio"]["daily"] is None
    assert summary["accounts"]["Demo"]["weekly"] is None


def test_performance_tracker_handles_zero_reference_balance(tmp_path: Path) -> None:
    tracker = PerformanceTracker(tmp_path)

    first_day = datetime(2024, 4, 1, 21, 0, tzinfo=timezone.utc)
    tracker.record(
        generated_at=first_day,
        portfolio_balance=0.0,
        account_balances={"Demo": 0.0},
    )

    second_day = datetime(2024, 4, 2, 21, 0, tzinfo=timezone.utc)
    summary = tracker.record(
        generated_at=second_day,
        portfolio_balance=500.0,
        account_balances={"Demo": 500.0},
    )

    portfolio_daily = summary["portfolio"]["daily"]
    assert portfolio_daily is not None
    assert pytest.approx(portfolio_daily["pnl"]) == 500.0
    assert portfolio_daily["pct_change"] is None

    account_daily = summary["accounts"]["Demo"]["daily"]
    assert account_daily is not None
    assert pytest.approx(account_daily["pnl"]) == 500.0
    assert account_daily["pct_change"] is None
