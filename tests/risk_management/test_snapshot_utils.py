import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_management.snapshot_utils import build_presentable_snapshot


def test_snapshot_utils_preserves_position_fields() -> None:
    snapshot = {
        "generated_at": "2024-03-02T00:00:00+00:00",
        "accounts": [
            {
                "name": "Demo",
                "balance": 1_000,
                "positions": [
                    {
                        "symbol": "BTC/USDT",
                        "side": "long",
                        "notional": 500,
                        "entry_price": 100,
                        "mark_price": 110,
                        "liquidation_price": 80,
                        "wallet_exposure_pct": 0.5,
                        "unrealized_pnl": 50,
                        "daily_realized_pnl": 10,
                        "max_drawdown_pct": 0.2,
                        "take_profit_price": 120,
                        "stop_loss_price": 90,
                    }
                ],
                "daily_realized_pnl": 10,
                "open_orders": [],
            }
        ],
        "alert_thresholds": {},
        "notification_channels": [],
        "account_stop_losses": {
            "Demo": {
                "threshold_pct": 5.0,
                "baseline_balance": 1_000.0,
                "current_balance": 950.0,
                "current_drawdown_pct": 0.05,
                "triggered": False,
                "active": True,
                "triggered_at": None,
            }
        },
        "performance": {
            "portfolio": {
                "current_balance": 1_000.0,
                "latest_snapshot": {
                    "date": "2024-03-01",
                    "balance": 980.0,
                    "timestamp": "2024-03-01T21:00:00+00:00",
                },
                "daily": {
                    "pnl": 20.0,
                    "since": "2024-03-01",
                    "reference_balance": 960.0,
                },
            },
            "accounts": {
                "Demo": {
                    "current_balance": 1_000.0,
                    "latest_snapshot": {
                        "date": "2024-03-01",
                        "balance": 985.0,
                        "timestamp": "2024-03-01T21:00:00+00:00",
                    },
                    "daily": {
                        "pnl": 15.0,
                        "since": "2024-03-01",
                        "reference_balance": 985.0,
                    },
                }
            },
        },
    }

    view = build_presentable_snapshot(snapshot)

    assert view["accounts"][0]["positions"][0]["daily_realized_pnl"] == 10
    assert view["accounts"][0]["positions"][0]["liquidation_price"] == 80
    assert view["accounts"][0]["positions"][0]["take_profit_price"] == 120
    assert view["accounts"][0]["positions"][0]["stop_loss_price"] == 90
    assert view["accounts"][0]["positions"][0]["max_drawdown_pct"] == 0.2

    stop_loss = view["accounts"][0]["stop_loss"]
    assert stop_loss["threshold_pct"] == 5.0
    assert stop_loss["current_balance"] == 950.0

    performance = view["accounts"][0]["performance"]
    assert performance["daily"] == 15.0
    assert performance["since"]["daily"] == "2024-03-01"

    portfolio_perf = view["portfolio"]["performance"]
    assert portfolio_perf["daily"] == 20.0
    assert portfolio_perf["latest_snapshot"]["balance"] == 980.0

    assert view["account_stop_losses"]["Demo"]["current_balance"] == 950.0
