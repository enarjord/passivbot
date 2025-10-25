import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.dashboard import build_dashboard
from risk_management.realtime import RealtimeDataFetcher
from risk_management.snapshot_utils import build_presentable_snapshot


class StubAccountClient:
    def __init__(self, name: str, balance: float, positions: list[dict[str, float]]) -> None:
        self.name = name
        self.balance = balance
        self.positions = positions
        self.closed = False

    async def fetch(self) -> dict:
        return {"name": self.name, "balance": self.balance, "positions": list(self.positions)}

    async def close(self) -> None:
        self.closed = True


class FailingAccountClient:
    def __init__(self) -> None:
        self.closed = False

    async def fetch(self) -> dict:
        raise RuntimeError("simulated failure")

    async def close(self) -> None:
        self.closed = True


def test_realtime_fetcher_combines_accounts() -> None:
    config = RealtimeConfig(
        accounts=[
            AccountConfig(name="Demo A", exchange="binance", credentials={}),
            AccountConfig(name="Demo B", exchange="okx", credentials={}),
        ],
        alert_thresholds={
            "wallet_exposure_pct": 0.5,
            "position_wallet_exposure_pct": 0.3,
            "max_drawdown_pct": 0.4,
            "loss_threshold_pct": -0.2,
        },
        notification_channels=["email:test@example.com"],
    )
    clients = [
        StubAccountClient(
            "Demo A",
            10_000,
            [
                {
                    "symbol": "BTCUSDT",
                    "side": "long",
                    "notional": 2_500,
                    "entry_price": 62_000,
                    "mark_price": 63_000,
                    "wallet_exposure_pct": 0.25,
                    "unrealized_pnl": 200,
                    "max_drawdown_pct": 0.1,
                }
            ],
        ),
        StubAccountClient("Demo B", 5_000, []),
    ]
    fetcher = RealtimeDataFetcher(config, account_clients=clients)
    snapshot = asyncio.run(fetcher.fetch_snapshot())

    assert snapshot["accounts"][0]["name"] == "Demo A"
    assert snapshot["accounts"][0]["positions"][0]["symbol"] == "BTCUSDT"

    view = build_presentable_snapshot(snapshot)
    assert view["accounts"][0]["positions"][0]["symbol"] == "BTCUSDT"
    assert view["alerts"] == []

    dashboard = build_dashboard(snapshot)
    assert "Demo A" in dashboard

    asyncio.run(fetcher.close())
    assert clients[0].closed and clients[1].closed


def test_realtime_fetcher_reports_errors() -> None:
    config = RealtimeConfig(
        accounts=[AccountConfig(name="Problematic", exchange="binance", credentials={})],
        alert_thresholds={},
        notification_channels=[],
    )
    fetcher = RealtimeDataFetcher(config, account_clients=[FailingAccountClient()])
    snapshot = asyncio.run(fetcher.fetch_snapshot())

    assert "account_messages" in snapshot
    assert "Problematic" in snapshot["account_messages"]

    view = build_presentable_snapshot(snapshot)
    assert view["accounts"] == []
    assert view["hidden_accounts"][0]["name"] == "Problematic"

    asyncio.run(fetcher.close())
