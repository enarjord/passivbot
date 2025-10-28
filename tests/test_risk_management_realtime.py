import asyncio
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
        self.config = type("Config", (), {"name": name})()

    async def fetch(self) -> dict:
        return {"name": self.name, "balance": self.balance, "positions": list(self.positions)}

    async def close(self) -> None:
        self.closed = True

    async def kill_switch(self, symbol: Optional[str] = None) -> dict:
        return {
            "cancelled_orders": [],
            "failed_order_cancellations": [],
            "closed_positions": [],
            "failed_position_closures": [],
        }

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[dict] = None,
    ) -> Mapping[str, Any]:
        return {"order": {"symbol": symbol, "type": order_type, "side": side}, "raw": {}}

    async def cancel_order(
        self, order_id: str, symbol: Optional[str] = None, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:
        return {"cancelled": True}

    async def close_position(self, symbol: str) -> Mapping[str, Any]:
        return {"closed_positions": [{"symbol": symbol}]}

    async def list_order_types(self) -> Sequence[str]:
        return ["limit", "market"]

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        return {"cancelled_orders": [], "failed_order_cancellations": []}

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        return {"closed_positions": [], "failed_position_closures": []}


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


def test_account_stop_loss_state_updates(tmp_path: Path) -> None:
    config = RealtimeConfig(
        accounts=[AccountConfig(name="Demo", exchange="binance", credentials={})],
        alert_thresholds={},
        notification_channels=[],
        reports_dir=tmp_path,
    )
    client = StubAccountClient("Demo", 10_000.0, [])
    fetcher = RealtimeDataFetcher(config, account_clients=[client])

    snapshot = asyncio.run(fetcher.fetch_snapshot())
    assert "account_stop_losses" not in snapshot

    asyncio.run(fetcher.set_account_stop_loss("Demo", 10.0))

    client.balance = 9_200.0
    snapshot = asyncio.run(fetcher.fetch_snapshot())
    account_states = snapshot.get("account_stop_losses") or {}
    assert "Demo" in account_states
    state = account_states["Demo"]
    assert state["threshold_pct"] == 10.0
    assert state["current_balance"] == 9_200.0
    assert state["current_drawdown_pct"] == pytest.approx((10_000.0 - 9_200.0) / 10_000.0)
    assert not state["triggered"]

    client.balance = 6_000.0
    snapshot = asyncio.run(fetcher.fetch_snapshot())
    state = snapshot["account_stop_losses"]["Demo"]
    assert state["triggered"] is True
    assert state["current_balance"] == 6_000.0

    asyncio.run(fetcher.clear_account_stop_loss("Demo"))
    snapshot = asyncio.run(fetcher.fetch_snapshot())
    assert "account_stop_losses" not in snapshot or "Demo" not in snapshot["account_stop_losses"]

    asyncio.run(fetcher.close())
