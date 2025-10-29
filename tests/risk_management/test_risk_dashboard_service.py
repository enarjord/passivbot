from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("passlib")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.account_clients import AccountClientProtocol
from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.realtime import RealtimeDataFetcher
from risk_management.web import RiskDashboardService


class _SilentNotifications:
    def __init__(self) -> None:
        self.daily_calls: List[tuple[Mapping[str, Any], float]] = []
        self.alert_calls: List[Mapping[str, Any]] = []

    def send_daily_snapshot(self, snapshot: Mapping[str, Any], portfolio_balance: float) -> None:
        self.daily_calls.append((snapshot, portfolio_balance))

    def dispatch_alerts(self, snapshot: Mapping[str, Any]) -> None:
        self.alert_calls.append(snapshot)


@dataclass
class RecordingAccountClient(AccountClientProtocol):
    name: str
    balances: Iterable[float]

    def __post_init__(self) -> None:
        self.config = AccountConfig(name=self.name, exchange="paper", credentials={})
        self._balances = list(self.balances)
        self.kill_switch_calls: List[Optional[str]] = []
        self.cancel_all_calls: List[Optional[str]] = []
        self.close_all_calls: List[Optional[str]] = []

    async def fetch(self) -> Mapping[str, Any]:  # pragma: no cover - exercised indirectly
        if self._balances:
            balance = self._balances.pop(0)
        else:
            balance = 0.0
        return {"name": self.config.name, "balance": balance, "positions": []}

    async def close(self) -> None:  # pragma: no cover - helper for interface completeness
        return None

    async def kill_switch(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self.kill_switch_calls.append(symbol)
        return {"cancelled_orders": [], "closed_positions": [], "symbol": symbol}

    async def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:  # pragma: no cover - unused in these tests
        return {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "amount": amount,
            "price": price,
            "params": dict(params) if params else None,
        }

    async def cancel_order(
        self, order_id: str, symbol: Optional[str] = None, params: Optional[Mapping[str, Any]] = None
    ) -> Mapping[str, Any]:  # pragma: no cover - unused in these tests
        return {"order_id": order_id, "symbol": symbol, "params": dict(params) if params else None}

    async def close_position(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - unused here
        return {"closed_positions": [symbol]}

    async def list_order_types(self) -> Sequence[str]:  # pragma: no cover - unused in these tests
        return ("limit", "market")

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self.cancel_all_calls.append(symbol)
        return {"cancelled_orders": [], "symbol": symbol}

    async def close_all_positions(self, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self.close_all_calls.append(symbol)
        return {"closed_positions": [], "symbol": symbol}


def _build_service(*clients: RecordingAccountClient) -> RiskDashboardService:
    config = RealtimeConfig(accounts=[client.config for client in clients])
    fetcher = RealtimeDataFetcher(config, account_clients=list(clients))
    fetcher._notifications = _SilentNotifications()  # type: ignore[attr-defined]
    return RiskDashboardService(fetcher)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_service_kill_switch_propagates_to_account_clients() -> None:
    client = RecordingAccountClient("Alpha", balances=[1_000.0])
    service = _build_service(client)

    result = await service.trigger_kill_switch("Alpha", symbol="BTCUSDT")

    assert client.kill_switch_calls == ["BTCUSDT"]
    assert result["Alpha"]["symbol"] == "BTCUSDT"
    await service.close()


@pytest.mark.anyio
async def test_service_cancel_all_orders_reaches_client() -> None:
    client = RecordingAccountClient("Alpha", balances=[1_000.0])
    service = _build_service(client)

    await service.cancel_all_orders("Alpha", symbol="ETHUSDT")

    assert client.cancel_all_calls == ["ETHUSDT"]
    await service.close()


@pytest.mark.anyio
async def test_account_stop_loss_updates_after_balance_drop() -> None:
    client = RecordingAccountClient("Alpha", balances=[1_000.0, 950.0, 800.0])
    service = _build_service(client)

    # Prime baseline balance
    await service.fetch_snapshot()
    state = await service.set_account_stop_loss("Alpha", 10.0)
    assert state["threshold_pct"] == 10.0

    await service.fetch_snapshot()
    intermediate = service.get_account_stop_loss("Alpha")
    assert intermediate is not None
    assert intermediate["triggered"] is False
    assert intermediate["current_drawdown_pct"] == pytest.approx(0.05, rel=1e-6)

    await service.fetch_snapshot()
    triggered = service.get_account_stop_loss("Alpha")
    assert triggered is not None
    assert triggered["triggered"] is True
    assert triggered["triggered_at"] is not None
    assert triggered["current_drawdown_pct"] == pytest.approx(0.2, rel=1e-6)

    await service.close()
