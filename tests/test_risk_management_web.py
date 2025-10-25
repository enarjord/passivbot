import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("passlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("passlib")

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from passlib.context import CryptContext

from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.web import AuthManager, RiskDashboardService, create_app


class StubFetcher:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.closed = False
        self.kill_requests: list[str | None] = []

    async def fetch_snapshot(self) -> dict:
        return self.snapshot

    async def close(self) -> None:
        self.closed = True

    async def execute_kill_switch(self, account_name: str | None = None) -> dict:
        self.kill_requests.append(account_name)
        return {"status": "ok"}


@pytest.fixture
def sample_snapshot() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": now,
        "accounts": [
            {
                "name": "Demo Account",
                "balance": 12_000,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "notional": 3_000,
                        "entry_price": 62_500,
                        "mark_price": 63_200,
                        "liquidation_price": 52_000,
                        "wallet_exposure_pct": 0.25,
                        "unrealized_pnl": 210,
                        "max_drawdown_pct": 0.12,
                    }
                ],
            }
        ],
        "alert_thresholds": {
            "wallet_exposure_pct": 0.65,
            "position_wallet_exposure_pct": 0.25,
            "max_drawdown_pct": 0.25,
            "loss_threshold_pct": -0.08,
        },
        "notification_channels": ["email:risk-team@example.com"],
    }


@pytest.fixture
def auth_manager() -> AuthManager:
    context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = context.hash("admin123")
    return AuthManager(secret_key="super-secret", users={"admin": password_hash})


def create_test_app(snapshot: dict, auth_manager: AuthManager) -> tuple[TestClient, StubFetcher]:
    fetcher = StubFetcher(snapshot)
    service = RiskDashboardService(fetcher)  # type: ignore[arg-type]
    config = RealtimeConfig(accounts=[AccountConfig(name="Demo", exchange="binance", credentials={})])
    app = create_app(config, service=service, auth_manager=auth_manager)
    return TestClient(app), fetcher


def test_web_dashboard_auth_flow(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        response = client.get("/", allow_redirects=False)
        assert response.status_code in {302, 303}
        assert response.headers["location"].endswith("/login")

        response = client.get("/login")
        assert response.status_code == 200
        assert "Sign in" in response.text

        response = client.post("/login", data={"username": "admin", "password": "wrong"})
        assert response.status_code == 401
        assert "Invalid username" in response.text

        response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert response.status_code in {302, 303}

        response = client.get("/")
        assert response.status_code == 200
        assert "Demo Account" in response.text

        api_response = client.get("/api/snapshot")
        assert api_response.status_code == 200
        payload = api_response.json()
        assert payload["accounts"][0]["name"] == "Demo Account"

        logout_response = client.post("/logout", allow_redirects=False)
        assert logout_response.status_code in {302, 303}

    assert fetcher.closed


def test_kill_switch_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303}

        response = client.post("/api/accounts/Demo%20Account/kill-switch")
        assert response.status_code == 200
        assert fetcher.kill_requests[-1] == "Demo Account"
