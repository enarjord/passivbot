import hmac
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("passlib")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi.testclient import TestClient
from risk_management.configuration import AccountConfig, RealtimeConfig
from risk_management.web import AuthManager, RiskDashboardService, create_app


def _patch_httpx_for_starlette() -> None:
    """Allow Starlette's TestClient to run against legacy httpx releases.

    Older httpx versions (e.g. <0.25) do not accept the ``app`` keyword that
    newer Starlette/FastAPI releases pass when initialising ``httpx.Client``.
    When that happens the constructor raises ``TypeError: unexpected keyword``
    and the tests crash during collection.  We patch ``httpx.Client.__init__``
    to accept the extra parameter and delegate to the original implementation
    so the rest of the behaviour stays untouched.
    """

    parameters = inspect.signature(httpx.Client.__init__).parameters
    if "app" in parameters:
        return

    original_init = httpx.Client.__init__

    def _compat_init(self, *args, app=None, **kwargs):  # type: ignore[override]
        return original_init(self, *args, **kwargs)

    httpx.Client.__init__ = _compat_init  # type: ignore[assignment]


_patch_httpx_for_starlette()


class StubFetcher:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot
        self.closed = False
        self.kill_requests: List[Tuple[Optional[str], Optional[str]]] = []

    async def fetch_snapshot(self) -> dict:
        return self.snapshot

    async def close(self) -> None:
        self.closed = True

    async def execute_kill_switch(
        self,
        account_name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> dict:
        self.kill_requests.append((account_name, symbol))
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


class _TestingAuthManager(AuthManager):
    """Simplified AuthManager that avoids bcrypt backend requirements."""

    def __init__(self) -> None:
        super().__init__(
            secret_key="super-secret",
            users={"admin": "admin123"},
            https_only=False,
        )

    def authenticate(self, username: str, password: str) -> bool:  # type: ignore[override]
        stored = self.users.get(username)
        if stored is None:
            return False
        return hmac.compare_digest(stored, password)


@pytest.fixture
def auth_manager() -> AuthManager:
    return _TestingAuthManager()


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
        # Starlette's TestClient may surface a 307 redirect when working with
        # newer httpx releases, while older stacks returned 302/303.
        assert response.status_code in {302, 303, 307}
        assert urlparse(response.headers["location"]).path == "/login"

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
        assert response.status_code in {302, 303, 307}

        response = client.get("/")
        assert response.status_code == 200
        assert "Demo Account" in response.text

        api_response = client.get("/api/snapshot")
        assert api_response.status_code == 200
        payload = api_response.json()
        assert payload["accounts"][0]["name"] == "Demo Account"

        logout_response = client.post("/logout", allow_redirects=False)
        assert logout_response.status_code in {302, 303, 307}

    assert fetcher.closed


def test_kill_switch_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post("/api/accounts/Demo%20Account/kill-switch")
        assert response.status_code == 200
        assert fetcher.kill_requests[-1] == ("Demo Account", None)


def test_letsencrypt_challenge_mount(tmp_path: Path, auth_manager: AuthManager) -> None:
    fetcher = StubFetcher({"generated_at": "", "accounts": [], "alert_thresholds": {}, "notification_channels": []})
    service = RiskDashboardService(fetcher)  # type: ignore[arg-type]
    config = RealtimeConfig(accounts=[AccountConfig(name="Demo", exchange="binance", credentials={})])
    challenge_dir = tmp_path / "acme"
    app = create_app(
        config,
        service=service,
        auth_manager=auth_manager,
        letsencrypt_challenge_dir=challenge_dir,
    )
    assert challenge_dir.exists()
    assert any(route.path == "/.well-known/acme-challenge" for route in app.routes)
