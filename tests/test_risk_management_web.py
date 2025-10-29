import hmac
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
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


class StubRiskService:
    def __init__(
        self,
        snapshot: dict,
        *,
        kill_switch_responses: Optional[List[dict]] = None,
        order_types: Optional[Sequence[str]] = None,
        error_sequences: Optional[Mapping[str, Sequence[Exception]]] = None,
    ) -> None:
        self.snapshot = snapshot
        self.closed = False
        self.kill_requests: List[Tuple[Optional[str], Optional[str]]] = []
        self._kill_switch_responses: List[dict] = list(kill_switch_responses or [])
        self.order_types = list(order_types or ["limit", "market"])
        self.placed_orders: List[Tuple[str, Dict[str, Any]]] = []
        self.cancelled_orders: List[Tuple[str, str, Optional[str]]] = []
        self.closed_positions: List[Tuple[str, str]] = []
        self.stop_loss: Optional[Dict[str, Any]] = None
        self.account_stop_losses: Dict[str, Dict[str, Any]] = {}
        self.cancel_all_orders_calls: List[Tuple[str, Optional[str]]] = []
        self.close_all_positions_calls: List[Tuple[str, Optional[str]]] = []
        self._error_sequences: Dict[str, List[Exception]] = {
            key: list(seq) for key, seq in (error_sequences or {}).items()
        }


    async def fetch_snapshot(self) -> dict:
        return self.snapshot

    async def close(self) -> None:
        self.closed = True

    async def trigger_kill_switch(
        self,
        account_name: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> dict:
        self.kill_requests.append((account_name, symbol))

        if self._kill_switch_responses:
            return self._kill_switch_responses.pop(0)

        return {"status": "ok"}

    async def place_order(
        self,
        account_name: str,
        *,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        self._maybe_raise("place_order")
        payload = {
            "symbol": symbol,
            "order_type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": dict(params) if isinstance(params, Mapping) else None,
        }
        self.placed_orders.append((account_name, payload))
        return {"order": {"symbol": symbol, "side": side, "type": order_type}, "raw": payload}

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        self._maybe_raise("cancel_order")
        self.cancelled_orders.append((account_name, order_id, symbol))
        return {"cancelled": True}

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        self._maybe_raise("close_position")
        self.closed_positions.append((account_name, symbol))
        return {"closed_positions": [{"symbol": symbol}]}

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        return list(self.order_types)

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        return dict(self.stop_loss) if self.stop_loss else None

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        self.stop_loss = {
            "threshold_pct": float(threshold_pct),
            "baseline_balance": 1000.0,
            "triggered": False,
            "triggered_at": None,
            "active": True,
        }
        return dict(self.stop_loss)

    async def clear_portfolio_stop_loss(self) -> None:
        self.stop_loss = None

    def get_account_stop_loss(self, account_name: str) -> Optional[Dict[str, Any]]:
        if not any(acc.get("name") == account_name for acc in self.snapshot.get("accounts", [])):
            raise ValueError(f"Account '{account_name}' not found")
        state = self.account_stop_losses.get(account_name)
        return dict(state) if state is not None else None

    async def set_account_stop_loss(self, account_name: str, threshold_pct: float) -> Dict[str, Any]:
        if not any(acc.get("name") == account_name for acc in self.snapshot.get("accounts", [])):
            raise ValueError(f"Account '{account_name}' not found")
        state = {
            "threshold_pct": float(threshold_pct),
            "baseline_balance": 1000.0,
            "current_balance": 1000.0,
            "current_drawdown_pct": 0.0,
            "triggered": False,
            "triggered_at": None,
            "active": True,
        }
        self.account_stop_losses[account_name] = state
        return dict(state)

    async def clear_account_stop_loss(self, account_name: str) -> None:
        if not any(acc.get("name") == account_name for acc in self.snapshot.get("accounts", [])):
            raise ValueError(f"Account '{account_name}' not found")
        self.account_stop_losses.pop(account_name, None)

    async def cancel_all_orders(self, account_name: str, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self._maybe_raise("cancel_all_orders")
        self.cancel_all_orders_calls.append((account_name, symbol))
        return {"cancelled_orders": [], "failed_order_cancellations": []}

    async def close_all_positions(self, account_name: str, symbol: Optional[str] = None) -> Mapping[str, Any]:
        self._maybe_raise("close_all_positions")
        self.close_all_positions_calls.append((account_name, symbol))
        return {"closed_positions": [], "failed_position_closures": []}

    def _maybe_raise(self, key: str) -> None:
        queue = self._error_sequences.get(key)
        if not queue:
            return
        exc = queue.pop(0)
        if not queue:
            self._error_sequences.pop(key, None)
        raise exc


class StubPerformanceRepository:
    def __init__(
        self,
        *,
        portfolio_series: Sequence[Mapping[str, Any]],
        account_series: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> None:
        self.portfolio_series = [dict(item) for item in portfolio_series]
        self.account_series = {name: [dict(entry) for entry in history] for name, history in account_series.items()}

    def get_portfolio_series(
        self, *, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[dict[str, Any]]:
        return self._filter(self.portfolio_series, start=start, end=end)

    def get_account_series(
        self, account_name: str, *, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[dict[str, Any]]:
        if account_name not in self.account_series:
            raise KeyError(account_name)
        return self._filter(self.account_series[account_name], start=start, end=end)

    def _filter(
        self,
        series: Sequence[Mapping[str, Any]],
        *,
        start: Optional[str],
        end: Optional[str],
    ) -> List[dict[str, Any]]:
        if start == "invalid" or end == "invalid":
            raise ValueError("invalid date value")
        if start and end and start > end:
            raise ValueError("start date cannot be after end date")
        filtered: List[dict[str, Any]] = []
        for entry in series:
            date_str = str(entry.get("date"))
            if start and date_str < start:
                continue
            if end and date_str > end:
                continue
            filtered.append({
                "date": date_str,
                "balance": float(entry.get("balance", 0.0)),
                "timestamp": entry.get("timestamp"),
            })
        return filtered


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
                        "daily_realized_pnl": 15,
                        "max_drawdown_pct": 0.12,
                    }
                ],
                "daily_realized_pnl": 15,
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


def create_test_app(
    snapshot: dict,
    auth_manager: AuthManager,
    *,
    kill_switch_responses: Optional[List[dict]] = None,
    performance_repository: Optional[Any] = None,
) -> tuple[TestClient, StubFetcher]:
    fetcher = StubFetcher(snapshot, kill_switch_responses=kill_switch_responses)
    ) -> tuple[TestClient, StubRiskService]:
    fetcher = StubRiskService(snapshot, kill_switch_responses=kill_switch_responses)
    service = RiskDashboardService(fetcher)  # type: ignore[arg-type]
    config = RealtimeConfig(accounts=[AccountConfig(name="Demo", exchange="binance", credentials={})])
    app = create_app(
        config,
        service=service,
        auth_manager=auth_manager,
        performance_repository=performance_repository,
    )
    return TestClient(app), fetcher


def create_async_test_app(
    snapshot: dict,
    auth_manager: AuthManager,
    *,
    kill_switch_responses: Optional[List[dict]] = None,
    error_sequences: Optional[Mapping[str, Sequence[Exception]]] = None,
) -> tuple[httpx.AsyncClient, StubFetcher]:
    fetcher = StubFetcher(
        snapshot,
        kill_switch_responses=kill_switch_responses,
        error_sequences=error_sequences,
    )
    service = RiskDashboardService(fetcher)  # type: ignore[arg-type]
    config = RealtimeConfig(accounts=[AccountConfig(name="Demo", exchange="binance", credentials={})])
    app = create_app(config, service=service, auth_manager=auth_manager)
    transport = getattr(httpx, "ASGITransport", None)
    if transport is not None:
        client = httpx.AsyncClient(transport=transport(app=app), base_url="http://testserver")
    else:  # pragma: no cover - legacy httpx fallback
        client = httpx.AsyncClient(app=app, base_url="http://testserver")
    return client, fetcher


AssertionFn = Callable[[StubFetcher, httpx.Response], None]


@dataclass(frozen=True)
class AsyncFlowScenario:
    identifier: str
    method: str
    path: str
    payload: Optional[Mapping[str, Any]]
    error_sequences: Optional[Mapping[str, Sequence[Exception]]]
    expected_status: int
    assertion: AssertionFn


def _assert_place_order_success(fetcher: StubFetcher, response: httpx.Response) -> None:
    payload = response.json()
    assert payload["order"]["type"] == "limit"
    assert fetcher.placed_orders[-1][0] == "Demo Account"
    assert fetcher.placed_orders[-1][1]["symbol"] == "BTCUSDT"


def _assert_error_detail(expected: str) -> AssertionFn:
    def _checker(_fetcher: StubFetcher, response: httpx.Response) -> None:
        assert response.json()["detail"] == expected

    return _checker


def _assert_cancel_success(fetcher: StubFetcher, response: httpx.Response) -> None:
    payload = response.json()
    assert payload["cancelled"] is True
    assert fetcher.cancelled_orders[-1] == ("Demo Account", "42", "BTCUSDT")


def _assert_close_success(fetcher: StubFetcher, response: httpx.Response) -> None:
    payload = response.json()
    assert payload["closed_positions"][0]["symbol"] == "BTCUSDT"
    assert fetcher.closed_positions[-1] == ("Demo Account", "BTCUSDT")


ASYNC_FLOW_SCENARIOS: Sequence[AsyncFlowScenario] = (
    AsyncFlowScenario(
        identifier="place-order-success",
        method="POST",
        path="/api/trading/accounts/Demo%20Account/orders",
        payload={
            "symbol": "BTCUSDT",
            "order_type": "limit",
            "side": "buy",
            "amount": 1.25,
            "price": 42_000,
        },
        error_sequences=None,
        expected_status=200,
        assertion=_assert_place_order_success,
    ),
    AsyncFlowScenario(
        identifier="place-order-missing-account",
        method="POST",
        path="/api/trading/accounts/Unknown/orders",
        payload={
            "symbol": "BTCUSDT",
            "order_type": "limit",
            "side": "buy",
            "amount": 1.25,
            "price": 42_000,
        },
        error_sequences={"place_order": [ValueError("Account 'Unknown' not found")]},
        expected_status=404,
        assertion=_assert_error_detail("Account 'Unknown' not found"),
    ),
    AsyncFlowScenario(
        identifier="place-order-invalid",
        method="POST",
        path="/api/trading/accounts/Demo%20Account/orders",
        payload={
            "symbol": "BTCUSDT",
            "order_type": "limit",
            "side": "buy",
            "amount": 1.25,
            "price": 42_000,
        },
        error_sequences={"place_order": [RuntimeError("order rejected")]},
        expected_status=400,
        assertion=_assert_error_detail("order rejected"),
    ),
    AsyncFlowScenario(
        identifier="cancel-order-success",
        method="DELETE",
        path="/api/trading/accounts/Demo%20Account/orders/42",
        payload={"symbol": "BTCUSDT"},
        error_sequences=None,
        expected_status=200,
        assertion=_assert_cancel_success,
    ),
    AsyncFlowScenario(
        identifier="cancel-order-not-found",
        method="DELETE",
        path="/api/trading/accounts/Demo%20Account/orders/42",
        payload={"symbol": "BTCUSDT"},
        error_sequences={"cancel_order": [ValueError("Unknown order")]},
        expected_status=404,
        assertion=_assert_error_detail("Unknown order"),
    ),
    AsyncFlowScenario(
        identifier="cancel-order-rejected",
        method="DELETE",
        path="/api/trading/accounts/Demo%20Account/orders/42",
        payload={"symbol": "BTCUSDT"},
        error_sequences={"cancel_order": [RuntimeError("cannot cancel")]},
        expected_status=400,
        assertion=_assert_error_detail("cannot cancel"),
    ),
    AsyncFlowScenario(
        identifier="close-position-success",
        method="POST",
        path="/api/trading/accounts/Demo%20Account/positions/BTCUSDT/close",
        payload=None,
        error_sequences=None,
        expected_status=200,
        assertion=_assert_close_success,
    ),
    AsyncFlowScenario(
        identifier="close-position-not-found",
        method="POST",
        path="/api/trading/accounts/Demo%20Account/positions/BTCUSDT/close",
        payload=None,
        error_sequences={"close_position": [ValueError("no position")]},
        expected_status=404,
        assertion=_assert_error_detail("no position"),
    ),
    AsyncFlowScenario(
        identifier="close-position-failed",
        method="POST",
        path="/api/trading/accounts/Demo%20Account/positions/BTCUSDT/close",
        payload=None,
        error_sequences={"close_position": [RuntimeError("close failed")]},
        expected_status=400,
        assertion=_assert_error_detail("close failed"),
    ),
)


async def _async_login(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/login",
        data={"username": "admin", "password": "admin123"},
        follow_redirects=False,
    )
    assert response.status_code in {302, 303, 307}


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", ASYNC_FLOW_SCENARIOS, ids=lambda sc: sc.identifier)
async def test_async_trading_flows(
    sample_snapshot: dict, auth_manager: AuthManager, scenario: AsyncFlowScenario
) -> None:
    client, fetcher = create_async_test_app(
        sample_snapshot,
        auth_manager,
        error_sequences=scenario.error_sequences,
    )
    try:
        await _async_login(client)
        request_kwargs: Dict[str, Any] = {}
        if scenario.payload is not None:
            request_kwargs["json"] = scenario.payload
        response = await client.request(scenario.method, scenario.path, **request_kwargs)
        assert response.status_code == scenario.expected_status
        scenario.assertion(fetcher, response)
    finally:
        await client.aclose()


def _build_accounts_snapshot() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": now,
        "accounts": [
            {
                "name": "Alpha",
                "balance": 15_000,
                "daily_realized_pnl": 120,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "side": "long",
                        "notional": 5_000,
                        "entry_price": 48_000,
                        "mark_price": 49_200,
                        "liquidation_price": 35_000,
                        "wallet_exposure_pct": 0.2,
                        "unrealized_pnl": 150,
                        "daily_realized_pnl": 25,
                        "max_drawdown_pct": 0.1,
                    }
                ],
            },
            {
                "name": "Bravo",
                "balance": 12_500,
                "daily_realized_pnl": -50,
                "positions": [
                    {
                        "symbol": "ETHUSDT",
                        "side": "short",
                        "notional": 3_500,
                        "entry_price": 3_200,
                        "mark_price": 3_150,
                        "liquidation_price": 3_800,
                        "wallet_exposure_pct": 0.28,
                        "unrealized_pnl": 80,
                        "daily_realized_pnl": -15,
                        "max_drawdown_pct": 0.22,
                    }
                ],
            },
            {
                "name": "Charlie",
                "balance": 9_800,
                "daily_realized_pnl": 40,
                "positions": [
                    {
                        "symbol": "SOLUSDT",
                        "side": "long",
                        "notional": 1_200,
                        "entry_price": 95,
                        "mark_price": 101,
                        "liquidation_price": 60,
                        "wallet_exposure_pct": 0.12,
                        "unrealized_pnl": 72,
                        "daily_realized_pnl": 18,
                        "max_drawdown_pct": 0.05,
                    }
                ],
            },
        ],
        "alert_thresholds": {
            "wallet_exposure_pct": 0.65,
            "position_wallet_exposure_pct": 0.25,
            "max_drawdown_pct": 0.3,
            "loss_threshold_pct": -0.08,
        },
        "notification_channels": [],
    }


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
    kill_response = {
        "Demo Account": {
            "cancelled_orders": [],
            "failed_order_cancellations": [],
            "closed_positions": [],
            "failed_position_closures": [],
        }
    }
    client, fetcher = create_test_app(
        sample_snapshot, auth_manager, kill_switch_responses=[kill_response]
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post("/api/accounts/Demo%20Account/kill-switch")
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["results"] == kill_response
        assert fetcher.kill_requests[-1] == ("Demo Account", None)


def test_trading_panel_page(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, _ = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.get("/trading-panel")
        assert response.status_code == 200
        assert "Trading panel" in response.text


def test_snapshot_api_paginates_and_sorts(auth_manager: AuthManager) -> None:
    snapshot = _build_accounts_snapshot()
    client, _ = create_test_app(snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        first_page = client.get("/api/snapshot", params={"page_size": 2, "sort": "name", "sort_order": "asc"})
        assert first_page.status_code == 200
        payload = first_page.json()
        assert payload["accounts"][0]["name"] == "Alpha"
        assert payload["accounts"][1]["name"] == "Bravo"
        assert payload["accounts_meta"]["page"] == 1
        assert payload["accounts_meta"]["pages"] == 2

        second_page = client.get("/api/snapshot", params={"page_size": 2, "page": 2, "sort": "name", "sort_order": "asc"})
        assert second_page.status_code == 200
        second_payload = second_page.json()
        assert len(second_payload["accounts"]) == 1
        assert second_payload["accounts"][0]["name"] == "Charlie"
        assert second_payload["accounts_meta"]["page"] == 2


def test_snapshot_api_filters_by_search_and_exposure(auth_manager: AuthManager) -> None:
    snapshot = _build_accounts_snapshot()
    client, _ = create_test_app(snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        search_response = client.get("/api/snapshot", params={"search": "eth", "page_size": 5})
        assert search_response.status_code == 200
        search_payload = search_response.json()
        assert len(search_payload["accounts"]) == 1
        assert search_payload["accounts"][0]["name"] == "Bravo"
        assert search_payload["accounts_meta"]["filtered"] == 1

        short_response = client.get("/api/snapshot", params={"exposure": "net_short", "page_size": 5})
        assert short_response.status_code == 200
        short_payload = short_response.json()
        assert len(short_payload["accounts"]) == 1
        assert short_payload["accounts"][0]["name"] == "Bravo"


def test_snapshot_api_validates_query_params(auth_manager: AuthManager) -> None:
    snapshot = _build_accounts_snapshot()
    client, _ = create_test_app(snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        invalid_sort = client.get("/api/snapshot", params={"sort": "invalid"})
        assert invalid_sort.status_code == 400

        invalid_page = client.get("/api/snapshot", params={"page": 0})
        assert invalid_page.status_code == 400

        too_large_page_size = client.get("/api/snapshot", params={"page_size": 10_000})
        assert too_large_page_size.status_code == 400


def test_position_kill_switch_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    kill_response = {
        "Demo Account": {
            "cancelled_orders": [],
            "failed_order_cancellations": [],
            "closed_positions": [
                {"symbol": "BTC/USDT:USDT", "side": "sell", "amount": 1, "price": 62_000}
            ],
            "failed_position_closures": [],
        }
    }
    client, fetcher = create_test_app(
        sample_snapshot, auth_manager, kill_switch_responses=[kill_response]
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post(
            "/api/accounts/Demo%20Account/positions/BTC%2FUSDT%3AUSDT/kill-switch"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["results"] == kill_response
        assert fetcher.kill_requests[-1] == ("Demo Account", "BTC/USDT:USDT")


def test_kill_switch_endpoint_reports_failures(
    sample_snapshot: dict, auth_manager: AuthManager
) -> None:
    failure_response = {
        "Demo Account": {
            "cancelled_orders": [],
            "failed_order_cancellations": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "order_id": "123",
                    "error": "Something went wrong",
                }
            ],
            "closed_positions": [],
            "failed_position_closures": [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "sell",
                    "amount": 1,
                    "error": "Order's position side does not match user's setting.",
                }
            ],
        }
    }
    client, fetcher = create_test_app(
        sample_snapshot, auth_manager, kill_switch_responses=[failure_response]
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post("/api/accounts/Demo%20Account/kill-switch")
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is False
        assert payload["results"] == failure_response
        assert any(
            "Failed to close position" in error or "Failed to close" in error
            for error in payload["errors"]
        )
        assert fetcher.kill_requests[-1] == ("Demo Account", None)


def test_position_kill_switch_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post(
            "/api/accounts/Demo%20Account/positions/BTC%2FUSDT%3AUSDT/kill-switch"
        )
        assert response.status_code == 200
        assert fetcher.kill_requests[-1] == ("Demo Account", "BTC/USDT:USDT")


def test_account_stop_loss_endpoints(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.get("/api/trading/accounts/Demo%20Account/stop-loss")
        assert response.status_code == 200
        assert response.json()["stop_loss"] is None

        response = client.post(
            "/api/trading/accounts/Demo%20Account/stop-loss",
            json={"threshold_pct": 7.5},
        )
        assert response.status_code == 200
        assert fetcher.account_stop_losses["Demo Account"]["threshold_pct"] == 7.5

        response = client.get("/api/trading/accounts/Demo%20Account/stop-loss")
        assert response.status_code == 200
        payload = response.json()
        assert payload["stop_loss"]["threshold_pct"] == 7.5

        response = client.delete("/api/trading/accounts/Demo%20Account/stop-loss")
        assert response.status_code == 200
        assert "Demo Account" not in fetcher.account_stop_losses


def test_cancel_all_orders_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post("/api/trading/accounts/Demo%20Account/orders/cancel-all")
        assert response.status_code == 200
        assert fetcher.cancel_all_orders_calls[-1] == ("Demo Account", None)


def test_close_all_positions_endpoint(sample_snapshot: dict, auth_manager: AuthManager) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.post("/api/trading/accounts/Demo%20Account/positions/close-all")
        assert response.status_code == 200
        assert fetcher.close_all_positions_calls[-1] == ("Demo Account", None)


def test_letsencrypt_challenge_mount(tmp_path: Path, auth_manager: AuthManager) -> None:
    fetcher = StubRiskService({"generated_at": "", "accounts": [], "alert_thresholds": {}, "notification_channels": []})
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


def test_portfolio_history_and_cashflows(
    sample_snapshot: dict, auth_manager: AuthManager
) -> None:
    client, fetcher = create_test_app(sample_snapshot, auth_manager)
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        snapshot_response = client.get("/api/snapshot")
        assert snapshot_response.status_code == 200

        history_response = client.get("/api/history/portfolio")
        assert history_response.status_code == 200
        payload = history_response.json()
        assert "summary" in payload
        assert "series" in payload

        create_cashflow = client.post(
            "/api/history/cashflows",
            json={
                "type": "deposit",
                "amount": 1250,
                "currency": "USDT",
                "account": "Fund A",
                "note": "Initial funding",
            },
        )
        assert create_cashflow.status_code == 201

        list_response = client.get("/api/history/cashflows")
        assert list_response.status_code == 200
        entries = list_response.json().get("cashflows", [])
        assert any(entry.get("note") == "Initial funding" for entry in entries)

        report_response = client.get("/api/reports/portfolio")
        assert report_response.status_code == 200
        assert report_response.headers["content-type"].startswith("text/csv")


def _build_performance_series() -> tuple[List[dict[str, Any]], Dict[str, List[dict[str, Any]]]]:
    portfolio = [
        {"date": "2024-01-01", "balance": 1000.0, "timestamp": "2024-01-01T16:00:00+00:00"},
        {"date": "2024-01-02", "balance": 1100.0, "timestamp": "2024-01-02T16:00:00+00:00"},
        {"date": "2024-01-03", "balance": 1080.0, "timestamp": "2024-01-03T16:00:00+00:00"},
    ]
    accounts = {
        "Demo": [
            {"date": "2024-01-01", "balance": 500.0, "timestamp": "2024-01-01T16:00:00+00:00"},
            {"date": "2024-01-02", "balance": 520.0, "timestamp": "2024-01-02T16:00:00+00:00"},
            {"date": "2024-01-03", "balance": 515.0, "timestamp": "2024-01-03T16:00:00+00:00"},
        ]
    }
    return portfolio, accounts


def test_portfolio_performance_endpoint_returns_series(
    auth_manager: AuthManager,
) -> None:
    snapshot = _build_accounts_snapshot()
    portfolio, accounts = _build_performance_series()
    repository = StubPerformanceRepository(portfolio_series=portfolio, account_series=accounts)
    client, _ = create_test_app(
        snapshot,
        auth_manager,
        performance_repository=repository,
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.get("/api/performance/portfolio")
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"series": portfolio}


def test_account_performance_endpoint_filters_by_range(
    auth_manager: AuthManager,
) -> None:
    snapshot = _build_accounts_snapshot()
    portfolio, accounts = _build_performance_series()
    repository = StubPerformanceRepository(portfolio_series=portfolio, account_series=accounts)
    client, _ = create_test_app(
        snapshot,
        auth_manager,
        performance_repository=repository,
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        response = client.get(
            "/api/performance/accounts/Demo",
            params={"start": "2024-01-02", "end": "2024-01-03"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["account"] == "Demo"
        assert payload["series"] == accounts["Demo"][1:]


def test_account_performance_endpoint_handles_errors(
    auth_manager: AuthManager,
) -> None:
    snapshot = _build_accounts_snapshot()
    portfolio, accounts = _build_performance_series()
    repository = StubPerformanceRepository(portfolio_series=portfolio, account_series=accounts)
    client, _ = create_test_app(
        snapshot,
        auth_manager,
        performance_repository=repository,
    )
    with client:
        login_response = client.post(
            "/login",
            data={"username": "admin", "password": "admin123"},
            allow_redirects=False,
        )
        assert login_response.status_code in {302, 303, 307}

        missing = client.get("/api/performance/accounts/Unknown")
        assert missing.status_code == 404

        invalid = client.get(
            "/api/performance/portfolio",
            params={"start": "invalid"},
        )
        assert invalid.status_code == 400
