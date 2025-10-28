import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Dict, Mapping, TypeVar

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_endpoint_overrides import ResolvedEndpointOverride
from risk_management import account_clients as module
from risk_management.account_clients import _apply_credentials, _instantiate_ccxt_client
from risk_management.configuration import AccountConfig
from risk_management.realized_pnl import fetch_realized_pnl_history


T = TypeVar("T")


def run_async(coro: Awaitable[T]) -> T:
    """Execute ``coro`` in a fresh event loop to avoid cross-test interference."""

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()


class DummyClient:
    def __init__(self) -> None:
        self.headers = {"Existing": "1"}
        self.options = {"existing": True}


def test_apply_credentials_merges_and_sets_sensitive_fields() -> None:
    client = DummyClient()

    credentials = {
        "apiKey": "key",
        "secret": "secret",
        "password": "pass",
        "headers": {"X-First": "A"},
        "options": {"defaultType": "swap"},
        "ccxt": {
            "uid": "123",
            "headers": {"X-Nested": "B"},
        },
    }

    _apply_credentials(client, credentials)

    assert client.apiKey == "key"
    assert client.secret == "secret"
    assert client.password == "pass"
    assert client.uid == "123"
    assert client.headers == {"Existing": "1", "X-First": "A", "X-Nested": "B"}
    assert client.options == {"existing": True, "defaultType": "swap"}


def test_apply_credentials_formats_header_placeholders() -> None:
    client = DummyClient()

    client.headers["Authorization"] = "Bearer {apiKey}:{secret}"
    credentials = {"apiKey": "alpha", "secret": "beta"}

    _apply_credentials(client, credentials)

    assert client.headers["Authorization"] == "Bearer alpha:beta"


def test_instantiate_ccxt_client_applies_custom_endpoints(monkeypatch) -> None:
    class DummyExchange:
        def __init__(self, params):
            self.params = params
            self.hostname = "bybit.com"
            self.urls = {
                "api": {"public": "https://api.bybit.com/v5"},
                "host": "https://api.bybit.com",
            }
            self.headers = {}
            self.options = {}
            self.has = {}

    class DummyNamespace:
        def __init__(self):
            self.bybit = DummyExchange

    monkeypatch.setattr(module, "load_ccxt_instance", None)
    monkeypatch.setattr(module, "ccxt_async", DummyNamespace())
    monkeypatch.setattr(module, "normalize_exchange_name", lambda exchange: "bybit")

    override = ResolvedEndpointOverride(
        exchange_id="bybit",
        rest_domain_rewrites={"https://api.bybit.com": "https://proxy.example"},
    )

    def fake_resolve(exchange_id: str):
        assert exchange_id == "bybit"
        return override

    monkeypatch.setattr(module, "resolve_custom_endpoint_override", fake_resolve)

    client = _instantiate_ccxt_client("bybit", {})

    assert client.urls["api"]["public"] == "https://proxy.example/v5"
    assert client.urls["host"] == "https://proxy.example"


def test_instantiate_ccxt_client_respects_load_helper_override(monkeypatch) -> None:
    class DummyExchange:
        def __init__(self) -> None:
            self.hostname = "binance.com"
            self.urls = {
                "api": {"public": "https://api.binance.com/v3"},
                "host": "https://api.binance.com",
            }
            self.headers: Dict[str, str] = {}
            self.options: Dict[str, Any] = {}
            self.has: Dict[str, Any] = {}

    calls: list[Dict[str, Any]] = []

    def fake_load(exchange_id: str, enable_rate_limit: bool = True, apply_custom_endpoints: bool = True):
        calls.append(
            {
                "exchange_id": exchange_id,
                "enable_rate_limit": enable_rate_limit,
                "apply_custom_endpoints": apply_custom_endpoints,
            }
        )
        return DummyExchange()

    monkeypatch.setattr(module, "load_ccxt_instance", fake_load)
    monkeypatch.setattr(module, "ccxt_async", None)
    monkeypatch.setattr(module, "normalize_exchange_name", lambda exchange: "binanceusdm")

    override = ResolvedEndpointOverride(
        exchange_id="binanceusdm",
        rest_domain_rewrites={"https://api.binance.com": "https://proxy.example"},
    )

    client = module._instantiate_ccxt_client(
        "binanceusdm",
        {},
        custom_endpoint_override=override,
    )

    assert calls and calls[-1]["apply_custom_endpoints"] is False
    assert client.urls["api"]["public"] == "https://proxy.example/v3"
    assert client.urls["host"] == "https://proxy.example"

    direct_client = module._instantiate_ccxt_client(
        "binanceusdm",
        {},
        custom_endpoint_override=None,
    )

    assert len(calls) == 2
    assert calls[-1]["apply_custom_endpoints"] is False
    assert direct_client.urls["api"]["public"] == "https://api.binance.com/v3"
    assert direct_client.urls["host"] == "https://api.binance.com"


def test_translate_ccxt_error_auth(monkeypatch) -> None:
    dummy_client = object()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any], **kwargs):
        return dummy_client

    override = ResolvedEndpointOverride(exchange_id="binanceusdm")

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)
    monkeypatch.setattr(module, "resolve_custom_endpoint_override", lambda exch: override)
    monkeypatch.setattr(module, "get_custom_endpoint_source", lambda: Path("/tmp/custom.json"))

    config = AccountConfig(name="Binance", exchange="binanceusdm", credentials={})
    client = module.CCXTAccountClient(config)

    error = module.BaseError('binanceusdm {"msg":"Invalid API key","source":"mltech"}')

    translated = client._translate_ccxt_error(error)

    assert isinstance(translated, module.AuthenticationError)
    message = str(translated)
    assert "Invalid API key" in message
    assert "mltech" in message
    assert "API key" in message


def test_translate_ccxt_error_generic(monkeypatch) -> None:
    dummy_client = object()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any], **kwargs):
        return dummy_client

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)
    monkeypatch.setattr(module, "resolve_custom_endpoint_override", lambda exch: None)
    monkeypatch.setattr(module, "get_custom_endpoint_source", lambda: None)

    config = AccountConfig(name="Binance", exchange="binanceusdm", credentials={})
    client = module.CCXTAccountClient(config)

    error = module.BaseError("binanceusdm {\"message\":\"rate limit exceeded\"}")

    translated = client._translate_ccxt_error(error)

    assert isinstance(translated, RuntimeError)
    message = str(translated)
    assert "rate limit exceeded" in message
    assert "See logs" in message


def test_fetch_realized_pnl_history_binance_uses_income_endpoint(monkeypatch) -> None:
    class DummyIncomeClient:
        def __init__(self) -> None:
            self.calls: list[Mapping[str, Any]] = []

        async def fetch_income(self, params=None):  # type: ignore[override]
            self.calls.append(dict(params or {}))
            return [
                {"amount": "1.5"},
                {"info": {"income": "0.5"}},
            ]

    dummy = DummyIncomeClient()

    realized = run_async(
        fetch_realized_pnl_history(
            "binanceusdm",
            dummy,
            since=940_000,
            until=1_000_000,
        )
    )

    assert realized == pytest.approx(2.0)
    assert dummy.calls, "fetch_income should have been called"
    params = dummy.calls[-1]
    assert params["incomeType"] == "REALIZED_PNL"
    assert params["startTime"] == 940_000
    assert params["endTime"] == 1_000_000


def test_fetch_realized_pnl_history_bybit_paginates_closed_pnl() -> None:
    class DummyBybitClient:
        def __init__(self) -> None:
            self.calls: list[Mapping[str, Any]] = []
            self._responses = [
                {
                    "result": {
                        "list": [{"pnl": "1.2"}, {"closedPnl": "-0.2"}],
                        "nextPageCursor": "cursor123",
                    }
                },
                {"result": {"list": [{"pnl": "0.3"}]}}
            ]

        async def private_get_v5_position_closed_pnl(self, params=None):  # type: ignore[override]
            self.calls.append(dict(params or {}))
            return self._responses.pop(0)

    dummy = DummyBybitClient()

    realized = run_async(
        fetch_realized_pnl_history(
            "bybit",
            dummy,
            since=1_940_000,
            until=2_000_000,
            limit=100,
        )
    )

    assert realized == pytest.approx(1.3)
    assert len(dummy.calls) == 2
    assert dummy.calls[0]["limit"] == 100
    assert dummy.calls[1]["cursor"] == "cursor123"


def test_fetch_realized_pnl_history_okx_sums_trade_pnl() -> None:
    class DummyOkxClient:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, Any, Mapping[str, Any]]] = []

        async def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):  # type: ignore[override]
            params = dict(params or {})
            self.calls.append((symbol, since, limit, params))
            return [
                {"pnl": "0.5"},
                {"info": {"fillPnl": "-0.2"}},
            ]

    dummy = DummyOkxClient()

    realized = run_async(
        fetch_realized_pnl_history(
            "okx",
            dummy,
            since=500_000,
            until=600_000,
            symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
            limit=50,
        )
    )

    assert realized == pytest.approx(0.6)
    assert len(dummy.calls) == 2
    first_call = dummy.calls[0]
    assert first_call[0] == "BTC/USDT:USDT"
    assert first_call[1] == 500_000
    assert first_call[2] == 50
    assert first_call[3]["until"] == 600_000


def test_account_fetch_uses_realized_history_when_requested(monkeypatch) -> None:
    class DummyExchange:
        def __init__(self) -> None:
            self.markets = {}
            self.options = {}

        async def load_markets(self):  # type: ignore[override]
            self.markets = {"BTCUSDT": {}}

        async def fetch_balance(self, params=None):  # type: ignore[override]
            return {"total": {"USDT": 1_000}, "info": {"totalWalletBalance": "1000"}}

        async def fetch_positions(self, params=None):  # type: ignore[override]
            return [
                {
                    "symbol": "BTCUSDT",
                    "contracts": "1",
                    "entryPrice": "100",
                    "markPrice": "110",
                    "unrealizedPnl": "10",
                    "dailyRealizedPnl": "0",
                }
            ]

        async def fetch_open_orders(self, symbol=None, params=None):  # type: ignore[override]
            return []

        async def close(self):  # type: ignore[override]
            return None

    dummy_client = DummyExchange()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any], **kwargs):
        assert exchange == "bybit"
        assert "custom_endpoint_override" in kwargs
        return dummy_client

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)

    async def fake_collect(self, symbols):  # type: ignore[override]
        return {}

    monkeypatch.setattr(module.CCXTAccountClient, "_collect_symbol_metrics", fake_collect)

    async def fake_fetch_realized(self, positions, *, now_ms=None):  # type: ignore[override]
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTCUSDT"
        return 7.5

    monkeypatch.setattr(module.CCXTAccountClient, "_fetch_realized_pnl", fake_fetch_realized)

    config = AccountConfig(
        name="Bybit",
        exchange="bybit",
        settle_currency="USDT",
        credentials={},
        params={
            "realized_pnl": {
                "mode": "always",
                "lookback_ms": 60_000,
                "since_ms": 1_940_000,
                "until_ms": 2_000_000,
            }
        },
    )

    client = module.CCXTAccountClient(config)

    result = run_async(client.fetch())

    assert result["daily_realized_pnl"] == pytest.approx(7.5)
    assert result["positions"][0]["daily_realized_pnl"] == 0.0
