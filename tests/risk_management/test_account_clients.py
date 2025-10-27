import asyncio
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_endpoint_overrides import ResolvedEndpointOverride
from risk_management import account_clients as module
from risk_management.account_clients import _apply_credentials, _instantiate_ccxt_client
from risk_management.configuration import AccountConfig
from risk_management.realized_pnl import fetch_realized_pnl_history


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

def test_fetch_realized_pnl_history_binance_uses_income_endpoint(monkeypatch) -> None:
    class DummyIncomeClient:
def test_fetch_realized_pnl_binance_uses_income_endpoint(monkeypatch) -> None:
    class DummyClient:
        def __init__(self) -> None:
            self.calls: list[Mapping[str, Any]] = []

        async def fetch_income(self, params=None):  # type: ignore[override]
            self.calls.append(dict(params or {}))
            return [
                {"amount": "1.5"},
                {"info": {"income": "0.5"}},
            ]

    dummy = DummyIncomeClient()

    realized = asyncio.run(
        fetch_realized_pnl_history(
            "binanceusdm",
            dummy,
            since=940_000,
            until=1_000_000,
        )
    )

    dummy = DummyClient()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any]):
        assert exchange == "binanceusdm"
        return dummy

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)

    config = AccountConfig(
        name="Binance",
        exchange="binanceusdm",
        settle_currency="USDT",
        credentials={},
        params={"realized_pnl": {"lookback_ms": 60_000}},
    )

    client = module.CCXTAccountClient(config)
    realized = asyncio.run(client._fetch_realized_pnl([], now_ms=1_000_000))
    assert realized == pytest.approx(2.0)
    assert dummy.calls, "fetch_income should have been called"
    params = dummy.calls[-1]
    assert params["incomeType"] == "REALIZED_PNL"
    assert params["startTime"] == 940_000
    assert params["endTime"] == 1_000_000

def test_fetch_realized_pnl_history_bybit_paginates_closed_pnl() -> None:
    class DummyBybitClient:

def test_fetch_realized_pnl_bybit_paginates_closed_pnl(monkeypatch) -> None:
    class DummyClient:
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

    realized = asyncio.run(
        fetch_realized_pnl_history(
            "bybit",
            dummy,
            since=1_940_000,
            until=2_000_000,
            limit=100,
        )
    )

    dummy = DummyClient()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any]):
        assert exchange == "bybit"
        return dummy

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)

    config = AccountConfig(
        name="Bybit",
        exchange="bybit",
        settle_currency="USDT",
        credentials={},
        params={"realized_pnl": {"lookback_ms": 60_000, "limit": 100}},
    )

    client = module.CCXTAccountClient(config)
    realized = asyncio.run(client._fetch_realized_pnl([], now_ms=2_000_000))

    assert realized == pytest.approx(1.3)
    assert len(dummy.calls) == 2
    assert dummy.calls[0]["limit"] == 100
    assert dummy.calls[1]["cursor"] == "cursor123"

def test_fetch_realized_pnl_history_okx_sums_trade_pnl(monkeypatch) -> None:
    class DummyOkxClient:
def test_fetch_realized_pnl_okx_sums_trade_pnl(monkeypatch) -> None:
    class DummyClient:
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

    realized = asyncio.run(
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

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any]):
        assert exchange == "bybit"
        return dummy_client

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)

    async def fake_collect(self, symbols):  # type: ignore[override]
        return {}

    monkeypatch.setattr(module.CCXTAccountClient, "_collect_symbol_metrics", fake_collect)

    async def fake_fetch_realized(exchange_id, client, **kwargs):  # type: ignore[override]
        assert exchange_id == "bybit"
        assert kwargs["account_name"] == "Bybit"
        assert kwargs["since"] == 1_940_000
        assert kwargs["until"] == 2_000_000
        return 7.5

    monkeypatch.setattr(module, "fetch_realized_pnl_history", fake_fetch_realized)

    config = AccountConfig(
        name="Bybit",
        exchange="bybit",
    dummy = DummyClient()

    def fake_instantiate(exchange: str, credentials: Mapping[str, Any]):
        assert exchange == "okx"
        return dummy

    monkeypatch.setattr(module, "_instantiate_ccxt_client", fake_instantiate)

    config = AccountConfig(
        name="OKX",
        exchange="okx",
        settle_currency="USDT",
        credentials={},
        params={
            "realized_pnl": {

                "mode": "always",
                "lookback_ms": 60_000,
                "since_ms": 1_940_000,
                "until_ms": 2_000_000,

                "lookback_ms": 60_000,
                "symbols": ["BTC/USDT:USDT"],
            }
        },
    )

    client = module.CCXTAccountClient(config)

    result = asyncio.run(client.fetch())

    assert result["daily_realized_pnl"] == pytest.approx(7.5)
    assert result["positions"][0]["daily_realized_pnl"] == 0.0
    positions = [{"symbol": "BTC/USDT:USDT"}]
    realized = asyncio.run(client._fetch_realized_pnl(positions, now_ms=500_000))

    assert realized == pytest.approx(0.3)
    assert dummy.calls
    symbol, since, limit, params = dummy.calls[-1]
    assert symbol == "BTC/USDT:USDT"
    assert params["until"] == 500_000
