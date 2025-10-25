import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_endpoint_overrides import ResolvedEndpointOverride
from risk_management.account_clients import _apply_credentials, _instantiate_ccxt_client


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
    from risk_management import account_clients as module

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
