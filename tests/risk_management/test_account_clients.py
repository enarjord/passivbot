import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.account_clients import _apply_credentials


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
