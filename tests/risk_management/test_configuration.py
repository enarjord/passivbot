"""Tests for realtime risk management configuration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.configuration import (  # noqa: E402
    _merge_credentials,
    _normalise_credentials,
    load_realtime_config,
)


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "configs" / "risk" / "realtime.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _base_payload() -> dict:
    return {
        "accounts": [
            {
                "name": "Example",
                "exchange": "binanceusdm",
                "credentials": {"key": "abc", "secret": "def"},
            }
        ],
    }


def test_custom_endpoint_path_resolves_relative(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["custom_endpoints"] = {"path": "../custom_endpoints.json", "autodiscover": False}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    expected_path = (config_path.parent / Path("../custom_endpoints.json")).resolve()
    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(expected_path)


def test_custom_endpoint_path_keeps_absolute(tmp_path: Path) -> None:
    payload = _base_payload()
    absolute_path = tmp_path / "custom" / "endpoints.json"
    payload["custom_endpoints"] = {"path": str(absolute_path), "autodiscover": True}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(absolute_path.resolve())
    assert config.custom_endpoints.autodiscover is True


def test_normalise_credentials_supports_aliases() -> None:
    payload = {
        "key": " key-value ",
        "api_secret": " secret-value ",
        "passPhrase": " pass ",
        "uid": " 123 ",
        "exchange": "binance",
        "headers": {"X-Test": "1"},
        "options": {"defaultType": "swap"},
        "ccxt_config": {"login": "demo"},
        "wallet_address": " wallet ",
        "private_key": " private ",
    }

    normalised = _normalise_credentials(payload)

    assert normalised == {
        "apiKey": "key-value",
        "secret": "secret-value",
        "password": "pass",
        "uid": "123",
        "headers": {"X-Test": "1"},
        "options": {"defaultType": "swap"},
        "ccxt": {"login": "demo"},
        "walletAddress": "wallet",
        "privateKey": "private",
    }


def test_merge_credentials_prioritises_primary_values() -> None:
    primary = {"apiKey": "primary", "headers": {"X-Primary": "1"}}
    secondary = {"key": "secondary", "headers": {"X-Secondary": "2"}, "exchange": "binance"}

    merged = _merge_credentials(primary, secondary)

    assert merged["apiKey"] == "primary"
    assert merged["headers"] == {"X-Secondary": "2", "X-Primary": "1"}
    assert "exchange" not in merged


def test_load_realtime_config_supports_nested_user_entries(tmp_path: Path) -> None:
    api_keys_path = tmp_path / "api-keys.json"
    api_keys_path.write_text(
        json.dumps(
            {
                "referrals": {"binance": "https://example.com"},
                "binance_01": {"exchange": "binance", "key": "a", "secret": "b"},
                "users": {
                    "okx_01": {
                        "exchange": "okx",
                        "key": "c",
                        "secret": "d",
                        "passphrase": "p",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "realtime.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys_file": "../api-keys.json",
                "accounts": [
                    {
                        "name": "Binance",
                        "api_key_id": "binance_01",
                        "exchange": "binance",
                    },
                    {
                        "name": "OKX",
                        "api_key_id": "okx_01",
                        "exchange": "okx",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)

    assert len(config.accounts) == 2
    binance = config.accounts[0]
    okx = config.accounts[1]

    assert binance.credentials["apiKey"] == "a"
    assert binance.credentials["secret"] == "b"

    assert okx.credentials["apiKey"] == "c"
    assert okx.credentials["secret"] == "d"
    assert okx.credentials["password"] == "p"
    assert config.config_root == config_path.parent.resolve()


def test_load_realtime_config_expands_user_path(tmp_path: Path, monkeypatch) -> None:
    home_api_keys = tmp_path / "api-keys.json"
    home_api_keys.write_text(
        json.dumps({"binance": {"exchange": "binance", "key": "x", "secret": "y"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(tmp_path))

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys_file": "~/api-keys.json",
                "accounts": [
                    {
                        "name": "Binance",
                        "api_key_id": "binance",
                        "exchange": "binance",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)
    assert config.accounts[0].credentials["apiKey"] == "x"
    assert config.config_root == config_path.parent.resolve()
