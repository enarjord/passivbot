"""Tests for realtime risk management configuration helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.configuration import (  # noqa: E402
    _ensure_debug_logging_enabled,
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


def test_debug_logging_enabled_for_global_flag(tmp_path: Path, monkeypatch) -> None:
    payload = _base_payload()
    payload["debug_api_payloads"] = True
    config_path = _write_config(tmp_path, payload)

    calls: List[None] = []

    def record_call() -> None:
        calls.append(None)

    monkeypatch.setattr("risk_management.configuration._ensure_debug_logging_enabled", record_call)

    load_realtime_config(config_path)

    assert calls, "expected debug logging to be enabled when global flag is set"


def test_debug_logging_enabled_for_account_flag(tmp_path: Path, monkeypatch) -> None:
    payload = _base_payload()
    payload["accounts"][0]["debug_api_payloads"] = True
    config_path = _write_config(tmp_path, payload)

    calls: List[None] = []

    def record_call() -> None:
        calls.append(None)

    monkeypatch.setattr("risk_management.configuration._ensure_debug_logging_enabled", record_call)

    load_realtime_config(config_path)

    assert calls, "expected debug logging to be enabled when account flag is set"


def test_default_logging_sets_info_levels(tmp_path: Path) -> None:
    payload = _base_payload()
    config_path = _write_config(tmp_path, payload)

    root_logger = logging.getLogger()
    risk_logger = logging.getLogger("risk_management")

    original_root_level = root_logger.level
    original_root_handlers = list(root_logger.handlers)
    original_risk_level = risk_logger.level
    original_risk_handlers = list(risk_logger.handlers)

    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    root_handler = logging.StreamHandler()
    root_handler.setLevel(logging.ERROR)
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.ERROR)

    for handler in risk_logger.handlers:
        risk_logger.removeHandler(handler)
    risk_handler = logging.StreamHandler()
    risk_handler.setLevel(logging.ERROR)
    risk_logger.addHandler(risk_handler)
    risk_logger.setLevel(logging.ERROR)

    try:
        load_realtime_config(config_path)

        assert root_logger.level == logging.INFO
        assert root_handler.level == logging.INFO
        assert risk_logger.level == logging.INFO
        assert risk_handler.level == logging.INFO
    finally:
        risk_logger.removeHandler(risk_handler)
        for handler in original_risk_handlers:
            risk_logger.addHandler(handler)
        risk_logger.setLevel(original_risk_level)

        root_logger.removeHandler(root_handler)
        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_root_level)


def test_debug_logging_promotes_root_and_risk_loggers(monkeypatch) -> None:
    root_logger = logging.getLogger()
    risk_logger = logging.getLogger("risk_management")

    original_root_level = root_logger.level
    original_root_handlers = list(root_logger.handlers)
    original_risk_level = risk_logger.level
    original_risk_handlers = list(risk_logger.handlers)

    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)
    root_handler = logging.StreamHandler()
    root_handler.setLevel(logging.WARNING)
    root_logger.addHandler(root_handler)
    root_logger.setLevel(logging.WARNING)

    for handler in risk_logger.handlers:
        risk_logger.removeHandler(handler)
    risk_handler = logging.StreamHandler()
    risk_handler.setLevel(logging.WARNING)
    risk_logger.addHandler(risk_handler)
    risk_logger.setLevel(logging.WARNING)

    monkeypatch.setattr(
        "risk_management.configuration._configure_default_logging", lambda debug_level=2: False
    )

    try:
        _ensure_debug_logging_enabled()

        assert root_logger.level == logging.DEBUG
        assert root_handler.level == logging.DEBUG
        assert risk_logger.level == logging.DEBUG
        assert risk_handler.level == logging.DEBUG
    finally:
        risk_logger.removeHandler(risk_handler)
        for handler in original_risk_handlers:
            risk_logger.addHandler(handler)
        risk_logger.setLevel(original_risk_level)

        root_logger.removeHandler(root_handler)
        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_root_level)


def test_auth_https_only_flag_respected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["auth"] = {
        "secret_key": "abc",
        "users": {"demo": "hashed"},
        "https_only": False,
    }

    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.auth is not None
    assert config.auth.https_only is False


def test_load_realtime_config_parses_email_settings(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["email"] = {
        "host": "smtp.example.com",
        "port": 2525,
        "username": "alerts@example.com",
        "password": "secret",
        "sender": "alerts@example.com",
        "use_tls": False,
        "use_ssl": True,
    }
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.email is not None
    assert config.email.host == "smtp.example.com"
    assert config.email.port == 2525
    assert config.email.username == "alerts@example.com"
    assert config.email.password == "secret"
    assert config.email.sender == "alerts@example.com"
    assert config.email.use_tls is False
    assert config.email.use_ssl is True


def test_load_realtime_config_discovers_api_keys_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "passivbot"
    repo_root.mkdir()

    api_keys_path = repo_root / "api-keys.json"
    api_keys_path.write_text(
        json.dumps({"binance": {"exchange": "binance", "key": "auto", "secret": "secret"}}),
        encoding="utf-8",
    )

    config_dir = repo_root / "risk_management"
    config_dir.mkdir()
    config_path = config_dir / "realtime.json"
    config_path.write_text(
        json.dumps(
            {
                "accounts": [
                    {
                        "name": "Binance",
                        "exchange": "binance",
                        "api_key_id": "binance",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)

    assert config.accounts[0].credentials["apiKey"] == "auto"

