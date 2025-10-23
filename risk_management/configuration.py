"""Utilities for loading realtime risk management configuration files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass()
class AccountConfig:
    """Configuration for a single exchange account."""

    name: str
    exchange: str
    settle_currency: str = "USDT"
    api_key_id: str | None = None
    credentials: Dict[str, Any] = field(default_factory=dict)
    symbols: List[str] | None = None
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass()
class AuthConfig:
    """Settings for session authentication in the web dashboard."""

    secret_key: str
    users: Mapping[str, str]
    session_cookie_name: str = "risk_dashboard_session"


@dataclass()
class RealtimeConfig:
    """Top level realtime configuration."""

    accounts: List[AccountConfig]
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    auth: AuthConfig | None = None
    account_messages: Dict[str, str] = field(default_factory=dict)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc


def _normalise_credentials(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise credential keys to ccxt's expected names."""

    mapping = {
        "key": "apiKey",
        "apikey": "apiKey",
        "api_key": "apiKey",
        "secret": "secret",
        "password": "password",
        "passphrase": "password",
        "uid": "uid",
    }
    normalised: Dict[str, Any] = {}
    for raw_key, value in data.items():
        if value is None:
            continue
        key = mapping.get(raw_key, raw_key)
        normalised[key] = value
    return normalised


def _merge_credentials(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(secondary)
    merged.update(primary)
    return _normalise_credentials(merged)


def _parse_accounts(
    accounts_raw: Iterable[Mapping[str, Any]],
    api_keys: Mapping[str, Mapping[str, Any]] | None,
) -> List[AccountConfig]:
    accounts: List[AccountConfig] = []
    for raw in accounts_raw:
        if not raw.get("enabled", True):
            continue
        api_key_id = raw.get("api_key_id")
        credentials: Mapping[str, Any] = raw.get("credentials", {})
        exchange = raw.get("exchange")
        if api_key_id:
            if api_keys is None:
                raise ValueError(
                    f"Account '{raw.get('name')}' references api_key_id '{api_key_id}' but no api key file was provided"
                )
            if api_key_id not in api_keys:
                raise ValueError(
                    f"Account '{raw.get('name')}' references unknown api_key_id '{api_key_id}'"
                )
            key_payload = api_keys[api_key_id]
            if not exchange:
                exchange = key_payload.get("exchange")
            credentials = _merge_credentials(credentials, key_payload)
        else:
            credentials = _normalise_credentials(credentials)
        if not exchange:
            raise ValueError(
                f"Account '{raw.get('name')}' must specify an exchange either directly or via the api key entry."
            )
        account = AccountConfig(
            name=str(raw.get("name", exchange)),
            exchange=str(exchange),
            settle_currency=str(raw.get("settle_currency", "USDT")),
            api_key_id=api_key_id,
            credentials=dict(credentials),
            symbols=list(raw.get("symbols") or []) or None,
            params=dict(raw.get("params", {})),
            enabled=bool(raw.get("enabled", True)),
        )
        accounts.append(account)
    return accounts


def _parse_auth(auth_raw: Mapping[str, Any] | None) -> AuthConfig | None:
    if not auth_raw:
        return None
    secret_key = auth_raw.get("secret_key")
    if not secret_key:
        raise ValueError("Authentication configuration requires a 'secret_key'.")
    users_raw = auth_raw.get("users")
    if not users_raw:
        raise ValueError("Authentication configuration requires at least one user entry.")
    if isinstance(users_raw, Mapping):
        users = dict(users_raw)
    else:
        users = {str(entry["username"]): str(entry["password_hash"]) for entry in users_raw}
    session_cookie = str(auth_raw.get("session_cookie_name", "risk_dashboard_session"))
    return AuthConfig(secret_key=str(secret_key), users=users, session_cookie_name=session_cookie)


def load_realtime_config(path: Path) -> RealtimeConfig:
    """Load a realtime configuration file."""

    config = _load_json(path)
    api_keys_file = config.get("api_keys_file")
    api_keys: Mapping[str, Any] | None = None
    if api_keys_file:
        api_keys_path = (path.parent / api_keys_file).resolve()
        api_keys_raw = _load_json(api_keys_path)
        api_keys = {
            key: value
            for key, value in api_keys_raw.items()
            if isinstance(value, Mapping) and key != "referrals"
        }
    accounts_raw = config.get("accounts")
    if not accounts_raw:
        raise ValueError("Realtime configuration must include at least one account entry.")
    accounts = _parse_accounts(accounts_raw, api_keys)
    alert_thresholds = {str(k): float(v) for k, v in config.get("alert_thresholds", {}).items()}
    notification_channels = [str(item) for item in config.get("notification_channels", [])]
    auth = _parse_auth(config.get("auth"))
    return RealtimeConfig(
        accounts=accounts,
        alert_thresholds=alert_thresholds,
        notification_channels=notification_channels,
        auth=auth,
    )
