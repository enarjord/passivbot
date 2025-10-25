"""Utilities for loading realtime risk management configuration files."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Set


logger = logging.getLogger(__name__)


def _ensure_debug_logging_enabled() -> None:
    """Raise logging verbosity when debug API payloads are requested."""

    root_logger = logging.getLogger()
    if root_logger.level in {
        logging.NOTSET,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    } or root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)

    risk_logger = logging.getLogger("risk_management")
    if risk_logger.level in {logging.NOTSET} or risk_logger.level > logging.DEBUG:
        risk_logger.setLevel(logging.DEBUG)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Return a boolean for ``value`` supporting common string representations."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "default", "auto"}:
            return default
        if lowered in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "disable"}:
            return False
    return bool(value)


@dataclass()
class CustomEndpointSettings:
    """Settings controlling how custom endpoint overrides are loaded."""

    path: str | None = None
    autodiscover: bool = True


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
    debug_api_payloads: bool = False


@dataclass()
class AuthConfig:
    """Settings for session authentication in the web dashboard."""

    secret_key: str
    users: Mapping[str, str]
    session_cookie_name: str = "risk_dashboard_session"
    https_only: bool = True


@dataclass()
class EmailSettings:
    """SMTP configuration used to dispatch alert emails."""

    host: str
    port: int = 587
    username: str | None = None
    password: str | None = None
    use_tls: bool = True
    use_ssl: bool = False
    sender: str | None = None


@dataclass()
class GrafanaDashboardConfig:
    """Description of a Grafana dashboard or panel to embed."""

    title: str
    url: str
    description: str | None = None
    height: int | None = None


@dataclass()
class GrafanaConfig:
    """Settings for embedding Grafana dashboards in the web UI."""

    dashboards: List[GrafanaDashboardConfig] = field(default_factory=list)
    default_height: int = 600
    theme: str = "dark"
    base_url: str | None = None


@dataclass()
class RealtimeConfig:
    """Top level realtime configuration."""

    accounts: List[AccountConfig]
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list)
    auth: AuthConfig | None = None
    account_messages: Dict[str, str] = field(default_factory=dict)
    custom_endpoints: CustomEndpointSettings | None = None
    email: EmailSettings | None = None
    config_root: Path | None = None
    debug_api_payloads: bool = False
    reports_dir: Path | None = None
    grafana: GrafanaConfig | None = None


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc


def _normalise_credentials(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise credential keys to ccxt's expected names."""

    key_aliases = {
        "key": "apiKey",
        "apikey": "apiKey",
        "api_key": "apiKey",
        "api-key": "apiKey",
        "secret": "secret",
        "secret_key": "secret",
        "secretkey": "secret",
        "secret-key": "secret",
        "apisecret": "secret",
        "api_secret": "secret",
        "api-secret": "secret",
        "password": "password",
        "passphrase": "password",
        "pass_phrase": "password",
        "pass-phrase": "password",
        "uid": "uid",
        "user_id": "uid",
        "wallet_address": "walletAddress",
        "walletaddress": "walletAddress",
        "private_key": "privateKey",
        "privatekey": "privateKey",
        "ccxt_config": "ccxt",
        "ccxtconfig": "ccxt",
    }
    normalised: Dict[str, Any] = {}
    for raw_key, value in data.items():
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                continue
        key_lookup = raw_key.lower().replace(" ", "").replace("-", "_")
        key = key_aliases.get(key_lookup, raw_key)
        if key == "exchange":
            # ``exchange`` is metadata in api key files and should not be
            # treated as authentication input for ccxt clients.
            continue
        if key in {"headers", "options"} and isinstance(value, Mapping):
            existing = normalised.setdefault(key, {})
            existing.update(value)
            continue
        normalised[key] = value
    return normalised


def _merge_credentials(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _normalise_credentials(secondary)
    primary_normalised = _normalise_credentials(primary)
    for key, value in primary_normalised.items():
        if key in {"headers", "options"} and isinstance(value, Mapping):
            existing = merged.setdefault(key, {})
            existing.update(value)
        else:
            merged[key] = value
    return merged


def _iter_candidate_roots(config_root: Path | None) -> Iterable[Path]:
    """Yield directories to inspect when auto-discovering shared files."""

    module_root = Path(__file__).resolve().parent
    repository_root = module_root.parent

    bases = [config_root, Path.cwd(), module_root, repository_root]

    seen: Set[Path] = set()
    for base in bases:
        if base is None:
            continue
        try:
            resolved = base.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved
        for parent in resolved.parents:
            if parent in seen:
                continue
            seen.add(parent)
            yield parent


def _discover_api_keys_path(config_root: Path | None) -> Path | None:
    """Return the first ``api-keys.json`` found relative to common roots."""

    for root in _iter_candidate_roots(config_root):
        candidate = root / "api-keys.json"
        if candidate.is_file():
            return candidate
    return None


def _parse_custom_endpoints(settings: Any) -> CustomEndpointSettings | None:
    """Return structured custom endpoint settings from ``settings``."""

    if settings is None:
        return None
    if isinstance(settings, Mapping):
        path_raw = settings.get("path")
        path = str(path_raw).strip() if path_raw not in (None, "") else None
        autodiscover = bool(settings.get("autodiscover", True))
        return CustomEndpointSettings(path=path or None, autodiscover=autodiscover)
    value = str(settings).strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"none", "off", "disable"}:
        return CustomEndpointSettings(path=None, autodiscover=False)
    return CustomEndpointSettings(path=value, autodiscover=False)


def _parse_email_settings(settings: Any) -> EmailSettings | None:
    """Return SMTP settings when provided in the realtime configuration."""

    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError("Email settings must be provided as an object in the configuration file.")

    host_raw = settings.get("host")
    if not host_raw or not str(host_raw).strip():
        raise ValueError("Email settings must include a non-empty 'host'.")
    host = str(host_raw).strip()

    port_raw = settings.get("port", 587)
    try:
        port = int(port_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Email settings 'port' must be an integer.") from exc

    username = settings.get("username")
    password = settings.get("password")
    sender = settings.get("sender")
    use_tls = _coerce_bool(settings.get("use_tls"), True)
    use_ssl = _coerce_bool(settings.get("use_ssl"), False)

    return EmailSettings(
        host=host,
        port=port,
        username=str(username).strip() if username not in (None, "") else None,
        password=str(password).strip() if password not in (None, "") else None,
        sender=str(sender).strip() if sender not in (None, "") else None,
        use_tls=use_tls,
        use_ssl=use_ssl,
    )


def _parse_grafana_config(settings: Any) -> GrafanaConfig | None:
    """Return Grafana embedding settings from ``settings``."""

    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise TypeError("Grafana settings must be provided as an object in the configuration file.")

    dashboards_raw = settings.get("dashboards")
    if dashboards_raw in (None, []):
        return None
    if not isinstance(dashboards_raw, Iterable):
        raise TypeError("Grafana 'dashboards' must be an array of dashboard definitions.")

    dashboards: List[GrafanaDashboardConfig] = []
    for entry in dashboards_raw:
        if not isinstance(entry, Mapping):
            raise TypeError(
                "Each Grafana dashboard entry must be an object with at least a title and url."
            )
        url_raw = entry.get("url")
        if not url_raw or not str(url_raw).strip():
            raise ValueError("Grafana dashboard entries require a non-empty 'url'.")
        title_raw = entry.get("title", "Grafana dashboard")
        description_raw = entry.get("description")
        height_raw = entry.get("height")

        height: int | None = None
        if height_raw not in (None, ""):
            try:
                height = int(height_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Grafana dashboard 'height' must be an integer when provided."
                ) from exc
            if height <= 0:
                raise ValueError(
                    "Grafana dashboard 'height' must be greater than zero when provided."
                )

        dashboards.append(
            GrafanaDashboardConfig(
                title=str(title_raw).strip() or "Grafana dashboard",
                url=str(url_raw).strip(),
                description=str(description_raw).strip()
                if description_raw not in (None, "")
                else None,
                height=height,
            )
        )

    default_height_raw = settings.get("default_height", 600)
    try:
        default_height = int(default_height_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grafana 'default_height' must be an integer.") from exc
    if default_height <= 0:
        raise ValueError("Grafana 'default_height' must be greater than zero.")

    theme_raw = settings.get("theme", "dark")
    theme = str(theme_raw).strip() or "dark"

    base_url_raw = settings.get("base_url")
    base_url = str(base_url_raw).strip() if base_url_raw not in (None, "") else None

    return GrafanaConfig(
        dashboards=dashboards,
        default_height=default_height,
        theme=theme,
        base_url=base_url,
    )


def _parse_accounts(
    accounts_raw: Iterable[Mapping[str, Any]],
    api_keys: Mapping[str, Mapping[str, Any]] | None,
    debug_api_payloads_default: bool = False,
) -> List[AccountConfig]:
    accounts: List[AccountConfig] = []
    debug_requested = False
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
        debug_api_payloads = _coerce_bool(
            raw.get("debug_api_payloads"), debug_api_payloads_default
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
            debug_api_payloads=debug_api_payloads,
        )
        accounts.append(account)
        if debug_api_payloads:
            debug_requested = True
    if debug_requested:
        _ensure_debug_logging_enabled()
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
    https_only = _coerce_bool(auth_raw.get("https_only"), True)
    return AuthConfig(
        secret_key=str(secret_key),
        users=users,
        session_cookie_name=session_cookie,
        https_only=https_only,
    )


def load_realtime_config(path: Path) -> RealtimeConfig:
    """Load a realtime configuration file."""

    config = _load_json(path)
    config_root = path.parent.resolve()
    api_keys_file = config.get("api_keys_file")
    api_keys: Dict[str, Mapping[str, Any]] | None = None
    api_keys_path: Path | None = None
    if api_keys_file:
        api_keys_path = Path(str(api_keys_file)).expanduser()
        if not api_keys_path.is_absolute():
            api_keys_path = (path.parent / api_keys_path).resolve()
        else:
            api_keys_path = api_keys_path.resolve()
    else:
        api_keys_path = _discover_api_keys_path(config_root)
        if api_keys_path:
            logger.info("Using api keys from %s", api_keys_path)
    if api_keys_path:
        api_keys_raw = _load_json(api_keys_path)
        flattened: Dict[str, Mapping[str, Any]] = {}
        for key, value in api_keys_raw.items():
            if key == "referrals" or not isinstance(value, Mapping):
                continue
            if key.lower() == "users":
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, Mapping):
                        flattened[sub_key] = sub_value
                continue
            flattened[key] = value
        api_keys = flattened
    accounts_raw = config.get("accounts")
    if not accounts_raw:
        raise ValueError("Realtime configuration must include at least one account entry.")
    debug_api_payloads_default = _coerce_bool(config.get("debug_api_payloads"), False)
    if debug_api_payloads_default:
        _ensure_debug_logging_enabled()

    accounts = _parse_accounts(accounts_raw, api_keys, debug_api_payloads_default)
    alert_thresholds = {str(k): float(v) for k, v in config.get("alert_thresholds", {}).items()}
    notification_channels = [str(item) for item in config.get("notification_channels", [])]
    auth = _parse_auth(config.get("auth"))
    custom_endpoints = _parse_custom_endpoints(config.get("custom_endpoints"))
    email_settings = _parse_email_settings(config.get("email"))
    grafana_settings = _parse_grafana_config(config.get("grafana"))
    reports_dir_value = config.get("reports_dir")
    reports_dir: Path | None = None
    if reports_dir_value:
        candidate = Path(str(reports_dir_value)).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        else:
            candidate = candidate.resolve()
        reports_dir = candidate

    if custom_endpoints and custom_endpoints.path:
        resolved_path = Path(custom_endpoints.path).expanduser()
        if not resolved_path.is_absolute():
            resolved_path = (path.parent / resolved_path).resolve()
        else:
            resolved_path = resolved_path.resolve()
        custom_endpoints = CustomEndpointSettings(
            path=str(resolved_path),
            autodiscover=custom_endpoints.autodiscover,
        )

    return RealtimeConfig(
        accounts=accounts,
        alert_thresholds=alert_thresholds,
        notification_channels=notification_channels,
        auth=auth,
        custom_endpoints=custom_endpoints,
        email=email_settings,
        config_root=config_root,
        debug_api_payloads=debug_api_payloads_default,
        reports_dir=reports_dir,
        grafana=grafana_settings,
    )
