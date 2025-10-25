"""Account client implementations for the risk management tooling."""

from __future__ import annotations

import abc
import asyncio
import json
import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency in some envs
    import ccxt.async_support as ccxt_async
    from ccxt.base.errors import BaseError
except ModuleNotFoundError:  # pragma: no cover - allow tests without ccxt
    ccxt_async = None  # type: ignore[assignment]

    class BaseError(Exception):
        """Fallback error when ccxt is unavailable."""

        pass

from custom_endpoint_overrides import (
    apply_rest_overrides_to_ccxt,
    resolve_custom_endpoint_override,
)

try:  # pragma: no cover - passivbot is optional when running tests
    from passivbot.utils import load_ccxt_instance, normalize_exchange_name
except (ModuleNotFoundError, ImportError):  # pragma: no cover - allow running without passivbot
    load_ccxt_instance = None  # type: ignore[assignment]

    def normalize_exchange_name(exchange: str) -> str:  # type: ignore[override]
        return exchange


logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    """Coerce non-serialisable objects into JSON-compatible types."""

    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _stringify_payload(payload: Any) -> str:
    """Return a JSON string representation for logging purposes."""

    try:
        return json.dumps(payload, ensure_ascii=False, default=_json_default, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        return repr(payload)


class AccountClientProtocol(abc.ABC):
    """Abstract interface for realtime account clients."""

    config: "AccountConfig"

    @abc.abstractmethod
    async def fetch(self) -> Dict[str, Any]:
        """Return a mapping with account balance, positions, and metadata."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close any open network connections."""


def _set_exchange_field(client: Any, key: str, value: Any, aliases: Sequence[str]) -> None:
    """Assign ``value`` to ``key`` on ``client`` and store aliases when possible."""

    config = getattr(client, "config", None)
    keys = tuple(dict.fromkeys((key, *aliases)))  # preserve order, drop duplicates
    for attr in keys:
        try:
            setattr(client, attr, value)
        except Exception:
            logger.debug("Ignored unsupported credential attribute %s", attr)
        if isinstance(config, MutableMapping):
            try:
                config[attr] = value
            except Exception:
                logger.debug("Failed to persist credential %s in exchange config", attr)


def _format_header_placeholders(
    headers: MutableMapping[str, Any], values: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Expand placeholder tokens in ``headers`` using ``values`` as the source."""

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    alias_map = {
        "apiKey": ("api_key", "key"),
        "secret": ("apiSecret", "secret_key", "secretKey"),
        "password": ("passphrase", "pass_phrase"),
        "uid": ("user_id",),
        "walletAddress": ("wallet_address",),
        "privateKey": ("private_key",),
    }

    substitutions: Dict[str, str] = {}

    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)):
            substitutions[key] = str(value)

    for canonical, aliases in alias_map.items():
        canonical_value = substitutions.get(canonical)
        if canonical_value is None:
            continue
        for alias in aliases:
            substitutions.setdefault(alias, canonical_value)

    formatter = _DefaultDict(substitutions)

    updated = False
    for header_key, header_value in list(headers.items()):
        if isinstance(header_value, str) and "{" in header_value and "}" in header_value:
            try:
                formatted = header_value.format_map(formatter)
            except Exception:  # pragma: no cover - defensive against malformed format strings
                continue
            if formatted != header_value:
                headers[header_key] = formatted
                updated = True

    if updated:
        # ``headers`` may be an exchange-specific structure; normalise to ``dict`` to
        # avoid subtle mutation bugs when ccxt clones the mapping.
        return dict(headers)

    return None


def _apply_credentials(client: Any, credentials: Mapping[str, Any]) -> None:
    """Populate authentication fields on a ccxt client."""

    sensitive_fields = {"apiKey", "secret", "password", "uid", "login", "walletAddress", "privateKey"}
    alias_map = {
        "apiKey": ("api_key", "key"),
        "secret": ("apiSecret", "secret_key", "secretKey"),
        "password": ("passphrase",),
        "uid": (),
        "login": (),
        "walletAddress": ("wallet_address",),
        "privateKey": ("private_key",),
    }

    for key, value in credentials.items():
        if value is None:
            continue
        if key in sensitive_fields:
            aliases = alias_map.get(key, ())
            _set_exchange_field(client, key, value, aliases)
        elif key == "headers" and isinstance(value, Mapping):
            headers = getattr(client, "headers", {}) or {}
            headers.update(value)
            client.headers = headers
        elif key == "options" and isinstance(value, Mapping):
            options = getattr(client, "options", None)
            if isinstance(options, MutableMapping):
                options.update(value)
            else:
                setattr(client, "options", dict(value))
        elif key == "ccxt" and isinstance(value, Mapping):
            # Some configurations expose an explicit ``ccxt`` block mirroring
            # passivbot's "ccxt_config" support. Apply the known keys while
            # falling back to attribute assignment for any extras.
            _apply_credentials(client, value)
        else:
            try:
                setattr(client, key, value)
            except Exception:
                logger.debug("Ignored unsupported credential field %s", key)

    headers = getattr(client, "headers", None)
    if isinstance(headers, MutableMapping):
        formatted = _format_header_placeholders(headers, credentials)
        if formatted is not None:
            client.headers = formatted


def _disable_fetch_currencies(client: Any) -> None:
    """Disable ccxt currency lookups that require authenticated endpoints."""

    options = getattr(client, "options", None)
    if isinstance(options, MutableMapping):
        # ccxt exchanges often respect ``options['fetchCurrencies']`` when deciding
        # whether to hit authenticated endpoints while loading markets.
        options["fetchCurrencies"] = False
        # Suppress any warnings about skipping currency downloads without keys.
        options["warnOnFetchCurrenciesWithoutApiKey"] = False

    has = getattr(client, "has", None)
    if isinstance(has, MutableMapping):
        # Some exchange implementations consult ``has['fetchCurrencies']``
        # instead of the options flag, therefore toggle both to cover either
        # code path.
        has["fetchCurrencies"] = False


def _instantiate_ccxt_client(exchange_id: str, credentials: Mapping[str, Any]) -> Any:
    """Instantiate a ccxt async client honoring passivbot customisations when available."""

    normalized = normalize_exchange_name(exchange_id)
    rate_limited = bool(credentials.get("enableRateLimit", True))

    if load_ccxt_instance is not None:
        client = load_ccxt_instance(normalized, enable_rate_limit=rate_limited)
        _apply_credentials(client, credentials)
        _disable_fetch_currencies(client)
        override = resolve_custom_endpoint_override(normalized)
        apply_rest_overrides_to_ccxt(client, override)
        return client

    if ccxt_async is None:
        raise RuntimeError(
            "ccxt is required to create realtime exchange clients. Install it via 'pip install ccxt'."
        )

    try:
        exchange_class = getattr(ccxt_async, normalized)
    except AttributeError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt.") from exc

    params: MutableMapping[str, Any] = dict(credentials)
    params.setdefault("enableRateLimit", rate_limited)
    client = exchange_class(params)
    _apply_credentials(client, credentials)
    _disable_fetch_currencies(client)
    override = resolve_custom_endpoint_override(normalized)
    apply_rest_overrides_to_ccxt(client, override)
    return client


class CCXTAccountClient(AccountClientProtocol):
    """Realtime account client backed by ccxt asynchronous exchanges."""

    def __init__(self, config: "AccountConfig") -> None:
        from .configuration import AccountConfig  # lazy import to avoid cycle

        if not isinstance(config, AccountConfig):  # pragma: no cover - defensive
            raise TypeError("config must be an AccountConfig instance")

        self.config = config
        credentials = dict(config.credentials)
        credentials.setdefault("enableRateLimit", True)
        self.client = _instantiate_ccxt_client(config.exchange, credentials)
        self._balance_params = dict(config.params.get("balance", {}))
        self._positions_params = dict(config.params.get("positions", {}))
        self._markets_loaded: asyncio.Lock | None = None
        self._debug_api_payloads = bool(config.debug_api_payloads)

    def _log_exchange_payload(
        self, operation: str, payload: Any, params: Mapping[str, Any] | None
    ) -> None:
        if not self._debug_api_payloads:
            return
        params_repr = _stringify_payload(params or {}) if params else "{}"
        payload_repr = _stringify_payload(payload)
        logger.debug(
            "[%s] %s response params=%s payload=%s",
            self.config.name,
            operation,
            params_repr,
            payload_repr,
        )

    async def _ensure_markets(self) -> None:
        lock = self._markets_loaded
        if lock is None:
            lock = asyncio.Lock()
            self._markets_loaded = lock
        async with lock:
            if getattr(self.client, "markets", None):
                return
            await self.client.load_markets()

    async def fetch(self) -> Dict[str, Any]:
        await self._ensure_markets()
        balance_raw = await self.client.fetch_balance(params=self._balance_params)
        self._log_exchange_payload("fetch_balance", balance_raw, self._balance_params)
        from .realtime import _extract_balance, _parse_position  # circular safe import

        balance_value = _extract_balance(balance_raw, self.config.settle_currency)
        positions_raw: Iterable[Mapping[str, Any]] = []
        positions: list[Dict[str, Any]] = []
        if hasattr(self.client, "fetch_positions"):
            try:
                positions_raw = await self.client.fetch_positions(params=self._positions_params)
                self._log_exchange_payload(
                    "fetch_positions", positions_raw, self._positions_params
                )
            except BaseError as exc:
                logger.warning(
                    "Failed to fetch positions for %s: %s", self.config.name, exc, exc_info=True
                )
                if self._debug_api_payloads:
                    logger.debug(
                        "[%s] fetch_positions params=%s raised=%s",
                        self.config.name,
                        _stringify_payload(self._positions_params),
                        exc,
                    )
        for position_raw in positions_raw or []:
            parsed = _parse_position(position_raw, balance_value)
            if parsed is not None:
                positions.append(parsed)
        if not hasattr(self.client, "fetch_positions") and self._debug_api_payloads:
            logger.debug(
                "[%s] fetch_positions not available on exchange client", self.config.name
            )
        return {
            "name": self.config.name,
            "balance": balance_value,
            "positions": positions,
        }

    async def close(self) -> None:
        await self.client.close()


__all__ = [
    "AccountClientProtocol",
    "CCXTAccountClient",
]
