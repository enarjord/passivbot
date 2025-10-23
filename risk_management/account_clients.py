"""Account client implementations for the risk management tooling."""

from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any, Dict, Iterable, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency in some envs
    import ccxt.async_support as ccxt_async
    from ccxt.base.errors import BaseError
except ModuleNotFoundError:  # pragma: no cover - allow tests without ccxt
    ccxt_async = None  # type: ignore[assignment]

    class BaseError(Exception):
        """Fallback error when ccxt is unavailable."""

        pass

try:  # pragma: no cover - passivbot is optional when running tests
    from passivbot.utils import load_ccxt_instance, normalize_exchange_name
except (ModuleNotFoundError, ImportError):  # pragma: no cover - allow running without passivbot
    load_ccxt_instance = None  # type: ignore[assignment]

    def normalize_exchange_name(exchange: str) -> str:  # type: ignore[override]
        return exchange


logger = logging.getLogger(__name__)


class AccountClientProtocol(abc.ABC):
    """Abstract interface for realtime account clients."""

    config: "AccountConfig"

    @abc.abstractmethod
    async def fetch(self) -> Dict[str, Any]:
        """Return a mapping with account balance, positions, and metadata."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close any open network connections."""


def _apply_credentials(client: Any, credentials: Mapping[str, Any]) -> None:
    """Populate authentication fields on a ccxt client."""

    for key, value in credentials.items():
        if key in ("apiKey", "secret", "password", "uid", "login"):
            setattr(client, key, value)
        elif key == "headers" and isinstance(value, Mapping):
            headers = getattr(client, "headers", {}) or {}
            headers.update(value)
            client.headers = headers
        else:
            try:
                setattr(client, key, value)
            except Exception:
                logger.debug("Ignored unsupported credential field %s", key)


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
        from .realtime import _extract_balance, _parse_position  # circular safe import

        balance_value = _extract_balance(balance_raw, self.config.settle_currency)
        positions_raw: Iterable[Mapping[str, Any]] = []
        positions: list[Dict[str, Any]] = []
        if hasattr(self.client, "fetch_positions"):
            try:
                positions_raw = await self.client.fetch_positions(params=self._positions_params)
            except BaseError as exc:
                logger.warning(
                    "Failed to fetch positions for %s: %s", self.config.name, exc, exc_info=True
                )
        for position_raw in positions_raw or []:
            parsed = _parse_position(position_raw, balance_value)
            if parsed is not None:
                positions.append(parsed)
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
