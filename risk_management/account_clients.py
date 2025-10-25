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

    @abc.abstractmethod
    async def kill_switch(self) -> Dict[str, Any]:
        """Cancel open orders and close all positions for the account."""


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


def _suppress_open_orders_warning(client: Any) -> None:
    """Prevent ccxt from escalating open-order symbol warnings to exceptions."""

    options = getattr(client, "options", None)
    if isinstance(options, MutableMapping):
        options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    else:
        setattr(client, "options", {"warnOnFetchOpenOrdersWithoutSymbol": False})


def _is_symbol_specific_open_orders_warning(error: BaseError) -> bool:
    """Return ``True`` when ``error`` is the ccxt warning about missing symbols."""

    message = str(error)
    return (
        "fetchOpenOrders() WARNING" in message
        and "without specifying a symbol" in message
        and "warnOnFetchOpenOrdersWithoutSymbol" in message
    )


def _instantiate_ccxt_client(exchange_id: str, credentials: Mapping[str, Any]) -> Any:
    """Instantiate a ccxt async client honoring passivbot customisations when available."""

    normalized = normalize_exchange_name(exchange_id)
    rate_limited = bool(credentials.get("enableRateLimit", True))

    def _suppress_open_orders_warning(client: Any) -> None:
        """Prevent ccxt from raising on informational open-order warnings."""

        options = getattr(client, "options", None)
        if isinstance(options, MutableMapping):
            options.setdefault("warnOnFetchOpenOrdersWithoutSymbol", False)
        else:
            setattr(client, "options", {"warnOnFetchOpenOrdersWithoutSymbol": False})

    if load_ccxt_instance is not None:
        client = load_ccxt_instance(normalized, enable_rate_limit=rate_limited)
        _apply_credentials(client, credentials)
        _disable_fetch_currencies(client)
        _suppress_open_orders_warning(client)
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
    _suppress_open_orders_warning(client)
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
        self._orders_params = dict(config.params.get("orders", {}))
        self._close_params = dict(config.params.get("close", {}))
        self._markets_loaded: asyncio.Lock | None = None
        self._debug_api_payloads = bool(config.debug_api_payloads)

    def _refresh_open_order_preferences(self) -> None:
        """Re-apply exchange options that silence noisy open-order warnings."""

        try:
            _suppress_open_orders_warning(self.client)
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "Failed to update open-order warning preference for %s", self.config.name
            )

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
        from .realtime import (  # circular safe import
            _extract_balance,
            _parse_order,
            _parse_position,
        )

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

        open_orders: list[Dict[str, Any]] = []
        if hasattr(self.client, "fetch_open_orders"):
            try:
                self._refresh_open_order_preferences()
                raw_orders: Iterable[Mapping[str, Any]] | None = None
                if self.config.symbols:
                    combined: list[Mapping[str, Any]] = []
                    for symbol in self.config.symbols:
                        params = dict(self._orders_params)
                        raw = await self.client.fetch_open_orders(symbol, params=params)
                        self._log_exchange_payload(
                            "fetch_open_orders",
                            raw,
                            {**params, "symbol": symbol},
                        )
                        if raw:
                            combined.extend(raw)
                    raw_orders = combined
                else:
                    raw_orders = await self.client.fetch_open_orders(params=self._orders_params)
                    self._log_exchange_payload(
                        "fetch_open_orders", raw_orders, self._orders_params
                    )
                for order_raw in raw_orders or []:
                    parsed_order = _parse_order(order_raw)
                    if parsed_order is not None:
                        open_orders.append(parsed_order)
            except BaseError as exc:
                if _is_symbol_specific_open_orders_warning(exc):
                    logger.info(
                        "Exchange %s requires a symbol when fetching open orders; skipping.",
                        self.config.name,
                    )
                else:
                    logger.warning(
                        "Failed to fetch open orders for %s: %s",
                        self.config.name,
                        exc,
                        exc_info=True,
                    )
                if self._debug_api_payloads:
                    logger.debug(
                        "[%s] fetch_open_orders params=%s raised=%s",
                        self.config.name,
                        _stringify_payload(self._orders_params),
                        exc,
                    )
        elif self._debug_api_payloads:
            logger.debug(
                "[%s] fetch_open_orders not available on exchange client", self.config.name
            )

        return {
            "name": self.config.name,
            "balance": balance_value,
            "positions": positions,
            "open_orders": open_orders,
        }

    async def close(self) -> None:
        await self.client.close()

    async def kill_switch(self) -> Dict[str, Any]:
        await self._ensure_markets()
        summary: Dict[str, Any] = {
            "cancelled_orders": [],
            "failed_order_cancellations": [],
            "closed_positions": [],
            "failed_position_closures": [],
        }
        await self._cancel_open_orders(summary)
        await self._close_positions(summary)
        return summary

    async def _cancel_open_orders(self, summary: Dict[str, Any]) -> None:
        if hasattr(self.client, "cancel_all_orders"):
            try:
                if self.config.symbols:
                    for symbol in self.config.symbols:
                        await self.client.cancel_all_orders(symbol, params=self._orders_params)
                        summary["cancelled_orders"].append({"symbol": symbol})
                else:
                    await self.client.cancel_all_orders(params=self._orders_params)
                    summary["cancelled_orders"].append({"symbol": None})
                return
            except BaseError as exc:
                logger.warning(
                    "cancel_all_orders failed for %s: %s", self.config.name, exc, exc_info=True
                )
                summary["failed_order_cancellations"].append(
                    {"error": str(exc), "method": "cancel_all_orders"}
                )

        if not hasattr(self.client, "fetch_open_orders"):
            return

        try:
            self._refresh_open_order_preferences()
            symbols = self.config.symbols or [None]
            for symbol in symbols:
                params = dict(self._orders_params)
                orders = await self.client.fetch_open_orders(symbol, params=params)
                for order in orders or []:
                    order_id = order.get("id") or order.get("clientOrderId")
                    if not order_id:
                        continue
                    try:
                        if symbol:
                            await self.client.cancel_order(order_id, symbol, params=params)
                        else:
                            await self.client.cancel_order(order_id, params=params)
                        summary["cancelled_orders"].append(
                            {"symbol": symbol or order.get("symbol"), "order_id": order_id}
                        )
                    except BaseError as exc:
                        logger.warning(
                            "Failed to cancel order %s on %s: %s",
                            order_id,
                            self.config.name,
                            exc,
                            exc_info=True,
                        )
                        summary["failed_order_cancellations"].append(
                            {
                                "symbol": symbol or order.get("symbol"),
                                "order_id": order_id,
                                "error": str(exc),
                            }
                        )
        except BaseError as exc:
            if _is_symbol_specific_open_orders_warning(exc):
                logger.info(
                    "Exchange %s requires a symbol when cancelling open orders; skipping.",
                    self.config.name,
                )
            else:
                logger.warning(
                    "Failed to enumerate open orders for %s: %s",
                    self.config.name,
                    exc,
                    exc_info=True,
                )

    async def _close_positions(self, summary: Dict[str, Any]) -> None:
        if not hasattr(self.client, "fetch_positions"):
            return

        try:
            positions = await self.client.fetch_positions(params=self._positions_params)
        except BaseError as exc:
            logger.warning(
                "Failed to fetch positions for kill switch on %s: %s",
                self.config.name,
                exc,
                exc_info=True,
            )
            summary["failed_position_closures"].append({"error": str(exc)})
            return

        for position in positions or []:
            size = position.get("contracts") or position.get("size") or position.get("amount")
            symbol = position.get("symbol") or position.get("id")
            if not symbol:
                continue
            try:
                qty = abs(float(size))
            except (TypeError, ValueError):
                continue
            if qty == 0:
                continue
            side = "sell" if float(size) > 0 else "buy"
            params = dict(self._close_params)
            params.setdefault("reduceOnly", True)
            try:
                await self.client.create_order(symbol, "market", side, qty, params=params)
                summary["closed_positions"].append({"symbol": symbol, "side": side, "amount": qty})
            except BaseError as exc:
                logger.warning(
                    "Failed to close position %s on %s: %s",
                    symbol,
                    self.config.name,
                    exc,
                    exc_info=True,
                )
                summary["failed_position_closures"].append(
                    {"symbol": symbol, "side": side, "amount": qty, "error": str(exc)}
                )


__all__ = [
    "AccountClientProtocol",
    "CCXTAccountClient",
]
