"""Standalone helpers for provisioning ccxt exchange clients."""

from __future__ import annotations

import logging

try:  # pragma: no cover - optional dependency in tests
    import ccxt.async_support as ccxt
except ModuleNotFoundError:  # pragma: no cover - allow unit tests without ccxt
    ccxt = None  # type: ignore[assignment]

from custom_endpoint_overrides import apply_rest_overrides_to_ccxt, resolve_custom_endpoint_override

logger = logging.getLogger(__name__)


def normalize_exchange_name(exchange: str) -> str:
    """Return the canonical ccxt identifier for the provided ``exchange``."""

    if not exchange:
        return exchange
    lowered = exchange.lower()
    if lowered == "binance":
        return "binanceusdm"
    if lowered.endswith("usdm") or lowered.endswith("futures"):
        return lowered
    if ccxt is None:  # pragma: no cover - unavailable during certain tests
        return lowered
    valid = set(getattr(ccxt, "exchanges", []))
    for suffix in ("usdm", "futures"):
        candidate = f"{lowered}{suffix}"
        if candidate in valid:
            return candidate
    return lowered


def load_ccxt_instance(exchange_id: str, enable_rate_limit: bool = True):
    """Instantiate an async ccxt client with custom endpoint overrides applied."""

    if ccxt is None:  # pragma: no cover - unavailable during certain tests
        raise RuntimeError("ccxt is not installed; install the 'ccxt' extra to use realtime features")

    normalized = normalize_exchange_name(exchange_id)
    try:
        client = getattr(ccxt, normalized)({"enableRateLimit": bool(enable_rate_limit)})
    except Exception as exc:  # pragma: no cover - surface exchange availability issues
        raise RuntimeError(f"ccxt exchange '{normalized}' not available: {exc}") from exc

    try:
        client.options.setdefault("defaultType", "swap")
    except Exception:  # pragma: no cover - options not always mutable
        pass

    try:
        override = resolve_custom_endpoint_override(normalized)
        apply_rest_overrides_to_ccxt(client, override)
    except Exception as exc:  # pragma: no cover - defensive guard around optional overrides
        logger.warning("Failed to apply custom endpoint override for %s: %s", normalized, exc)

    return client


__all__ = ["load_ccxt_instance", "normalize_exchange_name"]
