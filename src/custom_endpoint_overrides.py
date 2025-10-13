"""
Utility helpers for loading and applying custom REST endpoint overrides.

These helpers are intentionally isolated from the rest of the codebase for now.
Subsequent integration steps can import this module to mutate ccxt exchange
instances or other HTTP clients before any network calls are made.

Configuration overview (formalized in ``custom_endpoints.json.example``):

{
    "defaults": {
        "disable_ws": false,
        "rest": {
            "rewrite_domains": {
                "fapi.binance.com": "proxy.example.exchange"
            },
            "url_overrides": {
                "fapiPrivate": "https://proxy.example.exchange/fapi/v1"
            },
            "extra_headers": {
                "X-Demo-Header": "example"
            }
        }
    },
    "exchanges": {
        "binanceusdm": {
            "disable_ws": true,
            "rest": {
                "rewrite_domains": {
                    "fapi.binance.com": "proxy.example.exchange"
                },
                "url_overrides": {
                    "fapiPrivate": "https://proxy.example.exchange/fapi/v1",
                    "fapiPrivateV2": "https://proxy.example.exchange/fapi/v2",
                    "fapiPrivateV3": "https://proxy.example.exchange/fapi/v3"
                }
            }
        }
    }
}

Only REST overrides are handled at this stage. If ``disable_ws`` is ``True``
websocket helpers should decide whether to skip initialisation entirely.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# Canonical fallback values for optional sections
_BASE_EXCHANGE_TEMPLATE = {
    "disable_ws": False,
    "rest": {
        "rewrite_domains": {},
        "url_overrides": {},
        "extra_headers": {},
    },
}

DEFAULT_CONFIG_SEARCH_PATHS: Tuple[str, ...] = (os.path.join("configs", "custom_endpoints.json"),)


class CustomEndpointConfigError(RuntimeError):
    """Raised when the custom endpoint configuration cannot be parsed."""


@dataclass(frozen=True)
class ResolvedEndpointOverride:
    """
    Represents the fully merged override for a single exchange.

    ``rest_domain_rewrites`` maps hostname (or full base URLs) to replacement
    hostnames. ``rest_url_overrides`` replaces concrete ccxt URL keys (e.g.
    ``fapiPrivate``) with explicit URLs. ``rest_extra_headers`` lists headers
    that downstream HTTP clients should send alongside all REST requests routed
    through the override.
    """

    exchange_id: str
    rest_domain_rewrites: Dict[str, str] = field(default_factory=dict)
    rest_url_overrides: Dict[str, str] = field(default_factory=dict)
    rest_extra_headers: Dict[str, str] = field(default_factory=dict)
    disable_ws: bool = False

    def is_noop(self) -> bool:
        return (
            not self.disable_ws
            and not self.rest_domain_rewrites
            and not self.rest_url_overrides
            and not self.rest_extra_headers
        )

    def rewrite_url(self, url: str, *, hostname: Optional[str] = None) -> str:
        """
        Return ``url`` with domain-level rewrites applied.

        Any configured replacement that matches the start of ``url`` is applied.
        Matches can be either bare hostnames (``fapi.binance.com``) or full base
        URLs (``https://fapi.binance.com``).
        """
        if not url:
            return url
        resolved_url = url
        if hostname and "{hostname}" in url:
            resolved_url = url.replace("{hostname}", hostname)

        for old, new in self.rest_domain_rewrites.items():
            if not old:
                continue

            candidates = {old}
            if hostname and "{hostname}" in old:
                candidates.add(old.replace("{hostname}", hostname))

            for candidate in candidates:
                if not candidate:
                    continue
                if resolved_url.startswith(candidate):
                    suffix = resolved_url[len(candidate) :]
                    return new.rstrip("/") + suffix
                if "://" not in candidate:
                    needle = "://" + candidate
                    idx = resolved_url.find(needle)
                    if idx != -1:
                        prefix = resolved_url[: idx + 3]
                        suffix = resolved_url[idx + len(needle) :]
                        return prefix + new + suffix
        return resolved_url

    def apply_to_api_urls(
        self, urls: Mapping[str, str], *, hostname: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Return a new ``dict`` with REST URL overrides applied to the provided
        ccxt ``urls['api']`` mapping.
        """
        updated = dict(urls)
        for key, value in self.rest_url_overrides.items():
            updated[key] = value
        for key, value in list(updated.items()):
            updated[key] = self.rewrite_url(value, hostname=hostname)
        return updated


class CustomEndpointConfig:
    """
    High-level helper for custom endpoint configuration.

    This class loads raw JSON structures and exposes a mergeable API so that
    future integration steps can resolve overrides per exchange.
    """

    def __init__(
        self,
        *,
        source_path: Optional[Path],
        defaults: Mapping[str, object],
        exchanges: Mapping[str, Mapping[str, object]],
    ) -> None:
        self._source_path = source_path
        self._defaults = _ensure_exchange_shape(defaults)
        self._exchanges = {
            key.lower(): _ensure_exchange_shape(value) for key, value in exchanges.items()
        }

    @property
    def source_path(self) -> Optional[Path]:
        return self._source_path

    def available_exchanges(self) -> Iterable[str]:
        return self._exchanges.keys()

    def get_override(self, exchange_id: str) -> Optional[ResolvedEndpointOverride]:
        """
        Resolve the override for ``exchange_id`` (case insensitive). Returns
        ``None`` if no customisation exists.
        """
        if not exchange_id:
            return None
        key = exchange_id.lower()
        merged = _deep_merge_dicts(self._defaults, self._exchanges.get(key))
        resolved = _build_resolved(exchange_id=key, payload=merged)
        return None if resolved.is_noop() else resolved

    def is_empty(self) -> bool:
        return (
            not self.available_exchanges()
            and _build_resolved(exchange_id="defaults", payload=self._defaults).is_noop()
        )


def load_custom_endpoint_config(
    path: Optional[str] = None,
    *,
    search_paths: Iterable[str] = DEFAULT_CONFIG_SEARCH_PATHS,
) -> CustomEndpointConfig:
    """
    Load custom endpoint configuration from JSON.

    If ``path`` is provided it takes precedence. Otherwise the loader searches
    ``search_paths`` in order and returns the first file found. Missing files
    result in an empty configuration rather than an error.
    """
    candidate_path: Optional[Path] = None
    if path:
        candidate_path = Path(path).expanduser().resolve()
        if not candidate_path.is_file():
            raise CustomEndpointConfigError(f"custom endpoint config not found: {candidate_path}")
    else:
        for entry in search_paths:
            resolved = Path(entry).expanduser().resolve()
            if resolved.is_file():
                candidate_path = resolved
                break

    if not candidate_path:
        return CustomEndpointConfig(source_path=None, defaults=_BASE_EXCHANGE_TEMPLATE, exchanges={})

    try:
        with candidate_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise CustomEndpointConfigError(
            f"failed to parse custom endpoint config ({candidate_path}): {exc}"
        ) from exc
    except Exception as exc:
        raise CustomEndpointConfigError(
            f"failed to read custom endpoint config ({candidate_path}): {exc}"
        ) from exc

    defaults = data.get("defaults", {})
    exchanges = data.get("exchanges", {})
    if not isinstance(exchanges, Mapping):
        raise CustomEndpointConfigError("'exchanges' section must be an object mapping exchange ids")

    config = CustomEndpointConfig(
        source_path=candidate_path,
        defaults=defaults,
        exchanges=exchanges,
    )

    logger.debug(
        "Loaded custom endpoint config from %s (exchanges: %s)",
        candidate_path,
        ", ".join(sorted(config.available_exchanges())) or "none",
    )
    return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_exchange_shape(data: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise CustomEndpointConfigError("exchange override must be an object")
    payload = _deep_merge_dicts(_BASE_EXCHANGE_TEMPLATE, data)
    rest = payload.get("rest", {})
    if not isinstance(rest, Mapping):
        raise CustomEndpointConfigError("'rest' override must be an object when provided")
    for key in ("rewrite_domains", "url_overrides", "extra_headers"):
        value = rest.get(key, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise CustomEndpointConfigError(f"'rest.{key}' must be an object mapping strings")
        rest[key] = {str(k): str(v) for k, v in value.items()}
    payload["disable_ws"] = bool(payload.get("disable_ws", False))
    payload["rest"] = dict(rest)
    return dict(payload)


def _deep_merge_dicts(
    base: Mapping[str, object],
    override: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    result: Dict[str, object] = dict(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _build_resolved(exchange_id: str, payload: Mapping[str, object]) -> ResolvedEndpointOverride:
    rest = payload.get("rest", {})
    return ResolvedEndpointOverride(
        exchange_id=exchange_id,
        rest_domain_rewrites=dict(rest.get("rewrite_domains", {})),
        rest_url_overrides=dict(rest.get("url_overrides", {})),
        rest_extra_headers=dict(rest.get("extra_headers", {})),
        disable_ws=bool(payload.get("disable_ws", False)),
    )


_CONFIG_CACHE: Optional[CustomEndpointConfig] = None
_CONFIG_LOAD_PARAMS: Tuple[Optional[str], bool] = (None, True)
_CONFIG_SOURCE_PATH: Optional[Path] = None


def configure_custom_endpoint_loader(
    path: Optional[str],
    *,
    autodiscover: bool = True,
    preloaded: Optional[CustomEndpointConfig] = None,
) -> None:
    """
    Configure the loader to use a specific path or disable auto-discovery.

    Args:
        path: Explicit JSON filepath to load; when provided the loader ignores
              auto-discovery. Use ``None`` together with ``autodiscover=False``
              to disable custom endpoints entirely.
        autodiscover: Whether to search default locations when ``path`` is None.
        preloaded: Optional already-parsed configuration to reuse, avoiding
              an additional file read on next access.
    """
    global _CONFIG_CACHE, _CONFIG_LOAD_PARAMS, _CONFIG_SOURCE_PATH
    _CONFIG_LOAD_PARAMS = (path, bool(autodiscover))
    if preloaded is not None:
        _CONFIG_SOURCE_PATH = preloaded.source_path
    else:
        _CONFIG_SOURCE_PATH = Path(path).expanduser().resolve() if path else None
    _CONFIG_CACHE = preloaded


def get_cached_custom_endpoint_config() -> CustomEndpointConfig:
    """
    Return the cached custom endpoint configuration, loading it on first use.

    If loading fails due to parsing errors the function logs the issue and
    returns an empty configuration to keep the application running.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        path_override, autodiscover = _CONFIG_LOAD_PARAMS
        try:
            if path_override is not None:
                _CONFIG_CACHE = load_custom_endpoint_config(path_override)
            elif autodiscover:
                _CONFIG_CACHE = load_custom_endpoint_config()
            else:
                _CONFIG_CACHE = CustomEndpointConfig(
                    source_path=None,
                    defaults={},
                    exchanges={},
                )
            if _CONFIG_CACHE is not None:
                global _CONFIG_SOURCE_PATH
                _CONFIG_SOURCE_PATH = _CONFIG_CACHE.source_path
        except CustomEndpointConfigError as exc:
            logger.error("Failed to load custom endpoint config: %s", exc)
            _CONFIG_CACHE = CustomEndpointConfig(
                source_path=None,
                defaults={},
                exchanges={},
            )
    return _CONFIG_CACHE


def resolve_custom_endpoint_override(exchange_id: str) -> Optional[ResolvedEndpointOverride]:
    """
    Return the resolved override for ``exchange_id`` or ``None`` when not found.

    ``exchange_id`` should be the normalized ccxt exchange identifier
    (e.g. ``binanceusdm``).
    """
    config = get_cached_custom_endpoint_config()
    return config.get_override(exchange_id) if config else None


def get_custom_endpoint_source() -> Optional[Path]:
    """Return the filesystem path the current overrides were loaded from."""
    return _CONFIG_SOURCE_PATH


def apply_rest_overrides_to_ccxt(
    exchange,
    override: Optional[ResolvedEndpointOverride],
) -> None:
    """
    Mutate a ccxt exchange instance so that REST requests honour ``override``.

    The helper updates ``exchange.urls['api']`` and merges any ``extra_headers``.
    The exchange instance is modified in-place.
    """
    if not override:
        return
    try:
        urls = getattr(exchange, "urls", {})
        if isinstance(urls, Mapping) and "api" in urls:
            original_api = dict(urls["api"])
            hostname = getattr(exchange, "hostname", None)
            updated = override.apply_to_api_urls(original_api, hostname=hostname)
            exchange.urls["api"] = updated
            for key, original_value in original_api.items():
                new_value = updated.get(key)
                if new_value != original_value:
                    logger.info(
                        "Custom endpoint active for %s.%s: %s -> %s",
                        override.exchange_id,
                        key,
                        original_value,
                        new_value,
                    )
            for key in updated:
                if key not in original_api:
                    logger.info(
                        "Custom endpoint added for %s.%s: %s",
                        override.exchange_id,
                        key,
                        updated[key],
                    )
        headers = getattr(exchange, "headers", {}) or {}
        if override.rest_extra_headers:
            merged = dict(headers)
            merged.update(override.rest_extra_headers)
            exchange.headers = merged
            logger.info(
                "Custom endpoint headers for %s merged: %s",
                override.exchange_id,
                override.rest_extra_headers,
            )
    except Exception as exc:
        logger.warning(
            "Failed to apply custom endpoint override for %s: %s",
            getattr(override, "exchange_id", "unknown"),
            exc,
        )


__all__ = [
    "CustomEndpointConfig",
    "CustomEndpointConfigError",
    "ResolvedEndpointOverride",
    "apply_rest_overrides_to_ccxt",
    "configure_custom_endpoint_loader",
    "get_cached_custom_endpoint_config",
    "get_custom_endpoint_source",
    "load_custom_endpoint_config",
    "resolve_custom_endpoint_override",
]
