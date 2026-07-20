from __future__ import annotations

import re
import sys

EXCEPTION_TYPE_MAX_LEN = 80
_SENSITIVE_EXCEPTION_TYPE_RE = re.compile(
    r"(?i)(?:api_?key|apikey|authorization|cookie|passphrase|password|private_?key|"
    r"privatekey|secret|signature|token|wallet_?address|walletaddress)"
)
_EXCEPTION_STATUS_RE = re.compile(r"[0-9]{1,3}")
_EXCEPTION_CODE_RE = re.compile(r"-?[0-9]{1,12}")
_TRUSTED_EXCEPTION_MODULE_PREFIXES = (
    "aiohttp",
    "asyncio",
    "binance_ohlcv_archive",
    "builtins",
    "candlestick_manager",
    "ccxt",
    "custom_endpoint_overrides",
    "exchanges",
    "fill_events_manager",
    "hlcv_preparation",
    "hlcvs_manifest",
    "httpx",
    "live",
    "metrics_schema",
    "pareto_store",
    "passivbot",
    "passivbot_exceptions",
    "requests",
    "ssl",
    "urllib3",
    "websockets",
)


def _trusted_exception_module(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in _TRUSTED_EXCEPTION_MODULE_PREFIXES
    )


def _module_exports_exception_class(module: str, name: str, cls: type) -> bool:
    try:
        module_obj = sys.modules.get(module)
        if type(module_obj) is not type(sys):
            return False
        namespace = type(sys).__getattribute__(module_obj, "__dict__")
        return type(namespace) is dict and namespace.get(name) is cls
    except BaseException:
        return False


def bounded_exception_type(exc: BaseException) -> str:
    try:
        mro = type.__getattribute__(type(exc), "__mro__")
        if type(mro) is not tuple:
            return "Error"
        for cls in mro:
            module = type.__getattribute__(cls, "__module__")
            name = type.__getattribute__(cls, "__name__")
            if type(module) is not str or not _trusted_exception_module(module):
                continue
            if (
                type(name) is str
                and name
                and name.isascii()
                and name.isidentifier()
                and not _SENSITIVE_EXCEPTION_TYPE_RE.search(name)
                and _module_exports_exception_class(module, name, cls)
            ):
                return name[:EXCEPTION_TYPE_MAX_LEN]
        return "Error"
    except BaseException:
        return "Error"


def exception_text_contains(
    exc: BaseException,
    needles: tuple[str, ...],
    *,
    chunk_chars: int = 4096,
) -> bool:
    """Inspect exception text in bounded temporary chunks without returning it."""
    try:
        text = str(exc)
        if type(text) is not str:
            return False
        lowered_needles = tuple(
            needle.lower()
            for needle in needles
            if type(needle) is str and needle
        )
        if not lowered_needles or type(chunk_chars) is not int or chunk_chars <= 0:
            return False
        overlap = max(len(needle) for needle in lowered_needles) - 1
        for start in range(0, len(text), chunk_chars):
            chunk_start = max(0, start - overlap)
            lowered_chunk = text[chunk_start : start + chunk_chars].lower()
            if any(needle in lowered_chunk for needle in lowered_needles):
                return True
        return False
    except BaseException:
        return False


def _exact_scalar_text(value: object) -> str | None:
    try:
        if type(value) is str:
            return value
        if type(value) is int:
            if int.bit_length(value) > 160:
                return None
            return str(value)
        return None
    except BaseException:
        return None


def _bounded_exception_attribute(
    exc: BaseException,
    names: tuple[str, ...],
    pattern: re.Pattern[str],
) -> str | None:
    try:
        for name in names:
            try:
                value = getattr(exc, name, None)
            except BaseException:
                continue
            text = _exact_scalar_text(value)
            if (
                text is not None
                and len(text) <= 80
                and text.isascii()
                and pattern.fullmatch(text)
                and not _SENSITIVE_EXCEPTION_TYPE_RE.search(text)
            ):
                return text
        try:
            info = getattr(exc, "info", None)
        except BaseException:
            info = None
        if type(info) is dict:
            for name in names:
                value = info.get(name)
                text = _exact_scalar_text(value)
                if (
                    text is not None
                    and len(text) <= 80
                    and text.isascii()
                    and pattern.fullmatch(text)
                    and not _SENSITIVE_EXCEPTION_TYPE_RE.search(text)
                ):
                    return text
        return None
    except BaseException:
        return None


def bounded_exception_status(exc: BaseException) -> str | None:
    return _bounded_exception_attribute(
        exc,
        ("http_status", "status", "status_code", "statusCode"),
        _EXCEPTION_STATUS_RE,
    )


def bounded_exception_code(exc: BaseException) -> str | None:
    return _bounded_exception_attribute(
        exc,
        ("code", "exact", "error_code", "retCode", "errorCode"),
        _EXCEPTION_CODE_RE,
    )
