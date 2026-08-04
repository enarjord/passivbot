from __future__ import annotations

import json
import re
import sys

EXCEPTION_TYPE_MAX_LEN = 80
DIAGNOSTIC_MESSAGE_MAX_LEN = 4096
DIAGNOSTIC_MESSAGE_UNAVAILABLE = "<unavailable>"
DIAGNOSTIC_REDACTED = "[redacted]"
_DIAGNOSTIC_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----.*?"
    r"(?:-----END(?: [A-Z0-9]+)* PRIVATE KEY-----|\Z)",
    re.DOTALL,
)
_DIAGNOSTIC_SENSITIVE_HEADER_NAME_PATTERN = (
    r"(?:authorization|proxy-authorization|cookie|set-cookie|key|sign|signature|"
    r"broker-?key|"
    r"kc-api-partner-sign|x-bapi-sign|"
    r"(?:[a-z0-9]+-)*(?:api-?key|api-(?:secret|sign(?:ature)?|passphrase)|"
    r"access-(?:key|secret|sign(?:ature)?|passphrase)|"
    r"auth-(?:key|secret|sign(?:ature)?|token)|"
    r"security-token|private-key|request-signature))"
)
_DIAGNOSTIC_SERIALIZED_SENSITIVE_HEADER_RE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_-])({_DIAGNOSTIC_SENSITIVE_HEADER_NAME_PATTERN})"
    r"([\"'])(\s*[:=]\s*)([\"'])(?:\\.|[^\"'])*\4"
)
_DIAGNOSTIC_SENSITIVE_HEADER_RE = re.compile(
    rf"(?i)(?<![A-Za-z0-9_-])({_DIAGNOSTIC_SENSITIVE_HEADER_NAME_PATTERN})"
    r"([\"']?\s*[:=]\s*[\"']?)(?:bearer|basic)?\s*[^,\"'\s;}]+"
)
_DIAGNOSTIC_SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])"
    r"(api[-_]?(?:key|secret|signature)|apikey|accessToken|refreshToken|"
    r"clientSecret|privateKey|requestSignature|request[-_]?signature|"
    r"broker[-_]?key|"
    r"auth[-_]?(?:signature|token)|hmac[-_]?signature|exchange[-_]?signature|"
    r"access[-_]?token|refresh[-_]?token|secret|token|signature|password|passphrase|"
    r"private[-_]?key)"
    r"([\"']?\s*(?:[:=/])\s*)(?:[\"'][^\"']*[\"']|[^,\s;&\"'}]+)"
)
_DIAGNOSTIC_SENSITIVE_CLI_RE = re.compile(
    r"(?i)(--(?:api[-_]?key|apikey|api[-_]?secret|secret|token|signature|password|"
    r"passphrase|private[-_]?key)\s+)(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_DIAGNOSTIC_AUTH_SCHEME_RE = re.compile(
    r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+"
)
_DIAGNOSTIC_STANDALONE_SECRET_RE = re.compile(
    r"(?i)\b(?:sk_(?:live|test)_[A-Za-z0-9]+|sk-[A-Za-z0-9_-]{16,}|"
    r"AKIA[0-9A-Z]{16})\b"
)
_DIAGNOSTIC_URL_USERINFO_RE = re.compile(
    r"(?i)\b(https?://)[^/@\s:]+:[^/@\s]+@"
)
_EXCEPTION_STATUS_RE = re.compile(r"[0-9]{1,3}")
_EXCEPTION_CODE_RE = re.compile(r"-?[0-9]{1,12}")
_EXCHANGE_ERROR_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}")
_EXCHANGE_ERROR_PAYLOAD_MAX_LEN = 8192
_EXCHANGE_ERROR_REASON_MAX_LEN = 160
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
                and _module_exports_exception_class(module, name, cls)
            ):
                return name[:EXCEPTION_TYPE_MAX_LEN]
        return "Error"
    except BaseException:
        return "Error"


def sanitize_diagnostic_text(
    value: str,
    *,
    max_len: int = DIAGNOSTIC_MESSAGE_MAX_LEN,
) -> str:
    """Return bounded one-line diagnostics with authentication secrets redacted.

    Operational values such as symbols, prices, quantities, order identifiers,
    exchange messages, URLs, and account state are intentionally retained.  The
    sanitizer removes only credential-bearing values and private-key material.
    """
    if type(value) is not str or type(max_len) is not int or max_len <= 0:
        return DIAGNOSTIC_MESSAGE_UNAVAILABLE
    try:
        scan_len = max(16_384, max_len * 4)
        truncated = len(value) > scan_len
        text = value[:scan_len]
        text = _DIAGNOSTIC_PRIVATE_KEY_BLOCK_RE.sub(
            DIAGNOSTIC_REDACTED, text
        )
        text = _DIAGNOSTIC_SERIALIZED_SENSITIVE_HEADER_RE.sub(
            rf"\1\2\3\4{DIAGNOSTIC_REDACTED}\4", text
        )
        text = _DIAGNOSTIC_SENSITIVE_HEADER_RE.sub(
            rf"\1\2{DIAGNOSTIC_REDACTED}", text
        )
        text = _DIAGNOSTIC_SENSITIVE_VALUE_RE.sub(
            rf"\1\2{DIAGNOSTIC_REDACTED}", text
        )
        text = _DIAGNOSTIC_SENSITIVE_CLI_RE.sub(
            rf"\1{DIAGNOSTIC_REDACTED}", text
        )
        text = _DIAGNOSTIC_AUTH_SCHEME_RE.sub(
            rf"\1 {DIAGNOSTIC_REDACTED}", text
        )
        text = _DIAGNOSTIC_STANDALONE_SECRET_RE.sub(
            DIAGNOSTIC_REDACTED, text
        )
        text = _DIAGNOSTIC_URL_USERINFO_RE.sub(
            rf"\1{DIAGNOSTIC_REDACTED}@", text
        )
        text = " ".join(text.split())
        if not text:
            return "<empty>"
        suffix = "...<truncated>"
        if truncated or len(text) > max_len:
            if max_len <= len(suffix):
                return suffix[:max_len]
            return f"{text[: max_len - len(suffix)]}{suffix}"
        return text
    except BaseException:
        return DIAGNOSTIC_MESSAGE_UNAVAILABLE


def sanitized_exception_message(
    exc: BaseException,
    *,
    max_len: int = DIAGNOSTIC_MESSAGE_MAX_LEN,
) -> str:
    """Render an exception without allowing hostile accessors or secrets to escape."""
    try:
        message = str(exc)
    except BaseException:
        return DIAGNOSTIC_MESSAGE_UNAVAILABLE
    return sanitize_diagnostic_text(message, max_len=max_len)


def _text_contains(
    text: str,
    needles: tuple[str, ...],
    *,
    case_sensitive: bool = False,
    chunk_chars: int = 4096,
) -> bool:
    if type(text) is not str or type(case_sensitive) is not bool:
        return False
    search_needles = tuple(
        needle if case_sensitive else needle.lower()
        for needle in needles
        if type(needle) is str and needle
    )
    if not search_needles or type(chunk_chars) is not int or chunk_chars <= 0:
        return False
    overlap = max(len(needle) for needle in search_needles) - 1
    for start in range(0, len(text), chunk_chars):
        chunk_start = max(0, start - overlap)
        chunk = text[chunk_start : start + chunk_chars]
        searchable_chunk = chunk if case_sensitive else chunk.lower()
        if any(needle in searchable_chunk for needle in search_needles):
            return True
    return False


def exception_text_contains(
    exc: BaseException,
    needles: tuple[str, ...],
    *,
    case_sensitive: bool = False,
    chunk_chars: int = 4096,
) -> bool:
    """Inspect exception text in bounded temporary chunks without returning it."""
    try:
        return _text_contains(
            str(exc),
            needles,
            case_sensitive=case_sensitive,
            chunk_chars=chunk_chars,
        )
    except BaseException:
        return False


def exception_type_name_contains(
    exc: BaseException,
    needles: tuple[str, ...],
    *,
    case_sensitive: bool = False,
) -> bool:
    """Inspect the real class name for control flow without projecting that name."""
    try:
        cls = type(exc)
        name = str.__str__(type.__dict__["__name__"].__get__(cls, type))
        return _text_contains(name, needles, case_sensitive=case_sensitive)
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


def _exception_payload_mapping(exc: BaseException) -> dict | None:
    """Return a small structured exchange payload without retaining raw exception text."""
    try:
        info = getattr(exc, "info", None)
    except BaseException:
        info = None
    if type(info) is dict:
        return info
    try:
        args = getattr(exc, "args", ())
    except BaseException:
        return None
    if type(args) is not tuple:
        return None
    for arg in args:
        if type(arg) is not str or len(arg) > _EXCHANGE_ERROR_PAYLOAD_MAX_LEN:
            continue
        start = arg.find("{")
        if start < 0:
            continue
        try:
            payload, _end = json.JSONDecoder().raw_decode(arg[start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if type(payload) is dict:
            return payload
    return None


def _bounded_exchange_error_label(value: object) -> str | None:
    text = _exact_scalar_text(value)
    if (
        text is None
        or not text.isascii()
        or not _EXCHANGE_ERROR_LABEL_RE.fullmatch(text)
    ):
        return None
    return text


def _bounded_exchange_error_reason(value: object) -> str | None:
    if type(value) is not str or not value or len(value) > _EXCHANGE_ERROR_PAYLOAD_MAX_LEN:
        return None
    try:
        text = sanitize_diagnostic_text(
            value,
            max_len=_EXCHANGE_ERROR_REASON_MAX_LEN,
        )
        text = "".join(
            char if char.isascii() and char.isprintable() and char not in {"|"} else "_"
            for char in text
        ).strip()
    except BaseException:
        return None
    if not text:
        return None
    return text


def _bounded_payload_scalar(
    payload: dict,
    names: tuple[str, ...],
    pattern: re.Pattern[str],
) -> str | None:
    for name in names:
        text = _exact_scalar_text(payload.get(name))
        if (
            text is not None
            and len(text) <= 80
            and text.isascii()
            and pattern.fullmatch(text)
        ):
            return text
    return None


def bounded_exchange_error_context_from_mapping(payload: dict) -> dict[str, str]:
    """Extract bounded diagnostics from a CCXT result mapping.

    Only the normalized result, its exact ``info`` mapping, and a bounded number
    of exact OKX per-order ``data`` mappings are inspected. Raw payloads are
    never retained.
    """
    result: dict[str, str] = {}
    if type(payload) is not dict:
        return result
    info = payload.get("info")
    envelopes = (payload, info) if type(info) is dict else (payload,)
    order_details: list[dict] = []
    for envelope in envelopes:
        data = envelope.get("data")
        if type(data) is not list:
            continue
        for member in data[:8]:
            if type(member) is dict and (
                "sCode" in member or "sMsg" in member
            ):
                order_details.append(member)
    # Prefer actionable per-order details over generic envelope values such as
    # OKX's top-level code "1" and empty message.
    payloads = (*order_details, *envelopes)
    for candidate in payloads:
        status = _bounded_payload_scalar(
            candidate,
            ("http_status", "status", "status_code", "statusCode"),
            _EXCEPTION_STATUS_RE,
        )
        if status is not None:
            result["error_status"] = status
            break
    for candidate in payloads:
        code = _bounded_payload_scalar(
            candidate,
            ("sCode", "code", "exact", "error_code", "retCode", "errorCode"),
            _EXCEPTION_CODE_RE,
        )
        if code is not None:
            result["error_code"] = code
            break
    for candidate in payloads:
        for key in ("label", "error", "name"):
            label = _bounded_exchange_error_label(candidate.get(key))
            if label is not None:
                result["error_label"] = label
                break
        if "error_label" in result:
            break
    for candidate in payloads:
        for key in ("sMsg", "message", "msg", "retMsg", "detail", "reason"):
            reason = _bounded_exchange_error_reason(candidate.get(key))
            if reason is not None:
                result["error_reason"] = reason
                break
        if "error_reason" in result:
            break
    return result


def bounded_exchange_error_context(exc: BaseException) -> dict[str, str]:
    """Extract only bounded, sanitized fields from a structured exchange error."""
    result: dict[str, str] = {}
    status = bounded_exception_status(exc)
    code = bounded_exception_code(exc)
    if status is not None:
        result["error_status"] = status
    if code is not None:
        result["error_code"] = code
    payload = _exception_payload_mapping(exc)
    if payload is None:
        return result
    for key, value in bounded_exchange_error_context_from_mapping(payload).items():
        result.setdefault(key, value)
    return result
