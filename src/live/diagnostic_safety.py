from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
import re
import sys

EXCEPTION_TYPE_MAX_LEN = 80
_SENSITIVE_EXCEPTION_TYPE_RE = re.compile(
    r"(?i)(?:api_?key|apikey|authorization|cookie|passphrase|password|private_?key|"
    r"privatekey|secret|signature|token|wallet_?address|walletaddress)"
)
_EXCEPTION_STATUS_RE = re.compile(r"[0-9]{1,3}")
_EXCEPTION_CODE_RE = re.compile(r"-?[0-9]{1,12}")
_EXCHANGE_ERROR_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}")
# An unlabelled opaque value can still be a credential.
_EXCHANGE_ERROR_SECRET_VALUE_RE = re.compile(r"[A-Za-z0-9+/_=-]{24,}")
_PUBLIC_WALLET_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]{40}")
# Cap the prefix scan at each word boundary, even in long hyphenated input.
_EXCHANGE_ERROR_FIELD_PREFIX = r"(?<![a-z0-9])(?:[a-z0-9]+[-_]+){0,4}"
# A quote or status clause fixes the label start. Scan the full prefix
# there without backtracking within its components.
_EXCHANGE_ERROR_ANCHORED_FIELD_PREFIX = r"(?:[a-z0-9]++[-_]++)*"
_EXCHANGE_ERROR_AUTH_HEADER_LABEL = (
    r"(?:authorization|cookie)(?>[-_ ]+header\b|)"
)
_EXCHANGE_ERROR_AUTH_HEADER_RE = re.compile(
    _EXCHANGE_ERROR_FIELD_PREFIX
    + _EXCHANGE_ERROR_AUTH_HEADER_LABEL
    + r"""['"]?\s*[:=].*""",
    re.IGNORECASE | re.DOTALL,
)
_EXCHANGE_ERROR_PEM_RE = re.compile(
    r"-----BEGIN [^\r\n]*PRIVATE KEY-----.*?(?:-----END [^\r\n]*PRIVATE KEY-----|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_EXCHANGE_ERROR_CREDENTIAL_NAME = (
    r"(?:"
    + _EXCHANGE_ERROR_AUTH_HEADER_LABEL
    + r"|api[-_ ]?(?:key|secret|token|passphrase)|private[-_ ]?key|"
    r"client[-_ ]?secret|access[-_ ]?(?:key|secret|sign(?:ature)?|token|passphrase)|"
    r"refresh[-_ ]?token|session[-_ ]?(?:id|key|token)|signing[-_ ]?key|"
    r"secret(?:[-_ ]?key)?|signature|passphrase|password|token|"
    r"listen[-_ ]?key|seed[-_ ]?phrase|mnemonic)(?![\w-])"
)
_EXCHANGE_ERROR_CREDENTIAL_LABEL = (
    _EXCHANGE_ERROR_FIELD_PREFIX + _EXCHANGE_ERROR_CREDENTIAL_NAME
)
# A closing quote must be followed by a boundary. Another quote of the same
# kind in the remainder makes the value ambiguous, so redact the whole suffix.
_EXCHANGE_ERROR_CREDENTIAL_VALUE = (
    r"(?:"
    r"""(?:"(?:\\.|[^"\\\r\n])*"(?=[^"]*\Z)|'(?:\\.|[^'\\\r\n])*'(?=[^']*\Z))"""
    r"(?=\Z|\s|[.,!?;:)\]](?=\Z|\s))"
    r"|[\s\S]+)"
)
_SAFE_CREDENTIAL_STATUS = (
    r"(?!(?:(?:is |was |has )?(?:expired|invalid|missing|rejected|revoked|disabled|"
    r"incorrect|required|not found|not recognized|does not exist)|"
    r"for the configured private key|failed: <redacted-auth>)(?=\s*(?:[.!?](?:\s|$)|$)))"
)
_EXCHANGE_ERROR_AUTH_SCHEME_RE = re.compile(
    r"\b(?:bearer|basic)(?>\s+(?:authentication|auth|token)\b|)"
    r"\s+(?![\s.!?,;:=])"
    + _SAFE_CREDENTIAL_STATUS
    + _EXCHANGE_ERROR_CREDENTIAL_VALUE,
    re.IGNORECASE,
)
_EXCHANGE_ERROR_CREDENTIAL_RE = re.compile(
    rf"""(?P<label>(?:{_EXCHANGE_ERROR_CREDENTIAL_LABEL}|"""
    rf"""{_EXCHANGE_ERROR_FIELD_PREFIX}(?:key|sign|credentials?)(?![\w-])|"""
    r"""\b(?:seed|session)(?![\w-])|"""
    r"""\b(?:basic|bearer)(?:[-_ ]+(?:auth(?:entication)?|token))?)['"]?\s*[:=]\s*)"""
    + _EXCHANGE_ERROR_CREDENTIAL_VALUE,
    re.IGNORECASE,
)
_EXCHANGE_ERROR_CREDENTIAL_ECHO_RE = re.compile(
    rf"""(?P<label>(?P<quote>['"]){_EXCHANGE_ERROR_ANCHORED_FIELD_PREFIX}"""
    rf"""{_EXCHANGE_ERROR_CREDENTIAL_NAME}(?P=quote)\s*|"""
    rf"""(?<!['"]){_EXCHANGE_ERROR_CREDENTIAL_LABEL}(?:\s+|(?=['"(\[,;])))"""
    # Keep standalone diagnostic wording, including a credential followed by
    # a known status clause. Other trailing text may be a bare credential echo.
    r"(?![\s.!?:=]|<redacted)"
    + _SAFE_CREDENTIAL_STATUS
    + _EXCHANGE_ERROR_CREDENTIAL_VALUE,
    re.IGNORECASE,
)
_EXCHANGE_ERROR_REVERSE_CREDENTIAL_RE = re.compile(
    r"""(?P<quote>['"])\s+(?P<clause>(?:"""
    r"(?:is|was)\s+(?:(?:an?|the)\s+)?(?:invalid|incorrect|expired|revoked)|"
    r"(?:(?:is|was)\s+not|isn't|wasn't)\s+(?:(?:an?|the)\s+)?"
    r"(?:valid|correct|recognized|accepted))\s+"
    + _EXCHANGE_ERROR_ANCHORED_FIELD_PREFIX
    + _EXCHANGE_ERROR_CREDENTIAL_NAME + ")",
    re.IGNORECASE,
)
# Exact stock text with no echoed values. Keep the safety decision independent
# of a potentially spoofed exchange code or message prefix.
_SAFE_EXCHANGE_REJECTION_TEXT = frozenset({
    "invalid api-key, ip, or permissions for action.",
    "signature for this request is not valid.",
    "the api key does not have trade permission.",
    "your token balance is too low",
})
_BASIC_ORDER_PRICE_TICK_REASON_RE = re.compile(
    r"Basic order validation failed: price tick \d{1,16}(?:\.\d{1,16})?\.?",
    re.IGNORECASE | re.ASCII,
)
_EXCHANGE_ERROR_EMBEDDED_PAYLOAD_RE = re.compile(r"\{|\[\s*[\[{\x22\x27]")
_HYPERLIQUID_UNRECOGNIZED_WALLET_RE = re.compile(
    r"(?:L1 error: )?User or API Wallet 0x[0-9a-fA-F]{40} does not exist\.?"
)
HYPERLIQUID_UNRECOGNIZED_WALLET = "HYPERLIQUID_UNRECOGNIZED_WALLET"
_EXCHANGE_ERROR_EMAIL_RE = re.compile(
    r"(?i)\b[A-Z0-9._%+-]{1,64}@[A-Z0-9.-]{1,253}\.[A-Z]{2,63}\b"
)
_EXCHANGE_ERROR_URL_RE = re.compile(r"(?i)\bhttps?://[^\s]+")
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
                and not _SENSITIVE_EXCEPTION_TYPE_RE.search(name)
                and _module_exports_exception_class(module, name, cls)
            ):
                return name[:EXCEPTION_TYPE_MAX_LEN]
        return "Error"
    except BaseException:
        return "Error"


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
        or _SENSITIVE_EXCEPTION_TYPE_RE.search(text)
    ):
        return None
    return text


def _redact_reverse_credential_echoes(text: str) -> str:
    # Start with the status clause. An earlier quote inside the value must
    # not let a later quote become the apparent start of the credential.
    parts = []
    offset = 0
    for match in _EXCHANGE_ERROR_REVERSE_CREDENTIAL_RE.finditer(text):
        opening = text.find(match["quote"], offset, match.start())
        if opening >= 0:
            parts.append(text[offset:opening])
        parts.extend(("<redacted> ", match["clause"]))
        offset = match.end()
    parts.append(text[offset:])
    return "".join(parts)


def _bounded_exchange_error_reason(value: object) -> str | None:
    if type(value) is not str or not value or len(value) > _EXCHANGE_ERROR_PAYLOAD_MAX_LEN:
        return None
    text = value.strip()
    if (
        text.lower() in _SAFE_EXCHANGE_REJECTION_TEXT
        or _BASIC_ORDER_PRICE_TICK_REASON_RE.fullmatch(text)
    ):
        return text
    try:
        # A nested request/response dump can contain short credentials or a
        # replayable action. Keep the reason preceding it and drop the dump.
        prefix, *payload = _EXCHANGE_ERROR_EMBEDDED_PAYLOAD_RE.split(value, maxsplit=1)
        text = prefix.rstrip() + " <redacted-payload>" if payload else prefix
        text = _redact_reverse_credential_echoes(text)
        text = _EXCHANGE_ERROR_URL_RE.sub("<redacted-url>", text)
        if "@" in text:
            text = _EXCHANGE_ERROR_EMAIL_RE.sub("<redacted-email>", text)
        text = _EXCHANGE_ERROR_AUTH_HEADER_RE.sub("<redacted-header>", text)
        text = _EXCHANGE_ERROR_AUTH_SCHEME_RE.sub("<redacted-auth>", text)
        text = _EXCHANGE_ERROR_PEM_RE.sub("<redacted-key>", text)
        text = _EXCHANGE_ERROR_CREDENTIAL_RE.sub(r"\g<label><redacted>", text)
        text = _EXCHANGE_ERROR_CREDENTIAL_ECHO_RE.sub(r"\g<label><redacted>", text)
        text = _EXCHANGE_ERROR_SECRET_VALUE_RE.sub(
            lambda match: (
                match[0] if _PUBLIC_WALLET_ADDRESS_RE.fullmatch(match[0]) else "<redacted>"
            ),
            text,
        )
        text = " ".join(text.split())
        text = "".join(
            char if char.isascii() and char.isprintable() and char not in {"|"} else "_"
            for char in text
        ).strip()
    except BaseException:
        return None
    if not text:
        return None
    if len(text) > _EXCHANGE_ERROR_REASON_MAX_LEN:
        return f"{text[: _EXCHANGE_ERROR_REASON_MAX_LEN - 3]}..."
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
            and not _SENSITIVE_EXCEPTION_TYPE_RE.search(text)
        ):
            return text
    return None


def _hyperliquid_error_details(envelopes: tuple[dict, ...]) -> dict[str, str]:
    """Read the two Hyperliquid error envelopes without exposing their bodies."""
    for envelope in envelopes:
        response = envelope.get("response")
        status = envelope.get("status")
        if status == "err" and type(response) is str:
            reason = _bounded_exchange_error_reason(response)
            if reason is None:
                continue
            if _HYPERLIQUID_UNRECOGNIZED_WALLET_RE.fullmatch(response):
                return {
                    "error_label": HYPERLIQUID_UNRECOGNIZED_WALLET,
                    "error_reason": "Hyperliquid does not recognize the API wallet.",
                }
            return {"error_reason": reason}
        if status != "ok" or type(response) is not dict:
            continue
        if response.get("type") not in ("order", "cancel"):
            continue
        data = response.get("data")
        statuses = data.get("statuses") if type(data) is dict else None
        if type(statuses) is list:
            for member in statuses[:8]:
                if type(member) is dict:
                    reason = _bounded_exchange_error_reason(member.get("error"))
                    if reason is not None:
                        return {"error_reason": reason}
    return {}


def bounded_exchange_error_context_from_mapping(payload: dict) -> dict[str, str]:
    """Extract bounded diagnostics from normalized results and known exchange envelopes."""
    result: dict[str, str] = {}
    if type(payload) is not dict:
        return result
    info = payload.get("info")
    envelopes = (payload, info) if type(info) is dict else (payload,)
    result.update(_hyperliquid_error_details(envelopes))
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
        if "error_label" not in result:
            for key in ("label", "error", "name"):
                label = _bounded_exchange_error_label(candidate.get(key))
                if label is not None:
                    result["error_label"] = label
                    break
        if "error_reason" not in result:
            for key in ("sMsg", "message", "msg", "retMsg", "detail", "reason"):
                reason = _bounded_exchange_error_reason(candidate.get(key))
                if reason is not None:
                    result["error_reason"] = reason
                    break
    return result


def hyperliquid_wallet_rejected(exc: BaseException) -> bool:
    """Classify only the exact structured response, not generic error labels."""
    payload = _exception_payload_mapping(exc)
    return (
        payload is not None
        and _hyperliquid_error_details((payload,)).get("error_label")
        == HYPERLIQUID_UNRECOGNIZED_WALLET
    )


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


def bounded_traceback_detail_inner(exc: BaseException) -> dict[str, Any]:
    """Return a durable frame chain without exception text, locals, or source lines."""
    exceptions: list[dict[str, Any]] = []
    visited: set[int] = set()
    current: BaseException | None = exc
    relation = "raised"
    total_frames = 0
    truncated = False
    source_root = Path(__file__).resolve().parents[2]
    while current is not None and id(current) not in visited:
        if len(exceptions) >= 8:
            truncated = True
            break
        visited.add(id(current))
        frames: list[dict[str, Any]] = []
        tb = current.__traceback__
        while tb is not None:
            if total_frames >= 64:
                truncated = True
                break
            raw_filename = str(tb.tb_frame.f_code.co_filename)
            try:
                resolved = Path(raw_filename).resolve()
                relative = resolved.relative_to(source_root)
                filename = relative.as_posix()
            except (OSError, RuntimeError, ValueError):
                filename = os.path.basename(raw_filename)
            if not re.fullmatch(r"[A-Za-z0-9_./<>-]{1,240}", filename):
                filename = os.path.basename(raw_filename)
            if not re.fullmatch(r"[A-Za-z0-9_./<>-]{1,240}", filename):
                filename = "unknown"
            function = str(tb.tb_frame.f_code.co_name)
            if not re.fullmatch(r"[A-Za-z0-9_<>.-]{1,96}", function):
                function = "unknown"
            frames.append(
                {
                    "file": filename,
                    "function": function,
                    "line": max(0, int(tb.tb_lineno)),
                }
            )
            total_frames += 1
            tb = tb.tb_next
        exceptions.append(
            {
                "relation": relation,
                "error_type": bounded_exception_type(current),
                "frames": frames,
            }
        )
        if current.__cause__ is not None:
            current = current.__cause__
            relation = "cause"
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            relation = "context"
        else:
            current = None
    return {
        "exceptions": exceptions,
        "frame_count": total_frames,
        "truncated": truncated,
        "includes_exception_text": False,
        "includes_locals": False,
    }


def bounded_traceback_detail(exc: BaseException) -> dict[str, Any]:
    """Isolate optional traceback projection from the execution failure policy."""
    try:
        return bounded_traceback_detail_inner(exc)
    except BaseException:
        return {
            "exceptions": [],
            "frame_count": 0,
            "truncated": True,
            "unavailable_reason": "projection_failed",
            "includes_exception_text": False,
            "includes_locals": False,
        }


def log_process_recovery(bot: Any) -> None:
    """Clear repeat diagnostics only after an admitted ordinary plan returns."""
    try:
        state = getattr(bot, "_process_failure_state", None)
        if state:
            logging.info(
                "[bot] recovered from previous run failure; ordinary execution resumed"
            )
            state.clear()
    except Exception:
        pass  # Optional diagnostics cannot alter execution or its return value.
