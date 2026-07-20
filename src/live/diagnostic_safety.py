from __future__ import annotations

import re

EXCEPTION_TYPE_MAX_LEN = 80
_SENSITIVE_EXCEPTION_TYPE_RE = re.compile(
    r"(?i)(?:api_?key|apikey|authorization|cookie|passphrase|password|private_?key|"
    r"privatekey|secret|signature|token|wallet_?address|walletaddress)"
)
_EXCEPTION_STATUS_RE = re.compile(r"[0-9]{1,3}")
_EXCEPTION_CODE_RE = re.compile(r"-?[A-Za-z0-9][A-Za-z0-9_-]{0,47}")


def bounded_exception_type(exc: BaseException) -> str:
    try:
        name = type(exc).__name__
        if type(name) is not str:
            return "Error"
        if (
            not name
            or not name.isascii()
            or not name.isidentifier()
            or _SENSITIVE_EXCEPTION_TYPE_RE.search(name)
        ):
            return "Error"
        return name[:EXCEPTION_TYPE_MAX_LEN]
    except BaseException:
        return "Error"


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
