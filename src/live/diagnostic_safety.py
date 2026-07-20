from __future__ import annotations

import re

EXCEPTION_TYPE_MAX_LEN = 80
_SENSITIVE_EXCEPTION_TYPE_RE = re.compile(
    r"(?i)(?:^|_)(?:api_?key|apikey|authorization|auth|cookie|passphrase|password|"
    r"private_?key|privatekey|secret|signature|token|wallet_?address|walletaddress)"
    r"(?:_|$)"
)


def bounded_exception_type(exc: BaseException) -> str:
    try:
        name = type(exc).__name__
        if not isinstance(name, str):
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
