from __future__ import annotations

import re

EXCEPTION_TYPE_MAX_LEN = 80
_SENSITIVE_EXCEPTION_TYPE_RE = re.compile(
    r"(?i)(?:api_?key|apikey|authorization|cookie|passphrase|password|private_?key|"
    r"privatekey|secret|signature|token|wallet_?address|walletaddress)"
)


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
