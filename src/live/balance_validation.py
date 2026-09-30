"""Shared current-balance validation, independent of HSL history recovery."""
import math
from live.event_bus import ReasonCodes

class RiskInputUnavailable(RuntimeError):
    """An authoritative input cannot support live risk evaluation."""

    def __init__(self, reason: str, **details):
        self.reason = reason
        self.details = details
        super().__init__(reason)

def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None

def validate_balances(raw, sizing):
    if not all(math.isfinite(value) and value > 0.0 for value in (raw, sizing)):
        raise RiskInputUnavailable(
            ReasonCodes.CURRENT_BALANCE_UNAVAILABLE,
            balance_raw=_number(raw),
            balance=_number(sizing),
        )
