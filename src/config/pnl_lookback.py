from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


FULL_HISTORY_LOOKBACK_DAYS_SENTINEL = -1.0
ONE_MINUTE_MS = 60_000
ONE_DAY_MS = 86_400_000.0


@dataclass(frozen=True)
class PnlsLookback:
    mode: str
    days: float

    @property
    def is_all(self) -> bool:
        return self.mode == "all"

    @property
    def display_value(self) -> str | float:
        return "all" if self.is_all else self.days

    def to_backtest_days_value(self) -> float:
        return FULL_HISTORY_LOOKBACK_DAYS_SENTINEL if self.is_all else self.days

    def to_window_ms(self, *, minimum_ms: int = ONE_MINUTE_MS) -> int | None:
        if self.is_all:
            return None
        return max(int(minimum_ms), int(round(self.days * ONE_DAY_MS)))

    def to_start_ms(self, now_ms: int, *, minimum_ms: int = ONE_MINUTE_MS) -> int | None:
        lookback_ms = self.to_window_ms(minimum_ms=minimum_ms)
        if lookback_ms is None:
            return None
        return int(now_ms) - lookback_ms

    def event_history_start_ms(self, now_ms: int) -> int | None:
        return self.to_start_ms(now_ms, minimum_ms=1)

    def hsl_window_ms(self) -> int | None:
        return self.to_window_ms(minimum_ms=ONE_MINUTE_MS)

    def balance_history_start_ms(self, now_ms: int) -> int | None:
        return self.to_start_ms(now_ms, minimum_ms=ONE_MINUTE_MS)

    def fill_cache_age_limit_ms(self, now_ms: int) -> int | None:
        if self.is_all:
            return None
        return self.event_history_start_ms(now_ms)


def parse_pnls_max_lookback_days(
    value: Any,
    *,
    field_name: str = "live.pnls_max_lookback_days",
) -> PnlsLookback:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "all":
            return PnlsLookback(mode="all", days=math.inf)
        value = normalized
    try:
        days = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be >= 0 or 'all', got {value!r}") from exc
    if not math.isfinite(days):
        raise ValueError(f"{field_name} must be >= 0 or 'all', got {value!r}")
    if days < 0.0:
        raise ValueError(f"{field_name} must be >= 0 or 'all', got {value!r}")
    return PnlsLookback(mode="window", days=days)


def normalize_pnls_max_lookback_days_config_value(
    value: Any,
    *,
    field_name: str = "live.pnls_max_lookback_days",
) -> str | float:
    parsed = parse_pnls_max_lookback_days(value, field_name=field_name)
    return parsed.display_value
