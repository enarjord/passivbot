from __future__ import annotations

from typing import Any


WE_EXCESS_ALLOWANCE_MODE_BOUNDED = "bounded"
WE_EXCESS_ALLOWANCE_MODE_LEGACY_RAW = "legacy_raw"
WE_EXCESS_ALLOWANCE_MODES = frozenset(
    {
        WE_EXCESS_ALLOWANCE_MODE_BOUNDED,
        WE_EXCESS_ALLOWANCE_MODE_LEGACY_RAW,
    }
)


def normalize_we_excess_allowance_mode(
    value: Any,
    *,
    path: str = "risk_we_excess_allowance_mode",
) -> str:
    if value is None:
        return WE_EXCESS_ALLOWANCE_MODE_BOUNDED
    mode = str(value).strip().lower()
    if mode not in WE_EXCESS_ALLOWANCE_MODES:
        joined = ", ".join(sorted(WE_EXCESS_ALLOWANCE_MODES))
        raise ValueError(f"{path} must be one of: {joined}")
    return mode


def effective_we_excess_allowance_pct(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
    total_wallet_exposure_limit: float = 0.0,
    risk_we_excess_allowance_mode: str = WE_EXCESS_ALLOWANCE_MODE_BOUNDED,
) -> float:
    base_limit = float(wallet_exposure_limit)
    raw_allowance = max(0.0, float(risk_we_excess_allowance_pct))
    mode = normalize_we_excess_allowance_mode(risk_we_excess_allowance_mode)
    if mode == WE_EXCESS_ALLOWANCE_MODE_LEGACY_RAW:
        return raw_allowance
    total_limit = float(total_wallet_exposure_limit)
    if base_limit > 0.0 and total_limit > 0.0:
        return min(raw_allowance, max(0.0, total_limit / base_limit - 1.0))
    return raw_allowance


def wallet_exposure_limit_with_allowance(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
    total_wallet_exposure_limit: float = 0.0,
    risk_we_excess_allowance_mode: str = WE_EXCESS_ALLOWANCE_MODE_BOUNDED,
) -> float:
    base_limit = float(wallet_exposure_limit)
    return base_limit * (
        1.0
        + effective_we_excess_allowance_pct(
            wallet_exposure_limit=base_limit,
            risk_we_excess_allowance_pct=risk_we_excess_allowance_pct,
            total_wallet_exposure_limit=total_wallet_exposure_limit,
            risk_we_excess_allowance_mode=risk_we_excess_allowance_mode,
        )
    )
