"""Bounded per-coin exposure headroom for admission and diagnostics."""

import math


def effective_we_excess_allowance_pct(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
    total_wallet_exposure_limit: float = 0.0
) -> float:
    base = float(wallet_exposure_limit)
    total = float(total_wallet_exposure_limit)
    if not (math.isfinite(base) and base > 0 and math.isfinite(total) and total > 0):
        return 0.0
    return min(
        max(0.0, float(risk_we_excess_allowance_pct)), max(0.0, total / base - 1.0)
    )


def wallet_exposure_limit_with_allowance(
    *,
    wallet_exposure_limit: float,
    risk_we_excess_allowance_pct: float,
    total_wallet_exposure_limit: float = 0.0
) -> float:
    base = float(wallet_exposure_limit)
    if not (math.isfinite(base) and base > 0):
        return 0.0
    return base * (
        1
        + effective_we_excess_allowance_pct(
            wallet_exposure_limit=base,
            risk_we_excess_allowance_pct=risk_we_excess_allowance_pct,
            total_wallet_exposure_limit=total_wallet_exposure_limit,
        )
    )
