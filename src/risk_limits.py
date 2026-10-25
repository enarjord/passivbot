"""Bounded per-coin exposure headroom for admission and diagnostics."""

import math


def effective_we_excess_allowance_fraction(
    *,
    wallet_exposure_limit_ratio: float,
    risk_we_excess_allowance_fraction: float,
    total_wallet_exposure_limit_ratio: float = 0.0
) -> float:
    base = float(wallet_exposure_limit_ratio)
    total = float(total_wallet_exposure_limit_ratio)
    if not (math.isfinite(base) and base > 0 and math.isfinite(total) and total > 0):
        return 0.0
    return min(
        max(0.0, float(risk_we_excess_allowance_fraction)), max(0.0, total / base - 1.0)
    )


def wallet_exposure_limit_ratio_with_allowance(
    *,
    wallet_exposure_limit_ratio: float,
    risk_we_excess_allowance_fraction: float,
    total_wallet_exposure_limit_ratio: float = 0.0
) -> float:
    base = float(wallet_exposure_limit_ratio)
    if not (math.isfinite(base) and base > 0):
        return 0.0
    return base * (
        1
        + effective_we_excess_allowance_fraction(
            wallet_exposure_limit_ratio=base,
            risk_we_excess_allowance_fraction=risk_we_excess_allowance_fraction,
            total_wallet_exposure_limit_ratio=total_wallet_exposure_limit_ratio,
        )
    )
