"""Prevent GPU screening from silently reusing a different strategy EMA band."""

from optimization.bounds import Bound

from .model import gpu_side_enabled


def validate_independent_unstuck_scope(config: dict) -> None:
    kind = config.get("live", {}).get("strategy_kind", "trailing_martingale")
    bounds = config.get("optimize", {}).get("bounds", {})
    overrides = config.get("coin_overrides", {}) or {}
    for side in ("long", "short"):
        if not gpu_side_enabled(config, side):
            continue
        base = config.get("bot", {}).get(side, {})
        for coin, patch in [("global", {}), *overrides.items()]:
            side_patch = patch.get("bot", {}).get(side, {})
            if side_patch.get("wallet_exposure_limit") == 0:
                # Backtests start flat and per-coin WEL is not an optimizer gene.
                continue
            unstuck = {**base.get("unstuck", {}), **side_patch.get("unstuck", {})}
            if not (
                unstuck.get("enabled", True) and unstuck.get("ema_gating_enabled", True)
            ):
                continue

            # Skip only when both the configured reducer and every candidate are inactive.
            # A zero starting value with a positive search range can still consume the band.
            def fixed_inactive_control(key):
                value = unstuck.get(key)
                if value is None or value > 0.0:
                    return False
                if key in side_patch.get("unstuck", {}):
                    return True
                bound = bounds.get(side, {}).get("unstuck", {}).get(key)
                return (
                    bound is None
                    or Bound.from_config(f"{side}_unstuck_{key}", bound).high <= 0.0
                )

            if any(
                fixed_inactive_control(key)
                for key in ("loss_allowance_pct", "close_pct", "threshold")
            ):
                continue
            strategy_patch = side_patch.get("strategy", {}).get(kind, {})
            strategy = {**base.get("strategy", {}).get(kind, {}), **strategy_patch}
            for key in ("ema_span_0", "ema_span_1"):
                if key not in unstuck:
                    continue  # Legacy direct callers have not introduced independent spans.
                bound = bounds.get(side, {}).get("unstuck", {}).get(key)
                strategy_bound = (
                    bounds.get(side, {}).get("strategy", {}).get(kind, {}).get(key)
                )
                bound = (
                    Bound.from_config(f"{side}_unstuck_{key}", bound)
                    if bound is not None
                    else None
                )
                strategy_bound = (
                    Bound.from_config(f"{side}_{key}", strategy_bound)
                    if strategy_bound is not None
                    else None
                )
                independently_varies = (
                    key not in side_patch.get("unstuck", {})
                    and bound is not None
                    and bound.low != bound.high
                ) or (
                    key not in strategy_patch
                    and strategy_bound is not None
                    and strategy_bound.low != strategy_bound.high
                )
                effective_unstuck = (
                    bound.low
                    if bound is not None and key not in side_patch.get("unstuck", {})
                    else unstuck[key]
                )
                effective_strategy = (
                    strategy_bound.low
                    if strategy_bound is not None and key not in strategy_patch
                    else strategy.get(key)
                )
                if (
                    unstuck[key] != strategy.get(key)
                    or effective_unstuck != effective_strategy
                    or independently_varies
                ):
                    raise ValueError(
                        f"Apple MPS GPU screening does not yet model independent unstuck EMA spans "
                        f"({coin} {side}.{key}). Use optimize.backend='pymoo' or 'deap'. "
                        "Matching fixed strategy/unstuck spans or disabled unstuck EMA gating remain supported. "
                        "Exact CPU validation cannot correct an unmodeled GPU screening gate."
                    )
