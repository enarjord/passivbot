"""Prevent GPU screening from silently reusing a different strategy EMA band."""

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
                return bound is None or (
                    isinstance(bound, (tuple, list))
                    and len(bound) >= 2
                    and max(bound[:2]) <= 0.0
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
                varies = (
                    lambda b: isinstance(b, (tuple, list))
                    and len(b) >= 2
                    and b[0] != b[1]
                )
                independently_varies = (
                    key not in side_patch.get("unstuck", {}) and varies(bound)
                ) or (key not in strategy_patch and varies(strategy_bound))
                effective_unstuck = (
                    bound[0]
                    if isinstance(bound, (tuple, list))
                    and bound
                    and key not in side_patch.get("unstuck", {})
                    else unstuck[key]
                )
                effective_strategy = (
                    strategy_bound[0]
                    if isinstance(strategy_bound, (tuple, list))
                    and strategy_bound
                    and key not in strategy_patch
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
