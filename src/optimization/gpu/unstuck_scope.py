"""Prevent GPU screening from silently reusing a different strategy EMA band."""


def validate_independent_unstuck_scope(config: dict) -> None:
    kind = config.get("live", {}).get("strategy_kind", "trailing_martingale")
    bounds = config.get("optimize", {}).get("bounds", {})
    overrides = config.get("coin_overrides", {}) or {}
    for side in ("long", "short"):
        approved = config.get("live", {}).get("approved_coins")
        if isinstance(approved, dict) and side in approved and not approved[side]:
            continue
        base = config.get("bot", {}).get(side, {})
        if base.get("risk", {}).get("total_wallet_exposure_limit", 0.0) <= 0.0:
            continue
        for coin, patch in [("global", {}), *overrides.items()]:
            side_patch = patch.get("bot", {}).get(side, {})
            unstuck = {**base.get("unstuck", {}), **side_patch.get("unstuck", {})}
            if not (
                unstuck.get("enabled", True) and unstuck.get("ema_gating_enabled", True)
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
