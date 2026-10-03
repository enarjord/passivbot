"""Passive valuations for account logs; never a trading or risk input."""

from __future__ import annotations

import math

import passivbot_rust as pbr

DIAGNOSTIC_MAX_AGE_MS = 60_000


def remember_position_quotes(bot, snapshots, positions, *, now_ms):
    """Retain only held-symbol quotes already obtained for position logging."""
    held = {p["symbol"] for p in positions if float(p["size"]) != 0.0}
    previous = getattr(bot, "_diagnostic_position_quotes", {})
    bot._diagnostic_position_quotes = {
        symbol: quote
        for symbol, quote in {**previous, **snapshots}.items()
        if symbol in held and 0 <= now_ms - quote.fetched_ms <= DIAGNOSTIC_MAX_AGE_MS
    }


def balance_equity_observation(bot, *, balance_raw, now_ms):
    """Value every held leg with cached quotes, including position-log fallback.

    The 60-second diagnostic allowance does not alter market caches, freshness
    epochs, HSL, or trading permissions. Fallback age is time since observation,
    not a claim about the timestamp of the underlying completed candle.
    """
    def unavailable(reason):
        return {"equity": None, "equity_unavailable_reason": reason}

    try:
        strict_age = bot._live_market_snapshot_max_age_ms()
        pending = getattr(bot, "_authoritative_pending_confirmations", {})
        max_age = 0
        for name in ("balance", "positions"):
            state = bot.freshness_ledger.surfaces[name]
            age = now_ms - state.updated_ms
            if state.updated_ms <= 0 or not 0 <= age <= DIAGNOSTIC_MAX_AGE_MS:
                return unavailable("account_state_stale")
            if int(pending.get(name, 0)) > state.epoch:
                return unavailable("account_confirmation_pending")
            max_age = max(max_age, age)
        if not math.isfinite(balance_raw):
            return unavailable("invalid_balance")
        equity = balance_raw
        sources = set()
        fallback = False
        cache = getattr(getattr(bot, "market_snapshot_provider", None), "_cache", {})
        diagnostic = getattr(bot, "_diagnostic_position_quotes", {})
        for symbol, sides in bot.positions.items():
            for side in ("long", "short"):
                position = sides.get(side)
                if position is None:
                    continue
                size = float(position["size"])
                if not math.isfinite(size):
                    return unavailable("invalid_position")
                if size == 0.0:
                    continue
                basis, multiplier = float(position["price"]), float(bot.c_mults[symbol])
                if (
                    not math.isfinite(basis) or basis <= 0
                    or not math.isfinite(multiplier) or multiplier <= 0
                ):
                    return unavailable("invalid_position")
                candidates = [
                    q for q in (cache.get(symbol), diagnostic.get(symbol))
                    if q is not None and q.is_valid()
                    and 0 <= now_ms - q.fetched_ms <= DIAGNOSTIC_MAX_AGE_MS
                ]
                if not candidates:
                    return unavailable("price_missing_or_stale")
                quote = max(candidates, key=lambda q: q.fetched_ms)
                # Source is a bounded classification, never arbitrary connector text.
                candle = quote.source == "completed_candle_fallback"
                sources.add("completed_candle" if candle else "market_quote")
                fallback |= candle
                max_age = max(max_age, now_ms - quote.fetched_ms)
                calc = pbr.calc_pnl_long if side == "long" else pbr.calc_pnl_short
                pnl = float(calc(basis, quote.last, size, multiplier))
                if not math.isfinite(pnl):
                    return unavailable("invalid_pnl")
                equity += pnl
        if not math.isfinite(equity):
            return unavailable("invalid_pnl")
        return {
            "equity": float(equity),
            "equity_estimated": fallback or max_age > strict_age,
            "equity_valuation_source": (
                next(iter(sources)) if len(sources) == 1 else "mixed" if sources else "flat"
            ),
            "equity_observation_age_ms": int(max_age),
        }
    except Exception:
        # Optional telemetry cannot inhibit refresh or reveal exception content.
        return unavailable("valuation_error")
