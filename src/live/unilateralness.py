"""Completed-candle transport for Rust RMS directionality; no projected flat tails."""

import asyncio
import logging
import math

import ccxt
import numpy as np
import passivbot_rust
from candlestick_manager import OhlcvFetchError
from config.entry_cooldown import uses_adverse_rms


def scoring_enabled(bot, side, symbols):
    if bot.bot_value(side, "forager_score_weights")["unilateralness"] <= 0:
        return False
    if bot.is_forager_mode(side):
        return True
    # A held ineligible symbol can consume a slot even when the approved
    # universe fits the configured cap. Keep a conservative input superset:
    # with occupied slots and competing flat symbols, Rust may need ranking.
    # Rust retains exact mode, eligibility, effective-slot and cost decisions.
    held = sum(
        bot.positions.get(symbol, {}).get(side, {}).get("size", 0.0) != 0.0
        for symbol in symbols
    )
    return held > 0 and len(symbols) - held > 1


def adverse_enabled(bot, side, symbol):
    weights = bot.bp(side, "entry_cooldown_weights_minutes", symbol)
    if weights["adverse_directionality"] <= 0 or not bot.is_approved(side, symbol):
        return False
    return uses_adverse_rms({
        "entry_cooldown_weights_minutes": weights,
        **{key: bot.bp(side, key, symbol) for key in (
            "risk_entry_cooldown_minutes", "entry_cooldown_min_duration_minutes",
            "entry_cooldown_max_duration_minutes",
        )},
    })


async def load(bot, symbols, cache_only_symbols, forager_age_by_symbol=None):
    result = {symbol: {} for symbol in symbols}
    ranking = {symbol: {} for symbol in symbols}
    required = {}
    # Share canonical EMA invalidation: candle repairs and new gap evidence
    # invalidate these results even when the completed-candle cutoff is unchanged.
    replay_cache = getattr(bot.cm, "_ema_cache", None)
    if replay_cache is not None:
        for symbol, cached in replay_cache.items():
            if symbol not in result:
                for key in list(cached):
                    if key[0] == "signed_unilateralness":
                        del cached[key]
    source_ends = {}
    ages = forager_age_by_symbol or {}
    scoring_sides = {
        side for side in ("long", "short")
        if bot.is_pside_enabled(side) and scoring_enabled(bot, side, symbols)
    }
    end = int(bot.get_exchange_time()) // 60_000 * 60_000 - 60_000
    for symbol in symbols:
        current_spans = set()
        forager_spans = set()
        for side in ("long", "short"):
            if not bot.is_pside_enabled(side):
                continue
            span = float(bot.bot_value(side, "unilateralness_ema_span_1m"))
            if adverse_enabled(bot, side, symbol):
                current_spans.add(span)
            if side in scoring_sides:
                forager_spans.add(span)
        spans = current_spans | forager_spans
        required[symbol] = (current_spans, forager_spans)
        cached = replay_cache.get(symbol, {}) if replay_cache is not None else {}
        for key in list(cached):
            if key[0] == "signed_unilateralness" and key[1] not in spans:
                del cached[key]
        if not spans:
            continue
        if any(not math.isfinite(s) or not 1 <= s <= 100_000 for s in spans):
            raise ValueError("unilateralness EMA span must be between 1 and 100000")
        keys = {span: ("signed_unilateralness", span, "60000") for span in spans}
        allowed_age = max(0, int(ages.get(symbol, 0)))
        if all(
            key in cached and (
                cached[key][1] == end
                or (symbol in cache_only_symbols and 0 <= end - cached[key][1] <= allowed_age)
            )
            for key in keys.values()
        ):
            ranking[symbol] = {span: cached[key][0] for span, key in keys.items()}
            result[symbol] = {
                span: cached[key][0] for span, key in keys.items() if cached[key][1] == end
            }
            source_ends[symbol] = min(cached[key][1] for key in keys.values())
            continue
        # A failed refresh must not leave an older score available on a later load.
        for key in keys.values():
            cached.pop(key, None)
        count = math.ceil(max(spans) * 20.0) + 1
        start = end - (count - 1) * 60_000 - math.ceil(allowed_age / 60_000) * 60_000
        try:
            rows = await bot.cm.get_candles(
                symbol,
                start_ts=start,
                end_ts=end,
                timeframe="1m",
                standardize=False,
                allow_remote_fetch=symbol not in cache_only_symbols,
            )
        except (ccxt.NetworkError, OhlcvFetchError, OSError, asyncio.TimeoutError) as exc:
            logging.warning(
                "[ema] unilateralness unavailable | symbol=%s error_type=%s",
                symbol,
                type(exc).__name__,
            )
            continue
        rows = rows[(rows["ts"] >= start) & (rows["ts"] <= end)]
        for span in sorted(spans):
            n = math.ceil(span * 20.0) + 1
            window = rows[-n:]
            if (
                len(window) != n
                or end - int(window["ts"][-1]) > allowed_age
                or np.any(np.diff(window["ts"]) != 60_000)
            ):
                continue
            # Rust rejects malformed prices. Missing history stays absent, never a zero score.
            value = passivbot_rust.calc_signed_unilateralness(
                window["c"].astype(float).tolist(), span
            )
            source_ends[symbol] = int(window["ts"][-1])
            ranking[symbol][span] = value
            if replay_cache is not None:
                # Fetching may have replaced the symbol's cache. Store the actual
                # source cutoff; only cache-only ranking can reuse an older window.
                replay_cache.setdefault(symbol, {})[keys[span]] = (
                    value, source_ends[symbol], int(bot.get_exchange_time())
                )
            if window["ts"][-1] == end:
                result[symbol][span] = value
            else:
                logging.debug(
                    "[ema] cached unilateralness | symbol=%s span=%s source=completed_candles age_ms=%d max_age_ms=%d",
                    symbol,
                    span,
                    end - int(window["ts"][-1]),
                    allowed_age,
                )
        await asyncio.sleep(0)
    final_end = int(bot.get_exchange_time()) // 60_000 * 60_000 - 60_000
    if final_end != end:
        # All symbols were read against one cutoff. A rollover during a slow
        # load invalidates current cooldown inputs; do not privilege later fetches.
        for symbol in symbols:
            result[symbol].clear()
            if final_end - source_ends.get(symbol, end) > max(0, int(ages.get(symbol, 0))):
                ranking[symbol].clear()
    missing = {}
    for symbol, (current_spans, forager_spans) in required.items():
        unavailable = {
            "current": sorted(current_spans - result[symbol].keys()),
            "forager": sorted(forager_spans - ranking[symbol].keys()),
        }
        if any(unavailable.values()):
            missing[symbol] = unavailable
    return result, ranking, missing
