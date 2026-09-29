"""Completed-candle transport for Rust RMS directionality; no projected flat tails."""

import asyncio
import logging
import math

import ccxt
import numpy as np
import passivbot_rust
from candlestick_manager import OhlcvFetchError
from config.entry_cooldown import uses_adverse_rms


def scoring_enabled(bot, side):
    return (
        bot.bot_value(side, "forager_score_weights")["unilateralness"] > 0
        and bot.is_forager_mode(side)
    )


def adverse_enabled(bot, side, symbol):
    weights = bot.bp(side, "entry_cooldown_weights_minutes", symbol)
    if weights["adverse_directionality"] <= 0:
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
    ages = forager_age_by_symbol or {}
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
            if scoring_enabled(bot, side):
                forager_spans.add(span)
        spans = current_spans | forager_spans
        required[symbol] = (current_spans, forager_spans)
        if not spans:
            continue
        if any(not math.isfinite(s) or not 1 <= s <= 100_000 for s in spans):
            raise ValueError("unilateralness EMA span must be between 1 and 100000")
        count = math.ceil(max(spans) * 20.0) + 1
        allowed_age = max(0, int(ages.get(symbol, 0)))
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
            ranking[symbol][span] = value
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
    missing = {}
    for symbol, (current_spans, forager_spans) in required.items():
        unavailable = {
            "current": sorted(current_spans - result[symbol].keys()),
            "forager": sorted(forager_spans - ranking[symbol].keys()),
        }
        if any(unavailable.values()):
            missing[symbol] = unavailable
    return result, ranking, missing
