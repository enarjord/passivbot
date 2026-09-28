"""Completed-candle transport for Rust RMS directionality; no projected flat tails."""

import asyncio
import logging
import math

import ccxt
import numpy as np
import passivbot_rust
from candlestick_manager import OhlcvFetchError


async def load(bot, symbols, cache_only_symbols, forager_age_by_symbol=None):
    result = {symbol: {} for symbol in symbols}
    ranking = {symbol: {} for symbol in symbols}
    missing = set()
    ages = forager_age_by_symbol or {}
    end = int(bot.get_exchange_time()) // 60_000 * 60_000 - 60_000
    for symbol in symbols:
        spans = set()
        for side in ("long", "short"):
            weights = bot.bp(side, "entry_cooldown_weights_minutes", symbol)
            scoring = bot.bot_value(side, "forager_score_weights")
            if weights["adverse_directionality"] > 0 or scoring["unilateralness"] > 0:
                spans.add(float(bot.bot_value(side, "unilateralness_ema_span_1m")))
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
            missing.add(symbol)
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
                missing.add(symbol)
                continue
            # Rust rejects malformed prices. Missing history stays absent, never a zero score.
            value = passivbot_rust.calc_signed_unilateralness(
                window["c"].astype(float).tolist(), span
            )
            ranking[symbol][span] = value
            if window["ts"][-1] == end:
                result[symbol][span] = value
            else:
                missing.add(symbol)
                logging.debug(
                    "[ema] cached unilateralness | symbol=%s span=%s source=completed_candles age_ms=%d max_age_ms=%d",
                    symbol,
                    span,
                    end - int(window["ts"][-1]),
                    allowed_age,
                )
        await asyncio.sleep(0)
    return result, ranking, missing
