"""Completed-candle transport for Rust RMS directionality; no projected flat tails."""

import asyncio
import logging
import math
import time

import ccxt
import numpy as np
import passivbot_rust
from candlestick_manager import OhlcvFetchError
from config.entry_cooldown import uses_adverse_rms
from live.console_health import token
from live.diagnostic_safety import bounded_exception_type


RMS_CACHE_KIND = "signed_rms_directionality"


def history_candle_count(span):
    if not math.isfinite(span) or not 1 <= span <= 100_000:
        raise ValueError("RMS directionality EMA span must be between 1 and 100000")
    return math.ceil(span * 20.0) + 1


def scoring_enabled(bot, side, symbols):
    if bot.bot_value(side, "forager_score_weights")["rms_directionality"] <= 0:
        return False
    if bot.is_forager_mode(side):
        return True
    # With a fixed universe that fits the cap, an ordinary eligible held coin
    # consumes one candidate and one slot equally. Only held coins outside that
    # set can force ranking of the remaining flat candidates. Keep a conservative
    # superset here; Rust retains exact modes, costs and final slot decisions.
    held = {
        symbol for symbol in symbols
        if bot.positions.get(symbol, {}).get(side, {}).get("size", 0.0) != 0.0
    }
    if not held or len(symbols) - len(held) <= 1:
        return False
    approved = bot.approved_coins_minus_ignored_coins[side]
    unavailable = getattr(bot, "_orchestrator_ema_unavailable_symbols", set())
    markets = getattr(bot, "markets_dict", {})
    return any(
        symbol not in approved
        or bot.bp(side, "wallet_exposure_limit_ratio", symbol) == 0.0
        or not markets.get(symbol, {}).get("active", True)
        or symbol in unavailable
        for symbol in held
    )


def adverse_enabled(bot, side, symbol):
    weights = bot.bp(side, "entry_cooldown_weights_minutes", symbol)
    if weights["adverse_directionality"] <= 0:
        return False
    held = getattr(bot, "positions", {}).get(symbol, {}).get(side, {}).get("size", 0.0) != 0.0
    # Approval only governs initials. Held graceful-stop positions still DCA;
    # conservatively supply their inputs while Rust decides effective modes.
    if not held and not bot.is_approved(side, symbol):
        return False
    return uses_adverse_rms({
        "entry_cooldown_weights_minutes": weights,
        **{key: bot.bp(side, key, symbol) for key in (
            "risk_entry_cooldown_minutes", "entry_cooldown_min_duration_minutes",
            "entry_cooldown_max_duration_minutes",
        )},
    })


def _log_read_failure(bot, symbol, exc):
    error_type = bounded_exception_type(exc)
    logging.debug(
        "[ema] RMS directionality read failed | symbol=%s error_type=%s",
        token(symbol, 32), error_type,
    )
    states = getattr(bot, "_rms_read_warning_state", None)
    if states is None:
        states = {}
        bot._rms_read_warning_state = states
    now = time.monotonic()
    last, repeats = states.get(error_type, (None, 0))
    if last is not None and now - last < 300:
        states[error_type] = (last, repeats + 1)
        return
    logging.warning(
        "[ema] RMS directionality unavailable | sample=%s error_type=%s failures=%d",
        token(symbol, 32), error_type, repeats + 1,
    )
    states[error_type] = (now, 0)


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
                    if key[0] == RMS_CACHE_KIND:
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
            span = float(bot.bot_value(side, "rms_directionality_ema_span_minutes"))
            if adverse_enabled(bot, side, symbol):
                current_spans.add(span)
            if side in scoring_sides:
                forager_spans.add(span)
        spans = current_spans | forager_spans
        required[symbol] = (current_spans, forager_spans)
        cached = replay_cache.get(symbol, {}) if replay_cache is not None else {}
        for key in list(cached):
            if key[0] == RMS_CACHE_KIND and key[1] not in spans:
                del cached[key]
        if not spans:
            continue
        counts = {span: history_candle_count(span) for span in spans}
        keys = {span: (RMS_CACHE_KIND, span, "60000") for span in spans}
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
            source_ends[symbol] = {span: cached[key][1] for span, key in keys.items()}
            continue
        start = end - (max(counts.values()) - 1) * 60_000 - math.ceil(allowed_age / 60_000) * 60_000
        read_args = dict(start_ts=start, end_ts=end, timeframe="1m", standardize=False)
        remote = symbol not in cache_only_symbols
        try:
            try:
                rows = await bot.cm.get_candles(symbol, allow_remote_fetch=remote, **read_args)
            except asyncio.TimeoutError:
                if not remote:
                    raise
                # The shared deadline applies to remote work. A timed out
                # refresh can still leave a complete window in the local cache.
                logging.debug("[ema] RMS directionality retrying local candles | symbol=%s", symbol)
                rows = await bot.cm.get_candles(symbol, allow_remote_fetch=False, **read_args)
        except (ccxt.NetworkError, OhlcvFetchError, OSError, asyncio.TimeoutError) as exc:
            _log_read_failure(bot, symbol, exc)
            continue

        # Find the latest complete segment for each span. A partial segment
        # after a gap does not invalidate an older, still timely window.
        rows = rows[(rows["ts"] >= start) & (rows["ts"] <= end)]
        ts = rows["ts"]
        segment_ends = np.flatnonzero(np.diff(ts, append=-1) != 60_000) + 1
        segment_starts = np.r_[0, segment_ends][:len(segment_ends)]
        recent = end - ts[segment_ends - 1] <= allowed_age
        for span, count in sorted(counts.items()):
            ends = segment_ends[recent & (segment_ends - segment_starts >= count)]
            key = keys[span]
            # Read the canonical memo again after I/O. Ingestion can have
            # invalidated it if a correction touched the source window.
            memo = replay_cache.get(symbol, {}).get(key) if replay_cache is not None else None
            if memo is not None and 0 <= end - memo[1] <= allowed_age and (
                not len(ends) or memo[1] >= int(rows["ts"][ends[-1] - 1])
            ):
                value, source_end, _ = memo
            elif len(ends):
                stop = int(ends[-1])
                window = rows[stop - count:stop]
                value = passivbot_rust.calc_signed_rms_directionality(window["c"].astype(float).tolist(), span)
                source_end = int(window["ts"][-1])
                if replay_cache is not None:
                    replay_cache.setdefault(symbol, {})[key] = (value, source_end, int(bot.get_exchange_time()))
            else:
                logging.debug(
                    "[ema] RMS history incomplete | symbol=%s required=%d longest_recent=%d",
                    symbol, count, max(segment_ends[recent] - segment_starts[recent], default=0),
                )
                continue
            source_ends.setdefault(symbol, {})[span] = source_end
            ranking[symbol][span] = value
            if source_end == end:
                result[symbol][span] = value
            else:
                logging.debug(
                    "[ema] cached rms_directionality | symbol=%s span=%s source=completed_candles age_ms=%d max_age_ms=%d",
                    symbol, span, end - source_end, allowed_age,
                )
        await asyncio.sleep(0)
    final_end = int(bot.get_exchange_time()) // 60_000 * 60_000 - 60_000
    for symbol in symbols:
        # Earlier symbols may also have been corrected while later reads
        # awaited I/O. Recheck the canonical memo and age at the output cutoff.
        cached = replay_cache.get(symbol, {}) if replay_cache is not None else None
        allowed_age = max(0, int(ages.get(symbol, 0)))
        for span, value in list(ranking[symbol].items()):
            source_end = source_ends[symbol][span]
            memo = cached.get((RMS_CACHE_KIND, span, "60000")) if cached is not None else None
            if (
                not 0 <= final_end - source_end <= allowed_age
                or (cached is not None and (memo is None or memo[:2] != (value, source_end)))
            ):
                ranking[symbol].pop(span)
            if final_end != end or span not in ranking[symbol]:
                result[symbol].pop(span, None)
    missing = {}
    for symbol, (current_spans, forager_spans) in required.items():
        unavailable = {
            "current": sorted(current_spans - result[symbol].keys()),
            "forager": sorted(forager_spans - ranking[symbol].keys()),
        }
        if any(unavailable.values()):
            missing[symbol] = unavailable
    return result, ranking, missing
