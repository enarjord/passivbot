"""Live consumer policy and diagnostics for completed-candle RMS directionality."""

import asyncio
import logging
import time

import ccxt
from candlestick_manager import ONE_MIN_MS, OhlcvFetchError
from config.entry_cooldown import uses_adverse_rms
from live.console_health import token
from live.diagnostic_safety import bounded_exception_type
from live.event_bus import EventTags, EventTypes, emit_event, format_rms_read_failed_console
from live.events import run_diagnostic_step


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
    states = getattr(bot, "_rms_read_warning_state", None)
    if states is None:
        states = {}
        bot._rms_read_warning_state = states
    now = time.monotonic()
    last, repeats = states.get(error_type, (None, 0))
    show_warning = last is None or now - last >= 300
    states[error_type] = (now, 0) if show_warning else (last, min(repeats + 1, 999_998))
    data = {
        "sample": token(symbol, 32),
        "error_type": error_type,
        "failures": repeats + 1 if show_warning else 1,
    }
    event = emit_event(bot, {
        "event_type": EventTypes.EMA_RMS_READ_FAILED,
        "level": "warning" if show_warning else "debug",
        "component": "ema.rms",
        "tags": [EventTags.EMA, EventTags.UNAVAILABLE],
        "status": "degraded",
        "data": data,
    })
    console = getattr(bot, "_live_event_console_available", None)
    if event is None or not (callable(console) and console()):
        logging.log(
            logging.WARNING if show_warning else logging.DEBUG,
            format_rms_read_failed_console(data),
        )


async def read_required(bot, symbols, cache_only_symbols, forager_age_by_symbol=None):
    """Collect the strict cooldown and bounded-age ranking RMS inputs."""
    current = {symbol: {} for symbol in symbols}
    ranking = {symbol: {} for symbol in symbols}
    active_sides = [side for side in ("long", "short") if bot.is_pside_enabled(side)]
    side_spans = {
        side: float(bot.bot_value(side, "rms_directionality_ema_span_minutes"))
        for side in active_sides
    }
    ranking_spans = {
        side_spans[side] for side in active_sides if scoring_enabled(bot, side, symbols)
    }
    required = {
        symbol: {
            "current": {
                side_spans[side] for side in active_sides if adverse_enabled(bot, side, symbol)
            },
            "forager": ranking_spans,
        }
        for symbol in symbols
    }
    bot.cm.prune_rms_emas(required)
    ages = {
        symbol: max(0, int((forager_age_by_symbol or {}).get(symbol, 0)))
        for symbol in symbols
    }
    end = int(bot.get_exchange_time()) // ONE_MIN_MS * ONE_MIN_MS - ONE_MIN_MS
    observations = {}
    for symbol, spans in required.items():
        if not any(spans.values()):
            continue
        try:
            observations[symbol] = await bot.cm.get_rms_emas(
                symbol,
                spans,
                end_ts=end,
                forager_max_age_ms=ages[symbol],
                allow_remote_fetch=symbol not in cache_only_symbols,
            )
        except (ccxt.NetworkError, OhlcvFetchError, OSError, TimeoutError) as exc:
            run_diagnostic_step(
                "record RMS candle read failure",
                lambda exc=exc: _log_read_failure(bot, symbol, exc),
            )
        await asyncio.sleep(0)

    final_end = int(bot.get_exchange_time()) // ONE_MIN_MS * ONE_MIN_MS - ONE_MIN_MS
    missing = {}
    for symbol, spans in required.items():
        for scope, values in observations.get(symbol, {}).items():
            output = current[symbol] if scope == "current" else ranking[symbol]
            age_ms = 0 if scope == "current" else ages[symbol]
            for span, observation in values.items():
                value, source_end = observation
                # Earlier symbols may have been corrected during a later read.
                if (
                    0 <= final_end - source_end <= age_ms
                    and bot.cm.cached_rms_ema(symbol, scope, span) == observation
                ):
                    output[span] = value
        unavailable = {
            "current": sorted(spans["current"] - current[symbol].keys()),
            "forager": sorted(spans["forager"] - ranking[symbol].keys()),
        }
        if any(unavailable.values()):
            missing[symbol] = unavailable
    return current, ranking, missing
