"""Live factual snapshot -> shared Rust HSL decisions.

This adapter does no I/O and has no previous-decision input. Capture and evaluation
run without an await; async acquisition belongs before capture, execution after it.
Public runtime activation remains separately gated while orchestration is integrated.
"""
from dataclasses import dataclass
import json
import math

import passivbot_rust as pbr

from config.hsl_revised import FIELDS, engine
from live.hsl_revised_inputs import capture_fills
from passivbot_exceptions import FatalBotException


@dataclass(frozen=True)
class Scope:
    mode: str
    pside: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class Request:
    scope: Scope
    payload: str
    execution_type: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Unavailable:
    scope: Scope
    reason: str


@dataclass(frozen=True)
class Decision:
    scope: Scope
    action: str | None
    execution_type: str
    payload: str
    reasons: tuple[str, ...]


class InvalidHslOutput(FatalBotException):
    """Malformed native intent is a fatal producer error, never unavailability."""


def _finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (not positive or value > 0))


def _policies(bot, symbols):
    mode = bot.config["live"]["hsl_signal_mode"]
    if mode == "unified":
        policy = dict(bot.config["bot"]["hsl"])
        if policy["enabled"]:
            yield Scope(mode), policy, 1
    elif mode in {"coin", "pside"}:
        for side in ("long", "short"):
            for symbol in sorted(symbols) if mode == "coin" else (None,):
                policy = {key: bot.bp(side, "hsl_" + key, symbol) for key in FIELDS}
                if policy["enabled"]:
                    # Coin budgets use configured slots, not current open bags or
                    # the number of symbols whose history happened to arrive.
                    slots = int(round(bot.bot_value(side, "n_positions"))) if mode == "coin" else 1
                    yield Scope(mode, side, symbol), policy, slots
    else:
        raise ValueError("invalid revised HSL signal mode")


def capture(bot, quotes, candle_sources, *, symbols, now_ms, utc_now_ms,
            max_current_age_ms, fills_started_ms=None, fills_completed_ms=None):
    """Copy a current account cohort and canonical history into immutable requests.

    ``quotes`` are factual MarketSnapshots (UTC fetch times). Candle open and
    fill event times use the exchange clock; candle capture times use UTC. The optional fill interval is an
    observed remote fetch interval in UTC, never synthesized from a cache read.
    Missing interval evidence degrades lifecycle proof, not numeric evaluation.
    ``positions`` must be the complete successfully committed account snapshot;
    an omitted symbol in that complete response is an observed flat position.

    Missing current quotes are scoped before Rust. No part of a malformed Rust
    result can become a partial decision batch. Callers must not await between
    capture/evaluation and installing the resulting current permissions.
    """
    if engine(bot.config) != "revised":
        raise ValueError("revised HSL adapter requires revised engine")
    if (any(type(x) is not int for x in (now_ms, utc_now_ms, max_current_age_ms))
            or max_current_age_ms < 0):
        raise ValueError("invalid revised HSL capture clock")
    lookback_days = float(bot.config["live"]["pnls_max_lookback_days"])
    if not 1 <= lookback_days <= 90:
        raise ValueError("revised HSL requires 1..90 day lookback")
    lookback_ms = math.floor(lookback_days * 86_400_000 + .5)
    start = max(0, now_ms - lookback_ms)
    offset = now_ms - utc_now_ms
    ledger = bot._ensure_freshness_ledger()
    balance_state, position_state = (ledger.surfaces[key] for key in ("balance", "positions"))
    balance = bot.get_raw_balance()
    events = bot._pnls_manager.get_events(start_ms=start) if bot._pnls_manager is not None else []
    tape = capture_fills(events, bot.c_mults)
    by_pair = {(p.symbol, p.pside): p for p in tape.pairs}
    # Old flat history is intentionally outside the configured authority window.
    relevant = {p.symbol for p in tape.pairs if any(f.timestamp >= start for f in p.fills)}
    relevant.update(symbol for symbol, sides in bot.positions.items()
                    if any(side["size"] != 0 for side in sides.values()))
    if bot.config["live"]["hsl_signal_mode"] == "coin":
        relevant.update(symbols)
    policies = tuple(_policies(bot, relevant))

    def fresh(timestamp):
        return type(timestamp) is int and timestamp > 0 and 0 <= utc_now_ms - timestamp <= max_current_age_ms

    problem = None
    if not _finite(balance, positive=True) or not fresh(balance_state.updated_ms):
        problem = "current_balance_unavailable"
    elif not fresh(position_state.updated_ms):
        problem = "current_positions_unavailable"
    if problem:
        return (), tuple(Unavailable(scope, problem) for scope, _, _ in policies)
    for timestamp in (fills_started_ms, fills_completed_ms):
        if timestamp is not None and (type(timestamp) is not int or timestamp <= 0 or timestamp > utc_now_ms):
            raise ValueError("invalid observed fill capture interval")
    if (fills_started_ms is not None and fills_completed_ms is not None
            and fills_started_ms > fills_completed_ms):
        raise ValueError("reversed observed fill capture interval")

    pairs, pair_problems, pair_reasons = {}, {}, {}
    projected = {}
    for scope, _, _ in policies:
        selected_symbols = (scope.symbol,) if scope.symbol is not None else sorted(relevant)
        selected_sides = (scope.pside,) if scope.pside else ("long", "short")
        for symbol in selected_symbols:
            for side in selected_sides:
                key = (symbol, side)
                if key in pairs or key in pair_problems:
                    continue
                # Complete account snapshot absence means flat; no price/basis is
                # invented for exposure. Flat pairs still retain their cashflows.
                position = bot.positions.get(symbol, {}).get(side, {"size": 0.0, "price": 0.0})
                size, basis = position["size"], position["price"]
                if (not _finite(size) or (size != 0 and not _finite(basis, positive=True))
                        or (side == "long" and size < 0) or (side == "short" and size > 0)):
                    pair_problems[key] = "current_position_unavailable"
                    continue
                multiplier = bot.c_mults.get(symbol)
                if not _finite(multiplier, positive=True):
                    pair_problems[key] = "current_contract_metadata_unavailable"
                    continue
                if symbol not in projected:
                    source = candle_sources.get(symbol)
                    candles = [{**row, "available_at": row["available_at"] + offset}
                               for row in source.payload()] if source is not None else []
                    projected[symbol] = json.loads(pbr.hsl_revised_prices(json.dumps(
                        dict(candles=candles, start=start, end=now_ms), allow_nan=False)))
                prices = projected[symbol]
                quote = quotes.get(symbol)
                quote_valid = (quote is not None and quote.is_valid()
                               and 0 < quote.fetched_ms <= utc_now_ms)
                if quote_valid and (size == 0 or fresh(quote.fetched_ms)):
                    mark, mark_at = quote.last, quote.fetched_ms + offset
                    flat_price_reason = None
                elif size == 0 and prices["rows"]:
                    # Flat UPNL is zero. A retained source close can value this
                    # history without requiring a live quote from a delisted market.
                    last = prices["rows"][-1]
                    mark, mark_at = last["close"], last["source_end"]
                    flat_price_reason = "flat_historical_close"
                else:
                    pair_problems[key] = "current_mark_unavailable"
                    continue
                fills = by_pair.get(key)
                reasons = set(tape.reasons) | set(prices["reasons"])
                if flat_price_reason:
                    reasons.add(flat_price_reason)
                if fills is not None:
                    reasons.update(fills.reasons)
                source = candle_sources.get(symbol)
                if source is not None:
                    for source_tape in source.tapes:
                        reasons.update(source_tape.reasons)
                    for failure in source.failures:
                        reasons.add("candle_" + failure.stage + "_unavailable:" + failure.timeframe)
                pair_reasons[key] = reasons
                pairs[key] = dict(
                    symbol=symbol, position=dict(size=size, basis=basis if size != 0 else 0.0,
                        mark=mark, multiplier=multiplier, inverse=bot.inverse, pside=side,
                        quantity_step=bot.qty_steps.get(symbol)),
                    position_at=position_state.updated_ms + offset,
                    mark_at=mark_at,
                    fills_started_at=None if fills_started_ms is None else fills_started_ms + offset,
                    fills_at=None if fills_completed_ms is None else fills_completed_ms + offset,
                    prices_at=now_ms, prices={str(row["timestamp"]): row["close"] for row in prices["rows"]},
                    fills=fills.payload() if fills is not None else [],
                    revisions=[position_state.revision, 0, 0, 0],
                    fills_position_anchor=None)
    requests, unavailable = [], []
    for scope, policy, slots in policies:
        keys = [(symbol, side) for symbol in ((scope.symbol,) if scope.symbol else sorted(relevant))
                for side in ((scope.pside,) if scope.pside else ("long", "short"))]
        problems = sorted({pair_problems[key] for key in keys if key in pair_problems})
        if problems:
            unavailable.append(Unavailable(scope, ",".join(problems)))
            continue
        snapshot = dict(now=now_ms, start=start, balance=balance,
            balance_at=balance_state.updated_ms + offset, config_at=now_ms,
            max_current_age_ms=max_current_age_ms, mode=scope.mode, pside=scope.pside,
            symbol=scope.symbol, pairs=[pairs[key] for key in keys])
        payload = dict(snapshot=snapshot, slots=slots, span=policy["ema_span_minutes"],
            threshold=policy["red_threshold"],
            cooldown_ms=math.floor(policy["cooldown_minutes_after_red"] * 60_000 + .5),
            restart=policy["restart_after_red_policy"],
            intervention=bot.config["live"]["hsl_position_during_cooldown_policy"])
        reasons = set(tape.reasons)
        for key in keys:
            reasons.update(pair_reasons[key])
        requests.append(Request(scope, json.dumps(payload, allow_nan=False),
                                policy["panic_close_order_type"], tuple(sorted(reasons))))
    return tuple(requests), tuple(unavailable)


def evaluate(requests):
    """Evaluate the entire captured batch before exposing any current intent."""
    try:
        return _evaluate(requests)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidHslOutput("invalid revised HSL native evaluation") from exc


def _evaluate(requests):
    decisions = []
    for request in requests:
        submitted = json.loads(request.payload)
        output = json.loads(pbr.hsl_revised_evaluate(request.payload))
        decision = output["decision"]
        inactive = request.scope.mode == "coin" and submitted["slots"] == 0
        if (decision is None) != inactive:
            raise InvalidHslOutput("unexpected revised HSL scope activity")
        if decision is not None:
            if (decision["timestamp"] != submitted["snapshot"]["now"]
                    or decision["action"] not in {"normal", "panic", "halted"}
                    or any(not _finite(decision[key]) or not 0 <= decision[key] <= 1 for key in ("raw", "ema"))):
                raise InvalidHslOutput("invalid revised HSL decision envelope")
        reasons = tuple(sorted(set(request.reasons) | set(output["reasons"])))
        decisions.append(Decision(request.scope, None if decision is None else decision["action"],
            request.execution_type, json.dumps(output, allow_nan=False), reasons))
    return tuple(decisions)
