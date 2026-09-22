"""Live factual snapshot -> shared Rust HSL decisions.

This adapter does no I/O and has no previous-decision input. Capture and evaluation
run without an await; async acquisition belongs before capture, execution after it.
Public runtime activation remains separately gated while orchestration is integrated.
"""
from dataclasses import dataclass, replace
from collections import Counter
import json
import math

import passivbot_rust as pbr

from config.hsl_revised import FIELDS, engine
from live.hsl_revised_inputs import FillTape, capture_fills
from passivbot_exceptions import FatalBotException


@dataclass(frozen=True)
class PositionObservation:
    """An actual complete position read, never a reconstructed timestamp.

    A later identical account read need not erase an earlier still-fresh read
    which precedes a fill-tail fetch. The owner must discard this factual cache
    on account invalidation; the adapter also checks values and observation times.
    Omitted flat rows and the planner's explicit zero-size rows mean the same
    thing in a complete account response.
    """
    payload: str
    observed_ms: int
    generation: int


def observe_positions(bot):
    return PositionObservation(
        json.dumps(sorted((symbol, side, position['size'], position['price'])
                          for symbol, sides in bot.positions.items()
                          for side, position in sides.items() if position['size'] != 0)),
        bot._ensure_freshness_ledger().surfaces['positions'].updated_ms,
        int(getattr(bot, '_account_invalidation_generation', 0)))


def observe_open_orders(bot):
    """Immutable complete order facts, ignoring bucket/order iteration and flat padding.

    Keep the complete normalized row rather than inferring a venue-independent
    subset of reconciliation inputs. A harmless metadata change may defer a plan;
    it must never let changed resting orders inherit its permission.
    """
    return tuple(sorted((symbol, json.dumps(order, sort_keys=True, allow_nan=False))
                        for symbol, orders in bot.open_orders.items() for order in orders))


@dataclass(frozen=True)
class FillObservation:
    """Immutable normalized facts copied at successful remote-fetch completion."""
    manager: object
    start_ms: int
    tape: FillTape
    interval: tuple[int, int]


def observe_fills(bot, interval):
    now = int(bot.get_exchange_time())
    start = max(0, now - math.floor(float(bot.config['live']['pnls_max_lookback_days']) * 86_400_000 + .5))
    manager = bot._pnls_manager
    return FillObservation(manager, start,
        capture_fills(manager.get_events(start_ms=start), bot.c_mults), interval)


def observed_fill_interval(bot, tape, start, now):
    observation = getattr(bot, '_hsl_revised_fill_observation', None)
    if (observation is None or observation.manager is not bot._pnls_manager
            or bot._pnls_manager is None or start < observation.start_ms):
        return None

    def retained_facts(value):
        # Event order is not execution sequence. Expired facts cannot invalidate
        # the receipt of retained facts. Completeness diagnostics are facts too:
        # undated/unattributed rows and quality-only corrections cannot inherit a
        # successful acquisition interval from a different canonical tape.
        return (value.reasons, {
            (pair.symbol, pair.pside): (pair.reasons, Counter(
                fill for fill in pair.fills if start <= fill.timestamp <= now))
            for pair in value.pairs
            if pair.reasons or any(start <= fill.timestamp <= now for fill in pair.fills)})

    return observation.interval if retained_facts(tape) == retained_facts(observation.tape) else None


@dataclass(frozen=True)
class Scope:
    mode: str
    pside: str | None = None
    symbol: str | None = None


@dataclass(frozen=True)
class Request:
    scope: Scope
    metadata: str
    execution_type: str
    reasons: tuple[str, ...]
    price_grids: tuple[object, ...]
    mark_observed_ms: tuple[int, ...]

    @property
    def payload(self):
        """Materialize a standalone reference/debug snapshot only on demand."""
        value = json.loads(self.metadata)
        for pair, grid in zip(value["snapshot"]["pairs"], self.price_grids, strict=True):
            pair["prices"] = grid.values()
        return json.dumps(value, allow_nan=False)


@dataclass(frozen=True)
class Unavailable:
    scope: Scope
    reason: str


@dataclass(frozen=True)
class Decision:
    threshold: float
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


def _source_reasons(source):
    if source is None:
        return set()
    return ({reason for tape in source.tapes for reason in tape.reasons}
            | {"candle_" + failure.stage + "_unavailable:" + failure.timeframe
               for failure in source.failures})


def _policies(bot, pairs):
    mode = bot.config["live"]["hsl_signal_mode"]
    if mode == "unified":
        policy = dict(bot.config["bot"]["hsl"])
        if policy["enabled"]:
            yield Scope(mode), policy, 1
    elif mode in {"coin", "pside"}:
        for side in ("long", "short"):
            for symbol in sorted(s for s, pside in pairs if pside == side) if mode == "coin" else (None,):
                policy = {key: bot.config["bot"][side]["hsl"][key] for key in FIELDS}
                if symbol is not None:
                    # init_coin_overrides resolves authored identifiers to
                    # exchange symbols. Merge its canonical partial HSL block;
                    # bp's legacy global fallback expects removed flat keys.
                    policy.update(bot.coin_overrides.get(symbol, {}).get("bot", {}).get(side, {}).get("hsl", {}))
                if policy["enabled"]:
                    # Coin budgets use configured slots, not current open bags or
                    # the number of symbols whose history happened to arrive.
                    slots = int(round(bot.bot_value(side, "n_positions"))) if mode == "coin" else 1
                    yield Scope(mode, side, symbol), policy, slots
    else:
        raise ValueError("invalid revised HSL signal mode")


def capture(bot, quotes, candle_sources, *, symbols, now_ms, utc_now_ms,
            max_current_age_ms, fills_started_ms=None, fills_completed_ms=None,
            position_observation=None, use_observed_fills=False, target=None):
    """Copy a current account cohort and canonical history into immutable requests.

    ``quotes`` are factual MarketSnapshots (UTC fetch times). Candle open and
    fill event times use the exchange clock; candle capture times use UTC. The optional fill interval is an
    observed remote fetch interval in UTC, never synthesized from a cache read.
    Missing interval evidence degrades lifecycle proof, not numeric evaluation.
    ``symbols`` maps each position side to its currently eligible symbols; it
    cannot grant a symbol eligibility on the opposite side.
    ``target`` optionally selects the one symbol/position side whose order needs
    permission; an aggregate scope still retains every contributing pair.
    ``positions`` must be the complete successfully committed account snapshot;
    an omitted symbol in that complete response is an observed flat position.

    Missing current quotes are scoped before Rust. No part of a malformed Rust
    result can become a partial decision batch. Callers must not await between
    capture/evaluation and installing the resulting current permissions.
    """
    if engine(bot.config) != "revised":
        raise ValueError("revised HSL adapter requires revised engine")
    if (not isinstance(symbols, dict) or set(symbols) != {"long", "short"}
            or any(not isinstance(selected, (list, tuple, set, frozenset))
                   or any(not isinstance(symbol, str) or not symbol for symbol in selected)
                   for selected in symbols.values())):
        raise ValueError("revised HSL symbols require explicit long/short membership")
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
    if position_observation is not None:
        current = observe_positions(bot)
        if (not isinstance(position_observation, PositionObservation)
                or position_observation.payload != current.payload
                or position_observation.generation != current.generation
                or type(position_observation.observed_ms) is not int
                or not 0 < position_observation.observed_ms <= current.observed_ms):
            raise ValueError("position observation does not match current account facts")
        position_state = replace(position_state, updated_ms=position_observation.observed_ms)
    balance = bot.get_raw_balance()
    events = bot._pnls_manager.get_events(start_ms=start) if bot._pnls_manager is not None else []
    tape = capture_fills(events, bot.c_mults)
    if use_observed_fills:
        if fills_started_ms is not None or fills_completed_ms is not None:
            raise ValueError('cannot combine observed fills with a supplied fill interval')
        interval = observed_fill_interval(bot, tape, start, now_ms)
        if interval is not None:
            fills_started_ms, fills_completed_ms = interval
    by_pair = {(p.symbol, p.pside): p for p in tape.pairs}
    # Old flat history is intentionally outside the configured authority window.
    relevant = {(p.symbol, p.pside) for p in tape.pairs
                if any(start <= f.timestamp <= now_ms for f in p.fills)}
    relevant.update((symbol, side) for symbol, sides in bot.positions.items()
                    for side, position in sides.items() if position["size"] != 0)
    global_reasons = set(tape.reasons)
    if fills_started_ms is None or fills_completed_ms is None:
        global_reasons.add("fill_capture_unknown")
    if any(f.timestamp > now_ms for p in tape.pairs for f in p.fills):
        global_reasons.add("future_fill_outside_evaluation")
    if bot.config["live"]["hsl_signal_mode"] == "coin":
        relevant.update((symbol, side) for side, selected in symbols.items() for symbol in selected)
    if target is not None and (not isinstance(target, tuple) or len(target) != 2
            or not isinstance(target[0], str) or not target[0] or target[1] not in {"long", "short"}):
        raise ValueError("invalid revised HSL permission target")
    policies = tuple((scope, policy, slots) for scope, policy, slots in _policies(bot, relevant)
        if target is None or ((scope.symbol is None or scope.symbol == target[0])
                             and (scope.pside is None or scope.pside == target[1])))

    def fresh(timestamp):
        return type(timestamp) is int and timestamp > 0 and 0 <= utc_now_ms - timestamp <= max_current_age_ms

    pending = getattr(bot, '_authoritative_pending_confirmations', {})
    balance_invalidated = int(pending.get('balance', 0)) > balance_state.epoch
    positions_invalidated = int(pending.get('positions', 0)) > position_state.epoch
    problem = None
    if balance_invalidated or not _finite(balance, positive=True) or not fresh(balance_state.updated_ms):
        problem = "current_balance_unavailable"
    elif positions_invalidated or not fresh(position_state.updated_ms):
        problem = "current_positions_unavailable"
    if problem:
        return (), tuple(Unavailable(scope, problem) for scope, _, _ in policies)
    for timestamp in (fills_started_ms, fills_completed_ms):
        if timestamp is not None and (type(timestamp) is not int or timestamp <= 0 or timestamp > utc_now_ms):
            raise ValueError("invalid observed fill capture interval")
    if (fills_started_ms is not None and fills_completed_ms is not None
            and fills_started_ms > fills_completed_ms):
        raise ValueError("reversed observed fill capture interval")

    scoped_keys = {}
    for scope, _, _ in policies:
        keys = []
        for symbol, side in sorted(relevant):
            if ((scope.symbol is None or scope.symbol == symbol)
                    and (scope.pside is None or scope.pside == side)):
                keys.append((symbol, side))
        scoped_keys[scope] = keys

    pairs, pair_problems, pair_reasons, flat_coins = {}, {}, {}, {}
    projected = {}
    for scope, _, _ in policies:
        for symbol, side in scoped_keys[scope]:
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
            history = by_pair.get(key)
            quote = quotes.get(symbol)
            source = candle_sources.get(symbol)
            if (scope.mode == "coin" and size == 0 and quote is None
                    and (source is None or not any(t.candles for t in source.tapes))
                    and bot._pnls_manager is not None
                    and (history is None or (not history.fills and not history.reasons))
                    and not tape.reasons and fills_started_ms is not None
                    and fills_completed_ms == position_state.updated_ms):
                # This is the native contract's simultaneous factual flat/empty
                # cohort, not a cache read relabelled as a fill observation.
                flat_coins[scope] = dict(symbol=symbol, pside=side,
                    position_at=position_state.updated_ms + offset,
                    fills_at=fills_completed_ms + offset, history_start=start)
                pair_reasons[key] = global_reasons | _source_reasons(source)
                continue
            multiplier = bot.c_mults.get(symbol)
            if not _finite(multiplier, positive=True):
                pair_problems[key] = "current_contract_metadata_unavailable"
                continue
            if symbol not in projected:
                source = candle_sources.get(symbol)
                candles = [(c.start, c.minutes, c.open, c.high, c.low, c.close,
                            c.available_at + offset)
                           for tape in source.tapes for c in tape.candles] if source is not None else []
                projected[symbol] = pbr.hsl_revised_native_price_grid(start, now_ms, candles)
            _grid, last_price, price_reasons = projected[symbol]
            quote = quotes.get(symbol)
            flat_fill_prices = [(fill.timestamp, fill.price) for fill in history.fills
                                if start <= fill.timestamp <= now_ms and _finite(fill.price, positive=True)] if history else []
            quote_valid = (quote is not None and quote.is_valid()
                           and 0 < quote.fetched_ms <= utc_now_ms)
            if quote_valid and (size == 0 or fresh(quote.fetched_ms)):
                mark, mark_at = quote.last, quote.fetched_ms + offset
                flat_price_reason = None
            elif size == 0 and last_price is not None:
                # Flat UPNL is zero. A retained source close can value this
                # history without requiring a live quote from a delisted market.
                mark, mark_at = last_price
                flat_price_reason = "flat_historical_close"
            elif size == 0 and flat_fill_prices:
                # A flat position has zero current UPNL. Its latest factual fill
                # price suffices for sparse replay when a delisted market has
                # neither quotes nor candles. The tie-break selects a price,
                # never an execution ordering or a mark for held exposure.
                mark_at, mark = max(flat_fill_prices)
                flat_price_reason = "flat_historical_fill_price"
            else:
                pair_problems[key] = "current_mark_unavailable"
                continue
            fills = by_pair.get(key)
            reasons = global_reasons | set(price_reasons)
            if flat_price_reason:
                reasons.add(flat_price_reason)
            if fills is not None:
                reasons.update(fills.reasons)
            reasons.update(_source_reasons(source))
            pair_reasons[key] = reasons
            pairs[key] = dict(
                symbol=symbol, position=dict(size=size, basis=basis if size != 0 else 0.0,
                    mark=mark, multiplier=multiplier, inverse=bot.inverse, pside=side,
                    quantity_step=bot.qty_steps.get(symbol)),
                position_at=position_state.updated_ms + offset,
                mark_at=mark_at,
                fills_started_at=None if fills_started_ms is None else fills_started_ms + offset,
                fills_at=None if fills_completed_ms is None else fills_completed_ms + offset,
                prices_at=now_ms, prices={},
                fills=fills.payload() if fills is not None else [],
                revisions=[position_state.revision, 0, 0, 0],
                fills_position_anchor=None)
    requests, unavailable = [], []
    for scope, policy, slots in policies:
        keys = scoped_keys[scope]
        problems = sorted({pair_problems[key] for key in keys if key in pair_problems})
        if problems:
            unavailable.append(Unavailable(scope, ",".join(problems)))
            continue
        snapshot = dict(now=now_ms, start=start, balance=balance,
            balance_at=balance_state.updated_ms + offset, config_at=now_ms,
            max_current_age_ms=max_current_age_ms, mode=scope.mode, pside=scope.pside,
            symbol=scope.symbol, pairs=[pairs[key] for key in keys if key in pairs])
        if scope in flat_coins:
            snapshot["flat_coin"] = flat_coins[scope]
        payload = dict(snapshot=snapshot, slots=slots, span=policy["ema_span_minutes"],
            threshold=policy["red_threshold"],
            cooldown_ms=math.floor(policy["cooldown_minutes_after_red"] * 60_000 + .5),
            restart=policy["restart_after_red_policy"])
        reasons = set(global_reasons)
        # An attributed row may have no usable timestamp and therefore cannot
        # establish a price-bearing contributor. Its quality still belongs to
        # the known aggregate scope, without inventing market requirements.
        if scope.mode != "coin":
            for pair in tape.pairs:
                if scope.pside is None or pair.pside == scope.pside:
                    reasons.update(pair.reasons)
        for key in keys:
            reasons.update(pair_reasons.get(key, ()))
        requests.append(Request(scope, json.dumps(payload, allow_nan=False),
                                policy["panic_close_order_type"], tuple(sorted(reasons)),
                                tuple(projected[key[0]][0] for key in keys if key in pairs),
                                tuple(pair["mark_at"] - offset for pair in snapshot["pairs"]
                                      if pair["position"]["size"] != 0)))
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
        submitted = json.loads(request.metadata)
        output = json.loads(pbr.hsl_revised_evaluate_grids(request.metadata, request.price_grids))
        decision = output["decision"]
        inactive = request.scope.mode == "coin" and submitted["slots"] == 0
        if (decision is None) != inactive:
            raise InvalidHslOutput("unexpected revised HSL scope activity")
        if decision is not None:
            snapshot = submitted["snapshot"]
            exposed = any(pair["position"]["size"] != 0 for pair in snapshot["pairs"])
            red, flat = decision["red_at"], decision["flat_at"]
            if (type(decision["timestamp"]) is not int or decision["timestamp"] != snapshot["now"]
                    or decision["action"] not in {"normal", "panic", "halted"}
                    or any(not _finite(decision[key]) or decision[key] < 0 for key in ("raw", "ema"))
                    or any(t is not None and (type(t) is not int or not snapshot["start"] <= t <= snapshot["now"])
                           for t in (red, flat))
                    or not isinstance(decision["reason"], str) or not decision["reason"]
                    or type(decision["numeric_range_approximation"]) is not bool
                    or (decision["action"] == "normal" and (red is not None or flat is not None))
                    or (decision["action"] != "normal" and red is None)
                    or (decision["action"] == "panic" and (not exposed or flat is not None))
                    or (decision["action"] == "halted" and exposed)
                    or (flat is not None and (red is None or flat < red))):
                raise InvalidHslOutput("invalid revised HSL decision envelope")
        reasons = tuple(sorted(set(request.reasons) | set(output["reasons"])))
        decisions.append(Decision(submitted["threshold"], request.scope, None if decision is None else decision["action"],
            request.execution_type, json.dumps(output, allow_nan=False), reasons))
    return tuple(decisions)
