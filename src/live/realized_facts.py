"""Deterministic account-wide ordering of canonical realized fill facts.

The caller owns fill coverage and authoritative-PnL readiness. This module does
not repair the ledger, select revisions, infer execution order from identifiers,
or use replay-derived position after-state as evidence. Unknown same-time order
uses a stable, explicitly approximate ordering; fills remain individual events.
"""

from dataclasses import dataclass
from itertools import groupby
import json
import math
from collections.abc import Iterable, Mapping


class RealizedFactsError(ValueError):
    """Canonical facts conflict or cannot safely supply realized memory."""


@dataclass(frozen=True)
class RealizedFacts:
    fills: tuple[dict, ...]
    reasons: tuple[str, ...]
    duplicate_count: int
    ambiguous_cohort_count: int
    proven_pair_chain_count: int


@dataclass(frozen=True)
class _Fill:
    key: tuple[str, str, str]
    timestamp: int
    pnl: float
    fee: float
    net: float
    chain: tuple[float, float] | None
    structural: tuple

    def payload(self):
        return {
            "identity": json.dumps(self.key, separators=(",", ":"), ensure_ascii=False),
            "timestamp": self.timestamp,
            "net": self.net,
        }


_MISSING = object()


def _field(event, key, *, required=True):
    value = event.get(key, _MISSING) if isinstance(event, Mapping) else getattr(event, key, _MISSING)
    if required and value is _MISSING:
        raise RealizedFactsError(f"realized fill is missing {key}")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RealizedFactsError(f"realized fill {field} must be a nonempty canonical string")
    return value


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise RealizedFactsError(f"realized fill {field} must be finite")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise RealizedFactsError(f"realized fill {field} must be finite") from exc
    if not math.isfinite(number):
        raise RealizedFactsError(f"realized fill {field} must be finite")
    return number


def _has_chain_evidence(raw):
    if raw is _MISSING or raw is None:
        return False
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        data = item.get("data", item)
        if isinstance(data, Mapping):
            info = data.get("info")
            if isinstance(info, Mapping) and "startPosition" in info:
                return True
    return False


def _chain_helpers():
    # Keep ordinary imports and cohorts without raw chain evidence independent
    # of the exchange/strategy/native-extension dependency graph. These helpers
    # are the same factual proof used by FillEventsManager, not a second solver.
    from fill_events_manager import (
        _hyperliquid_fill_position_chain,
        _unique_position_chain_order,
    )

    return _hyperliquid_fill_position_chain, _unique_position_chain_order


def _normalize(event):
    symbol = _text(_field(event, "symbol"), "symbol")
    pside = _field(event, "position_side")
    if pside not in ("long", "short"):
        raise RealizedFactsError("realized fill position_side must be long or short")
    identity = _text(_field(event, "id"), "id")
    timestamp = _field(event, "timestamp")
    if type(timestamp) is not int or not 0 <= timestamp <= 2**63 - 1:
        raise RealizedFactsError("realized fill timestamp must be an integer in the clock domain")
    pnl = _number(_field(event, "pnl"), "pnl")
    fee = _number(_field(event, "fee_paid"), "fee_paid")
    net = pnl + fee
    if not math.isfinite(net):
        raise RealizedFactsError("realized fill net PnL is outside the finite domain")

    # These optional structural facts detect conflicting repeated identities.
    # They never replace required PnL or prove chronology by themselves.
    side = _field(event, "side", required=False)
    qty = _field(event, "qty", required=False)
    price = _field(event, "price", required=False)
    if side is not _MISSING and side not in ("buy", "sell"):
        raise RealizedFactsError("realized fill side must be buy or sell")
    if qty is not _MISSING:
        qty = _number(qty, "qty")
    if price is not _MISSING:
        price = _number(price, "price")
    raw = _field(event, "raw", required=False)
    chain = None
    if _has_chain_evidence(raw):
        chain_for, _ = _chain_helpers()
        chain = chain_for({"raw": raw, "position_side": pside,
                           **({"side": side} if side is not _MISSING else {})})
        if chain is not None:
            chain = tuple(float(value) for value in chain)
            if any(not math.isfinite(value) or value < 0.0 for value in chain):
                chain = None
    return _Fill((symbol, pside, identity), timestamp, pnl, fee, net, chain,
                 (side, qty, price))


def canonical_realized_facts(events: Iterable[object]) -> RealizedFacts:
    """Return individual Rust payloads plus stable ordering diagnostics.

    At one timestamp, symbol/position-side blocks are sorted lexically. Within
    each block a unique independently proven raw position chain wins; otherwise
    identity text is a deterministic tie-breaker, never an execution sequence.
    Cross-pair interleaving is unknown and always disclosed. Exact duplicates
    collapse; conflicting copies of one identity fail instead of choosing a row.
    Input facts and their raw/derived metadata are never modified.
    """
    unique = {}
    duplicate_count = 0
    for event in events:
        fill = _normalize(event)
        previous = unique.get(fill.key)
        if previous is not None:
            if previous != fill:
                raise RealizedFactsError("conflicting realized fill identity")
            duplicate_count += 1
        else:
            unique[fill.key] = fill

    reasons = set()
    if duplicate_count:
        reasons.add("duplicate_realized_fill_collapsed")
    ordered = []
    ambiguous_count = 0
    proven_count = 0
    values = sorted(unique.values(), key=lambda fill: (fill.timestamp, fill.key))
    for _, cohort_rows in groupby(values, key=lambda fill: fill.timestamp):
        cohort = list(cohort_rows)
        pair_blocks = [list(rows) for _, rows in groupby(cohort, key=lambda fill: fill.key[:2])]
        ambiguous = len(pair_blocks) > 1
        for block in pair_blocks:
            if len(block) > 1:
                chain_order = None
                if all(fill.chain is not None for fill in block):
                    _, solve_chain = _chain_helpers()
                    chain_order = solve_chain([fill.chain for fill in block])
                if chain_order is None:
                    ambiguous = True
                else:
                    block = [block[index] for index in chain_order]
                    proven_count += 1
            ordered.extend(fill.payload() for fill in block)
        if ambiguous:
            reasons.add("ambiguous_same_time_realized_order")
            ambiguous_count += 1
    return RealizedFacts(tuple(ordered), tuple(sorted(reasons)), duplicate_count,
                         ambiguous_count, proven_count)
