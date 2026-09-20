"""Test-only scope-boundary and snapshot experiments for the revised HSL.

These compose the independent oracle; they are not installed in the live loop.
No cache/journal state or prior decisions are accepted as reconstruction inputs.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import groupby

from hsl_reference import (
    Fill, Observation, Position, dec, ordered_fills, reconstruct, signal,
)


def _number(value):
    # Capture valid numeric values by value; preserve missing/malformed historical
    # fields as immutable text/None so the oracle can disclose their degradation.
    if value is None:
        return None
    try:
        return dec(value)
    except (ValueError, ArithmeticError):
        return str(value)


def _quantity(fill):
    try:
        value = dec(fill.delta)
        return value, value != 0
    except (ValueError, ArithmeticError):
        return Decimal(0), False


@dataclass(frozen=True)
class Pair:
    symbol: str
    position: Position
    position_at: int
    mark_at: int
    fills: tuple
    prices: tuple

    @property
    def key(self):
        return self.symbol, self.position.pside


def capture_pair(symbol, position, position_at, mark_at, fills, prices):
    size, basis, mark, multiplier = position.validate()
    copied_fills = tuple(replace(f, delta=_number(f.delta), price=_number(f.price),
                                realized=_number(f.realized), fee=_number(f.fee)) for f in fills)
    return Pair(symbol, replace(position, size=size, basis=basis, mark=mark,
                                multiplier=multiplier), position_at, mark_at,
                copied_fills, tuple(sorted((t, dec(p)) for t, p in prices.items())))


@dataclass(frozen=True)
class Settings:
    ema_span: object = 1
    threshold: object = ".05"
    slots: int = 1


@dataclass(frozen=True)
class Snapshot:
    now: int
    start: int
    balance: Decimal
    balance_at: int
    pairs: tuple
    settings: Settings
    # balance, positions, marks, fills, prices, config; monotonic producer versions.
    revisions: tuple


def capture(now, start, balance, balance_at, pairs, settings=Settings(), revisions=(0,) * 6,
            max_current_age=120_000):
    pairs = tuple(sorted(pairs, key=lambda p: p.key))
    balance = dec(balance)
    if start > now or balance <= 0 or max_current_age < 0:
        raise ValueError("invalid minimum snapshot")
    if len({p.key for p in pairs}) != len(pairs) or len(revisions) != 6:
        raise ValueError("invalid snapshot shape")
    observed = [balance_at, *(p.position_at for p in pairs), *(p.mark_at for p in pairs)]
    if any(not now - max_current_age <= t <= now for t in observed):
        raise ValueError("unusable current observation")
    for p in pairs:
        p.position.validate()
    settings = replace(settings, ema_span=dec(settings.ema_span), threshold=dec(settings.threshold))
    return Snapshot(now, start, balance, balance_at, pairs, settings, tuple(revisions))


def selected_pairs(snapshot, mode, *, pside=None, symbol=None):
    if mode == "unified":
        if pside is not None or symbol is not None:
            raise ValueError("unified has no side/symbol selector")
        return snapshot.pairs
    if mode not in ("pside", "coin") or pside not in ("long", "short"):
        raise ValueError("invalid scope")
    if (mode == "coin") != (symbol is not None):
        raise ValueError("invalid symbol selector")
    return tuple(p for p in snapshot.pairs if p.position.pside == pside
                 and (symbol is None or p.symbol == symbol))


@dataclass(frozen=True)
class Step:
    pair: tuple
    fill: Fill
    before: Decimal
    after: Decimal
    clean_tail: bool


@dataclass(frozen=True)
class Boundary:
    timestamp: int
    # Exact prefix identities distinguish multiple flats in the same millisecond.
    consumed: tuple
    observation: Observation


@dataclass(frozen=True)
class BoundaryTrace:
    boundaries: tuple
    reasons: frozenset


def _steps(pair, start, now):
    direction = 1 if pair.position.pside == "long" else -1
    ordered, reasons = ordered_fills(pair.fills, start, now, direction)
    if any(f.timestamp > pair.position_at for f in ordered):
        reasons.add("post_position_fill")
    ordered = [f for f in ordered if f.timestamp <= pair.position_at]
    # Conflicting current revisions have no canonical position transition. Keep
    # their uncertainty local in time; an older damaged prefix does not taint a
    # later independently reconstructible episode.
    versions = {}
    for f in pair.fills:
        versions.setdefault(f.identity, []).append(f)
    conflicts = []
    for group in versions.values():
        newest = [f for f in group if f.revision == max(x.revision for x in group)]
        if any(f != newest[0] for f in newest):
            conflicts.extend(f.timestamp for f in newest if start <= f.timestamp <= pair.position_at)
    steps = []
    after = abs(dec(pair.position.size))
    clean = True
    for f in reversed(ordered):
        delta, valid = _quantity(f)
        before = after - delta * direction
        if not valid or before < 0 or any(t >= f.timestamp for t in conflicts):
            clean = False
        if before < 0:
            reasons.add("clamped_quantity")
        before = max(Decimal(0), before)
        steps.append(Step(pair.key, f, before, after, clean))
        after = before
    return list(reversed(steps)), reasons, conflicts


def scope_boundaries(snapshot, mode, *, pside=None, symbol=None):
    """Supported scope flats from observed positions and in-window fills.

    No exact-history gate: uncertainty may suppress a lifecycle boundary but never
    the separate current risk calculation. Absence of a detected gap is not proof
    that all exchange fills were delivered. Cross-pair simultaneous fills are a
    cohort; per-pair sequence numbers do not prove a global exchange ordering.
    """
    pairs = selected_pairs(snapshot, mode, pside=pside, symbol=symbol)
    paths, conflicts, reasons = {}, {}, set()
    for pair in pairs:
        paths[pair.key], quality, conflicts[pair.key] = _steps(pair, snapshot.start, snapshot.now)
        reasons.update(quality)
    sizes = {p.key: (paths[p.key][0].before if paths[p.key] else abs(dec(p.position.size)))
             for p in pairs}
    consumed = {p.key: [] for p in pairs}
    timeline = sorted((s for steps in paths.values() for s in steps), key=lambda s: s.fill.timestamp)
    boundaries = []
    uncertain_episode = False
    for timestamp, cohort in groupby(timeline, key=lambda s: s.fill.timestamp):
        cohort = list(cohort)
        one_pair = len({s.pair for s in cohort}) == 1
        seqs = [s.fill.sequence for s in cohort]
        exact_order = one_pair and all(s is not None for s in seqs) and len(set(seqs)) == len(seqs)
        # Unique per-pair sequence permits an exact flatten before a same-time
        # reopening. All other cohorts are applied as a whole before testing flat.
        groups = [[s] for s in sorted(cohort, key=lambda s: s.fill.sequence)] if exact_order else [cohort]
        for group in groups:
            was_exposed = any(sizes.values())
            for step in group:
                delta, valid = _quantity(step.fill)
                direction = 1 if step.pair[1] == "long" else -1
                if not valid or step.before + direction * delta != step.after:
                    uncertain_episode = True
                sizes[step.pair] = step.after
                consumed[step.pair].append(step.fill)
            had_exposure = was_exposed or any(s.before > 0 or s.after > 0 for s in group)
            if any(sizes.values()):
                continue
            # A contradictory opening followed by a partial reduction must not
            # turn that reduction into a false flatten. An inferred flat may
            # separate a later clean episode, but is not exported as a lifecycle
            # boundary and cannot release an existing halt.
            uncertain = uncertain_episode
            uncertain_episode = False
            if not had_exposure:
                continue
            if uncertain:
                reasons.add("uncertain_episode_flat")
                continue
            if any(timestamp > p.position_at for p in pairs):
                reasons.add("boundary_after_position_anchor")
                continue
            if (any(not s.clean_tail for s in group)
                    or any(t >= timestamp for ts in conflicts.values() for t in ts)
                    or any(not s.clean_tail for steps in paths.values() for s in steps
                           if s.fill.timestamp > timestamp)):
                reasons.add("uncertain_flat")
                continue
            realized = Decimal(0)
            for pair in pairs:
                flat = replace(pair.position, size=Decimal(0), basis=Decimal(0))
                history = reconstruct(flat, consumed[pair.key], {}, snapshot.start, timestamp)
                realized += history.rows[-1].pnl
                reasons.update(history.reasons)
            ids = tuple((p.key, tuple(f.identity for f in consumed[p.key])) for p in pairs)
            boundaries.append(Boundary(timestamp, ids, Observation(timestamp, realized, Decimal(0))))
    return BoundaryTrace(tuple(boundaries), frozenset(reasons))


@dataclass(frozen=True)
class PairEstimate:
    history: object
    signal: object
    reasons: frozenset


def estimate_pair(snapshot, key):
    """Snapshot experiment for a pair with a supplied historical price grid.

    Scope composition and all-candles-absent composition remain separate reference
    work. Fills beyond the position anchor are isolated, not counted against an
    older size and then counted again when positions catch up.
    """
    pair = next(p for p in snapshot.pairs if p.key == key)
    direction = 1 if pair.position.pside == "long" else -1
    all_fills, quality = ordered_fills(pair.fills, snapshot.start, snapshot.now, direction)
    fills = [f for f in all_fills if f.timestamp <= pair.position_at]
    if len(fills) != len(all_fills):
        quality.add("post_position_fill")
    if len({snapshot.balance_at, pair.position_at, pair.mark_at}) > 1:
        quality.add("snapshot_skew")
    prices = {t: p for t, p in pair.prices if snapshot.start <= t <= snapshot.now}
    if not prices:
        raise ValueError("snapshot experiment requires a historical price grid; use the minimal-history oracle separately")
    history = reconstruct(pair.position, fills, prices, snapshot.start, snapshot.now)
    if snapshot.settings.slots <= 0:
        raise ValueError("pair experiment requires an active coin slot")
    result = signal(history.rows, snapshot.balance / snapshot.settings.slots,
                    snapshot.settings.ema_span, snapshot.settings.threshold)
    return PairEstimate(history, result, frozenset(quality | set(history.reasons)))


@dataclass(frozen=True)
class Evaluation:
    snapshot: Snapshot
    value: object
    evaluations: int
    revalidated: bool
    reasons: frozenset


def evaluate_bounded(observe, compute, max_attempts=2):
    """Deterministic race model, not a production installation/execution gate.

    A stable result may be installed for its immutable source snapshot. After
    bounded churn return an explicitly unvalidated estimate of the latest snapshot,
    preserving its usable history. It is not permission to install stale state or
    size orders without independently fresh execution inputs. The reference bounds
    evaluation count, not CPU cost of the eventual Rust incremental implementation.
    """
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    snapshot = observe()
    for attempt in range(1, max_attempts + 1):
        value = compute(snapshot)
        current = observe()
        if current == snapshot:
            return Evaluation(snapshot, value, attempt, True, frozenset())
        snapshot = current
    return Evaluation(snapshot, compute(snapshot), max_attempts + 1, False,
                      frozenset({"revision_churn"}))
