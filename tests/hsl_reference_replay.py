"""Test-only scope-boundary and snapshot experiments for the revised HSL.

These compose the independent oracle; they are not installed in the live loop.
No cache/journal state or prior decisions are accepted as reconstruction inputs.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import groupby

from hsl_reference import (
    Candle, Fill, MINUTE, Observation, Position, dec, minute_prices, ordered_fills,
    reconstruct, signal,
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
    fills_started_at: int | None
    fills_at: int | None
    prices_at: int
    fills: tuple
    prices: tuple
    # position, mark, fills, prices producer revisions for this pair only.
    revisions: tuple
    # Optional causal acquisition proof, bound to the exact observed position.
    fills_position_anchor: tuple | None

    @property
    def key(self):
        return self.symbol, self.position.pside


def capture_pair(symbol, position, position_at, mark_at, fills, prices, *,
                 fills_started_at=None, fills_at=None, prices_at, revisions=(0,) * 4,
                 fills_after_position=False):
    """Capture normalized inputs, with fetch-completion times separate from events.

    Fill.sequence is actual per-pair exchange/simulator ordering supplied by the
    fixture, not a numeric trade ID or local arrival index. Unknown order is None.
    This proposed-contract oracle does not parse/validate connector provenance;
    production normalization must establish it before supplying such a sequence.
    fills_after_position is explicit causal acquisition evidence: this exact
    position was observed before initiating the tail request. It is never inferred
    from equal millisecond timestamps, and is invalidated by position changes.
    """
    size, basis, mark, multiplier = position.validate()
    copied_fills = tuple(replace(f, delta=_number(f.delta), price=_number(f.price),
                                realized=_number(f.realized), fee=_number(f.fee)) for f in fills)
    captured = Pair(symbol, replace(position, size=size, basis=basis, mark=mark,
                                multiplier=multiplier), position_at, mark_at,
                fills_started_at, fills_at,
                prices_at,
                copied_fills, tuple(sorted((t, dec(p)) for t, p in prices.items())), tuple(revisions), None)
    # Explicit fixture/acquisition evidence, never inferred from equal clocks.
    return replace(captured, fills_position_anchor=position_anchor(captured)) if fills_after_position else captured


def position_anchor(pair):
    p = pair.position
    return (pair.position_at, p.size, p.basis, p.multiplier, p.inverse, p.pside, pair.revisions[0])


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
    config_at: int
    pairs: tuple
    settings: Settings
    # balance, positions, marks, fills, prices, config; monotonic producer versions.
    revisions: tuple


def capture(now, start, balance, balance_at, pairs, settings=Settings(), revisions=(0,) * 6,
            max_current_age=120_000, config_at=None):
    pairs = tuple(sorted(pairs, key=lambda p: p.key))
    balance = dec(balance)
    if start > now or balance <= 0 or max_current_age < 0:
        raise ValueError("invalid minimum snapshot")
    if len({p.key for p in pairs}) != len(pairs) or len(revisions) != 6:
        raise ValueError("invalid snapshot shape")
    observed = [balance_at, *(p.position_at for p in pairs), *(p.mark_at for p in pairs)]
    if any(not now - max_current_age <= t <= now for t in observed):
        raise ValueError("unusable current observation")
    config_at = now if config_at is None else config_at
    if any(t is not None and t > now for t in [config_at, *(p.fills_at for p in pairs),
                             *(p.fills_started_at for p in pairs),
                             *(p.prices_at for p in pairs)]):
        raise ValueError("future source capture")
    if any(p.fills_started_at is not None and p.fills_at is not None
           and p.fills_started_at > p.fills_at for p in pairs):
        raise ValueError("reversed fill request interval")
    if any(not isinstance(r, int) or r < 0 for r in revisions):
        raise ValueError("invalid producer revision")
    for p in pairs:
        p.position.validate()
        if len(p.revisions) != 4 or any(not isinstance(r, int) or r < 0 for r in p.revisions):
            raise ValueError("invalid pair revisions")
    settings = replace(settings, ema_span=dec(settings.ema_span), threshold=dec(settings.threshold))
    return Snapshot(now, start, balance, balance_at, config_at, pairs, settings, tuple(revisions))


def snapshot_quality(snapshot, keys=None):
    pairs = snapshot.pairs if keys is None else tuple(p for p in snapshot.pairs if p.key in keys)
    reasons = set()
    if any(p.fills_started_at is None or p.fills_at is None for p in pairs):
        reasons.add("fill_capture_unknown")
    if any(p.fills_started_at is not None and (p.fills_started_at < p.position_at
           or (p.fills_started_at == p.position_at and p.fills_position_anchor != position_anchor(p)))
           for p in pairs):
        reasons.add("fills_before_position")
    if any(p.prices_at < p.mark_at for p in pairs):
        reasons.add("prices_before_mark")
    if any(t > p.prices_at for p in pairs for t, _ in p.prices):
        reasons.add("post_capture_price")
    if any(p.position_at < f.timestamp <= snapshot.now for p in pairs
           for f in latest_variants(p.fills)):
        reasons.add("post_position_fill")
    if any(f.timestamp == p.position_at for p in pairs for f in latest_variants(p.fills)):
        reasons.add("position_fill_timestamp_tie")
    if any(f.timestamp > min(snapshot.now, p.fills_at if p.fills_at is not None else snapshot.now)
           for p in pairs for f in latest_variants(p.fills)):
        reasons.add("post_capture_fill")
    return reasons


def latest_variants(fills):
    versions = {}
    for fill in fills:
        versions.setdefault(fill.identity, []).append(fill)
    return tuple(f for group in versions.values() for f in group
                 if f.revision == max(x.revision for x in group))


def project(snapshot, keys):
    if keys is None:
        return snapshot
    keys = frozenset(keys)
    if not keys.issubset({p.key for p in snapshot.pairs}):
        raise ValueError("missing current scope positions; absent is not flat")
    # Balance/config revisions are account-wide. Position/mark/history revisions
    # are per pair for scoped evaluation; aggregate tokens would reintroduce churn
    # from unrelated pairs even when all selected content stayed unchanged.
    revisions = (snapshot.revisions[0], 0, 0, 0, 0, snapshot.revisions[5])
    return replace(snapshot, pairs=tuple(p for p in snapshot.pairs if p.key in keys),
                   revisions=revisions)


def source_revisions(snapshot):
    result = {("global", i): r for i, r in enumerate(snapshot.revisions)}
    result.update({(p.key, i): r for p in snapshot.pairs for i, r in enumerate(p.revisions)})
    for p in snapshot.pairs:
        for f in latest_variants(p.fills):
            result[("fill", p.key, f.identity)] = f.revision
    return result


def source_times(snapshot):
    result = {"balance": snapshot.balance_at, "config": snapshot.config_at}
    for p in snapshot.pairs:
        for name in ("position_at", "mark_at", "fills_started_at", "fills_at", "prices_at"):
            result[(p.key, name)] = getattr(p, name)
    return result


def fill_identity_times(snapshot):
    result = {}
    for p in snapshot.pairs:
        end = min(snapshot.now, p.fills_at if p.fills_at is not None else snapshot.now)
        for f in latest_variants(p.fills):
            if f.timestamp > end:
                continue
            times = result.setdefault((p.key, f.identity), set())
            if snapshot.start <= f.timestamp <= end:
                times.add(f.timestamp)
    return result


def selected_pairs(snapshot, mode, *, pside=None, symbol=None):
    if mode == "unified":
        if pside is not None or symbol is not None:
            raise ValueError("unified has no side/symbol selector")
        return snapshot.pairs
    if mode not in ("pside", "coin") or pside not in ("long", "short"):
        raise ValueError("invalid scope")
    if (mode == "coin") != (symbol is not None):
        raise ValueError("invalid symbol selector")
    pairs = tuple(p for p in snapshot.pairs if p.position.pside == pside
                  and (symbol is None or p.symbol == symbol))
    if mode == "coin" and not pairs:
        raise ValueError("missing current coin position; absent is not flat")
    return pairs


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
    # Full canonical prefix distinguishes both tied flats and corrected evidence.
    consumed: tuple
    observation: Observation
    lifecycle_eligible: bool = True


@dataclass(frozen=True)
class BoundaryTrace:
    boundaries: tuple
    reasons: frozenset


def _steps(pair, start, now):
    direction = 1 if pair.position.pside == "long" else -1
    ordered, reasons = ordered_fills(pair.fills, start, now, direction)
    if any(f.timestamp > pair.position_at for f in ordered):
        reasons.add("post_position_fill")
    end = min(pair.position_at, pair.fills_at if pair.fills_at is not None else now)
    ordered = [f for f in ordered if f.timestamp <= end]
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
            if any(pair.position_at < f.timestamp <= now for f in newest):
                reasons.add("post_position_fill")
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
    reasons.update(snapshot_quality(snapshot, {p.key for p in pairs}))
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
            fill_quality = snapshot_quality(snapshot, {p.key for p in pairs}) & {
                "fills_before_position", "fill_capture_unknown"
            }
            if fill_quality:
                reasons.update(fill_quality)
                continue
            if "position_fill_timestamp_tie" in snapshot_quality(snapshot, {p.key for p in pairs}):
                reasons.add("position_fill_timestamp_tie")
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
            # A mixed unordered cohort supplies an estimated risk row, not proof
            # of a lifecycle reset. sequence means actual exchange/simulator order,
            # never an arbitrary ID or Python list index (see capture contract).
            ambiguous = not exact_order and any(
                len({(_quantity(s.fill)[0] > 0) for s in group if s.pair == p.key}) > 1
                for p in pairs
            )
            if ambiguous:
                reasons.add("estimated_flat")
            tape = tuple((p.key, tuple(consumed[p.key])) for p in pairs)
            boundaries.append(Boundary(timestamp, tape, Observation(timestamp, realized, Decimal(0)),
                                       not ambiguous and not reasons.intersection({
                                           "post_position_fill", "position_fill_timestamp_tie", "post_capture_fill"
                                       })))
    return BoundaryTrace(tuple(boundaries), frozenset(reasons))


@dataclass(frozen=True)
class PairEstimate:
    history: object
    signal: object
    reasons: frozenset


def estimate_pair(snapshot, key):
    """Snapshot experiment for a pair with a supplied historical price grid.

    Sparse minute closes are normalized with ffill/bfill. Scope composition and
    all-candles-absent composition remain separate reference work. Fills beyond the
    position anchor are isolated, not counted against an older size and then counted
    again when positions catch up.
    """
    pair = next(p for p in snapshot.pairs if p.key == key)
    direction = 1 if pair.position.pside == "long" else -1
    all_fills, quality = ordered_fills(pair.fills, snapshot.start, snapshot.now, direction)
    quality.update(snapshot_quality(snapshot, {key}))
    end = min(pair.position_at, pair.fills_at if pair.fills_at is not None else snapshot.now)
    fills = [f for f in all_fills if f.timestamp <= end]
    if any(f.timestamp > pair.position_at for f in all_fills):
        quality.add("post_position_fill")
    if len({snapshot.balance_at, pair.position_at, pair.mark_at}) > 1:
        quality.add("snapshot_skew")
    prices = {t: p for t, p in pair.prices if snapshot.start <= t <= min(snapshot.now, pair.prices_at)}
    if not prices:
        raise ValueError("snapshot experiment requires a historical price grid; use the minimal-history oracle separately")
    expected = set(range(((snapshot.start + MINUTE - 1) // MINUTE) * MINUTE,
                         snapshot.now + 1, MINUTE))
    if not expected.issubset(prices):
        quality.add("filled_price_grid")
    # Apply the agreed ffill/bfill convention instead of compressing EMA time or
    # turning a historical gap into a readiness veto. Inputs are minute closes.
    prices = minute_prices([Candle(t - MINUTE, 1, p, p, p, p) for t, p in prices.items()],
                           snapshot.start, snapshot.now)
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


def evaluate_bounded(observe, compute, max_attempts=2, *, scope_keys=None):
    """Deterministic race model, not a production installation/execution gate.

    A stable result with nonregressing revisions and compatible source captures
    may be installed for its immutable source snapshot. This does not certify
    complete exchange history or atomic account observations. After
    bounded churn return an explicitly unvalidated estimate of the latest snapshot,
    preserving its usable history. It is not permission to install stale state or
    size orders without independently fresh execution inputs. The reference bounds
    evaluation count, not CPU cost of the eventual Rust incremental implementation.
    scope_keys projects coin/pside calculation, comparison and source quality;
    None means the unified scope. Only global balance/config remain shared.
    """
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    scope_keys = None if scope_keys is None else frozenset(scope_keys)
    snapshot = project(observe(), scope_keys)
    high_water = source_revisions(snapshot)
    time_water = source_times(snapshot)
    fill_water = fill_identity_times(snapshot)
    last_time = snapshot.now
    regression = False
    capture_regression = False
    missing_identity = False
    for attempt in range(1, max_attempts + 1):
        value = compute(snapshot)
        current = project(observe(), scope_keys)
        revisions = source_revisions(current)
        regression |= current.now < last_time or any(r < high_water.get(k, r) for k, r in revisions.items())
        high_water.update({k: max(r, high_water.get(k, r)) for k, r in revisions.items()})
        for k, t in source_times(current).items():
            prior = time_water.get(k)
            if prior is not None and (t is None or t < prior):
                capture_regression = True
            if t is not None:
                time_water[k] = t if prior is None else max(prior, t)
        current_fills = fill_identity_times(current)
        missing_identity |= any(k not in current_fills and any(current.start <= t <= current.now for t in ts)
                                for k, ts in fill_water.items())
        fill_water.update(current_fills)
        last_time = max(last_time, current.now)
        quality = snapshot_quality(current, scope_keys)
        if regression:
            quality.add("revision_regression")
        if capture_regression:
            quality.add("source_capture_regression")
        if missing_identity:
            quality.add("missing_fill_identity")
        if current == snapshot:
            return Evaluation(snapshot, value, attempt, not quality, frozenset(quality))
        snapshot = current
    return Evaluation(snapshot, compute(snapshot), max_attempts + 1, False,
                      frozenset(quality | {"revision_churn"}))
