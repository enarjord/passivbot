"""Independent, offline oracle for the *proposed* HSL contract.

No production imports and no exchange I/O. Decimal arithmetic makes expected
values independent of Rust/f64 implementation details. This is not a live fallback.
The functions expose intermediate estimates so tests cannot pass on a bool alone.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import groupby


MINUTE = 60_000
DAY = 1440 * MINUTE


def dec(value):
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite input")
    return result


@dataclass(frozen=True)
class Observation:
    timestamp: int
    pnl: Decimal
    upnl: Decimal


@dataclass(frozen=True)
class Signal:
    equity: tuple
    peaks: tuple
    raw: tuple
    ema: tuple
    panic: tuple


def signal(rows, budget, span, threshold, *, entry_reference=None, point_references=None, anchor=None):
    """Batch-reseeded signal; repeated samples in a minute replace its EMA input.

    Rows may include known flatten boundaries as well as minute closes. Consumers
    must inspect boundary decisions before resetting the episode. Nothing is latched
    in this function. entry_reference seeds an estimated entry peak without adding a row.
    """
    budget, span, threshold = map(dec, (budget, span, threshold))
    if budget <= 0 or span < 1 or not 0 <= threshold <= 1 or not rows:
        raise ValueError("invalid signal domain")
    if any(a.timestamp > b.timestamp for a, b in zip(rows, rows[1:])):
        raise ValueError("observations must be ordered")
    with localcontext() as ctx:
        ctx.prec = 80
        values = [dec(r.pnl) + dec(r.upnl) for r in rows]
        endpoint = values[-1] if anchor is None else dec(anchor.pnl) + dec(anchor.upnl)
        equity = [budget + (x - endpoint) for x in values]
        if anchor is None:
            equity[-1] = budget
        peak = dec(entry_reference) if entry_reference is not None else equity[0]
        alpha = Decimal(2) / (span + 1)
        raw, smooth, peaks = [], [], []
        previous_minute = None
        baseline = None
        refs = [None] * len(rows) if point_references is None else point_references
        if len(refs) != len(rows):
            raise ValueError("invalid reference count")
        for row, value, reference in zip(rows, equity, refs):
            peak = max(peak, value)
            if reference is not None:
                peak = max(peak, budget + dec(reference))
            # An entirely nonpositive historical segment cannot define a ratio.
            # Represent complete impairment until a positive peak exists. This
            # explicit reference choice affects old EMA samples, not current B.
            drawdown = (peak - value) / peak if peak > 0 else Decimal(1)
            minute = row.timestamp // MINUTE
            if minute != previous_minute:
                baseline = smooth[-1] if smooth else None
            ema = drawdown if baseline is None else alpha * drawdown + (1 - alpha) * baseline
            peaks.append(peak)
            raw.append(drawdown)
            smooth.append(ema)
            previous_minute = minute
        return Signal(tuple(equity), tuple(peaks), tuple(raw), tuple(smooth),
                      tuple(min(d, e) > threshold for d, e in zip(raw, smooth)))


def minimal_signal(upnl, budget, span, threshold, timestamp=0):
    """A synthetic entry reference is not an earlier zero-drawdown EMA sample."""
    budget, upnl = dec(budget), dec(upnl)
    return signal([Observation(timestamp, Decimal(0), upnl)], budget, span, threshold,
                  entry_reference=max(budget, budget - upnl))


def aggregate(series):
    """Currency aggregation on an explicitly aligned timeline, never DD averaging."""
    if not series or not series[0]:
        raise ValueError("empty aggregate")
    times = [r.timestamp for r in series[0]]
    if any([r.timestamp for r in rows] != times for rows in series):
        raise ValueError("unaligned aggregate")
    return tuple(Observation(t, sum((dec(s[i].pnl) for s in series), Decimal(0)),
                             sum((dec(s[i].upnl) for s in series), Decimal(0)))
                 for i, t in enumerate(times))


def scope_budget(mode, balance, slots=None):
    balance = dec(balance)
    if balance <= 0:
        raise ValueError("balance must be positive")
    if mode in ("pside", "unified"):
        return balance
    if mode != "coin" or slots is None or int(slots) != slots or slots < 0:
        raise ValueError("invalid scope")
    # Inactive coin side: no replacement divisor and no numeric HSL evaluation.
    return balance / slots if slots else None


@dataclass(frozen=True)
class Position:
    size: object
    basis: object
    mark: object
    multiplier: object = 1
    inverse: bool = False
    pside: str = "long"

    def validate(self):
        size, basis, mark, multiplier = map(dec, (self.size, self.basis, self.mark, self.multiplier))
        if mark <= 0 or multiplier <= 0 or (size and basis <= 0):
            raise ValueError("invalid current position")
        if self.pside not in ("long", "short") or (size > 0 and self.pside != "long") or (size < 0 and self.pside != "short"):
            raise ValueError("position side disagrees with signed size")
        return size, basis, mark, multiplier


def pnl(position, size, basis, price):
    size, basis, price = map(dec, (size, basis, price))
    if not size:
        return Decimal(0)
    if basis <= 0 or price <= 0:
        raise ValueError("invalid price")
    multiplier = dec(position.multiplier)
    if position.inverse:
        return size * multiplier * (1 / basis - 1 / price)
    return size * multiplier * (price - basis)


@dataclass(frozen=True)
class Fill:
    identity: str
    timestamp: int
    delta: object  # signed buy/sell quantity; short increases are negative
    price: object
    realized: object = None  # gross realized PnL, None means unknown
    fee: object = 0  # signed balance impact, normally negative
    sequence: int | None = None
    revision: int = 0


@dataclass(frozen=True)
class History:
    rows: tuple
    sizes: tuple
    bases: tuple
    reasons: frozenset
    # Canonical fill and cumulative net realized cashflow after that exact fill.
    # Kept independently of prices so candle absence cannot erase known losses.
    cashflows: tuple = ()


def ordered_fills(fills, start, end, direction):
    reasons = set()
    by_id = {}
    for fill in fills:
        by_id.setdefault(fill.identity, []).append(fill)
    selected = []
    for versions in by_id.values():
        revision = max(f.revision for f in versions)
        latest = [f for f in versions if f.revision == revision]
        if any(f != latest[0] for f in latest):
            reasons.add("conflicting_identity")
            continue
        f = latest[0]
        if not start <= f.timestamp <= end:
            continue
        try:
            if not dec(f.delta):
                raise ValueError("zero delta")
        except (ValueError, ArithmeticError):
            reasons.add("invalid_quantity")
            # Unknown position transition does not erase independent cashflow.
            # Zero is an estimator-local quantity omission, not a ledger repair.
        selected.append(f)
    ordered = []
    for _, cohort in groupby(sorted(selected, key=lambda f: f.timestamp), key=lambda f: f.timestamp):
        cohort = list(cohort)
        sequences = [f.sequence for f in cohort]
        if len(cohort) > 1 and (any(s is None for s in sequences) or len(set(sequences)) != len(cohort)):
            reasons.add("estimated_fill_order")
        # Retain known relative order even if other rows lack sequences. Place
        # unknown rows after that ordered subset, increases before reductions.
        # Equal/absent sequences use identity as final deterministic tie-breaker.
        cohort.sort(key=lambda f: (f.sequence is None, f.sequence if f.sequence is not None else 0,
                                   _estimated_delta(f) * direction < 0, f.identity))
        ordered.extend(cohort)
    return ordered, reasons


def reconstruct(position, fills, prices, start, end):
    """One pair, coherent current snapshot; prices are minute-end estimates.

    No I/O/retry policy is modeled. Malformed historical price/fee/PnL fields
    degrade individual estimates; invalid current inputs are errors. This layer
    deliberately does not decide whether an inferred flat is lifecycle evidence.
    """
    size, current_basis, mark, _ = position.validate()
    if start > end or any(not start <= t <= end or dec(p) <= 0 for t, p in prices.items()):
        raise ValueError("invalid price grid")
    direction = Decimal(1) if position.pside == "long" else Decimal(-1)
    ordered, reasons = ordered_fills(fills, start, end, direction)
    after = abs(size)
    steps = []
    for f in reversed(ordered):
        delta = _estimated_delta(f) * direction
        before = after - delta
        if before < 0:
            reasons.add("clamped_quantity")
            before = Decimal(0)
        steps.append((f, before, after, delta))
        after = before
    steps.reverse()
    basis = next((dec(f.price) for f in ordered if _positive(f.price)),
                 current_basis if current_basis > 0 else mark)
    if after:
        reasons.add("estimated_opening_basis")
    quantity = after
    if not quantity:
        basis = Decimal(0)
    states = [(start - 1, quantity, basis, Decimal(0))]
    cumulative = Decimal(0)
    cashflows = []
    for f, before, after, delta in steps:
        price = dec(f.price) if _positive(f.price) else (
            basis if basis > 0 else current_basis if current_basis > 0 else mark)
        if not _positive(f.price):
            reasons.add("estimated_fill_price")
        if delta > 0:
            if position.inverse and before and basis:
                basis = (before + delta) / (before / basis + delta / price)
            else:
                basis = (before * basis + delta * price) / (before + delta)
        gross = dec(f.realized) if _usable(f.realized) else (
            pnl(position, direction * min(before, -delta), basis, price) if delta < 0 else Decimal(0))
        if not _usable(f.realized):
            reasons.add("estimated_realized_pnl")
        fee = dec(f.fee) if _usable(f.fee) else Decimal(0)
        if not _usable(f.fee):
            reasons.add("unknown_fee")
        cumulative += gross + fee
        cashflows.append((f, cumulative))
        if not after:
            basis = Decimal(0)
        states.append((f.timestamp, after, basis, cumulative))
    if size and basis != current_basis:
        reasons.add("current_basis_reconciliation")
    rows, sizes, bases = [], [], []
    grid = dict(prices)
    grid[end] = mark
    for t, price in sorted(grid.items()):
        _, q, b, realized = next(s for s in reversed(states) if s[0] <= t)
        if t == end:
            q, b = abs(size), current_basis
        rows.append(Observation(t, realized, pnl(position, direction * q, b, price)))
        sizes.append(direction * q)
        bases.append(b)
    return History(tuple(rows), tuple(sizes), tuple(bases), frozenset(reasons), tuple(cashflows))


def _usable(value):
    try:
        dec(value)
        return True
    except (ValueError, ArithmeticError):
        return False


def _positive(value):
    return _usable(value) and dec(value) > 0


def _estimated_delta(fill):
    return dec(fill.delta) if _usable(fill.delta) else Decimal(0)


@dataclass(frozen=True)
class Candle:
    start: int
    minutes: int
    open: object
    high: object
    low: object
    close: object
    available_at: int | None = None


def minute_prices(candles, start, end):
    """Close-only grid with coarse zigzag, then in-window ffill/leading bfill.

    The entire coarse source must be within the window and available by end;
    a real minute contributes only its close, whose timestamp must be in-window.
    Empty output explicitly requests the minimal-history branch; no fake candle.
    """
    selected = {}
    for c in candles:
        finish = c.start + c.minutes * MINUTE
        if c.minutes not in (1, 5, 15, 60) or finish > end:
            continue
        if (c.minutes == 1 and finish < start) or (c.minutes != 1 and c.start < start):
            continue
        if c.available_at is not None and c.available_at > end:
            continue
        if c.minutes == 1:
            if not _positive(c.close):
                continue
            path = [dec(c.close)]
        else:
            if not all(_positive(v) for v in (c.open, c.high, c.low, c.close)):
                continue
            o, h, l, close = map(dec, (c.open, c.high, c.low, c.close))
            if not l <= min(o, close) <= max(o, close) <= h:
                continue
            a, b, last = c.minutes // 3, 2 * c.minutes // 3, c.minutes - 1
            waypoints = [(0, o), (a, l if close >= o else h),
                         (b, h if close >= o else l), (last, close)]
            path = []
            for i in range(c.minutes):
                left, right = next((x, y) for x, y in zip(waypoints, waypoints[1:])
                                   if x[0] <= i <= y[0])
                path.append(left[1] + (right[1] - left[1]) * dec(i - left[0]) / (right[0] - left[0]))
        for i, price in enumerate(path):
            t = c.start + (i + 1) * MINUTE
            if t not in selected or c.minutes < selected[t][0]:
                selected[t] = (c.minutes, price)
            elif c.minutes == selected[t][0] and price != selected[t][1]:
                raise ValueError("canonicalize conflicting candles before sampling")
    if not selected:
        return {}
    first = selected[min(selected)][1]
    result, previous = {}, first
    for t in range(((start + MINUTE - 1) // MINUTE) * MINUTE, end + 1, MINUTE):
        if t in selected:
            previous = selected[t][1]
        result[t] = previous
    return result


@dataclass(frozen=True)
class LifecycleEvidence:
    """Exchange-reconstructible times, not persisted decisions.

    Tests construct these from explicit synthetic fill-boundary traces. None
    means that the available tape does not establish the event. Historical
    classification from ambiguous fills remains a later integration obligation.
    """
    red_at: int | None = None
    flat_at: int | None = None


def permission(now, lookback, cooldown, restart, intervention, evidence, *, exposed, red_now):
    """Lifecycle algebra, recomputed each call with no saved-state argument."""
    if restart not in ("always", "never") or intervention not in ("panic", "normal"):
        raise ValueError("removed policy")
    if lookback <= 0 or cooldown < 0:
        raise ValueError("invalid duration")
    start = now - lookback
    red_at, flat_at = evidence.red_at, evidence.flat_at
    red_at = red_at if red_at is not None and start <= red_at <= now else None
    flat_at = flat_at if flat_at is not None and start <= flat_at <= now else None
    if red_now and exposed:
        return "panic"
    if red_at is None and flat_at is None:
        return "normal"
    if flat_at is None or (red_at is not None and flat_at < red_at):
        return "panic" if exposed else "halted"
    # Cooldown is anchored at the stop's actual flatten, while `never` is
    # anchored at its imposing RED event. These may expire at different times.
    halted = red_at is not None if restart == "never" else now < flat_at + cooldown
    if not halted:
        return "normal"
    # A proven flat followed by fresh current exposure establishes reappearance
    # without needing its opening fill. An unfinished panic never reaches here.
    if exposed:
        return "panic" if intervention == "panic" else "normal"
    return "halted"
