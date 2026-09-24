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
        endpoint = dec(rows[-1].pnl) if anchor is None else dec(anchor.pnl)
        equity = [budget + (x - endpoint) for x in values]
        if anchor is None:
            equity[-1] = budget + dec(rows[-1].upnl)
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
                  entry_reference=budget)


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


def quantity_path(ordered, current, direction):
    """Explain each reduction locally; current position does not alter the prefix."""
    runs, subtotal = {}, Decimal(0)
    for i in reversed(range(len(ordered))):
        delta = _estimated_delta(ordered[i]) * direction
        subtotal = Decimal(0) if delta > 0 else subtotal - delta
        runs[i] = subtotal
    opening = runs.get(0, Decimal(0))
    if ordered and all(_estimated_delta(f) * direction <= 0 for f in ordered):
        opening += abs(current)
    quantity, steps = opening, []
    for i, fill in enumerate(ordered):
        delta = _estimated_delta(fill) * direction
        before = max(quantity, runs[i])
        steps.append((fill, before, before + delta, delta))
        quantity = before + delta
    return steps, opening


def reconstruct(position, fills, prices, start, end):
    """One pair, coherent current snapshot; prices are minute-end estimates.

    No I/O/retry policy is modeled. Malformed historical price/fee/PnL fields
    degrade individual estimates; invalid current inputs are errors. This layer
    supplies the same estimated position path used by lifecycle reconstruction.
    """
    size, current_basis, mark, _ = position.validate()
    if start > end or any(not start <= t <= end or dec(p) <= 0 for t, p in prices.items()):
        raise ValueError("invalid price grid")
    direction = Decimal(1) if position.pside == "long" else Decimal(-1)
    ordered, reasons = ordered_fills(fills, start, end, direction)
    steps, quantity = quantity_path(ordered, abs(size), direction)
    if quantity:
        reasons.add("estimated_opening_quantity")
    basis = next((dec(f.price) for f in ordered if _positive(f.price)),
                 current_basis if current_basis > 0 else mark)
    if quantity:
        reasons.add("estimated_opening_basis")
    if not quantity:
        basis = Decimal(0)
    states = [(start - 1, quantity, basis, Decimal(0))]
    cumulative = Decimal(0)
    cashflows = []
    prior = quantity
    for f, before, after, delta in steps:
        if before != prior:
            reasons.add("local_quantity_reconciliation")
        prior = after
        price = dec(f.price) if _positive(f.price) else (
            basis if basis > 0 else current_basis if current_basis > 0 else mark)
        if not _positive(f.price):
            reasons.add("estimated_fill_price")
        if before and not basis:
            basis = price
            reasons.add("local_basis_reconciliation")
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
    tail = steps[-1][2] if steps else quantity
    if not tail and size:
        reasons.add("estimated_current_opening")
    if tail != abs(size):
        reasons.add("current_quantity_reconciliation")
        if not size and ordered:
            reasons.add("current_flat_timestamp_estimate")
    rows, sizes, bases = [], [], []
    grid = dict(prices)
    grid[end] = mark
    for t, price in sorted(grid.items()):
        _, q, b, realized = next(s for s in reversed(states) if s[0] <= t)
        if t == end or (not size and tail != 0 and ordered and t >= ordered[-1].timestamp):
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
    """Latest episode's reconstructed terminal signal and flatten time."""
    terminal_red: bool = False
    flat_at: int | None = None


def permission(now, lookback, cooldown, restart, evidence, *, exposed, red_now):
    """Latest-episode lifecycle algebra with no saved decision argument."""
    if restart not in ("always", "never"):
        raise ValueError("removed policy")
    if lookback <= 0 or cooldown < 0:
        raise ValueError("invalid duration")
    if exposed:
        return "panic" if red_now else "normal"
    flat = evidence.flat_at
    if not evidence.terminal_red or flat is None or not now-lookback <= flat <= now:
        return "normal"
    return "halted" if restart == "never" or now < flat + cooldown else "normal"
