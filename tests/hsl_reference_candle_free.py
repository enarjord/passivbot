"""Independent candle-free scope oracle; no production callers.

This is the minimal-history construction enriched by available realized cashflows.
It intentionally has one actual EMA observation. Cashflow peaks are references,
not fabricated past equity samples or proof of lifecycle boundaries.
"""

from dataclasses import dataclass
from decimal import Decimal
from itertools import groupby

from hsl_reference import Observation, dec, ordered_fills, pnl, reconstruct, scope_budget, signal
from hsl_reference_replay import causal_fills, selected_pairs, snapshot_quality


@dataclass(frozen=True)
class CandleFreeEstimate:
    signal: object  # None only for an explicitly inactive zero-slot coin scope.
    realized: Decimal
    realized_peak: Decimal
    upnl: Decimal
    reasons: frozenset


def estimate_candle_free(snapshot, mode, *, pside=None, symbol=None):
    """All selected pairs lack usable candles; not a mixed-history dispatcher.

    Normalize/correct before clipping and isolate post-position events just as
    the snapshot reference does. Missing prices affect historical UPNL knowledge,
    not independently usable gross PnL/fees. Estimate missing cashflow components
    with the existing reconstruction rules; never add known losses twice.

    A single pair's actual sequence may reveal an intratimestamp realized peak.
    Cross-pair timestamps are indivisible cohorts: pair sequence numbers cannot
    manufacture global peaks. No historical UPNL or EMA delay is invented.
    Episode reset/controller selection is the caller's separate responsibility.
    """
    pairs = selected_pairs(snapshot, mode, pside=pside, symbol=symbol)
    budget = scope_budget(mode, snapshot.balance, snapshot.settings.slots)
    if budget is None:
        return CandleFreeEstimate(None, dec(0), dec(0), dec(0), frozenset({"inactive_scope"}))
    quality = set(snapshot_quality(snapshot, {p.key for p in pairs}))
    if len({snapshot.balance_at, *(p.position_at for p in pairs),
            *(p.mark_at for p in pairs)}) > 1:
        quality.add("snapshot_skew")
    quality.add("candle_free_reference")
    timelines = []
    upnl = Decimal(0)
    for pair in pairs:
        if any(snapshot.start <= t <= min(snapshot.now, pair.prices_at) for t, _ in pair.prices):
            raise ValueError("candle-free oracle cannot discard supplied historical prices")
        direction = 1 if pair.position.pside == "long" else -1
        fills, reasons = ordered_fills(causal_fills(pair, snapshot.now),
                                      snapshot.start, snapshot.now, direction)
        quality.update(reasons)
        limit = min(pair.position_at, pair.fills_at if pair.fills_at is not None else snapshot.now)
        fills = [f for f in fills if f.timestamp <= limit]
        history = reconstruct(pair.position, fills, {}, snapshot.start, snapshot.now)
        quality.update(history.reasons)
        previous = Decimal(0)
        for fill, cumulative in history.cashflows:
            timelines.append((pair.key, fill, cumulative - previous))
            previous = cumulative
        p = pair.position
        upnl += pnl(p, p.size, p.basis, p.mark)
    realized = peak = Decimal(0)
    for _, cohort in groupby(sorted(timelines, key=lambda row: row[1].timestamp),
                             key=lambda row: row[1].timestamp):
        cohort = list(cohort)
        sequences = [f.sequence for _, f, _ in cohort]
        exact = (len({key for key, _, _ in cohort}) == 1
                 and all(seq is not None for seq in sequences)
                 and len(set(sequences)) == len(sequences))
        if exact:
            groups = [[row] for row in sorted(cohort, key=lambda row: row[1].sequence)]
        else:
            groups = [cohort]
            if len(cohort) > 1:
                quality.add("cohort_cashflow_peak")
        for group in groups:
            realized += sum((delta for _, _, delta in group), Decimal(0))
            peak = max(peak, realized)
    # Preserve known cashflow peaks and current losses, but do not invent past
    # UPNL observations. Gains offset losses in currency before any division.
    endpoint = realized
    reference = budget + peak - endpoint
    result = signal([Observation(snapshot.now, realized, upnl)], budget,
                    snapshot.settings.ema_span, snapshot.settings.threshold,
                    entry_reference=reference)
    return CandleFreeEstimate(result, realized, peak, upnl, frozenset(quality))
