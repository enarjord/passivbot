"""Immutable factual history transport for revised HSL (no trading decisions).

Consume the manager's *current canonical batch*, not a concatenation of snapshots.
Rust owns clipping, ordering, reconstruction and estimation. These copies never
use the manager's derived psize/pprice or infer execution sequence from trade IDs.
"""

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

from fill_events_manager import (
    FillEvent,
    FEE_QUALITY_EXACT,
    FEE_QUALITY_CONVERTED,
    PNL_CONTRACT_CURRENT,
    PNL_SOURCE_AUTHORITATIVE,
    PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED,
    PNL_SOURCE_PENDING,
)


def _number(value):
    """Missing or damaged historical components stay explicitly missing."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _identified(value):
    return isinstance(value, str) and value.strip().lower() not in ("", "none", "null", "nan")


def _timestamp(value):
    # Do not truncate fractional timestamps or invent the time of an undated fill.
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
        if result != value or not 0 <= result <= 2**63 - 1:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return result


@dataclass(frozen=True)
class Fill:
    identity: str
    timestamp: int
    delta: float | None
    price: float | None
    realized: float | None
    fee: float | None
    # The manager has already resolved revisions. No cross-snapshot merging and
    # no inference from local arrival order or numeric exchange IDs is allowed.
    sequence: None = None
    revision: int = 0


@dataclass(frozen=True)
class PairFills:
    symbol: str
    pside: str
    fills: tuple[Fill, ...]
    reasons: tuple[str, ...]

    def payload(self):
        return [asdict(fill) for fill in self.fills]


@dataclass(frozen=True)
class FillTape:
    pairs: tuple[PairFills, ...]
    # Rows without usable attribution/time are excluded and disclosed globally.
    reasons: tuple[str, ...]


def capture_fills(events: Iterable[FillEvent], multipliers: Mapping[str, float]) -> FillTape:
    """Copy canonical manager fills in native contract quantities.

    This consumes the manager's normalized contract, including its supported
    optional-field defaults (complete supplied PnL and native unit multiplier 1).
    It does not demand raw-field-presence certificates from normalized objects.
    ``multipliers`` is current factual market metadata. A missing/changed normalized
    fill multiplier makes that fill's quantity unknown; independently usable cashflows
    survive. Unknown accounting contracts cannot supply gross/fee amounts. Pending
    PnL is absent, including a producer's placeholder zero. Estimated PnL under the
    current contract remains usable and disclosed. No historical defect blocks
    other fields or pairs, and no factual ledger is changed.
    """
    grouped, quality, global_reasons = {}, {}, set()
    for event in events:
        symbol, pside = event.symbol, event.position_side
        if not _identified(symbol) or pside not in ("long", "short"):
            global_reasons.add("unattributed_fill")
            continue
        key = symbol, pside
        reasons = quality.setdefault(key, set())
        rows = grouped.setdefault(key, [])
        timestamp = _timestamp(event.timestamp)
        if not _identified(event.id) or timestamp is None or timestamp == 0:
            reasons.add("unidentified_or_undated_fill")
            continue
        quantity = _number(event.qty)
        if (event.side not in ("buy", "sell") or quantity is None
                or (event.side == "buy" and quantity < 0)
                or (event.side == "sell" and quantity > 0)):
            quantity = None
            reasons.add("invalid_fill_quantity")
        fill_multiplier = _number(event.c_mult)
        current_multiplier = _number(multipliers.get(symbol))
        if (fill_multiplier is None or fill_multiplier <= 0
                or current_multiplier is None or current_multiplier <= 0
                or not math.isclose(fill_multiplier, current_multiplier, rel_tol=1e-12, abs_tol=0.0)):
            quantity = None
            reasons.add("fill_contract_units_unavailable")
        price = _number(event.price)
        if price is None or price <= 0:
            price = None
            reasons.add("invalid_fill_price")
        realized, fee = _number(event.pnl), _number(event.fee_paid)
        if event.pnl_contract != PNL_CONTRACT_CURRENT:
            realized = fee = None
            reasons.add("fill_accounting_contract_unavailable")
        else:
            if event.pnl_status == "pending" or event.pnl_source == PNL_SOURCE_PENDING:
                realized = None
                reasons.add("pending_realized_pnl")
            elif event.pnl_status != "complete":
                realized = None
                reasons.add("unknown_pnl_completeness")
            elif event.pnl_source not in (
                PNL_SOURCE_AUTHORITATIVE, PNL_SOURCE_AUTHORITATIVE_CYCLE_RECONCILED
            ):
                reasons.add("estimated_realized_pnl")
            if realized is None:
                reasons.add("missing_realized_pnl")
            if fee is None:
                reasons.add("missing_fill_fee")
            elif event.fee_quality not in (FEE_QUALITY_EXACT, FEE_QUALITY_CONVERTED):
                reasons.add("estimated_fill_fee")
        rows.append(Fill(event.id, timestamp, quantity, price, realized, fee))
    return FillTape(
        tuple(PairFills(*key, tuple(grouped[key]), tuple(sorted(quality[key])))
              for key in sorted(grouped)),
        tuple(sorted(global_reasons)),
    )


@dataclass(frozen=True)
class Candle:
    start: int
    minutes: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    available_at: int


@dataclass(frozen=True)
class CandleTape:
    candles: tuple[Candle, ...]
    reasons: tuple[str, ...]

    def payload(self):
        return [asdict(candle) for candle in self.candles]


def capture_candles(rows, *, minutes: int, observed_at: int) -> CandleTape:
    """Copy one factual manager resolution before any resampling or gap filling.

    ``rows`` has the manager's structured-array fields (ts/o/h/l/c), or equivalent mappings. ``observed_at`` is capture time, not exchange
    candle time. The Rust price primitive handles completeness, source conflicts,
    window clipping and the approved ephemeral projections.
    """
    if type(minutes) is not int or minutes not in (1, 5, 15, 60):
        raise ValueError("unsupported revised HSL candle resolution")
    observed_at = _timestamp(observed_at)
    if observed_at is None:
        raise ValueError("invalid revised HSL candle capture time")
    candles, reasons = [], set()
    for row in rows:
        timestamp = _timestamp(row["ts"])
        if timestamp is None:
            reasons.add("undated_candle")
            continue
        values = tuple(_number(row[field]) for field in ("o", "h", "l", "c"))
        if any(value is None for value in values):
            reasons.add("missing_candle_component")
        candles.append(Candle(timestamp, minutes, *values, observed_at))
    return CandleTape(tuple(candles), tuple(sorted(reasons)))
