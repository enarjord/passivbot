"""Fill events management module.

Provides a reusable manager that keeps local cache of canonicalised fill events,
fetches fresh data from the exchange when requested, and exposes convenient query
APIs (PnL summaries, cumulative PnL, last fill timestamps, etc.).

Currently implements a Bitget fetcher; the design is extensible to other
exchanges.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import random
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

from ccxt.base.errors import RateLimitExceeded

try:
    from utils import ts_to_date  # type: ignore
except ImportError:  # pragma: no cover - fallback for package-relative execution
    from .utils import ts_to_date

from config_utils import format_config, load_config
from logging_setup import configure_logging
from procedures import load_user_info
from pure_funcs import ensure_millis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate Limit Coordination
# ---------------------------------------------------------------------------

# Default rate limits per exchange (calls per minute)
_DEFAULT_RATE_LIMITS: Dict[str, Dict[str, int]] = {
    "binance": {"fetch_my_trades": 1200, "fetch_income_history": 120, "default": 1200},
    "bybit": {"fetch_my_trades": 120, "fetch_positions_history": 120, "default": 120},
    "bitget": {"fill_history": 120, "fetch_order": 60, "default": 120},
    "hyperliquid": {"fetch_my_trades": 120, "default": 120},
    "gateio": {"fetch_closed_orders": 120, "default": 120},
    "kucoin": {"fetch_my_trades": 120, "fetch_positions_history": 120, "fetch_order": 60, "default": 120},
}

# Window for rate limit tracking (ms)
_RATE_LIMIT_WINDOW_MS = 60_000

# Default jitter range for staggered startup (seconds)
_STARTUP_JITTER_MIN = 0.0
_STARTUP_JITTER_MAX = 30.0


class RateLimitCoordinator:
    """Coordinates rate limiting across multiple bot instances via shared temp file.

    Each exchange has a temp file that logs recent API calls. Instances check this
    file before making API calls and add jitter if approaching rate limits.
    """

    def __init__(
        self,
        exchange: str,
        user: str,
        *,
        temp_dir: Optional[Path] = None,
        window_ms: int = _RATE_LIMIT_WINDOW_MS,
        limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self.exchange = exchange.lower()
        self.user = user
        self.window_ms = window_ms
        self.limits = limits or _DEFAULT_RATE_LIMITS.get(self.exchange, {"default": 120})

        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "passivbot_rate_limits"
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_file = self.temp_dir / f"{self.exchange}.json"

    def _load_calls(self) -> List[Dict[str, object]]:
        """Load recent API calls from temp file."""
        if not self.temp_file.exists():
            return []
        try:
            with self.temp_file.open("r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data.get("calls", [])
        except Exception as exc:
            logger.debug("RateLimitCoordinator: failed to load %s: %s", self.temp_file, exc)
            return []

    def _save_calls(self, calls: List[Dict[str, object]]) -> None:
        """Save API calls to temp file atomically."""
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        # Prune old entries
        cutoff = now_ms - self.window_ms
        calls = [c for c in calls if c.get("timestamp_ms", 0) > cutoff]

        data = {
            "calls": calls,
            "window_ms": self.window_ms,
            "limits": self.limits,
            "last_update": now_ms,
        }

        tmp_file = self.temp_file.with_suffix(".tmp")
        try:
            with tmp_file.open("w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(data, f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(tmp_file, self.temp_file)
        except Exception as exc:
            logger.debug("RateLimitCoordinator: failed to save %s: %s", self.temp_file, exc)

    def get_current_usage(self, endpoint: str) -> int:
        """Get current call count for an endpoint in the current window."""
        calls = self._load_calls()
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        cutoff = now_ms - self.window_ms
        return sum(1 for c in calls if c.get("endpoint") == endpoint and c.get("timestamp_ms", 0) > cutoff)

    def get_limit(self, endpoint: str) -> int:
        """Get rate limit for an endpoint."""
        return self.limits.get(endpoint, self.limits.get("default", 120))

    def record_call(self, endpoint: str) -> None:
        """Record an API call."""
        calls = self._load_calls()
        calls.append({
            "endpoint": endpoint,
            "timestamp_ms": int(datetime.now(tz=timezone.utc).timestamp() * 1000),
            "user": self.user,
        })
        self._save_calls(calls)

    async def wait_if_needed(self, endpoint: str) -> float:
        """Check rate limit and wait if needed. Returns time waited (seconds)."""
        current = self.get_current_usage(endpoint)
        limit = self.get_limit(endpoint)

        if current >= limit:
            # At or over limit - wait for full window
            wait_time = self.window_ms / 1000.0
            logger.info(
                "RateLimitCoordinator: %s:%s at limit (%d/%d), waiting %.1fs",
                self.exchange, endpoint, current, limit, wait_time
            )
            await asyncio.sleep(wait_time)
            return wait_time
        elif current >= limit * 0.8:
            # Approaching limit - add jitter
            jitter = random.uniform(0.1, 2.0)
            logger.debug(
                "RateLimitCoordinator: %s:%s approaching limit (%d/%d), jitter %.2fs",
                self.exchange, endpoint, current, limit, jitter
            )
            await asyncio.sleep(jitter)
            return jitter

        return 0.0

    @staticmethod
    async def startup_jitter(
        min_seconds: float = _STARTUP_JITTER_MIN,
        max_seconds: float = _STARTUP_JITTER_MAX,
    ) -> float:
        """Apply random jitter at startup to stagger multiple bot launches."""
        jitter = random.uniform(min_seconds, max_seconds)
        if jitter > 0:
            logger.info("RateLimitCoordinator: startup jitter %.2fs", jitter)
            await asyncio.sleep(jitter)
        return jitter


def _format_ms(ts: Optional[int]) -> str:
    if ts is None:
        return "None"
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _day_key(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _merge_fee_lists(
    fees_a: Optional[Sequence], fees_b: Optional[Sequence]
) -> Optional[List[Dict[str, object]]]:
    def to_list(fees):
        if not fees:
            return []
        if isinstance(fees, dict):
            return [fees]
        return list(fees)

    merged: Dict[str, Dict[str, object]] = {}
    for entry in to_list(fees_a) + to_list(fees_b):
        if not isinstance(entry, dict):
            continue
        currency = str(entry.get("currency") or entry.get("code") or "")
        if currency not in merged:
            merged[currency] = dict(entry)
            try:
                merged[currency]["cost"] = float(entry.get("cost", 0.0))
            except Exception:
                merged[currency]["cost"] = 0.0
        else:
            try:
                merged[currency]["cost"] += float(entry.get("cost", 0.0))
            except Exception:
                pass
    if not merged:
        return None
    return [dict(value) for value in merged.values()]


def _fee_cost(fees: Optional[Sequence]) -> float:
    """Sum fee costs defensively, tolerating missing/partial structures."""
    total = 0.0
    if not fees:
        return total
    items: Sequence
    if isinstance(fees, dict):
        items = [fees]
    else:
        try:
            items = list(fees)
        except Exception:
            return total
    for entry in items:
        if not isinstance(entry, dict):
            continue
        try:
            total += float(entry.get("cost", 0.0))
        except Exception:
            continue
    return total


def ensure_qty_signage(events: List[Dict[str, object]]) -> None:
    """Normalize qty sign convention: buys positive, sells negative."""
    for ev in events:
        side = str(ev.get("side") or "").lower()
        qty = float(ev.get("qty") or ev.get("amount") or 0.0)
        if qty == 0.0:
            continue
        if side == "buy" and qty < 0:
            ev["qty"] = abs(qty)
        elif side == "sell" and qty > 0:
            ev["qty"] = -abs(qty)


def annotate_positions_inplace(
    events: List[Dict[str, object]],
    state: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    *,
    recompute_pnl: bool = False,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Given a list of events (expected in chronological order), compute position
    size (psize) and vwap (pprice) per (symbol, position_side) after each fill.
    Qty sign is assumed already normalized (buy +, sell -).
    If recompute_pnl is True, realized PnL is recomputed per fill from positions.
    """
    positions: Dict[Tuple[str, str], Tuple[float, float]] = state or {}
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for ev in events:
        key = (
            str(ev.get("symbol") or ""),
            str(ev.get("position_side") or ev.get("pside") or "long").lower(),
        )
        grouped[key].append(ev)

    def _add_reduce(pos_side: str, qty_signed: float) -> Tuple[float, float]:
        if pos_side == "short":
            add_amt = max(-qty_signed, 0.0)  # sells are negative -> add
            reduce_amt = max(qty_signed, 0.0)  # buys positive -> reduce short
        else:
            add_amt = max(qty_signed, 0.0)  # buys add to long
            reduce_amt = max(-qty_signed, 0.0)  # sells reduce long
        return add_amt, reduce_amt

    for key, evs in grouped.items():
        # sort by time to ensure chronological
        evs.sort(key=lambda x: x.get("timestamp", 0))
        # First forward pass: get tentative sizes with clamping to zero
        forward_sizes: List[float] = []
        pos_size = positions.get(key, (0.0, 0.0))[0]
        for ev in evs:
            qty_signed = float(ev.get("qty") or ev.get("amount") or 0.0) * float(
                ev.get("c_mult", 1.0) or 1.0
            )
            add_amt, reduce_amt = _add_reduce(key[1], qty_signed)
            pos_size = max(pos_size + add_amt - reduce_amt, 0.0)
            forward_sizes.append(pos_size)
        # Backward pass: reconcile sizes starting from final
        reconciled_sizes: List[float] = [0.0] * len(evs)
        current = forward_sizes[-1] if forward_sizes else 0.0
        for idx in range(len(evs) - 1, -1, -1):
            ev = evs[idx]
            qty_signed = float(ev.get("qty") or ev.get("amount") or 0.0) * float(
                ev.get("c_mult", 1.0) or 1.0
            )
            add_amt, reduce_amt = _add_reduce(key[1], qty_signed)
            before = max(current - add_amt + reduce_amt, 0.0)
            reconciled_sizes[idx] = current
            current = before

        # Final forward pass to compute pprice with reconciled sizes
        pos_size = positions.get(key, (0.0, 0.0))[0]
        vwap = positions.get(key, (0.0, 0.0))[1]
        for ev, after_size in zip(evs, reconciled_sizes):
            qty_signed = float(ev.get("qty") or ev.get("amount") or 0.0) * float(
                ev.get("c_mult", 1.0) or 1.0
            )
            price = float(ev.get("price") or 0.0)
            add_amt, reduce_amt = _add_reduce(key[1], qty_signed)
            before_size = max(after_size - add_amt + reduce_amt, 0.0)
            if recompute_pnl:
                realized = 0.0
                if reduce_amt > 0 and before_size > 0 and price > 0 and vwap >= 0:
                    close_qty = min(before_size, reduce_amt)
                    if key[1] == "short":
                        realized = (vwap - price) * close_qty
                    else:
                        realized = (price - vwap) * close_qty
                    ev["pnl"] = realized - _fee_cost(ev.get("fees"))
            if add_amt > 0:
                if before_size <= 0:
                    vwap = price
                else:
                    vwap = ((before_size * vwap) + (add_amt * price)) / max(
                        before_size + add_amt, 1e-12
                    )
            if after_size <= 1e-12:
                vwap = 0.0
            ev["psize"] = round(after_size, 12)
            ev["pprice"] = vwap
            pos_size, _ = positions.get(key, (0.0, 0.0))
        positions[key] = (reconciled_sizes[-1] if reconciled_sizes else pos_size, vwap)

    return positions


def compute_realized_pnls_from_trades(
    trades: List[Dict[str, object]],
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], Tuple[float, float]]]:
    """
    Compute realized PnL per trade by reconstructing positions from fills.

    Tracks positions separately per (symbol, position_side) so hedged longs/shorts
    do not interfere. Position_size is always kept as a positive magnitude for the
    given side; reductions trigger realized PnL.

    Returns:
        per_trade_pnl: mapping trade_id -> realized pnl (gross, without fees)
        final_positions: mapping (symbol, position_side) -> (pos_size, vwap)
    """
    per_trade: Dict[str, float] = {}
    positions: Dict[Tuple[str, str], Tuple[float, float]] = {}

    for trade in sorted(trades, key=lambda x: x.get("timestamp", 0)):
        trade_id = str(trade.get("id") or "")
        if not trade_id:
            continue
        symbol = str(trade.get("symbol") or "")
        side = str(trade.get("side") or "").lower()
        pos_side = str(trade.get("position_side") or trade.get("pside") or "long").lower()
        qty = abs(float(trade.get("qty") or trade.get("amount") or 0.0))
        price = float(trade.get("price") or 0.0)
        if qty <= 0 or price <= 0 or not symbol:
            per_trade[trade_id] = 0.0
            continue

        key = (symbol, pos_side)
        pos_size, vwap = positions.get(key, (0.0, 0.0))

        # Determine whether this trade adds or reduces for this side
        if pos_side == "short":
            adds = side == "sell"
        else:  # long or unknown
            adds = side == "buy"

        realized = 0.0
        if not adds:
            # reducing position
            if pos_size > 0:
                closing_qty = min(pos_size, qty)
                if pos_side == "short":
                    realized += (vwap - price) * closing_qty
                else:
                    realized += (price - vwap) * closing_qty
                pos_size -= closing_qty
                if pos_size < 1e-12:
                    pos_size = 0.0
                    vwap = 0.0
                leftover = qty - closing_qty
                if leftover > 0:
                    # trade overshoots and becomes a new position in trade direction
                    pos_size = leftover
                    vwap = price
        else:
            # adding to position
            new_size = pos_size + qty
            if pos_size == 0.0:
                vwap = price
            else:
                vwap = ((pos_size * vwap) + (qty * price)) / (pos_size + qty)
            pos_size = new_size

        positions[key] = (pos_size, vwap)
        per_trade[trade_id] = realized

    return per_trade, positions


def _coalesce_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Group events sharing timestamp/symbol/pb_type/side/position."""
    aggregated: Dict[Tuple, Dict[str, object]] = {}
    order: List[Tuple] = []
    for ev in events:
        key = (
            ev.get("timestamp"),
            ev.get("symbol"),
            ev.get("pb_order_type"),
            ev.get("side"),
            ev.get("position_side"),
        )
        if key not in aggregated:
            aggregated[key] = dict(ev)
            aggregated[key]["id"] = str(ev.get("id", ""))
            aggregated[key]["qty"] = float(ev.get("qty", 0.0))
            aggregated[key]["pnl"] = float(ev.get("pnl", 0.0))
            aggregated[key]["fees"] = _merge_fee_lists(ev.get("fees"), None)
            aggregated[key]["_price_numerator"] = float(ev.get("price", 0.0)) * float(
                ev.get("qty", 0.0)
            )
            order.append(key)
        else:
            agg = aggregated[key]
            agg["id"] = f"{agg['id']}+{ev.get('id', '')}".strip("+")
            agg["qty"] = float(agg.get("qty", 0.0)) + float(ev.get("qty", 0.0))
            agg["pnl"] = float(agg.get("pnl", 0.0)) + float(ev.get("pnl", 0.0))
            agg["fees"] = _merge_fee_lists(agg.get("fees"), ev.get("fees"))
            agg["_price_numerator"] = float(agg.get("_price_numerator", 0.0)) + float(
                ev.get("price", 0.0)
            ) * float(ev.get("qty", 0.0))
            if not agg.get("client_order_id") and ev.get("client_order_id"):
                agg["client_order_id"] = ev.get("client_order_id")
            if not agg.get("pb_order_type"):
                agg["pb_order_type"] = ev.get("pb_order_type")
    coalesced: List[Dict[str, object]] = []
    for key in order:
        agg = aggregated[key]
        qty = float(agg.get("qty", 0.0))
        price_numerator = float(agg.get("_price_numerator", 0.0))
        if qty > 0:
            agg["price"] = price_numerator / qty
        agg.pop("_price_numerator", None)
        fees = agg.get("fees")
        if isinstance(fees, list) and len(fees) == 1:
            agg["fees"] = fees[0]
        coalesced.append(agg)
    return coalesced


def _check_pagination_progress(
    previous: Optional[Tuple[Tuple[str, object], ...]],
    params: Dict[str, object],
    context: str,
) -> Optional[Tuple[Tuple[str, object], ...]]:
    params_key = tuple(sorted(params.items()))
    if previous == params_key:
        logger.warning(
            "%s: repeated params detected; aborting pagination (%s)",
            context,
            dict(params),
        )
        return None
    logger.debug("%s: fetching with params %s", context, dict(params))
    return params_key


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


def _normalize_raw_field(raw: object) -> List[Dict[str, object]]:
    """Normalize raw field to List[Dict] format.

    Handles migration from old Dict format to new List[Dict] format.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        # Already in new format - validate and return
        return [dict(item) if isinstance(item, dict) else {"data": item} for item in raw]
    if isinstance(raw, dict):
        # Old format: single dict -> wrap in list with "legacy" source
        return [{"source": "legacy", "data": raw}]
    # Unknown format
    return [{"source": "unknown", "data": str(raw)}]


@dataclass(frozen=True)
class FillEvent:
    """Canonical representation of a single fill event."""

    id: str
    timestamp: int
    datetime: str
    symbol: str
    side: str
    qty: float
    price: float
    pnl: float
    fees: Optional[Sequence]
    pb_order_type: str
    position_side: str
    client_order_id: str
    psize: float = 0.0
    pprice: float = 0.0
    raw: List[Dict[str, object]] = None  # List of raw payloads from multiple sources

    @property
    def key(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "datetime": self.datetime,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "pnl": self.pnl,
            "fees": self.fees,
            "pb_order_type": self.pb_order_type,
            "position_side": self.position_side,
            "client_order_id": self.client_order_id,
            "psize": self.psize,
            "pprice": self.pprice,
            "raw": self.raw if self.raw is not None else [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "FillEvent":
        required = [
            "id",
            "timestamp",
            "symbol",
            "side",
            "qty",
            "price",
            "pnl",
            "pb_order_type",
            "position_side",
            "client_order_id",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Fill event missing required keys: {missing}")
        return cls(
            id=str(data["id"]),
            timestamp=int(data["timestamp"]),
            datetime=str(data.get("datetime") or ts_to_date(int(data["timestamp"]))),
            symbol=str(data["symbol"]),
            side=str(data["side"]).lower(),
            qty=float(data["qty"]),
            price=float(data["price"]),
            pnl=float(data["pnl"]),
            fees=data.get("fees"),
            pb_order_type=str(data["pb_order_type"]),
            position_side=str(data["position_side"]).lower(),
            client_order_id=str(data["client_order_id"]),
            psize=float(data.get("psize", 0.0)),
            pprice=float(data.get("pprice", 0.0)),
            raw=_normalize_raw_field(data.get("raw")),
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# Maximum retry attempts before marking gap as persistent
_GAP_MAX_RETRIES = 3

# Gap confidence levels
GAP_CONFIDENCE_UNKNOWN = 0.0
GAP_CONFIDENCE_SUSPICIOUS = 0.3
GAP_CONFIDENCE_LIKELY_LEGITIMATE = 0.7
GAP_CONFIDENCE_CONFIRMED = 1.0

# Gap reasons
GAP_REASON_AUTO = "auto_detected"
GAP_REASON_FETCH_FAILED = "fetch_failed"
GAP_REASON_CONFIRMED = "confirmed_legitimate"
GAP_REASON_MANUAL = "manual"


class KnownGap(TypedDict, total=False):
    """Gap metadata stored in metadata.json known_gaps."""

    start_ts: int  # Gap start timestamp (ms)
    end_ts: int  # Gap end timestamp (ms)
    retry_count: int  # Number of fetch attempts (max 3)
    reason: str  # auto_detected, fetch_failed, confirmed_legitimate, manual
    added_at: int  # Timestamp when gap was first detected
    confidence: float  # 0.0=unknown, 0.3=suspicious, 0.7=likely_ok, 1.0=confirmed


class CacheMetadata(TypedDict, total=False):
    """Cache metadata stored in metadata.json."""

    last_refresh_ms: int  # Timestamp of last successful refresh
    oldest_event_ts: int  # Oldest event timestamp in cache
    newest_event_ts: int  # Newest event timestamp in cache
    known_gaps: List[KnownGap]  # List of known gaps


class FillEventCache:
    """JSON cache storing fills split by UTC day."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata: Optional[CacheMetadata] = None

    def load(self) -> List[FillEvent]:
        files = sorted(self.root.glob("*.json"))
        events: List[FillEvent] = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh) or []
            except Exception as exc:
                logger.warning("FillEventCache.load: failed to read %s (%s)", path, exc)
                continue
            for raw in payload:
                try:
                    events.append(FillEvent.from_dict(raw))
                except Exception:
                    logger.debug("FillEventCache.load: skipping malformed record in %s", path)
        events.sort(key=lambda ev: ev.timestamp)
        logger.info(
            "FillEventCache.load: loaded %d events from %d files in %s",
            len(events),
            len(files),
            self.root,
        )
        return events

    def save(self, events: Sequence[FillEvent]) -> None:
        day_map: Dict[str, List[FillEvent]] = defaultdict(list)
        for event in events:
            day_map[_day_key(event.timestamp)].append(event)
        for day in day_map:
            day_map[day].sort(key=lambda ev: ev.timestamp)
        self.save_days(day_map)

    def save_days(self, day_events: Dict[str, Sequence[FillEvent]]) -> None:
        for day, events in day_events.items():
            path = self.root / f"{day}.json"
            payload = [event.to_dict() for event in sorted(events, key=lambda ev: ev.timestamp)]
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        current = json.load(fh)
                except Exception:
                    current = None
                if current == payload:
                    logger.debug("FillEventCache.save_days: %s unchanged", path.name)
                    continue
            tmp_path = path.with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp_path, path)
            logger.info(
                "FillEventCache.save_days: wrote %d events to %s",
                len(payload),
                path,
            )

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def load_metadata(self) -> CacheMetadata:
        """Load cache metadata from disk."""
        if self._metadata is not None:
            return self._metadata

        default: CacheMetadata = {
            "last_refresh_ms": 0,
            "oldest_event_ts": 0,
            "newest_event_ts": 0,
            "known_gaps": [],
        }

        if not self.metadata_path.exists():
            self._metadata = default
            return self._metadata

        try:
            with self.metadata_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                data = default
            # Ensure all keys exist
            for key in default:
                data.setdefault(key, default[key])
            self._metadata = data
        except Exception as exc:
            logger.warning("FillEventCache.load_metadata: failed to read %s (%s)", self.metadata_path, exc)
            self._metadata = default

        return self._metadata

    def save_metadata(self, metadata: Optional[CacheMetadata] = None) -> None:
        """Save cache metadata to disk atomically."""
        if metadata is not None:
            self._metadata = metadata

        if self._metadata is None:
            return

        tmp_path = self.metadata_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._metadata, fh, indent=2)
            os.replace(tmp_path, self.metadata_path)
            logger.debug("FillEventCache.save_metadata: wrote to %s", self.metadata_path)
        except Exception as exc:
            logger.error("FillEventCache.save_metadata: failed to write %s (%s)", self.metadata_path, exc)

    def update_metadata_from_events(self, events: Sequence[FillEvent]) -> None:
        """Update metadata timestamps based on events."""
        if not events:
            return

        metadata = self.load_metadata()
        timestamps = [ev.timestamp for ev in events]
        oldest = min(timestamps)
        newest = max(timestamps)

        current_oldest = metadata.get("oldest_event_ts", 0)
        current_newest = metadata.get("newest_event_ts", 0)

        if current_oldest == 0 or oldest < current_oldest:
            metadata["oldest_event_ts"] = oldest
        if newest > current_newest:
            metadata["newest_event_ts"] = newest

        metadata["last_refresh_ms"] = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        self.save_metadata(metadata)

    def get_known_gaps(self) -> List[KnownGap]:
        """Return list of known gaps."""
        return self.load_metadata().get("known_gaps", [])

    def add_known_gap(
        self,
        start_ts: int,
        end_ts: int,
        *,
        reason: str = GAP_REASON_AUTO,
        confidence: float = GAP_CONFIDENCE_UNKNOWN,
    ) -> None:
        """Add or update a known gap."""
        metadata = self.load_metadata()
        gaps = metadata.get("known_gaps", [])
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

        # Check for overlapping gap to update
        for gap in gaps:
            if gap["start_ts"] <= end_ts and gap["end_ts"] >= start_ts:
                # Overlapping - merge
                gap["start_ts"] = min(gap["start_ts"], start_ts)
                gap["end_ts"] = max(gap["end_ts"], end_ts)
                gap["retry_count"] = gap.get("retry_count", 0) + 1
                if gap["retry_count"] >= _GAP_MAX_RETRIES:
                    gap["confidence"] = max(gap.get("confidence", 0), GAP_CONFIDENCE_LIKELY_LEGITIMATE)
                logger.info(
                    "FillEventCache.add_known_gap: updated gap %s → %s (retry_count=%d)",
                    _format_ms(gap["start_ts"]),
                    _format_ms(gap["end_ts"]),
                    gap["retry_count"],
                )
                self.save_metadata(metadata)
                return

        # New gap
        new_gap: KnownGap = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "retry_count": 0,
            "reason": reason,
            "added_at": now_ms,
            "confidence": confidence,
        }
        gaps.append(new_gap)
        metadata["known_gaps"] = gaps
        logger.info(
            "FillEventCache.add_known_gap: added new gap %s → %s (reason=%s)",
            _format_ms(start_ts),
            _format_ms(end_ts),
            reason,
        )
        self.save_metadata(metadata)

    def clear_gap(self, start_ts: int, end_ts: int) -> bool:
        """Remove a gap that has been filled. Returns True if a gap was removed."""
        metadata = self.load_metadata()
        gaps = metadata.get("known_gaps", [])
        original_count = len(gaps)

        # Remove gaps that are fully contained in the filled range
        remaining = []
        for gap in gaps:
            if gap["start_ts"] >= start_ts and gap["end_ts"] <= end_ts:
                logger.info(
                    "FillEventCache.clear_gap: removed gap %s → %s",
                    _format_ms(gap["start_ts"]),
                    _format_ms(gap["end_ts"]),
                )
                continue
            # Partial overlap - trim the gap
            if gap["start_ts"] < start_ts < gap["end_ts"]:
                gap["end_ts"] = start_ts
            if gap["start_ts"] < end_ts < gap["end_ts"]:
                gap["start_ts"] = end_ts
            if gap["start_ts"] < gap["end_ts"]:
                remaining.append(gap)

        if len(remaining) != original_count:
            metadata["known_gaps"] = remaining
            self.save_metadata(metadata)
            return True
        return False

    def should_retry_gap(self, gap: KnownGap) -> bool:
        """Check if a gap should be retried (retry_count < max)."""
        return gap.get("retry_count", 0) < _GAP_MAX_RETRIES

    def get_coverage_summary(self) -> Dict[str, object]:
        """Return a summary of cache coverage for debugging."""
        metadata = self.load_metadata()
        gaps = metadata.get("known_gaps", [])

        persistent_gaps = [g for g in gaps if not self.should_retry_gap(g)]
        retryable_gaps = [g for g in gaps if self.should_retry_gap(g)]

        total_gap_ms = sum(g["end_ts"] - g["start_ts"] for g in gaps)

        return {
            "oldest_event_ts": metadata.get("oldest_event_ts", 0),
            "newest_event_ts": metadata.get("newest_event_ts", 0),
            "last_refresh_ms": metadata.get("last_refresh_ms", 0),
            "total_gaps": len(gaps),
            "persistent_gaps": len(persistent_gaps),
            "retryable_gaps": len(retryable_gaps),
            "total_gap_hours": total_gap_ms / (1000 * 60 * 60) if total_gap_ms > 0 else 0,
            "gaps": [
                {
                    "start": _format_ms(g["start_ts"]),
                    "end": _format_ms(g["end_ts"]),
                    "retry_count": g.get("retry_count", 0),
                    "reason": g.get("reason", "unknown"),
                    "confidence": g.get("confidence", 0),
                }
                for g in gaps
            ],
        }


# ---------------------------------------------------------------------------
# Fetcher infrastructure
# ---------------------------------------------------------------------------


class BaseFetcher:
    """Abstract interface for exchange-specific fill fetchers."""

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        raise NotImplementedError


class BitgetFetcher(BaseFetcher):
    """Fetches and enriches fill events from Bitget."""

    def __init__(
        self,
        api,
        *,
        product_type: str = "USDT-FUTURES",
        history_limit: int = 100,
        detail_calls_per_minute: int = 120,
        detail_concurrency: int = 10,
        now_func: Optional[Callable[[], int]] = None,
        symbol_resolver: Optional[Callable[[Optional[str]], str]] = None,
    ) -> None:
        self.api = api
        self.product_type = product_type
        self.history_limit = history_limit
        self.detail_calls_per_minute = max(1, detail_calls_per_minute)
        self._detail_call_timestamps: deque[int] = deque()
        self.detail_concurrency = max(1, detail_concurrency)
        self._rate_lock = asyncio.Lock()
        self._now_func = now_func or (lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000))
        if symbol_resolver is None:
            raise ValueError("BitgetFetcher requires a symbol_resolver callable")
        self._symbol_resolver = symbol_resolver

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        buffer_step_ms = 24 * 60 * 60 * 1000
        end_time = int(until_ms) if until_ms is not None else self._now_func() + buffer_step_ms
        params: Dict[str, object] = {
            "productType": self.product_type,
            "limit": self.history_limit,
            "endTime": end_time,
        }
        events: Dict[str, Dict[str, object]] = {}

        detail_hits = 0
        detail_fetches = 0
        max_fetches = 400
        fetch_count = 0

        logger.info(
            "BitgetFetcher.fetch: start (since=%s, until=%s, limit=%d)",
            _format_ms(since_ms),
            _format_ms(until_ms),
            self.history_limit,
        )

        while True:
            if fetch_count >= max_fetches:
                logger.warning(
                    "BitgetFetcher.fetch: reached maximum pagination depth (%d)",
                    max_fetches,
                )
                break
            fetch_count += 1
            payload = await self.api.private_mix_get_v2_mix_order_fill_history(dict(params))
            fill_list = payload.get("data", {}).get("fillList") or []
            if fetch_count > 1:
                logger.info(
                    "BitgetFetcher.fetch: fetch #%d endTime=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("endTime")),
                    len(fill_list),
                )
            if not fill_list:
                if since_ms is None:
                    logger.debug("BitgetFetcher.fetch: empty batch without start bound; stopping")
                    break
                end_param = int(params.get("endTime", self._now_func()))
                if end_param <= since_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: empty batch and cursor reached start; stopping"
                    )
                    break
                new_end_time = max(since_ms, end_param - buffer_step_ms)
                if new_end_time == end_param:
                    new_end_time = max(since_ms, end_param - 1)
                params["endTime"] = new_end_time
                logger.debug(
                    "BitgetFetcher.fetch: empty batch, continuing with endTime=%s",
                    _format_ms(params["endTime"]),
                )
                continue
            logger.debug(
                "BitgetFetcher.fetch: received batch size=%d endTime=%s",
                len(fill_list),
                params.get("endTime"),
            )
            batch_ids: List[str] = []
            pending_tasks: List[asyncio.Task[int]] = []
            for raw in fill_list:
                event = self._normalize_fill(raw)
                event_id = event["id"]
                if not event_id:
                    continue
                batch_ids.append(event_id)
                if event_id in detail_cache:
                    client_oid, pb_type = detail_cache[event_id]
                    event["client_order_id"] = client_oid
                    event["pb_order_type"] = pb_type
                    detail_hits += 1
                if not event.get("client_order_id"):
                    pending_tasks.append(
                        asyncio.create_task(self._enrich_with_details(event, detail_cache))
                    )
                    if len(pending_tasks) >= self.detail_concurrency:
                        detail_fetches += await self._flush_detail_tasks(pending_tasks)
                events[event_id] = event
            detail_fetches += await self._flush_detail_tasks(pending_tasks)
            if on_batch:
                batch_events = [
                    dict(events[event_id])
                    for event_id in batch_ids
                    if events[event_id].get("client_order_id")
                ]
                if batch_events:
                    on_batch(batch_events)
            oldest = min(int(raw["cTime"]) for raw in fill_list)
            if len(fill_list) < self.history_limit:
                if since_ms is None:
                    logger.debug(
                        "BitgetFetcher.fetch: short batch size=%d without start bound; stopping",
                        len(fill_list),
                    )
                    break
                end_param = int(params.get("endTime", oldest))
                if end_param - since_ms < buffer_step_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: short batch size=%d close to requested start; stopping",
                        len(fill_list),
                    )
                    break
                new_end_time = max(since_ms, min(end_param, oldest) - 1)
                if new_end_time <= since_ms:
                    logger.debug(
                        "BitgetFetcher.fetch: rewound endTime to start boundary; stopping",
                    )
                    break
                params["endTime"] = new_end_time
                logger.debug(
                    "BitgetFetcher.fetch: short batch size=%d, continuing with endTime=%s",
                    len(fill_list),
                    _format_ms(params["endTime"]),
                )
                continue
            first_ts = min(ev["timestamp"] for ev in events.values()) if events else None
            if since_ms is not None and first_ts is not None and first_ts <= since_ms:
                break
            params["endTime"] = max(since_ms or oldest, oldest - 1)

        ordered = sorted(events.values(), key=lambda ev: ev["timestamp"])
        if since_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] >= since_ms]
        if until_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] <= until_ms]
        logger.info(
            "BitgetFetcher.fetch: done (events=%d, detail_cache_hits=%d, detail_fetches=%d)",
            len(ordered),
            detail_hits,
            detail_fetches,
        )
        return ordered

    async def _enrich_with_details(
        self,
        event: Dict[str, object],
        cache: Dict[str, Tuple[str, str]],
    ) -> int:
        if not event.get("order_id"):
            return 0
        logger.debug(
            "BitgetFetcher._enrich_with_details: fetching detail for order %s %s",
            event["order_id"],
            event.get("datetime"),
        )
        await self._respect_rate_limit()
        order_details = await self.api.private_mix_get_v2_mix_order_detail(
            {
                "productType": self.product_type,
                "orderId": event["order_id"],
                "symbol": event["symbol_external"],
            }
        )
        client_oid = (
            order_details.get("data", {}).get("clientOid")
            if isinstance(order_details, dict)
            else None
        )
        if client_oid:
            pb_type = custom_id_to_snake(client_oid)
            event["client_order_id"] = client_oid
            event["pb_order_type"] = pb_type
            cache[event["id"]] = (client_oid, pb_type)
            logger.debug(
                "BitgetFetcher._enrich_with_details: cached clientOid=%s for trade %s, pb_order_type %s",
                client_oid,
                event["id"],
                pb_type,
            )
            return 1
        else:
            logger.debug(
                "BitgetFetcher._enrich_with_details: no clientOid returned for order %s",
                event["order_id"],
            )
            return 1

    async def _respect_rate_limit(self) -> None:
        window_ms = 60_000
        max_calls = self.detail_calls_per_minute
        q = self._detail_call_timestamps
        while True:
            async with self._rate_lock:
                now = self._now_func()
                window_start = now - window_ms
                while q and q[0] <= window_start:
                    q.popleft()
                if len(q) < max_calls:
                    q.append(now)
                    return
                wait_ms = q[0] + window_ms - now
            if wait_ms > 0:
                logger.debug(
                    "BitgetFetcher._respect_rate_limit: sleeping %.3fs to respect %d calls/min",
                    wait_ms / 1000,
                    max_calls,
                )
                await asyncio.sleep(wait_ms / 1000)
            else:
                await asyncio.sleep(0)

    async def _flush_detail_tasks(self, tasks: List[asyncio.Task[int]]) -> int:
        if not tasks:
            return 0
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()
        total = 0
        for res in results:
            if isinstance(res, Exception):
                logger.error(
                    "BitgetFetcher._flush_detail_tasks: detail fetch failed: %s",
                    res,
                )
                continue
            total += res or 0
        return total

    def _normalize_fill(self, raw: Dict[str, object]) -> Dict[str, object]:
        timestamp = int(raw["cTime"])
        side, position_side = deduce_side_pside(raw)
        return {
            "id": raw.get("tradeId"),
            "order_id": raw.get("orderId"),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": self._resolve_symbol(raw.get("symbol")),
            "symbol_external": raw.get("symbol"),
            "side": side,
            "qty": float(raw.get("baseVolume", 0.0)),
            "price": float(raw.get("price", 0.0)),
            "pnl": float(raw.get("profit", 0.0)),
            "fees": raw.get("feeDetail"),
            "pb_order_type": raw.get("pb_order_type", ""),
            "position_side": position_side,
            "client_order_id": raw.get("client_order_id"),
            "raw": [{"source": "fill_history", "data": dict(raw)}],
        }

    def _resolve_symbol(self, market_symbol: Optional[str]) -> str:
        if not market_symbol:
            return ""
        try:
            resolved = self._symbol_resolver(market_symbol)
        except Exception as exc:
            logger.warning(
                "BitgetFetcher._resolve_symbol: resolver failed for %s (%s); using fallback",
                market_symbol,
                exc,
            )
            resolved = None
        if resolved:
            return resolved
        logger.warning(
            "BitgetFetcher._resolve_symbol: unresolved symbol '%s'; falling back to raw value",
            market_symbol,
        )
        return str(market_symbol)


class BinanceFetcher(BaseFetcher):
    """Fetch realised PnL events for Binance by combining income and trade history."""

    def __init__(
        self,
        api,
        *,
        symbol_resolver: Callable[[str], str],
        now_func: Optional[Callable[[], int]] = None,
        positions_provider: Optional[Callable[[], Iterable[str]]] = None,
        open_orders_provider: Optional[Callable[[], Iterable[str]]] = None,
        income_limit: int = 1000,
        trade_limit: int = 1000,
    ) -> None:
        self.api = api
        if symbol_resolver is None:
            raise ValueError("BinanceFetcher requires a symbol_resolver callable")
        self._symbol_resolver = symbol_resolver
        self._positions_provider = positions_provider or (lambda: ())
        self._open_orders_provider = open_orders_provider or (lambda: ())
        self.income_limit = min(1000, max(1, income_limit))  # cap to max 1000
        self._now_func = now_func or (lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000))
        self.trade_limit = max(1, trade_limit)

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        logger.info(
            "BinanceFetcher.fetch: start since=%s until=%s",
            _format_ms(since_ms),
            _format_ms(until_ms),
        )
        income_events = await self._fetch_income(since_ms, until_ms)
        symbol_pool = set(self._collect_symbols(self._positions_provider))
        symbol_pool.update(self._collect_symbols(self._open_orders_provider))
        symbol_pool.update(ev["symbol"] for ev in income_events if ev.get("symbol"))
        if detail_cache is None:
            detail_cache = {}

        trade_events: Dict[str, Dict[str, object]] = {}
        trade_tasks: Dict[str, asyncio.Task[List[Dict[str, object]]]] = {}
        for symbol in sorted(symbol_pool):
            if not symbol:
                continue
            trade_tasks[symbol] = asyncio.create_task(
                self._fetch_symbol_trades(symbol, since_ms, until_ms)
            )
        for symbol, task in trade_tasks.items():
            try:
                trades = await task
            except RateLimitExceeded as exc:  # pragma: no cover - depends on live API
                logger.warning(
                    "BinanceFetcher.fetch: rate-limited fetching trades for %s (%s)", symbol, exc
                )
                trades = []
            except Exception as exc:
                logger.error("BinanceFetcher.fetch: error fetching trades for %s (%s)", symbol, exc)
                trades = []
            for trade in trades:
                event = self._normalize_trade(trade)
                cached = detail_cache.get(event["id"])
                if cached:
                    event.setdefault("client_order_id", cached[0])
                    if cached[1]:
                        event.setdefault("pb_order_type", cached[1])
                trade_events[event["id"]] = event

        merged: Dict[str, Dict[str, object]] = {}
        for ev in income_events:
            merged[ev["id"]] = ev

        def _event_from_trade(trade: Dict[str, object]) -> Dict[str, object]:
            symbol = trade.get("symbol") or self._resolve_symbol(trade.get("info", {}).get("symbol"))
            timestamp = int(trade.get("timestamp") or 0)
            client_oid = trade.get("client_order_id") or ""
            event: Dict[str, object] = {
                "id": str(trade.get("id")),
                "timestamp": timestamp,
                "datetime": ts_to_date(timestamp) if timestamp else "",
                "symbol": symbol or "",
                "side": trade.get("side") or "",
                "qty": float(trade.get("qty") or 0.0),
                "price": float(trade.get("price") or 0.0),
                "pnl": float(trade.get("pnl") or 0.0),
                "fees": trade.get("fees"),
                "pb_order_type": trade.get("pb_order_type") or "",
                "position_side": trade.get("position_side") or "unknown",
                "client_order_id": client_oid,
                "order_id": trade.get("order_id") or "",
                "info": trade.get("info"),
            }
            return event

        def _merge_trade_into_event(event: Dict[str, object], trade: Dict[str, object]) -> None:
            if not event.get("symbol") and trade.get("symbol"):
                event["symbol"] = trade["symbol"]
            if not event.get("side") and trade.get("side"):
                event["side"] = trade["side"]
            if float(event.get("qty", 0.0)) == 0.0 and trade.get("qty") is not None:
                event["qty"] = float(trade.get("qty", 0.0))
            if float(event.get("price", 0.0)) == 0.0 and trade.get("price") is not None:
                event["price"] = float(trade.get("price", 0.0))
            if not event.get("fees") and trade.get("fees"):
                event["fees"] = trade["fees"]
            if (event.get("position_side") in (None, "", "unknown")) and trade.get("position_side"):
                event["position_side"] = trade["position_side"]
            if trade.get("client_order_id"):
                event["client_order_id"] = trade["client_order_id"]
            if trade.get("order_id"):
                event["order_id"] = trade["order_id"]
            if trade.get("info"):
                event["info"] = trade["info"]
            if trade.get("pb_order_type"):
                event["pb_order_type"] = trade["pb_order_type"]

        if trade_events:
            for event_id, trade in trade_events.items():
                if event_id not in merged:
                    merged[event_id] = _event_from_trade(trade)
                event = merged[event_id]
                _merge_trade_into_event(event, trade)

        for event_id, event in merged.items():
            cached = detail_cache.get(event_id)
            if cached:
                client_oid, pb_type = cached
                if client_oid:
                    event["client_order_id"] = client_oid
                if pb_type and pb_type != "unknown":
                    event["pb_order_type"] = pb_type

        enrichment_tasks: List[asyncio.Task[Optional[Tuple[str, str]]]] = []
        enrichment_events: List[Tuple[Dict[str, object], str]] = []
        if merged:
            for event_id, event in merged.items():
                has_client = bool(event.get("client_order_id"))
                has_type = bool(event.get("pb_order_type")) and event["pb_order_type"] != "unknown"
                if has_client and has_type:
                    continue
                trade = trade_events.get(event_id)
                order_id = None
                symbol = None
                if trade:
                    order_id = trade.get("order_id")
                    symbol = trade.get("symbol") or event.get("symbol")
                else:
                    order_id = event.get("order_id")
                    symbol = event.get("symbol")
                if not order_id or not symbol:
                    continue
                enrichment_events.append((event, event_id))
                enrichment_tasks.append(
                    asyncio.create_task(
                        self._enrich_with_order_details(
                            str(order_id),
                            str(symbol),
                        )
                    )
                )
        if enrichment_tasks:
            detail_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
            for (event, event_id), res in zip(enrichment_events, detail_results):
                if isinstance(res, Exception):
                    logger.debug(
                        "BinanceFetcher.fetch: fetch_order failed for %s (%s)",
                        event.get("id"),
                        res,
                    )
                    continue
                if not res:
                    continue
                client_oid, pb_type = res
                event["client_order_id"] = client_oid
                if pb_type:
                    event["pb_order_type"] = pb_type
                if event_id:
                    detail_cache[event_id] = (client_oid, pb_type or "")

        for event_id, ev in merged.items():
            client_oid = ev.get("client_order_id")
            if client_oid and not ev.get("pb_order_type"):
                ev["pb_order_type"] = custom_id_to_snake(str(client_oid))
            if not ev.get("pb_order_type"):
                ev["pb_order_type"] = ""
            ev["client_order_id"] = str(client_oid) if client_oid is not None else ""
            if event_id and ev.get("client_order_id"):
                detail_cache[event_id] = (ev["client_order_id"], ev["pb_order_type"])

        ordered = sorted(merged.values(), key=lambda ev: ev["timestamp"])
        if since_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] >= since_ms]
        if until_ms is not None:
            ordered = [ev for ev in ordered if ev["timestamp"] <= until_ms]

        if on_batch and ordered:
            on_batch(ordered)

        logger.info(
            "BinanceFetcher.fetch: done events=%d (symbols=%d)",
            len(ordered),
            len(symbol_pool),
        )
        return ordered

    async def _enrich_with_order_details(
        self,
        order_id: Optional[str],
        symbol: Optional[str],
    ) -> Optional[Tuple[str, str]]:
        if not order_id or not symbol:
            return None
        try:
            detail = await self.api.fetch_order(order_id, symbol)
        except Exception as exc:  # pragma: no cover - live API dependent
            logger.debug(
                "BinanceFetcher._enrich_with_order_details: fetch_order failed for %s (%s)",
                order_id,
                exc,
            )
            return None
        info = detail.get("info") if isinstance(detail, dict) else detail
        if not isinstance(info, dict):
            return None
        client_oid = info.get("clientOrderId") or info.get("clientOrderID")
        if not client_oid:
            return None
        client_oid = str(client_oid)
        return client_oid, custom_id_to_snake(client_oid)

    async def _fetch_income(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {"incomeType": "REALIZED_PNL", "limit": self.income_limit}
        if until_ms is None:
            if since_ms is None:
                logger.debug(f"BinanceFetcher._fetch_income.fapiprivate_get_income params={params}")
                payload = await self.api.fapiprivate_get_income(params=params)
                return sorted(
                    [self._normalize_income(x) for x in payload], key=lambda x: x["timestamp"]
                )
            until_ms = self._now_func() + 1000 * 60 * 60
        week_buffer_ms = 1000 * 60 * 60 * 24 * 6.95
        params["startTime"] = int(since_ms)
        params["endTime"] = int(min(until_ms, since_ms + week_buffer_ms))
        events = []
        previous_key: Optional[Tuple[Tuple[str, object], ...]] = None
        fetch_count = 0
        while True:
            key = _check_pagination_progress(
                previous_key,
                params,
                "BinanceFetcher._fetch_income",
            )
            if key is None:
                break
            previous_key = key
            fetch_count += 1
            payload = await self.api.fapiprivate_get_income(params=params)
            if fetch_count > 1:
                logger.info(
                    "BinanceFetcher._fetch_income: fetch #%d startTime=%s endTime=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("startTime")),
                    _format_ms(params.get("endTime")),
                    len(payload) if payload else 0,
                )
            if payload == []:
                if params["startTime"] + week_buffer_ms >= until_ms:
                    break
                params["startTime"] = int(params["startTime"] + week_buffer_ms)
                params["endTime"] = int(min(until_ms, params["startTime"] + week_buffer_ms))
                continue
            events.extend(
                sorted([self._normalize_income(x) for x in payload], key=lambda x: x["timestamp"])
            )
            params["startTime"] = int(events[-1]["timestamp"]) + 1
            params["endTime"] = int(min(until_ms, params["startTime"] + week_buffer_ms))
            if params["startTime"] > until_ms:
                break
        return events

    async def _fetch_symbol_trades(
        self,
        ccxt_symbol: str,
        since_ms: Optional[int],
        until_ms: Optional[int],
    ) -> List[Dict[str, object]]:
        limit = min(1000, max(1, self.trade_limit))
        try:
            if since_ms is None and until_ms is None:
                return await self.api.fetch_my_trades(ccxt_symbol, limit=limit)

            end_bound = until_ms or (self._now_func() + 60 * 60 * 1000)
            start_bound = since_ms or max(0, end_bound - 7 * 24 * 60 * 60 * 1000)
            week_span = int(7 * 24 * 60 * 60 * 1000 * 0.999)
            params: Dict[str, object] = {}
            fetched: Dict[str, Dict[str, object]] = {}
            previous_key: Optional[Tuple[Tuple[str, object], ...]] = None
            fetch_count = 0

            cursor = int(start_bound)
            while cursor <= end_bound:
                window_end = int(min(end_bound, cursor + week_span))
                params["startTime"] = cursor
                params["endTime"] = window_end
                param_key = _check_pagination_progress(
                    previous_key,
                    params,
                    f"BinanceFetcher._fetch_symbol_trades({ccxt_symbol})",
                )
                if param_key is None:
                    break
                previous_key = param_key
                fetch_count += 1
                batch = await self.api.fetch_my_trades(
                    ccxt_symbol,
                    limit=limit,
                    params=dict(params),
                )
                if fetch_count > 1:
                    logger.info(
                        "BinanceFetcher._fetch_symbol_trades: fetch #%d symbol=%s start=%s end=%s size=%d",
                        fetch_count,
                        ccxt_symbol,
                        _format_ms(params["startTime"]),
                        _format_ms(params["endTime"]),
                        len(batch) if batch else 0,
                    )
                if not batch:
                    cursor = window_end + 1
                    continue
                for trade in batch:
                    trade_id = str(
                        trade.get("id")
                        or (trade.get("info") or {}).get("id")
                        or f"{trade.get('order')}-{trade.get('timestamp')}"
                    )
                    fetched[trade_id] = trade
                last_ts = int(
                    batch[-1].get("timestamp")
                    or (batch[-1].get("info") or {}).get("time")
                    or params["endTime"]
                )
                if last_ts >= end_bound or len(batch) < limit:
                    cursor = last_ts + 1
                    if cursor > end_bound:
                        break
                else:
                    cursor = last_ts + 1

            ordered = sorted(
                fetched.values(),
                key=lambda tr: int(tr.get("timestamp") or (tr.get("info") or {}).get("time") or 0),
            )
            return ordered
        except Exception as exc:  # pragma: no cover - depends on live API
            logger.error("BinanceFetcher._fetch_symbol_trades: error %s (%s)", ccxt_symbol, exc)
            return []

    def _normalize_income(self, entry: Dict[str, object]) -> Dict[str, object]:
        trade_id = entry.get("tradeId") or entry.get("id") or f"income-{entry.get('time')}"
        timestamp = int(entry.get("time") or entry.get("timestamp") or 0)
        raw_symbol = entry.get("symbol")
        ccxt_symbol = self._resolve_symbol(raw_symbol)
        pnl = float(entry.get("income") or entry.get("pnl") or 0.0)
        position_side = str(entry.get("positionSide") or entry.get("pside") or "unknown").lower()
        return {
            "id": str(trade_id),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": ccxt_symbol,
            "side": entry.get("side") or "",
            "qty": 0.0,
            "price": 0.0,
            "pnl": pnl,
            "fees": None,
            "pb_order_type": "",
            "position_side": position_side or "unknown",
            "client_order_id": entry.get("clientOrderId") or "",
        }

    def _normalize_trade(self, trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info") or {}
        trade_id = trade.get("id") or info.get("id")
        timestamp = int(trade.get("timestamp") or info.get("time") or info.get("T") or 0)
        pnl = float(info.get("realizedPnl") or trade.get("pnl") or 0.0)
        position_side = str(
            info.get("positionSide") or trade.get("position_side") or "unknown"
        ).lower()
        fees = trade.get("fees") or trade.get("fee")
        client_order_id = (
            trade.get("clientOrderId")
            or info.get("clientOrderId")
            or info.get("origClientOrderId")
            or info.get("clientOrderID")
            or ""
        )
        symbol = trade.get("symbol")
        if symbol and "/" not in symbol:
            symbol = self._resolve_symbol(symbol)
        order_id = (
            trade.get("order")
            or info.get("orderId")
            or info.get("origClientOrderId")
            or info.get("orderID")
        )
        return {
            "id": str(trade_id),
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": symbol or "",
            "side": trade.get("side") or "",
            "qty": float(trade.get("amount") or trade.get("qty") or 0.0),
            "price": float(trade.get("price") or 0.0),
            "pnl": pnl,
            "fees": fees,
            "pb_order_type": "",
            "position_side": position_side or "unknown",
            "client_order_id": client_order_id,
            "order_id": str(order_id) if order_id else "",
            "info": info,
            "raw": [{"source": "fetch_my_trades", "data": dict(trade)}],
        }

    def _collect_symbols(self, provider: Callable[[], Iterable[str]]) -> List[str]:
        try:
            items = provider() or []
        except Exception as exc:
            logger.warning("BinanceFetcher._collect_symbols: provider failed (%s)", exc)
            return []
        symbols: List[str] = []
        for raw in items:
            normalized = self._resolve_symbol(raw)
            if normalized:
                symbols.append(normalized)
        return symbols

    def _resolve_symbol(self, value: Optional[str]) -> str:
        if not value:
            return ""
        try:
            resolved = self._symbol_resolver(value)
            if resolved:
                return resolved
        except Exception as exc:
            logger.warning("BinanceFetcher._resolve_symbol: resolver failed for %s (%s)", value, exc)
        return str(value)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class FillEventsManager:
    """High-level interface around cached/fetched fill events."""

    def __init__(
        self,
        *,
        exchange: str,
        user: str,
        fetcher: BaseFetcher,
        cache_path: Path,
        rate_limit_coordinator: Optional[RateLimitCoordinator] = None,
    ) -> None:
        self.exchange = exchange
        self.user = user
        self.fetcher = fetcher
        self.cache = FillEventCache(cache_path)
        self.rate_limiter = rate_limit_coordinator or RateLimitCoordinator(exchange, user)
        self._events: List[FillEvent] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            cached = self.cache.load()
            filtered = []
            dropped = 0
            for ev in cached:
                if getattr(ev, "raw", None) is None:
                    dropped += 1
                    continue
                filtered.append(ev)
            self._events = sorted(filtered, key=lambda ev: ev.timestamp)
            logger.info(
                "FillEventsManager.ensure_loaded: loaded %d cached events (dropped %d without raw)",
                len(self._events),
                dropped,
            )
            self._loaded = True

    async def refresh(
        self,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> None:
        await self.ensure_loaded()
        requested_start = start_ms
        logger.info(
            "FillEventsManager.refresh: start=%s end=%s current_cache=%d (requested_start=%s)",
            _format_ms(start_ms),
            _format_ms(end_ms),
            len(self._events),
            _format_ms(requested_start),
        )
        detail_cache = {
            ev.id: (ev.client_order_id, ev.pb_order_type) for ev in self._events if ev.client_order_id
        }
        updated_map: Dict[str, FillEvent] = {ev.id: ev for ev in self._events}
        added_ids: set[str] = set()

        def handle_batch(batch: List[Dict[str, object]]) -> None:
            ensure_qty_signage(batch)
            days_touched: set[str] = set()
            for raw in batch:
                raw.setdefault("raw", [])
                try:
                    event = FillEvent.from_dict(raw)
                except ValueError as exc:
                    logger.warning(
                        "FillEventsManager.refresh: skipping malformed event %s (error=%s)",
                        raw.get("id"),
                        exc,
                    )
                    continue
                prev = updated_map.get(event.id)
                if prev is not None and event.timestamp < prev.timestamp:
                    continue
                updated_map[event.id] = event
                if prev is None:
                    added_ids.add(event.id)
                day = _day_key(event.timestamp)
                days_touched.add(day)
            if not days_touched:
                return
            day_payload = self._events_for_days(updated_map.values(), days_touched)
            self.cache.save_days(day_payload)
            days_list = sorted(days_touched)
            preview = ", ".join(days_list[:5])
            if len(days_list) > 5:
                preview += ", ..."
            logger.info(
                "FillEventsManager.refresh: persisted %d day files (%s)",
                len(day_payload),
                preview,
            )

        await self.fetcher.fetch(start_ms, end_ms, detail_cache, on_batch=handle_batch)

        self._events = sorted(updated_map.values(), key=lambda ev: ev.timestamp)

        # Update cache metadata with timestamps
        if self._events:
            self.cache.update_metadata_from_events(self._events)

            # If we successfully fetched data for a gap range, clear it
            if start_ms is not None and end_ms is not None and added_ids:
                self.cache.clear_gap(start_ms, end_ms)

        logger.info(
            "FillEventsManager.refresh: merged events=%d (added=%d)",
            len(self._events),
            len(added_ids),
        )

    async def refresh_latest(self, *, overlap: int = 20) -> None:
        """Fetch only the most recent fills, overlapping by `overlap` events."""
        await self.ensure_loaded()
        if not self._events:
            logger.info("FillEventsManager.refresh_latest: cache empty, falling back to full refresh")
        start_ms = None
        if self._events:
            idx = max(0, len(self._events) - overlap)
            start_ms = self._events[idx].timestamp
        await self.refresh(start_ms=start_ms, end_ms=None)

    async def refresh_range(
        self,
        start_ms: int,
        end_ms: Optional[int],
        *,
        gap_hours: float = 12.0,
        overlap: int = 20,
        force_refetch_gaps: bool = False,
    ) -> None:
        """Fill missing data between `start_ms` and `end_ms` using gap heuristics.

        Args:
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds (or None for now)
            gap_hours: Threshold for detecting gaps (default 12 hours)
            overlap: Number of events to overlap when fetching latest
            force_refetch_gaps: If True, retry even persistent gaps
        """
        await self.ensure_loaded()
        intervals: List[Tuple[int, int]] = []

        # Get known gaps from cache metadata
        known_gaps = self.cache.get_known_gaps()

        def is_in_persistent_gap(ts_start: int, ts_end: int) -> bool:
            """Check if interval is fully within a persistent (max retries) gap."""
            if force_refetch_gaps:
                return False
            for gap in known_gaps:
                if ts_start >= gap["start_ts"] and ts_end <= gap["end_ts"]:
                    if not self.cache.should_retry_gap(gap):
                        return True
            return False

        if not self._events:
            logger.info("FillEventsManager.refresh_range: cache empty, refreshing entire interval")
            await self.refresh(start_ms=start_ms, end_ms=end_ms)
            await self.refresh_latest(overlap=overlap)
            return

        events_sorted = self._events
        earliest = events_sorted[0].timestamp
        latest = events_sorted[-1].timestamp
        gap_ms = max(1, int(gap_hours * 60.0 * 60.0 * 1000.0))

        # Fetch older data before earliest cached if requested
        if start_ms < earliest:
            upper = earliest if end_ms is None else min(earliest, end_ms)
            if start_ms < upper and not is_in_persistent_gap(start_ms, upper):
                intervals.append((start_ms, upper))

        # Detect large gaps in cached data
        prev_ts = earliest
        for ev in events_sorted[1:]:
            cur_ts = ev.timestamp
            if end_ms is not None and cur_ts > end_ms:
                break
            if cur_ts - prev_ts >= gap_ms:
                gap_start = max(prev_ts, start_ms)
                gap_end = end_ms if end_ms is not None else cur_ts
                if gap_start < gap_end:
                    if is_in_persistent_gap(gap_start, gap_end):
                        logger.debug(
                            "FillEventsManager.refresh_range: skipping persistent gap %s → %s",
                            _format_ms(gap_start),
                            _format_ms(gap_end),
                        )
                    else:
                        intervals.append((gap_start, gap_end))
                        # Record as potential gap for tracking
                        self.cache.add_known_gap(
                            gap_start,
                            gap_end,
                            reason=GAP_REASON_AUTO,
                            confidence=GAP_CONFIDENCE_SUSPICIOUS,
                        )
                break
            prev_ts = cur_ts

        # Fetch newer data after latest cached if requested (if not already covered)
        if end_ms is not None and end_ms > latest and (not intervals or intervals[-1][1] != end_ms):
            lower = max(latest, start_ms)
            if lower < end_ms and not is_in_persistent_gap(lower, end_ms):
                intervals.append((lower, end_ms))

        merged = self._merge_intervals(intervals)
        if merged:
            logger.info(
                "FillEventsManager.refresh_range: refreshing %d intervals: %s",
                len(merged),
                ", ".join(f"{_format_ms(start)} → {_format_ms(end)}" for start, end in merged),
            )
        else:
            logger.info("FillEventsManager.refresh_range: no gaps detected in requested interval")

        for start, end in merged:
            await self.refresh(start_ms=start, end_ms=end)

        await self.refresh_latest(overlap=overlap)

    def get_events(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[FillEvent]:
        events = self._events
        if start_ms is not None:
            events = [ev for ev in events if ev.timestamp >= start_ms]
        if end_ms is not None:
            events = [ev for ev in events if ev.timestamp <= end_ms]
        if symbol:
            events = [ev for ev in events if ev.symbol == symbol]
        # Annotate positions on a copy so cache on disk remains untouched
        payload = [ev.to_dict() for ev in events]
        ensure_qty_signage(payload)
        annotate_positions_inplace(payload, recompute_pnl=(self.exchange.lower() == "kucoin"))
        return [FillEvent.from_dict(ev) for ev in payload]

    def get_pnl_sum(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> float:
        events = self.get_events(start_ms, end_ms, symbol)
        return float(sum(ev.pnl for ev in events))

    def get_pnl_cumsum(
        self,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[Tuple[int, float]]:
        events = self.get_events(start_ms, end_ms, symbol)
        total = 0.0
        result = []
        for ev in events:
            total += ev.pnl
            result.append((ev.timestamp, total))
        return result

    def get_last_timestamp(self, symbol: Optional[str] = None) -> Optional[int]:
        events = self._events
        if symbol:
            events = [ev for ev in events if ev.symbol == symbol]
        if not events:
            return None
        return max(ev.timestamp for ev in events)

    def reconstruct_positions(
        self, current_positions: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        positions: Dict[str, float] = dict(current_positions or {})
        for ev in self._events:
            key = f"{ev.symbol}:{ev.position_side}"
            positions[key] = positions.get(key, 0.0) + ev.qty
        return positions

    def reconstruct_equity_curve(self, starting_equity: float = 0.0) -> List[Tuple[int, float]]:
        total = starting_equity
        points: List[Tuple[int, float]] = []
        for ev in self._events:
            total += ev.pnl
            points.append((ev.timestamp, total))
        return points

    def get_coverage_summary(self) -> Dict[str, object]:
        """Return a summary of cache coverage and known gaps."""
        summary = self.cache.get_coverage_summary()
        summary["events_count"] = len(self._events)
        summary["exchange"] = self.exchange
        summary["user"] = self.user
        if self._events:
            summary["first_event"] = _format_ms(self._events[0].timestamp)
            summary["last_event"] = _format_ms(self._events[-1].timestamp)
            # Count unique symbols
            symbols = set(ev.symbol for ev in self._events)
            summary["symbols_count"] = len(symbols)
            summary["symbols"] = sorted(symbols)
        return summary

    @staticmethod
    def _events_for_days(
        events: Iterable[FillEvent], days: Iterable[str]
    ) -> Dict[str, List[FillEvent]]:
        target = {day: [] for day in days}
        for event in events:
            day = _day_key(event.timestamp)
            if day in target:
                target[day].append(event)
        for day_events in target.values():
            day_events.sort(key=lambda ev: ev.timestamp)
        return target

    @staticmethod
    def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        cleaned = [(int(start), int(end)) for start, end in intervals if end > start]
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = []
        cur_start, cur_end = cleaned[0]
        for start, end in cleaned[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((cur_start, cur_end))
        return merged


class BybitFetcher(BaseFetcher):
    """Fetches fill events from Bybit using trades + positions history."""

    def __init__(
        self,
        api,
        *,
        category: str = "linear",
        trade_limit: int = 100,
        position_limit: int = 100,
        overlap_days: float = 3.0,
        max_span_days: float = 6.5,
    ) -> None:
        self.api = api
        self.category = category
        self.trade_limit = max(1, min(trade_limit, 100))
        self.position_limit = max(1, min(position_limit, 100))
        self._default_span_ms = int(overlap_days * 24 * 60 * 60 * 1000)
        self._max_span_ms = int(max_span_days * 24 * 60 * 60 * 1000)

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        end_ms = until_ms or (self._now_ms() + 60 * 60 * 1000)
        start_ms = since_ms or max(0, end_ms - self._default_span_ms)

        trades = await self._fetch_my_trades(start_ms, end_ms)
        positions = await self._fetch_positions_history(start_ms, end_ms)

        events = self._combine(trades, positions, detail_cache)
        events = [
            ev
            for ev in events
            if (since_ms is None or ev["timestamp"] >= since_ms)
            and (until_ms is None or ev["timestamp"] <= until_ms)
        ]
        events.sort(key=lambda ev: ev["timestamp"])
        events = _coalesce_events(events)

        if on_batch and events:
            day_map = defaultdict(list)
            for ev in events:
                day_map[_day_key(ev["timestamp"])].append(ev)
            for day in sorted(day_map):
                on_batch(day_map[day])

        logger.info(
            "BybitFetcher.fetch: done (events=%d, trades=%d, positions=%d)",
            len(events),
            len(trades),
            len(positions),
        )
        return events

    async def _fetch_my_trades(self, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        params = {
            "type": "swap",
            "subType": self.category,
            "limit": self.trade_limit,
            "endTime": int(end_ms),
        }
        results: List[Dict[str, object]] = []
        max_fetches = 200
        fetch_count = 0
        prev_params = None
        while True:
            new_key = _check_pagination_progress(
                prev_params,
                params,
                "BybitFetcher._fetch_my_trades",
            )
            if new_key is None:
                break
            prev_params = new_key
            fetch_count += 1
            batch = await self.api.fetch_my_trades(params=params)
            if fetch_count > 1:
                logger.info(
                    "BybitFetcher._fetch_my_trades: fetch #%d endTime=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("endTime")),
                    len(batch) if batch else 0,
                )
            if not batch:
                break
            batch.sort(key=lambda x: x["timestamp"])
            results.extend(batch)
            if len(batch) < self.trade_limit:
                if params["endTime"] - start_ms < self._max_span_ms:
                    break
                params["endTime"] = max(start_ms, params["endTime"] - self._max_span_ms)
                continue
            first_ts = batch[0]["timestamp"]
            if first_ts <= start_ms:
                break
            if params["endTime"] == first_ts:
                break
            params["endTime"] = int(first_ts)
            if fetch_count >= max_fetches:
                logger.warning("BybitFetcher._fetch_my_trades: max fetches reached")
                break
        ordered = sorted(
            results,
            key=lambda x: int(x.get("info", {}).get("updatedTime") or x.get("timestamp") or 0),
        )
        return ordered

    async def _fetch_positions_history(self, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        params = {
            "limit": self.position_limit,
            "endTime": int(end_ms),
        }
        results: List[Dict[str, object]] = []
        max_fetches = 200
        fetch_count = 0
        prev_params = None
        while True:
            new_key = _check_pagination_progress(
                prev_params,
                params,
                "BybitFetcher._fetch_positions_history",
            )
            if new_key is None:
                break
            prev_params = new_key
            fetch_count += 1
            batch = await self.api.fetch_positions_history(params=params)
            if fetch_count > 1:
                logger.info(
                    "BybitFetcher._fetch_positions_history: fetch #%d endTime=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("endTime")),
                    len(batch) if batch else 0,
                )
            if not batch:
                break
            batch.sort(key=lambda x: x["timestamp"])
            results.extend(batch)
            if len(batch) < self.position_limit:
                if params["endTime"] - start_ms < self._max_span_ms:
                    break
                params["endTime"] = max(start_ms, params["endTime"] - self._max_span_ms)
                continue
            first_ts = batch[0]["timestamp"]
            if first_ts <= start_ms:
                break
            if params["endTime"] == first_ts:
                break
            params["endTime"] = int(first_ts)
            if fetch_count >= max_fetches:
                logger.warning("BybitFetcher._fetch_positions_history: max fetches reached")
                break
        return results

    def _combine(
        self,
        trades: List[Dict[str, object]],
        positions: List[Dict[str, object]],
        detail_cache: Dict[str, Tuple[str, str]],
    ) -> List[Dict[str, object]]:
        pnls: Dict[str, float] = defaultdict(float)
        symbol_realized: Dict[str, float] = defaultdict(float)
        symbol_closed_qty: Dict[str, float] = defaultdict(float)
        for entry in positions:
            order_id = str(entry.get("info", {}).get("orderId", entry.get("orderId", "")))
            if not order_id:
                continue
            pnl = float(entry.get("realizedPnl") or entry.get("info", {}).get("closedPnl") or 0.0)
            pnls[order_id] += pnl
            symbol = entry.get("symbol")
            if symbol:
                symbol_realized[symbol] += pnl
                closed = float(
                    entry.get("info", {}).get("closedSize") or entry.get("contracts") or 0.0
                )
                symbol_closed_qty[symbol] += closed

        order_total_qty: Dict[str, float] = defaultdict(float)
        symbol_order_qty: Dict[str, float] = defaultdict(float)
        symbol_unknown_trade_qty: Dict[str, float] = defaultdict(float)
        for trade in trades:
            order_id = str(trade.get("info", {}).get("orderId", trade.get("order")))
            qty = abs(float(trade.get("amount") or trade.get("info", {}).get("execQty") or 0.0))
            symbol = trade.get("symbol") or trade.get("info", {}).get("symbol")
            if order_id and order_id in pnls:
                order_total_qty[order_id] += qty
                if symbol:
                    symbol_order_qty[symbol] += qty
            elif symbol:
                symbol_unknown_trade_qty[symbol] += qty

        order_remaining_qty = dict(order_total_qty)
        order_remaining_pnl = dict(pnls)
        symbol_remaining_pnl = dict(symbol_realized)
        symbol_remaining_qty: Dict[str, float] = {}
        for sym, closed in symbol_closed_qty.items():
            remaining = max(closed - symbol_order_qty.get(sym, 0.0), 0.0)
            symbol_remaining_qty[sym] = remaining
        for sym, qty in symbol_unknown_trade_qty.items():
            symbol_remaining_qty[sym] = symbol_remaining_qty.get(sym, 0.0) + qty

        events: List[Dict[str, object]] = []
        for trade in trades:
            event = self._normalize_trade(trade)
            order_id = event.get("order_id")
            cache_entry = detail_cache.get(event["id"])
            allocated = False
            if order_id and order_id in order_remaining_pnl and order_remaining_qty[order_id] > 0:
                remaining_qty = order_remaining_qty[order_id]
                remaining_pnl = order_remaining_pnl[order_id]
                qty = abs(event["qty"])
                if remaining_qty <= qty * 1.0000001:
                    event["pnl"] = remaining_pnl
                else:
                    event["pnl"] = remaining_pnl * (qty / remaining_qty)
                order_remaining_qty[order_id] = max(0.0, remaining_qty - qty)
                order_remaining_pnl[order_id] = remaining_pnl - event["pnl"]
                symbol_remaining_pnl[event["symbol"]] = (
                    symbol_remaining_pnl.get(event["symbol"], 0.0) - event["pnl"]
                )
                symbol_remaining_qty[event["symbol"]] = max(
                    0.0, symbol_remaining_qty.get(event["symbol"], 0.0) - qty
                )
                allocated = True
            if cache_entry:
                event["client_order_id"], event["pb_order_type"] = cache_entry
                if not event["pb_order_type"]:
                    event["pb_order_type"] = "unknown"
            elif event["client_order_id"]:
                pb_type = custom_id_to_snake(event["client_order_id"])
                event["pb_order_type"] = pb_type or "unknown"
            else:
                event["pb_order_type"] = "unknown"
            if (
                event["pb_order_type"] == "unknown"
                and not allocated
                and abs(event.get("pnl", 0.0)) < 1e-12
            ):
                symbol = event["symbol"]
                remaining_symbol_qty = symbol_remaining_qty.get(symbol, 0.0)
                remaining_symbol_pnl = symbol_remaining_pnl.get(symbol, 0.0)
                qty = abs(event["qty"])
                if remaining_symbol_qty > 0:
                    if remaining_symbol_qty <= qty * 1.0000001:
                        event["pnl"] = remaining_symbol_pnl
                    else:
                        event["pnl"] = remaining_symbol_pnl * (qty / remaining_symbol_qty)
                    symbol_remaining_qty[symbol] = max(0.0, remaining_symbol_qty - qty)
                    symbol_remaining_pnl[symbol] = remaining_symbol_pnl - event["pnl"]
                else:
                    event["pnl"] = remaining_symbol_pnl
            events.append(event)

        remaining_orders = [k for k, v in order_remaining_pnl.items() if abs(v) > 1e-6]
        if remaining_orders:
            logger.warning(
                "BybitFetcher._combine: residual PnL for orders %s (values=%s)",
                remaining_orders,
                [order_remaining_pnl[k] for k in remaining_orders],
            )
        remaining_symbols = [k for k, v in symbol_remaining_pnl.items() if abs(v) > 1e-6]
        if remaining_symbols:
            logger.debug(
                "BybitFetcher._combine: remaining symbol-level PnL after distribution %s",
                {k: symbol_remaining_pnl[k] for k in remaining_symbols},
            )
        return events

    @staticmethod
    def _normalize_trade(trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info", {})
        order_id = str(info.get("orderId", trade.get("order")))
        trade_id = str(trade.get("id") or info.get("execId") or order_id)
        timestamp = int(trade.get("timestamp") or info.get("execTime", 0))
        qty = float(trade.get("amount") or info.get("execQty", 0.0))
        side = str(trade.get("side") or info.get("side", "")).lower()
        price = float(trade.get("price") or info.get("execPrice", 0.0))
        closed_size = float(info.get("closedSize") or info.get("closeSize") or 0.0)
        position_side = BybitFetcher._determine_position_side(side, closed_size)
        pnl = float(trade.get("pnl") or 0.0)
        client_order_id = info.get("orderLinkId") or trade.get("clientOrderId")
        fee = trade.get("fee")
        symbol = trade.get("symbol") or info.get("symbol")

        return {
            "id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp),
            "symbol": symbol,
            "side": side,
            "qty": abs(qty),
            "price": price,
            "pnl": pnl,
            "fees": fee,
            "pb_order_type": "",
            "position_side": position_side,
            "client_order_id": client_order_id or "",
            "raw": [{"source": "fetch_my_trades", "data": dict(trade)}],
        }

    @staticmethod
    def _determine_position_side(side: str, closed_size: float) -> str:
        if side == "buy":
            return "short" if closed_size else "long"
        if side == "sell":
            return "long" if closed_size else "short"
        return "long"

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


class HyperliquidFetcher(BaseFetcher):
    """Fetches fill events via ccxt.fetch_my_trades for Hyperliquid."""

    def __init__(
        self,
        api,
        *,
        trade_limit: int = 500,
        symbol_resolver: Optional[Callable[[Optional[str]], str]] = None,
    ) -> None:
        self.api = api
        self.trade_limit = max(1, trade_limit)
        self._symbol_resolver = symbol_resolver

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {"limit": self.trade_limit}
        if since_ms is not None:
            params["since"] = int(since_ms)

        collected: Dict[str, Dict[str, object]] = {}
        max_fetches = 200
        fetch_count = 0

        prev_params = None
        while True:
            check_params = dict(params)
            check_params["_page"] = fetch_count
            new_key = _check_pagination_progress(
                prev_params,
                check_params,
                "HyperliquidFetcher.fetch",
            )
            if new_key is None:
                break
            prev_params = new_key
            try:
                trades = await self.api.fetch_my_trades(params=params)
            except RateLimitExceeded as exc:
                logger.debug(
                    "HyperliquidFetcher.fetch: rate limit exceeded, sleeping briefly (%s)",
                    exc,
                )
                await asyncio.sleep(1.0)
                continue
            fetch_count += 1
            if fetch_count > 1:
                logger.info(
                    "HyperliquidFetcher.fetch: fetch #%d since=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("since")),
                    len(trades) if trades else 0,
                )
            if not trades:
                break
            before_count = len(collected)
            for trade in trades:
                event = self._normalize_trade(trade)
                ts = event["timestamp"]
                if since_ms is not None and ts < since_ms:
                    continue
                if until_ms is not None and ts > until_ms:
                    continue
                collected[event["id"]] = event
            added = len(collected) - before_count
            if len(trades) < self.trade_limit:
                break
            last_ts = int(
                trades[-1].get("timestamp")
                or trades[-1].get("info", {}).get("time")
                or trades[-1].get("info", {}).get("updatedTime")
                or 0
            )
            if last_ts <= 0:
                break
            if until_ms is not None and last_ts >= until_ms:
                break
            if added <= 0:
                logger.debug(
                    "HyperliquidFetcher.fetch: no new trades added on page (last_ts=%s), stopping",
                    last_ts,
                )
                break
            params["since"] = last_ts
            if fetch_count >= max_fetches:
                logger.warning(
                    "HyperliquidFetcher.fetch: reached maximum pagination depth (%d)",
                    max_fetches,
                )
                break

        events = sorted(collected.values(), key=lambda ev: ev["timestamp"])
        events = _coalesce_events(events)
        annotate_positions_inplace(events)

        for event in events:
            cache_entry = detail_cache.get(event["id"])
            if cache_entry:
                event["client_order_id"], event["pb_order_type"] = cache_entry
            elif event["client_order_id"]:
                event["pb_order_type"] = custom_id_to_snake(event["client_order_id"])
            else:
                event["pb_order_type"] = "unknown"
            if not event["pb_order_type"]:
                event["pb_order_type"] = "unknown"

        if on_batch and events:
            on_batch(events)

        return events

    @staticmethod
    def _normalize_trade(trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info", {}) or {}
        trade_id = str(trade.get("id") or info.get("hash") or info.get("tid") or "")
        order_id = str(trade.get("order") or info.get("oid") or "")
        timestamp = int(
            trade.get("timestamp")
            or info.get("time")
            or info.get("tradeTime")
            or info.get("updatedTime")
            or 0
        )
        symbol_raw = trade.get("symbol") or info.get("symbol") or info.get("coin")
        side = str(trade.get("side") or info.get("side") or "").lower()
        qty = abs(float(trade.get("amount") or info.get("sz") or 0.0))
        price = float(trade.get("price") or info.get("px") or 0.0)
        pnl = float(trade.get("pnl") or info.get("closedPnl") or 0.0)
        fee = trade.get("fee") or {"currency": info.get("feeToken"), "cost": info.get("fee")}
        client_order_id = trade.get("clientOrderId") or info.get("cloid") or info.get("clOrdId") or ""
        direction = str(info.get("dir", "")).lower()
        if "short" in direction:
            position_side = "short"
        elif "long" in direction:
            position_side = "long"
        else:
            position_side = "long" if side == "buy" else "short"
        return {
            "id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp) if timestamp else "",
            "symbol": str(symbol_raw or ""),
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "fees": fee,
            "pb_order_type": "",
            "position_side": position_side,
            "client_order_id": str(client_order_id or ""),
            "raw": [{"source": "fetch_my_trades", "data": trade}],
            "c_mult": float(info.get("contractMultiplier") or info.get("multiplier") or 1.0),
        }


class GateioFetcher(BaseFetcher):
    """Fetches fill events via ccxt.fetch_closed_orders for Gate.io."""

    def __init__(
        self,
        api,
        *,
        trade_limit: int = 100,
    ) -> None:
        self.api = api
        self.trade_limit = max(1, trade_limit)

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        params: Dict[str, object] = {
            "status": "finished",
            "limit": self.trade_limit,
            "offset": 0,
        }

        collected: Dict[str, Dict[str, object]] = {}
        max_fetches = 400
        fetch_count = 0

        while True:
            new_key = _check_pagination_progress(
                None,
                dict(params, _page=fetch_count),
                "GateioFetcher.fetch",
            )
            if new_key is None:
                break
            fetch_count += 1
            try:
                orders = await self.api.fetch_closed_orders(params=params)
            except RateLimitExceeded as exc:  # pragma: no cover - live API
                logger.debug("GateioFetcher.fetch: rate-limited (%s); sleeping", exc)
                await asyncio.sleep(1.0)
                continue
            if fetch_count > 1:
                logger.info(
                    "GateioFetcher.fetch: fetch #%d offset=%s size=%d",
                    fetch_count,
                    params.get("offset"),
                    len(orders) if orders else 0,
                )
            if not orders:
                break
            for order in orders:
                event = self._normalize_order(order)
                ts = event["timestamp"]
                if since_ms is not None and ts < since_ms:
                    continue
                if until_ms is not None and ts > until_ms:
                    continue
                collected[event["id"]] = event
            if on_batch:
                on_batch(list(collected.values()))
            if len(orders) < self.trade_limit:
                break
            if since_ms is not None:
                oldest = min(ev["timestamp"] for ev in collected.values()) if collected else None
                if oldest is not None and oldest <= since_ms:
                    break
            params["offset"] = params.get("offset", 0) + self.trade_limit
            if fetch_count >= max_fetches:
                logger.warning("GateioFetcher.fetch: reached pagination cap (%d)", max_fetches)
                break

        ordered = sorted(collected.values(), key=lambda ev: ev["timestamp"])
        return ordered

    def _normalize_order(self, order: Dict[str, object]) -> Dict[str, object]:
        info = order.get("info", {}) or {}
        order_id = str(order.get("id") or info.get("id") or info.get("order_id") or "")
        ts_raw = (
            order.get("lastTradeTimestamp")
            or info.get("update_time_ms")
            or info.get("update_time")
            or order.get("timestamp")
            or info.get("create_time_ms")
            or info.get("create_time")
            or 0
        )
        try:
            timestamp = int(ensure_millis(float(ts_raw)))
        except Exception:
            try:
                timestamp = int(float(ts_raw))
            except Exception:
                timestamp = 0
        symbol = str(order.get("symbol") or info.get("symbol") or info.get("contract") or "")
        side = str(order.get("side") or info.get("side") or "").lower()
        qty = abs(float(order.get("amount") or info.get("size") or info.get("amount") or 0.0))
        price = float(order.get("price") or info.get("price") or 0.0)
        pnl = float(info.get("pnl") or 0.0)
        pnl_margin = float(info.get("pnl_margin") or 0.0)
        reduce_only = bool(order.get("reduce_only") or info.get("reduce_only") or False)
        client_order_id = (
            order.get("clientOrderId") or info.get("text") or info.get("client_order_id") or ""
        )
        pb_type = custom_id_to_snake(str(client_order_id)) if client_order_id else "unknown"
        is_close = abs(pnl) > 0.0 or abs(pnl_margin) > 0.0 or reduce_only
        position_side = self._determine_position_side(side, is_close)

        return {
            "id": order_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp) if timestamp else "",
            "symbol": str(symbol or ""),
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "fees": None,
            "pb_order_type": pb_type or "unknown",
            "position_side": position_side,
            "client_order_id": str(client_order_id or ""),
            "raw": [{"source": "fetch_closed_orders", "data": dict(order)}],
        }

    @staticmethod
    def _determine_position_side(side: str, is_close: bool) -> str:
        side = side.lower()
        if is_close:
            if side == "buy":
                return "short"
            if side == "sell":
                return "long"
        else:
            if side == "buy":
                return "long"
            if side == "sell":
                return "short"
        return "long"


class KucoinFetcher(BaseFetcher):
    """Fetches fill events for Kucoin by combining trade and position history."""

    def __init__(
        self, api, *, trade_limit: int = 1000, now_func: Optional[Callable[[], int]] = None
    ) -> None:
        self.api = api
        self.trade_limit = max(1, trade_limit)
        self._symbol_resolver = None
        self._now_func = now_func or (lambda: int(datetime.now(tz=timezone.utc).timestamp() * 1000))

    async def fetch(
        self,
        since_ms: Optional[int],
        until_ms: Optional[int],
        detail_cache: Dict[str, Tuple[str, str]],
        on_batch: Optional[Callable[[List[Dict[str, object]]], None]] = None,
    ) -> List[Dict[str, object]]:
        trades = await self._fetch_trades(since_ms, until_ms)
        if not trades:
            return []

        # Compute local realized PnL from trades (gross), subtract fees when available
        local_pnls, _ = compute_realized_pnls_from_trades(trades)

        closes = [
            t
            for t in trades
            if (t["side"] == "sell" and t["position_side"] == "long")
            or (t["side"] == "buy" and t["position_side"] == "short")
        ]
        events: Dict[str, Dict[str, object]] = {}
        for t in trades:
            ev = dict(t)
            fee_cost = _fee_cost(ev.get("fees"))
            ev["pnl"] = local_pnls.get(ev["id"], 0.0) - fee_cost
            events[ev["id"]] = ev

        if closes:
            ph = await self._fetch_positions_history(
                start_ms=closes[0]["timestamp"] - 60_000,
                end_ms=closes[-1]["timestamp"] + 60_000,
            )
            self._match_pnls(closes, ph, events)
            self._log_discrepancies(local_pnls, ph)

        ordered = sorted(events.values(), key=lambda ev: ev["timestamp"])
        await self._enrich_with_order_details_bulk(ordered, detail_cache)
        if on_batch and ordered:
            on_batch(ordered)
        return ordered

    async def _fetch_trades(
        self, since_ms: Optional[int], until_ms: Optional[int]
    ) -> List[Dict[str, object]]:
        now_ms = self._now_func()
        until_ts = int(until_ms) if until_ms is not None else now_ms + 3_600_000
        since_ts = int(since_ms) if since_ms is not None else until_ts - 24 * 60 * 60 * 1000
        buffer_ms = int(24 * 60 * 60 * 1000 * 0.99)
        limit = min(self.trade_limit, 1000)

        collected: Dict[str, Dict[str, object]] = {}
        max_fetches = 400
        start_at = since_ts
        prev_params = None
        fetch_count = 0

        while start_at < until_ts and fetch_count < max_fetches:
            fetch_count += 1
            end_at = min(start_at + buffer_ms, until_ts)
            params: Dict[str, object] = {
                "startAt": int(start_at),
                "endAt": int(end_at),
                "limit": limit,
            }
            key = _check_pagination_progress(prev_params, dict(params), "KucoinFetcher._fetch_trades")
            if key is None:
                break
            prev_params = key
            batch = await self.api.fetch_my_trades(params=params)
            if fetch_count > 1:
                logger.info(
                    "KucoinFetcher._fetch_trades: fetch #%d startAt=%s endAt=%s size=%d",
                    fetch_count,
                    _format_ms(params["startAt"]),
                    _format_ms(params["endAt"]),
                    len(batch) if batch else 0,
                )
            if not batch:
                start_at += buffer_ms
                continue

            batch_sorted = sorted(batch, key=lambda x: x.get("timestamp", 0))
            for trade in batch_sorted:
                event = self._normalize_trade(trade)
                ts = event["timestamp"]
                if ts < since_ts or ts > until_ts:
                    continue
                key = (event.get("id") or "", event.get("order_id") or "")
                collected[key] = event

            last_ts = int(batch_sorted[-1].get("timestamp", start_at))
            if last_ts <= start_at:
                start_at = start_at + buffer_ms
            else:
                start_at = last_ts + 1

        if fetch_count >= max_fetches:
            logger.warning("KucoinFetcher._fetch_trades: reached pagination cap (%d)", max_fetches)

        return sorted(collected.values(), key=lambda ev: ev["timestamp"])

    async def _fetch_positions_history(self, start_ms: int, end_ms: int) -> List[Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        max_fetches = 400
        fetch_count = 0
        buffer_ms = int(24 * 60 * 60 * 1000 * 0.99)
        limit = 200
        now_ms = self._now_func()
        until_ts = int(end_ms) if end_ms is not None else now_ms + 3_600_000
        since_ts = int(start_ms) if start_ms is not None else until_ts - 24 * 60 * 60 * 1000

        start_at = since_ts
        prev_params = None
        while start_at < until_ts and fetch_count < max_fetches:
            end_at = min(start_at + buffer_ms, until_ts)
            params: Dict[str, object] = {"from": int(start_at), "to": int(end_at), "limit": limit}
            key = _check_pagination_progress(
                prev_params, dict(params), "KucoinFetcher._fetch_positions_history"
            )
            if key is None:
                break
            prev_params = key
            fetch_count += 1
            batch = await self.api.fetch_positions_history(params=params)
            if fetch_count > 1:
                logger.info(
                    "KucoinFetcher._fetch_positions_history: fetch #%d from=%s to=%s size=%d",
                    fetch_count,
                    _format_ms(params.get("from")),
                    _format_ms(params.get("to")),
                    len(batch) if batch else 0,
                )
            if not batch:
                start_at += buffer_ms
                continue
            batch_sorted = sorted(batch, key=lambda x: x.get("lastUpdateTimestamp", 0))
            for pos in batch_sorted:
                close_id = str(pos.get("info", {}).get("closeId") or pos.get("id") or "")
                results[close_id] = pos
            last_ts = int(batch_sorted[-1].get("lastUpdateTimestamp", end_at))
            if last_ts <= start_at:
                start_at += buffer_ms
            else:
                start_at = last_ts + 1

        if fetch_count >= max_fetches:
            logger.warning(
                "KucoinFetcher._fetch_positions_history: reached pagination cap (%d)", max_fetches
            )

        return sorted(results.values(), key=lambda x: x.get("lastUpdateTimestamp", 0))

    def _match_pnls(
        self,
        closes: List[Dict[str, object]],
        positions: List[Dict[str, object]],
        events: Dict[str, Dict[str, object]],
    ) -> None:
        closes_by_symbol: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for c in closes:
            closes_by_symbol[c["symbol"]].append(c)
        positions_by_symbol: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for p in positions:
            positions_by_symbol[p.get("symbol", "")].append(p)

        seen_trade_ids: set[str] = set()
        for symbol, pos_list in positions_by_symbol.items():
            if symbol not in closes_by_symbol:
                continue
            for p in pos_list:
                candidates = sorted(
                    [c for c in closes_by_symbol[symbol] if c["id"] not in seen_trade_ids],
                    key=lambda c: abs(c["timestamp"] - p.get("lastUpdateTimestamp", 0)),
                )
                if not candidates:
                    continue
                best = candidates[0]
                events[best["id"]]["pnl"] = float(p.get("realizedPnl", 0.0))
                seen_trade_ids.add(best["id"])

    @staticmethod
    def _log_discrepancies(local_pnls: Dict[str, float], positions: List[Dict[str, object]]) -> None:
        if not positions or not local_pnls:
            return
        # Aggregate by symbol for a rough reconciliation
        pos_sum: Dict[str, float] = defaultdict(float)
        for p in positions:
            sym = p.get("symbol") or p.get("info", {}).get("symbol") or ""
            if not sym:
                continue
            try:
                pos_sum[sym] += float(p.get("realizedPnl", 0.0))
            except Exception:
                continue
        if not pos_sum:
            return
        # Local aggregate by symbol inferred from trade ids is not available here; report global sums
        local_total = sum(local_pnls.values())
        remote_total = sum(pos_sum.values())
        if abs(local_total - remote_total) > max(1e-8, 0.05 * (abs(remote_total) + 1e-8)):
            logger.warning(
                "KucoinFetcher: local PnL sum %.6f differs from positions_history sum %.6f",
                local_total,
                remote_total,
            )

    @staticmethod
    def _normalize_trade(trade: Dict[str, object]) -> Dict[str, object]:
        info = trade.get("info", {}) or {}
        trade_id = str(trade.get("id") or info.get("tradeId") or info.get("id") or "")
        order_id = str(trade.get("order") or info.get("orderId") or "")
        ts_raw = (
            info.get("tradeTime")
            or trade.get("timestamp")
            or info.get("createdAt")
            or info.get("updatedTime")
            or 0
        )
        try:
            timestamp = int(ensure_millis(float(ts_raw)))
        except Exception:
            try:
                timestamp = int(float(ts_raw))
            except Exception:
                timestamp = 0
        symbol = str(trade.get("symbol") or "")
        side = str(trade.get("side") or info.get("side") or "").lower()
        qty = abs(float(trade.get("amount") or info.get("size") or info.get("amount") or 0.0))
        price = float(trade.get("price") or info.get("price") or 0.0)
        fee = trade.get("fee")
        reduce_only = bool(trade.get("reduceOnly") or info.get("closeOrder") or False)
        close_fee_pay = float(info.get("closeFeePay") or 0.0)
        position_side = KucoinFetcher._determine_position_side(side, reduce_only, close_fee_pay)

        return {
            "id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "datetime": ts_to_date(timestamp) if timestamp else "",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": 0.0,
            "fees": fee,
            "pb_order_type": "",
            "position_side": position_side,
            "client_order_id": str(trade.get("clientOrderId") or info.get("clientOid") or ""),
            "raw": [{"source": "fetch_my_trades", "data": dict(trade)}],
        }

    @staticmethod
    def _determine_position_side(side: str, reduce_only: bool, close_fee_pay: float) -> str:
        side = side.lower()
        if side == "buy":
            return "short" if close_fee_pay != 0.0 or reduce_only else "long"
        if side == "sell":
            return "long" if close_fee_pay != 0.0 or reduce_only else "short"
        return "long"

    async def _enrich_with_order_details_bulk(
        self, events: List[Dict[str, object]], detail_cache: Dict[str, Tuple[str, str]]
    ) -> None:
        if events is None:
            return
        detail_cache = detail_cache or {}
        pending: List[Tuple[Dict[str, object], str, str, str]] = []  # (ev, ev_id, order_id, symbol)
        for ev in events:
            cached = detail_cache.get(ev.get("id"))
            if cached:
                ev["client_order_id"], ev["pb_order_type"] = cached
            has_client = bool(ev.get("client_order_id"))
            has_type = bool(ev.get("pb_order_type")) and ev["pb_order_type"] != "unknown"
            if has_client and has_type:
                continue
            order_id = ev.get("order_id")
            symbol = ev.get("symbol")
            if not order_id:
                ev.setdefault("pb_order_type", "unknown")
                continue
            pending.append((ev, ev.get("id"), order_id, symbol))

        if pending:
            # Limit concurrency to avoid overwhelming the API
            sem = asyncio.Semaphore(8)
            total = len(pending)
            completed = 0
            last_log_time = time.time()
            log_interval = 5.0  # Log progress every 5 seconds

            async def throttled_fetch(order_id: str, symbol: str) -> Optional[Tuple[str, str]]:
                nonlocal completed, last_log_time
                async with sem:
                    result = await self._enrich_with_order_details(order_id, symbol)
                    completed += 1
                    now = time.time()
                    if total > 50 and (now - last_log_time >= log_interval):
                        last_log_time = now
                        pct = int(100 * completed / total)
                        logger.info(
                            "KucoinFetcher: enriching order details %d/%d (%d%%)",
                            completed,
                            total,
                            pct,
                        )
                    return result

            tasks = [throttled_fetch(order_id, symbol) for _, _, order_id, symbol in pending]
            if total > 50:
                logger.info(
                    "KucoinFetcher: enriching %d events with order details (concurrency=8)...",
                    total,
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if total > 50:
                logger.info("KucoinFetcher: enrichment complete (%d events)", total)
            for (ev, ev_id, _, _), res in zip(pending, results):
                if isinstance(res, Exception) or res is None:
                    ev.setdefault("pb_order_type", ev.get("pb_order_type") or "unknown")
                    continue
                client_oid, pb_type = res
                ev["client_order_id"] = client_oid or ev.get("client_order_id") or ""
                ev["pb_order_type"] = pb_type or "unknown"
                if ev_id:
                    detail_cache[ev_id] = (ev["client_order_id"], ev["pb_order_type"])
        for ev in events:
            if not ev.get("pb_order_type"):
                ev["pb_order_type"] = "unknown"

    async def _enrich_with_order_details(
        self, order_id: Optional[str], symbol: Optional[str]
    ) -> Optional[Tuple[str, str]]:
        if not order_id:
            return None
        try:
            detail = await self.api.fetch_order(order_id, symbol)
        except Exception as exc:  # pragma: no cover - live API dependent
            logger.debug(
                "KucoinFetcher._enrich_with_order_details: fetch_order failed for %s (%s)",
                order_id,
                exc,
            )
            return None
        info = detail.get("info") if isinstance(detail, dict) else detail
        if not isinstance(info, dict):
            return None
        client_oid = (
            detail.get("clientOrderId")
            or info.get("clientOrderId")
            or info.get("clientOid")
            or info.get("clientOid")
        )
        if not client_oid:
            return None
        client_oid = str(client_oid)
        return client_oid, custom_id_to_snake(client_oid)


# ---------------------------------------------------------------------------
# Utilities for Bitget integration
# ---------------------------------------------------------------------------


def custom_id_to_snake(client_oid: str) -> str:
    """Placeholder import shim; real implementation lives in passivbot."""
    try:
        from passivbot import custom_id_to_snake as _real

        return _real(client_oid)
    except Exception:
        return client_oid or ""


def deduce_side_pside(elm: dict) -> Tuple[str, str]:
    """Import helper from exchanges.bitget when available."""
    try:
        from exchanges.bitget import deduce_side_pside as _real

        return _real(elm)
    except Exception:
        side = str(elm.get("side", "buy")).lower()
        return side or "buy", "long"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


EXCHANGE_BOT_CLASSES: Dict[str, Tuple[str, str]] = {
    "binance": ("exchanges.binance", "BinanceBot"),
    "bitget": ("exchanges.bitget", "BitgetBot"),
    "bybit": ("exchanges.bybit", "BybitBot"),
    "hyperliquid": ("exchanges.hyperliquid", "HyperliquidBot"),
    "gateio": ("exchanges.gateio", "GateIOBot"),
    "kucoin": ("exchanges.kucoin", "KucoinBot"),
}


def _parse_time_arg(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        ts = int(value)
        if ts < 10**11:
            ts *= 1000
        return ts
    except ValueError:
        pass
    try:
        if value.lower() == "now":
            dt = datetime.now(tz=timezone.utc)
        else:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"Unable to parse datetime '{value}'")


def _parse_log_level(value: str) -> int:
    mapping = {"warning": 0, "warn": 0, "info": 1, "debug": 2, "trace": 3}
    if value is None:
        return 1
    value = str(value).strip().lower()
    if value in mapping:
        return mapping[value]
    try:
        lvl = int(float(value))
        return max(0, min(3, lvl))
    except Exception:
        return 1


def _extract_symbol_pool(config: dict, override: Optional[List[str]]) -> List[str]:
    if override:
        return sorted({sym for sym in override if sym})
    live = config.get("live", {})
    approved = live.get("approved_coins")
    symbols: List[str] = []
    if isinstance(approved, dict):
        for vals in approved.values():
            if isinstance(vals, list):
                symbols.extend(vals)
    elif isinstance(approved, list):
        symbols.extend(approved)
    return sorted({sym for sym in symbols if sym})


def _symbol_resolver(bot) -> Callable[[Optional[str]], str]:
    def resolver(raw: Optional[str]) -> str:
        if not raw:
            return ""
        if isinstance(raw, str) and "/" in raw:
            return raw
        value = "" if raw is None else str(raw)
        if not value:
            return ""
        # Prefer the bot's coin_to_symbol mapping which handles exchange quirks
        try:
            mapped = bot.coin_to_symbol(value, verbose=False)
            if mapped:
                return mapped
        except Exception:
            pass
        upper = value.upper()
        for quote in ("USDT", "USDC", "USD"):
            if upper.endswith(quote) and len(upper) > len(quote):
                base = upper[: -len(quote)]
                if base:
                    return f"{base}/{quote}:{quote}"
        if ":" in value and "/" not in value:
            base, _, quote = value.partition(":")
            if base and quote:
                return f"{base}/{quote}:{quote}"
        return value

    return resolver


def _build_fetcher_for_bot(bot, symbols: List[str]) -> BaseFetcher:
    exchange = getattr(bot, "exchange", "").lower()
    resolver = _symbol_resolver(bot)
    static_provider = lambda: symbols  # noqa: E731
    if exchange == "binance":
        return BinanceFetcher(
            api=bot.cca,
            symbol_resolver=resolver,
            positions_provider=static_provider,
            open_orders_provider=static_provider,
        )
    if exchange == "bitget":
        return BitgetFetcher(
            api=bot.cca,
            symbol_resolver=lambda value: resolver(value),
        )
    if exchange == "bybit":
        return BybitFetcher(api=bot.cca)
    if exchange == "hyperliquid":
        return HyperliquidFetcher(
            api=bot.cca,
            symbol_resolver=lambda value: resolver(value),
        )
    if exchange == "gateio":
        return GateioFetcher(
            api=bot.cca,
        )
    if exchange == "kucoin":
        return KucoinFetcher(api=bot.cca)
    raise ValueError(f"Unsupported exchange '{exchange}' for fill events CLI")


def _instantiate_bot(config: dict):
    live = config.get("live", {})
    user = str(live.get("user") or "").strip()
    if not user:
        raise ValueError("Config missing live.user to determine bot exchange")
    user_info = load_user_info(user)
    exchange = str(user_info.get("exchange") or "").lower()
    if not exchange:
        raise ValueError(f"User '{user}' has no exchange configured in api-keys.json")
    bot_cls_info = EXCHANGE_BOT_CLASSES.get(exchange)
    if bot_cls_info is None:
        raise ValueError(f"No bot class registered for exchange '{exchange}'")
    module = import_module(bot_cls_info[0])
    bot_cls = getattr(module, bot_cls_info[1])
    return bot_cls(config)


async def _run_cli(args: argparse.Namespace) -> None:
    config = load_config(args.config, verbose=False)
    config = format_config(config, verbose=False)
    live = config.setdefault("live", {})
    if args.user:
        live["user"] = args.user
    bot = _instantiate_bot(config)
    try:
        symbol_pool = _extract_symbol_pool(config, args.symbols)
        fetcher = _build_fetcher_for_bot(bot, symbol_pool)
        cache_root = Path(args.cache_root)
        cache_path = cache_root / bot.exchange / bot.user
        manager = FillEventsManager(
            exchange=bot.exchange,
            user=bot.user,
            fetcher=fetcher,
            cache_path=cache_path,
        )
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = _parse_time_arg(args.start) or (
            now_ms - int(args.lookback_days * 24 * 60 * 60 * 1000)
        )
        end_ms = _parse_time_arg(args.end) or now_ms
        if start_ms >= end_ms:
            raise ValueError("start time must be earlier than end time")
        logger.info(
            "fill_events_manager CLI | exchange=%s user=%s start=%s end=%s cache=%s",
            bot.exchange,
            bot.user,
            _format_ms(start_ms),
            _format_ms(end_ms),
            cache_path,
        )
        await manager.refresh_range(start_ms, end_ms)
        events = manager.get_events(start_ms, end_ms)
        logger.info("fill_events_manager CLI: events=%d written to %s", len(events), cache_path)
    finally:
        try:
            await bot.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill events cache refresher")
    parser.add_argument(
        "--config", "-c", type=str, default="configs/template.json", help="Config path"
    )
    parser.add_argument("--user", "-u", type=str, required=True, help="Live user identifier")
    parser.add_argument("--start", "-s", type=str, help="Start datetime (ms or ISO)")
    parser.add_argument("--end", "-e", type=str, help="End datetime (ms or ISO)")
    parser.add_argument(
        "--lookback-days",
        "-d",
        type=float,
        default=30.0,
        help="Default lookback window in days when start is omitted",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="info",
        help="Logging verbosity (warning/info/debug/trace or 0-3)",
    )
    parser.add_argument(
        "--cache-root",
        "-r",
        type=str,
        default="caches/fill_events",
        help="Root directory for fill events cache (default: caches/fill_events)",
    )
    parser.add_argument(
        "--symbols",
        "-S",
        nargs="*",
        default=None,
        help="Optional explicit symbol list to fetch",
    )
    args = parser.parse_args()
    configure_logging(debug=_parse_log_level(args.log_level))
    try:
        asyncio.run(_run_cli(args))
    except KeyboardInterrupt:
        logger.info("fill_events_manager CLI interrupted by user")


if __name__ == "__main__":
    main()
