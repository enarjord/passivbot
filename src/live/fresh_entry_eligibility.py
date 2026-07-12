"""Dependency-light diagnostics for why fresh initial entries were not created."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


_PSIDES = frozenset(("long", "short"))
_FACT_ALIASES = {
    "evaluated": "evaluated",
    "evaluated_count": "evaluated",
    "ideal": "ideal",
    "ideal_count": "ideal",
    "satisfied": "satisfied",
    "satisfied_count": "satisfied",
    "blocked": "blocked",
    "blocked_count": "blocked",
    "protective": "protective",
    "protective_count": "protective",
    "eligible": "eligible",
    "eligible_count": "eligible",
}
_OUTCOMES = (
    "eligible",
    "blocked_candidate",
    "already_satisfied",
    "protective_only",
    "no_candidate",
)
_SYMBOL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")
_REASON_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class _TraceRecord:
    facts: dict[str, int] = field(
        default_factory=lambda: {fact: 0 for fact in set(_FACT_ALIASES.values())}
    )
    reason_counts: dict[str, int] = field(default_factory=dict)


class FreshEntryEligibilityTrace:
    """Collect bounded, per-position-side eligibility facts without affecting orders.

    This object intentionally does not decode Passivbot client IDs. Callers that already
    decoded an order can use :meth:`record_count` to report the corresponding fact.
    """

    MAX_LABEL_LENGTH = 120
    MAX_REASON_KEYS_PER_RECORD = 16
    _REASON_OVERFLOW = "reason_overflow"
    _UNCLASSIFIED_CANDIDATE = "unclassified_candidate"
    _RUST_NO_INITIAL_CANDIDATE = "rust_no_initial_candidate"
    _PROTECTIVE_ACTIONS_ONLY = "protective_actions_only"

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], _TraceRecord] = {}

    def record_evaluated(self, symbol: str, pside: str) -> None:
        """Record that a symbol/position side was considered for a fresh entry."""
        self.record_count(symbol, pside, "evaluated")

    def record_ideal_orders(self, orders: Iterable[Mapping[str, object]] | object) -> None:
        """Record readable, non-protective initial-entry candidates from ``orders``."""
        self._record_initial_orders(orders, "ideal")

    def record_satisfied_orders(
        self, orders: Iterable[Mapping[str, object]] | object, reason: str
    ) -> None:
        """Record readable initial entries already accounted for by an existing order."""
        self._record_initial_orders(orders, "satisfied", reason=reason)

    def record_blocked_orders(
        self, orders: Iterable[Mapping[str, object]] | object, reason: str
    ) -> None:
        """Record readable initial entries blocked before creation."""
        self._record_initial_orders(orders, "blocked", reason=reason)

    def record_protective_orders(
        self, orders: Iterable[Mapping[str, object]] | object
    ) -> None:
        """Record reduce-only or panic orders separately from fresh-entry candidates."""
        for order in self._iter_orders(orders):
            if not self._is_protective(order):
                continue
            pair = self._order_pair(order)
            if pair is not None:
                self.record_count(*pair, "protective")

    def record_eligible_orders(
        self, orders: Iterable[Mapping[str, object]] | object
    ) -> None:
        """Record readable initial entries that remain eligible for creation."""
        self._record_initial_orders(orders, "eligible")

    def record_count(
        self,
        symbol: str,
        pside: str,
        fact: str,
        count: int = 1,
        reason: str | None = None,
    ) -> None:
        """Record already-classified facts supplied by an integration point.

        Labels and counts are validated here because this direct API is the boundary
        between arbitrary integration data and the queryable diagnostic payload.
        """
        key = self._validate_pair(symbol, pside)
        normalized_fact = self._validate_fact(fact)
        normalized_count = self._validate_count(count)
        normalized_reason = self._validate_reason(reason) if reason is not None else None
        if normalized_count == 0:
            return
        record = self._records.setdefault(key, _TraceRecord())
        record.facts[normalized_fact] += normalized_count
        if normalized_reason is not None:
            self._add_bounded_reason(
                record.reason_counts,
                normalized_reason,
                normalized_count,
                self.MAX_REASON_KEYS_PER_RECORD,
            )

    def to_event_data(self, max_records: int = 32) -> dict[str, object]:
        """Return a deterministic, bounded event-safe view of the collected facts."""
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 0:
            raise ValueError("max_records must be a nonnegative integer")
        record_limit = max_records
        records: list[dict[str, object]] = []
        records_total = 0
        outcome_counts = {outcome: 0 for outcome in _OUTCOMES}
        event_reason_counts: dict[str, int] = {}
        evaluated_count = 0

        for (symbol, pside), record in sorted(self._records.items()):
            records_total += 1
            outcome, reason_counts = self._record_outcome(record)
            outcome_counts[outcome] += 1
            evaluated_count += record.facts["evaluated"]
            for reason, count in reason_counts.items():
                event_reason_counts[reason] = event_reason_counts.get(reason, 0) + count
            if len(records) < record_limit:
                records.append(
                    {
                        "symbol": symbol,
                        "pside": pside,
                        "outcome": outcome,
                        "evaluated_count": record.facts["evaluated"],
                        "ideal_count": record.facts["ideal"],
                        "satisfied_count": record.facts["satisfied"],
                        "blocked_count": record.facts["blocked"],
                        "protective_count": record.facts["protective"],
                        "eligible_count": record.facts["eligible"],
                        "reason_counts": dict(sorted(reason_counts.items())),
                    }
                )

        return {
            "evaluated_count": evaluated_count,
            "outcome_counts": outcome_counts,
            "reason_counts": dict(sorted(event_reason_counts.items())),
            "records_total": records_total,
            "records": records,
            "records_truncated": records_total > record_limit,
        }

    def _record_initial_orders(
        self, orders: Iterable[Mapping[str, object]] | object, fact: str, reason: str | None = None
    ) -> None:
        if reason is not None:
            # Validate once, before any partial batch changes are recorded.
            self._validate_reason(reason)
        for order in self._iter_orders(orders):
            if not self._is_initial_entry(order) or self._is_protective(order):
                continue
            pair = self._order_pair(order)
            if pair is not None:
                self.record_count(*pair, fact, reason=reason)

    @staticmethod
    def _iter_orders(
        orders: Iterable[Mapping[str, object]] | object,
    ) -> Iterable[Mapping[str, object]]:
        if isinstance(orders, (str, bytes, Mapping)):
            return ()
        try:
            iterator = iter(orders)  # type: ignore[arg-type]
        except TypeError:
            return ()
        return (order for order in iterator if isinstance(order, Mapping))

    def _order_pair(self, order: Mapping[str, object]) -> tuple[str, str] | None:
        symbol = order.get("symbol")
        pside = (
            order.get("position_side")
            or order.get("positionSide")
            or order.get("pside")
        )
        try:
            return self._validate_pair(symbol, pside)
        except ValueError:
            return None

    @staticmethod
    def _normalized_text(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return _NORMALIZE_RE.sub("_", value.lower()).strip("_")

    def _is_initial_entry(self, order: Mapping[str, object]) -> bool:
        pb_type = self._normalized_text(order.get("pb_order_type"))
        custom_id = self._normalized_text(order.get("custom_id"))
        return pb_type.startswith("entry_initial_") or custom_id.startswith("entry_initial_")

    def _is_protective(self, order: Mapping[str, object]) -> bool:
        reduce_only = order.get("reduce_only", order.get("reduceOnly", False))
        if isinstance(reduce_only, str):
            if reduce_only.lower() in {"true", "1", "yes"}:
                return True
        elif bool(reduce_only):
            return True
        return "panic" in self._normalized_text(order.get("pb_order_type"))

    def _record_outcome(self, record: _TraceRecord) -> tuple[str, dict[str, int]]:
        facts = record.facts
        unaccounted = max(
            0,
            facts["ideal"] - facts["eligible"] - facts["blocked"] - facts["satisfied"],
        )
        if facts["eligible"] > 0:
            outcome = "eligible"
        elif facts["blocked"] > 0 or unaccounted > 0:
            outcome = "blocked_candidate"
        elif facts["satisfied"] > 0:
            outcome = "already_satisfied"
        elif facts["ideal"] == 0 and facts["protective"] > 0:
            outcome = "protective_only"
        else:
            outcome = "no_candidate"

        reason_counts = dict(record.reason_counts)
        if unaccounted:
            self._add_bounded_reason(
                reason_counts,
                self._UNCLASSIFIED_CANDIDATE,
                unaccounted,
                self.MAX_REASON_KEYS_PER_RECORD,
            )
        elif outcome == "protective_only" and not reason_counts:
            self._add_bounded_reason(
                reason_counts,
                self._PROTECTIVE_ACTIONS_ONLY,
                max(1, facts["protective"]),
                self.MAX_REASON_KEYS_PER_RECORD,
            )
        elif outcome == "no_candidate" and not reason_counts:
            self._add_bounded_reason(
                reason_counts,
                self._RUST_NO_INITIAL_CANDIDATE,
                1,
                self.MAX_REASON_KEYS_PER_RECORD,
            )
        return outcome, reason_counts

    @classmethod
    def _add_bounded_reason(
        cls, reason_counts: dict[str, int], reason: str, count: int, limit: int
    ) -> None:
        if reason in reason_counts:
            reason_counts[reason] += count
            return
        if len(reason_counts) < limit:
            reason_counts[reason] = count
            return
        if cls._REASON_OVERFLOW in reason_counts:
            reason_counts[cls._REASON_OVERFLOW] += count
            return
        # Reserve one bounded bucket for any reasons beyond this point.
        last_reason = sorted(reason_counts)[-1]
        overflow_count = reason_counts.pop(last_reason)
        reason_counts[cls._REASON_OVERFLOW] = overflow_count + count

    @classmethod
    def _validate_pair(cls, symbol: object, pside: object) -> tuple[str, str]:
        return cls._validate_symbol(symbol), cls._validate_pside(pside)

    @classmethod
    def _validate_symbol(cls, symbol: object) -> str:
        if (
            not isinstance(symbol, str)
            or len(symbol) > cls.MAX_LABEL_LENGTH
            or not _SYMBOL_RE.fullmatch(symbol)
        ):
            raise ValueError("symbol must be a safe query label no longer than 120 characters")
        return symbol

    @staticmethod
    def _validate_pside(pside: object) -> str:
        if not isinstance(pside, str) or pside.lower() not in _PSIDES:
            raise ValueError("pside must be 'long' or 'short'")
        return pside.lower()

    @staticmethod
    def _validate_fact(fact: object) -> str:
        if not isinstance(fact, str) or fact not in _FACT_ALIASES:
            raise ValueError(
                "fact must be one of evaluated, ideal, satisfied, blocked, protective, eligible"
            )
        return _FACT_ALIASES[fact]

    @classmethod
    def _validate_reason(cls, reason: object) -> str:
        if (
            not isinstance(reason, str)
            or len(reason) > cls.MAX_LABEL_LENGTH
            or not _REASON_RE.fullmatch(reason)
        ):
            raise ValueError("reason must be a safe query label no longer than 120 characters")
        return reason

    @staticmethod
    def _validate_count(count: object) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        return count
