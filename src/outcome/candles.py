from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from outcome.models import (
    NormalizedOutcomeTrade,
    OutcomeCandle1s,
    OutcomeSide,
    OutcomeSignalCandle1s,
)


class AmbiguousTradeOrderingError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedCoverage:
    """Half-open interval during which the collector is known to have observed every fill."""

    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("verified coverage must be a non-empty non-negative interval")

    def contains_second(self, timestamp_ms: int) -> bool:
        return self.start_ms <= timestamp_ms and timestamp_ms + 1_000 <= self.end_ms


def _immutable_trade_fields(trade: NormalizedOutcomeTrade) -> tuple[object, ...]:
    return (
        trade.asset_id,
        trade.outcome,
        trade.native_side,
        trade.native_price,
        trade.canonical_yes_price,
        trade.qty,
        trade.exchange_time_ms,
        trade.source_event_id,
        trade.economic_event_id,
        trade.sequence_id,
    )


def _deduplicate(trades: Sequence[NormalizedOutcomeTrade]) -> list[tuple[int, NormalizedOutcomeTrade]]:
    seen: dict[tuple[str, ...], tuple[object, ...]] = {}
    retained: list[tuple[int, NormalizedOutcomeTrade]] = []
    for index, trade in enumerate(trades):
        key = trade.deduplication_key
        if key is not None:
            if key in seen:
                if seen[key] != _immutable_trade_fields(trade):
                    raise ValueError(
                        "conflicting outcome trade evidence for an immutable "
                        "source or sequence identity"
                    )
                continue
            seen[key] = _immutable_trade_fields(trade)
        retained.append((index, trade))
    return retained


def _deduplicate_economic_events(
    trades: Sequence[NormalizedOutcomeTrade],
) -> list[tuple[int, NormalizedOutcomeTrade]]:
    retained = _deduplicate(trades)
    side_occurrences: dict[
        tuple[str, ...],
        dict[OutcomeSide, int],
    ] = defaultdict(lambda: defaultdict(int))
    seen_occurrences: dict[
        tuple[tuple[str, ...], int],
        tuple[float, float, int],
    ] = {}
    economic: list[tuple[int, NormalizedOutcomeTrade]] = []
    for item in retained:
        trade = item[1]
        key = trade.economic_deduplication_key
        if key is not None:
            # Hyperliquid can mirror one merged-book match into one YES and one NO
            # record.  Its available pairing tuple is not globally unique when a
            # transaction contains repeated equal-price/equal-size matches, so pair
            # the nth retained record from each native side instead of dropping every
            # record after the first tuple occurrence.
            occurrence = side_occurrences[key][trade.outcome]
            side_occurrences[key][trade.outcome] += 1
            occurrence_key = (key, occurrence)
            if occurrence_key in seen_occurrences:
                economic_fields = (
                    trade.canonical_yes_price,
                    trade.qty,
                    trade.exchange_time_ms,
                )
                if seen_occurrences[occurrence_key] != economic_fields:
                    raise ValueError(
                        "conflicting mirrored outcome trade evidence for one "
                        "economic event occurrence"
                    )
                continue
            seen_occurrences[occurrence_key] = (
                trade.canonical_yes_price,
                trade.qty,
                trade.exchange_time_ms,
            )
        economic.append(item)
    return economic


def _is_verified_second(timestamp_ms: int, coverage: Sequence[VerifiedCoverage]) -> bool:
    return any(interval.contains_second(timestamp_ms) for interval in coverage)


def _sequence_sort_key(sequence_id: str | None) -> tuple[int, int | str]:
    if sequence_id is None:
        return (2, "")
    if sequence_id.isdigit():
        return (0, int(sequence_id))
    return (1, sequence_id)


def _sort_and_validate_ordering(
    retained: list[tuple[int, NormalizedOutcomeTrade]],
) -> None:
    by_exchange_time: dict[int, list[tuple[int, NormalizedOutcomeTrade]]] = defaultdict(list)
    ordering_mode: dict[int, str] = {}
    for item in retained:
        by_exchange_time[item[1].exchange_time_ms].append(item)
    for exchange_time_ms, same_time in by_exchange_time.items():
        if len(same_time) <= 1:
            ordering_mode[exchange_time_ms] = "received"
            continue
        received_times = [item[1].received_time_ms for item in same_time]
        sequences = [item[1].sequence_id for item in same_time]
        collector_sequences = [item[1].collector_sequence for item in same_time]
        has_unique_receive_order = len(set(received_times)) == len(received_times)
        has_unique_source_sequence = all(sequence is not None for sequence in sequences) and len(
            set(sequences)
        ) == len(sequences)
        has_unique_collector_sequence = all(
            sequence is not None for sequence in collector_sequences
        ) and len(set(collector_sequences)) == len(collector_sequences)
        ordering_mode[exchange_time_ms] = (
            "source"
            if has_unique_source_sequence
            else "collector"
            if has_unique_collector_sequence
            else "received"
        )
        if (
            not has_unique_receive_order
            and not has_unique_source_sequence
            and not has_unique_collector_sequence
        ):
            raise AmbiguousTradeOrderingError(
                "multiple outcome trades share an exchange timestamp without a unique "
                f"source sequence, collector sequence, or receive order at {exchange_time_ms}"
            )

    def chronology_key(
        item: tuple[int, NormalizedOutcomeTrade],
    ) -> tuple[int, int, tuple[int, int | str], int, int]:
        source_index, trade = item
        mode = ordering_mode[trade.exchange_time_ms]
        if mode == "source":
            mode_priority = 0
            sequence_key = _sequence_sort_key(trade.sequence_id)
        elif mode == "collector":
            mode_priority = 1
            sequence_key = (0, int(trade.collector_sequence or 0))
        else:
            mode_priority = 2
            sequence_key = (2, "")
        return (
            trade.exchange_time_ms,
            mode_priority,
            sequence_key,
            trade.received_time_ms,
            source_index,
        )

    retained.sort(key=chronology_key)


def trades_to_1s_candles(
    trades: Iterable[NormalizedOutcomeTrade],
    *,
    verified_coverage: Sequence[VerifiedCoverage] = (),
) -> list[OutcomeCandle1s]:
    """Build canonical-price candles for one market and one native outcome book.

    Trade prices must already be transformed into the canonical YES coordinate. Unknown
    collection gaps break carry-forward continuity; only fully covered no-trade seconds are
    materialized.
    """

    trade_list = list(trades)
    if not trade_list:
        return []
    market_keys = {(trade.venue, trade.market_id, trade.outcome) for trade in trade_list}
    if len(market_keys) != 1:
        raise ValueError("trades_to_1s_candles requires one venue, market, and native outcome book")
    outcome = trade_list[0].outcome

    retained = _deduplicate(trade_list)
    _sort_and_validate_ordering(retained)
    by_second: dict[int, list[NormalizedOutcomeTrade]] = defaultdict(list)
    for _, trade in retained:
        by_second[(trade.exchange_time_ms // 1_000) * 1_000].append(trade)

    first_second = min(by_second)
    last_second = max(
        max(by_second),
        max((interval.end_ms - 1) // 1_000 * 1_000 for interval in verified_coverage)
        if verified_coverage
        else max(by_second),
    )
    candles: list[OutcomeCandle1s] = []
    prior_close: float | None = None
    for timestamp_ms in range(first_second, last_second + 1_000, 1_000):
        second_trades = by_second.get(timestamp_ms)
        if second_trades:
            prices = [trade.canonical_yes_price for trade in second_trades]
            candle = OutcomeCandle1s(
                timestamp_ms=timestamp_ms,
                source_outcome=outcome,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(trade.qty for trade in second_trades),
                trade_count=len(second_trades),
                carried_forward=False,
            )
            candles.append(candle)
            prior_close = candle.close
        elif prior_close is not None and _is_verified_second(timestamp_ms, verified_coverage):
            candles.append(
                OutcomeCandle1s(
                    timestamp_ms=timestamp_ms,
                    source_outcome=outcome,
                    open=prior_close,
                    high=prior_close,
                    low=prior_close,
                    close=prior_close,
                    volume=0.0,
                    trade_count=0,
                    carried_forward=True,
                )
            )
        else:
            prior_close = None
    return candles


def trades_to_1s_candles_by_native_book(
    trades: Iterable[NormalizedOutcomeTrade],
    *,
    verified_coverage: Sequence[VerifiedCoverage] = (),
) -> dict[OutcomeSide, list[OutcomeCandle1s]]:
    grouped: dict[OutcomeSide, list[NormalizedOutcomeTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.outcome].append(trade)
    return {
        outcome: trades_to_1s_candles(
            outcome_trades,
            verified_coverage=verified_coverage,
        )
        for outcome, outcome_trades in grouped.items()
    }


def trades_to_canonical_signal_1s_candles(
    trades: Iterable[NormalizedOutcomeTrade],
    *,
    verified_coverage: Sequence[VerifiedCoverage] = (),
) -> list[OutcomeSignalCandle1s]:
    """Combine actual YES and canonicalized NO fills into one dense signal series."""

    trade_list = list(trades)
    if not trade_list:
        return []
    market_keys = {(trade.venue, trade.market_id) for trade in trade_list}
    if len(market_keys) != 1:
        raise ValueError("canonical outcome signal candles require one venue and market")
    retained = _deduplicate_economic_events(trade_list)
    _sort_and_validate_ordering(retained)
    by_second: dict[int, list[NormalizedOutcomeTrade]] = defaultdict(list)
    for _, trade in retained:
        by_second[(trade.exchange_time_ms // 1_000) * 1_000].append(trade)
    first_second = min(by_second)
    last_second = max(
        max(by_second),
        max((interval.end_ms - 1) // 1_000 * 1_000 for interval in verified_coverage)
        if verified_coverage
        else max(by_second),
    )
    candles: list[OutcomeSignalCandle1s] = []
    prior_close: float | None = None
    for timestamp_ms in range(first_second, last_second + 1_000, 1_000):
        second_trades = by_second.get(timestamp_ms)
        if second_trades:
            prices = [trade.canonical_yes_price for trade in second_trades]
            candle = OutcomeSignalCandle1s(
                timestamp_ms=timestamp_ms,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(trade.qty for trade in second_trades),
                trade_count=len(second_trades),
                carried_forward=False,
            )
            candles.append(candle)
            prior_close = candle.close
        elif prior_close is not None and _is_verified_second(timestamp_ms, verified_coverage):
            candles.append(
                OutcomeSignalCandle1s(
                    timestamp_ms=timestamp_ms,
                    open=prior_close,
                    high=prior_close,
                    low=prior_close,
                    close=prior_close,
                    volume=0.0,
                    trade_count=0,
                    carried_forward=True,
                )
            )
        else:
            prior_close = None
    return candles
