from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage
from outcome.models import (
    NormalizedOutcomeTrade,
    OutcomeBookLevel,
    OutcomeBookSnapshot,
    OutcomeOrderSide,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
    OutcomeSide,
    OutcomeVenue,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def market():
    return hyperliquid.normalize_market(
        json.loads((FIXTURES / "hyperliquid_price_binary.json").read_text())
    )


def trade(event_id: str | None, timestamp_ms: int = 1_234) -> NormalizedOutcomeTrade:
    return NormalizedOutcomeTrade(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="913",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        native_side=OutcomeOrderSide.BUY,
        native_price=0.4,
        canonical_yes_price=0.4,
        qty=2.0,
        exchange_time_ms=timestamp_ms,
        received_time_ms=timestamp_ms + 10,
        source_event_id=event_id,
        raw_payload={"coin": "#9130", "tid": 1},
    )


def test_archive_deduplicates_explicit_event_identity_and_round_trips(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    original = replace(trade("event-1"), collector_sequence=7)

    assert archive.append_trade(original, collector_session="session-1") is True
    assert archive.append_trade(replace(original, received_time_ms=9_999)) is False
    loaded = archive.load_trades(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=0,
        end_ms=10_000,
    )

    assert loaded == [original]


def test_archive_does_not_invent_identity_for_source_without_unique_ids(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    no_identity = trade(None)

    assert archive.append_trade(no_identity) is True
    assert archive.append_trade(no_identity) is True
    loaded = archive.load_trades(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=0,
        end_ms=10_000,
    )
    assert len(loaded) == 2


def test_market_metadata_archive_round_trips_versions_and_rejects_id_reuse(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    original = market()
    updated_constraints = replace(original, min_order_qty=50.0)

    assert archive.append_market_metadata(
        original,
        observed_at_ms=1_000,
        observation_source="outcomeMeta",
    )
    assert not archive.append_market_metadata(
        original,
        observed_at_ms=2_000,
        observation_source="outcomeMeta",
    )
    assert archive.append_market_metadata(
        updated_constraints,
        observed_at_ms=3_000,
        observation_source="outcomeMeta",
    )
    assert archive.load_market_metadata(original.venue, original.market_id) == [
        original,
        updated_constraints,
    ]

    with pytest.raises(ValueError, match="conflicting immutable metadata"):
        archive.append_market_metadata(
            replace(original, description="different immutable contract"),
            observed_at_ms=4_000,
            observation_source="outcomeMeta",
        )


def test_verified_coverage_is_merged_but_collection_gaps_are_not_marked_covered(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    archive.record_verified_coverage(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        VerifiedCoverage(1_000, 3_000),
        collector_session="session-1",
    )
    archive.record_gap(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=3_000,
        end_ms=4_000,
        reason="websocket_disconnect",
        collector_session="session-1",
    )
    archive.record_verified_coverage(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        VerifiedCoverage(4_000, 6_000),
        collector_session="session-2",
    )

    assert archive.load_verified_coverage(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=0,
        end_ms=10_000,
    ) == [VerifiedCoverage(1_000, 3_000), VerifiedCoverage(4_000, 6_000)]


def test_verified_price_grid_coverage_is_stored_separately_from_fill_coverage(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    archive.record_verified_price_grid_coverage(
        OutcomeVenue.POLYMARKET,
        "condition-1",
        VerifiedCoverage(1_000, 3_000),
        collector_session="grid-1",
    )
    archive.record_verified_price_grid_coverage(
        OutcomeVenue.POLYMARKET,
        "condition-1",
        VerifiedCoverage(3_000, 5_000),
        collector_session="grid-2",
    )

    assert archive.load_verified_price_grid_coverage(
        OutcomeVenue.POLYMARKET,
        "condition-1",
        start_ms=0,
        end_ms=10_000,
    ) == [VerifiedCoverage(1_000, 5_000)]
    assert archive.load_verified_coverage(
        OutcomeVenue.POLYMARKET,
        "condition-1",
        "yes",
        start_ms=0,
        end_ms=10_000,
    ) == []


def test_book_archive_keeps_raw_snapshots_without_using_them_as_trades(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    book = OutcomeBookSnapshot(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="913",
        asset_id="+9130",
        outcome=OutcomeSide.YES,
        timestamp_ms=1_234,
        received_time_ms=1_250,
        bids=(OutcomeBookLevel(native_price=0.4, qty=10.0, order_count=2),),
        asks=(OutcomeBookLevel(native_price=0.42, qty=12.0, order_count=1),),
        raw_payload={
            "coin": "#9130",
            "time": 1_234,
            "levels": [
                [{"px": "0.4", "sz": "10", "n": 2}],
                [{"px": "0.42", "sz": "12", "n": 1}],
            ],
        },
    )

    assert archive.append_book(book, collector_session="session-1") is True
    assert archive.append_book(replace(book, received_time_ms=9_999)) is False
    assert archive.load_books(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=0,
        end_ms=10_000,
    ) == [book]
    assert archive.load_trades(
        OutcomeVenue.HYPERLIQUID,
        "913",
        "+9130",
        start_ms=0,
        end_ms=10_000,
    ) == []


def test_archive_round_trips_dynamic_price_grid_events(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    change = OutcomePriceGridChange(
        venue=OutcomeVenue.POLYMARKET,
        market_id="condition",
        timestamp_ms=1_234,
        received_time_ms=1_250,
        old_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.01),
        new_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001),
        raw_payload={"event_type": "tick_size_change"},
    )

    assert archive.append_price_grid_change(change, collector_session="session-1") is True
    assert archive.append_price_grid_change(replace(change, received_time_ms=9_999)) is False
    assert archive.load_price_grid_changes(
        OutcomeVenue.POLYMARKET,
        "condition",
        start_ms=0,
        end_ms=10_000,
    ) == [change]
