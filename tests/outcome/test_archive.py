from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
import time

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
    OutcomeSettlementEvidence,
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


def test_archive_rejects_conflicting_duplicate_trade_identities(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    source_identified = trade("event-1")
    sequence_identified = replace(
        trade(None, timestamp_ms=2_000),
        sequence_id="sequence-1",
    )

    assert archive.append_trade(source_identified) is True
    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        archive.append_trade(replace(source_identified, qty=3.0))

    assert archive.append_trade(sequence_identified) is True
    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        archive.append_trade(
            replace(
                sequence_identified,
                native_price=0.41,
                canonical_yes_price=0.41,
            )
        )


def test_archive_trade_identities_are_unique_across_outcome_assets(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    outcome_market = market()
    source_identified = trade("event-1")
    sequence_identified = replace(
        trade(None, timestamp_ms=2_000),
        sequence_id="sequence-1",
    )
    opposite_source = replace(
        source_identified,
        asset_id=outcome_market.no_asset.asset_id,
        outcome=OutcomeSide.NO,
        native_price=0.6,
    )
    opposite_sequence = replace(
        sequence_identified,
        asset_id=outcome_market.no_asset.asset_id,
        outcome=OutcomeSide.NO,
        native_price=0.6,
    )

    assert archive.append_trade(source_identified) is True
    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        archive.append_trade(opposite_source)

    assert archive.append_trade(sequence_identified) is True
    with pytest.raises(ValueError, match="conflicting outcome trade evidence"):
        archive.append_trade(opposite_sequence)


def test_archive_rejects_conflicting_duplicate_settlement_identity(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    original = OutcomeSettlementEvidence(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="913",
        yes_fraction=1.0,
        payout_unit=1.0,
        settlement_time_ms=5_000,
        capital_release_time_ms=5_000,
        received_time_ms=5_100,
        source_event_id="settlement-1",
        evidence_source="hyperliquid_user_fill",
        observed_yes_qty=2.0,
        observed_no_qty=0.0,
        collateral_payout=2.0,
        fee=0.0,
        fee_asset="USDC",
        raw_payload={"dir": "Settlement"},
    )

    assert archive.append_settlement(original) is True
    assert archive.append_settlement(replace(original, received_time_ms=5_200)) is False
    assert (
        archive.append_settlement(
            replace(
                original,
                received_time_ms=5_300,
                evidence_source="hyperliquid_user_fills_by_time",
            )
        )
        is False
    )
    with pytest.raises(ValueError, match="conflicting outcome settlement evidence"):
        archive.append_settlement(replace(original, yes_fraction=0.0))


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
    assert archive.append_market_metadata(
        original,
        observed_at_ms=4_000,
        observation_source="outcomeMeta",
    )
    assert archive.load_market_metadata(original.venue, original.market_id) == [
        original,
        updated_constraints,
        original,
    ]
    assert (
        archive.load_market_metadata_at(
            original.venue,
            original.market_id,
            observed_at_or_before_ms=2_500,
        )
        == original
    )
    assert (
        archive.load_market_metadata_at(
            original.venue,
            original.market_id,
            observed_at_or_before_ms=3_500,
        )
        == updated_constraints
    )

    with pytest.raises(ValueError, match="conflicting immutable metadata"):
        archive.append_market_metadata(
            replace(original, description="different immutable contract"),
            observed_at_ms=4_000,
            observation_source="outcomeMeta",
        )


def test_market_metadata_versions_quote_asset_as_mutable_transport_state(tmp_path):
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    original = market()
    updated_quote_asset = replace(original, quote_asset="pUSD")

    assert archive.append_market_metadata(
        original,
        observed_at_ms=1_000,
        observation_source="outcomeMeta",
    )
    assert archive.append_market_metadata(
        updated_quote_asset,
        observed_at_ms=2_000,
        observation_source="outcomeMeta",
    )
    assert archive.load_market_metadata(original.venue, original.market_id) == [
        original,
        updated_quote_asset,
    ]


def test_market_metadata_fingerprint_check_is_serialized_with_insert(tmp_path):
    db_path = tmp_path / "concurrent-metadata.sqlite"
    first_archive = OutcomeTradeArchive(db_path)
    second_archive = OutcomeTradeArchive(db_path)
    original = market()
    conflicting = replace(original, description="conflicting immutable contract")
    started = threading.Event()
    errors: list[BaseException] = []

    def append_conflicting() -> None:
        started.set()
        try:
            second_archive.append_market_metadata(
                conflicting,
                observed_at_ms=1_000,
                observation_source="collector-2",
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            second_archive.close()

    with first_archive.write_transaction():
        assert first_archive.append_market_metadata(
            original,
            observed_at_ms=1_000,
            observation_source="collector-1",
        )
        writer = threading.Thread(target=append_conflicting)
        writer.start()
        assert started.wait(timeout=1.0)
        time.sleep(0.05)
        assert writer.is_alive()

    writer.join(timeout=2.0)
    assert not writer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "conflicting immutable metadata" in str(errors[0])
    assert first_archive.load_market_metadata(original.venue, original.market_id) == [
        original
    ]


def test_market_metadata_schema_migration_preserves_later_state_reversion(tmp_path):
    db_path = tmp_path / "legacy-metadata.sqlite"
    original = market()
    updated = replace(original, min_order_qty=50.0)
    archive = OutcomeTradeArchive(db_path)
    assert archive.append_market_metadata(
        original,
        observed_at_ms=1_000,
        observation_source="legacy",
    )
    archive.close()

    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            ALTER TABLE outcome_market_metadata
                RENAME TO outcome_market_metadata_current;
            CREATE TABLE outcome_market_metadata (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT NOT NULL,
                market_id TEXT NOT NULL,
                observed_at_ms INTEGER NOT NULL,
                observation_source TEXT NOT NULL,
                contract_fingerprint TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                normalized_market_json TEXT NOT NULL,
                collector_session TEXT,
                archived_at_ms INTEGER NOT NULL,
                UNIQUE(venue, market_id, payload_sha256)
            );
            INSERT INTO outcome_market_metadata
            SELECT * FROM outcome_market_metadata_current;
            DROP TABLE outcome_market_metadata_current;
            CREATE INDEX outcome_market_metadata_lookup
                ON outcome_market_metadata(
                    venue, market_id, observed_at_ms, record_id
                );
            """
        )

    migrated = OutcomeTradeArchive(db_path)
    assert migrated.append_market_metadata(
        updated,
        observed_at_ms=2_000,
        observation_source="migrated",
    )
    assert migrated.append_market_metadata(
        original,
        observed_at_ms=2_000,
        observation_source="migrated",
    )
    assert migrated.load_market_metadata(original.venue, original.market_id) == [
        original,
        updated,
        original,
    ]


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
    assert (
        archive.append_price_grid_change(
            replace(
                change,
                received_time_ms=10_000,
                raw_payload={
                    "event_type": "tick_size_change",
                    "asset_id": "complementary-token",
                },
            ),
            collector_session="session-1",
        )
        is False
    )
    assert archive.load_price_grid_changes(
        OutcomeVenue.POLYMARKET,
        "condition",
        start_ms=0,
        end_ms=10_000,
    ) == [change]
