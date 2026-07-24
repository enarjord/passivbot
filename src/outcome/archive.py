from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Iterable

from outcome.candles import VerifiedCoverage
from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeAsset,
    NormalizedOutcomeMarket,
    NormalizedOutcomeTrade,
    OutcomeBookLevel,
    OutcomeBookSnapshot,
    OutcomeCapabilities,
    OutcomeFeeMetadata,
    OutcomeOrderSide,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
    OutcomeSettlementEvidence,
    OutcomeSide,
    OutcomeVenue,
)


def _utc_ms() -> int:
    return int(time.time() * 1_000)


def _market_payload(market: NormalizedOutcomeMarket) -> dict:
    return {
        "venue": market.venue.value,
        "market_id": market.market_id,
        "title": market.title,
        "description": market.description,
        "quote_asset": market.quote_asset,
        "yes_asset": {
            **asdict(market.yes_asset),
            "side": market.yes_asset.side.value,
        },
        "no_asset": {
            **asdict(market.no_asset),
            "side": market.no_asset.side.value,
        },
        "payout_unit": market.payout_unit,
        "price_grid": asdict(market.price_grid),
        "qty_step": market.qty_step,
        "min_order_qty": market.min_order_qty,
        "min_order_notional": market.min_order_notional,
        "lifecycle": asdict(market.lifecycle),
        "capabilities": asdict(market.capabilities),
        "fee_metadata": asdict(market.fee_metadata),
        "native_metadata": dict(market.native_metadata),
    }


def _market_from_payload(payload: dict) -> NormalizedOutcomeMarket:
    yes_asset = dict(payload["yes_asset"])
    yes_asset["side"] = OutcomeSide(yes_asset["side"])
    no_asset = dict(payload["no_asset"])
    no_asset["side"] = OutcomeSide(no_asset["side"])
    return NormalizedOutcomeMarket(
        venue=OutcomeVenue(payload["venue"]),
        market_id=str(payload["market_id"]),
        title=str(payload["title"]),
        description=str(payload["description"]),
        quote_asset=str(payload["quote_asset"]),
        yes_asset=NormalizedOutcomeAsset(**yes_asset),
        no_asset=NormalizedOutcomeAsset(**no_asset),
        payout_unit=float(payload["payout_unit"]),
        price_grid=OutcomePriceGridMetadata(**payload["price_grid"]),
        qty_step=payload["qty_step"],
        min_order_qty=payload["min_order_qty"],
        min_order_notional=payload["min_order_notional"],
        lifecycle=MarketLifecycle(**payload["lifecycle"]),
        capabilities=OutcomeCapabilities(**payload["capabilities"]),
        fee_metadata=OutcomeFeeMetadata(**payload["fee_metadata"]),
        native_metadata=payload["native_metadata"],
    )


def _contract_fingerprint(market: NormalizedOutcomeMarket) -> str:
    immutable = {
        "venue": market.venue.value,
        "market_id": market.market_id,
        "title": market.title,
        "description": market.description,
        "quote_asset": market.quote_asset,
        "yes_asset": {
            **asdict(market.yes_asset),
            "side": market.yes_asset.side.value,
        },
        "no_asset": {
            **asdict(market.no_asset),
            "side": market.no_asset.side.value,
        },
        "payout_unit": market.payout_unit,
        "trading_open_time_ms": market.lifecycle.trading_open_time_ms,
        "scheduled_event_time_ms": market.lifecycle.scheduled_event_time_ms,
        "capabilities": asdict(market.capabilities),
    }
    encoded = json.dumps(
        immutable,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class OutcomeTradeArchive:
    """Durable raw normalized fills and collector-continuity evidence."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self.db_path))
            connection.row_factory = sqlite3.Row
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS outcome_market_metadata (
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
                CREATE INDEX IF NOT EXISTS outcome_market_metadata_lookup
                    ON outcome_market_metadata(
                        venue, market_id, observed_at_ms, record_id
                    );

                CREATE TABLE IF NOT EXISTS outcome_trades (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    native_side TEXT NOT NULL,
                    native_price REAL NOT NULL,
                    canonical_yes_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    exchange_time_ms INTEGER NOT NULL,
                    received_time_ms INTEGER NOT NULL,
                    source_event_id TEXT,
                    economic_event_id TEXT,
                    sequence_id TEXT,
                    collector_sequence INTEGER,
                    collector_session TEXT,
                    source_cursor TEXT,
                    raw_payload_json TEXT NOT NULL,
                    archived_at_ms INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS outcome_trades_event_identity
                    ON outcome_trades(venue, market_id, asset_id, source_event_id)
                    WHERE source_event_id IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS outcome_trades_sequence_identity
                    ON outcome_trades(venue, market_id, asset_id, sequence_id)
                    WHERE sequence_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS outcome_trades_time_lookup
                    ON outcome_trades(venue, market_id, asset_id, exchange_time_ms, record_id);

                CREATE TABLE IF NOT EXISTS outcome_books (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    exchange_time_ms INTEGER NOT NULL,
                    received_time_ms INTEGER NOT NULL,
                    bids_json TEXT NOT NULL,
                    asks_json TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    collector_session TEXT,
                    archived_at_ms INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS outcome_books_payload_identity
                    ON outcome_books(
                        venue, market_id, asset_id, exchange_time_ms, payload_sha256
                    );
                CREATE INDEX IF NOT EXISTS outcome_books_time_lookup
                    ON outcome_books(
                        venue, market_id, asset_id, exchange_time_ms, record_id
                    );

                CREATE TABLE IF NOT EXISTS outcome_price_grid_changes (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    exchange_time_ms INTEGER NOT NULL,
                    received_time_ms INTEGER NOT NULL,
                    old_grid_json TEXT NOT NULL,
                    new_grid_json TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    collector_session TEXT,
                    archived_at_ms INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS outcome_price_grid_change_identity
                    ON outcome_price_grid_changes(
                        venue, market_id, exchange_time_ms, payload_sha256
                    );
                CREATE INDEX IF NOT EXISTS outcome_price_grid_change_time_lookup
                    ON outcome_price_grid_changes(
                        venue, market_id, exchange_time_ms, record_id
                    );

                CREATE TABLE IF NOT EXISTS outcome_settlements (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    yes_fraction REAL NOT NULL,
                    payout_unit REAL NOT NULL,
                    settlement_time_ms INTEGER NOT NULL,
                    received_time_ms INTEGER NOT NULL,
                    source_event_id TEXT NOT NULL,
                    evidence_source TEXT NOT NULL,
                    observed_yes_qty REAL NOT NULL,
                    observed_no_qty REAL NOT NULL,
                    collateral_payout REAL NOT NULL,
                    fee REAL NOT NULL,
                    fee_asset TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    collector_session TEXT,
                    source_cursor TEXT,
                    archived_at_ms INTEGER NOT NULL,
                    UNIQUE(venue, market_id, source_event_id)
                );
                CREATE INDEX IF NOT EXISTS outcome_settlements_time_lookup
                    ON outcome_settlements(venue, market_id, settlement_time_ms, record_id);

                CREATE TABLE IF NOT EXISTS outcome_verified_coverage (
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    collector_session TEXT NOT NULL,
                    recorded_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(venue, market_id, asset_id, start_ms, end_ms, collector_session)
                );

                CREATE TABLE IF NOT EXISTS outcome_collection_gaps (
                    venue TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    collector_session TEXT,
                    recorded_at_ms INTEGER NOT NULL,
                    PRIMARY KEY(venue, market_id, asset_id, start_ms, end_ms, reason)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(outcome_trades)").fetchall()
            }
            if "economic_event_id" not in columns:
                connection.execute(
                    "ALTER TABLE outcome_trades ADD COLUMN economic_event_id TEXT"
                )
            if "collector_sequence" not in columns:
                connection.execute(
                    "ALTER TABLE outcome_trades ADD COLUMN collector_sequence INTEGER"
                )

    def append_market_metadata(
        self,
        market: NormalizedOutcomeMarket,
        *,
        observed_at_ms: int,
        observation_source: str,
        collector_session: str | None = None,
    ) -> bool:
        if observed_at_ms < 0:
            raise ValueError("outcome market observation time must be non-negative")
        if not observation_source.strip():
            raise ValueError("outcome market observation source must not be empty")
        payload_json = json.dumps(
            _market_payload(market),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        payload_sha256 = hashlib.sha256(payload_json.encode()).hexdigest()
        contract_fingerprint = _contract_fingerprint(market)
        connection = self._connect()
        existing_fingerprints = {
            row["contract_fingerprint"]
            for row in connection.execute(
                """
                SELECT DISTINCT contract_fingerprint
                FROM outcome_market_metadata
                WHERE venue = ? AND market_id = ?
                """,
                (market.venue.value, market.market_id),
            ).fetchall()
        }
        if existing_fingerprints and existing_fingerprints != {contract_fingerprint}:
            raise ValueError(
                f"outcome market {market.venue.value}:{market.market_id} "
                "has conflicting immutable metadata"
            )
        with connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outcome_market_metadata (
                    venue, market_id, observed_at_ms, observation_source,
                    contract_fingerprint, payload_sha256, normalized_market_json,
                    collector_session, archived_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market.venue.value,
                    market.market_id,
                    int(observed_at_ms),
                    observation_source,
                    contract_fingerprint,
                    payload_sha256,
                    payload_json,
                    collector_session,
                    _utc_ms(),
                ),
            )
        return cursor.rowcount == 1

    def load_market_metadata(
        self,
        venue: OutcomeVenue,
        market_id: str,
    ) -> list[NormalizedOutcomeMarket]:
        rows = self._connect().execute(
            """
            SELECT normalized_market_json
            FROM outcome_market_metadata
            WHERE venue = ? AND market_id = ?
            ORDER BY observed_at_ms, record_id
            """,
            (venue.value, str(market_id)),
        ).fetchall()
        return [
            _market_from_payload(json.loads(row["normalized_market_json"]))
            for row in rows
        ]

    def append_trade(
        self,
        trade: NormalizedOutcomeTrade,
        *,
        collector_session: str | None = None,
        source_cursor: str | None = None,
    ) -> bool:
        raw_payload_json = json.dumps(
            dict(trade.raw_payload),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outcome_trades (
                    venue, market_id, asset_id, outcome, native_side,
                    native_price, canonical_yes_price, qty,
                    exchange_time_ms, received_time_ms,
                    source_event_id, economic_event_id, sequence_id, collector_sequence,
                    collector_session, source_cursor, raw_payload_json, archived_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.venue.value,
                    trade.market_id,
                    trade.asset_id,
                    trade.outcome.value,
                    trade.native_side.value,
                    trade.native_price,
                    trade.canonical_yes_price,
                    trade.qty,
                    trade.exchange_time_ms,
                    trade.received_time_ms,
                    trade.source_event_id,
                    trade.economic_event_id,
                    trade.sequence_id,
                    trade.collector_sequence,
                    collector_session,
                    source_cursor,
                    raw_payload_json,
                    _utc_ms(),
                ),
            )
        return cursor.rowcount == 1

    def append_trades(
        self,
        trades: Iterable[NormalizedOutcomeTrade],
        *,
        collector_session: str | None = None,
    ) -> tuple[int, int]:
        inserted = 0
        ignored = 0
        for trade in trades:
            if self.append_trade(trade, collector_session=collector_session):
                inserted += 1
            else:
                ignored += 1
        return inserted, ignored

    def append_book(
        self,
        book: OutcomeBookSnapshot,
        *,
        collector_session: str | None = None,
    ) -> bool:
        bids_json = json.dumps(
            [
                {
                    "native_price": level.native_price,
                    "qty": level.qty,
                    "order_count": level.order_count,
                }
                for level in book.bids
            ],
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        asks_json = json.dumps(
            [
                {
                    "native_price": level.native_price,
                    "qty": level.qty,
                    "order_count": level.order_count,
                }
                for level in book.asks
            ],
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        raw_payload_json = json.dumps(
            dict(book.raw_payload),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        payload_sha256 = hashlib.sha256(raw_payload_json.encode()).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outcome_books (
                    venue, market_id, asset_id, outcome,
                    exchange_time_ms, received_time_ms,
                    bids_json, asks_json, raw_payload_json, payload_sha256,
                    collector_session, archived_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book.venue.value,
                    book.market_id,
                    book.asset_id,
                    book.outcome.value,
                    book.timestamp_ms,
                    book.received_time_ms,
                    bids_json,
                    asks_json,
                    raw_payload_json,
                    payload_sha256,
                    collector_session,
                    _utc_ms(),
                ),
            )
        return cursor.rowcount == 1

    def append_price_grid_change(
        self,
        change: OutcomePriceGridChange,
        *,
        collector_session: str | None = None,
    ) -> bool:
        def encode_grid(grid: OutcomePriceGridMetadata) -> str:
            return json.dumps(
                {
                    "kind": grid.kind,
                    "fixed_step": grid.fixed_step,
                    "max_significant_figures": grid.max_significant_figures,
                    "max_decimal_places": grid.max_decimal_places,
                },
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )

        raw_payload_json = json.dumps(
            dict(change.raw_payload),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        payload_sha256 = hashlib.sha256(raw_payload_json.encode()).hexdigest()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outcome_price_grid_changes (
                    venue, market_id, exchange_time_ms, received_time_ms,
                    old_grid_json, new_grid_json, raw_payload_json, payload_sha256,
                    collector_session, archived_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change.venue.value,
                    change.market_id,
                    change.timestamp_ms,
                    change.received_time_ms,
                    encode_grid(change.old_grid),
                    encode_grid(change.new_grid),
                    raw_payload_json,
                    payload_sha256,
                    collector_session,
                    _utc_ms(),
                ),
            )
        return cursor.rowcount == 1

    def append_settlement(
        self,
        settlement: OutcomeSettlementEvidence,
        *,
        collector_session: str | None = None,
        source_cursor: str | None = None,
    ) -> bool:
        raw_payload_json = json.dumps(
            dict(settlement.raw_payload),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO outcome_settlements (
                    venue, market_id, yes_fraction, payout_unit,
                    settlement_time_ms, received_time_ms,
                    source_event_id, evidence_source,
                    observed_yes_qty, observed_no_qty, collateral_payout,
                    fee, fee_asset, raw_payload_json,
                    collector_session, source_cursor, archived_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    settlement.venue.value,
                    settlement.market_id,
                    settlement.yes_fraction,
                    settlement.payout_unit,
                    settlement.settlement_time_ms,
                    settlement.received_time_ms,
                    settlement.source_event_id,
                    settlement.evidence_source,
                    settlement.observed_yes_qty,
                    settlement.observed_no_qty,
                    settlement.collateral_payout,
                    settlement.fee,
                    settlement.fee_asset,
                    raw_payload_json,
                    collector_session,
                    source_cursor,
                    _utc_ms(),
                ),
            )
        return cursor.rowcount == 1

    def load_trades(
        self,
        venue: OutcomeVenue,
        market_id: str,
        asset_id: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[NormalizedOutcomeTrade]:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("trade query must use a non-empty non-negative interval")
        rows = self._connect().execute(
            """
            SELECT *
            FROM outcome_trades
            WHERE venue = ? AND market_id = ? AND asset_id = ?
              AND exchange_time_ms >= ? AND exchange_time_ms < ?
            ORDER BY exchange_time_ms, received_time_ms, record_id
            """,
            (venue.value, str(market_id), str(asset_id), int(start_ms), int(end_ms)),
        ).fetchall()
        return [
            NormalizedOutcomeTrade(
                venue=OutcomeVenue(row["venue"]),
                market_id=row["market_id"],
                asset_id=row["asset_id"],
                outcome=OutcomeSide(row["outcome"]),
                native_side=OutcomeOrderSide(row["native_side"]),
                native_price=row["native_price"],
                canonical_yes_price=row["canonical_yes_price"],
                qty=row["qty"],
                exchange_time_ms=row["exchange_time_ms"],
                received_time_ms=row["received_time_ms"],
                source_event_id=row["source_event_id"],
                economic_event_id=row["economic_event_id"],
                sequence_id=row["sequence_id"],
                collector_sequence=row["collector_sequence"],
                raw_payload=json.loads(row["raw_payload_json"]),
            )
            for row in rows
        ]

    def load_books(
        self,
        venue: OutcomeVenue,
        market_id: str,
        asset_id: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[OutcomeBookSnapshot]:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("book query must use a non-empty non-negative interval")
        rows = self._connect().execute(
            """
            SELECT *
            FROM outcome_books
            WHERE venue = ? AND market_id = ? AND asset_id = ?
              AND exchange_time_ms >= ? AND exchange_time_ms < ?
            ORDER BY exchange_time_ms, received_time_ms, record_id
            """,
            (venue.value, str(market_id), str(asset_id), int(start_ms), int(end_ms)),
        ).fetchall()

        def levels(encoded: str) -> tuple[OutcomeBookLevel, ...]:
            return tuple(OutcomeBookLevel(**item) for item in json.loads(encoded))

        return [
            OutcomeBookSnapshot(
                venue=OutcomeVenue(row["venue"]),
                market_id=row["market_id"],
                asset_id=row["asset_id"],
                outcome=OutcomeSide(row["outcome"]),
                timestamp_ms=row["exchange_time_ms"],
                received_time_ms=row["received_time_ms"],
                bids=levels(row["bids_json"]),
                asks=levels(row["asks_json"]),
                raw_payload=json.loads(row["raw_payload_json"]),
            )
            for row in rows
        ]

    def load_price_grid_changes(
        self,
        venue: OutcomeVenue,
        market_id: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[OutcomePriceGridChange]:
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError("price-grid query must use a non-empty non-negative interval")
        rows = self._connect().execute(
            """
            SELECT *
            FROM outcome_price_grid_changes
            WHERE venue = ? AND market_id = ?
              AND exchange_time_ms >= ? AND exchange_time_ms < ?
            ORDER BY exchange_time_ms, received_time_ms, record_id
            """,
            (venue.value, str(market_id), int(start_ms), int(end_ms)),
        ).fetchall()

        def decode_grid(encoded: str) -> OutcomePriceGridMetadata:
            return OutcomePriceGridMetadata(**json.loads(encoded))

        return [
            OutcomePriceGridChange(
                venue=OutcomeVenue(row["venue"]),
                market_id=row["market_id"],
                timestamp_ms=row["exchange_time_ms"],
                received_time_ms=row["received_time_ms"],
                old_grid=decode_grid(row["old_grid_json"]),
                new_grid=decode_grid(row["new_grid_json"]),
                raw_payload=json.loads(row["raw_payload_json"]),
            )
            for row in rows
        ]

    def load_settlements(
        self,
        venue: OutcomeVenue,
        market_id: str,
    ) -> list[OutcomeSettlementEvidence]:
        rows = self._connect().execute(
            """
            SELECT *
            FROM outcome_settlements
            WHERE venue = ? AND market_id = ?
            ORDER BY settlement_time_ms, received_time_ms, record_id
            """,
            (venue.value, str(market_id)),
        ).fetchall()
        return [
            OutcomeSettlementEvidence(
                venue=OutcomeVenue(row["venue"]),
                market_id=row["market_id"],
                yes_fraction=row["yes_fraction"],
                payout_unit=row["payout_unit"],
                settlement_time_ms=row["settlement_time_ms"],
                received_time_ms=row["received_time_ms"],
                source_event_id=row["source_event_id"],
                evidence_source=row["evidence_source"],
                observed_yes_qty=row["observed_yes_qty"],
                observed_no_qty=row["observed_no_qty"],
                collateral_payout=row["collateral_payout"],
                fee=row["fee"],
                fee_asset=row["fee_asset"],
                raw_payload=json.loads(row["raw_payload_json"]),
            )
            for row in rows
        ]

    def record_verified_coverage(
        self,
        venue: OutcomeVenue,
        market_id: str,
        asset_id: str,
        coverage: VerifiedCoverage,
        *,
        collector_session: str,
    ) -> None:
        if not collector_session:
            raise ValueError("collector_session must not be empty")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO outcome_verified_coverage (
                    venue, market_id, asset_id, start_ms, end_ms,
                    collector_session, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    venue.value,
                    str(market_id),
                    str(asset_id),
                    coverage.start_ms,
                    coverage.end_ms,
                    collector_session,
                    _utc_ms(),
                ),
            )

    def load_verified_coverage(
        self,
        venue: OutcomeVenue,
        market_id: str,
        asset_id: str,
        *,
        start_ms: int,
        end_ms: int,
    ) -> list[VerifiedCoverage]:
        rows = self._connect().execute(
            """
            SELECT start_ms, end_ms
            FROM outcome_verified_coverage
            WHERE venue = ? AND market_id = ? AND asset_id = ?
              AND end_ms > ? AND start_ms < ?
            ORDER BY start_ms, end_ms
            """,
            (venue.value, str(market_id), str(asset_id), int(start_ms), int(end_ms)),
        ).fetchall()
        intervals = [
            VerifiedCoverage(
                start_ms=max(start_ms, row["start_ms"]),
                end_ms=min(end_ms, row["end_ms"]),
            )
            for row in rows
        ]
        if not intervals:
            return []
        merged = [intervals[0]]
        for interval in intervals[1:]:
            previous = merged[-1]
            if interval.start_ms <= previous.end_ms:
                merged[-1] = VerifiedCoverage(
                    start_ms=previous.start_ms,
                    end_ms=max(previous.end_ms, interval.end_ms),
                )
            else:
                merged.append(interval)
        return merged

    def record_gap(
        self,
        venue: OutcomeVenue,
        market_id: str,
        asset_id: str,
        *,
        start_ms: int,
        end_ms: int,
        reason: str,
        collector_session: str | None = None,
    ) -> None:
        if start_ms < 0 or end_ms <= start_ms or not reason.strip():
            raise ValueError("gap must have a valid interval and non-empty reason")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO outcome_collection_gaps (
                    venue, market_id, asset_id, start_ms, end_ms,
                    reason, collector_session, recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    venue.value,
                    str(market_id),
                    str(asset_id),
                    int(start_ms),
                    int(end_ms),
                    reason,
                    collector_session,
                    _utc_ms(),
                ),
            )
