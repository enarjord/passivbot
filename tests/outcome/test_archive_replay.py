from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid
from outcome.archive import OutcomeTradeArchive, authoritative_settlement_evidence
from outcome.archive_replay import (
    build_archived_ema_anchor_replay,
    consolidated_archived_market,
)
from outcome.candles import VerifiedCoverage
from outcome.models import (
    MarketLifecycle,
    NormalizedOutcomeTrade,
    OutcomeOrderSide,
    OutcomeSettlementEvidence,
    OutcomeSide,
    OutcomeVenue,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def hyperliquid_market_with_fixture_constraints(raw_market):
    return replace(
        hyperliquid.normalize_market(raw_market),
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )


def test_full_contract_replay_rejects_midmarket_constraint_changes():
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    initial = hyperliquid_market_with_fixture_constraints(raw_market)
    changed = replace(initial, min_order_qty=initial.min_order_qty + 1.0)

    with pytest.raises(ValueError, match="changing quantity or minimum-order"):
        consolidated_archived_market([initial, changed])


def test_full_contract_replay_requires_metadata_observed_by_trading_open(tmp_path):
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    market = replace(
        hyperliquid_market_with_fixture_constraints(raw_market),
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            trading_close_time_ms=5_000,
            scheduled_event_time_ms=5_000,
        ),
    )
    archive = OutcomeTradeArchive(tmp_path / "late-metadata.sqlite")
    archive.append_market_metadata(
        market,
        observed_at_ms=1_500,
        observation_source="late_import",
    )

    with pytest.raises(ValueError, match="metadata observed by trading open"):
        build_archived_ema_anchor_replay(
            archive,
            venue=market.venue,
            market_id=market.market_id,
            fee_schedule={
                "maker_rate": 0.0,
                "taker_rate": 0.0,
                "formula": "notional",
            },
            requested_collateral=10.0,
            strategy_params={"execution_mode": "accumulate_pairs"},
        )


def test_full_contract_archive_builds_authoritative_settled_replay(tmp_path):
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    market = replace(
        hyperliquid.normalize_market(raw_market),
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            trading_close_time_ms=5_000,
            scheduled_event_time_ms=5_000,
        ),
    )
    archive = OutcomeTradeArchive(tmp_path / "outcomes.sqlite")
    archive.append_market_metadata(
        market,
        observed_at_ms=500,
        observation_source="fixture",
    )
    for index, (asset, side, price, timestamp_ms) in enumerate(
        (
            (market.yes_asset, OutcomeSide.YES, 0.4, 1_500),
            (market.no_asset, OutcomeSide.NO, 0.55, 3_500),
        ),
        start=1,
    ):
        archive.append_trade(
            NormalizedOutcomeTrade(
                venue=OutcomeVenue.HYPERLIQUID,
                market_id=market.market_id,
                asset_id=asset.asset_id,
                outcome=side,
                native_side=OutcomeOrderSide.BUY,
                native_price=price,
                canonical_yes_price=(
                    price if side is OutcomeSide.YES else 1.0 - price
                ),
                qty=2.0,
                exchange_time_ms=timestamp_ms,
                received_time_ms=timestamp_ms + 10,
                source_event_id=f"trade-{index}",
                sequence_id=f"{index:04d}",
            )
        )
    for asset in (market.yes_asset, market.no_asset):
        archive.record_verified_coverage(
            market.venue,
            market.market_id,
            asset.asset_id,
            VerifiedCoverage(1_000, 5_000),
            collector_session="fixture",
        )
    archive.append_settlement(
        OutcomeSettlementEvidence(
            venue=market.venue,
            market_id=market.market_id,
            yes_fraction=1.0,
            payout_unit=1.0,
            settlement_time_ms=5_100,
            capital_release_time_ms=5_100,
            received_time_ms=5_200,
            source_event_id="settlement",
            evidence_source="fixture",
            observed_yes_qty=2.0,
            observed_no_qty=2.0,
            collateral_payout=2.0,
            fee=0.0,
            fee_asset="USDC",
        )
    )

    replay = build_archived_ema_anchor_replay(
        archive,
        venue=market.venue,
        market_id=market.market_id,
        fee_schedule={
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "formula": "notional",
            "settlement_rate": 0.0,
        },
        requested_collateral=100.0,
        strategy_params={"execution_mode": "accumulate_pairs"},
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )

    assert replay.actual_fill_records == 2
    assert replay.coverage == VerifiedCoverage(1_000, 5_000)
    assert replay.settlement.yes_fraction == 1.0
    assert replay.payload["market"]["qty_step"] == 1.0
    assert replay.payload["market"]["min_qty"] == 1.0
    assert replay.payload["market"]["min_notional"] == 10.0
    assert replay.payload["settlement_time_ms"] == 5_100
    assert [row["timestamp_ms"] for row in replay.payload["signal_candles"]] == [
        1_000,
        2_000,
        3_000,
        4_000,
    ]


def test_authoritative_settlement_must_match_market_payout_unit():
    settlement = OutcomeSettlementEvidence(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="binary-1",
        yes_fraction=1.0,
        payout_unit=2.0,
        settlement_time_ms=5_100,
        capital_release_time_ms=5_100,
        received_time_ms=5_200,
        source_event_id="settlement",
        evidence_source="fixture",
        observed_yes_qty=1.0,
        observed_no_qty=0.0,
        collateral_payout=2.0,
        fee=0.0,
        fee_asset="USDC",
    )

    with pytest.raises(ValueError, match="settlement payout unit.*disagrees"):
        authoritative_settlement_evidence(
            [settlement],
            market_id=settlement.market_id,
            payout_unit=1.0,
        )


def test_authoritative_settlement_requires_one_resolution_timestamp():
    settlement = OutcomeSettlementEvidence(
        venue=OutcomeVenue.HYPERLIQUID,
        market_id="binary-1",
        yes_fraction=1.0,
        payout_unit=1.0,
        settlement_time_ms=5_100,
        capital_release_time_ms=5_100,
        received_time_ms=5_200,
        source_event_id="resolution",
        evidence_source="fixture_resolution",
        observed_yes_qty=1.0,
        observed_no_qty=0.0,
        collateral_payout=1.0,
        fee=0.0,
        fee_asset="USDC",
    )
    contradictory = replace(
        settlement,
        settlement_time_ms=5_101,
        capital_release_time_ms=5_200,
        received_time_ms=5_300,
        source_event_id="redemption",
        evidence_source="fixture_redemption",
    )

    with pytest.raises(ValueError, match="conflicting settlement evidence"):
        authoritative_settlement_evidence(
            [settlement, contradictory],
            market_id=settlement.market_id,
            payout_unit=1.0,
        )


def test_replay_merges_later_actual_close_into_initial_market_terms(tmp_path):
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    initial = replace(
        hyperliquid_market_with_fixture_constraints(raw_market),
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            scheduled_event_time_ms=5_000,
            accepting_orders=True,
        ),
    )
    discovery = replace(initial, min_order_qty=2.0)
    closed = replace(
        initial,
        lifecycle=replace(
            initial.lifecycle,
            trading_close_time_ms=5_000,
            accepting_orders=False,
        ),
    )
    archive = OutcomeTradeArchive(tmp_path / "lifecycle.sqlite")
    archive.append_market_metadata(
        discovery,
        observed_at_ms=100,
        observation_source="discovery",
    )
    archive.append_market_metadata(
        initial,
        observed_at_ms=500,
        observation_source="live",
    )
    archive.append_market_metadata(
        closed,
        observed_at_ms=5_100,
        observation_source="closed",
    )
    for asset in (initial.yes_asset, initial.no_asset):
        archive.record_verified_coverage(
            initial.venue,
            initial.market_id,
            asset.asset_id,
            VerifiedCoverage(1_000, 5_000),
            collector_session="fixture",
        )
    archive.append_trade(
        NormalizedOutcomeTrade(
            venue=initial.venue,
            market_id=initial.market_id,
            asset_id=initial.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.5,
            canonical_yes_price=0.5,
            qty=1.0,
            exchange_time_ms=1_500,
            received_time_ms=1_600,
            source_event_id="trade",
        )
    )
    archive.append_settlement(
        OutcomeSettlementEvidence(
            venue=initial.venue,
            market_id=initial.market_id,
            yes_fraction=0.0,
            payout_unit=1.0,
            settlement_time_ms=5_200,
            capital_release_time_ms=5_200,
            received_time_ms=5_300,
            source_event_id="settlement",
            evidence_source="fixture",
            observed_yes_qty=0.0,
            observed_no_qty=0.0,
            collateral_payout=0.0,
            fee=0.0,
            fee_asset="USDC",
        )
    )

    replay = build_archived_ema_anchor_replay(
        archive,
        venue=initial.venue,
        market_id=initial.market_id,
        fee_schedule={
            "maker_rate": 0.0,
            "taker_rate": 0.0,
            "formula": "notional",
            "settlement_rate": 0.0,
        },
        requested_collateral=10.0,
        strategy_params={"execution_mode": "accumulate_pairs"},
    )

    assert replay.market.lifecycle.trading_close_time_ms == 5_000
    assert replay.market.lifecycle.accepting_orders is False
    assert replay.market.min_order_qty == 1.0


def test_replay_does_not_treat_resolution_as_polymarket_capital_release(tmp_path):
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    market = replace(
        hyperliquid_market_with_fixture_constraints(raw_market),
        venue=OutcomeVenue.POLYMARKET,
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            trading_close_time_ms=5_000,
            scheduled_event_time_ms=5_000,
        ),
    )
    archive = OutcomeTradeArchive(tmp_path / "resolution-only.sqlite")
    archive.append_market_metadata(
        market,
        observed_at_ms=500,
        observation_source="fixture",
    )
    archive.append_settlement(
        OutcomeSettlementEvidence(
            venue=market.venue,
            market_id=market.market_id,
            yes_fraction=1.0,
            payout_unit=1.0,
            settlement_time_ms=5_100,
            capital_release_time_ms=None,
            received_time_ms=5_200,
            source_event_id="condition-resolution",
            evidence_source="polymarket_ctf_condition_resolution",
            observed_yes_qty=0.0,
            observed_no_qty=0.0,
            collateral_payout=0.0,
            fee=0.0,
            fee_asset="USDC",
        )
    )

    with pytest.raises(ValueError, match="no authoritative capital release"):
        build_archived_ema_anchor_replay(
            archive,
            venue=market.venue,
            market_id=market.market_id,
            fee_schedule={"maker_rate": 0.0, "taker_rate": 0.0, "formula": "notional"},
            requested_collateral=10.0,
            strategy_params={"execution_mode": "accumulate_pairs"},
        )


def test_polymarket_replay_requires_separate_verified_price_grid_history(tmp_path):
    raw_market = json.loads(
        (FIXTURES / "hyperliquid_price_binary.json").read_text()
    )
    market = replace(
        hyperliquid_market_with_fixture_constraints(raw_market),
        venue=OutcomeVenue.POLYMARKET,
        lifecycle=MarketLifecycle(
            trading_open_time_ms=1_000,
            trading_close_time_ms=5_000,
            scheduled_event_time_ms=5_000,
        ),
    )
    archive = OutcomeTradeArchive(tmp_path / "grid-coverage.sqlite")
    archive.append_market_metadata(
        market,
        observed_at_ms=500,
        observation_source="fixture",
    )
    archive.append_trade(
        NormalizedOutcomeTrade(
            venue=market.venue,
            market_id=market.market_id,
            asset_id=market.yes_asset.asset_id,
            outcome=OutcomeSide.YES,
            native_side=OutcomeOrderSide.BUY,
            native_price=0.5,
            canonical_yes_price=0.5,
            qty=1.0,
            exchange_time_ms=1_500,
            received_time_ms=1_600,
            source_event_id="trade",
        )
    )
    for asset in (market.yes_asset, market.no_asset):
        archive.record_verified_coverage(
            market.venue,
            market.market_id,
            asset.asset_id,
            VerifiedCoverage(1_000, 5_000),
            collector_session="fills",
        )
    archive.append_settlement(
        OutcomeSettlementEvidence(
            venue=market.venue,
            market_id=market.market_id,
            yes_fraction=1.0,
            payout_unit=1.0,
            settlement_time_ms=5_100,
            capital_release_time_ms=5_200,
            received_time_ms=5_300,
            source_event_id="redemption",
            evidence_source="fixture_redemption",
            observed_yes_qty=1.0,
            observed_no_qty=0.0,
            collateral_payout=1.0,
            fee=0.0,
            fee_asset="USDC",
        )
    )
    kwargs = {
        "venue": market.venue,
        "market_id": market.market_id,
        "fee_schedule": {"maker_rate": 0.0, "taker_rate": 0.0, "formula": "notional"},
        "requested_collateral": 10.0,
        "strategy_params": {"execution_mode": "accumulate_pairs"},
    }

    with pytest.raises(ValueError, match="price-grid coverage"):
        build_archived_ema_anchor_replay(archive, **kwargs)

    archive.record_verified_price_grid_coverage(
        market.venue,
        market.market_id,
        VerifiedCoverage(1_000, 5_000),
        collector_session="grid",
    )
    replay = build_archived_ema_anchor_replay(archive, **kwargs)

    assert replay.payload["settlement_time_ms"] == 5_100
    assert replay.settlement.capital_release_time_ms == 5_200
