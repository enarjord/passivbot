from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.archive_replay import load_verified_trade_window
from outcome.backtest_input import build_trade_derived_ema_anchor_input
from outcome.candles import VerifiedCoverage
from outcome.evaluation import ema_warmup_observations
from outcome.models import (
    MarketLifecycle,
    OutcomeFeeMetadata,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
)
from tools.evaluate_archived_outcome_portfolio import (
    _add_constraint_arguments as _add_portfolio_constraint_arguments,
    _load_archived_fee_market,
    _require_shared_quote_asset,
    _rust_fee_formula,
    _rust_fee_rates,
    _validate_constraint_arguments as _validate_portfolio_constraint_arguments,
)
from tools.evaluate_hip4_outcome_window import (
    _add_constraint_arguments,
    _add_window_phase_arguments,
    _window_market_spec,
)
from tools.evaluate_polymarket_outcome_window import (
    _add_constraint_arguments as _add_polymarket_constraint_arguments,
    _evaluation_assumptions as _polymarket_evaluation_assumptions,
    _load_archived_market_and_grid_window,
    _require_fee_free_market,
    _window_market_spec as _polymarket_window_market_spec,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize(
    ("span_seconds", "candle_interval_seconds", "expected"),
    (
        (1.0, 1.0, 1),
        (1.9, 1.0, 2),
        (2.0, 1.0, 2),
        (2.1, 1.0, 3),
        (2.1, 0.5, 5),
    ),
)
def test_ema_warmup_observations_cover_full_span(
    span_seconds,
    candle_interval_seconds,
    expected,
):
    assert (
        ema_warmup_observations(
            span_seconds,
            candle_interval_seconds=candle_interval_seconds,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("span_seconds", "candle_interval_seconds"),
    (
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1.0, 0.0),
        (1.0, float("nan")),
    ),
)
def test_ema_warmup_observations_reject_invalid_durations(
    span_seconds,
    candle_interval_seconds,
):
    with pytest.raises(ValueError, match="finite and positive"):
        ema_warmup_observations(
            span_seconds,
            candle_interval_seconds=candle_interval_seconds,
        )


def test_archived_polymarket_fee_curve_maps_to_probability_variance():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))

    assert _rust_fee_formula(market, "archived") == "probability_variance"
    with pytest.raises(ValueError, match="explicitly fee-free"):
        _require_fee_free_market(market)

    _require_fee_free_market(
        replace(
            market,
            fee_metadata=OutcomeFeeMetadata(
                formula="venue_reported_zero",
                maker_rate=0.0,
                taker_rate=0.0,
            ),
        )
    )


def test_archived_fee_formula_fails_closed_when_not_representable():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    unsupported = replace(
        market,
        fee_metadata=OutcomeFeeMetadata(
            formula="polymarket_probability_curve",
            parameters={"exponent": 2.0},
        ),
    )

    with pytest.raises(ValueError, match="exponent is not supported"):
        _rust_fee_formula(unsupported, "archived")

    hip4 = hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json"))
    with pytest.raises(ValueError, match="no Rust translation"):
        _rust_fee_formula(hip4, "archived")
    assert _rust_fee_formula(hip4, "notional") == "notional"


def test_archived_zero_fee_metadata_forces_zero_rust_rates():
    market = replace(
        polymarket.normalize_market(fixture("polymarket_binary.json")),
        fee_metadata=OutcomeFeeMetadata(
            formula="venue_reported_zero",
            maker_rate=0.0,
            taker_rate=0.0,
        ),
    )

    assert _rust_fee_formula(market, "archived") == "notional"
    assert _rust_fee_rates(
        market,
        "archived",
        maker_rate=0.01,
        taker_rate=0.02,
        settlement_rate=0.03,
    ) == (0.0, 0.0, 0.0)
    assert _rust_fee_rates(
        market,
        "notional",
        maker_rate=0.01,
        taker_rate=0.02,
        settlement_rate=0.03,
    ) == (0.01, 0.02, 0.03)


def test_shared_wallet_portfolio_requires_one_quote_asset():
    replay = argparse.Namespace(
        market=polymarket.normalize_market(fixture("polymarket_binary.json"))
    )
    other_quote_asset = argparse.Namespace(
        market=replace(replay.market, quote_asset="pUSD")
    )

    assert _require_shared_quote_asset([replay]) == replay.market.quote_asset
    with pytest.raises(ValueError, match="requires one shared quote asset"):
        _require_shared_quote_asset([replay, other_quote_asset])


def test_archived_portfolio_accepts_explicit_missing_constraint_assumptions():
    parser = argparse.ArgumentParser()
    _add_portfolio_constraint_arguments(parser)
    args = parser.parse_args(
        [
            "--qty-step",
            "1",
            "--min-order-qty",
            "2",
            "--min-order-notional",
            "0",
        ]
    )

    _validate_portfolio_constraint_arguments(parser, args)

    assert args.qty_step == 1.0
    assert args.min_order_qty == 2.0
    assert args.min_order_notional == 0.0


def test_archived_fee_formula_uses_opening_state_and_rejects_later_transition(
    tmp_path,
):
    discovered = polymarket.normalize_market(fixture("polymarket_binary.json"))
    lifecycle = MarketLifecycle(
        trading_open_time_ms=2_000,
        trading_close_time_ms=5_000,
        scheduled_event_time_ms=5_000,
    )
    fee_free = replace(
        discovered,
        lifecycle=lifecycle,
        fee_metadata=OutcomeFeeMetadata(
            formula="venue_reported_zero",
            maker_rate=0.0,
            taker_rate=0.0,
        ),
    )
    opening = replace(discovered, lifecycle=lifecycle)
    archive = OutcomeTradeArchive(tmp_path / "fee-history.sqlite")
    archive.append_market_metadata(
        fee_free,
        observed_at_ms=500,
        observation_source="discovery",
    )
    archive.append_market_metadata(
        opening,
        observed_at_ms=1_500,
        observation_source="opening",
    )

    selected = _load_archived_fee_market(
        archive,
        venue=opening.venue,
        market_id=opening.market_id,
        fee_formula="archived",
    )
    assert selected.fee_metadata == opening.fee_metadata

    archive.append_market_metadata(
        fee_free,
        observed_at_ms=2_500,
        observation_source="fee_transition",
    )
    with pytest.raises(ValueError, match="fee transitions are not supported"):
        _load_archived_fee_market(
            archive,
            venue=opening.venue,
            market_id=opening.market_id,
            fee_formula="archived",
        )


def test_polymarket_window_uses_start_metadata_and_archived_grid_changes(tmp_path):
    discovered = polymarket.normalize_market(fixture("polymarket_binary.json"))
    fee_free = OutcomeFeeMetadata(
        formula="venue_reported_zero",
        maker_rate=0.0,
        taker_rate=0.0,
    )
    start_market = replace(discovered, fee_metadata=fee_free)
    current_market = replace(
        start_market,
        price_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001),
    )
    change = OutcomePriceGridChange(
        venue=start_market.venue,
        market_id=start_market.market_id,
        timestamp_ms=3_000,
        received_time_ms=3_100,
        old_grid=start_market.price_grid,
        new_grid=current_market.price_grid,
        raw_payload={"event_type": "tick_size_change"},
    )
    archive = OutcomeTradeArchive(tmp_path / "polymarket-window.sqlite")
    archive.append_market_metadata(
        start_market,
        observed_at_ms=1_000,
        observation_source="gamma",
    )
    archive.append_market_metadata(
        current_market,
        observed_at_ms=3_100,
        observation_source="gamma",
    )
    archive.append_price_grid_change(change, collector_session="grid")
    archive.record_verified_price_grid_coverage(
        start_market.venue,
        start_market.market_id,
        VerifiedCoverage(1_000, 5_000),
        collector_session="grid",
    )

    market, changes = _load_archived_market_and_grid_window(
        archive,
        current_market,
        start_ms=2_000,
        end_ms=5_000,
    )

    assert market.price_grid == start_market.price_grid
    assert changes == [change]


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("min_order_qty", 99.0),
        (
            "fee_metadata",
            OutcomeFeeMetadata(
                formula="notional",
                maker_rate=0.001,
                taker_rate=0.002,
            ),
        ),
    ),
)
def test_polymarket_window_rejects_unmodeled_metadata_transitions(
    tmp_path,
    changed_field,
    changed_value,
):
    discovered = polymarket.normalize_market(fixture("polymarket_binary.json"))
    opening_market = replace(
        discovered,
        fee_metadata=OutcomeFeeMetadata(
            formula="venue_reported_zero",
            maker_rate=0.0,
            taker_rate=0.0,
        ),
    )
    changed_market = replace(opening_market, **{changed_field: changed_value})
    archive = OutcomeTradeArchive(tmp_path / f"polymarket-{changed_field}.sqlite")
    archive.append_market_metadata(
        opening_market,
        observed_at_ms=1_000,
        observation_source="gamma",
    )
    archive.append_market_metadata(
        changed_market,
        observed_at_ms=3_000,
        observation_source="gamma",
    )
    archive.record_verified_price_grid_coverage(
        opening_market.venue,
        opening_market.market_id,
        VerifiedCoverage(1_000, 5_000),
        collector_session="grid",
    )

    with pytest.raises(ValueError, match=f"metadata transitions: {changed_field}"):
        _load_archived_market_and_grid_window(
            archive,
            changed_market,
            start_ms=2_000,
            end_ms=5_000,
        )


def test_polymarket_window_reports_all_explicit_constraint_assumptions():
    assumptions = _polymarket_evaluation_assumptions(
        qty_step=0.25,
        min_order_notional=2.5,
    )

    assert assumptions["qty_step"] == 0.25
    assert assumptions["min_order_notional"] == 2.5


def test_polymarket_window_applies_latest_pre_window_grid_change(tmp_path):
    discovered = polymarket.normalize_market(fixture("polymarket_binary.json"))
    fee_free = OutcomeFeeMetadata(
        formula="venue_reported_zero",
        maker_rate=0.0,
        taker_rate=0.0,
    )
    opening_market = replace(discovered, fee_metadata=fee_free)
    changed_market = replace(
        opening_market,
        price_grid=OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001),
    )
    change = OutcomePriceGridChange(
        venue=opening_market.venue,
        market_id=opening_market.market_id,
        timestamp_ms=3_000,
        received_time_ms=3_100,
        old_grid=opening_market.price_grid,
        new_grid=changed_market.price_grid,
        raw_payload={"event_type": "tick_size_change"},
    )
    archive = OutcomeTradeArchive(tmp_path / "polymarket-pre-window-grid.sqlite")
    archive.append_market_metadata(
        opening_market,
        observed_at_ms=1_000,
        observation_source="gamma",
    )
    archive.append_market_metadata(
        changed_market,
        observed_at_ms=5_000,
        observation_source="gamma",
    )
    archive.append_price_grid_change(change, collector_session="grid")
    archive.record_verified_price_grid_coverage(
        opening_market.venue,
        opening_market.market_id,
        VerifiedCoverage(1_000, 6_000),
        collector_session="grid",
    )

    market, changes = _load_archived_market_and_grid_window(
        archive,
        changed_market,
        start_ms=4_000,
        end_ms=6_000,
    )

    assert market.price_grid == changed_market.price_grid
    assert changes == []


def test_polymarket_window_rejects_discontinuous_pre_window_grid_changes(
    tmp_path,
):
    discovered = polymarket.normalize_market(fixture("polymarket_binary.json"))
    fee_free = OutcomeFeeMetadata(
        formula="venue_reported_zero",
        maker_rate=0.0,
        taker_rate=0.0,
    )
    opening_market = replace(discovered, fee_metadata=fee_free)
    first_grid = OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.001)
    unrelated_grid = OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.002)
    final_grid = OutcomePriceGridMetadata(kind="fixed_step", fixed_step=0.0001)
    archive = OutcomeTradeArchive(tmp_path / "polymarket-broken-grid-chain.sqlite")
    archive.append_market_metadata(
        opening_market,
        observed_at_ms=1_000,
        observation_source="gamma",
    )
    archive.append_price_grid_change(
        OutcomePriceGridChange(
            venue=opening_market.venue,
            market_id=opening_market.market_id,
            timestamp_ms=2_000,
            received_time_ms=2_100,
            old_grid=opening_market.price_grid,
            new_grid=first_grid,
            raw_payload={"event_type": "tick_size_change"},
        ),
        collector_session="grid",
    )
    archive.append_price_grid_change(
        OutcomePriceGridChange(
            venue=opening_market.venue,
            market_id=opening_market.market_id,
            timestamp_ms=3_000,
            received_time_ms=3_100,
            old_grid=unrelated_grid,
            new_grid=final_grid,
            raw_payload={"event_type": "tick_size_change"},
        ),
        collector_session="grid",
    )
    archive.record_verified_price_grid_coverage(
        opening_market.venue,
        opening_market.market_id,
        VerifiedCoverage(1_000, 6_000),
        collector_session="grid",
    )

    with pytest.raises(ValueError, match="do not form a continuous chain"):
        _load_archived_market_and_grid_window(
            archive,
            opening_market,
            start_ms=4_000,
            end_ms=6_000,
        )


def test_window_uses_covered_preceding_fill_as_signal_seed(tmp_path):
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    seed = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.yes_asset.asset_id,
            "price": "0.40",
            "size": "2",
            "side": "BUY",
            "timestamp": "1500",
        },
        market,
        received_time_ms=1_600,
        collector_sequence=1,
    )
    in_window = polymarket.normalize_market_ws_trade(
        {
            "event_type": "last_trade_price",
            "market": market.market_id,
            "asset_id": market.no_asset.asset_id,
            "price": "0.55",
            "size": "3",
            "side": "SELL",
            "timestamp": "4200",
        },
        market,
        received_time_ms=4_300,
        collector_sequence=2,
    )
    archive = OutcomeTradeArchive(tmp_path / "window-seed.sqlite")
    archive.append_trade(seed, collector_session="fills")
    archive.append_trade(in_window, collector_session="fills")
    for asset in (market.yes_asset, market.no_asset):
        archive.record_verified_coverage(
            market.venue,
            market.market_id,
            asset.asset_id,
            VerifiedCoverage(1_000, 6_000),
            collector_session="fills",
        )

    trades, coverage = load_verified_trade_window(
        archive,
        market,
        start_ms=3_000,
        end_ms=6_000,
    )
    payload = build_trade_derived_ema_anchor_input(
        market_spec={"market_id": market.market_id},
        trades=trades,
        verified_coverage=(coverage,),
        fee_schedule={},
        starting_collateral=100.0,
        strategy_params={},
        settlement_time_ms=6_000,
        yes_fraction=0.0,
        candle_start_ms=3_000,
    )

    assert coverage == VerifiedCoverage(1_000, 6_000)
    assert [candle["timestamp_ms"] for candle in payload["signal_candles"]] == [
        3_000,
        4_000,
        5_000,
    ]
    assert [candle["volume"] for candle in payload["signal_candles"]] == [
        0.0,
        3.0,
        0.0,
    ]
    assert payload["signal_candles"][0]["close"] == pytest.approx(0.4)
    assert [
        (candle["timestamp_ms"], candle["outcome"])
        for candle in payload["execution_candles"]
    ] == [(4_000, "no")]


def test_hip4_window_uses_requested_synthetic_lifecycle_boundaries():
    market = replace(
        hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json")),
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )

    spec = _window_market_spec(
        market,
        start_ms=2_000,
        end_ms=5_000,
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )

    assert spec["trading_opens_ms"] == 2_000
    assert spec["order_entry_opens_ms"] == 2_000
    assert spec["trading_closes_ms"] == 5_000
    assert spec["scheduled_event_ms"] == 5_000
    assert spec["min_notional"] == 10.0


def test_polymarket_window_applies_synthetic_close_before_rust_translation():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))
    active_market = replace(
        market,
        lifecycle=replace(
            market.lifecycle,
            trading_close_time_ms=None,
        ),
    )

    spec = _polymarket_window_market_spec(
        active_market,
        start_ms=2_000,
        end_ms=5_000,
        qty_step=0.01,
        min_order_notional=0.0,
    )

    assert spec["trading_opens_ms"] == 2_000
    assert spec["order_entry_opens_ms"] == 2_000
    assert spec["trading_closes_ms"] == 5_000
    assert spec["scheduled_event_ms"] == 5_000
    assert spec["min_notional"] == 0.0


def test_hip4_window_close_phases_default_to_disabled_and_are_configurable():
    parser = argparse.ArgumentParser()
    _add_window_phase_arguments(parser)

    defaults = parser.parse_args([])
    configured = parser.parse_args(
        [
            "--risk-reduction-only-ms-before-close",
            "30000",
            "--entry-cutoff-ms-before-close",
            "5000",
        ]
    )

    assert defaults.risk_reduction_only_ms_before_close == 0
    assert defaults.entry_cutoff_ms_before_close == 0
    assert configured.risk_reduction_only_ms_before_close == 30_000
    assert configured.entry_cutoff_ms_before_close == 5_000


def test_hip4_window_requires_explicit_quantity_constraint_assumptions():
    parser = argparse.ArgumentParser()
    _add_constraint_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    configured = parser.parse_args(
        [
            "--qty-step",
            "1",
            "--min-order-qty",
            "10",
            "--min-order-notional",
            "5",
        ]
    )

    assert configured.qty_step == 1.0
    assert configured.min_order_qty == 10.0
    assert configured.min_order_notional == 5.0


def test_polymarket_window_requires_explicit_quantity_and_notional_assumptions():
    parser = argparse.ArgumentParser()
    _add_polymarket_constraint_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--qty-step", "0.01"])
    configured = parser.parse_args(
        ["--qty-step", "0.01", "--min-order-notional", "0"]
    )

    assert configured.qty_step == 0.01
    assert configured.min_order_notional == 0.0
