from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.archive import OutcomeTradeArchive
from outcome.candles import VerifiedCoverage
from outcome.models import (
    OutcomeFeeMetadata,
    OutcomePriceGridChange,
    OutcomePriceGridMetadata,
)
from tools.evaluate_archived_outcome_portfolio import _rust_fee_formula
from tools.evaluate_hip4_outcome_window import (
    _add_window_phase_arguments,
    _window_market_spec,
)
from tools.evaluate_polymarket_outcome_window import (
    _load_archived_market_and_grid_window,
    _require_fee_free_market,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


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


def test_hip4_window_uses_requested_synthetic_lifecycle_boundaries():
    market = replace(
        hyperliquid.normalize_market(fixture("hyperliquid_price_binary.json")),
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )

    spec = _window_market_spec(market, start_ms=2_000, end_ms=5_000)

    assert spec["trading_opens_ms"] == 2_000
    assert spec["order_entry_opens_ms"] == 2_000
    assert spec["trading_closes_ms"] == 5_000
    assert spec["scheduled_event_ms"] == 5_000


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
