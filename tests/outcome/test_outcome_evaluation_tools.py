from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid, polymarket
from outcome.models import OutcomeFeeMetadata
from tools.evaluate_archived_outcome_portfolio import _rust_fee_formula


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_archived_polymarket_fee_curve_maps_to_probability_variance():
    market = polymarket.normalize_market(fixture("polymarket_binary.json"))

    assert _rust_fee_formula(market, "archived") == "probability_variance"


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
