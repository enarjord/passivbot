from __future__ import annotations

import asyncio

import ccxt
import ccxt.prediction as ccxt_prediction
import pytest

from outcome.ccxt_transport import create_ccxt_prediction_client
from outcome.models import OutcomeVenue


def test_prediction_connectors_exist_on_pinned_ccxt_release():
    assert ccxt.__version__ == "4.5.68"
    assert callable(ccxt_prediction.hyperliquid)
    assert callable(ccxt_prediction.polymarket)


@pytest.mark.parametrize(
    ("venue", "expected_type"),
    [
        (OutcomeVenue.HYPERLIQUID, ccxt_prediction.hyperliquid),
        (OutcomeVenue.POLYMARKET, ccxt_prediction.polymarket),
    ],
)
def test_prediction_client_factory_disables_builder_attribution_by_default(
    venue, expected_type
):
    source_config = {"options": {"custom": "retained"}}

    client = create_ccxt_prediction_client(venue, source_config)

    assert isinstance(client, expected_type)
    assert client.options["custom"] == "retained"
    assert client.options["builderFee"] is False
    assert source_config == {"options": {"custom": "retained"}}


def test_prediction_client_factory_requires_explicit_builder_attribution_opt_in():
    client = create_ccxt_prediction_client(
        "hyperliquid",
        {"options": {"builderFee": True, "feeRate": "0%"}},
        allow_builder_attribution=True,
    )

    assert client.options["builderFee"] is True
    assert client.options["feeRate"] == "0%"


def test_hyperliquid_prediction_initialization_cannot_approve_builder_by_default(
    monkeypatch,
):
    client = create_ccxt_prediction_client("hyperliquid")
    approvals: list[tuple[str, str]] = []

    async def fake_load_markets():
        return {}

    async def fake_approve_builder_fee(builder: str, fee_rate: str):
        approvals.append((builder, fee_rate))

    monkeypatch.setattr(client, "load_markets", fake_load_markets)
    monkeypatch.setattr(client, "approve_builder_fee", fake_approve_builder_fee)

    asyncio.run(client.initialize_client())

    assert approvals == []


def test_prediction_client_factory_rejects_non_mapping_options():
    with pytest.raises(TypeError, match="options must be a mapping"):
        create_ccxt_prediction_client("polymarket", {"options": []})
