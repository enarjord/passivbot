from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from outcome.adapters import hyperliquid
from outcome.hyperliquid_live import (
    HyperliquidOutcomeLifecycleState,
    HyperliquidOutcomeLiveClient,
    OutcomeCreateDeadlineExpired,
    OutcomeMutationDisabled,
)
from outcome.models import OutcomeOrderSide, OutcomeSide


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "outcome"


def market_fixture():
    payload = json.loads((FIXTURES / "hyperliquid_price_binary.json").read_text())
    market = hyperliquid.normalize_market(payload)
    return payload, replace(
        market,
        qty_step=1.0,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )


def spot_state():
    return {
        "balances": [
            {
                "coin": "USDC",
                "entryNtl": "0.0",
                "hold": "5.0",
                "token": 0,
                "total": "100.0",
            },
            {
                "coin": "+9130",
                "entryNtl": "4.2",
                "hold": "2",
                "total": "12",
            },
            {
                "coin": "+9990",
                "entryNtl": "1",
                "hold": "0",
                "total": "1",
            },
        ],
        "tokenToAvailableAfterMaintenance": [[0, "90.0"]],
    }


def open_orders():
    return [
        {
            "coin": "#9131",
            "limitPx": "0.6",
            "oid": 123,
            "origSz": "20",
            "side": "B",
            "sz": "15",
            "timestamp": 1784900000000,
            "cloid": "0x00000000000000000000000000000001",
        },
        {
            "coin": "#9990",
            "limitPx": "0.5",
            "oid": 999,
            "origSz": "20",
            "side": "B",
            "sz": "20",
            "timestamp": 1784900000000,
        },
    ]


def user_fills():
    return [
        {
            "coin": "#9130",
            "crossed": False,
            "dir": "Buy",
            "fee": "0.0",
            "feeToken": "+9130",
            "hash": "0xabc",
            "oid": 456,
            "px": "0.4",
            "side": "B",
            "startPosition": "0",
            "sz": "25",
            "tid": 789,
            "time": 1784900000100,
        },
        {
            "coin": "#9990",
            "crossed": True,
            "dir": "Buy",
            "fee": "0.0",
            "feeToken": "+9990",
            "hash": "0xdef",
            "oid": 999,
            "px": "0.5",
            "side": "B",
            "startPosition": "0",
            "sz": "20",
            "tid": 999,
            "time": 1784900000100,
        },
    ]


def user_fill_with_rebate():
    fill = dict(user_fills()[0])
    fill["fee"] = "-0.001"
    return [fill]


def settlement_fills(market, *, yes_fraction: float):
    timestamp_ms = market.lifecycle.scheduled_event_time_ms + 1_000
    return [
        {
            "coin": market.yes_asset.market_data_symbol,
            "crossed": True,
            "dir": "Settlement",
            "fee": "0.0",
            "feeToken": "USDC",
            "hash": "0xsettlement",
            "oid": 1001,
            "px": str(yes_fraction),
            "side": "A",
            "startPosition": "12",
            "sz": "12",
            "tid": 2001,
            "time": timestamp_ms,
        },
        {
            "coin": market.no_asset.market_data_symbol,
            "crossed": True,
            "dir": "Settlement",
            "fee": "0.0",
            "feeToken": "USDC",
            "hash": "0xsettlement",
            "oid": 1002,
            "px": str(1.0 - yes_fraction),
            "side": "A",
            "startPosition": "3",
            "sz": "3",
            "tid": 2002,
            "time": timestamp_ms,
        },
    ]


def user_fees():
    return {
        "userAddRate": "0.000105",
        "userCrossRate": "0.000315",
        "userSpotAddRate": "0.00028",
        "userSpotCrossRate": "0.00049",
    }


def book():
    return {
        "coin": "#9130",
        "time": 1784900000200,
        "levels": [
            [
                {"px": "0.39", "sz": "50", "n": 2},
                {"px": "0.38", "sz": "75", "n": 1},
            ],
            [
                {"px": "0.41", "sz": "40", "n": 1},
                {"px": "0.42", "sz": "60", "n": 3},
            ],
        ],
    }


class FakeSession:
    def __init__(self, outcome_meta):
        self.outcome_meta = outcome_meta
        self.meta_outcomes = [outcome_meta]
        self.user_fills_payload = user_fills()
        self.user_fills_by_time_payload = user_fills()
        self.private_requests = []
        self.public_requests = []
        self.signed = []

    async def publicPostInfo(self, payload):
        self.public_requests.append(payload)
        request_type = payload["type"]
        if request_type == "spotClearinghouseState":
            return spot_state()
        if request_type == "frontendOpenOrders":
            return open_orders()
        if request_type == "userFills":
            return self.user_fills_payload
        if request_type == "userFillsByTime":
            if isinstance(self.user_fills_by_time_payload, Exception):
                raise self.user_fills_by_time_payload
            return self.user_fills_by_time_payload
        if request_type == "userFees":
            return user_fees()
        if request_type == "outcomeMeta":
            return {"outcomes": self.meta_outcomes, "questions": []}
        if request_type == "l2Book":
            return book()
        raise AssertionError(f"unexpected public request {payload!r}")

    async def privatePostExchange(self, payload):
        self.private_requests.append(payload)
        if payload["action"]["type"] == "cancel":
            return {
                "status": "ok",
                "response": {
                    "type": "cancel",
                    "data": {"statuses": ["success"]},
                },
            }
        return {
            "status": "ok",
            "response": {
                "type": "order",
                "data": {"statuses": [{"resting": {"oid": 777}}]},
            },
        }

    def milliseconds(self):
        return 1784900000300

    def sign_l1_action(self, action, nonce, vault_address=None, expires_after=None):
        self.signed.append((action, nonce, vault_address, expires_after))
        return {"r": "0x1", "s": "0x2", "v": 27}


def test_hip4_action_builder_uses_official_asset_id_and_strict_constraints():
    _, market = market_fixture()
    action = hyperliquid.build_limit_order_action(
        market,
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price="0.4",
        qty="25",
        client_order_id="0x00000000000000000000000000000001",
    )
    assert action == {
        "type": "order",
        "orders": [
            {
                "a": 100009130,
                "b": True,
                "p": "0.4",
                "s": "25",
                "r": False,
                "t": {"limit": {"tif": "Alo"}},
                "c": "0x00000000000000000000000000000001",
            }
        ],
        "grouping": "na",
    }
    with pytest.raises(ValueError, match="notional"):
        hyperliquid.build_limit_order_action(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="24",
        )
    with pytest.raises(ValueError, match="significant"):
        hyperliquid.build_limit_order_action(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.400001",
            qty="25",
        )


def test_hip4_action_builder_fails_closed_without_authoritative_quantity_constraints():
    payload = json.loads((FIXTURES / "hyperliquid_price_binary.json").read_text())
    market = hyperliquid.normalize_market(payload)

    with pytest.raises(ValueError, match="quantity step is unavailable"):
        hyperliquid.build_limit_order_action(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="25",
        )


@pytest.mark.asyncio
async def test_account_snapshot_keeps_outcome_state_separate_and_surfaces_unknown_assets():
    payload, market = market_fixture()
    client = HyperliquidOutcomeLiveClient(
        FakeSession(payload),
        account_address="0xaccount",
    )
    snapshot = await client.fetch_account_snapshot((market,))

    assert snapshot.collateral.unheld == pytest.approx(95.0)
    assert snapshot.collateral.conservative_available == pytest.approx(90.0)
    assert snapshot.fee_rates.user_add_rate == pytest.approx(0.000105)
    assert snapshot.fee_rates.user_spot_add_rate == pytest.approx(0.00028)
    assert snapshot.fee_rates.conservative_maker_rate == pytest.approx(0.00028)
    assert snapshot.fee_rates.conservative_taker_rate == pytest.approx(0.00049)
    yes = snapshot.balance("913", OutcomeSide.YES)
    no = snapshot.balance("913", OutcomeSide.NO)
    assert yes.total_qty == 12.0
    assert yes.available_qty == 10.0
    assert no.total_qty == 0.0
    assert snapshot.open_orders[0].outcome is OutcomeSide.NO
    assert snapshot.recent_fills[0].fee == 0.0
    assert snapshot.recent_fills[0].is_maker is True
    assert snapshot.unknown_outcome_balance_coins == ("+9990",)
    assert snapshot.unknown_outcome_order_coins == ("#9990",)
    assert snapshot.unknown_outcome_fill_coins == ("#9990",)


@pytest.mark.parametrize(
    "maintenance",
    [
        None,
        [],
        [[1, "90.0"]],
        [[0, "90.0"], [0, "80.0"]],
    ],
    ids=[
        "field-absent",
        "empty-array",
        "quote-token-omitted",
        "quote-token-duplicated",
    ],
)
def test_collateral_rejects_missing_quote_maintenance_availability(maintenance):
    state = spot_state()
    if maintenance is None:
        del state["tokenToAvailableAfterMaintenance"]
    else:
        state["tokenToAvailableAfterMaintenance"] = maintenance

    with pytest.raises(ValueError, match="maintenance availability"):
        hyperliquid.normalize_collateral_balance(state, quote_asset="USDC")


def test_collateral_rejects_quote_balance_without_token_identifier():
    state = spot_state()
    del state["balances"][0]["token"]

    with pytest.raises(ValueError, match="quote balance must contain a token identifier"):
        hyperliquid.normalize_collateral_balance(state, quote_asset="USDC")


@pytest.mark.asyncio
async def test_account_snapshot_timestamp_is_conservative_across_concurrent_reads(
    monkeypatch,
):
    payload, market = market_fixture()
    clock = {"seconds": 1_784_900_000.0}

    class UnevenSession(FakeSession):
        async def publicPostInfo(self, request):
            result = await super().publicPostInfo(request)
            if request["type"] == "userFees":
                clock["seconds"] += 20.0
            return result

    monkeypatch.setattr(
        "outcome.hyperliquid_live.time.time",
        lambda: clock["seconds"],
    )
    client = HyperliquidOutcomeLiveClient(
        UnevenSession(payload),
        account_address="0xaccount",
    )

    snapshot = await client.fetch_account_snapshot((market,))

    assert snapshot.received_time_ms == 1_784_900_000_000
    assert clock["seconds"] == 1_784_900_020.0


@pytest.mark.asyncio
async def test_account_snapshot_rejects_missing_required_user_fee_rate():
    payload, market = market_fixture()
    session = FakeSession(payload)
    original = session.publicPostInfo

    async def malformed_user_fees(request):
        if request["type"] == "userFees":
            result = user_fees()
            del result["userSpotAddRate"]
            return result
        return await original(request)

    session.publicPostInfo = malformed_user_fees
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    with pytest.raises(ValueError, match="userSpotAddRate"):
        await client.fetch_account_snapshot((market,))


@pytest.mark.asyncio
async def test_expired_market_uses_account_settlement_fills_as_authoritative_resolution():
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    session.user_fills_payload = settlement_fills(market, yes_fraction=1.0)
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    snapshot = await client.fetch_account_snapshot((market,))
    evidence = snapshot.settlement(market.market_id)
    assert evidence is not None
    assert evidence.yes_fraction == 1.0
    assert evidence.observed_yes_qty == 12.0
    assert evidence.observed_no_qty == 3.0
    assert evidence.collateral_payout == 12.0
    assert evidence.fee == 0.0

    lifecycle = await client.fetch_market_lifecycle(
        market,
        account=snapshot,
        now_ms=market.lifecycle.scheduled_event_time_ms + 2_000,
    )
    assert lifecycle.state is HyperliquidOutcomeLifecycleState.SETTLED
    assert lifecycle.settlement == evidence


@pytest.mark.asyncio
async def test_expired_market_recovers_settlement_from_time_ranged_fill_history():
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    session.user_fills_payload = []
    session.user_fills_by_time_payload = settlement_fills(
        market,
        yes_fraction=0.0,
    )
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")
    now_ms = market.lifecycle.scheduled_event_time_ms + 2_000

    snapshot = await client.fetch_account_snapshot((market,))
    assert snapshot.settlement(market.market_id) is None
    lifecycle = await client.fetch_market_lifecycle(
        market,
        account=snapshot,
        now_ms=now_ms,
    )

    assert lifecycle.state is HyperliquidOutcomeLifecycleState.SETTLED
    assert lifecycle.settlement is not None
    assert lifecycle.settlement.yes_fraction == 0.0
    assert lifecycle.settlement.evidence_source == "hyperliquid_user_fills_by_time"
    assert {
        "type": "userFillsByTime",
        "user": "0xaccount",
        "startTime": market.lifecycle.scheduled_event_time_ms,
        "endTime": now_ms,
        "aggregateByTime": False,
    } in session.public_requests


@pytest.mark.asyncio
async def test_missing_market_before_expiry_fails_instead_of_inventing_lifecycle():
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    with pytest.raises(ValueError, match="disappeared before scheduled expiry"):
        await client.fetch_market_lifecycle(
            market,
            now_ms=market.lifecycle.scheduled_event_time_ms - 1,
        )


@pytest.mark.asyncio
async def test_expired_market_without_settlement_evidence_stays_explicitly_unresolved():
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    lifecycle = await client.fetch_market_lifecycle(
        market,
        now_ms=market.lifecycle.scheduled_event_time_ms + 1,
    )
    assert (
        lifecycle.state
        is HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT
    )
    assert lifecycle.settlement is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history_payload", "expected_error"),
    [
        (TimeoutError("history unavailable"), "TimeoutError: history unavailable"),
        ({"malformed": True}, "ValueError: Hyperliquid userFillsByTime response"),
    ],
)
async def test_expired_market_settlement_recovery_failure_remains_cancel_eligible(
    history_payload,
    expected_error,
):
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    session.user_fills_payload = []
    session.user_fills_by_time_payload = history_payload
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    snapshot = await client.fetch_account_snapshot((market,))
    lifecycle = await client.fetch_market_lifecycle(
        market,
        account=snapshot,
        now_ms=market.lifecycle.scheduled_event_time_ms + 1,
    )

    assert lifecycle.state is HyperliquidOutcomeLifecycleState.EXPIRED_AWAITING_SETTLEMENT
    assert lifecycle.settlement is None
    assert lifecycle.settlement_recovery_error.startswith(expected_error)


def test_account_fill_preserves_negative_fee_as_authoritative_rebate():
    _, market = market_fixture()
    fills = hyperliquid.normalize_account_fills(user_fill_with_rebate(), (market,))

    assert len(fills) == 1
    assert fills[0].fee == pytest.approx(-0.001)
    assert fills[0].is_maker is True


def test_account_fill_rejects_conflicting_duplicate_identity():
    _, market = market_fixture()
    first = user_fills()[0]
    conflicting = dict(first, px="0.41")

    with pytest.raises(ValueError, match="conflicting duplicate Hyperliquid account fill"):
        hyperliquid.normalize_account_fills((first, conflicting), (market,))

    assert len(hyperliquid.normalize_account_fills((first, dict(first)), (market,))) == 1


@pytest.mark.asyncio
async def test_mutations_are_disabled_by_default_before_any_private_request():
    payload, market = market_fixture()
    session = FakeSession(payload)
    client = HyperliquidOutcomeLiveClient(session, account_address="0xaccount")

    with pytest.raises(OutcomeMutationDisabled):
        await client.submit_limit_order(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="25",
        )
    assert session.private_requests == []


@pytest.mark.asyncio
async def test_enabled_post_only_order_preflights_state_book_and_current_market(monkeypatch):
    payload, market = market_fixture()
    session = FakeSession(payload)
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xaccount",
        allow_mutations=True,
    )
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: 1784900000.0)

    result = await client.submit_limit_order(
        market,
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price="0.4",
        qty="25",
    )

    assert result.kind == "resting"
    assert result.order_id == "777"
    request = session.private_requests[0]
    assert request["action"]["orders"][0]["a"] == 100009130
    assert request["action"]["orders"][0]["t"] == {"limit": {"tif": "Alo"}}
    assert request["nonce"] == 1784900000300


@pytest.mark.asyncio
async def test_close_all_sell_may_clear_current_residual_below_minimum_quantity(
    monkeypatch,
):
    payload, base_market = market_fixture()
    market = replace(
        base_market,
        min_order_qty=5.0,
        min_order_notional=0.5,
    )

    class ResidualSession(FakeSession):
        async def publicPostInfo(self, request):
            if request["type"] == "spotClearinghouseState":
                self.public_requests.append(request)
                return {
                    "balances": [
                        {
                            "coin": "USDC",
                            "entryNtl": "0",
                            "hold": "0",
                            "token": 0,
                            "total": "100",
                        },
                        {
                            "coin": "+9130",
                            "entryNtl": "4.8",
                            "hold": "0",
                            "total": "12",
                        },
                        {
                            "coin": "+9131",
                            "entryNtl": "4.4",
                            "hold": "0",
                            "total": "11",
                        },
                    ],
                    "tokenToAvailableAfterMaintenance": [[0, "100"]],
                }
            if request["type"] == "frontendOpenOrders":
                self.public_requests.append(request)
                return []
            return await super().publicPostInfo(request)

    session = ResidualSession(payload)
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xaccount",
        allow_mutations=True,
    )
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: 1784900000.0)

    result = await client.submit_limit_order(
        market,
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.SELL,
        native_price="0.6",
        qty="1",
        close_all=True,
    )

    assert result.kind == "resting"
    assert session.private_requests[0]["action"]["orders"][0]["s"] == "1"

    with pytest.raises(ValueError, match="entire current residual"):
        await client.submit_limit_order(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.SELL,
            native_price="0.6",
            qty="2",
            close_all=True,
        )

    below_notional_market = replace(
        market,
        min_order_qty=1.0,
        min_order_notional=10.0,
    )
    with pytest.raises(ValueError, match="notional is below the minimum"):
        await client.submit_limit_order(
            below_notional_market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.SELL,
            native_price="0.6",
            qty="1",
            close_all=True,
        )


@pytest.mark.asyncio
async def test_order_rechecks_create_deadline_after_public_preflight(monkeypatch):
    payload, market = market_fixture()
    session = FakeSession(payload)
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xaccount",
        allow_mutations=True,
    )
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: 1784900000.0)

    with pytest.raises(OutcomeCreateDeadlineExpired) as exc_info:
        await client.submit_limit_order(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="25",
            create_deadline_ms=8_000,
            wall_clock_ms=lambda: 8_001,
        )

    assert exc_info.value.observed_at_ms == 8_001
    assert exc_info.value.deadline_ms == 8_000
    assert session.public_requests
    assert session.private_requests == []


@pytest.mark.asyncio
async def test_vault_mutations_query_and_sign_for_the_vault_trading_account(monkeypatch):
    payload, market = market_fixture()
    session = FakeSession(payload)
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xsigner",
        vault_address="0xvault",
        allow_mutations=True,
    )
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: 1784900000.0)

    await client.submit_limit_order(
        market,
        outcome=OutcomeSide.YES,
        side=OutcomeOrderSide.BUY,
        native_price="0.4",
        qty="25",
    )
    await client.fetch_settlement_evidence(
        market,
        start_time_ms=market.lifecycle.scheduled_event_time_ms,
        end_time_ms=market.lifecycle.scheduled_event_time_ms + 1_000,
    )

    account_request_types = {
        "spotClearinghouseState",
        "frontendOpenOrders",
        "userFills",
        "userFillsByTime",
        "userFees",
    }
    account_requests = [
        request
        for request in session.public_requests
        if request["type"] in account_request_types
    ]
    assert {request["user"] for request in account_requests} == {"0xvault"}
    assert session.private_requests[0]["vaultAddress"] == "0xvault"
    assert session.signed[0][2] == "0xvault"


@pytest.mark.asyncio
async def test_post_only_buy_preflight_reserves_conservative_account_maker_fee(monkeypatch):
    payload, market = market_fixture()
    session = FakeSession(payload)
    original = session.publicPostInfo

    async def limited_collateral(request):
        if request["type"] == "spotClearinghouseState":
            return {
                "balances": [
                    {
                        "coin": "USDC",
                        "entryNtl": "0.0",
                        "hold": "0.0",
                        "token": 0,
                        "total": "10.0",
                    }
                ],
                "tokenToAvailableAfterMaintenance": [[0, "10.0"]],
            }
        return await original(request)

    session.publicPostInfo = limited_collateral
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xaccount",
        allow_mutations=True,
    )
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: 1784900000.0)

    with pytest.raises(ValueError, match="insufficient HIP-4 collateral"):
        await client.submit_limit_order(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="25",
        )
    assert session.private_requests == []


@pytest.mark.asyncio
async def test_expired_market_rejects_new_order_but_allows_authoritative_cancel(
    monkeypatch,
):
    payload, market = market_fixture()
    session = FakeSession(payload)
    session.meta_outcomes = []
    client = HyperliquidOutcomeLiveClient(
        session,
        account_address="0xaccount",
        allow_mutations=True,
    )
    observed_seconds = (market.lifecycle.scheduled_event_time_ms + 2_000) / 1_000
    monkeypatch.setattr("outcome.hyperliquid_live.time.time", lambda: observed_seconds)

    with pytest.raises(ValueError, match="not active"):
        await client.submit_limit_order(
            market,
            outcome=OutcomeSide.YES,
            side=OutcomeOrderSide.BUY,
            native_price="0.4",
            qty="25",
        )
    assert session.private_requests == []

    result = await client.cancel_order(
        market,
        outcome=OutcomeSide.NO,
        order_id=123,
        expected_client_order_id="0x00000000000000000000000000000001",
    )

    assert result.kind == "cancelled"
    assert session.private_requests[0]["action"] == {
        "type": "cancel",
        "cancels": [{"a": 100009131, "o": 123}],
    }
