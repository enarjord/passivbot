from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from exchanges.hyperliquid import HyperliquidBot
from live.order_churn_gate import OrderChurnGateState


def _bot(payload_or_exception):
    bot = HyperliquidBot.__new__(HyperliquidBot)
    bot.user_info = {"wallet_address": "0x" + "1" * 40}
    bot._hl_order_churn_rate_snapshot = None
    bot._hl_order_churn_local_actions = {}
    bot._hl_order_churn_local_action_sequence = 0
    bot._hl_order_churn_rate_next_refresh_monotonic = 0.0
    bot._hl_order_churn_rate_request_started_monotonic = 0.0

    async def public_post_info(request):
        assert request == {
            "type": "userRateLimit",
            "user": bot.user_info["wallet_address"],
        }
        if isinstance(payload_or_exception, BaseException):
            raise payload_or_exception
        return dict(payload_or_exception)

    bot.cca = SimpleNamespace(publicPostInfo=public_post_info)
    return bot


@pytest.mark.asyncio
async def test_headroom_uses_cap_minus_used_plus_unconsumed_reserved_surplus():
    bot = _bot(
        {"nRequestsUsed": 7, "nRequestsCap": 10, "nRequestsSurplus": 0}
    )
    assert await bot._order_churn_far_create_headroom() == 3
    bot._record_order_churn_signed_action_attempts(2)
    assert await bot._order_churn_far_create_headroom() == 1

    surplus_bot = _bot(
        {"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 25}
    )
    assert await surplus_bot._order_churn_far_create_headroom() == 35
    surplus_bot._record_order_churn_signed_action_attempts(2)
    assert await surplus_bot._order_churn_far_create_headroom() == 33


@pytest.mark.asyncio
async def test_usage_beyond_normal_cap_is_valid_exhausted_state():
    bot = _bot(
        {"nRequestsUsed": 11, "nRequestsCap": 10, "nRequestsSurplus": 0}
    )

    assert await bot._order_churn_far_create_headroom() == 0
    assert bot._hl_order_churn_rate_snapshot["headroom"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"nRequestsUsed": 1, "nRequestsCap": 10, "nRequestsSurplus": 1},
        {"nRequestsUsed": -1, "nRequestsCap": 10, "nRequestsSurplus": 0},
        {"nRequestsUsed": 1.5, "nRequestsCap": 10, "nRequestsSurplus": 0},
    ],
)
async def test_invalid_or_contradictory_headroom_fails_closed_for_far_class(payload):
    bot = _bot(payload)
    assert await bot._order_churn_far_create_headroom() is None
    assert bot._hl_order_churn_rate_snapshot is None
    assert bot._hl_order_churn_rate_next_refresh_monotonic > 0.0


@pytest.mark.asyncio
async def test_headroom_endpoint_failure_is_backed_off_and_does_not_invent_capacity():
    bot = _bot(RuntimeError("offline"))
    assert await bot._order_churn_far_create_headroom() is None
    assert await bot._order_churn_far_create_headroom() is None


@pytest.mark.asyncio
async def test_action_attempted_during_headroom_fetch_is_debited_from_response():
    bot = _bot({"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0})

    async def public_post_info(_request):
        bot._record_order_churn_signed_action_attempts(1)
        return {"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0}

    bot.cca.publicPostInfo = public_post_info

    assert await bot._order_churn_far_create_headroom() == 9
    assert len(bot._hl_order_churn_local_actions) == 1


def test_completed_actions_before_first_snapshot_need_no_local_debit():
    bot = _bot({"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0})
    tokens = bot._record_order_churn_signed_action_attempts(5)
    assert len(bot._hl_order_churn_local_actions) == 5
    bot._complete_order_churn_signed_action_attempts(tokens)
    assert bot._hl_order_churn_local_actions == {}


def test_expired_snapshot_preserves_unresolved_local_debits(monkeypatch):
    bot = _bot({"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0})
    bot._hl_order_churn_rate_snapshot = {
        "expires_monotonic": 5.0,
        "request_started_monotonic": 1.0,
        "headroom": 10,
    }
    monkeypatch.setattr("exchanges.hyperliquid.time.monotonic", lambda: 10.0)

    tokens = bot._record_order_churn_signed_action_attempts(1)

    assert bot._hl_order_churn_rate_snapshot is not None
    assert set(bot._hl_order_churn_local_actions) == set(tokens)


@pytest.mark.asyncio
async def test_unresolved_action_started_before_refresh_is_carried(monkeypatch):
    bot = _bot({"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0})
    clock = [1.0]
    monkeypatch.setattr("exchanges.hyperliquid.time.monotonic", lambda: clock[0])
    tokens = bot._record_order_churn_signed_action_attempts(1)

    async def public_post_info(_request):
        clock[0] = 3.0
        bot._complete_order_churn_signed_action_attempts(tokens)
        return {"nRequestsUsed": 0, "nRequestsCap": 10, "nRequestsSurplus": 0}

    bot.cca.publicPostInfo = public_post_info
    clock[0] = 2.0
    assert await bot._order_churn_far_create_headroom() == 9
    assert set(bot._hl_order_churn_local_actions) == set(tokens)

    bot._hl_order_churn_rate_snapshot["expires_monotonic"] = 0.0
    clock[0] = 40.0
    assert await bot._order_churn_far_create_headroom() == 10
    assert bot._hl_order_churn_local_actions == {}


def test_precreate_cost_counts_only_unconfigured_symbols():
    bot = object.__new__(HyperliquidBot)
    bot.already_updated_exchange_config_symbols = {"BTC/USDC:USDC"}

    assert bot._order_churn_precreate_signed_action_costs(
        {"BTC/USDC:USDC", "ETH/USDC:USDC"}
    ) == {"ETH/USDC:USDC": 1}


@pytest.mark.asyncio
async def test_failed_config_connector_attempt_is_debited(monkeypatch):
    bot = object.__new__(HyperliquidBot)
    bot.user_info = {"is_vault": False}
    bot.exchange = "hyperliquid"
    bot._order_churn_gate_enabled_for_connector = True
    bot._order_churn_gate_state = OrderChurnGateState()
    bot.live_value = lambda key: {
        "order_replacement_churn_gate_activation_count": 10,
        "order_replacement_churn_gate_window_minutes": 10.0,
    }[key]
    bot._emit_order_churn_actions_accounted_event = MagicMock()
    bot._calc_leverage_for_symbol = lambda _symbol: 5
    bot._get_margin_mode_for_symbol = lambda _symbol: "cross"
    bot._record_order_churn_signed_action_attempts = MagicMock()
    bot._complete_order_churn_signed_action_attempts = MagicMock()
    bot.cca = SimpleNamespace(
        set_margin_mode=AsyncMock(side_effect=RuntimeError("rejected"))
    )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("exchanges.hyperliquid.asyncio.sleep", no_sleep)

    await bot.update_exchange_config_by_symbols(["BTC/USDC:USDC"])

    assert len(bot._order_churn_gate_state.action_attempt_timestamps) == 1
    bot._emit_order_churn_actions_accounted_event.assert_called_once()
    assert bot._emit_order_churn_actions_accounted_event.call_args.kwargs[
        "action_kind"
    ] == "config"
    bot._record_order_churn_signed_action_attempts.assert_called_once_with(1)
    bot._complete_order_churn_signed_action_attempts.assert_called_once()
    bot.cca.set_margin_mode.assert_awaited_once()
