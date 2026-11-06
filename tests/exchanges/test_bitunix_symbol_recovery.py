"""Offline coverage for connector-scoped product-admission recovery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import ccxt
import pytest

from ccxt_contracts import build_contract_bot
from exchanges.bitunix import BitunixClient
from exchanges.ccxt_bot import CCXTBot
from live import executor
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes
from passivbot import Passivbot


BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
NOW = 10_000_000
COOLDOWN_MS = 6 * 60 * 60 * 1000


def _api_error(code=710003):
    with pytest.raises(ccxt.ExchangeError) as raised:
        BitunixClient._raise_api_error(code, "synthetic venue response")
    return raised.value


def _bot(exchange="bitunix", *, cooldown_hours=6.0):
    bot = build_contract_bot(exchange)
    bot.config["live"]["recovery"] = {
        "exchange_symbol_unavailable_cooldown_hours": cooldown_hours
    }
    bot.config["live"]["execution"]["max_n_creations_per_batch"] = 10
    bot._exchange_config_retry_after_ms = {}
    bot._exchange_config_retry_attempts = {}
    bot.stop_signal_received = False
    bot.has_position = lambda *, symbol, pside=None: any(
        abs(float(side.get("size", 0))) > 0
        for side in bot.positions.get(symbol, {}).values()
    )
    return bot


def _order(symbol=BTC, *, close=False):
    return {
        "symbol": symbol,
        "position_side": "long",
        "side": "sell" if close else "buy",
        "qty": 0.01,
        "price": 100.0,
        "type": "limit",
        "reduce_only": close,
        "pb_order_type": "close_grid_long" if close else "entry_grid_normal_long",
        "custom_id": "synthetic-close" if close else "synthetic-entry",
    }


@pytest.mark.parametrize("code", [710003, "710003"])
def test_bitunix_classifies_exact_structured_api_error(code):
    error = _api_error(code)

    assert type(error) is ccxt.ExchangeError
    assert error.code == "710003"
    assert _bot()._classify_exchange_symbol_unavailable_error(error) == (
        "bitunix_product_permission_unavailable"
    )


@pytest.mark.parametrize(
    "error",
    [
        ccxt.ExchangeError("synthetic message mentions 710003"),
        ccxt.ExchangeError('{"code":710003,"msg":"synthetic"}'),
        _api_error(710004),
        _api_error("0710003"),
        _api_error("7100030"),
        SimpleNamespace(code="710003"),
    ],
)
def test_bitunix_does_not_infer_permission_from_text_or_other_codes(error):
    bot = _bot()

    assert bot._classify_exchange_symbol_unavailable_error(error) is None
    assert not bot._activate_exchange_symbol_unavailable_cooldown(BTC, error, now_ms=NOW)
    assert bot._active_exchange_symbol_unavailable_cooldowns(now_ms=NOW) == {}


@pytest.mark.parametrize("exchange", ["binance", "okx", "weex"])
def test_bitunix_code_has_no_cross_connector_meaning(exchange):
    bot = _bot(exchange)

    assert not bot._activate_exchange_symbol_unavailable_cooldown(
        BTC, _api_error(), now_ms=NOW
    )
    assert bot._active_exchange_symbol_unavailable_cooldowns(now_ms=NOW) == {}


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("strong_mode", ["manual", "panic"])
def test_cooldown_is_per_market_and_preserves_held_and_strong_modes(held, strong_mode):
    bot = _bot()
    if held:
        bot.positions[BTC] = {"long": {"size": 0.01}}
    assert bot._activate_exchange_symbol_unavailable_cooldown(
        BTC, _api_error(), now_ms=NOW
    )
    modes = {
        "long": {BTC: "normal", ETH: "normal"},
        "short": {BTC: strong_mode, ETH: strong_mode},
    }

    unavailable = bot._apply_exchange_symbol_unavailable_planning_policy(
        [BTC, ETH], modes, now_ms=NOW
    )

    assert unavailable == {BTC}
    assert modes["long"][BTC] == ("tp_only" if held else "graceful_stop")
    assert modes["short"][BTC] == strong_mode
    assert modes["long"][ETH] == "normal"
    assert modes["short"][ETH] == strong_mode
    assert bot._exchange_symbol_cooldown_blocks_tradability(BTC, unavailable) is not held
    assert not bot._exchange_symbol_cooldown_blocks_tradability(ETH, unavailable)


def test_held_graceful_stop_promotes_to_tp_only_without_weakening_manual():
    bot = _bot()
    bot.positions[BTC] = {"short": {"size": -0.01}}
    bot._activate_exchange_symbol_unavailable_cooldown(BTC, _api_error(), now_ms=NOW)
    modes = {"long": {BTC: "graceful_stop"}, "short": {BTC: "manual"}}

    bot._apply_exchange_symbol_unavailable_planning_policy([BTC], modes, now_ms=NOW)

    assert modes == {"long": {BTC: "tp_only"}, "short": {BTC: "manual"}}


def test_expiry_clears_retry_state_and_allows_retry_without_persistent_gate(monkeypatch):
    bot = _bot()
    bot._exchange_config_retry_attempts[BTC] = 3
    bot._activate_exchange_symbol_unavailable_cooldown(BTC, _api_error(), now_ms=NOW)
    assert bot._exchange_config_retry_after_ms[BTC] == NOW + COOLDOWN_MS
    assert _bot()._active_exchange_symbol_unavailable_cooldowns(now_ms=NOW) == {}
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW + COOLDOWN_MS - 1)
    assert not bot._order_requires_exchange_config_before_create(_order(close=True))

    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW + COOLDOWN_MS)

    assert bot._order_requires_exchange_config_before_create(_order(close=True))
    assert bot._active_exchange_symbol_unavailable_cooldowns() == {}
    assert BTC not in bot._exchange_symbol_unavailable_reason_by_symbol
    assert BTC not in bot._exchange_config_retry_after_ms
    assert BTC not in bot._exchange_config_retry_attempts


def test_zero_cooldown_disables_permission_recovery(monkeypatch):
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW)
    bot = _bot(cooldown_hours=0)

    assert not bot._activate_exchange_symbol_unavailable_cooldown(BTC, _api_error())
    assert bot._active_exchange_symbol_unavailable_cooldowns() == {}
    assert bot._order_requires_exchange_config_before_create(_order(close=True))


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"reduce_only": False},
        {"reduce_only": None},
        {"reduce_only": "true"},
        {"reduce_only": 1},
        {"reduceOnly": True},
        {"info": {"reduceOnly": True}},
        {"reduce_only": False, "info": {"reduceOnly": True}},
    ],
)
def test_config_bypass_requires_explicit_normalized_reduce_only(monkeypatch, fields):
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW)
    bot = _bot()
    bot._activate_exchange_symbol_unavailable_cooldown(BTC, _api_error())
    order = {"symbol": BTC, "side": "sell", "pb_order_type": "close_grid_long", **fields}

    assert bot._order_requires_exchange_config_before_create(order)
    assert not bot._order_requires_exchange_config_before_create(_order(close=True))
    assert bot._order_requires_exchange_config_before_create(_order(ETH, close=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [710003, "710003"])
@pytest.mark.parametrize("cooldown_hours", [0, 6.0])
async def test_actual_batch_failure_retains_exception_and_consumes_normal_budget(
    monkeypatch, code, cooldown_hours
):
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW)
    bot = _bot(cooldown_hours=cooldown_hours)
    errors = []

    async def reject_at_connector(_order):
        error = _api_error(code)
        errors.append(error)
        raise error

    bot.execute_order = AsyncMock(side_effect=reject_at_connector)
    bot.restart_bot = AsyncMock()
    entry = _order()

    results = await CCXTBot.execute_orders(bot, [entry])

    assert results == errors
    assert results[0] is errors[0]
    bot.execute_order.assert_awaited_once_with(entry)
    assert bot.error_counts == [NOW]
    bot.restart_bot.assert_not_awaited()
    expected = {BTC: NOW + COOLDOWN_MS} if cooldown_hours else {}
    assert bot._active_exchange_symbol_unavailable_cooldowns() == expected
    assert bot._exchange_config_retry_after_ms == expected


@pytest.mark.asyncio
async def test_permission_error_remains_ambiguous_in_actual_shared_batch(monkeypatch):
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW)
    bot = _bot()
    bot.debug_mode = False
    bot.recent_order_executions = []
    bot.get_exchange_time = lambda: NOW
    bot.log_order_action = Mock()
    bot._log_order_action_summary = Mock()
    bot._live_event_sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[bot._live_event_sink], monitor_sinks=[]
    )
    error = _api_error()
    bot.execute_order = AsyncMock(side_effect=error)
    entry = _order()
    try:
        result = await executor.execute_orders_parent(bot, [entry])

        assert result == []
        assert bot.error_counts == [NOW]
        assert bot.recent_order_executions[0]["symbol"] == BTC
        assert bot.orders_emitted_to_exchange[0]["status"] == "create_error_ambiguous"
        assert bot.orders_emitted_to_exchange[0]["custom_id"] == entry["custom_id"]
        assert bot._active_exchange_symbol_unavailable_cooldowns() == {BTC: NOW + COOLDOWN_MS}
        assert bot._live_event_pipeline.flush(timeout=2)
        ambiguous = [
            event for event in bot._live_event_sink.events
            if event.event_type == EventTypes.EXECUTION_AMBIGUOUS
        ]
        assert len(ambiguous) == 1
        assert ambiguous[0].reason_code == "result_exception"
        assert not [
            event for event in bot._live_event_sink.events
            if event.event_type == EventTypes.EXECUTION_CREATE_SUCCEEDED
        ]
    finally:
        assert bot._live_event_pipeline.close(timeout=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("cooldown_state", ["ordinary_backoff", "active", "expired"])
async def test_actual_executor_only_bypasses_pending_setup_for_cooled_held_close(
    monkeypatch, cooldown_state
):
    monkeypatch.setattr("passivbot.utc_ms", lambda: NOW)
    bot = _bot()
    bot.positions[BTC] = {"long": {"size": 0.01}}
    bot.debug_mode = False
    bot.balance_threshold = 0
    bot.state_change_detected_by_symbol = set()
    bot.get_raw_balance = lambda: 100.0
    bot.order_was_recently_updated = lambda _order: 0
    bot._exchange_config_success_pause_seconds = lambda: 0
    bot._current_planning_snapshot = SimpleNamespace(account_invalidation_generation=0)
    bot.execute_cancellations_parent = AsyncMock(return_value=[])
    bot.execute_orders_parent = AsyncMock(return_value=[])
    bot._schedule_forager_candidate_candle_refresh = Mock()
    bot._live_event_sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[bot._live_event_sink], monitor_sinks=[]
    )
    if cooldown_state == "active":
        bot._activate_exchange_symbol_unavailable_cooldown(BTC, _api_error())
    elif cooldown_state == "expired":
        bot._activate_exchange_symbol_unavailable_cooldown(
            BTC, _api_error(), now_ms=NOW - COOLDOWN_MS
        )
    else:
        # An unrelated ordinary setup backoff must retain the inherited close policy.
        bot._exchange_config_retry_after_ms[BTC] = NOW + 60_000
    async def configure(symbols):
        if cooldown_state == "expired" and symbols == [BTC]:
            raise ccxt.NetworkError("synthetic setup failure")

    bot.update_exchange_config_by_symbols = AsyncMock(side_effect=configure)

    async def fresh_market(_bot, orders, **_kwargs):
        return list(orders)

    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", fresh_market)
    monkeypatch.setattr(Passivbot, "_track_order_wave_confirmation", lambda *_args: None)
    entry, close, other_entry = _order(), _order(close=True), _order(ETH)
    try:
        await executor.execute_order_plan(bot, [], [entry, close, other_entry])

        expected_config_calls = (
            [call([BTC]), call([ETH])] if cooldown_state == "expired" else [call([ETH])]
        )
        assert bot.update_exchange_config_by_symbols.await_args_list == expected_config_calls
        expected = [close, other_entry] if cooldown_state == "active" else [other_entry]
        bot.execute_orders_parent.assert_awaited_once_with(expected)
        assert not getattr(bot, "error_counts", [])
        assert bot._live_event_pipeline.flush(timeout=2)
        skipped = [
            event for event in bot._live_event_sink.events
            if event.event_type == EventTypes.EXECUTION_CREATE_SKIPPED
            and event.reason_code == ReasonCodes.PENDING_EXCHANGE_CONFIG
        ]
        assert len(skipped) == 1
        assert skipped[0].data["order_count"] == (1 if cooldown_state == "active" else 2)
        assert skipped[0].data["symbols"] == [BTC]
        assert skipped[0].data["protective_allowed_count"] == int(cooldown_state == "active")
    finally:
        assert bot._live_event_pipeline.close(timeout=2)
