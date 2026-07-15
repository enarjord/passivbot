import sys
import types
from unittest.mock import AsyncMock

import pytest

from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes

sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

from passivbot import Passivbot
from live.event_emitters import emit_exchange_config_refresh_event


def _make_bot_with_event_sink():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.bot_id = "bot_1"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    return bot, sink


def _raise_emit_failure(**_kwargs):
    raise RuntimeError("emit failed")


@pytest.mark.asyncio
async def test_maintenance_exchange_config_refresh_success_emits_event():
    bot, sink = _make_bot_with_event_sink()
    bot.init_markets = AsyncMock(return_value="ok")

    result = await bot._refresh_markets_for_maintenance()

    assert result == "ok"
    bot.init_markets.assert_awaited_once_with(verbose=False)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.EXCHANGE_CONFIG_REFRESH
    assert event.component == "exchange.config_refresh"
    assert event.status == "succeeded"
    assert event.level == "debug"
    assert event.reason_code == ReasonCodes.EXCHANGE_CONFIG_REFRESH
    assert event.data["context"] == "maintain_hourly_cycle"
    assert event.data["operation"] == "init_markets"
    assert event.data["elapsed_ms"] >= 0
    assert "error" not in event.data
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_maintenance_exchange_config_refresh_failure_is_sanitized_and_reraised():
    bot, sink = _make_bot_with_event_sink()
    exc = RuntimeError("binanceusdm apiKey=supersecret code=-4084")
    bot.init_markets = AsyncMock(side_effect=exc)

    with pytest.raises(RuntimeError) as raised:
        await bot._refresh_markets_for_maintenance()

    assert raised.value is exc
    bot.init_markets.assert_awaited_once_with(verbose=False)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.EXCHANGE_CONFIG_REFRESH
    assert event.status == "failed"
    assert event.level == "warning"
    assert event.reason_code == ReasonCodes.EXCHANGE_CONFIG_REFRESH_FAILED
    assert event.data["error_type"] == "RuntimeError"
    assert "supersecret" not in event.data["error"]
    assert "[redacted]" in event.data["error"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_maintenance_exchange_config_emit_failure_preserves_success():
    bot, _sink = _make_bot_with_event_sink()
    bot.init_markets = AsyncMock(return_value="ok")
    bot._emit_exchange_config_refresh_event = _raise_emit_failure

    assert await bot._refresh_markets_for_maintenance() == "ok"
    bot.init_markets.assert_awaited_once_with(verbose=False)


@pytest.mark.asyncio
async def test_maintenance_exchange_config_emit_failure_preserves_original_error():
    bot, _sink = _make_bot_with_event_sink()
    exc = RuntimeError("exchange failed")
    bot.init_markets = AsyncMock(side_effect=exc)
    bot._emit_exchange_config_refresh_event = _raise_emit_failure

    with pytest.raises(RuntimeError) as raised:
        await bot._refresh_markets_for_maintenance()

    assert raised.value is exc
    bot.init_markets.assert_awaited_once_with(verbose=False)


def test_exchange_config_outcome_metadata_is_bounded_and_value_safe():
    bot, sink = _make_bot_with_event_sink()

    emit_exchange_config_refresh_event(
        bot,
        context="update_exchange_config_by_symbols",
        operation="set_margin_mode",
        status="failed",
        symbol="BTC/USDT:USDT",
        outcome="failed",
        response_code="51039",
        error_type="RuntimeError",
        level="error",
    )
    emit_exchange_config_refresh_event(
        bot,
        context="update_exchange_config_by_symbols",
        operation="set_margin_mode",
        status="failed",
        symbol="https://example.invalid/api?apiKey=supersecret",
        outcome="failed apiKey=supersecret",
        response_code='{"code":"51039","apiKey":"supersecret"}',
        error_type="RuntimeError apiKey=supersecret",
        level="error",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 2
    valid_event, unsafe_event = sink.events
    assert valid_event.level == "error"
    assert valid_event.data["symbol"] == "BTC/USDT:USDT"
    assert valid_event.data["outcome"] == "failed"
    assert valid_event.data["response_code"] == "51039"
    assert valid_event.data["error_type"] == "RuntimeError"
    assert "error" not in valid_event.data
    assert not {
        "symbol",
        "outcome",
        "response_code",
        "error_type",
    }.intersection(unsafe_event.data)
    assert "supersecret" not in repr(unsafe_event.data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True
