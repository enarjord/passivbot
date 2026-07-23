import logging
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest


sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

from exchanges.okx import OKXBot
from live.event_bus import EventTypes
from live.event_emitters import emit_exchange_config_refresh_event


SYMBOL = "BTC/USDT:USDT"


def _make_bot(cca, emitter=None):
    bot = OKXBot.__new__(OKXBot)
    bot.cca = cca
    bot._get_margin_mode_for_symbol = lambda symbol: "cross"
    bot._calc_leverage_for_symbol = lambda symbol: 5
    bot._emit_exchange_config_refresh_event = emitter or Mock()
    return bot


def _outcome_event_kwargs(bot):
    bot._emit_exchange_config_refresh_event.assert_called_once()
    return bot._emit_exchange_config_refresh_event.call_args.kwargs


@pytest.mark.asyncio
async def test_okx_margin_response_emits_confirmed_outcome_and_keeps_info_log(caplog):
    cca = types.SimpleNamespace(
        set_margin_mode=AsyncMock(return_value={"code": "0", "apiKey": "SECRET"})
    )
    bot = _make_bot(cca)

    with caplog.at_level(logging.INFO):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    cca.set_margin_mode.assert_awaited_once_with("cross", symbol=SYMBOL, params={"lever": 5})
    assert _outcome_event_kwargs(bot) == {
        "context": "update_exchange_config_by_symbols",
        "operation": "set_margin_mode",
        "status": "succeeded",
        "symbol": SYMBOL,
        "outcome": "confirmed",
        "response_code": None,
        "error": None,
        "level": "info",
    }
    assert "margin=ok" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_okx_59107_emits_unchanged_debug_outcome_and_demotes_legacy_log(caplog):
    cca = types.SimpleNamespace(
        set_margin_mode=AsyncMock(side_effect=RuntimeError('{"code":"59107","apiKey":"SECRET"}'))
    )
    bot = _make_bot(cca)

    with caplog.at_level(logging.DEBUG):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    assert _outcome_event_kwargs(bot) == {
        "context": "update_exchange_config_by_symbols",
        "operation": "set_margin_mode",
        "status": "succeeded",
        "symbol": SYMBOL,
        "outcome": "unchanged",
        "response_code": "59107",
        "error": None,
        "level": "debug",
    }
    record = next(record for record in caplog.records if "margin=ok (unchanged)" in record.message)
    assert record.levelno == logging.DEBUG
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_okx_51039_emits_warning_failure_outcome_and_continues(caplog):
    exc = RuntimeError('{"code":"51039","apiKey":"SECRET"}')
    cca = types.SimpleNamespace(
        set_margin_mode=AsyncMock(side_effect=exc)
    )
    bot = _make_bot(cca)

    with caplog.at_level(logging.WARNING):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    assert _outcome_event_kwargs(bot) == {
        "context": "update_exchange_config_by_symbols",
        "operation": "set_margin_mode",
        "status": "failed",
        "symbol": SYMBOL,
        "outcome": "failed",
        "response_code": "51039",
        "error": exc,
        "level": "warning",
    }
    assert "unable to adjust margin mode/leverage" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_okx_generic_failure_emits_bounded_error_outcome(caplog):
    exc = RuntimeError("https://example.invalid/api?apiKey=SECRET")
    cca = types.SimpleNamespace(
        set_margin_mode=AsyncMock(side_effect=exc)
    )
    bot = _make_bot(cca)

    with caplog.at_level(logging.ERROR):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    assert _outcome_event_kwargs(bot) == {
        "context": "update_exchange_config_by_symbols",
        "operation": "set_margin_mode",
        "status": "failed",
        "symbol": SYMBOL,
        "outcome": "failed",
        "response_code": None,
        "error": exc,
        "level": "error",
    }
    assert "cross-margin update failed" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert "SECRET" not in caplog.text


@pytest.mark.asyncio
async def test_okx_hostile_failure_event_uses_bounded_exception_projection(caplog):
    hostile_error_type = type(
        "HostileTokenClass",
        (RuntimeError,),
        {"__module__": "hostile.example.invalid"},
    )
    exc = hostile_error_type(
        "credential=supersecret token=event-token "
        "https://example.invalid/config Traceback: hostile-message"
    )
    cca = types.SimpleNamespace(set_margin_mode=AsyncMock(side_effect=exc))
    bot = _make_bot(cca)
    bot._emit_live_event = Mock()
    bot._emit_exchange_config_refresh_event = lambda **kwargs: (
        emit_exchange_config_refresh_event(bot, **kwargs)
    )

    with caplog.at_level(logging.ERROR):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    bot._emit_live_event.assert_called_once()
    event_type, = bot._emit_live_event.call_args.args
    event_kwargs = bot._emit_live_event.call_args.kwargs
    assert event_type == EventTypes.EXCHANGE_CONFIG_REFRESH
    assert event_kwargs["data"]["error_type"] == "RuntimeError"
    event_repr = repr(event_kwargs)
    for unsafe_value in (
        "HostileTokenClass",
        "hostile-message",
        "example.invalid",
        "credential",
        "supersecret",
        "event-token",
        "Traceback",
    ):
        assert unsafe_value not in event_repr
    assert "cross-margin update failed" in caplog.text


@pytest.mark.asyncio
async def test_okx_hostile_event_failure_logs_bounded_type_and_preserves_success(caplog):
    hostile_error_type = type(
        "HostileTokenClass",
        (RuntimeError,),
        {"__module__": "hostile.example.invalid"},
    )
    emitter_error = hostile_error_type(
        "credential=supersecret token=event-token "
        "https://example.invalid/config Traceback: hostile-message"
    )
    cca = types.SimpleNamespace(set_margin_mode=AsyncMock(return_value={"code": "0"}))
    bot = _make_bot(cca, emitter=Mock(side_effect=emitter_error))

    with caplog.at_level(logging.DEBUG):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    assert any(
        record.getMessage()
        == "[event] failed to emit OKX exchange config-refresh outcome | "
        "error_type=RuntimeError"
        for record in caplog.records
    )
    assert "margin=ok" in caplog.text
    for unsafe_value in (
        "HostileTokenClass",
        "hostile-message",
        "example.invalid",
        "credential",
        "supersecret",
        "event-token",
        "Traceback",
    ):
        assert unsafe_value not in caplog.text


@pytest.mark.asyncio
async def test_okx_exchange_config_event_failure_does_not_change_success_control_flow(caplog):
    cca = types.SimpleNamespace(set_margin_mode=AsyncMock(return_value={"code": "0"}))
    emitter = Mock(side_effect=RuntimeError("event sink unavailable"))
    bot = _make_bot(cca, emitter=emitter)

    with caplog.at_level(logging.INFO):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    emitter.assert_called_once()
    cca.set_margin_mode.assert_awaited_once_with("cross", symbol=SYMBOL, params={"lever": 5})
    assert "margin=ok" in caplog.text


@pytest.mark.asyncio
async def test_okx_task_creation_failure_does_not_emit_false_outcome(caplog):
    cca = types.SimpleNamespace(set_margin_mode=AsyncMock())
    bot = _make_bot(cca)
    bot._calc_leverage_for_symbol = Mock(
        side_effect=RuntimeError("apiKey=SECRET task creation failed")
    )

    with caplog.at_level(logging.ERROR):
        await bot.update_exchange_config_by_symbols([SYMBOL])

    cca.set_margin_mode.assert_not_awaited()
    bot._emit_exchange_config_refresh_event.assert_not_called()
    assert "task creation failed" in caplog.text
    assert "cross-margin update failed" in caplog.text
    assert "SECRET" not in caplog.text
