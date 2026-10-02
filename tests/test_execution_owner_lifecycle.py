"""The single execution owner's lifecycle, independent of historical HSL replay."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import InvalidNonce, InvalidOrder
from live.hsl_live import Owner
from passivbot import Passivbot
from passivbot_exceptions import FatalBotException


def owner_bot():
    bot = SimpleNamespace(
        stop_signal_received=False,
        _shutdown_in_progress=False,
        _health_errors=0,
        live_value=lambda key: 0.1,
        _maybe_log_health_summary=lambda: None,
        _maybe_log_trailing_status=lambda: None,
    )
    instance = Owner(bot)
    return bot, instance


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ValueError("malformed producer"),
        FatalBotException("fatal"),
        asyncio.CancelledError(),
        InvalidOrder("bad request"),
    ],
)
async def test_execution_wrapper_propagates_failure_and_releases_ownership(failure):
    bot, instance = owner_bot()
    bot._hsl_live = instance
    instance.cycle = AsyncMock(side_effect=failure)
    bot._maybe_recover_exchange_time_sync = AsyncMock(return_value=False)
    with pytest.raises(type(failure)):
        await Passivbot.run_execution_loop(bot)
    assert bot._execution_loop_task is None
    assert bot._execution_loop_stopped.is_set()
    assert not instance._running
    instance.cycle.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True])
async def test_owner_retries_only_confirmed_clock_recovery(recovered):
    bot, instance = owner_bot()
    error = InvalidNonce("timestamp outside recvWindow")
    instance.cycle = AsyncMock(side_effect=[error, dict(updated=True)])
    bot._maybe_recover_exchange_time_sync = AsyncMock(return_value=recovered)
    delays = []

    async def delay(seconds, *, stage):
        delays.append(stage)
        if stage != "exchange_time_sync":
            bot.stop_signal_received = True

    bot._sleep_unless_shutdown = delay
    if recovered:
        await instance.run()
        assert instance.cycle.await_count == 2
        assert delays == ["exchange_time_sync", "hsl_execution_delay"]
        assert bot._health_errors == 1
    else:
        with pytest.raises(InvalidNonce):
            await instance.run()
        assert instance.cycle.await_count == 1
        assert not delays
    assert not instance._running


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["stop_signal_received", "_shutdown_in_progress"])
async def test_owner_shutdown_during_clock_failure_does_not_attempt_recovery(shutdown):
    bot, instance = owner_bot()

    async def cycle():
        setattr(bot, shutdown, True)
        raise InvalidNonce("timestamp outside recvWindow")

    instance.cycle = cycle
    bot._maybe_recover_exchange_time_sync = AsyncMock()
    await instance.run()
    bot._maybe_recover_exchange_time_sync.assert_not_awaited()
    assert not instance._running
