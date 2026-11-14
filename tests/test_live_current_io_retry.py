import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ccxt.base.errors import (
    DDoSProtection,
    InvalidNonce,
    NetworkError,
    OnMaintenance,
    RateLimitExceeded,
)

from live import hsl_live
from live.market_snapshot import MarketSnapshotUnavailable


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [NetworkError, DDoSProtection, OnMaintenance, RateLimitExceeded])
async def test_cycle_preserves_transport_category_and_retry_delay(error_type):
    failure = error_type("untrusted response")
    failure.retry_after_seconds = 45.0
    bot = SimpleNamespace(
        config={"live": {}},
        refresh_protective_authoritative_state=AsyncMock(side_effect=failure),
        _staged_planner_required_surfaces=lambda **kwargs: set(),
        _begin_live_event_cycle=lambda **kwargs: None,
    )

    result = await hsl_live.owner(bot).cycle()

    category = "network" if error_type is NetworkError else "exchange_backpressure"
    assert result == {
        "updated": False,
        "current_io_unavailable": True,
        "transport_failure": (category, error_type.__name__, 45.0),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "reason", "missing"),
    [
        (NetworkError, "ordinary_transport_unavailable", ()),
        (
            MarketSnapshotUnavailable,
            "prepared_market_inputs_unavailable",
            ("market_snapshot",),
        ),
    ],
)
async def test_completed_ordinary_input_failure_still_services_protection(
    error_type, reason, missing,
):
    bot = SimpleNamespace(
        config={"live": {}},
        refresh_protective_authoritative_state=AsyncMock(return_value=True),
        execute_order_plan_to_exchange=AsyncMock(),
        _begin_live_event_cycle=lambda **kwargs: None,
        _handle_staged_execution_precondition_error=lambda exc: (False, {}),
    )
    owner = hsl_live.owner(bot)
    owner.remember_position = Mock()
    owner.schedule_history = owner.schedule_sources = Mock()
    owner.protect = AsyncMock(return_value=True)
    owner._defer_ordinary = Mock()
    owner._ordinary_plan = AsyncMock(return_value=None)
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(error_type("ordinary input unavailable"))
    owner._ordinary = failed

    try:
        result = await owner.cycle()

        assert result == {
            "updated": True,
            "ordinary_completed": True,
            "ordinary_executed": False,
            "protective_work": True,
        }
        owner._defer_ordinary.assert_called_once_with(
            reason, phase="preparation", **({"missing": missing} if missing else {})
        )
        bot.refresh_protective_authoritative_state.assert_awaited_once_with(
            require_balance=True
        )
        owner.protect.assert_awaited_once()
        bot.execute_order_plan_to_exchange.assert_not_awaited()
        assert owner._ordinary is not failed and owner._ordinary is not None
        await owner._ordinary
        owner._ordinary_plan.assert_awaited_once()
    finally:
        owner.cancel_inputs()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_type", "retry_after", "category"),
    [(NetworkError, 45.0, "network"), (RateLimitExceeded, 0.0, "exchange_backpressure")],
)
async def test_completed_ordinary_input_failure_preserves_backpressure(
    error_type, retry_after, category,
):
    failure = error_type("ordinary request rejected")
    failure.retry_after_seconds = retry_after
    bot = SimpleNamespace(
        refresh_protective_authoritative_state=AsyncMock(),
        _begin_live_event_cycle=lambda **kwargs: None,
    )
    owner = hsl_live.owner(bot)
    owner.protect = AsyncMock()
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(failure)
    owner._ordinary = failed

    assert await owner.cycle() == {
        "updated": False,
        "current_io_unavailable": True,
        "transport_failure": (category, error_type.__name__, retry_after),
    }
    bot.refresh_protective_authoritative_state.assert_not_awaited()
    owner.protect.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_ordinary_input_nonce_failure_reaches_owner_recovery():
    bot = SimpleNamespace(_begin_live_event_cycle=lambda **kwargs: None)
    owner = hsl_live.owner(bot)
    failed = asyncio.get_running_loop().create_future()
    failure = InvalidNonce("invalid nonce")
    failed.set_exception(failure)
    owner._ordinary = failed

    with pytest.raises(InvalidNonce) as raised:
        await owner.cycle()
    assert raised.value is failure


@pytest.mark.asyncio
async def test_owner_backs_off_transport_and_recovers_after_success(monkeypatch, caplog):
    monkeypatch.setattr(hsl_live, "uniform", lambda start, end: 1.0)
    now = 0.0
    waits = []
    bot = SimpleNamespace(
        _process_failure_state={},
        _current_io_retry_state=hsl_live.CurrentIoRetry(),
        stop_signal_received=False,
        live_value=lambda key: 2.0,
        _maybe_log_health_summary=lambda: None,
        _maybe_log_trailing_status=lambda: None,
        _maybe_log_unstuck_status=lambda: None,
    )

    async def sleep(seconds, *, stage):
        nonlocal now
        waits.append((seconds, stage))
        now += seconds
        if stage == "hsl_execution_delay":
            bot.stop_signal_received = True

    bot._sleep_unless_shutdown = sleep
    transport = {
        "updated": False,
        "current_io_unavailable": True,
        "transport_failure": ("exchange_backpressure", "RateLimitExceeded", 0.0),
    }
    owner = hsl_live.Owner(bot)
    owner._schedule_clock = lambda: now
    owner.cycle = AsyncMock(side_effect=[transport] * 11 + [{"updated": True}])

    with caplog.at_level(logging.INFO):
        await owner.run()

    assert waits == [(delay, "current_io_retry") for delay in [10, 20, 40] + [60] * 8] + [
        (2, "hsl_execution_delay")
    ]
    assert caplog.text.count("current I/O retry") == 1
    assert caplog.text.count("outage persists") == 1
    assert "current I/O recovered" in caplog.text
    assert owner._current_io_retry.failures == 0


@pytest.mark.asyncio
async def test_owner_waits_through_server_delay_even_if_sleep_returns_early(monkeypatch):
    monkeypatch.setattr(hsl_live, "uniform", lambda start, end: 1.0)
    now = 0.0
    waits = []
    bot = SimpleNamespace(
        stop_signal_received=False,
        live_value=lambda key: 2.0,
        _maybe_log_health_summary=lambda: None,
        _maybe_log_trailing_status=lambda: None,
        _maybe_log_unstuck_status=lambda: None,
    )

    async def sleep(seconds, *, stage):
        nonlocal now
        waits.append((seconds, stage))
        if stage == "hsl_execution_delay":
            bot.stop_signal_received = True
        else:
            now += min(seconds, 300.0)

    cycle_times = []

    async def cycle():
        cycle_times.append(now)
        if len(cycle_times) == 1:
            return {
                "updated": False,
                "current_io_unavailable": True,
                "transport_failure": ("exchange_backpressure", "RateLimitExceeded", 600.0),
            }
        return {"updated": True}

    bot._sleep_unless_shutdown = sleep
    owner = hsl_live.Owner(bot)
    owner._schedule_clock = lambda: now
    owner.cycle = cycle
    await owner.run()

    assert cycle_times == [0.0, 600.0]
    assert waits == [(600.0, "current_io_retry"), (300.0, "current_io_retry"), (2.0, "hsl_execution_delay")]


def test_owner_reports_a_persistent_outage_after_a_long_server_wait(monkeypatch, caplog):
    monkeypatch.setattr(hsl_live, "uniform", lambda start, end: 1.0)
    retry = hsl_live.CurrentIoRetry()
    options = dict(category="exchange_backpressure", error_type="RateLimitExceeded",
                   retry_after=3600.0, execution_delay=2.0)
    retry.record_failure(now=0.0, **options)
    retry.record_failure(now=3600.0, **options)

    assert retry.retry_at == 7200.0
    assert caplog.text.count("outage persists") == 1


@pytest.mark.asyncio
async def test_startup_preparation_finishes_during_current_account_retry_after(monkeypatch):
    failure = RateLimitExceeded("untrusted response")
    failure.retry_after_seconds = 600.0
    refresh_attempted = asyncio.Event()
    preparation_done = asyncio.Event()

    async def refresh(**kwargs):
        refresh_attempted.set()
        raise failure

    async def prepare():
        await refresh_attempted.wait()
        await preparation_done.wait()
        return "prepared"

    bot = SimpleNamespace(
        stop_signal_received=False,
        live_value=lambda key: 2.0,
        refresh_protective_authoritative_state=AsyncMock(side_effect=refresh),
    )
    owner = hsl_live.Owner(bot)
    owner.poll_inputs = lambda: None
    owner.protect = AsyncMock()
    owner._schedule_clock = lambda: 1.0
    original_wait = asyncio.wait
    waits = []

    async def wait(tasks, *, timeout):
        waits.append(timeout)
        if len(waits) == 2:
            preparation_done.set()
            return await original_wait(tasks, timeout=timeout)
        return set(), set(tasks)

    monkeypatch.setattr(asyncio, "wait", wait)
    assert await owner.during_preparation(prepare()) == "prepared"
    assert owner._current_io_retry.retry_at == 601.0
    bot.refresh_protective_authoritative_state.assert_awaited_once()
    owner.protect.assert_not_awaited()


def test_ordinary_failure_diagnostic_recovery_preserves_replacement_retry_state():
    from live.diagnostic_safety import log_process_recovery

    process_diagnostics = {"repeat_signature": "test"}
    retry = hsl_live.CurrentIoRetry(failures=3, retry_at=120.0)
    first = SimpleNamespace(
        _process_failure_state=process_diagnostics, _current_io_retry_state=retry
    )
    owner = hsl_live.Owner(first)
    log_process_recovery(first)

    replacement = hsl_live.Owner(
        SimpleNamespace(
            _process_failure_state=process_diagnostics, _current_io_retry_state=retry
        )
    )
    assert process_diagnostics == {}
    assert replacement._current_io_retry is owner._current_io_retry
    assert replacement._current_io_retry.retry_at == 120.0
