import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import DDoSProtection, NetworkError, OnMaintenance, RateLimitExceeded

from live import hsl_live


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [NetworkError, DDoSProtection, OnMaintenance, RateLimitExceeded])
async def test_cycle_preserves_transport_category_and_retry_delay(error_type):
    failure = error_type("untrusted response")
    failure.response_headers = {"Retry-After": "45"}
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
