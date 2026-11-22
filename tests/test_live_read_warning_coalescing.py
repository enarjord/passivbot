import logging
from types import SimpleNamespace

import pytest

from live import rms_directionality, state_refresh


def test_rms_read_failures_warn_first_and_report_accumulated_repeats(monkeypatch, caplog):
    clock = [0]
    monkeypatch.setattr(rms_directionality.time, "monotonic", lambda: clock[0])
    bot = SimpleNamespace()
    with caplog.at_level(logging.DEBUG):
        rms_directionality._log_read_failure(bot, "AAA/USDT:USDT", OSError("private"))
        clock[0] = 1
        rms_directionality._log_read_failure(bot, "BBB/USDT:USDT", OSError("private"))
        rms_directionality._log_read_failure(bot, "BBB/USDT:USDT", TimeoutError("private"))
        clock[0] = 300
        rms_directionality._log_read_failure(bot, "BBB/USDT:USDT", OSError("private"))

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 3
    assert "error_type=OSError failures=1" in messages[0]
    assert "error_type=TimeoutError failures=1" in messages[1]
    assert "error_type=OSError failures=2" in messages[2]
    assert sum(r.levelno == logging.DEBUG for r in caplog.records) == 4
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_separate_slow_refresh_tasks_keep_structured_events_but_coalesce_info(
    monkeypatch, caplog
):
    clock = [10_000]
    events = []
    monkeypatch.setattr(state_refresh, "_utc_ms", lambda: clock[0])
    monkeypatch.setattr(
        state_refresh.event_emitters,
        "emit_state_refresh_progress_event",
        lambda bot, **kwargs: events.append(kwargs),
    )
    done = False

    async def wait(*args, **kwargs):
        nonlocal done
        done = not done

    bot = SimpleNamespace(
        config={},
        _sleep_unless_shutdown=wait,
        _shutdown_requested=lambda: done,
    )
    task = SimpleNamespace(done=lambda: False)

    async def refresh(surface):
        nonlocal done
        done = True
        await state_refresh.log_staged_refresh_progress_until(
            bot, {surface}, {}, {surface: task}, clock[0] - 10_000,
        )

    with caplog.at_level(logging.DEBUG):
        await refresh("fills")
        clock[0] = 25_000
        await refresh("fills")
        await refresh("open_orders")
        clock[0] = 310_000
        await refresh("fills")

    assert [event["level"] for event in events] == ["info"] * 4
    assert [record.levelno for record in caplog.records] == [
        logging.INFO, logging.DEBUG, logging.INFO, logging.INFO,
    ]
    assert "suppressed_refreshes=1" in caplog.records[-1].getMessage()
