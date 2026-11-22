import logging
from types import SimpleNamespace

import pytest

from live import rms_directionality, state_refresh
from live.event_bus import (
    ConsoleSummarySink, ListEventSink, LiveEventPipeline,
    format_hsl_status_console, split_health_console,
)


def test_rms_read_failures_warn_first_and_report_accumulated_repeats(monkeypatch, caplog):
    clock = [0]
    monkeypatch.setattr(rms_directionality.time, "monotonic", lambda: clock[0])
    events = []
    bot = SimpleNamespace(
        _live_event_pipeline=SimpleNamespace(emit=lambda event, **kwargs: events.append(event))
    )
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
    assert sum(r.levelno == logging.DEBUG for r in caplog.records) == 1
    assert [event["level"] for event in events] == ["warning", "debug", "warning", "warning"]
    assert [event["data"]["failures"] for event in events] == [1, 1, 1, 2]
    assert all(event["event_type"] == "ema.rms_read_failed" for event in events)
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
    assert events[-1]["suppressed_refreshes"] == 1
    assert "suppressed_refreshes=1" in caplog.records[-1].getMessage()

def test_hsl_actual_expiry_stays_visible_in_a_single_record():
    row = {
        "observation_status": "stale",
        "signal_mode": "position_side",
        "stale_reasons": ["input_expired"],
        "account_unavailable": [],
        "counts": {"green": 1, "inactive": 0, "red": 0, "unavailable": 0, "estimated": 1},
        "scopes": [{"symbol": "AAA/USDT:USDT", "pside": "long", "estimates": ["basis_estimated"]}],
    }
    expiry = format_hsl_status_console(row)
    assert "observation=stale" in expiry
    assert "stale_reason=input_expired" in expiry
    assert "last_green=1" in expiry
    assert len(list(split_health_console(expiry, "[risk]"))) == 1

    row["counts"]["unavailable"] = 1
    row["scopes"][0]["unavailable_reason"] = "current_mark_unavailable"
    assert "unavailable_reason=current_mark_unavailable" in format_hsl_status_console(row)

def test_rms_failure_has_one_structured_console_projection(caplog):
    durable = ListEventSink()
    pipeline = LiveEventPipeline(
        console_sink=ConsoleSummarySink(), structured_sinks=[durable],
    )
    bot = SimpleNamespace(
        _live_event_pipeline=pipeline,
        _live_event_console_available=lambda: True,
    )
    try:
        rms_directionality._log_read_failure(bot, "AAA/USDT:USDT", OSError("private"))
        assert pipeline.flush()
    finally:
        pipeline.close()
    assert [(event.level, event.data["failures"]) for event in durable.events] == [
        ("warning", 1)
    ]
    records = [line for line in caplog.messages if "RMS directionality unavailable" in line]
    assert len(records) == 1
    assert "error_type=OSError" in records[0]
