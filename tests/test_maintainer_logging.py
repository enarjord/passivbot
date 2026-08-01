import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from passivbot import Passivbot, shutdown_bot
from passivbot_exceptions import RestartBotException


class _Task:
    def __init__(self, result=True):
        self.result = result

    def cancel(self):
        return self.result


class _FailingTask:
    def __init__(self, message="cancel failed"):
        self.message = message

    def cancel(self):
        raise RuntimeError(self.message)


class _AsyncClient:
    def __init__(self, calls):
        self.calls = calls

    async def close(self):
        self.calls.append("cca")


def test_stop_data_maintainers_keeps_success_detail_at_debug(caplog):
    bot = SimpleNamespace(
        maintainers={"hourly": _Task()},
        WS_ohlcvs_1m_tasks={"BTC": _Task(False)},
    )

    with caplog.at_level(logging.DEBUG):
        result = Passivbot.stop_data_maintainers(bot)

    assert result == {"hourly": True}
    assert "stopped data maintainers: {'hourly': True}" in caplog.text
    assert "stopped ohlcvs watcher tasks {'BTC': False}" in caplog.text
    assert not [record for record in caplog.records if record.levelno == logging.INFO]


def test_stop_data_maintainers_keeps_cancellation_failures_at_error(caplog):
    bot = SimpleNamespace(
        maintainers={
            "hourly": _FailingTask(
                "token=maintainer-secret https://private.invalid/maintainer"
            )
        },
        WS_ohlcvs_1m_tasks={
            "BTC": _FailingTask(
                "authorization=websocket-secret https://private.invalid/websocket"
            )
        },
    )

    with caplog.at_level(logging.DEBUG):
        result = Passivbot.stop_data_maintainers(bot)

    assert result == {}
    assert "error stopping maintainer hourly (RuntimeError)" in caplog.text
    assert "error stopping WS_ohlcvs_1m_tasks BTC (RuntimeError)" in caplog.text
    assert "maintainer-secret" not in caplog.text
    assert "websocket-secret" not in caplog.text
    assert "private.invalid" not in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.asyncio
async def test_restart_request_defers_maintainer_cancellation_to_outer_cleanup():
    bot = Passivbot.__new__(Passivbot)
    stop_calls = []
    bot.stop_data_maintainers = lambda: stop_calls.append("stop")

    with pytest.raises(RestartBotException, match="Bot will restart"):
        await bot.restart_bot()

    assert stop_calls == []


@pytest.mark.asyncio
async def test_restart_cleanup_awaits_ws_owner_before_closing_resources():
    bot = Passivbot.__new__(Passivbot)
    calls = []
    child = None

    async def child_watcher():
        try:
            await asyncio.Event().wait()
        finally:
            calls.append("watcher_stopped")

    async def ws_owner():
        try:
            await asyncio.Event().wait()
        finally:
            child.cancel()
            await asyncio.gather(child, return_exceptions=True)
            calls.append("owner_stopped")

    class _Client:
        def __init__(self, label):
            self.label = label

        async def close(self):
            calls.append(self.label)

    class _Publisher:
        def close(self):
            calls.append("publisher")

    child = asyncio.create_task(child_watcher())
    owner = asyncio.create_task(ws_owner())
    await asyncio.sleep(0)
    bot.maintainers = {"maintain_forager_ws_candles": owner}
    bot.WS_ohlcvs_1m_tasks = {"BTC": child}
    bot.ccp = _Client("public")
    bot.cca = _Client("private")
    bot.monitor_publisher = _Publisher()
    bot._close_live_event_pipeline = lambda **_kwargs: calls.append("pipeline") or True

    await bot.cleanup_for_restart(timeout_seconds=1.0)

    assert calls == [
        "watcher_stopped",
        "owner_stopped",
        "public",
        "private",
        "pipeline",
        "publisher",
    ]
    assert bot.ccp is None
    assert bot.cca is None
    assert bot.monitor_publisher is None


@pytest.mark.asyncio
async def test_restart_cleanup_abandons_cancellation_resistant_maintainer(caplog):
    bot = Passivbot.__new__(Passivbot)
    release = asyncio.Event()
    cancellation_count = 0

    async def resistant_maintainer():
        nonlocal cancellation_count
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_count += 1

    class _Client:
        async def close(self):
            return None

    task = asyncio.create_task(resistant_maintainer())
    await asyncio.sleep(0)
    bot.maintainers = {"resistant": task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Client()
    bot.cca = _Client()
    bot.monitor_publisher = None
    bot._close_live_event_pipeline = lambda **_kwargs: True

    try:
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(
                bot.cleanup_for_restart(timeout_seconds=0.0), timeout=1.5
            )
        assert cancellation_count >= 1
        assert "maintainer cancellation grace expired" in caplog.text
        assert bot.ccp is None
        assert bot.cca is None
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_restart_cleanup_skips_publisher_close_after_pipeline_timeout(caplog):
    bot = Passivbot.__new__(Passivbot)
    closed_clients = []

    class _Client:
        def __init__(self, label):
            self.label = label

        async def close(self):
            closed_clients.append(self.label)

    class _Publisher:
        def close(self):
            raise AssertionError("publisher close must be skipped")

    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Client("public")
    bot.cca = _Client("private")
    bot.monitor_publisher = _Publisher()
    bot._close_live_event_pipeline = lambda **_kwargs: False

    with caplog.at_level(logging.WARNING):
        await bot.cleanup_for_restart()

    assert closed_clients == ["public", "private"]
    assert bot.monitor_publisher is None
    assert "monitor publisher close skipped" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_bot_redacts_close_failure_text(capsys):
    class _FailingBot:
        def stop_data_maintainers(self):
            return None

        async def close(self):
            raise RuntimeError(
                "signature=shutdown-secret https://private.invalid/shutdown"
            )

    await shutdown_bot(_FailingBot())

    output = capsys.readouterr().out
    assert "Error during shutdown (RuntimeError)." in output
    assert "shutdown-secret" not in output
    assert "private.invalid" not in output


@pytest.mark.asyncio
async def test_close_stops_maintainers_once_without_duplicate_info(caplog):
    calls = []

    def stop_data_maintainers():
        calls.append("stop")
        return {"hourly": True}

    bot = SimpleNamespace(
        stop_data_maintainers=stop_data_maintainers,
        cca=_AsyncClient(calls),
        ccp=None,
        _close_live_event_pipeline=lambda timeout: calls.append(("pipeline", timeout)),
    )

    with caplog.at_level(logging.DEBUG):
        await Passivbot.close(bot)

    assert calls == ["stop", "cca", ("pipeline", 2.0)]
    assert "stopped data maintainers" not in caplog.text
    assert not [record for record in caplog.records if record.levelno == logging.INFO]


@pytest.mark.asyncio
async def test_hourly_maintenance_jitter_is_debug_only(caplog, monkeypatch):
    bot = SimpleNamespace(stop_signal_received=True)
    monkeypatch.setattr("passivbot.random.uniform", lambda _start, _end: 42.0)

    with caplog.at_level(logging.DEBUG):
        await Passivbot.maintain_hourly_cycle(bot)

    assert "[hourly] starting maintenance cycle (jitter=42.0s)" in caplog.text
    assert not [record for record in caplog.records if record.levelno == logging.INFO]


@pytest.mark.asyncio
async def test_hourly_candle_audit_failure_is_redacted_and_continues(caplog, monkeypatch):
    now_ms = 4_000_000
    sleeps = []
    bot = SimpleNamespace(
        stop_signal_received=False,
        _mem_log_prev={"timestamp": now_ms},
        memory_snapshot_interval_ms=3_600_000,
        candle_disk_check_interval_ms=1,
        candle_disk_check_boot_delay_ms=0,
        _candle_disk_check_last_ms=0,
        start_time_ms=0,
        init_markets_last_update_ms=0,
        audit_required_candle_disk_coverage=AsyncMock(
            side_effect=RuntimeError("apiKey=audit-secret\ntraceback-value")
        ),
        _refresh_markets_for_maintenance=AsyncMock(),
        restart_bot_on_too_many_errors=AsyncMock(),
    )

    async def sleep(seconds):
        sleeps.append(seconds)
        bot.stop_signal_received = True

    monkeypatch.setattr("passivbot.utc_ms", lambda: now_ms)
    monkeypatch.setattr("passivbot.random.uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr("passivbot.asyncio.sleep", sleep)

    with caplog.at_level(logging.DEBUG):
        await Passivbot.maintain_hourly_cycle(bot)

    assert bot._candle_disk_check_last_ms == now_ms
    bot.audit_required_candle_disk_coverage.assert_awaited_once_with()
    bot._refresh_markets_for_maintenance.assert_awaited_once_with()
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    assert sleeps == [1]
    assert (
        "[candle] disk coverage audit failed | action=continue error_type=RuntimeError"
        in caplog.text
    )
    assert "audit-secret" not in caplog.text
    assert "traceback-value" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
async def test_hourly_cycle_failure_is_redacted_and_retries_after_five_seconds(caplog, monkeypatch):
    sleeps = []
    bot = SimpleNamespace(
        stop_signal_received=False,
        _mem_log_prev=None,
        _log_memory_snapshot=lambda now_ms: (_ for _ in ()).throw(
            RuntimeError("authorization=cycle-secret\ntraceback-value")
        ),
        restart_bot_on_too_many_errors=AsyncMock(),
    )

    async def sleep(seconds):
        sleeps.append(seconds)
        bot.stop_signal_received = True

    monkeypatch.setattr("passivbot.utc_ms", lambda: 1_000)
    monkeypatch.setattr("passivbot.random.uniform", lambda _start, _end: 0.0)
    monkeypatch.setattr("passivbot.asyncio.sleep", sleep)

    with caplog.at_level(logging.DEBUG):
        await Passivbot.maintain_hourly_cycle(bot)

    bot.restart_bot_on_too_many_errors.assert_awaited_once_with()
    assert sleeps == [5]
    assert (
        "[hourly] maintenance cycle failed | "
        "action=check_error_budget_then_retry error_type=RuntimeError"
        in caplog.text
    )
    assert "cycle-secret" not in caplog.text
    assert "traceback-value" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
