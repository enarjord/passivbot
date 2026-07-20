import logging
from types import SimpleNamespace

import pytest

from passivbot import Passivbot, shutdown_bot


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
