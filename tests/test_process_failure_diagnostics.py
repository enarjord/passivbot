import logging
from types import SimpleNamespace

import pytest
import passivbot as pb
from passivbot_exceptions import FatalBotException, RestartBotException


def crash():
    try:
        raise ValueError(
            "api_key=CHAIN_SECRET https://private.invalid?signature=SECRET"
        )
    except ValueError as cause:
        raise RuntimeError("password=OUTER_SECRET") from cause


def test_run_failure_console_preserves_chain_without_payloads(caplog, monkeypatch):
    monkeypatch.setattr(pb, "bot", SimpleNamespace(), raising=False)
    with caplog.at_level(logging.ERROR):
        try:
            crash()
        except RuntimeError as exc:
            pb._log_process_failure("passivbot error", exc)
    assert "Traceback" in caplog.text
    assert "test_process_failure_diagnostics.py:" in caplog.text
    assert "in crash" in caplog.text
    assert "raised: RuntimeError" in caplog.text
    assert "cause: ValueError" in caplog.text
    assert "action=restart" in caplog.text
    for forbidden in ("CHAIN_SECRET", "OUTER_SECRET", "private.invalid", "signature="):
        assert forbidden not in caplog.text


def test_known_missing_market_key_is_visible_but_arbitrary_key_is_not(
    caplog, monkeypatch
):
    symbol = "UNLISTED/USDT:USDT"
    bot = SimpleNamespace(coin_overrides={symbol: {}})
    monkeypatch.setattr(pb, "bot", bot, raising=False)
    with caplog.at_level(logging.ERROR):
        for key in (symbol, "api_key=SECRET", "OPAQUE_SECRET/USDT:USDT"):
            try:
                raise KeyError(key)
            except KeyError as exc:
                pb._log_process_failure("passivbot error", exc)
    assert f"missing_market_key={symbol}" in caplog.text
    assert "SECRET" not in caplog.text


def test_restarted_bots_share_repeat_suppression_but_changed_failure_prints(
    caplog, monkeypatch
):
    state = {}
    clock = [0.0]
    monkeypatch.setattr(pb.time, "monotonic", lambda: clock[0])

    def emit(key, incident):
        bot = SimpleNamespace(
            coin_overrides={key: {}},
            _startup_failure_context={"stage": "init_markets", "incident_id": incident},
        )
        monkeypatch.setattr(pb, "bot", bot, raising=False)
        try:
            raise KeyError(key)
        except KeyError as exc:
            pb._log_process_failure("passivbot error", exc, failure_state=state)

    with caplog.at_level(logging.ERROR):
        emit("AAA/USDT:USDT", "startup-1")
        clock[0] = 69
        emit("AAA/USDT:USDT", "startup-2")
        clock[0] = 301
        emit("AAA/USDT:USDT", "startup-3")
        emit("BBB/USDT:USDT", "startup-4")
    assert caplog.text.count("Traceback (") == 2
    assert "repeated=2" in caplog.text
    assert "incident_id=startup-2" not in caplog.text
    assert "incident_id=startup-3" in caplog.text
    assert "missing_market_key=BBB/USDT:USDT" in caplog.text


def test_diagnostic_projection_failure_keeps_failure_summary(caplog, monkeypatch):
    monkeypatch.setattr(pb, "bot", SimpleNamespace(), raising=False)

    def broken(_exc):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(pb, "_bounded_traceback_detail_inner", broken)
    with caplog.at_level(logging.ERROR):
        pb._log_process_failure(
            "passivbot fatal error", FatalBotException("SECRET"), action="stop"
        )
    assert "action=stop" in caplog.text
    assert "SECRET" not in caplog.text


def test_missing_key_context_is_optional_even_with_broken_observer(monkeypatch):
    class BrokenObserver:
        @property
        def markets_dict(self):
            raise RuntimeError("observer failure")

    assert (
        pb._process_failure_key_context(KeyError("AAA/USDT:USDT"), BrokenObserver())
        == ""
    )


def test_intentional_restart_does_not_create_failure_recovery_state(caplog, monkeypatch):
    monkeypatch.setattr(pb, "bot", SimpleNamespace(), raising=False)
    state = {}
    with caplog.at_level(logging.ERROR):
        try:
            raise RestartBotException("expected restart")
        except RestartBotException as exc:
            pb._log_process_failure("passivbot error", exc, failure_state=state)
    assert state == {}
    assert "Traceback" not in caplog.text
