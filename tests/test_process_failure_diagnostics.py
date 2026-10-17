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

    from live import diagnostic_safety
    monkeypatch.setattr(diagnostic_safety, "bounded_traceback_detail_inner", broken)
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


def test_intentional_restart_does_not_create_failure_recovery_state(
    caplog, monkeypatch
):
    monkeypatch.setattr(pb, "bot", SimpleNamespace(), raising=False)
    state = {}
    with caplog.at_level(logging.ERROR):
        try:
            raise RestartBotException("expected restart")
        except RestartBotException as exc:
            pb._log_process_failure("passivbot error", exc, failure_state=state)
    assert state == {}
    assert "Traceback" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_stage", ["load_config", "load_markets", "setup_bot"])
async def test_early_startup_failure_is_bounded_and_exits_unsuccessfully(
    failing_stage, caplog, monkeypatch
):
    from unittest.mock import AsyncMock, Mock

    config = pb.get_template_config()
    config["live"]["user"] = "offline_test"
    failure = RuntimeError("api_key=EARLY_SECRET https://private.invalid?signature=SIG")
    monkeypatch.setattr(pb.sys, "argv", ["passivbot"])
    monkeypatch.setattr(pb, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        pb, "load_input_config", Mock(return_value=(config, None, None))
    )
    monkeypatch.setattr(pb, "prepare_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        pb, "resolve_live_log_file_settings", lambda *a, **kw: {"log_file": None}
    )
    monkeypatch.setattr(pb, "configure_custom_endpoint_loader", lambda *a, **kw: None)
    monkeypatch.setattr(pb, "load_user_info", Mock(return_value={"exchange": "fake"}))
    monkeypatch.setattr(pb, "load_markets", AsyncMock())
    monkeypatch.setattr(pb, "parse_overrides", lambda config, **kwargs: config)
    monkeypatch.setattr(pb, "compile_runtime_config", lambda config, **kwargs: config)
    monkeypatch.setattr(pb, "setup_bot", Mock())
    target = "load_input_config" if failing_stage == "load_config" else failing_stage
    getattr(pb, target).side_effect = failure
    # A previous invocation must not supply a stale phase, identity, or key map.
    monkeypatch.setattr(
        pb,
        "bot",
        SimpleNamespace(_startup_failure_context={"stage": "stale"}),
        raising=False,
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as result:
            await pb.main()

    assert result.value.code == 1
    assert result.value.__suppress_context__ is True
    assert pb.bot is None
    assert "action=stop" in caplog.text
    assert f"stage={failing_stage}" in caplog.text
    assert "incident_id=process-" in caplog.text
    assert "Traceback" in caplog.text
    assert "in _run_live" in caplog.text
    assert "EARLY_SECRET" not in caplog.text
    assert "private.invalid" not in caplog.text
    if failing_stage != "setup_bot":
        pb.setup_bot.assert_not_called()
    if failing_stage == "load_config":
        pb.load_user_info.assert_not_called()
        pb.load_markets.assert_not_called()


@pytest.mark.asyncio
async def test_cli_help_keeps_successful_exit_without_failure_diagnostic(
    caplog, monkeypatch
):
    monkeypatch.setattr(pb.sys, "argv", ["passivbot", "--help"])
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as result:
            await pb.main()
    assert result.value.code == 0
    assert "passivbot startup error" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["warmup", "publish_startup_ready", "runtime"])
async def test_lifecycle_failure_preserves_phase_and_does_not_report_recovery(
    phase, caplog, monkeypatch
):
    from unittest.mock import AsyncMock
    from live.event_bus import ReasonCodes

    bot = pb.Passivbot.__new__(pb.Passivbot)
    bot.config = pb.get_template_config()
    bot.config["live"]["boot_stagger_seconds"] = 0
    bot.runtime_identity = SimpleNamespace(to_dict=lambda: {})
    bot._runtime_manifest_written = True
    bot.exchange, bot.user, bot.quote = "fake", "offline_test", "USDT"
    bot.start_time_ms = 1_000
    bot.debug_mode = False
    bot.stop_signal_received = False
    bot.user_info = {"exchange": "fake"}
    bot._log_startup_banner = bot._log_memory_snapshot = lambda: None
    bot._sleep_unless_shutdown = AsyncMock()
    bot._monitor_flush_snapshot = AsyncMock()
    events, stops = [], []
    bot._monitor_record_event = lambda *a, **kw: events.append((a, kw))
    bot._monitor_emit_stop = lambda *a, **kw: stops.append((a, kw))
    bot._emit_live_event = lambda *a, **kw: None
    bot.init_markets = AsyncMock()
    bot.start_data_maintainers = AsyncMock()
    bot.start_background_candle_warmup = AsyncMock()
    bot._log_silence_watchdog_stage = "idle"
    bot._live_event_current_cycle_id = "cy_1"
    failure = RuntimeError("api_key=LIFECYCLE_SECRET")
    bot.run_execution_loop = AsyncMock(side_effect=failure)
    owner = SimpleNamespace(warmup=AsyncMock(), during_preparation=lambda x: x)
    during_runtime = phase == "runtime"
    if phase == "warmup":
        owner.warmup.side_effect = failure
    elif phase == "publish_startup_ready":
        flush_failed = False

        async def flush(**kw):
            nonlocal flush_failed
            if bot._bot_ready and not flush_failed:
                flush_failed = True
                raise failure

        bot._monitor_flush_snapshot.side_effect = flush
    monkeypatch.setattr(pb.hsl_live, "owner", lambda _: owner)
    monkeypatch.setattr(pb, "format_approved_ignored_coins", AsyncMock())
    monkeypatch.setattr(
        pb.Passivbot, "_install_asyncio_runtime_exception_handler", lambda _: None
    )
    monkeypatch.setattr(pb.Passivbot, "_startup_timing_mark", lambda *a, **kw: None)
    monkeypatch.setattr(pb, "bot", bot, raising=False)
    state = {}
    pb._log_process_failure("passivbot error", failure, failure_state=state)
    bot._process_failure_state = state
    original_state = dict(state)

    caplog.clear()
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError) as raised:
        await bot.start_bot()
    assert raised.value is failure
    assert state == original_state
    assert "recovered" not in caplog.text
    incident = next(a[2] for a, _ in events if a[0] == "error.bot")
    expected_stage = {
        "warmup": "warmup_trading_ready_candles",
        "publish_startup_ready": "publish_startup_ready",
        "runtime": "execution_loop",
    }[phase]
    assert incident["stage"] == expected_stage
    assert incident["action"] == ("restart" if during_runtime else "stop_startup")
    assert incident["cycle"] == ("cy_1" if during_runtime else "not_started")
    assert incident["incident_id"].startswith(
        "runtime-" if during_runtime else "startup-"
    )
    assert stops[-1][0][0] == (
        ReasonCodes.RUNTIME_ERROR if during_runtime else ReasonCodes.STARTUP_ERROR
    )
    assert stops[-1][1]["payload"]["stage"] == expected_stage
    # Teardown no longer has to preserve the watchdog's mutable current stage.
    bot._log_silence_watchdog_stage = "cleanup"
    with caplog.at_level(logging.ERROR):
        pb._log_process_failure("passivbot error", failure, failure_state=state)
    assert f"stage={expected_stage}" in caplog.text
    assert "LIFECYCLE_SECRET" not in caplog.text


def test_terminal_restart_summary_retains_suppressed_failure(caplog, monkeypatch):
    state = {}
    clock = [10.0]
    monkeypatch.setattr(pb.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        pb, "bot", SimpleNamespace(_log_silence_watchdog_stage="execution_loop"),
        raising=False,
    )
    error = RuntimeError("password=TERMINAL_SECRET")
    with caplog.at_level(logging.ERROR):
        pb._log_process_failure("passivbot error", error, failure_state=state)
        clock[0] = 70.0
        pb._log_process_failure("passivbot error", error, failure_state=state)
    assert state["repeats"] == 1
    caplog.clear()
    clock[0] = 790.0
    with caplog.at_level(logging.ERROR):
        pb._log_restart_budget_exhausted([1, 2, 3], 2, state)
    assert "reason=restart_budget_exhausted restarts=3 limit=2 window=24h" in caplog.text
    assert (
        "last_failure error_type=RuntimeError stage=execution_loop failure_span_s=780"
        in caplog.text
    )
    assert "TERMINAL_SECRET" not in caplog.text
    assert all(len(r.message) <= 240 for r in caplog.records)


def test_process_recovery_is_once_and_isolates_sink_failure(caplog, monkeypatch):
    from live.diagnostic_safety import log_process_recovery
    bot = SimpleNamespace(_process_failure_state={"signature": "prior-failure"})
    with monkeypatch.context() as patch:
        def broken_sink(*args):
            raise RuntimeError("sink failure")
        patch.setattr(logging, "info", broken_sink)
        log_process_recovery(bot)
    assert bot._process_failure_state
    with caplog.at_level(logging.INFO):
        log_process_recovery(bot)
        log_process_recovery(bot)
    assert bot._process_failure_state == {}
    assert caplog.text.count("ordinary execution resumed") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["pending", "rejected", "failed", "write_failed", "admitted", "noop"]
)
async def test_recovery_waits_for_admitted_ordinary_execution(outcome, caplog):
    import asyncio
    from unittest.mock import AsyncMock
    from live.hsl_live import Owner

    state = {"signature": "prior-failure"}
    bot = SimpleNamespace(
        _process_failure_state=state,
        stop_signal_received=False,
        _begin_live_event_cycle=lambda **kw: None,
        refresh_protective_authoritative_state=AsyncMock(return_value=False),
        execute_order_plan_to_exchange=AsyncMock(),
        _handle_staged_execution_precondition_error=lambda exc: (False, {}),
    )
    if outcome == "write_failed":
        bot.execute_order_plan_to_exchange.side_effect = RuntimeError("ordinary failure")
    instance = Owner(bot)
    instance.protect = AsyncMock(return_value=False)
    instance._account_matches = lambda *args: outcome != "rejected"
    if outcome != "pending":
        instance._ordinary = asyncio.get_running_loop().create_future()
        if outcome == "failed":
            instance._ordinary.set_exception(RuntimeError("ordinary failure"))
        else:
            creates = [] if outcome == "noop" else [{"symbol": "TEST/USDT:USDT"}]
            instance._ordinary.set_result(([], creates, object(), object()))
    with caplog.at_level(logging.INFO):
        if outcome in {"failed", "write_failed"}:
            with pytest.raises(RuntimeError, match="ordinary failure"):
                await instance.cycle()
        else:
            await instance.cycle()
    recovered = outcome in {"admitted", "noop"}
    assert bool(state) == (not recovered)
    assert ("ordinary execution resumed" in caplog.text) is recovered
    assert bot.execute_order_plan_to_exchange.await_count == int(
        recovered or outcome == "write_failed"
    )


@pytest.mark.asyncio
async def test_restart_budget_and_cooldown_remain_unchanged(caplog, monkeypatch):
    from unittest.mock import AsyncMock, Mock

    config = pb.get_template_config()
    config["live"].update(user="offline_test", max_n_restarts_per_day=1)
    bot = SimpleNamespace(
        config=config, stop_signal_received=False,
        start_bot=AsyncMock(side_effect=RuntimeError("password=RESTART_SECRET")),
        cleanup_for_restart=AsyncMock(),
        _log_silence_watchdog_stage="execution_loop",
    )
    monkeypatch.setattr(pb.sys, "argv", ["passivbot"])
    monkeypatch.setattr(pb, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(pb, "load_input_config", lambda *a: (config, None, None))
    monkeypatch.setattr(pb, "prepare_config", lambda *a, **kw: config)
    monkeypatch.setattr(
        pb, "resolve_live_log_file_settings", lambda *a, **kw: {"log_file": None}
    )
    monkeypatch.setattr(pb, "configure_custom_endpoint_loader", lambda *a, **kw: None)
    monkeypatch.setattr(pb, "load_user_info", lambda *a: {"exchange": "fake"})
    monkeypatch.setattr(pb, "load_markets", AsyncMock())
    monkeypatch.setattr(pb, "parse_overrides", lambda config, **kw: config)
    monkeypatch.setattr(pb, "compile_runtime_config", lambda config, **kw: config)
    monkeypatch.setattr(pb, "setup_bot", Mock(return_value=bot))
    monkeypatch.setattr(pb, "bot", None, raising=False)
    sleep = AsyncMock()
    monkeypatch.setattr(pb.asyncio, "sleep", sleep)
    with caplog.at_level(logging.ERROR):
        await pb._run_live({})
    assert pb.setup_bot.call_count == 2  # Original len(restarts) > limit policy.
    assert bot.cleanup_for_restart.await_count == 2
    assert sleep.await_count == 2 * 61  # Original inclusive 60..0 cooldown.
    assert "restarts=2 limit=1 window=24h" in caplog.text
    assert "RESTART_SECRET" not in caplog.text
