import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from live import risk_input_recovery as recovery
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline


def make_bot(monkeypatch, *, hsl=True):
    clock = [1000.0]
    monkeypatch.setattr(recovery, "monotonic", lambda: clock[0])
    bot = SimpleNamespace(
        config={"live": {"risk_input_max_attempts": 10}},
        balance=100.0,
        balance_raw=100.0,
        stop_signal_received=False,
        _equity_hard_stop_enabled=lambda: hsl,
        _equity_hard_stop_signal_mode=lambda: "coin",
        _equity_hard_stop_start_coin_history_replay=AsyncMock(),
        _equity_hard_stop_initialize_from_history=AsyncMock(),
        _equity_hard_stop_check=AsyncMock(),
        _run_halted_hsl_protection_if_active=AsyncMock(return_value=False),
        _run_latched_hsl_supervisor_if_active=AsyncMock(return_value=False),
        _monitor_flush_snapshot=AsyncMock(),
        refresh_authoritative_state=AsyncMock(return_value=True),
    )
    bot.get_raw_balance = lambda: bot.balance_raw
    bot.get_hysteresis_snapped_balance = lambda: bot.balance

    async def sleep(seconds, *, stage):
        assert stage == "risk_inputs_waiting"
        clock[0] += seconds

    bot._sleep_unless_shutdown = AsyncMock(side_effect=sleep)
    return bot, clock


def invalid_history():
    recovery.validate_history_balances([60_000, 120_000], [-10.0, 100.0], current_balance=100.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("hsl", [True, False])
@pytest.mark.parametrize("field", ["balance", "balance_raw"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
async def test_current_balance_blocks_risk_until_valid(monkeypatch, hsl, field, value):
    bot, clock = make_bot(monkeypatch, hsl=hsl)
    setattr(bot, field, value)
    assert not await recovery.ensure_ready(bot)
    bot._equity_hard_stop_check.assert_not_awaited()
    await recovery.protect_and_wait(bot)
    bot._run_halted_hsl_protection_if_active.assert_not_awaited()
    bot._run_latched_hsl_supervisor_if_active.assert_not_awaited()
    setattr(bot, field, 100.0)
    assert await recovery.ensure_ready(bot)
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None
    assert bot._equity_hard_stop_check.await_count == int(hsl)


@pytest.mark.asyncio
async def test_startup_refreshes_zero_to_funded_without_restart(monkeypatch, caplog):
    bot, clock = make_bot(monkeypatch)
    bot.balance = bot.balance_raw = 0.0
    refreshes = []

    async def refresh():
        refreshes.append(clock[0])
        if len(refreshes) == 4:
            bot.balance = bot.balance_raw = 100.0
        return True

    bot.refresh_authoritative_state.side_effect = refresh
    with caplog.at_level(logging.INFO):
        await recovery.wait_for_startup(bot)
    assert len(refreshes) == 4
    bot._equity_hard_stop_start_coin_history_replay.assert_awaited_once()
    assert bot._risk_input_recovery is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING and "retry_count=" in r.message]) == 2
    assert "current_balance_unavailable" in caplog.text
    assert "resume_readiness_checks" in caplog.text


@pytest.mark.asyncio
async def test_startup_does_not_replay_on_failed_refresh_and_stops_cleanly(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    async def refresh():
        if clock[0] >= 1010.0:
            bot.stop_signal_received = True
        return False
    bot.refresh_authoritative_state.side_effect = refresh
    await recovery.wait_for_startup(bot)
    bot._equity_hard_stop_start_coin_history_replay.assert_not_awaited()
    assert clock[0] == 1010.0


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [True, False])
async def test_bad_history_backoff_diagnostics_protection_and_recovery(monkeypatch, caplog, startup):
    bot, clock = make_bot(monkeypatch)
    bot.config["live"]["risk_input_max_attempts"] = 11
    check = (bot._equity_hard_stop_start_coin_history_replay if startup
             else bot._equity_hard_stop_check)
    check.side_effect = invalid_history
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(structured_sinks=[sink], monitor_sinks=[])
    try:
        delays = []
        with caplog.at_level(logging.INFO):
            for i in range(10):
                assert not await recovery.ensure_ready(bot, startup=startup)
                state = bot._risk_input_recovery
                delays.append(state.retry_at - clock[0])
                # Retry pacing must not reconstruct unchanged history on each refresh.
                assert not await recovery.ensure_ready(bot, startup=startup)
                assert check.await_count == i + 1
                await recovery.protect_and_wait(bot)
                clock[0] = state.retry_at
            check.side_effect = None
            assert await recovery.ensure_ready(bot, startup=startup)
        assert delays == [5, 10, 20, 40, 80, 160, 300, 300, 300, 300]
        assert bot._run_halted_hsl_protection_if_active.await_count == 10
        assert bot._run_latched_hsl_supervisor_if_active.await_count == 10
        recovery.mark_ready(bot)
        assert bot._risk_input_recovery is None
        assert bot._live_event_pipeline.flush(timeout=2.0)
        events = [e for e in sink.events if e.event_type == EventTypes.RISK_INPUT_STATUS]
        assert events[0].data["balance_raw"] == 100.0
        assert events[0].data["first_invalid_balance"] == -10.0
        assert events[0].data["first_invalid_timestamp_ms"] == 60_000
        assert events[0].data["invalid_rows"] == 1
        assert events[-1].status == "succeeded"
        assert len(events) == check.await_count
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [ValueError("malformed payload"), TypeError("bad config")])
async def test_contract_errors_propagate_without_recovery_state(monkeypatch, exc):
    bot, _ = make_bot(monkeypatch)
    bot._equity_hard_stop_start_coin_history_replay.side_effect = exc
    with pytest.raises(type(exc)):
        await recovery.ensure_ready(bot, startup=True)
    assert not hasattr(bot, "_risk_input_recovery")


@pytest.mark.parametrize("values", [[0, 1], [-1, 1], [np.nan, 1], [np.inf, 1]])
def test_history_unavailability_is_explicit_and_bounded(values):
    with pytest.raises(recovery.RiskInputUnavailable) as raised:
        recovery.validate_history_balances([0, 60_000], values, current_balance=100)
    details = raised.value.details
    assert details["invalid_rows"] == 1
    assert details["first_invalid_balance"] is None or np.isfinite(details["first_invalid_balance"])
    assert len(details) == 6


@pytest.mark.parametrize("times,values", [([1], [1, 2]), ([[1]], [[1]]), ([1], ["secret=invalid"])])
def test_malformed_history_shape_and_types_remain_contract_errors(times, values):
    with pytest.raises((ValueError, TypeError)):
        recovery.validate_history_balances(times, values, current_balance=100)


def test_valid_empty_and_positive_history():
    recovery.validate_history_balances([], [], current_balance=100)
    recovery.validate_history_balances([60_000], [100], current_balance=100)


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [True, False])
async def test_coin_invalid_history_preserves_live_protection(compact):
    from test_hsl_coin_mode import make_coin_bot
    bot = make_coin_bot()
    state = bot._hsl_coin_state("long", "A")
    state["halted"] = True
    bot._runtime_forced_modes["long"]["A"] = "panic"
    states = bot._equity_hard_stop_coin
    modes = bot._runtime_forced_modes
    bot._equity_hard_stop_coin_protective_ready = True
    bot._equity_hard_stop_coin_replay_ready_pairs = {("long", "A")}
    payload = ({"hsl_coin_compact_replay": {
        "timestamps": [60_000, 120_000], "balances": [-10.0, 100.0],
        "realized_pnl": [0.0, 0.0], "pair_values": {},
    }} if compact else {"timeline": [
        {"timestamp": 60_000, "balance": -10.0},
        {"timestamp": 120_000, "balance": 100.0},
    ]})
    bot.get_balance_equity_history = AsyncMock(return_value=payload)
    with pytest.raises(recovery.RiskInputUnavailable):
        await bot._equity_hard_stop_initialize_coin_from_history()
    assert bot._equity_hard_stop_coin is states
    assert bot._hsl_coin_state("long", "A") is state
    assert state["halted"]
    assert bot._runtime_forced_modes is modes
    assert modes["long"]["A"] == "panic"
    assert bot._equity_hard_stop_coin_protective_ready
    assert bot._equity_hard_stop_coin_replay_ready_pairs == {("long", "A")}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_scoped_restart_respects_retry_deadline_and_preserves_state(monkeypatch, mode):
    from test_hsl_live_restart_replay import _live_restart_bot
    import passivbot_hsl as hsl
    bot, state = _live_restart_bot(mode, "expiry")
    clock = [1000.0]
    monkeypatch.setattr(recovery, "monotonic", lambda: clock[0])
    monkeypatch.setattr(hsl, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    bot._risk_input_recovery = recovery.RecoveryState(
        reason="hsl_history_balance_unavailable", retry_at=1005.0,
    )
    bot.get_balance_equity_history = AsyncMock()
    assert not await hsl._equity_hard_stop_replay_live_restart(
        bot, "long", "A" if mode == "coin" else None,
    )
    bot.get_balance_equity_history.assert_not_awaited()
    assert state["halted"]


@pytest.mark.asyncio
async def test_real_coin_startup_replay_recovers_from_invalid_history(monkeypatch):
    from test_hsl_coin_mode import make_coin_bot
    bot = make_coin_bot()
    bot.config["live"]["risk_input_max_attempts"] = 10
    clock = [1000.0]
    monkeypatch.setattr(recovery, "monotonic", lambda: clock[0])
    payload = {"hsl_coin_compact_replay": {
        "timestamps": [60_000, 120_000], "balances": [-1.0, 100.0],
        "realized_pnl": [0.0, 0.0], "pair_values": {},
    }}
    bot.get_balance_equity_history = AsyncMock(return_value=payload)
    assert not await recovery.ensure_ready(bot, startup=True)
    assert not bot._equity_hard_stop_coin_protective_ready
    payload["hsl_coin_compact_replay"]["balances"] = [100.0, 100.0]
    clock[0] = bot._risk_input_recovery.retry_at
    assert await recovery.ensure_ready(bot, startup=True)
    await bot._equity_hard_stop_coin_replay_task
    assert bot._equity_hard_stop_coin_protective_ready
    assert bot._equity_hard_stop_coin_initialized
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pside", "unified"])
async def test_aggregate_invalid_history_preserves_protection(mode):
    from test_hsl_live_restart_replay import _live_restart_bot
    bot, state = _live_restart_bot(mode, "expiry")
    original = bot.get_balance_equity_history
    async def invalid(**kwargs):
        history = await original(**kwargs)
        history["timeline"][0]["balance"] = -1.0
        return history
    bot.get_balance_equity_history = invalid
    with pytest.raises(recovery.RiskInputUnavailable):
        await bot._equity_hard_stop_initialize_from_history()
    assert bot._hsl_state("long") is state
    assert state["halted"]


@pytest.mark.asyncio
async def test_current_balance_failure_during_history_backoff_does_not_renew_budget(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_check.side_effect = invalid_history
    assert not await recovery.ensure_ready(bot)
    bot._risk_input_recovery.retry_at = clock[0] + 300.0
    bot.balance_raw = 0.0
    assert not await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery.attempts == 1
    assert bot._risk_input_recovery.retry_at == clock[0] + 300.0
    clock[0] += 300.0
    assert not await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery.reason == "current_balance_unavailable"
    assert bot._risk_input_recovery.attempts == 2
    assert bot._risk_input_recovery.retry_at == clock[0] + 10.0
    bot._equity_hard_stop_check.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_red_supervisor_returns_to_recovery_when_refresh_invalidates_balance(mode):
    from test_hsl_coin_mode import make_coin_bot
    import passivbot_hsl as hsl
    bot = make_coin_bot()
    bot.config["live"]["hsl_signal_mode"] = mode
    bot.stop_signal_received = False
    bot._equity_hard_stop_supervisor_running = False
    state = bot._hsl_coin_state("long", "A") if mode == "coin" else bot._hsl_state("long")
    bot._equity_hard_stop_coin_needs_panic_supervision = lambda *args: True
    bot._equity_hard_stop_runtime_red_latched = lambda pside: pside == "long"
    bot.balance_raw = 100.0
    bot.get_raw_balance = lambda: bot.balance_raw
    async def refresh():
        bot.balance_raw = 0.0
        return True
    bot.refresh_protective_authoritative_state = AsyncMock(side_effect=refresh)
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock()
    supervisor = (hsl._equity_hard_stop_run_coin_red_supervisor if mode == "coin"
                  else hsl._equity_hard_stop_run_red_supervisor)
    with pytest.raises(recovery.RiskInputUnavailable):
        await supervisor(bot)
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    assert not state["halted"]
    assert not bot._equity_hard_stop_supervisor_running


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [True, False])
async def test_risk_event_sink_failure_preserves_attempt_logs_and_terminal_stop(
    monkeypatch, caplog, raises
):
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch)
    bot.config["live"]["risk_input_max_attempts"] = 2
    bot.balance_raw = 0.0
    def fail(*args, **kwargs):
        if raises:
            raise RuntimeError("api_key=should-never-appear")
        return None
    bot._live_event_pipeline = SimpleNamespace(emit=fail, console_sink=object())
    bot.live_event_console_enabled = True
    with caplog.at_level(logging.WARNING):
        assert not await recovery.ensure_ready(bot)
        clock[0] = bot._risk_input_recovery.retry_at
        with pytest.raises(FatalBotException, match="2/2"):
            await recovery.ensure_ready(bot)
    attempts = [record for record in caplog.records if "retry_count=" in record.message]
    assert len(attempts) == 2
    assert attempts[0].levelno == logging.WARNING
    assert "retry_count=1" in attempts[0].message
    assert attempts[-1].levelno == logging.ERROR
    for detail in ("retry_count=2", "max_attempts=2", "balance_raw=0.0", "stop_without_restart"):
        assert detail in attempts[-1].message
    assert "should-never-appear" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [True, False])
@pytest.mark.parametrize("failure", ["current", "history", "alternating"])
async def test_budget_exhaustion_is_terminal_and_diagnostics_are_safe(
    monkeypatch, caplog, startup, failure
):
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch)
    check = (bot._equity_hard_stop_start_coin_history_replay if startup
             else bot._equity_hard_stop_check)
    check.side_effect = invalid_history
    with caplog.at_level(logging.INFO):
        for attempt in range(1, 11):
            bot.balance_raw = 0.0 if failure == "current" or (
                failure == "alternating" and attempt % 2 == 0
            ) else 100.0
            if attempt == 10:
                with pytest.raises(FatalBotException, match="10/10") as raised:
                    await recovery.ensure_ready(bot, startup=startup)
                assert isinstance(raised.value.__cause__, recovery.RiskInputUnavailable)
            else:
                assert not await recovery.ensure_ready(bot, startup=startup)
                # Polls and a different reason inside the deadline spend no attempt.
                before = bot._risk_input_recovery.attempts
                assert not await recovery.ensure_ready(bot, startup=startup)
                assert bot._risk_input_recovery.attempts == before
                clock[0] = bot._risk_input_recovery.retry_at
    attempts = [r for r in caplog.records if "retry_count=" in r.message]
    assert len(attempts) == 10
    assert attempts[-1].levelno == logging.ERROR
    assert "max_attempts=10" in attempts[-1].message
    assert "retry_delay_seconds=0.0" in attempts[-1].message
    assert "stop_without_restart" in attempts[-1].message
    assert caplog.text.count("Risk input traceback (") == 2
    assert "src/live/risk_input_recovery.py:" in caplog.text
    assert "in validate_" in caplog.text
    assert "resume_readiness_checks" not in caplog.text


@pytest.mark.asyncio
async def test_success_resets_episode_but_precheck_alone_does_not(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot.balance_raw = 0.0
    assert not await recovery.ensure_ready(bot)
    clock[0] = bot._risk_input_recovery.retry_at
    bot.balance_raw = 100.0
    assert await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery.attempts == 1
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None
    bot.balance_raw = 0.0
    assert not await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery.attempts == 1


@pytest.mark.asyncio
async def test_startup_permanent_failure_stops_at_configured_limit(monkeypatch):
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch, hsl=False)
    bot.config["live"]["risk_input_max_attempts"] = 3
    bot.balance_raw = 0.0
    with pytest.raises(FatalBotException, match="3/3"):
        await recovery.wait_for_startup(bot)
    assert clock[0] == 1015.0  # Initial failure plus retries after 5 and 10 seconds.
    assert bot.refresh_authoritative_state.await_count == 4
    assert bot._risk_input_recovery.attempts == 3


@pytest.mark.asyncio
async def test_protective_failures_share_budget_and_tracebacks_exclude_raw_text(monkeypatch, caplog):
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch)
    bot.config["live"]["risk_input_max_attempts"] = 2
    def fail():
        try:
            raise ValueError("api_key=PRIVATE_VALUE")
        except ValueError:
            invalid_history()
    bot._run_halted_hsl_protection_if_active.side_effect = fail
    with caplog.at_level(logging.WARNING):
        await recovery.protect_and_wait(bot)
        with pytest.raises(FatalBotException, match="2/2"):
            await recovery.protect_and_wait(bot)
    assert "PRIVATE_VALUE" not in caplog.text
    assert "ValueError" in caplog.text
    assert "invalid_history" in caplog.text


@pytest.mark.asyncio
async def test_exhaustion_exits_outer_lifecycle_without_full_restart(monkeypatch, caplog):
    import passivbot as pb
    from config import get_template_config
    bot, clock = make_bot(monkeypatch, hsl=False)
    config = get_template_config()
    config["live"]["risk_input_max_attempts"] = 2
    bot.config = config
    bot.balance_raw = 0.0
    bot.start_bot = AsyncMock()
    async def start():
        await recovery.wait_for_startup(bot)
    bot.start_bot.side_effect = start
    bot.cleanup_for_restart = AsyncMock()
    from unittest.mock import Mock
    setup = Mock(side_effect=[bot, AssertionError("must not recreate bot")])
    monkeypatch.setattr(pb, "bot", None, raising=False)
    monkeypatch.setattr(pb.sys, "argv", ["passivbot"])
    monkeypatch.setattr(pb, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(pb, "load_input_config", lambda *a: (config, None, None))
    monkeypatch.setattr(pb, "prepare_config", lambda *a, **k: config)
    monkeypatch.setattr(pb, "resolve_live_log_file_settings", lambda *a, **k: {"log_file": None})
    monkeypatch.setattr(pb, "configure_custom_endpoint_loader", lambda *a, **k: None)
    monkeypatch.setattr(pb, "load_user_info", lambda *a: {"exchange": "fake"})
    monkeypatch.setattr(pb, "load_markets", AsyncMock())
    monkeypatch.setattr(pb, "parse_overrides", lambda c, **k: c)
    monkeypatch.setattr(pb, "compile_runtime_config", lambda c, **k: c)
    monkeypatch.setattr(pb, "setup_bot", setup)
    with caplog.at_level(logging.INFO):
        await pb._run_live({})
    setup.assert_called_once()
    bot.cleanup_for_restart.assert_awaited_once()
    assert clock[0] == 1005.0
    assert "stop_without_restart" in caplog.text
    assert "action=stop" in caplog.text
    assert "restarting bot" not in caplog.text
