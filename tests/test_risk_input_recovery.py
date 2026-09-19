import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from live import risk_input_recovery as recovery
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline


def make_bot(monkeypatch, *, hsl=True):
    clock = [1000.0]
    monkeypatch.setattr("passivbot_hsl._equity_hard_stop_replay_live_restart", AsyncMock(return_value=True))
    monkeypatch.setattr(recovery, "monotonic", lambda: clock[0])
    bot = SimpleNamespace(
        config={"live": {"risk_input_max_attempts": 10, "hsl_unavailable_grace_seconds": 0.0}},
        balance=100.0,
        balance_raw=100.0,
        stop_signal_received=False,
        _equity_hard_stop_enabled=lambda *a, **k: hsl,
        positions={},
        open_orders={},
        refresh_protective_authoritative_state=AsyncMock(return_value=True),
        _equity_hard_stop_signal_mode=lambda: "coin",
        _equity_hard_stop_runtime_initialized=lambda *args: True,
        _equity_hard_stop_start_coin_history_replay=AsyncMock(),
        _equity_hard_stop_initialize_from_history=AsyncMock(),
        _equity_hard_stop_check=AsyncMock(),
        _run_halted_hsl_protection_if_active=AsyncMock(return_value=False),
        _run_latched_hsl_supervisor_if_active=AsyncMock(return_value=False),
        _monitor_flush_snapshot=AsyncMock(),
        refresh_authoritative_state=AsyncMock(return_value=True),
    )
    bot.get_exchange_time = lambda: int(clock[0] * 1000)
    bot._calc_upnl_sum_strict = AsyncMock(return_value=-100.0)
    bot.bot_value = lambda *args: 1
    bot._equity_hard_stop_config = lambda *args: {"red_threshold": 0.1}
    bot._pnls_manager = SimpleNamespace(get_events=lambda: [object()])
    bot._equity_hard_stop_check.side_effect = lambda: report_ready(bot)
    bot.live_value = lambda key: 5.0
    bot.get_raw_balance = lambda: bot.balance_raw
    bot.get_hysteresis_snapped_balance = lambda: bot.balance

    async def sleep(seconds, *, stage):
        assert stage in {"risk_inputs_waiting", "risk_input_protective_exit"}
        clock[0] += seconds

    bot._sleep_unless_shutdown = AsyncMock(side_effect=sleep)
    return bot, clock


def report_ready(bot):
    from live import hsl_protection
    health = hsl_protection.manager(bot)
    for scope in list(health.scopes):
        health.evaluated_successfully(scope, now_ms=bot.get_exchange_time())


def arm_exit(bot):
    from live import hsl_protection
    health = hsl_protection.manager(bot)
    for scope in hsl_protection.affected_scopes(bot, recovery._unready_hsl_targets(bot), {}):
        state = health.unavailable(scope, now_ms=bot.get_exchange_time() - 120_000,
                                   reason="emergency_threshold_crossed", grace_ms=120_000)
        state.exit_committed = True
    bot._risk_input_recovery = recovery.RecoveryState()


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
    assert bot._run_halted_hsl_protection_if_active.await_count == int(hsl)
    assert bot._run_latched_hsl_supervisor_if_active.await_count == int(hsl)
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

    async def refresh(**kwargs):
        refreshes.append(clock[0])
        if len(refreshes) == 4:
            bot.balance = bot.balance_raw = 100.0
        return True

    bot.refresh_authoritative_state.side_effect = refresh
    with caplog.at_level(logging.INFO):
        await recovery.wait_for_startup(bot)
    assert refreshes == [1000.0, 1005.0, 1010.0, 1015.0]
    bot._equity_hard_stop_start_coin_history_replay.assert_awaited_once()
    assert bot._risk_input_recovery is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING and "retry_count=" in r.message]) == 2
    assert "current_balance_unavailable" in caplog.text
    assert "resume_readiness_checks" in caplog.text


@pytest.mark.asyncio
async def test_startup_does_not_replay_on_failed_refresh_and_stops_cleanly(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    async def refresh(**kwargs):
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
                assert await recovery.ensure_ready(bot, startup=startup)
                state = bot._risk_input_recovery
                delays.append(state.retry_at - clock[0])
                # Retry pacing must not reconstruct unchanged history on each refresh.
                assert await recovery.ensure_ready(bot, startup=startup)
                assert check.await_count == i + 1
                await recovery.protect_and_wait(bot)
                clock[0] = state.retry_at
            check.side_effect = lambda: report_ready(bot)
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
    assert await recovery.ensure_ready(bot, startup=True)
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
    assert await recovery.ensure_ready(bot)
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
async def test_red_supervisor_executes_close_before_balance_repair(mode):
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
    async def refresh(**kwargs):
        bot.balance_raw = 0.0
        return True
    bot.refresh_protective_authoritative_state = AsyncMock(side_effect=refresh)
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    supervisor = (hsl._equity_hard_stop_run_coin_red_supervisor if mode == "coin"
                  else hsl._equity_hard_stop_run_red_supervisor)
    with pytest.raises(recovery.RiskInputUnavailable):
        await supervisor(bot)
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_awaited_once()
    bot.execute_order_plan_to_exchange.assert_awaited_once_with([], [], configure_creations=False)
    assert [call.kwargs for call in bot.refresh_protective_authoritative_state.await_args_list] == [
        {"require_balance": False}, {"require_balance": True}]
    assert not state["halted"]
    assert not bot._equity_hard_stop_supervisor_running


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [True, False])
async def test_risk_event_sink_failure_preserves_attempt_logs_and_terminal_stop(
    monkeypatch, caplog, raises
):
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch, hsl=False)
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
async def test_hsl_budget_exhaustion_escalates_but_keeps_recovering(
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
                assert (await recovery.ensure_ready(bot, startup=startup)) == (bot.balance_raw > 0.0)
            else:
                assert (await recovery.ensure_ready(bot, startup=startup)) == (bot.balance_raw > 0.0)
                # Polls and a different reason inside the deadline spend no attempt.
                before = bot._risk_input_recovery.attempts
                assert (await recovery.ensure_ready(bot, startup=startup)) == (bot.balance_raw > 0.0)
                assert bot._risk_input_recovery.attempts == before
                clock[0] = bot._risk_input_recovery.retry_at
    attempts = [r for r in caplog.records if "retry_count=" in r.message]
    assert len(attempts) == 10
    assert attempts[-1].levelno == logging.ERROR
    assert "max_attempts=10" in attempts[-1].message
    assert "retry_delay_seconds=0.0" not in attempts[-1].message
    assert "evaluate_emergency_after_grace_and_retry" in attempts[-1].message
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
    def fail(**kwargs):
        try:
            raise ValueError("api_key=PRIVATE_VALUE")
        except ValueError:
            invalid_history()
    bot._run_halted_hsl_protection_if_active.side_effect = fail
    with caplog.at_level(logging.WARNING):
        await recovery.protect_and_wait(bot)
        await recovery.protect_and_wait(bot)
        assert bot._risk_input_recovery.attempts == 2
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


@pytest.mark.asyncio
@pytest.mark.parametrize("console_fails", [False, True])
async def test_real_pipeline_risk_attempt_delivery_and_failure_fallback(
    monkeypatch, caplog, console_fails
):
    from live.event_bus import ConsoleSummarySink
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch, hsl=False)
    bot.config["live"]["risk_input_max_attempts"] = 2
    bot.balance_raw = 0.0
    class BrokenSink:
        def write(self, event):
            raise RuntimeError("api_key=CONSOLE_SECRET")
    structured = ListEventSink()
    pipeline = LiveEventPipeline(
        console_sink=BrokenSink() if console_fails else ConsoleSummarySink(),
        structured_sinks=[structured], monitor_sinks=[],
    )
    bot._live_event_pipeline = pipeline
    bot.live_event_console_enabled = True
    try:
        with caplog.at_level(logging.WARNING):
            assert not await recovery.ensure_ready(bot)
            clock[0] = bot._risk_input_recovery.retry_at
            with pytest.raises(FatalBotException, match="2/2"):
                await recovery.ensure_ready(bot)
        attempts = [r for r in caplog.records if "retry_count=" in r.message]
        assert len(attempts) == 2  # No duplicate on healthy console delivery.
        assert attempts[0].levelno == logging.WARNING
        assert attempts[-1].levelno == logging.ERROR
        for detail in ("retry_count=2", "max_attempts=2", "balance_raw=0.0", "stop_without_restart"):
            assert detail in attempts[-1].message
        assert "CONSOLE_SECRET" not in caplog.text
        assert pipeline.flush(timeout=2.0)
        events = [e for e in structured.events if e.event_type == EventTypes.RISK_INPUT_STATUS]
        assert [e.status for e in events] == ["deferred", "failed"]
        assert events[-1].data["traceback"]["frame_count"] > 0
    finally:
        assert pipeline.close(timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [False, True])
async def test_episode_evidence_uses_bounded_recovery_and_retains_scope(monkeypatch, startup):
    from live.hsl_episode import EpisodeEvidenceUnavailable
    from passivbot_exceptions import FatalBotException
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['risk_input_max_attempts'] = 3
    operation = bot._equity_hard_stop_start_coin_history_replay if startup else bot._equity_hard_stop_check
    operation.side_effect = EpisodeEvidenceUnavailable('position_mismatch', pside='long', symbol='A')
    emitted = []
    monkeypatch.setattr(recovery, '_emit', lambda *args, **kwargs: emitted.append(kwargs))
    assert await recovery.ensure_ready(bot, startup=startup)
    assert emitted[0]['details']['cause'] == 'position_mismatch'
    assert emitted[0]['details']['symbol'] == 'A'
    assert await recovery.ensure_ready(bot, startup=startup)
    assert operation.await_count == 1
    await recovery.protect_and_wait(bot)
    bot._run_halted_hsl_protection_if_active.assert_awaited_once()
    bot._run_latched_hsl_supervisor_if_active.assert_awaited_once()
    assert await recovery.ensure_ready(bot, startup=startup)
    clock[0] += 10
    assert await recovery.ensure_ready(bot, startup=startup)
    assert emitted[-1]['details']['action'] == 'evaluate_emergency_after_grace_and_retry'
    assert emitted[-1]['details']['blocked_seconds'] == 15.0


@pytest.mark.asyncio
async def test_episode_evidence_recovery_after_late_fill_and_unrelated_surface_is_strict(monkeypatch):
    from live.hsl_episode import EpisodeEvidenceUnavailable
    from live.state_refresh import AuthoritativeSurfaceUnavailable
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_check.side_effect = EpisodeEvidenceUnavailable('missing_opening_fill', pside='long', symbol='A')
    assert await recovery.ensure_ready(bot)
    clock[0] += 5
    bot._equity_hard_stop_check.side_effect = lambda: report_ready(bot)
    assert await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery is not None
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None
    bot._equity_hard_stop_check.side_effect = AuthoritativeSurfaceUnavailable('positions', 'bad')
    with pytest.raises(AuthoritativeSurfaceUnavailable):
        await recovery.ensure_ready(bot)


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [False, True])
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_emergency_exit_remains_committed_until_fresh_flat_confirmation(monkeypatch, startup, mode):
    from live.hsl_episode import EpisodeEvidenceUnavailable
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_signal_mode = lambda: mode
    bot._hsl_state = lambda side: {'halted': False}
    bot.positions = {'A': {'short': {'size': -2.0}}, 'UNMANAGED': {'short': {'size': -1.0}}}
    bot.open_orders = {'A': [{'id': 'entry', 'position_side': 'short', 'reduce_only': False}]}
    bot._equity_hard_stop_enabled = lambda side=None, symbol=None: symbol != 'UNMANAGED' if mode == 'coin' else True
    bot.live_value = lambda key: 0.25
    bot._sleep_unless_shutdown = AsyncMock()
    operation = bot._equity_hard_stop_start_coin_history_replay if startup and mode == 'coin' else bot._equity_hard_stop_initialize_from_history if startup else bot._equity_hard_stop_check
    operation.side_effect = EpisodeEvidenceUnavailable('position_mismatch', pside='short', symbol='A')
    plans = []
    async def plan(*, target_psides_by_symbol):
        plans.append(target_psides_by_symbol)
        return bot.open_orders['A'], [{'symbol': 'A', 'position_side': 'short', 'reduce_only': True}]
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(side_effect=plan)
    bot.execute_order_plan_to_exchange = AsyncMock()
    assert not await recovery.ensure_ready(bot, startup=startup)
    operation.side_effect = lambda: report_ready(bot)
    for remaining in (-2.0, -1.0):
        bot.positions['A']['short']['size'] = remaining
        assert not await recovery.ensure_ready(bot, startup=startup)
        await recovery.protect_and_wait(bot)
        assert bool(bot._hsl_protection_health.pending_exits())
    assert len(plans) == 2
    assert all(p['A'] == {'short'} for p in plans)
    assert all(('UNMANAGED' in p) == (mode != 'coin') for p in plans)
    assert operation.await_count == (2 if mode == "pside" and not startup else 1)
    assert bot.execute_order_plan_to_exchange.await_args.kwargs == {'configure_creations': False}
    # Submitted orders do not release the gate. Only fresh account confirmation does.
    bot.positions['A']['short']['size'] = 0.0
    bot.positions['UNMANAGED']['short']['size'] = 0.0
    bot.open_orders['A'] = []
    await recovery.protect_and_wait(bot)
    assert not bool(bot._hsl_protection_health.pending_exits())
    clock[0] = bot._risk_input_recovery.retry_at
    assert await recovery.ensure_ready(bot, startup=startup)
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None


@pytest.mark.asyncio
async def test_hsl_recovery_survives_limit_and_new_position_during_backoff(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['risk_input_max_attempts'] = 1
    bot._equity_hard_stop_check.side_effect = invalid_history
    for _ in range(12):
        assert await recovery.ensure_ready(bot)
        clock[0] = bot._risk_input_recovery.retry_at
    assert bot._risk_input_recovery.attempts == 12
    async def fresh(**kwargs):
        bot.positions = {'A': {'short': {'size': -1.0}}}
        return True
    bot.refresh_protective_authoritative_state.side_effect = fresh
    bot.live_value = lambda key: 0.25
    bot._sleep_unless_shutdown = AsyncMock()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_and_wait(bot)
    assert bool(bot._hsl_protection_health.pending_exits())
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_awaited_once_with(target_psides_by_symbol={'A': {'short'}})


@pytest.mark.asyncio
async def test_unready_exit_requires_fresh_account_and_cancels_flat_entries(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot.open_orders = {'A': [{'position_side': 'short', 'id': 'entry'}]}
    bot._equity_hard_stop_check.side_effect = invalid_history
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=(bot.open_orders['A'], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    bot.live_value = lambda key: 0.25
    assert not await recovery.ensure_ready(bot)
    bot.refresh_protective_authoritative_state.return_value = False
    await recovery.protect_and_wait(bot)
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    bot.refresh_protective_authoritative_state.return_value = True
    bot._sleep_unless_shutdown = AsyncMock()
    await recovery.protect_and_wait(bot)
    assert bot.execute_order_plan_to_exchange.await_args.args == (bot.open_orders['A'], [])


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['pending_pnl', 'degraded_pnl', 'fill_history_coverage', 'balance_consistency_check', 'fills_unavailable'])
async def test_startup_historical_readiness_failure_still_exits_exposure(monkeypatch, reason):
    bot, clock = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    bot._last_authoritative_block_reason = reason
    bot.refresh_authoritative_state.return_value = False
    bot.live_value = lambda key: 0.25
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], [{'reduce_only': True}]))
    async def execute(*args, **kwargs):
        bot.stop_signal_received = True
    bot.execute_order_plan_to_exchange = AsyncMock(side_effect=execute)
    bot._sleep_unless_shutdown = AsyncMock()
    await recovery.wait_for_startup(bot)
    bot._equity_hard_stop_start_coin_history_replay.assert_not_awaited()
    bot.execute_order_plan_to_exchange.assert_awaited_once()
    assert bool(bot._hsl_protection_health.pending_exits())


@pytest.mark.asyncio
async def test_unready_exit_does_not_starve_proven_cooldown_protection(monkeypatch):
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}, 'B': {'short': {'size': -2.0}}}
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_coin = {'short': {'B': {'halted': True}}}
    bot._equity_hard_stop_check.side_effect = invalid_history
    bot.live_value = lambda key: 0.25
    bot._sleep_unless_shutdown = AsyncMock()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    bot._run_halted_hsl_protection_if_active.return_value = True
    assert not await recovery.ensure_ready(bot)
    for _ in range(2):
        await recovery.protect_and_wait(bot)
    assert bot._run_halted_hsl_protection_if_active.await_count == 2
    assert bot._run_latched_hsl_supervisor_if_active.await_count == 2
    assert bot._run_halted_hsl_protection_if_active.await_args.kwargs == {"pace": False}
    assert bot._sleep_unless_shutdown.await_count == 2
    assert bot._sleep_unless_shutdown.await_args.args == (0.25,)
    assert bot.calc_protective_panic_orders_to_cancel_and_create.await_args.kwargs == {
        'target_psides_by_symbol': {'A': {'short'}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [False, True])
@pytest.mark.parametrize('exposure', ['position', 'order'])
async def test_successful_signal_recovery_does_not_panic_new_exposure(monkeypatch, startup, exposure):
    bot, clock = make_bot(monkeypatch)
    operation = bot._equity_hard_stop_start_coin_history_replay if startup else bot._equity_hard_stop_check
    operation.side_effect = invalid_history
    assert await recovery.ensure_ready(bot, startup=startup)
    await recovery.protect_and_wait(bot)
    assert not bool(bot._hsl_protection_health.pending_exits())
    clock[0] = bot._risk_input_recovery.retry_at
    operation.side_effect = lambda: report_ready(bot)
    if exposure == 'position':
        bot.positions = {'A': {'short': {'size': -1.0}}}
    else:
        bot.open_orders = {'A': [{'position_side': 'short'}]}
    assert await recovery.ensure_ready(bot, startup=startup)
    assert not bool(bot._hsl_protection_health.pending_exits())
    assert operation.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
@pytest.mark.parametrize('flat', [True, False])
async def test_single_pass_red_supervision_preserves_confirmations_and_yields(monkeypatch, mode, flat):
    import passivbot_hsl as hsl
    from unittest.mock import Mock
    state = {'halted': False, 'red_flat_confirmations': 0, 'pending_red_since_ms': 10,
             'cooldown_repanic_reset_pending': False}
    bot = SimpleNamespace(
        stop_signal_received=False, _equity_hard_stop_supervisor_running=False,
        _hsl_psides=lambda: ['short'], _hsl_state=lambda side: state,
        _hsl_coin_state=lambda side, symbol: state,
        _equity_hard_stop_coin={'short': {'A': state}},
        _equity_hard_stop_enabled=lambda *a, **k: True,
        _equity_hard_stop_signal_mode=lambda: mode,
        _equity_hard_stop_runtime_red_latched=lambda side: True,
        _equity_hard_stop_coin_needs_panic_supervision=lambda *a: not state['halted'],
        refresh_protective_authoritative_state=AsyncMock(return_value=True),
        get_raw_balance=lambda: 100.0, get_hysteresis_snapped_balance=lambda: 100.0,
        get_exchange_time=lambda: 1000,
        _equity_hard_stop_count_open_positions=lambda side: int(not flat),
        _equity_hard_stop_has_open_position_symbol=lambda *a: not flat,
        _equity_hard_stop_count_blocking_open_orders=lambda *a: (0, 0),
        _equity_hard_stop_count_blocking_open_orders_symbol=lambda *a: (0, 0),
        _equity_hard_stop_flatten_fill_timestamp_with_refresh=AsyncMock(return_value=20),
        _equity_hard_stop_compute_stop_event=AsyncMock(return_value={}),
        _equity_hard_stop_compute_coin_stop_event=AsyncMock(return_value={}),
        _equity_hard_stop_log_red_progress=Mock(),
        _calc_upnl_sum_strict=AsyncMock(return_value=0.0),
        _equity_hard_stop_realized_pnl_now=lambda *a: 0.0,
        _equity_hard_stop_apply_sample=lambda *a, **k: {'red_active_now': True},
        _equity_hard_stop_apply_coin_sample=lambda *a: {'red_active_now': True},
        _equity_hard_stop_set_red_runtime_forced_modes=Mock(),
        _equity_hard_stop_refresh_halted_runtime_forced_modes=Mock(),
        _equity_hard_stop_set_coin_runtime_forced_mode=Mock(),
        calc_protective_panic_orders_to_cancel_and_create=AsyncMock(return_value=([], [])),
        execute_order_plan_to_exchange=AsyncMock(), live_value=lambda key: 0.25,
    )
    async def finalize(*args, **kwargs):
        state['halted'] = True
    bot._equity_hard_stop_finalize_red_stop = AsyncMock(side_effect=finalize)
    bot._equity_hard_stop_finalize_coin_red_stop = AsyncMock(side_effect=finalize)
    monkeypatch.setattr(hsl.asyncio, 'sleep', AsyncMock())
    supervisor = hsl._equity_hard_stop_run_coin_red_supervisor if mode == 'coin' else hsl._equity_hard_stop_run_red_supervisor
    for n in (1, 2):
        await supervisor(bot, single_pass=True)
        assert bot.refresh_protective_authoritative_state.await_count == 2 * n
        assert not bot._equity_hard_stop_supervisor_running
        assert state['red_flat_confirmations'] == (n if flat else 0)
    assert state['halted'] == flat
    hsl.asyncio.sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['refresh', 'plan', 'execute'])
@pytest.mark.parametrize('failure', ['network', 'snapshot', 'restart', 'timeout', 'order_not_found'])
async def test_protective_transient_failure_keeps_exit_and_retries(monkeypatch, caplog, stage, failure):
    from ccxt.base.errors import NetworkError, OrderNotFound
    from passivbot_exceptions import RestartBotException
    from live.state_refresh import AuthoritativeSurfaceUnavailable
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    bot._equity_hard_stop_check.side_effect = invalid_history
    bot.live_value = lambda key: 0.25
    bot._sleep_unless_shutdown = AsyncMock()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    assert not await recovery.ensure_ready(bot)
    exc = {'network': NetworkError('api_key=private'),
           'snapshot': AuthoritativeSurfaceUnavailable('protective_planning_inputs', 'api_key=private'),
           'restart': RestartBotException('api_key=private'), 'timeout': TimeoutError('api_key=private'),
           'order_not_found': OrderNotFound('api_key=private')}[failure]
    operation = {'refresh': bot.refresh_protective_authoritative_state,
                 'plan': bot.calc_protective_panic_orders_to_cancel_and_create,
                 'execute': bot.execute_order_plan_to_exchange}[stage]
    operation.side_effect = exc
    with caplog.at_level(logging.WARNING):
        await recovery.protect_and_wait(bot)
    assert bool(bot._hsl_protection_health.pending_exits())
    assert 'api_key=private' not in caplog.text
    bot._sleep_unless_shutdown.assert_awaited_once_with(0.25, stage='risk_input_protective_exit')
    operation.side_effect = None
    await recovery.protect_and_wait(bot)
    assert bot.execute_order_plan_to_exchange.await_count >= 1
    assert bot.refresh_protective_authoritative_state.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['fatal', 'value', 'type', 'runtime', 'authentication', 'bad_request', 'invalid_order'])
async def test_protective_producer_defects_still_propagate(monkeypatch, kind):
    from passivbot_exceptions import FatalBotException
    from ccxt.base.errors import AuthenticationError, BadRequest, InvalidOrder
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    arm_exit(bot)
    error_type = {'fatal': FatalBotException, 'value': ValueError,
                  'type': TypeError, 'runtime': RuntimeError, 'authentication': AuthenticationError,
                  'bad_request': BadRequest, 'invalid_order': InvalidOrder}[kind]
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(side_effect=error_type('invalid producer'))
    with pytest.raises(error_type):
        await recovery.protect_and_wait(bot)
    assert bool(bot._hsl_protection_health.pending_exits())


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['market', 'snapshot'])
async def test_protective_reader_unavailability_is_classified_before_rust(monkeypatch, stage):
    from passivbot import Passivbot
    from live import planning_gates
    from live.state_refresh import AuthoritativeSurfaceUnavailable
    bot = SimpleNamespace(positions={'A': {'short': {'size': -1.0}}},
                          _get_orchestrator_market_snapshots=AsyncMock(return_value={}))
    from live.market_snapshot import MarketSnapshotUnavailable
    def unavailable(*a, **k):
        raise (MarketSnapshotUnavailable if stage == 'market' else RuntimeError)('live snapshot unavailable')
    if stage == 'market':
        bot._get_orchestrator_market_snapshots.side_effect = unavailable
    else:
        monkeypatch.setattr(planning_gates, 'build_protective_planning_snapshot', unavailable)
    with pytest.raises(AuthoritativeSurfaceUnavailable, match='protective_planning_inputs'):
        await Passivbot.calc_protective_panic_ideal_orders_orchestrator(
            bot, target_psides_by_symbol={'A': {'short'}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [True, False])
@pytest.mark.parametrize('exposed', [False, True])
@pytest.mark.parametrize('reason', ['fill_history_coverage', 'fills_unavailable'])
async def test_history_fetch_backoff_keeps_protective_refresh_running(monkeypatch, startup, exposed, reason):
    from passivbot import Passivbot
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['risk_input_max_attempts'] = 1
    bot.positions = {'A': {'short': {'size': -1.0 if exposed else 0.0}}}
    history_times, protective_times = [], []
    async def history():
        history_times.append(clock[0])
        bot._last_authoritative_block_reason = reason
        return False
    async def protective(**kwargs):
        protective_times.append(clock[0])
        # Keep a partial/unfilled exit beyond its first history retry deadline.
        if clock[0] >= 1020.0:
            bot.positions['A']['short']['size'] = 0.0
        return True
    async def sleep(seconds, *, stage):
        clock[0] += seconds
        if clock[0] >= 1040.0:
            bot.stop_signal_received = True
    bot.refresh_authoritative_state = AsyncMock(side_effect=history)
    bot.refresh_protective_authoritative_state = AsyncMock(side_effect=protective)
    bot._sleep_unless_shutdown = AsyncMock(side_effect=sleep)
    bot.live_value = lambda key: 1.0
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    bot._begin_live_event_cycle = lambda **kwargs: 'test'
    bot._set_log_silence_watchdog_context = lambda **kwargs: None
    bot._shutdown_requested = lambda: bot.stop_signal_received
    bot._handle_execution_loop_failure = AsyncMock(side_effect=AssertionError('unexpected restart path'))
    bot._emit_live_cycle_degraded = lambda **kwargs: None
    if startup:
        await recovery.wait_for_startup(bot)
    else:
        await Passivbot.run_execution_loop(bot)
    assert history_times == ([1000.0, 1021.0, 1026.0, 1031.0, 1036.0] if exposed else list(range(1000, 1040, 5)))
    assert len(protective_times) >= len(history_times)
    if exposed:
        assert sorted(set(t for t in protective_times if t < 1020)) == list(range(1000, 1020))
        assert bot.execute_order_plan_to_exchange.await_count == 20
    assert bot._risk_input_recovery.attempts <= len(history_times)
    bot._equity_hard_stop_check.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('unavailable', ['false', 'invalid_balance'])
async def test_incomplete_protective_account_uses_execution_cadence(monkeypatch, unavailable):
    bot, clock = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    arm_exit(bot)
    bot.live_value = lambda key: 0.25
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    if unavailable == 'false':
        bot.refresh_protective_authoritative_state.return_value = False
    else:
        bot.balance_raw = 0.0
    await recovery.protect_and_wait(bot)
    assert clock[0] == 1000.25
    assert bool(bot._hsl_protection_health.pending_exits())
    if unavailable == 'false':
        bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    else:
        bot.calc_protective_panic_orders_to_cancel_and_create.assert_awaited_once()
    bot._sleep_unless_shutdown.assert_awaited_once_with(0.25, stage='risk_input_protective_exit')


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['AuthenticationError', 'BadRequest', 'NotSupported', 'RequestTimeout', 'MarketSnapshotUnavailable', 'ValueError', 'TypeError', 'RuntimeError', 'KeyError'])
@pytest.mark.parametrize('path', ['primary', 'missing_symbol'])
@pytest.mark.parametrize('exchange', ['bybit', 'hyperliquid'])
async def test_real_snapshot_provider_preserves_failure_classification_in_protection(monkeypatch, kind, path, exchange):
    from ccxt.base import errors
    from market_snapshot import MarketSnapshotProvider
    from passivbot import Passivbot
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    arm_exit(bot)
    import builtins
    from live.market_snapshot import MarketSnapshotUnavailable
    error_type = (MarketSnapshotUnavailable if kind == "MarketSnapshotUnavailable"
                  else getattr(builtins, kind, None) or getattr(errors, kind))
    original = error_type('connector failure')
    async def fail(*args):
        raise original
    async def empty():
        return {}
    provider = MarketSnapshotProvider(
        exchange_name=exchange, fetch_tickers=fail if path == 'primary' else empty,
        fetch_tickers_for_symbols=fail,
    )
    from live import market_data
    bot.exchange = exchange
    bot.market_snapshot_provider = provider
    bot._live_market_snapshot_fetch_max_age_ms = lambda: 5000
    bot._log_symbols = lambda symbols, limit=12: ','.join(symbols[:limit])
    bot.symbol_ids = {}
    bot.cca = SimpleNamespace(fetch=AsyncMock(side_effect=errors.RequestTimeout('network timeout')))
    bot._hl_info_url = lambda: 'https://example.invalid/info'
    bot.fetch_tickers_for_symbols = fail
    bot._get_live_market_snapshots = lambda symbols, **kw: market_data.get_live_market_snapshots(bot, symbols, **kw)
    bot._get_orchestrator_market_snapshots = lambda symbols: market_data.get_orchestrator_market_snapshots(bot, symbols)
    async def plan(**kwargs):
        return await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, **kwargs)
    bot.calc_protective_panic_orders_to_cancel_and_create = plan
    bot.execute_order_plan_to_exchange = AsyncMock()
    if kind in {'RequestTimeout', 'MarketSnapshotUnavailable'}:
        await recovery.protect_and_wait(bot)
        bot._sleep_unless_shutdown.assert_awaited_once()
    else:
        with pytest.raises(type(original)) as caught:
            await recovery.protect_and_wait(bot)
        assert caught.value is original or caught.value.__cause__ is original
        bot._sleep_unless_shutdown.assert_not_awaited()
    assert bool(bot._hsl_protection_health.pending_exits())
    bot.execute_order_plan_to_exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_halted_owner_refresh_uses_shared_execution_cadence(monkeypatch):
    from passivbot import Passivbot
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_coin = {'short': {'A': {'halted': True}}}
    bot.positions = {'A': {'short': {'size': -1.0}}}
    bot._risk_input_recovery = recovery.RecoveryState()
    bot.refresh_protective_authoritative_state.side_effect = [False, True]
    bot.live_value = lambda key: 0.25
    bot._run_halted_hsl_protection_if_active = lambda **kw: Passivbot._run_halted_hsl_protection_if_active(bot, **kw)
    await recovery.protect_and_wait(bot)
    assert bot.refresh_protective_authoritative_state.await_count == 2
    bot._sleep_unless_shutdown.assert_awaited_once_with(0.25, stage='risk_input_protective_exit')
    assert clock[0] == 1000.25


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside', 'unified'])
async def test_grace_preserves_ordinary_readiness_and_successful_evaluation_resets(monkeypatch, mode):
    from live import hsl_protection
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot._equity_hard_stop_signal_mode = lambda: mode
    bot._hsl_state = lambda side: {'halted': False}
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_check.side_effect = invalid_history
    assert await recovery.ensure_ready(bot)
    health = hsl_protection.manager(bot)
    scope = hsl_protection.scope_for(bot, 'long', 'A')
    since = health.scopes[scope].unavailable_since_ms
    clock[0] += 119.0
    assert await recovery.ensure_ready(bot)
    assert health.scopes[scope].unavailable_since_ms == since
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is not None
    clock[0] += 61.0
    bot._equity_hard_stop_check.side_effect = lambda: report_ready(bot)
    assert await recovery.ensure_ready(bot)
    assert not health.pending_exits()
    assert health.scopes[scope].unavailable_since_ms is None
    recovery.mark_ready(bot)
    assert bot._risk_input_recovery is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "refresh", "plan", "execute"])
async def test_stuck_exit_does_not_starve_another_emergency_scope(monkeypatch, failure):
    from live import hsl_protection
    bot, clock = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    arm_exit(bot)
    bot.positions['B'] = {'short': {'size': -2.0}}
    health = hsl_protection.manager(bot)
    scope_b = hsl_protection.Scope('coin', 'short', 'B')
    health.unavailable(scope_b, now_ms=bot.get_exchange_time() - 120_000,
                       reason='missing_opening_fill', grace_ms=120_000)
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    if failure == "refresh":
        bot.refresh_protective_authoritative_state.side_effect = [OSError("transient"), True, True, True]
    elif failure == "plan":
        bot.calc_protective_panic_orders_to_cancel_and_create.side_effect = [OSError("transient"), ([], []), ([], [])]
    elif failure == "execute":
        bot.execute_order_plan_to_exchange.side_effect = [OSError("transient"), None, None]
    assert await recovery.protect_unready_hsl(bot)
    assert scope_b in health.pending_exits()
    if failure != "refresh":
        assert bot.calc_protective_panic_orders_to_cancel_and_create.await_args_list[0].kwargs == {
            "target_psides_by_symbol": {"A": {"short"}}}
    await recovery.protect_unready_hsl(bot)
    assert bot.calc_protective_panic_orders_to_cancel_and_create.await_args.kwargs == {
        'target_psides_by_symbol': {'A': {'short'}, 'B': {'short'}}}


@pytest.mark.asyncio
async def test_emergency_quote_outage_is_scoped_and_does_not_renew_grace(monkeypatch):
    from live import hsl_protection
    from live.market_snapshot import MarketSnapshotUnavailable
    bot, _ = make_bot(monkeypatch)
    bot.positions = {coin: {'short': {'size': -1.0}} for coin in ('A', 'B')}
    health = hsl_protection.manager(bot)
    for coin in ('A', 'B'):
        health.unavailable(hsl_protection.Scope('coin', 'short', coin),
                           now_ms=900_000, reason='history_timeout', grace_ms=0)
    async def upnl(pside, symbol):
        if symbol == 'A':
            raise MarketSnapshotUnavailable('no quote')
        return -100.0
    bot._calc_upnl_sum_strict = upnl
    await hsl_protection.evaluate_emergency(bot, recovery._unready_hsl_targets(bot))
    assert health.pending_exits() == {hsl_protection.Scope('coin', 'short', 'B')}
    a = health.scopes[hsl_protection.Scope('coin', 'short', 'A')]
    assert a.unavailable_since_ms == 900_000
    assert a.execution_blocked == 'MarketSnapshotUnavailable'


@pytest.mark.asyncio
async def test_missing_all_history_with_exposure_cannot_pass_normal_readiness(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._pnls_manager.get_events = lambda: []
    bot._calc_upnl_sum_strict.return_value = 100.0
    assert await recovery.ensure_ready(bot)
    bot._equity_hard_stop_check.assert_not_awaited()
    clock[0] += 120.0
    assert not await recovery.ensure_ready(bot)
    assert bool(bot._hsl_protection_health.pending_exits())


@pytest.mark.asyncio
async def test_new_position_during_replay_backoff_gets_its_own_grace(monkeypatch):
    from live import hsl_protection
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._equity_hard_stop_check.side_effect = invalid_history
    assert await recovery.ensure_ready(bot)
    clock[0] += 1.0
    bot.positions['B'] = {'short': {'size': -1.0}}
    assert await recovery.ensure_ready(bot)
    health = hsl_protection.manager(bot)
    assert health.scopes[hsl_protection.Scope('coin', 'short', 'B')].unavailable_since_ms == 1_001_000


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coin', 'pside'])
@pytest.mark.parametrize('failure', ['quote', 'episode'])
async def test_scoped_quote_failure_keeps_other_scope_evaluating(monkeypatch, mode, failure):
    from live import hsl_protection
    from live.market_snapshot import MarketSnapshotUnavailable
    from passivbot_hsl import _equity_hard_stop_scoped_upnl
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot._equity_hard_stop_signal_mode = lambda: mode
    bot._hsl_state = lambda side: {'halted': False}
    bot._equity_hard_stop_coin_initialized = True
    bot.positions = {'A': {'long': {'size': 1.0}}, 'B': {'short': {'size': -1.0}}}
    health = hsl_protection.manager(bot)
    seen = []
    async def upnl(side, symbol=None):
        if side == 'long':
            raise MarketSnapshotUnavailable('quote unavailable')
        return -100.0
    bot._calc_upnl_sum_strict = upnl
    async def check():
        for side, symbol in [('long', 'A'), ('short', 'B')]:
            key = (side, symbol if mode == 'coin' else None)
            if key in bot._hsl_readiness_excluded_pairs:
                continue
            if failure == 'episode' and side == 'long':
                from live.hsl_episode import EpisodeEvidenceUnavailable
                raise EpisodeEvidenceUnavailable('canonical_flatten_replay_unavailable', pside=key[0], symbol=key[1])
            await _equity_hard_stop_scoped_upnl(bot, *key)
            seen.append(key)
            hsl_protection.record_evaluation(bot, *key)
    bot._equity_hard_stop_check = check
    assert await recovery.ensure_ready(bot)
    clock[0] += 121.0
    assert await recovery.ensure_ready(bot)
    assert seen == [('short', 'B' if mode == 'coin' else None)] * 2
    assert not health.pending_exits()
    assert health.scopes[hsl_protection.scope_for(bot, 'short', 'B')].unavailable_since_ms is None


@pytest.mark.asyncio
async def test_disabled_hsl_retires_persisted_exit_before_reenable(monkeypatch, tmp_path):
    from live.hsl_protection import ProtectionHealth, Scope, Health
    path = tmp_path / 'protection.json'
    previous = ProtectionHealth(path)
    previous.scopes[Scope('coin', 'long', 'A')] = Health(exit_committed=True, exit_started_ms=100)
    previous.save()
    bot, _ = make_bot(monkeypatch, hsl=False)
    bot._hsl_protection_journal_path = path
    assert await recovery.ensure_ready(bot)
    assert not ProtectionHealth(path).pending_exits()
    bot._equity_hard_stop_enabled = lambda *a, **k: True
    bot._hsl_protection_health = ProtectionHealth(path)
    assert await recovery.ensure_ready(bot)
    assert not bot._hsl_protection_health.pending_exits()


@pytest.mark.asyncio
@pytest.mark.parametrize('sizing', [0.0, float('nan')])
async def test_emergency_uses_raw_balance_without_strategy_sizing_balance(monkeypatch, sizing):
    from live import hsl_protection
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot.balance = sizing
    assert not await recovery.ensure_ready(bot)
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    assert await recovery.protect_unready_hsl(bot)
    assert hsl_protection.manager(bot).pending_exits() == {hsl_protection.Scope('coin', 'long', 'A')}
    bot.execute_order_plan_to_exchange.assert_awaited_once()


@pytest.mark.asyncio
async def test_unified_exit_confirmation_waits_for_opposite_side_orders_and_positions(monkeypatch):
    from live import hsl_protection
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_signal_mode = lambda: 'unified'
    bot._hsl_state = lambda side: {'halted': False}
    bot.positions = {'A': {'long': {'size': 0.0}, 'short': {'size': -1.0}}}
    health = hsl_protection.manager(bot)
    scope = hsl_protection.Scope('unified', 'long')
    item = health.unavailable(scope, now_ms=bot.get_exchange_time(), reason='history', grace_ms=0)
    item.exit_committed = True
    item.exit_started_ms = bot.get_exchange_time()
    bot._risk_input_recovery = recovery.RecoveryState()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_unready_hsl(bot)
    assert item.exit_committed and item.exit_flat_ms is None
    bot.positions['A']['short']['size'] = 0.0
    bot.open_orders = {'A': [{'position_side': 'short'}]}
    await recovery.protect_unready_hsl(bot)
    assert item.exit_committed and item.exit_flat_ms is None
    bot.open_orders = {}
    clock[0] += 60.0
    await recovery.protect_unready_hsl(bot)
    assert not item.exit_committed and item.exit_confirmed_flat
    assert item.exit_flat_ms == bot.get_exchange_time()


@pytest.mark.asyncio
async def test_early_replay_failure_blocks_flat_initials_but_allows_held_adds(monkeypatch):
    from live import executor
    bot, _ = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._equity_hard_stop_check.side_effect = invalid_history
    assert await recovery.ensure_ready(bot)
    assert not getattr(bot, '_equity_hard_stop_coin_replay_pending_pairs', set())
    orders = [
        {'symbol': 'A', 'position_side': 'long', 'reduce_only': False},
        {'symbol': 'B', 'position_side': 'long', 'reduce_only': False},
        {'symbol': 'A', 'position_side': 'long', 'reduce_only': True},
    ]
    cls = SimpleNamespace(_emit_execution_create_filter_event=lambda *a, **k: None)
    filtered = executor._filter_hsl_replay_pending_creates(bot, cls, orders, None)
    assert filtered == [orders[0], orders[2]]
    bot._equity_hard_stop_coin_initialized = True
    assert executor._filter_hsl_replay_pending_creates(bot, cls, orders, None) == orders


@pytest.mark.asyncio
async def test_restored_exit_uses_configured_recovery_diagnostics(monkeypatch):
    bot, clock = make_bot(monkeypatch)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    arm_exit(bot)
    bot._risk_input_recovery = None
    bot.config['live']['risk_input_max_attempts'] = 3
    assert not await recovery.ensure_ready(bot)
    assert bot._risk_input_recovery.max_attempts == 3
    assert bot._risk_input_recovery.blocked_since == clock[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['pside', 'unified'])
@pytest.mark.parametrize('held', [False, True])
async def test_aggregate_replay_failure_blocks_flat_initials_but_keeps_held_adds(monkeypatch, mode, held):
    from live import executor
    bot, _ = make_bot(monkeypatch)
    bot._equity_hard_stop_signal_mode = lambda: mode
    bot._hsl_state = lambda side: {'halted': False}
    bot._equity_hard_stop_runtime_initialized = lambda side: False
    bot.positions = {'A': {'long': {'size': 1.0 if held else 0.0}}, 'B': {'long': {'size': 0.0}}}
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot._equity_hard_stop_check.side_effect = invalid_history
    assert await recovery.ensure_ready(bot)
    orders = [dict(symbol=symbol, position_side='long', side='buy', qty=1.0, reduce_only=False)
              for symbol in ('A', 'B')]
    close = dict(symbol='B', position_side='long', side='sell', qty=1.0, reduce_only=True)
    emitter = type('Emitter', (), {'_emit_execution_create_filter_event': staticmethod(lambda *a, **k: None)})
    assert executor._filter_hsl_replay_pending_creates(bot, emitter, orders + [close], None) == ([orders[0], close] if held else [close])
    bot._equity_hard_stop_runtime_initialized = lambda side: True
    assert executor._filter_hsl_replay_pending_creates(bot, emitter, orders, None) == orders


@pytest.mark.asyncio
async def test_partial_protective_wave_retains_quote_blockage_for_unexecuted_scope(monkeypatch):
    from live import hsl_protection
    bot, _ = make_bot(monkeypatch)
    bot.positions = {symbol: {'long': {'size': 1.0}} for symbol in ('A', 'B')}
    arm_exit(bot)
    bot._hsl_protective_unavailable_symbols = {'A'}
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_unready_hsl(bot)
    health = hsl_protection.manager(bot)
    assert health.scopes[hsl_protection.Scope('coin', 'long', 'A')].execution_blocked == 'MarketSnapshotUnavailable'
    assert health.scopes[hsl_protection.Scope('coin', 'long', 'B')].execution_blocked == ''
    bot._hsl_protective_unavailable_symbols.clear()
    await recovery.protect_unready_hsl(bot)
    assert all(item.execution_blocked == '' for item in health.scopes.values())


@pytest.mark.asyncio
async def test_confirmed_flat_scope_clears_quote_outage_without_another_plan(monkeypatch):
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    arm_exit(bot)
    bot.positions['A']['long']['size'] = 0.0
    bot._hsl_protective_unavailable_symbols = {'A', 'B'}
    bot.positions['B'] = {'long': {'size': 1.0}}
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_unready_hsl(bot)
    assert bot._hsl_protective_unavailable_symbols == {'B'}


@pytest.mark.asyncio
async def test_outer_refresh_protection_covers_existing_red_without_recovery_state(monkeypatch):
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._protective_panic_target_psides_by_symbol = lambda: {'A': {'long'}}
    bot._equity_hard_stop_coin_red_active = lambda: True
    bot._run_latched_hsl_supervisor_if_active.return_value = True
    assert await recovery.protect_before_history_refresh(bot)
    bot._run_latched_hsl_supervisor_if_active.assert_awaited_once()
    bot.refresh_authoritative_state.assert_not_awaited()
    # A stale panic mode alone cannot starve repairs after that side is flat.
    bot.positions = {'A': {'long': {'size': 0.0}, 'short': {'size': -1.0}}}
    assert not await recovery.protect_before_history_refresh(bot)
    assert bot._run_latched_hsl_supervisor_if_active.await_count == 1
    # Explicit panic configuration without an HSL latch still belongs to the
    # ordinary planner; this pre-refresh owner must not starve that planner.
    bot.positions['A']['long']['size'] = 1.0
    bot._equity_hard_stop_coin_red_active = lambda: False
    assert not await recovery.protect_before_history_refresh(bot)


@pytest.mark.asyncio
async def test_outer_refresh_protection_runs_cooldown_owner_before_repair(monkeypatch):
    bot, _ = make_bot(monkeypatch)
    bot._run_halted_hsl_protection_if_active.return_value = True
    assert await recovery.protect_before_history_refresh(bot)
    bot.refresh_authoritative_state.assert_not_awaited()
    bot._run_halted_hsl_protection_if_active.return_value = False
    assert not await recovery.protect_before_history_refresh(bot)


@pytest.mark.asyncio
async def test_known_red_wave_precedes_unready_emergency_balance(monkeypatch):
    bot, _ = make_bot(monkeypatch)
    seen = []
    async def normal(**kwargs):
        seen.append('normal_close')
        return True
    async def emergency(owner, health):
        seen.append('emergency_inputs')
        return False
    bot._run_latched_hsl_supervisor_if_active.side_effect = normal
    monkeypatch.setattr(recovery, '_evaluate_emergency_scopes', emergency)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    bot._protective_panic_target_psides_by_symbol = lambda: {'A': {'long'}}
    bot._equity_hard_stop_coin_red_active = lambda: True
    assert await recovery.protect_before_history_refresh(bot)
    assert seen == ['normal_close', 'emergency_inputs']


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive_key", ["n_positions", "total_wallet_exposure_limit"])
async def test_inactive_coin_scope_skips_new_emergency_but_preserves_committed_exit(monkeypatch, inactive_key):
    from live import hsl_protection as h
    from passivbot_hsl import _equity_hard_stop_coin_active_pside
    bot, _ = make_bot(monkeypatch)
    bot.positions = {"A": {"long": {"size": 1.0}, "short": {"size": -1.0}}}
    bot.bot_value = lambda side, key: 0 if side == "short" and key == inactive_key else 1
    assert not _equity_hard_stop_coin_active_pside(bot, "short", "A")
    health = h.manager(bot)
    inactive = h.Scope("coin", "short", "A")
    active = h.Scope("coin", "long", "A")
    for scope in (inactive, active):
        health.unavailable(scope, now_ms=0, reason="history", grace_ms=0)
    assert recovery._unready_hsl_targets(bot) == {"A": {"long"}}
    assert not h.affected_scopes(bot, {}, {"pside": "short", "symbol": "A"})
    # Even a caller supplying the inactive candidate cannot divide by zero.
    await h.evaluate_emergency(bot, {"A": {"long", "short"}}, refresh_fill_tail=False)
    assert health.pending_exits() == {active}
    bot._calc_upnl_sum_strict.assert_awaited_once_with("long", "A")
    health.scopes.setdefault(inactive, h.Health()).exit_committed = True
    health.scopes[inactive].exit_started_ms = bot.get_exchange_time()
    bot._risk_input_recovery = recovery.RecoveryState()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_unready_hsl(bot)
    assert bot.calc_protective_panic_orders_to_cancel_and_create.await_args.kwargs == {
        "target_psides_by_symbol": {"A": {"long", "short"}}}
    assert health.pending_exits() == {active, inactive}


@pytest.mark.asyncio
async def test_startup_restores_journal_before_any_ordinary_refresh(monkeypatch, tmp_path):
    from live.hsl_protection import ProtectionHealth, Scope, Health
    bot, _ = make_bot(monkeypatch)
    path = tmp_path / "journal.json"
    journal = ProtectionHealth(path)
    journal.scopes[Scope("coin", "long", "A")] = Health(exit_committed=True, exit_started_ms=100)
    journal.save()
    bot._hsl_protection_journal_path = path
    bot.refresh_authoritative_state = AsyncMock(side_effect=AssertionError("history before close"))
    async def refresh(*, require_balance):
        assert not require_balance
        bot.positions = {"A": {"long": {"size": 1.0}}}
        return True
    bot.refresh_protective_authoritative_state.side_effect = refresh
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    async def executed(*args, **kwargs):
        bot.stop_signal_received = True
    bot.execute_order_plan_to_exchange = AsyncMock(side_effect=executed)
    await recovery.wait_for_startup(bot)
    bot.execute_order_plan_to_exchange.assert_awaited_once()
    bot.refresh_authoritative_state.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["normal_red", "cooldown"])
async def test_outer_protection_gives_overdue_scope_a_turn_after_existing_closes(monkeypatch, owner):
    from live import hsl_protection as h
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot.positions = {symbol: {"long": {"size": 1.0}} for symbol in ("A", "B")}
    health = h.manager(bot)
    scope = h.Scope("coin", "long", "B")
    health.unavailable(scope, now_ms=bot.get_exchange_time(), reason="history", grace_ms=120000)
    calls = []
    async def close(**kwargs):
        calls.append("existing_close")
        return True
    if owner == "normal_red":
        bot._protective_panic_target_psides_by_symbol = lambda: {"A": {"long"}}
        bot._equity_hard_stop_coin_red_active = lambda: True
        bot._run_latched_hsl_supervisor_if_active.side_effect = close
    else:
        bot._run_halted_hsl_protection_if_active.side_effect = close
    async def fresh(*, require_balance):
        calls.append("balance" if require_balance else "positions_orders")
        return True
    bot.refresh_protective_authoritative_state.side_effect = fresh
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    for _ in range(2):
        assert await recovery.protect_before_history_refresh(bot)
        assert not health.pending_exits()
    clock[0] += 120.0
    calls.clear()
    assert await recovery.protect_before_history_refresh(bot)
    assert calls.index("existing_close") < calls.index("balance")
    assert scope in health.pending_exits()
    bot.execute_order_plan_to_exchange.assert_awaited_once()


@pytest.mark.asyncio
async def test_emergency_balance_timeout_does_not_monopolize_normal_close_waves(monkeypatch):
    import asyncio
    from live import hsl_protection as h
    bot, _ = make_bot(monkeypatch)
    bot.positions = {"A": {"long": {"size": 1.0}}}
    bot._protective_panic_target_psides_by_symbol = lambda: {"A": {"long"}}
    bot._equity_hard_stop_coin_red_active = lambda: True
    bot._run_latched_hsl_supervisor_if_active.return_value = True
    h.manager(bot).unavailable(h.Scope("coin", "long", "A"), now_ms=0, reason="history", grace_ms=0)
    cancelled = []
    async def hung(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)
    async def transport_timeout(**kwargs):
        from ccxt.base.errors import RequestTimeout
        try:
            await asyncio.wait_for(hung(**kwargs), timeout=0.01)
        except TimeoutError as exc:
            raise RequestTimeout("account request timed out") from exc
    bot.refresh_protective_authoritative_state.side_effect = transport_timeout
    for _ in range(2):
        assert await asyncio.wait_for(recovery.protect_before_history_refresh(bot), timeout=1.0)
    assert cancelled == [True, True]
    assert bot._run_latched_hsl_supervisor_if_active.await_count == 2


@pytest.mark.asyncio
async def test_new_exposure_observed_during_committed_exit_gets_its_own_grace(monkeypatch):
    from live import hsl_protection as h
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot.positions = {'A': {'long': {'size': 1.0}}}
    arm_exit(bot)
    async def fresh(**kwargs):
        bot.positions['B'] = {'long': {'size': 1.0}}
        return True
    bot.refresh_protective_authoritative_state.side_effect = fresh
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_before_history_refresh(bot)
    scope = h.Scope('coin', 'long', 'B')
    health = h.manager(bot)
    assert health.scopes[scope].unavailable_since_ms == int((clock[0]-5.0)*1000)
    assert not health.scopes[scope].exit_committed
    clock[0] += 120.0
    await recovery.protect_before_history_refresh(bot)
    assert health.scopes[scope].exit_committed


@pytest.mark.asyncio
async def test_stalled_emergency_quote_does_not_starve_next_scope(monkeypatch):
    import asyncio
    from live import hsl_protection as h
    bot, _ = make_bot(monkeypatch)
    bot.positions = {symbol: {'long': {'size': 1.0}} for symbol in ('A', 'B')}
    health = h.manager(bot)
    for symbol in ('A', 'B'):
        health.unavailable(h.Scope('coin', 'long', symbol), now_ms=0, reason='history', grace_ms=0)
    cancelled = []
    async def quote(side, symbol):
        if symbol == 'A':
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(symbol)
        return -100.0
    bot._calc_upnl_sum_strict = quote
    monkeypatch.setattr(h, '_EMERGENCY_QUOTE_TIMEOUT_SECONDS', 0.01)
    await asyncio.wait_for(h.evaluate_emergency(bot, recovery._unready_hsl_targets(bot)), timeout=1.0)
    assert cancelled == ['A']
    assert health.pending_exits() == {h.Scope('coin', 'long', 'B')}


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive_key", ["n_positions", "total_wallet_exposure_limit"])
async def test_inactive_interval_does_not_consume_reenabled_scope_grace(monkeypatch, tmp_path, inactive_key):
    from live import hsl_protection as h
    bot, clock = make_bot(monkeypatch)
    bot.config['live']['hsl_unavailable_grace_seconds'] = 120.0
    bot._hsl_protection_journal_path = tmp_path / 'protection.json'
    bot.positions = {'A': {'long': {'size': 1.0}}}
    health = h.manager(bot)
    scope = h.Scope('coin', 'long', 'A')
    health.unavailable(scope, now_ms=bot.get_exchange_time(), reason='history', grace_ms=120000)
    bot.bot_value = lambda side, key: 0 if key == inactive_key else 1
    h.reconcile_config(bot)
    assert scope not in h.ProtectionHealth(health.path).scopes
    clock[0] += 200.0
    bot.bot_value = lambda *args: 1
    bot._hsl_protection_health = h.ProtectionHealth(health.path)
    health = h.manager(bot)
    health.unavailable(scope, now_ms=bot.get_exchange_time(), reason='history', grace_ms=120000)
    await h.evaluate_emergency(bot, recovery._unready_hsl_targets(bot), refresh_fill_tail=False)
    assert not health.pending_exits()
    bot._calc_upnl_sum_strict.assert_not_awaited()
    clock[0] += 120.0
    await h.evaluate_emergency(bot, recovery._unready_hsl_targets(bot), refresh_fill_tail=False)
    assert health.pending_exits() == {scope}


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_read', ['account', 'manual_history'])
async def test_stalled_cooldown_reader_yields_to_due_emergency(monkeypatch, blocked_read):
    import asyncio
    import passivbot as pb
    import passivbot_hsl as hsl
    from live import hsl_protection as h
    bot, _ = make_bot(monkeypatch)
    bot.positions = {symbol: {'long': {'size': 1.0}} for symbol in ('A', 'B')}
    bot.open_orders = {'A': [{'position_side': 'long'}]}
    bot._equity_hard_stop_coin_initialized = True
    bot._equity_hard_stop_coin = {'long': {'A': {
        'halted': True, 'no_restart_latched': False,
        'cooldown_repanic_reset_pending': False, 'cooldown_until_ms': 9_000_000,
    }}}
    bot._equity_hard_stop_cooldown_position_policy = lambda: 'manual'
    bot._canonical_open_order_reduce_only = lambda order: False
    monkeypatch.setattr(hsl, '_equity_hard_stop_manual_cooldown_intervention', lambda *a, **kw: None)
    monkeypatch.setattr(pb, '_HSL_COOLDOWN_HISTORY_TIMEOUT_SECONDS', 0.01)
    cancelled = []
    async def hung():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(blocked_read)
    reads = []
    async def account(*, require_balance):
        reads.append(require_balance)
        if blocked_read == 'account' and len(reads) == 1:
            from ccxt.base.errors import RequestTimeout
            try:
                await asyncio.wait_for(hung(), timeout=0.01)
            except TimeoutError as exc:
                raise RequestTimeout('account request timed out') from exc
        return True
    bot.refresh_protective_authoritative_state.side_effect = account
    async def history(**kwargs):
        await hung()
    bot.update_pnls = history
    bot._run_halted_hsl_protection_if_active = lambda **kw: pb.Passivbot._run_halted_hsl_protection_if_active(bot, **kw)
    health = h.manager(bot)
    scope = h.Scope('coin', 'long', 'B')
    health.unavailable(scope, now_ms=0, reason='history', grace_ms=0)
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    assert await asyncio.wait_for(recovery.protect_before_history_refresh(bot), timeout=1.0)
    assert cancelled == [blocked_read]
    assert scope in health.pending_exits()
    bot.execute_order_plan_to_exchange.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('request_timeout_ms,latencies', [(30_000, (0.20, 0.15)), (90_000, (0.50, 0.10))])
async def test_committed_exit_allows_full_sequential_account_cohort(monkeypatch, request_timeout_ms, latencies):
    """Scale request seconds by 100: valid sequential reads outlast the old 30s cohort cap."""
    import asyncio
    from exchanges.binance import BinanceBot
    from live import hsl_protection as h
    bot, _ = make_bot(monkeypatch)
    bot.cca = SimpleNamespace(timeout=request_timeout_ms)
    bot.positions = {'A': {'long': {'size': 1.0}}}
    arm_exit(bot)
    calls = []
    original_wait_for = asyncio.wait_for

    async def scaled_wait_for(awaitable, timeout):
        return await original_wait_for(awaitable, timeout=None if timeout is None else timeout / 100.0)

    monkeypatch.setattr(asyncio, 'wait_for', scaled_wait_for)
    async def request(name, latency):
        await asyncio.wait_for(asyncio.sleep(latency), timeout=bot.cca.timeout / 1000.0)
        calls.append(name)
    async def positions():
        await request('positions', latencies[0])
        return [], [{'symbol': 'A', 'position_side': 'long', 'size': 1.0}]
    async def orders(symbols):
        assert symbols == {'A'}
        await request('orders', latencies[1])
        return []
    async def timed(surface, awaitable, timings):
        return await awaitable
    bot.capture_positions_snapshot = positions
    bot._select_open_order_symbols = lambda symbols, **kw: set(symbols)
    bot._fetch_open_orders_for_staged_symbols = orders
    bot._timed_authoritative_fetch = timed
    async def account(*, require_balance):
        assert not require_balance
        snapshot = await BinanceBot.capture_authoritative_state_staged_snapshot(
            bot, {'positions', 'open_orders'}, {})
        assert snapshot['positions'][0]['size'] == 1.0
        assert snapshot['open_orders'] == []
        return True
    bot.refresh_protective_authoritative_state.side_effect = account
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock(side_effect=lambda *a, **kw: calls.append('close'))

    assert await recovery._execute_emergency_exits(bot, h.manager(bot))

    assert calls == ['positions', 'orders', 'close']
    bot.execute_order_plan_to_exchange.assert_awaited_once()
