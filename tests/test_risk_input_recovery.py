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
        _equity_hard_stop_enabled=lambda *a, **k: hsl,
        positions={},
        open_orders={},
        refresh_protective_authoritative_state=AsyncMock(return_value=True),
        _equity_hard_stop_signal_mode=lambda: "coin",
        _equity_hard_stop_start_coin_history_replay=AsyncMock(),
        _equity_hard_stop_initialize_from_history=AsyncMock(),
        _equity_hard_stop_check=AsyncMock(),
        _run_halted_hsl_protection_if_active=AsyncMock(return_value=False),
        _run_latched_hsl_supervisor_if_active=AsyncMock(return_value=False),
        _monitor_flush_snapshot=AsyncMock(),
        refresh_authoritative_state=AsyncMock(return_value=True),
    )
    bot.live_value = lambda key: 5.0
    bot.get_raw_balance = lambda: bot.balance_raw
    bot.get_hysteresis_snapped_balance = lambda: bot.balance

    async def sleep(seconds, *, stage):
        assert stage in {"risk_inputs_waiting", "risk_input_protective_exit"}
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
    assert refreshes == [1000.0, 1005.0, 1015.0, 1035.0]
    bot._equity_hard_stop_start_coin_history_replay.assert_awaited_once()
    assert bot._risk_input_recovery is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING and "retry_count=" in r.message]) == 3
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
                assert not await recovery.ensure_ready(bot, startup=startup)
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
    assert "retry_delay_seconds=0.0" not in attempts[-1].message
    assert "protective_exit_and_retry" in attempts[-1].message
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
    assert not await recovery.ensure_ready(bot, startup=startup)
    assert emitted[0]['details']['cause'] == 'position_mismatch'
    assert emitted[0]['details']['symbol'] == 'A'
    assert not await recovery.ensure_ready(bot, startup=startup)
    assert operation.await_count == 1
    await recovery.protect_and_wait(bot)
    bot._run_halted_hsl_protection_if_active.assert_awaited_once()
    bot._run_latched_hsl_supervisor_if_active.assert_awaited_once()
    assert not await recovery.ensure_ready(bot, startup=startup)
    clock[0] += 10
    assert not await recovery.ensure_ready(bot, startup=startup)
    assert emitted[-1]['details']['action'] == 'protective_exit_and_retry'
    assert emitted[-1]['details']['blocked_seconds'] == 15.0


@pytest.mark.asyncio
async def test_episode_evidence_recovery_after_late_fill_and_unrelated_surface_is_strict(monkeypatch):
    from live.hsl_episode import EpisodeEvidenceUnavailable
    from live.state_refresh import AuthoritativeSurfaceUnavailable
    bot, clock = make_bot(monkeypatch)
    bot._equity_hard_stop_check.side_effect = EpisodeEvidenceUnavailable('missing_opening_fill', pside='long', symbol='A')
    assert not await recovery.ensure_ready(bot)
    clock[0] += 5
    bot._equity_hard_stop_check.side_effect = None
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
async def test_green_exposure_exits_before_recovery_even_if_history_recovers(monkeypatch, startup, mode):
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
    operation.side_effect = None
    for remaining in (-2.0, -1.0):
        bot.positions['A']['short']['size'] = remaining
        assert not await recovery.ensure_ready(bot, startup=startup)
        await recovery.protect_and_wait(bot)
        assert bot._risk_input_recovery.protective_exit_pending
    assert len(plans) == 2
    assert all(p['A'] == {'short'} for p in plans)
    assert all(('UNMANAGED' in p) == (mode != 'coin') for p in plans)
    assert operation.await_count == 1
    assert bot.execute_order_plan_to_exchange.await_args.kwargs == {'configure_creations': False}
    # Submitted orders do not release the gate. Only fresh account confirmation does.
    bot.positions['A']['short']['size'] = 0.0
    bot.positions['UNMANAGED']['short']['size'] = 0.0
    bot.open_orders['A'] = []
    await recovery.protect_and_wait(bot)
    assert not bot._risk_input_recovery.protective_exit_pending
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
        assert not await recovery.ensure_ready(bot)
        clock[0] = bot._risk_input_recovery.retry_at
    assert bot._risk_input_recovery.attempts == 12
    async def fresh():
        bot.positions = {'A': {'short': {'size': -1.0}}}
        return True
    bot.refresh_protective_authoritative_state.side_effect = fresh
    bot.live_value = lambda key: 0.25
    bot._sleep_unless_shutdown = AsyncMock()
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(return_value=([], []))
    bot.execute_order_plan_to_exchange = AsyncMock()
    await recovery.protect_and_wait(bot)
    assert bot._risk_input_recovery.protective_exit_pending
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
    assert bot._risk_input_recovery.protective_exit_pending


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
    assert bot._run_latched_hsl_supervisor_if_active.await_args.kwargs["single_pass"]
    assert bot.calc_protective_panic_orders_to_cancel_and_create.await_args.kwargs == {
        'target_psides_by_symbol': {'A': {'short'}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('startup', [False, True])
@pytest.mark.parametrize('exposure', ['position', 'order'])
async def test_new_exposure_on_successful_history_retry_still_commits_to_exit(monkeypatch, startup, exposure):
    bot, clock = make_bot(monkeypatch)
    operation = bot._equity_hard_stop_start_coin_history_replay if startup else bot._equity_hard_stop_check
    operation.side_effect = invalid_history
    assert not await recovery.ensure_ready(bot, startup=startup)
    await recovery.protect_and_wait(bot)
    assert not bot._risk_input_recovery.protective_exit_pending
    clock[0] = bot._risk_input_recovery.retry_at
    operation.side_effect = None
    if exposure == 'position':
        bot.positions = {'A': {'short': {'size': -1.0}}}
    else:
        bot.open_orders = {'A': [{'position_side': 'short'}]}
    assert not await recovery.ensure_ready(bot, startup=startup)
    assert bot._risk_input_recovery.protective_exit_pending
    assert operation.await_count == 1


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
        assert bot.refresh_protective_authoritative_state.await_count == n
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
    assert bot._risk_input_recovery.protective_exit_pending
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
    bot._risk_input_recovery = recovery.RecoveryState(protective_exit_pending=True)
    error_type = {'fatal': FatalBotException, 'value': ValueError,
                  'type': TypeError, 'runtime': RuntimeError, 'authentication': AuthenticationError,
                  'bad_request': BadRequest, 'invalid_order': InvalidOrder}[kind]
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock(side_effect=error_type('invalid producer'))
    with pytest.raises(error_type):
        await recovery.protect_and_wait(bot)
    assert bot._risk_input_recovery.protective_exit_pending


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['market', 'snapshot'])
async def test_protective_reader_unavailability_is_classified_before_rust(monkeypatch, stage):
    from passivbot import Passivbot
    from live import planning_gates
    from live.state_refresh import AuthoritativeSurfaceUnavailable
    bot = SimpleNamespace(positions={'A': {'short': {'size': -1.0}}},
                          _get_orchestrator_market_snapshots=AsyncMock(return_value={}))
    def unavailable(*a, **k):
        raise RuntimeError('live snapshot unavailable')
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
    async def protective():
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
    assert history_times == ([1000.0, 1025.0, 1035.0] if exposed else [1000.0, 1005.0, 1015.0, 1035.0])
    assert len(protective_times) > len(history_times)
    if exposed:
        assert protective_times[:20] == list(range(1000, 1020))
        assert bot.execute_order_plan_to_exchange.await_count == 20
    assert bot._risk_input_recovery.attempts == len(history_times)
    bot._equity_hard_stop_check.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('unavailable', ['false', 'invalid_balance'])
async def test_incomplete_protective_account_uses_execution_cadence(monkeypatch, unavailable):
    bot, clock = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    bot._risk_input_recovery = recovery.RecoveryState(protective_exit_pending=True)
    bot.live_value = lambda key: 0.25
    bot.calc_protective_panic_orders_to_cancel_and_create = AsyncMock()
    if unavailable == 'false':
        bot.refresh_protective_authoritative_state.return_value = False
    else:
        bot.balance_raw = 0.0
    await recovery.protect_and_wait(bot)
    assert clock[0] == 1000.25
    assert bot._risk_input_recovery.protective_exit_pending
    bot.calc_protective_panic_orders_to_cancel_and_create.assert_not_awaited()
    bot._sleep_unless_shutdown.assert_awaited_once_with(0.25, stage='risk_input_protective_exit')


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['AuthenticationError', 'BadRequest', 'NotSupported', 'RequestTimeout'])
@pytest.mark.parametrize('path', ['primary', 'missing_symbol'])
async def test_real_snapshot_provider_preserves_failure_classification_in_protection(monkeypatch, kind, path):
    from ccxt.base import errors
    from market_snapshot import MarketSnapshotProvider
    from passivbot import Passivbot
    bot, _ = make_bot(monkeypatch)
    bot.positions = {'A': {'short': {'size': -1.0}}}
    bot._risk_input_recovery = recovery.RecoveryState(protective_exit_pending=True)
    original = getattr(errors, kind)('connector failure')
    async def fail(*args):
        raise original
    async def empty():
        return {}
    provider = MarketSnapshotProvider(
        exchange_name='bybit', fetch_tickers=fail if path == 'primary' else empty,
        fetch_tickers_for_symbols=fail,
    )
    bot._get_orchestrator_market_snapshots = provider.get_snapshots
    async def plan(**kwargs):
        return await Passivbot.calc_protective_panic_ideal_orders_orchestrator(bot, **kwargs)
    bot.calc_protective_panic_orders_to_cancel_and_create = plan
    bot.execute_order_plan_to_exchange = AsyncMock()
    if kind == 'RequestTimeout':
        await recovery.protect_and_wait(bot)
        bot._sleep_unless_shutdown.assert_awaited_once()
    else:
        with pytest.raises(type(original)) as caught:
            await recovery.protect_and_wait(bot)
        assert caught.value is original
        bot._sleep_unless_shutdown.assert_not_awaited()
    assert bot._risk_input_recovery.protective_exit_pending
    bot.execute_order_plan_to_exchange.assert_not_awaited()
