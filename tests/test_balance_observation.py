from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from live import event_emitters, state_refresh
from live.balance_composition import normalize_okx_balance_composition
from live.event_bus import EventTypes, LiveEvent, ListEventSink, LiveEventPipeline, format_console_event

NOW = 1_700_000_000_000


@pytest.fixture
def observation(monkeypatch):
    monkeypatch.setattr(event_emitters, "utc_ms", lambda: NOW)
    structured, console = ListEventSink(), ListEventSink()
    pipeline = LiveEventPipeline(structured_sinks=[structured], console_sink=console)
    def emit(event_type, require_enqueue=False, defer_sync_sinks_until_enqueued=False, **kwargs):
        return pipeline.emit(LiveEvent(event_type, **kwargs), require_enqueue=require_enqueue,
                             defer_sync_sinks_until_enqueued=defer_sync_sinks_until_enqueued)

    bot = SimpleNamespace(
        config={"live": {}},
        balance_raw=100.0, balance=100.0, positions={},
        execution_scheduled=False, stop_signal_received=False,
        freshness_ledger=SimpleNamespace(surfaces={
            name: SimpleNamespace(updated_ms=NOW, epoch=1)
            for name in ("balance", "positions")
        }),
        _live_market_snapshot_max_age_ms=lambda: 10_000,
        _current_live_event_cycle_id=lambda: "cy_1",
        _emit_live_event=emit,
        live_event_console_enabled=True, _live_event_pipeline=pipeline,
    )
    bot.get_raw_balance = lambda: bot.balance_raw
    bot.get_hysteresis_snapped_balance = lambda: bot.balance
    bot.calc_upnl_sum = AsyncMock(side_effect=AssertionError("no price fetch"))
    bot.handle_balance_update = AsyncMock(side_effect=AssertionError("no legacy callback"))
    yield bot, pipeline, structured, console
    assert pipeline.close(timeout=2.0)


def flush(observation):
    bot, pipeline, structured, console = observation
    assert pipeline.flush(timeout=2.0)
    return structured.events, console.events


@pytest.mark.parametrize("balance", [0.0, 100.0])
@pytest.mark.parametrize("scheduled", [False, True])
def test_initial_snapshot_once_without_scheduling_or_fetch(observation, balance, scheduled):
    bot, _, _, _ = observation
    bot.balance_raw = bot.balance = balance
    bot.execution_scheduled = scheduled
    event_emitters.publish_committed_balance_observation(bot)
    event_emitters.publish_committed_balance_observation(bot)
    events, console = flush(observation)
    assert len(events) == len(console) == 1
    assert events[0].event_type == EventTypes.BALANCE_CHANGED
    assert events[0].data["initial_snapshot"] is True
    assert events[0].data["balance_raw"] == balance
    assert events[0].data["equity"] == balance
    assert "[balance] initial" in format_console_event(console[0])
    assert bot.execution_scheduled is scheduled
    assert not hasattr(bot, "_previous_balance_raw")
    assert not hasattr(bot, "_monitor_last_equity")
    bot.calc_upnl_sum.assert_not_awaited()
    bot.handle_balance_update.assert_not_awaited()
    del bot._balance_observation_signature  # New runtime starts with no presentation anchors.
    event_emitters.publish_committed_balance_observation(bot)
    events, console = flush(observation)
    assert len(console) == 2 and events[-1].data["initial_snapshot"] is True


def test_raw_and_composition_only_changes_stay_durable_but_quiet(observation):
    bot, _, _, _ = observation
    event_emitters.publish_committed_balance_observation(bot)
    bot.balance_raw = 101.0
    event_emitters.publish_committed_balance_observation(bot)
    bot._balance_composition = normalize_okx_balance_composition(
        {"info": {"data": [{"details": [{"ccy": "USDT", "cashBal": "3"}]}]}}
    )
    event_emitters.publish_committed_balance_observation(bot)
    bot.balance = 102.0
    event_emitters.publish_committed_balance_observation(bot)
    events, console = flush(observation)
    assert len(events) == 4 and len(console) == 2
    assert events[1].data["balance_raw_delta"] == 1.0
    assert events[1].data["balance_snapped_delta"] == 0.0
    assert events[2].data["balance_composition"]["asset_balances"][0]["amount"] == 3.0
    assert events[-1].data["balance_snapped_delta"] == 2.0
    assert "initial_snapshot" not in events[-1].data
    assert bot.execution_scheduled is False


@pytest.mark.parametrize("fresh", [True, False])
def test_equity_uses_cached_quotes_or_is_explicitly_unknown(observation, fresh):
    bot, _, _, _ = observation
    bot.positions = {"TEST/USDT:USDT": {"long": {"size": 2.0, "price": 10.0}}}
    quote = SimpleNamespace(last=12.0, fetched_ms=NOW if fresh else NOW - 20_000,
                            is_valid=lambda: True)
    bot.market_snapshot_provider = SimpleNamespace(_cache={"TEST/USDT:USDT": quote})
    bot.c_mults = {"TEST/USDT:USDT": 1.0}
    event_emitters.publish_committed_balance_observation(bot)
    events, console = flush(observation)
    assert events[0].data["equity"] == (104.0 if fresh else None)
    if not fresh:
        assert "equity=-" in format_console_event(console[0])
    quote.last = 13.0
    event_emitters.publish_committed_balance_observation(bot)
    assert len(flush(observation)[0]) == 1  # Equity jitter is not a balance change.
    bot.calc_upnl_sum.assert_not_awaited()


def test_observer_failure_is_isolated_and_redacted(observation, caplog):
    bot, _, _, _ = observation
    def fail(*args, **kwargs):
        raise RuntimeError("token=private-observer-secret")
    bot._emit_live_event = fail
    with caplog.at_level("DEBUG"):
        event_emitters.publish_committed_balance_observation(bot)
    assert bot.execution_scheduled is False
    assert flush(observation)[0] == []
    assert "private-observer-secret" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("require_balance,valid", [
    (True, True), (False, True), (True, False),
])
async def test_deferred_report_publishes_only_complete_balance_cohort(
    observation, require_balance, valid
):
    bot, _, _, _ = observation
    bot._begin_authoritative_refresh_epoch = Mock()
    bot._fetch_authoritative_state_staged_snapshot = AsyncMock(return_value={
        "balance": 100.0, "positions": [], "open_orders": []
    })
    bot._prepare_balance_snapshot = lambda balance: {"raw": balance} if valid else None
    bot._apply_open_orders_snapshot = AsyncMock(return_value=True)
    bot._apply_positions_snapshot = lambda rows: ([], rows)
    bot._commit_balance_snapshot = Mock()
    bot._record_authoritative_surface = Mock()
    bot._positions_signature = tuple
    bot._update_entry_cooldown_position_delta_guard = Mock()
    bot.get_exchange_time = lambda: NOW
    finalized = []
    bot._finalize_authoritative_refresh_consistency = lambda plan: finalized.append(plan)
    emit = bot._emit_live_event
    def after_commit(*args, **kwargs):
        assert finalized
        return emit(*args, **kwargs)
    bot._emit_live_event = after_commit
    assert await state_refresh.refresh_protective_authoritative_state(
        bot, require_balance=require_balance
    ) is (valid or not require_balance)
    assert flush(observation)[0] == []
    bot.log_position_changes = AsyncMock()
    await state_refresh.publish_protective_account_report(bot)
    events, _ = flush(observation)
    assert len(events) == int(require_balance and valid)
    bot.handle_balance_update.assert_not_awaited()
    bot.calc_upnl_sum.assert_not_awaited()
    assert bot.execution_scheduled is False


def test_disabled_console_uses_one_legacy_fallback(observation, caplog):
    bot, pipeline, _, _ = observation
    bot.live_event_console_enabled = False
    pipeline.console_sink = None
    with caplog.at_level("INFO"):
        event_emitters.publish_committed_balance_observation(bot)
        event_emitters.publish_committed_balance_observation(bot)
        bot.balance_raw = 101.0
        event_emitters.publish_committed_balance_observation(bot)
    assert sum("[balance]" in record.message for record in caplog.records) == 1
    assert len(flush(observation)[0]) == 2


@pytest.mark.parametrize("transition", [False, True])
@pytest.mark.parametrize("failure", ["none", "raise"])
def test_failed_publication_retries_without_losing_initial_or_delta(observation, transition, failure):
    bot, _, _, _ = observation
    accepted = bot._emit_live_event
    if transition:
        event_emitters.publish_committed_balance_observation(bot)
        bot.balance_raw = bot.balance = 110.0
    previous = getattr(bot, "_balance_observation_signature", None)
    def reject(event_type, **kwargs):
        assert kwargs["require_enqueue"] is True
        assert kwargs["defer_sync_sinks_until_enqueued"] is True
        if failure == "raise":
            raise RuntimeError("private-sink-secret")
        return None
    bot._emit_live_event = reject
    event_emitters.publish_committed_balance_observation(bot)
    assert getattr(bot, "_balance_observation_signature", None) == previous
    bot._emit_live_event = accepted
    event_emitters.publish_committed_balance_observation(bot)
    event_emitters.publish_committed_balance_observation(bot)
    events, console = flush(observation)
    assert len(events) == len(console) == (2 if transition else 1)
    assert events[-1].data.get("initial_snapshot", False) is (not transition)
    assert events[-1].data["balance_snapped_delta"] == (10.0 if transition else 100.0)
    assert bot.execution_scheduled is False


def test_successful_fallback_acknowledges_without_structured_acceptance(observation, caplog):
    bot, pipeline, _, _ = observation
    bot.live_event_console_enabled = False
    pipeline.console_sink = None
    bot._emit_live_event = lambda *args, **kwargs: None
    with caplog.at_level("INFO"):
        event_emitters.publish_committed_balance_observation(bot)
        event_emitters.publish_committed_balance_observation(bot)
    assert sum("[balance]" in record.message for record in caplog.records) == 1
    assert bot._balance_observation_signature[:2] == (100.0, 100.0)
    assert flush(observation)[0] == []


def test_console_failure_does_not_prevent_structured_acceptance(observation, monkeypatch):
    bot, _, _, _ = observation
    bot.live_event_console_enabled = False
    def fail(*args, **kwargs):
        raise RuntimeError("private-console-secret")
    monkeypatch.setattr(event_emitters.logging, "info", fail)
    event_emitters.publish_committed_balance_observation(bot)
    assert bot._balance_observation_signature[:2] == (100.0, 100.0)
    assert len(flush(observation)[0]) == 1
    assert bot.execution_scheduled is False
