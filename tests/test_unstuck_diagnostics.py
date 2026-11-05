"""Unstuck presentation follows native decisions without adding execution policy."""

from copy import deepcopy
import logging
from types import SimpleNamespace

import pytest

from live import unstuck_diagnostics as ud


@pytest.fixture
def presentation(monkeypatch):
    clock = [0]
    monkeypatch.setattr(ud, "utc_ms", lambda: clock[0])
    events = []
    bot = SimpleNamespace(
        _unstuck_native_observation=None,
        _live_event_console_available=lambda: True,
        has_open_unstuck_order=lambda: False,
        _emit_unstuck_status_event=lambda **kwargs: (
            events.append(deepcopy(kwargs)) or True
        ),
    )
    return bot, clock, events


def row(reason="waiting_ema", *, idx=0, side="long", allowance=20.0, **values):
    return {
        "symbol_idx": idx,
        "pside": side,
        "reason": reason,
        "proposed": reason
        in {"selected", "superseded", "loss_gate_blocked", "finalization_filtered"},
        "allowance": allowance,
        "exposure_ratio": 1.0,
        "exposure_trigger_ratio": 0.9,
        "current_price": 90.0,
        "ema_trigger_price": 100.0,
        **values,
    }


def publish(presentation, *, at, rows=None, symbols=None):
    bot, clock, events = presentation
    clock[0] = at
    ud.observe(
        bot,
        {"unstuck": [row()] if rows is None else rows},
        symbols or {0: "AAA/USDT:USDT"},
    )
    ud.maybe_log_status(bot)
    return events


def test_initial_observation_at_zero_and_twenty_minute_reminder(presentation):
    events = publish(presentation, at=0)
    assert len(events) == 1
    assert events[-1]["changed"] and events[-1]["operator_visible"]
    for minute in (1, 5, 10, 15, 19):
        publish(presentation, at=minute * 60_000)
    assert len(events) == 4  # initial plus three complete structured observations
    assert all(not event["operator_visible"] for event in events[1:])
    publish(presentation, at=ud.REMINDER_MS)
    assert len(events) == 5
    assert events[-1]["operator_visible"] and not events[-1]["changed"]


def test_startup_without_a_plan_is_explicit_then_recovers(presentation):
    bot, clock, events = presentation
    ud.maybe_log_status(bot)
    assert events[-1]["side_statuses"]["long"]["status"] == "awaiting_plan"
    assert "allowance" not in events[-1]["side_statuses"]["long"]
    publish(presentation, at=1)
    assert events[-1]["side_statuses"]["long"]["status"] == "waiting_ema"
    assert events[-1]["operator_visible"]


@pytest.mark.parametrize(
    "reason", ["allowance_exhausted", "inputs_unavailable", "mode_blocked"]
)
def test_semantic_blocker_and_recovery_are_immediate(presentation, reason):
    publish(presentation, at=0)
    events = publish(
        presentation,
        at=1,
        rows=[row(reason, allowance=0.0 if reason == "allowance_exhausted" else 20.0)],
    )
    assert events[-1]["changed"] and events[-1]["operator_visible"]
    assert events[-1]["side_statuses"]["long"]["status"] == reason
    publish(presentation, at=2)
    assert len(events) == 3 and events[-1]["changed"]


def test_candidate_churn_is_coalesced_until_five_minutes(presentation):
    symbols = {0: "AAA/USDT:USDT", 1: "BBB/USDT:USDT"}
    publish(presentation, at=0, symbols=symbols)
    events = publish(presentation, at=1, rows=[row(idx=1)], symbols=symbols)
    assert len(events) == 1
    publish(presentation, at=ud.OBSERVATION_MS - 1, rows=[row(idx=1)], symbols=symbols)
    assert len(events) == 1
    publish(presentation, at=ud.OBSERVATION_MS, rows=[row(idx=1)], symbols=symbols)
    assert len(events) == 2 and events[-1]["operator_visible"]
    assert events[-1]["side_statuses"]["long"]["samples"][0]["symbol"] == symbols[1]


def test_allowance_hysteresis_accumulates_against_last_visible_value(presentation):
    publish(presentation, at=0, rows=[row(allowance=100.0)])
    events = publish(presentation, at=ud.OBSERVATION_MS, rows=[row(allowance=103.0)])
    assert not events[-1]["operator_visible"]
    publish(presentation, at=2 * ud.OBSERVATION_MS, rows=[row(allowance=104.999)])
    assert not events[-1]["operator_visible"]
    publish(presentation, at=3 * ud.OBSERVATION_MS, rows=[row(allowance=105.0)])
    assert events[-1]["operator_visible"] and events[-1]["changed"]


def test_zero_allowance_crossing_is_visible_without_changing_native_reason(
    presentation,
):
    publish(presentation, at=0, rows=[row(allowance=0.0)])
    events = publish(presentation, at=ud.OBSERVATION_MS, rows=[row(allowance=0.01)])
    assert events[-1]["operator_visible"]
    assert events[-1]["side_statuses"]["long"]["status"] == "waiting_ema"


@pytest.mark.parametrize("reason", ["selected", "loss_gate_blocked", "superseded"])
def test_proposal_details_do_not_imply_final_selection(presentation, reason):
    proposal = row(reason, qty=-0.4, price=90.0, estimated_pnl=-4.0)
    events = publish(presentation, at=0, rows=[proposal])
    info = events[-1]["side_statuses"]["long"]
    assert info["selected_symbol"] == (
        "AAA/USDT:USDT" if reason == "selected" else None
    )
    assert info["samples"][0]["proposed"]
    assert info["samples"][0]["qty"] == -0.4
    lines = ud.format_sides(events[-1]["side_statuses"])
    assert any("qty=-0.4" in line and "est_pnl=-4" in line for line in lines)
    if reason != "selected":
        assert any("selected=none" in line for line in lines)


def test_selection_clear_is_an_immediate_transition(presentation):
    publish(
        presentation,
        at=0,
        rows=[row("selected", qty=-0.4, price=90.0, estimated_pnl=-4.0)],
    )
    events = publish(presentation, at=1, rows=[row()])
    assert len(events) == 2
    assert events[-1]["changed"] and events[-1]["operator_visible"]
    assert events[-1]["side_statuses"]["long"]["selected_symbol"] is None


def test_stale_plan_does_not_retain_a_current_selection(presentation):
    bot, clock, events = presentation
    publish(presentation, at=0, rows=[row("selected", qty=-0.4)])
    assert ud.snapshot(bot, ud.STALE_MS - 1)["long"]["status"] == "selected"
    clock[0] = ud.STALE_MS
    ud.maybe_log_status(bot)
    info = events[-1]["side_statuses"]["long"]
    assert info["status"] == "stale_plan" and info["selected_symbol"] is None
    assert info["plan_age_ms"] == ud.STALE_MS
    assert all(
        "qty=" not in line for line in ud.format_sides(events[-1]["side_statuses"])
    )
    publish(presentation, at=ud.STALE_MS + 1)
    assert events[-1]["side_statuses"]["long"]["status"] == "waiting_ema"
    assert events[-1]["operator_visible"]


def test_snapshot_open_order_state_changes_are_visible(presentation):
    bot, clock, events = presentation
    publish(presentation, at=0)
    bot.has_open_unstuck_order = lambda: True
    clock[0] = 1
    ud.maybe_log_status(bot)
    assert events[-1]["operator_visible"]
    assert events[-1]["side_statuses"]["long"]["has_open_order"]


def test_rows_are_bounded_and_coin_allowances_are_not_added(presentation):
    bot, clock, events = presentation
    symbols = {idx: f"COIN{idx}/USDT:USDT" for idx in range(50)}
    rows = [row(idx=idx, allowance=float(idx)) for idx in symbols]
    ud.observe(bot, {"unstuck": rows}, symbols)
    info = ud.snapshot(bot, clock[0])["long"]
    assert info["held_count"] == 50 and len(info["samples"]) == 3
    assert info["samples_omitted"] == 47
    assert info["allowance"] == 0.0 and info["allowance_max"] == 49.0
    assert sum(info["reason_counts"].values()) == 50
    rows[0]["allowance"] = 999
    assert ud.snapshot(bot, clock[0])["long"]["allowance"] == 0.0


def test_console_records_are_bounded_and_symbol_tokens_are_sanitized(presentation):
    unusual_symbol = "COIN\nwith whitespace\t" + "Z" * 200 + "/USDT:USDT"
    huge = 1.7976931348623157e308
    rows = [
        row(
            allowance=huge,
            exposure_ratio=huge,
            exposure_trigger_ratio=huge,
            current_price=1e-300,
            ema_trigger_price=huge,
        )
    ]
    events = publish(presentation, at=0, rows=rows, symbols={0: unusual_symbol})
    lines = ud.format_sides(events[-1]["side_statuses"])
    assert all(len(line) <= 240 for line in lines)
    assert all("\n" not in line and "\t" not in line for line in lines)
    assert any("ema_trigger=" in line for line in lines)


def test_only_one_console_projection_owner(presentation, caplog):
    bot, clock, events = presentation
    with caplog.at_level(logging.INFO):
        publish(presentation, at=0)
    assert not any("[unstuck]" in record.message for record in caplog.records)
    bot._live_event_console_available = lambda: False
    with caplog.at_level(logging.INFO):
        publish(presentation, at=1, rows=[row("allowance_exhausted", allowance=0.0)])
    assert any("[unstuck]" in record.message for record in caplog.records)


def test_malformed_or_absent_diagnostics_do_not_fabricate_allowance(presentation):
    bot, clock, events = presentation
    ud.observe(bot, {}, {})
    assert ud.snapshot(bot, 0)["long"]["status"] == "awaiting_plan"
    with pytest.raises(ValueError, match="non-finite"):
        ud.observe(
            bot, {"unstuck": [row(allowance=float("nan"))]}, {0: "AAA/USDT:USDT"}
        )


def test_malformed_new_observation_clears_previous_diagnostic(presentation):
    bot, clock, events = presentation
    publish(presentation, at=0, rows=[row("selected", qty=-0.4)])
    assert ud.snapshot(bot, 0)["long"]["selected_symbol"] == "AAA/USDT:USDT"
    with pytest.raises(ValueError, match="non-finite"):
        ud.observe(
            bot, {"unstuck": [row(allowance=float("nan"))]}, {0: "AAA/USDT:USDT"}
        )
    info = ud.snapshot(bot, 0)["long"]
    assert info["status"] == "awaiting_plan"
    assert info.get("selected_symbol") is None


def test_native_projection_uses_same_bounded_formatter_in_event_sink(presentation):
    from live.event_bus import EventTypes, LiveEvent, format_console_event

    events = publish(presentation, at=0)
    sides = events[-1]["side_statuses"]
    event = LiveEvent(
        EventTypes.UNSTUCK_STATUS,
        status="succeeded",
        reason_code="unstuck_status",
        data={"sides": sides, "changed": True, "operator_visible": True},
    )
    assert format_console_event(event).splitlines() == ud.format_sides(sides)


def test_failed_optional_event_does_not_advance_delivery_baseline(presentation):
    from live.events import run_diagnostic_step

    bot, clock, events = presentation
    ud.observe(bot, {"unstuck": [row()]}, {0: "AAA/USDT:USDT"})

    def broken_sink(**kwargs):
        raise OSError("optional sink")

    bot._emit_unstuck_status_event = broken_sink
    run_diagnostic_step("unstuck", lambda: ud.maybe_log_status(bot))
    assert getattr(bot, "_unstuck_visible_ms", None) is None
    bot._emit_unstuck_status_event = lambda **kwargs: (
        events.append(deepcopy(kwargs)) or True
    )
    ud.maybe_log_status(bot)
    assert len(events) == 1 and events[-1]["changed"]


def test_rejected_real_event_enqueue_retries_visible_observation(presentation):
    from live.event_emitters import emit_live_event, emit_unstuck_status_event

    bot, clock, events = presentation
    accepted = [False]

    class Pipeline:
        def emit(self, event, **kwargs):
            assert kwargs["require_enqueue"] is True
            events.append(event)
            return event if accepted[0] else None

    bot._live_event_pipeline = Pipeline()
    bot._current_live_event_cycle_id = lambda: "test"
    bot._emit_live_event = lambda *args, **kwargs: emit_live_event(bot, *args, **kwargs)
    bot._emit_unstuck_status_event = lambda **kwargs: emit_unstuck_status_event(
        bot, **kwargs
    )
    ud.observe(bot, {"unstuck": [row()]}, {0: "AAA/USDT:USDT"})
    ud.maybe_log_status(bot)
    assert getattr(bot, "_unstuck_visible_ms", None) is None
    accepted[0] = True
    clock[0] = 1
    ud.maybe_log_status(bot)
    assert len(events) == 2
    assert bot._unstuck_visible_ms == 1


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as native

    if getattr(native, "__is_stub__", False):
        pytest.skip("requires the compiled Rust extension")
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return native


@pytest.mark.parametrize("spent", [False, True])
def test_native_json_decision_reaches_operator_snapshot(
    presentation, require_real_passivbot_rust_module, spent
):
    from test_orchestrator_json_api import (
        make_input,
        make_symbol,
        bot_params_pair,
        compute,
    )

    params = {
        "unstuck_close_fraction": 0.5,
        "unstuck_exposure_trigger_ratio": 0.001,
        "unstuck_ema_gating_enabled": False,
        "unstuck_loss_allowance_fraction": 0.01,
        "risk_wel_enforcer_enabled": False,
        "risk_twel_enforcer_enabled": False,
    }
    value = make_input(
        balance=1_000.0,
        global_bp=bot_params_pair(long_overrides=params),
        symbols=[
            make_symbol(
                0,
                bid=99.0,
                ask=99.0,
                long_pos_size=10.0,
                long_pos_price=100.0,
                long_bp=params,
            )
        ],
    )
    value["global"]["unstuck_realized_pnl_drawdown"] = 1000.0 if spent else 0.0
    output = compute(require_real_passivbot_rust_module, value)
    bot, clock, events = presentation
    ud.observe(bot, output["diagnostics"], {0: "AAA/USDT:USDT"})
    ud.maybe_log_status(bot)
    info = events[-1]["side_statuses"]["long"]
    assert info["status"] == ("allowance_exhausted" if spent else "selected")
    assert info["allowance"] == (0.0 if spent else 10.0)
    if not spent:
        order = next(
            o for o in output["orders"] if o["order_type"] == "close_unstuck_long"
        )
        assert info["samples"][0]["qty"] == order["qty"]
        assert info["samples"][0]["price"] == order["price"]


def test_real_full_queue_does_not_duplicate_console_before_retry(presentation):
    from live.event_emitters import emit_live_event, emit_unstuck_status_event
    from live.event_bus import EventTypes, LiveEvent, LiveEventPipeline, ListEventSink

    bot, clock, events = presentation
    console = ListEventSink()
    pipeline = LiveEventPipeline(queue_maxsize=1, start=False, console_sink=console)
    bot._live_event_pipeline = pipeline
    bot._current_live_event_cycle_id = lambda: "test"
    bot._emit_live_event = lambda *args, **kwargs: emit_live_event(bot, *args, **kwargs)
    bot._emit_unstuck_status_event = lambda **kwargs: emit_unstuck_status_event(
        bot, **kwargs
    )
    pipeline.emit(LiveEvent(EventTypes.UNSTUCK_STATUS), require_enqueue=True)
    console.events.clear()
    ud.observe(bot, {"unstuck": [row()]}, {0: "AAA/USDT:USDT"})
    ud.maybe_log_status(bot)
    ud.maybe_log_status(bot)
    assert not [e for e in console.events if e.event_type == EventTypes.UNSTUCK_STATUS]
    assert getattr(bot, "_unstuck_visible_ms", None) is None
    pipeline.start()
    assert pipeline.flush(timeout=2.0)
    clock[0] = 1
    ud.maybe_log_status(bot)
    assert (
        len([e for e in console.events if e.event_type == EventTypes.UNSTUCK_STATUS])
        == 1
    )
    assert bot._unstuck_visible_ms == 1
    assert pipeline.close(timeout=2.0)


def test_console_sink_emits_individually_timestampable_records(presentation, caplog):
    from live.event_bus import ConsoleSummarySink, EventTypes, LiveEvent

    events = publish(presentation, at=0)
    event = LiveEvent(
        EventTypes.UNSTUCK_STATUS, data={"sides": events[-1]["side_statuses"]}
    )
    with caplog.at_level(logging.INFO):
        ConsoleSummarySink().write(event)
    lines = [
        record.message for record in caplog.records if "[unstuck]" in record.message
    ]
    assert len(lines) >= 3
    assert all("\n" not in line and len(line) <= 240 for line in lines)
