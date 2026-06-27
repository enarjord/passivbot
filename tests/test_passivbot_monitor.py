import asyncio
import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

import pytest

import live.event_emitters as live_event_emitters
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline, ReasonCodes


class RecorderPublisher:
    def __init__(self):
        self.events = []
        self.errors = []
        self.fills = []
        self.price_ticks = []
        self.completed_candles = []
        self.closed = False

    def record_event(self, kind, tags, payload=None, *, ts=None, symbol=None, pside=None):
        event = {
            "kind": kind,
            "tags": list(tags),
            "payload": payload or {},
            "ts": ts,
            "symbol": symbol,
            "pside": pside,
        }
        self.events.append(event)
        return event

    def record_error(self, kind, error, *, tags=None, payload=None, ts=None, symbol=None, pside=None):
        event = {
            "kind": kind,
            "error_type": type(error).__name__,
            "payload": payload or {},
            "tags": list(tags or []),
            "ts": ts,
            "symbol": symbol,
            "pside": pside,
        }
        self.errors.append(event)
        return event

    def record_fill(self, payload, *, ts=None, symbol=None, pside=None, raw_payload=None):
        entry = {
            "payload": payload,
            "ts": ts,
            "symbol": symbol,
            "pside": pside,
            "raw_payload": raw_payload,
        }
        self.fills.append(entry)
        return entry

    def record_price_tick(self, symbol, last, *, ts=None, bid=None, ask=None, source=None):
        entry = {
            "symbol": symbol,
            "last": last,
            "ts": ts,
            "bid": bid,
            "ask": ask,
            "source": source,
        }
        self.price_ticks.append(entry)
        return entry

    def record_completed_candles(self, symbol, timeframe, candles):
        entry = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": list(candles),
        }
        self.completed_candles.append(entry)
        return entry["candles"]

    def close(self):
        self.closed = True


def test_live_event_cycle_helpers_emit_structured_events():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _begin_live_event_cycle = pb_mod.Passivbot._begin_live_event_cycle
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_cycle_completed = pb_mod.Passivbot._emit_live_cycle_completed
        _emit_live_cycle_degraded = pb_mod.Passivbot._emit_live_cycle_degraded
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _set_live_event_context_ids = pb_mod.Passivbot._set_live_event_context_ids

        def __init__(self):
            self.exchange = "okx"
            self.user = "okx_01"
            self.bot_id = "bot_1"
            self._authoritative_refresh_epoch = 7
            self.execution_scheduled = True
            self._live_event_cycle_seq = 0
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    cycle_id = bot._begin_live_event_cycle(loop_start_ms=1000)
    bot._emit_live_cycle_degraded(
        cycle_id=cycle_id,
        reason_code="execution_barrier",
        data={"missing": ["open_orders"]},
    )
    assert bot._current_live_event_cycle_id() is None
    cycle_id_2 = bot._begin_live_event_cycle(loop_start_ms=1000)
    bot._emit_live_cycle_completed(
        cycle_id=cycle_id_2,
        loop_start_ms=1000,
        timings_ms={"execute": 3},
    )
    assert bot._current_live_event_cycle_id() is None
    bot._emit_live_event(
        EventTypes.BOT_STOPPING,
        component="lifecycle",
        tags=("bot", "lifecycle", "stop"),
        status="started",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = sink.events
    assert [event.event_type for event in events] == [
        EventTypes.CYCLE_STARTED,
        EventTypes.CYCLE_DEGRADED,
        EventTypes.CYCLE_STARTED,
        EventTypes.CYCLE_COMPLETED,
        EventTypes.BOT_STOPPING,
    ]
    assert [event.cycle_id for event in events] == [
        cycle_id,
        cycle_id,
        cycle_id_2,
        cycle_id_2,
        None,
    ]
    assert events[1].reason_code == "execution_barrier"
    assert events[3].data["orders_changed"] is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_startup_timing_mark_emits_structured_event(monkeypatch, caplog):
    import passivbot as pb_mod

    sink = ListEventSink()
    clock_values = iter([1_000, 3_500, 8_000])
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: next(clock_values))

    class FakeBot:
        _startup_timing_begin = pb_mod.Passivbot._startup_timing_begin
        _startup_timing_mark = pb_mod.Passivbot._startup_timing_mark
        _emit_live_event = pb_mod.Passivbot._emit_live_event

        def __init__(self):
            self.exchange = "gateio"
            self.user = "gateio_01"
            self.bot_id = "bot_1"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._startup_timing_begin()
    with caplog.at_level(logging.INFO):
        bot._startup_timing_mark("account")
        bot._startup_timing_mark("hsl", details="mode=coin")

    assert any("account-ready=2.50s" in record.message for record in caplog.records)
    assert any("hsl-ready=7.00s" in record.message for record in caplog.records)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = sink.events
    assert [event.event_type for event in events] == [
        EventTypes.BOT_STARTUP_TIMING,
        EventTypes.BOT_STARTUP_TIMING,
    ]
    assert events[0].component == "bot.startup"
    assert events[0].reason_code == "startup_phase_ready"
    assert events[0].data == {
        "phase": "account",
        "elapsed_ms": 2500,
        "since_previous_ms": 2500,
    }
    assert events[1].data == {
        "phase": "hsl",
        "elapsed_ms": 7000,
        "since_previous_ms": 4500,
        "details": "mode=coin",
    }
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_routine_planning_defer_summary_emits_live_event(monkeypatch):
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_planning_defer_summary_event = (
            pb_mod.Passivbot._emit_planning_defer_summary_event
        )
        _record_routine_completed_candle_defer = (
            pb_mod.Passivbot._record_routine_completed_candle_defer
        )

        def __init__(self):
            self.exchange = "kucoin"
            self.user = "kucoin_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_3"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )
            self._routine_completed_candle_defer_summary = {
                "window_start_ms": 1_000,
                "last_log_ms": 0,
                "count": 19,
                "symbols": {"XRP/USDT:USDT"},
            }

        def _log_symbols(self, symbols, limit=8):
            del limit
            return ",".join(str(symbol) for symbol in symbols)

    bot = FakeBot()
    monkeypatch.setattr(pb_mod.planning_gates, "_utc_ms", lambda: 1_861_000)

    bot._record_routine_completed_candle_defer(
        {
            "missing": ["completed_candles"],
            "required": ["balance", "completed_candles", "market_snapshot"],
            "context": "rust order calculation",
            "epoch": 42,
            "invalid": {
                "completed_candles": [
                    {
                        "reason": "signature_mismatch",
                        "mismatch_type": "completed_candle_target_changed",
                        "changed_symbols": ["BTC/USDT:USDT"],
                        "missing_symbols": ["ETH/USDT:USDT"],
                    }
                ]
            },
        }
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.event_type == EventTypes.PLANNING_DEFER_SUMMARY
    assert event.cycle_id == "cy_3"
    assert event.reason_code == "completed_candle_target_changed"
    assert event.status == "deferred"
    assert event.data["count"] == 20
    assert event.data["window_s"] == 1860
    assert event.data["symbols"] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "XRP/USDT:USDT",
    ]
    assert event.data["symbols_count"] == 3
    assert event.data["symbols_truncated"] is False
    assert event.data["missing"] == ["completed_candles"]
    assert event.data["invalid_surfaces"] == ["completed_candles"]
    assert bot._routine_completed_candle_defer_summary["count"] == 0

    bot._emit_planning_defer_summary_event(
        reason_code="completed_candle_target_changed",
        count=40,
        window_s=60,
        symbols=[f"S{i:02d}/USDT:USDT" for i in range(40)],
        details={},
    )
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    wide_event = sink.events[-1]
    assert wide_event.event_type == EventTypes.PLANNING_DEFER_SUMMARY
    assert wide_event.data["symbols_count"] == 40
    assert wide_event.data["symbols_truncated"] is True
    assert len(wide_event.data["symbols"]) == 32
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_planning_symbol_state_event_summarizes_and_bounds_records():
    import passivbot as pb_mod
    from live.planning_availability import (
        PlanningAvailability,
        PlanningAvailabilityRecord,
    )

    structured_sink = ListEventSink()
    monitor_sink = ListEventSink()

    class FakeBot:
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_planning_symbol_state_event = (
            pb_mod.Passivbot._emit_planning_symbol_state_event
        )

        def __init__(self):
            self.exchange = "gateio"
            self.user = "gateio_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_3"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[structured_sink],
                monitor_sinks=[monitor_sink],
            )

    records = tuple(
        PlanningAvailabilityRecord(
            cycle_id=7,
            snapshot_id="snap_7",
            symbol=f"S{i:02d}/USDT:USDT",
            position_side="long" if i % 2 == 0 else "short",
            order_class="initial_entry" if i % 3 else "hsl_panic_close",
            status="unavailable",
            reason_code=(
                "missing_canonical_candles" if i % 2 == 0 else "market_prices_too_old"
            ),
            required_surfaces=("balance", "market_prices", "canonical_candles"),
            unavailable_surfaces=(
                ("canonical_candles",) if i % 2 == 0 else ("market_prices",)
            ),
            packet_revisions=(("balance", 1),),
            surface_epochs=(("completed_candles", 1, 2),),
        )
        for i in range(40)
    )
    availability = PlanningAvailability(
        cycle_id=7,
        snapshot_id="snap_7",
        records=records,
    )
    bot = FakeBot()

    bot._emit_planning_symbol_state_event(
        availability,
        context="rust order calculation",
        sample_limit=5,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert len(structured_sink.events) == 1
    assert monitor_sink.events == structured_sink.events
    event = structured_sink.events[0]
    assert event.event_type == EventTypes.PLANNING_SYMBOL_STATE
    assert event.level == "debug"
    assert event.cycle_id == "cy_3"
    assert event.snapshot_id == "snap_7"
    assert event.status == "succeeded"
    assert event.reason_code == "snapshot_symbol_state"
    assert event.data["context"] == "rust order calculation"
    assert event.data["summary"]["cycle_id"] == 7
    assert event.data["summary"]["record_count"] == 40
    assert event.data["summary"]["status_counts"] == {"unavailable": 40}
    assert event.data["unavailable_count"] == 40
    assert event.data["unavailable_by_reason"] == {
        "market_prices_too_old": 20,
        "missing_canonical_candles": 20,
    }
    assert event.data["unavailable_by_surface"] == {
        "canonical_candles": 20,
        "market_prices": 20,
    }
    assert event.data["unavailable_symbols_count"] == 40
    assert event.data["unavailable_symbols_truncated"] is True
    assert len(event.data["unavailable_symbols"]) == 32
    assert event.data["records_sample_count"] == 5
    assert event.data["records_truncated"] is True
    assert event.data["records_sample"][0] == {
        "symbol": "S00/USDT:USDT",
        "pside": "long",
        "order_class": "hsl_panic_close",
        "reason_code": "missing_canonical_candles",
        "unavailable_surfaces": ["canonical_candles"],
        "required_surfaces": ["balance", "market_prices", "canonical_candles"],
    }
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_order_wave_summary_emits_live_event(caplog):
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _begin_order_wave = pb_mod.Passivbot._begin_order_wave
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_order_wave_completed_event = (
            pb_mod.Passivbot._emit_order_wave_completed_event
        )
        _log_order_wave_summary = pb_mod.Passivbot._log_order_wave_summary

        def __init__(self):
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self._order_wave_seq = 0
            self._live_event_current_cycle_id = "cy_9"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    wave = bot._begin_order_wave(
        [{"symbol": "BTC/USDT:USDT", "qty": 1, "price": 100.0}],
        [{"symbol": "ETH/USDT:USDT", "qty": 2, "price": 200.0}],
    )
    wave["cancel_posted"] = 1
    wave["create_posted"] = 1
    wave["cancel_ms"] = 11
    wave["create_ms"] = 22

    with caplog.at_level(logging.INFO):
        bot._log_order_wave_summary(wave)

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.ORDER_WAVE_COMPLETED
    assert event.cycle_id == "cy_9"
    assert event.order_wave_id == "ow_1"
    assert event.status == "succeeded"
    assert event.data["cancel_posted"] == 1
    assert event.data["create_posted"] == 1
    assert set(event.data["symbols"]) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_action_planned_emitter_records_bounded_summary():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_action_planned_event = pb_mod.Passivbot._emit_action_planned_event
        _emit_live_event = pb_mod.Passivbot._emit_live_event

        def __init__(self):
            self.exchange = "okx"
            self.user = "okx_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_10"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    orders = [
        {
            "symbol_idx": 0,
            "qty": 1.5,
            "price": 100.0,
            "order_type": "entry_grid_normal_long",
            "execution_type": "limit",
        },
        {
            "symbol_idx": 1,
            "qty": -2.0,
            "price": 101.0,
            "order_type": "close_grid_short",
            "execution_type": "market",
        },
    ]

    bot._emit_action_planned_event(
        orders=orders,
        idx_to_symbol={0: "BTC/USDT:USDT", 1: "ETH/USDT:USDT"},
        output_hash="out_hash",
        remote_call_id="rust_1",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.ACTION_PLANNED
    assert event.cycle_id == "cy_10"
    assert event.remote_call_id == "rust_1"
    assert event.status == "succeeded"
    assert event.reason_code == "rust_output_actions"
    assert event.raw_hash == "out_hash"
    assert event.data["order_count"] == 2
    assert event.data["by_order_type"] == {
        "close_grid_short": 1,
        "entry_grid_normal_long": 1,
    }
    assert event.data["by_pside"] == {"long": 1, "short": 1}
    assert event.data["by_execution_type"] == {"limit": 1, "market": 1}
    assert event.data["symbols"]["sample"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert event.data["orders_truncated"] is False
    assert event.data["orders_sample"][0] == {
        "index": 0,
        "symbol": "BTC/USDT:USDT",
        "symbol_idx": 0,
        "pside": "long",
        "order_type": "entry_grid_normal_long",
        "execution_type": "limit",
        "qty": 1.5,
        "price": 100.0,
    }
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_rust_orchestrator_emitters_record_bounded_summaries():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_rust_orchestrator_called_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_called_event
        )
        _emit_rust_orchestrator_returned_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_returned_event
        )

        def __init__(self):
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_rust"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._emit_rust_orchestrator_called_event(
        rust_call_id="rust_7",
        input_hash="input_hash",
        symbol_count=4,
        tradable_count=3,
        ema_unavailable_count=1,
        trailing_unavailable_count=2,
        hedge_mode=True,
        strategy_kind="recursive_grid",
    )
    bot._emit_rust_orchestrator_returned_event(
        rust_call_id="rust_7",
        status="succeeded",
        input_hash="input_hash",
        output_hash="output_hash",
        elapsed_ms=12,
        order_count=5,
        diagnostics={"min_cost": {}, "forager": {}},
    )
    bot._emit_rust_orchestrator_returned_event(
        rust_call_id="rust_8",
        status="failed",
        input_hash="failed_input_hash",
        elapsed_ms=9,
        error=ValueError("MissingEma { symbol_idx: 0 } apiKey=SECRET token SECRET"),
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    called, returned, failed = sink.events
    assert called.event_type == EventTypes.RUST_ORCHESTRATOR_CALLED
    assert called.cycle_id == "cy_rust"
    assert called.remote_call_id == "rust_7"
    assert called.raw_hash == "input_hash"
    assert called.data == {
        "symbol_count": 4,
        "tradable_count": 3,
        "ema_unavailable_count": 1,
        "trailing_unavailable_count": 2,
        "hedge_mode": True,
        "strategy_kind": "recursive_grid",
        "input_hash": "input_hash",
    }
    assert returned.event_type == EventTypes.RUST_ORCHESTRATOR_RETURNED
    assert returned.status == "succeeded"
    assert returned.raw_hash == "output_hash"
    assert returned.data["elapsed_ms"] == 12
    assert returned.data["input_hash"] == "input_hash"
    assert returned.data["output_hash"] == "output_hash"
    assert returned.data["order_count"] == 5
    assert returned.data["diagnostic_keys"] == ["forager", "min_cost"]
    assert failed.event_type == EventTypes.RUST_ORCHESTRATOR_RETURNED
    assert failed.level == "error"
    assert failed.status == "failed"
    assert failed.reason_code == "ValueError"
    assert failed.raw_hash == "failed_input_hash"
    assert failed.data["error_type"] == "ValueError"
    assert "MissingEma" in failed.data["error"]
    assert "SECRET" not in failed.data["error"]
    assert "[redacted]" in failed.data["error"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_rust_orchestrator_debug_profile_samples_are_bounded():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_rust_orchestrator_called_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_called_event
        )
        _emit_rust_orchestrator_returned_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_returned_event
        )

        def __init__(self):
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_rust"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    idx_to_symbol = {0: "BTC/USDT:USDT", 1: "ETH/USDT:USDT"}
    input_sample = live_event_emitters.rust_input_symbol_debug_sample(
        [
            {
                "symbol_idx": 0,
                "tradable": True,
                "order_book": {"bid": 100.0, "ask": 101.0},
                "effective_min_cost": 5.0,
                "emas": {
                    "m1": {"close": [[100.0, 1.0]], "log_range": [], "volume": []},
                    "h1": {"log_range": [[100.0, 0.01]]},
                },
                "long": {"position": {"size": 0.01}},
                "short": {"position": {"size": 0.0}},
            },
            {"symbol_idx": 1, "tradable": False, "emas": {}},
        ],
        idx_to_symbol=idx_to_symbol,
        limit=1,
    )
    output_sample = live_event_emitters.rust_output_order_debug_sample(
        [
            {
                "symbol_idx": 0,
                "order_type": "entry_grid_normal_long",
                "execution_type": "limit",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "price": 100.0,
                "reduce_only": False,
            },
            {"symbol_idx": 1, "order_type": "unstuck_close_short"},
        ],
        idx_to_symbol=idx_to_symbol,
        limit=1,
    )

    bot = FakeBot()
    bot._emit_rust_orchestrator_called_event(
        rust_call_id="rust_7",
        input_hash="input_hash",
        symbol_count=2,
        tradable_count=1,
        ema_unavailable_count=0,
        trailing_unavailable_count=0,
        hedge_mode=True,
        strategy_kind="recursive_grid",
        input_symbol_sample=input_sample,
    )
    bot._emit_rust_orchestrator_returned_event(
        rust_call_id="rust_7",
        status="succeeded",
        input_hash="input_hash",
        output_hash="output_hash",
        elapsed_ms=12,
        order_count=2,
        diagnostics={},
        output_order_sample=output_sample,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    called, returned = sink.events
    assert called.data["debug_profile"] == "rust"
    assert called.data["input_symbol_sample"]["count"] == 2
    assert called.data["input_symbol_sample"]["truncated"] == 1
    assert called.data["input_symbol_sample"]["sample"] == [
        {
            "symbol_idx": 0,
            "symbol": "BTC/USDT:USDT",
            "tradable": True,
            "has_bid": True,
            "has_ask": True,
            "effective_min_cost": 5.0,
            "m1_close_ema_count": 1,
            "m1_log_range_ema_count": 0,
            "m1_volume_ema_count": 0,
            "h1_log_range_ema_count": 1,
            "active_psides": ["long"],
        }
    ]
    assert returned.data["debug_profile"] == "rust"
    assert returned.data["output_order_sample"]["count"] == 2
    assert returned.data["output_order_sample"]["truncated"] == 1
    assert (
        returned.data["output_order_sample"]["sample"][0]["symbol"]
        == "BTC/USDT:USDT"
    )
    assert returned.data["output_order_sample"]["sample"][0]["order_type"] == (
        "entry_grid_normal_long"
    )
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_rust_orchestrator_emitters_are_best_effort(caplog):
    import passivbot as pb_mod

    class FailingBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_rust_orchestrator_called_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_called_event
        )
        _emit_rust_orchestrator_returned_event = (
            pb_mod.Passivbot._emit_rust_orchestrator_returned_event
        )

        def __init__(self):
            self._live_event_current_cycle_id = "cy_rust"

        def _emit_live_event(self, *args, **kwargs):
            raise RuntimeError("event sink failed")

    bot = FailingBot()

    with caplog.at_level(logging.DEBUG):
        bot._emit_rust_orchestrator_called_event(
            rust_call_id="rust_1",
            input_hash="input_hash",
            symbol_count=1,
            tradable_count=1,
            ema_unavailable_count=0,
            trailing_unavailable_count=0,
            hedge_mode=False,
            strategy_kind="recursive_grid",
        )
        bot._emit_rust_orchestrator_returned_event(
            rust_call_id="rust_1",
            status="succeeded",
            input_hash="input_hash",
            output_hash="output_hash",
            elapsed_ms=1,
            order_count=0,
            diagnostics={},
        )
        bot._emit_rust_orchestrator_returned_event(
            rust_call_id="rust_2",
            status="failed",
            input_hash="failed_input_hash",
            elapsed_ms=1,
            error=RuntimeError("rust failed"),
        )

    messages = [record.message for record in caplog.records]
    assert any(EventTypes.RUST_ORCHESTRATOR_CALLED in msg for msg in messages)
    assert sum(EventTypes.RUST_ORCHESTRATOR_RETURNED in msg for msg in messages) == 2


def test_health_summary_event_emitter_records_payload():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_health_summary_event = pb_mod.Passivbot._emit_health_summary_event

        def __init__(self):
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_health"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._emit_health_summary_event(
        {
            "uptime_ms": 60000,
            "last_loop_duration_ms": 250,
            "positions_long": 1,
            "positions_short": 0,
            "rss_bytes": 123456,
        }
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.HEALTH_SUMMARY
    assert event.level == "debug"
    assert event.component == "bot.health"
    assert event.tags == ("health", "resource")
    assert event.cycle_id == "cy_health"
    assert event.status == "succeeded"
    assert event.reason_code == "periodic_health_summary"
    assert event.data["rss_bytes"] == 123456
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_state_refresh_timing_event_emitter_records_payload():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_state_refresh_timing_event = (
            pb_mod.Passivbot._emit_state_refresh_timing_event
        )

        def __init__(self):
            self._live_event_current_cycle_id = "cy_state"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._emit_state_refresh_timing_event(
        plan={"open_orders", "balance"},
        timings_ms={"balance": 25, "open_orders": 40},
        wall_ms=45,
        sum_ms=65,
        max_surface_ms=40,
        residual_ms=5,
        pending_confirmations=True,
        meaningful_change=True,
        unusual_plan=False,
        epoch_changed={"open_orders"},
        level="info",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.STATE_REFRESH_TIMING
    assert event.level == "info"
    assert event.component == "state.refresh"
    assert event.tags == ("state", "refresh", "account")
    assert event.cycle_id == "cy_state"
    assert event.status == "succeeded"
    assert event.reason_code == ReasonCodes.STAGED_REFRESH_TIMING
    assert event.data["plan"] == ["balance", "open_orders"]
    assert event.data["timings_ms"] == {"balance": 25, "open_orders": 40}
    assert event.data["wall_ms"] == 45
    assert event.data["surface_sum_ms"] == 65
    assert event.data["surface_max_ms"] == 40
    assert event.data["residual_ms"] == 5
    assert event.data["pending_confirmations"] is True
    assert event.data["meaningful_change"] is True
    assert event.data["unusual_plan"] is False
    assert event.data["epoch_changed"] == ["open_orders"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_state_refresh_progress_event_emitter_records_payload():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_state_refresh_progress_event = (
            pb_mod.Passivbot._emit_state_refresh_progress_event
        )

        def __init__(self):
            self._live_event_current_cycle_id = "cy_progress"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._emit_state_refresh_progress_event(
        plan={"balance", "positions"},
        pending=("positions",),
        elapsed_ms=12_500,
        completed_timings_ms={"balance": 120},
        threshold_s=10.0,
        repeated=True,
        level="debug",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.STATE_REFRESH_PROGRESS
    assert event.level == "debug"
    assert event.component == "state.refresh"
    assert event.tags == ("state", "refresh", "timeout")
    assert event.cycle_id == "cy_progress"
    assert event.status == "degraded"
    assert event.reason_code == ReasonCodes.STAGED_REFRESH_PROGRESS
    assert event.data["plan"] == ["balance", "positions"]
    assert event.data["pending"] == ["positions"]
    assert event.data["elapsed_ms"] == 12_500
    assert event.data["completed_timings_ms"] == {"balance": 120}
    assert event.data["threshold_s"] == 10.0
    assert event.data["repeated"] is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_build_health_summary_payload_includes_resource_pressure(monkeypatch):
    import passivbot as pb_mod

    class FakePipeline:
        def health_snapshot(self):
            return {
                "event_queue_depth": 3,
                "event_queue_maxsize": 100,
                "event_dropped_total": 2,
                "event_drop_counts": {"remote_call.started": 2},
                "event_sink_error_total": 1,
                "event_sink_error_counts": {"monitor": 1},
                "event_degraded_count": 4,
                "event_pipeline_worker_alive": True,
            }

    class FakeBot:
        _build_health_summary_payload = pb_mod.Passivbot._build_health_summary_payload

        def __init__(self):
            self.positions = {
                "BTC/USDT:USDT": {
                    "long": {"size": 0.01},
                    "short": {"size": 0.0},
                },
                "ETH/USDT:USDT": {
                    "long": {"size": 0.0},
                    "short": {"size": -0.02},
                },
            }
            self._health_start_ms = 100_000
            self._last_loop_duration_ms = 1234
            self._monitor_last_equity = 1001.0
            self._health_orders_placed = 5
            self._health_orders_cancelled = 2
            self._health_fills = 7
            self._health_pnl = 8.5
            self._health_ws_reconnects = 1
            self._health_rate_limits = 2
            self.error_counts = [159_000, 10_000]
            self._live_event_pipeline = FakePipeline()

        def get_raw_balance(self):
            return 1000.0

        def get_hysteresis_snapped_balance(self):
            return 999.0

    monkeypatch.setattr(pb_mod.pb_monitor, "_get_process_rss_bytes", lambda: 123456)
    monkeypatch.setattr(pb_mod.pb_monitor, "_get_process_memory_percent", lambda: 12.5)
    monkeypatch.setattr(pb_mod.pb_monitor, "_get_open_fd_count", lambda: 42)
    monkeypatch.setattr(
        pb_mod.pb_monitor,
        "_get_loadavg_payload",
        lambda: {
            "loadavg_1m": 1.5,
            "loadavg_5m": 1.25,
            "loadavg_15m": 1.0,
            "cpu_count": 1,
        },
    )

    payload = FakeBot()._build_health_summary_payload(now_ms=160_000)

    assert payload["uptime_ms"] == 60_000
    assert payload["positions_long"] == 1
    assert payload["positions_short"] == 1
    assert payload["rss_bytes"] == 123456
    assert payload["memory_percent"] == 12.5
    assert payload["open_fds"] == 42
    assert payload["loadavg_1m"] == 1.5
    assert payload["loadavg_5m"] == 1.25
    assert payload["loadavg_15m"] == 1.0
    assert payload["cpu_count"] == 1
    assert payload["event_queue_depth"] == 3
    assert payload["event_queue_maxsize"] == 100
    assert payload["event_dropped_total"] == 2
    assert payload["event_drop_counts"] == {"remote_call.started": 2}
    assert payload["event_sink_error_total"] == 1
    assert payload["event_sink_error_counts"] == {"monitor": 1}
    assert payload["event_degraded_count"] == 4
    assert payload["event_pipeline_worker_alive"] is True


def test_log_health_summary_emits_structured_event(caplog, monkeypatch):
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_health_summary_event = pb_mod.Passivbot._emit_health_summary_event
        _format_duration = pb_mod.Passivbot._format_duration
        _log_health_summary = pb_mod.Passivbot._log_health_summary

        def __init__(self):
            self.exchange = "okx"
            self.user = "okx_01"
            self.bot_id = "bot_1"
            self.quote = "USDT"
            self.positions = {
                "BTC/USDT:USDT": {
                    "long": {"size": 0.01},
                    "short": {"size": 0.0},
                }
            }
            self._health_start_ms = 140000
            self._last_loop_duration_ms = 2500
            self._last_loop_timing_ms = {}
            self._health_orders_placed = 2
            self._health_orders_cancelled = 1
            self._health_fills = 3
            self._health_pnl = 1.25
            self._health_ws_reconnects = 4
            self._health_rate_limits = 5
            self.error_counts = [190000]
            self.candle_health_called = False
            self.payload_now_ms = None
            self._live_event_current_cycle_id = "cy_health"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

        def get_raw_balance(self):
            return 1000.0

        def get_hysteresis_snapped_balance(self):
            return 999.5

        def _build_health_summary_payload(self, *, now_ms=None):
            self.payload_now_ms = now_ms
            return {
                "uptime_ms": 60000,
                "last_loop_duration_ms": 2500,
                "positions_long": 1,
                "positions_short": 0,
                "balance_raw": 1000.0,
                "balance_snapped": 999.5,
                "orders_placed": 2,
                "orders_cancelled": 1,
                "fills": 3,
                "pnl": 1.25,
                "errors_last_hour": 1,
                "ws_reconnects": 4,
                "rate_limits": 5,
                "rss_bytes": 987654,
            }

        def _maybe_log_candle_health_summary(self):
            self.candle_health_called = True

    monkeypatch.setattr(pb_mod, "utc_ms", lambda: 200000)
    bot = FakeBot()

    with caplog.at_level(logging.INFO):
        bot._log_health_summary()

    assert "[health] uptime=1m0s" in caplog.text
    assert "positions=1 long, 0 short" in caplog.text
    assert "orders=+2/-1" in caplog.text
    assert bot.candle_health_called is True
    assert bot.payload_now_ms == 200000
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.HEALTH_SUMMARY
    assert event.cycle_id == "cy_health"
    assert event.data["rss_bytes"] == 987654
    assert event.data["errors_last_hour"] == 1
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_forager_and_ema_summary_emitters_emit_structured_events():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_ema_bundle_started_event = (
            pb_mod.Passivbot._emit_ema_bundle_started_event
        )
        _emit_ema_bundle_completed_event = (
            pb_mod.Passivbot._emit_ema_bundle_completed_event
        )
        _emit_ema_fallback_used_event = pb_mod.Passivbot._emit_ema_fallback_used_event
        _emit_ema_unavailable_event = pb_mod.Passivbot._emit_ema_unavailable_event
        _emit_candle_tail_projected_event = (
            pb_mod.Passivbot._emit_candle_tail_projected_event
        )
        _emit_candle_coverage_checked_event = (
            pb_mod.Passivbot._emit_candle_coverage_checked_event
        )
        _emit_cache_load_completed_event = (
            pb_mod.Passivbot._emit_cache_load_completed_event
        )
        _emit_cache_flush_completed_event = (
            pb_mod.Passivbot._emit_cache_flush_completed_event
        )
        _emit_cache_warmup_decision_event = (
            pb_mod.Passivbot._emit_cache_warmup_decision_event
        )
        _emit_forager_feature_unavailable_event = (
            pb_mod.Passivbot._emit_forager_feature_unavailable_event
        )
        _emit_forager_selection_event = pb_mod.Passivbot._emit_forager_selection_event
        _emit_live_event = pb_mod.Passivbot._emit_live_event

        def __init__(self):
            self.exchange = "gateio"
            self.user = "gateio_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_11"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._emit_forager_feature_unavailable_event(
        pside="long",
        symbols=["ETH/USDT:USDT", "BTC/USDT:USDT"],
        candidate_count=4,
        volume_count=3,
        log_range_count=2,
        max_age_ms=300_000,
        fetch_budget=1,
    )
    bot._emit_forager_selection_event(
        pside="long",
        candidate_count=4,
        eligible_count=2,
        selected_symbols=["BTC/USDT:USDT"],
        slots_open=True,
        max_n_positions=3,
        clip_pct=0.1,
        volatility_drop_pct=0.25,
        max_age_ms=300_000,
        fetch_budget=1,
        feature_unavailable_count=2,
        volatility_dropped_count=1,
    )
    bot._emit_ema_bundle_started_event(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        modes={
            "BTC/USDT:USDT": {"long": "normal", "short": ""},
            "ETH/USDT:USDT": {"long": "forager", "short": ""},
        },
    )
    bot._emit_ema_bundle_completed_event(
        symbols=["BTC/USDT:USDT", "ETH/USDT:USDT"],
        m1_close_emas={"BTC/USDT:USDT": {100.0: 50_000.0}},
        m1_volume_emas={"BTC/USDT:USDT": {60.0: 123.0}},
        m1_log_range_emas={"BTC/USDT:USDT": {60.0: 0.01}},
        h1_log_range_emas={},
        cache_only_symbols={"ETH/USDT:USDT"},
        projection_contexts={"ETH/USDT:USDT": {"tail_gap_age_ms": 120_000}},
    )
    bot._emit_ema_fallback_used_event(
        close_ema_recoveries={"BTC/USDT:USDT": [(100.0, 1)]},
        close_ema_fallbacks={"ETH/USDT:USDT": [(100.0, 120_000, 2, "stale")]},
        forager_cached_ema_fallbacks={
            "DOGE/USDT:USDT": [("qv", 60.0, 180_000)]
        },
    )
    bot._emit_ema_unavailable_event(
        optional_ema_drops={
            ("h1_log_range", "missing_optional"): [("SOL/USDT:USDT", 24.0)]
        },
        candidate_ema_unavailable_details={
            "required_missing": [("XRP/USDT:USDT", "ValueError", "missing")]
        },
        ema_unavailable_reasons={"cache_only_never_fetched": ["ADA/USDT:USDT"]},
    )
    bot._emit_candle_tail_projected_event(
        symbol="ETH/USDT:USDT",
        context={
            "timeframe": "1m",
            "latest_expected_ts": 180_000,
            "last_cached_ts": 120_000,
            "tail_gap_age_ms": 60_000,
            "tail_gap_candles": 1,
            "missing_candles": 1,
            "reason": "open_tail_gap_projection",
        },
        reason_code="late_open_tail_projection",
    )
    bot._emit_candle_coverage_checked_event(
        symbol="ETH/USDT:USDT",
        timeframe="1m",
        start_ts=120_000,
        end_ts=240_000,
        report={
            "ok": False,
            "timeframe": "1m",
            "missing_spans": [(180_000, 240_000)],
            "missing_candles": 2,
            "loaded_rows": 1,
        },
        context="required_disk_audit",
        required=True,
    )
    bot._emit_cache_load_completed_event(
        {
            "symbol": "ETH/USDT:USDT",
            "timeframe": "1m",
            "start_ts": 120_000,
            "end_ts": 240_000,
            "loaded_rows": 3,
            "loaded_start_ts": 120_000,
            "loaded_end_ts": 240_000,
            "days": 1,
            "primary_days": 1,
            "legacy_days": 0,
            "merged_days": 0,
            "source_days": {"primary": 1, "legacy": 0, "merged": 0},
            "elapsed_ms": 7,
            "suppressed_count": 2,
        }
    )
    bot._emit_cache_flush_completed_event(
        {
            "symbol": "ETH/USDT:USDT",
            "timeframe": "1m",
            "persisted_rows": 4,
            "persisted_start_ts": 300_000,
            "persisted_end_ts": 480_000,
            "suppressed_count": 2,
            "suppressed_rows": 8,
        }
    )
    bot._emit_cache_warmup_decision_event(
        context="trading-ready warmup",
        timeframe="1m",
        symbol_count=3,
        reused_count=1,
        cold_count=2,
        reason_counts={"missing_coverage": 2, "warm_cache_accepted": 1},
        elapsed_ms=1234,
        concurrency=4,
        ttl_ms=300_000,
        window_min_candles=120,
        window_max_candles=260,
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    events = sink.events
    assert [event.event_type for event in events] == [
        EventTypes.FORAGER_FEATURE_UNAVAILABLE,
        EventTypes.FORAGER_SELECTION,
        EventTypes.EMA_BUNDLE_STARTED,
        EventTypes.EMA_BUNDLE_COMPLETED,
        EventTypes.EMA_FALLBACK_USED,
        EventTypes.EMA_UNAVAILABLE,
        EventTypes.CANDLE_TAIL_PROJECTED,
        EventTypes.CANDLE_COVERAGE_CHECKED,
        EventTypes.CACHE_LOAD_COMPLETED,
        EventTypes.CACHE_FLUSH_COMPLETED,
        EventTypes.CACHE_WARMUP_DECISION,
    ]
    assert {event.cycle_id for event in events} == {"cy_11"}
    assert events[0].data["unavailable"]["count"] == 2
    assert events[1].data["selected_symbols"] == ["BTC/USDT:USDT"]
    assert events[2].status == "started"
    assert events[2].data["symbol_count"] == 2
    assert events[2].data["symbols"]["sample"] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]
    assert events[2].data["mode_counts"] == {"forager": 1, "normal": 1}
    assert events[3].data["cache_only"]["sample"] == ["ETH/USDT:USDT"]
    assert events[4].level == "warning"
    assert events[4].data["close_fallback_count"] == 1
    assert events[5].status == "degraded"
    assert events[5].data["candidate_unavailable"]["sample"] == ["XRP/USDT:USDT"]
    assert "debug" not in events[5].data
    assert events[6].symbol == "ETH/USDT:USDT"
    assert events[6].status == "recovered"
    assert events[6].reason_code == "late_open_tail_projection"
    assert events[6].data["latest_expected_ts"] == 180_000
    assert events[6].data["tail_gap_age_ms"] == 60_000
    assert events[7].component == "candle.coverage"
    assert events[7].symbol == "ETH/USDT:USDT"
    assert events[7].status == "degraded"
    assert events[7].level == "warning"
    assert events[7].reason_code == "required_candle_disk_coverage"
    assert events[7].data["coverage_ok"] is False
    assert events[7].data["missing_span_count"] == 1
    assert events[7].data["missing_spans_preview"] == [
        {"start_ts": 180_000, "end_ts": 240_000, "candles": 2}
    ]
    assert events[8].component == "cache.candles"
    assert events[8].symbol == "ETH/USDT:USDT"
    assert events[8].reason_code == "candle_disk_load_completed"
    assert events[8].data == {
        "timeframe": "1m",
        "start_ts": 120_000,
        "end_ts": 240_000,
        "loaded_rows": 3,
        "loaded_start_ts": 120_000,
        "loaded_end_ts": 240_000,
        "days": 1,
        "primary_days": 1,
        "legacy_days": 0,
        "merged_days": 0,
        "elapsed_ms": 7,
        "suppressed_count": 2,
        "source_days": {"legacy": 0, "merged": 0, "primary": 1},
    }
    assert events[9].component == "cache.candles"
    assert events[9].symbol == "ETH/USDT:USDT"
    assert events[9].reason_code == "candle_disk_flush_completed"
    assert events[9].data == {
        "timeframe": "1m",
        "persisted_rows": 4,
        "persisted_start_ts": 300_000,
        "persisted_end_ts": 480_000,
        "suppressed_count": 2,
        "suppressed_rows": 8,
    }
    assert events[10].component == "cache.warmup"
    assert events[10].reason_code == "warmup_cache_decision"
    assert events[10].data["context"] == "trading-ready warmup"
    assert events[10].data["symbol_count"] == 3
    assert events[10].data["reused_count"] == 1
    assert events[10].data["cold_count"] == 2
    assert events[10].data["cold_path_required"] is True
    assert events[10].data["reason_counts"] == {
        "missing_coverage": 2,
        "warm_cache_accepted": 1,
    }
    assert events[10].data["window_max_candles"] == 260
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_ema_unavailable_event_debug_profile_adds_parsed_readiness_detail():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_ema_unavailable_event = pb_mod.Passivbot._emit_ema_unavailable_event
        _emit_live_event = pb_mod.Passivbot._emit_live_event

        def __init__(self):
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self.live_event_debug_profiles = ("ema",)
            self._live_event_current_cycle_id = "cy_12"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._emit_ema_unavailable_event(
        candidate_ema_unavailable_details={
            "cache_only_fetch_failed": [
                (
                    "BTC/USDT:USDT",
                    "RuntimeError",
                    "[ema] missing required h1_log_range EMA for BTC/USDT:USDT: "
                    "span=672 reason=non-finite h1_log_range value nan; "
                    "span=1100 reason=non-finite h1_log_range value nan",
                ),
                (
                    "ETH/USDT:USDT",
                    "RuntimeError",
                    "[ema] missing required m1_log_range EMA for ETH/USDT:USDT: "
                    "span=500 reason=missing required window",
                ),
            ]
        },
        ema_unavailable_reasons={
            "never_fetched_cache_only": ["XRP/USDT:USDT", "ZEC/USDT:USDT"]
        },
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.EMA_UNAVAILABLE
    assert event.data["debug_profile"] == "ema"
    debug = event.data["debug"]
    assert debug["unavailable_groups"][0]["reason"] == "never_fetched_cache_only"
    group = debug["candidate_groups"][0]
    assert group["reason"] == "cache_only_fetch_failed"
    assert group["symbols"]["sample"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert group["ema_types"] == [
        {"ema_type": "h1_log_range", "count": 1},
        {"ema_type": "m1_log_range", "count": 1},
    ]
    assert group["spans"] == [500.0, 672.0, 1100.0]
    assert group["inner_reasons"] == [
        {"reason": "non-finite h1_log_range value nan", "count": 2},
        {"reason": "missing required window", "count": 1},
    ]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_candle_disk_coverage_audit_emits_structured_events(monkeypatch):
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeCandleManager:
        def __init__(self):
            self.calls = []

        def check_disk_coverage(
            self,
            symbol,
            start_ts,
            end_ts,
            *,
            timeframe,
            log_level,
        ):
            self.calls.append(
                {
                    "symbol": symbol,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "timeframe": timeframe,
                    "log_level": log_level,
                }
            )
            if timeframe == "1m":
                return {
                    "ok": False,
                    "timeframe": "1m",
                    "missing_spans": [(180_000, 240_000)],
                    "missing_candles": 2,
                    "loaded_rows": 1,
                }
            return {
                "ok": True,
                "timeframe": "1h",
                "missing_spans": [],
                "missing_candles": 0,
                "loaded_rows": 4,
            }

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.exchange = "okx"
    bot.user = "okx_01"
    bot.bot_id = "bot_1"
    bot.cm = FakeCandleManager()
    bot.config = {"live": {}}
    bot.active_symbols = ["BTC/USDT:USDT"]
    bot.open_orders = {}
    bot.positions = {}
    bot._live_event_current_cycle_id = "cy_12"
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    bot.get_max_n_positions = lambda pside: 1
    bot.get_current_n_positions = lambda pside: 1 if pside == "long" else 0
    bot.get_symbols_with_pos = (
        lambda pside: {"BTC/USDT:USDT"} if pside == "long" else set()
    )
    bot.get_symbols_approved_or_has_pos = lambda pside: {"BTC/USDT:USDT"}
    bot.is_forager_mode = lambda pside=None: False
    bot.has_position = lambda symbol: symbol == "BTC/USDT:USDT"

    monkeypatch.setattr(pb_mod, "utc_ms", lambda: 7_500_000)
    monkeypatch.setattr(
        pb_mod,
        "compute_live_warmup_windows",
        lambda *args, **kwargs: (
            {"BTC/USDT:USDT": 2},
            {"BTC/USDT:USDT": 1},
            {},
        ),
    )

    await bot.audit_required_candle_disk_coverage()

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    assert bot.cm.calls == [
        {
            "symbol": "BTC/USDT:USDT",
            "start_ts": 7_320_000,
            "end_ts": 7_440_000,
            "timeframe": "1m",
            "log_level": "debug",
        },
        {
            "symbol": "BTC/USDT:USDT",
            "start_ts": 0,
            "end_ts": 3_600_000,
            "timeframe": "1h",
            "log_level": "debug",
        },
    ]
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.CANDLE_COVERAGE_CHECKED
    ]
    assert [event.status for event in events] == ["degraded", "succeeded"]
    assert [event.level for event in events] == ["warning", "debug"]
    assert {event.cycle_id for event in events} == {"cy_12"}
    assert events[0].symbol == "BTC/USDT:USDT"
    assert events[0].data["timeframe"] == "1m"
    assert events[0].data["coverage_ok"] is False
    assert events[0].data["missing_candles"] == 2
    assert events[0].data["missing_spans_preview"] == [
        {"start_ts": 180_000, "end_ts": 240_000, "candles": 2}
    ]
    assert events[1].data["timeframe"] == "1h"
    assert events[1].data["coverage_ok"] is True


def test_candle_disk_load_handler_throttles_repeated_symbol_timeframe_events():
    import passivbot as pb_mod

    class FakeBot:
        _handle_candle_disk_load_event = (
            pb_mod.Passivbot._handle_candle_disk_load_event
        )

        def __init__(self):
            self._cache_load_event_throttle_seconds = 60.0
            self.emitted = []

        def _emit_cache_load_completed_event(self, payload):
            self.emitted.append(dict(payload))

    bot = FakeBot()
    payload = {
        "symbol": "ETH/USDT:USDT",
        "timeframe": "1m",
        "start_ts": 120_000,
        "end_ts": 240_000,
        "loaded_rows": 3,
    }

    bot._handle_candle_disk_load_event(payload)
    bot._handle_candle_disk_load_event({**payload, "start_ts": 180_000})

    assert len(bot.emitted) == 1
    assert "suppressed_count" not in bot.emitted[0]
    key = ("ETH/USDT:USDT", "1m")
    assert bot._cache_load_event_suppressed[key] == 1

    bot._cache_load_event_last_emit[key] -= 61.0
    bot._handle_candle_disk_load_event({**payload, "start_ts": 240_000})

    assert len(bot.emitted) == 2
    assert bot.emitted[1]["start_ts"] == 240_000
    assert bot.emitted[1]["suppressed_count"] == 1
    assert key not in bot._cache_load_event_suppressed

    bot._handle_candle_disk_load_event({**payload, "timeframe": "1h"})

    assert len(bot.emitted) == 3
    assert bot.emitted[2]["timeframe"] == "1h"


def test_candle_cache_flush_handler_throttles_repeated_symbol_timeframe_events():
    import passivbot as pb_mod

    class FakeBot:
        _handle_candle_cache_flush_event = (
            pb_mod.Passivbot._handle_candle_cache_flush_event
        )

        def __init__(self):
            self._cache_flush_event_throttle_seconds = 60.0
            self.emitted = []

        def _emit_cache_flush_completed_event(self, payload):
            self.emitted.append(dict(payload))

    bot = FakeBot()
    batch = np.array(
        [
            (60_000, 1.0, 2.0, 0.5, 1.5, 10.0),
            (120_000, 1.5, 2.5, 1.0, 2.0, 11.0),
        ],
        dtype=pb_mod.CANDLE_DTYPE,
    )

    bot._handle_candle_cache_flush_event("BTC/USDT:USDT", "1m", batch)
    bot._handle_candle_cache_flush_event("BTC/USDT:USDT", "1m", batch)

    assert len(bot.emitted) == 1
    assert bot.emitted[0] == {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1m",
        "persisted_rows": 2,
        "persisted_start_ts": 60_000,
        "persisted_end_ts": 120_000,
    }
    key = ("BTC/USDT:USDT", "1m")
    assert bot._cache_flush_event_suppressed[key] == {"count": 1, "rows": 2}

    bot._cache_flush_event_last_emit[key] -= 61.0
    bot._handle_candle_cache_flush_event("BTC/USDT:USDT", "1m", batch)

    assert len(bot.emitted) == 2
    assert bot.emitted[1]["suppressed_count"] == 1
    assert bot.emitted[1]["suppressed_rows"] == 2
    assert key not in bot._cache_flush_event_suppressed

    bot._handle_candle_cache_flush_event("BTC/USDT:USDT", "1h", batch)

    assert len(bot.emitted) == 3
    assert bot.emitted[2]["timeframe"] == "1h"


def test_candle_persist_handler_preserves_monitor_and_emits_flush_summary():
    import passivbot as pb_mod

    class FakeBot:
        _monitor_handle_candlestick_persist = (
            pb_mod.Passivbot._monitor_handle_candlestick_persist
        )
        _handle_candle_cache_flush_event = (
            pb_mod.Passivbot._handle_candle_cache_flush_event
        )
        _handle_candle_persist_event = pb_mod.Passivbot._handle_candle_persist_event

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._bot_ready = True
            self._cache_flush_event_throttle_seconds = 60.0
            self.flush_events = []

        def _emit_cache_flush_completed_event(self, payload):
            self.flush_events.append(dict(payload))

    bot = FakeBot()
    batch = np.array(
        [
            (60_000, 1.0, 2.0, 0.5, 1.5, 10.0),
            (120_000, 1.5, 2.5, 1.0, 2.0, 11.0),
        ],
        dtype=pb_mod.CANDLE_DTYPE,
    )

    bot._handle_candle_persist_event("BTC/USDT:USDT", "1m", batch)

    assert len(bot.monitor_publisher.completed_candles) == 1
    assert bot.monitor_publisher.completed_candles[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.flush_events == [
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1m",
            "persisted_rows": 2,
            "persisted_start_ts": 60_000,
            "persisted_end_ts": 120_000,
        }
    ]


def test_forager_and_ema_summary_emitters_are_best_effort_on_malformed_inputs(caplog):
    import passivbot as pb_mod

    bot = object()

    with caplog.at_level(logging.DEBUG):
        pb_mod.Passivbot._emit_forager_feature_unavailable_event(
            bot,
            pside="long",
            symbols=["BTC/USDT:USDT"],
            candidate_count="not-an-int",
            volume_count=1,
            log_range_count=1,
            max_age_ms=60_000,
            fetch_budget=1,
        )
        pb_mod.Passivbot._emit_forager_selection_event(
            bot,
            pside="long",
            candidate_count="not-an-int",
            eligible_count=1,
            selected_symbols=["BTC/USDT:USDT"],
            slots_open=True,
            max_n_positions=1,
            clip_pct=0.0,
            volatility_drop_pct=0.0,
            max_age_ms=60_000,
            fetch_budget=1,
        )
        pb_mod.Passivbot._emit_action_planned_event(
            bot,
            orders=[{"symbol_idx": 0, "order_type": "entry_grid_normal_long"}],
            idx_to_symbol=object(),
        )
        pb_mod.Passivbot._emit_ema_bundle_started_event(
            bot,
            symbols=object(),
            modes=object(),
        )
        pb_mod.Passivbot._emit_ema_bundle_completed_event(
            bot,
            symbols=["BTC/USDT:USDT"],
            m1_close_emas={"BTC/USDT:USDT": object()},
            m1_volume_emas={},
            m1_log_range_emas={},
            h1_log_range_emas={},
        )
        pb_mod.Passivbot._emit_ema_fallback_used_event(
            bot,
            close_ema_fallbacks={"BTC/USDT:USDT": object()},
        )
        pb_mod.Passivbot._emit_ema_unavailable_event(
            bot,
            candidate_ema_unavailable_details={
                "required_missing": [("BTC/USDT:USDT", "ValueError")]
            },
        )
        pb_mod.Passivbot._emit_candle_tail_projected_event(
            bot,
            symbol=object(),
            context=object(),
        )
        pb_mod.Passivbot._emit_cache_load_completed_event(
            bot,
            payload=object(),
        )
        pb_mod.Passivbot._emit_cache_flush_completed_event(
            bot,
            payload=object(),
        )
        pb_mod.Passivbot._emit_cache_warmup_decision_event(
            bot,
            context=object(),
            symbol_count=object(),
            reused_count=object(),
            cold_count=object(),
            reason_counts=object(),
        )
        pb_mod.Passivbot._emit_startup_timing_event(
            bot,
            phase=object(),
            elapsed_ms=object(),
            since_previous_ms=object(),
        )

    messages = [record.message for record in caplog.records]
    assert any(EventTypes.FORAGER_FEATURE_UNAVAILABLE in msg for msg in messages)
    assert any(EventTypes.FORAGER_SELECTION in msg for msg in messages)
    assert any(EventTypes.ACTION_PLANNED in msg for msg in messages)
    assert any(EventTypes.EMA_BUNDLE_STARTED in msg for msg in messages)
    assert any(EventTypes.EMA_BUNDLE_COMPLETED in msg for msg in messages)
    assert any(EventTypes.EMA_FALLBACK_USED in msg for msg in messages)
    assert any(EventTypes.EMA_UNAVAILABLE in msg for msg in messages)
    assert any(EventTypes.CANDLE_TAIL_PROJECTED in msg for msg in messages)
    assert any(EventTypes.CACHE_LOAD_COMPLETED in msg for msg in messages)
    assert any(EventTypes.CACHE_FLUSH_COMPLETED in msg for msg in messages)
    assert any(EventTypes.CACHE_WARMUP_DECISION in msg for msg in messages)
    assert any(EventTypes.BOT_STARTUP_TIMING in msg for msg in messages)


def _make_remote_fetch_event_bot(sink, *, cycle_id="cy_7", map_max=None):
    import passivbot as pb_mod

    class FakeBot:
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _handle_candle_remote_fetch_event = (
            pb_mod.Passivbot._handle_candle_remote_fetch_event
        )

        def __init__(self):
            self.exchange = "okx"
            self.user = "okx_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = cycle_id
            self._live_event_remote_call_seq = 0
            self._live_event_remote_call_ids = {}
            if map_max is not None:
                self._live_event_remote_call_map_max = map_max
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    return FakeBot()


def test_candle_remote_fetch_callback_emits_correlated_remote_call_events():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink)
    base = {
        "kind": "ccxt_fetch_ohlcv",
        "exchange": "okx",
        "symbol": "BTC/USDT:USDT",
        "tf": "1m",
        "since_ts": 123000,
    }

    bot._handle_candle_remote_fetch_event(
        {**base, "stage": "start", "limit": 100, "params": {"apiKey": "secret"}}
    )
    bot._handle_candle_remote_fetch_event(
        {
            **base,
            "stage": "ok",
            "rows": 100,
            "first_ts": 123000,
            "last_ts": 183000,
            "elapsed_ms": 42,
        }
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    assert [event.event_type for event in sink.events] == [
        EventTypes.REMOTE_CALL_STARTED,
        EventTypes.REMOTE_CALL_SUCCEEDED,
    ]
    started, succeeded = sink.events
    assert started.remote_call_id == succeeded.remote_call_id
    assert started.remote_call_group_id == "cy_7:candles"
    assert succeeded.remote_call_group_id == "cy_7:candles"
    assert started.cycle_id == "cy_7"
    assert succeeded.cycle_id == "cy_7"
    assert started.symbol == "BTC/USDT:USDT"
    assert succeeded.data["rows"] == 100
    assert started.data["params"]["apiKey"] == "[redacted]"
    assert bot._live_event_remote_call_seq == 1


def test_candle_remote_fetch_error_sanitizes_and_keeps_correlation():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink)
    base = {
        "kind": "ccxt_fetch_ohlcv",
        "exchange": "okx",
        "symbol": "BTC/USDT:USDT",
        "tf": "1m",
        "since_ts": 123000,
    }

    bot._handle_candle_remote_fetch_event({**base, "stage": "start"})
    bot._handle_candle_remote_fetch_event(
        {
            **base,
            "stage": "error",
            "error_type": "AuthError",
            "error": "token SECRET apiKey=abc Bearer xyz",
            "error_repr": "Auth(apiKey=SECRET, token=SECRET) " + ("x" * 700),
        }
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    started, failed = sink.events
    assert failed.event_type == EventTypes.REMOTE_CALL_FAILED
    assert failed.remote_call_id == started.remote_call_id
    assert failed.remote_call_group_id == started.remote_call_group_id
    assert "SECRET" not in failed.data["error"]
    assert "abc" not in failed.data["error"]
    assert "xyz" not in failed.data["error"].lower()
    assert "SECRET" not in failed.data["error_repr"]
    assert len(failed.data["error_repr"]) <= 514
    assert bot._live_event_remote_call_ids == {}


def test_candle_remote_fetch_url_is_sanitized_and_hashed():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink, cycle_id=None)
    url = "https://data.example/archive.zip?apiKey=SECRET&signature=abc#token"
    base = {
        "kind": "archive_http_get",
        "exchange": "binance",
        "url": url,
    }

    bot._handle_candle_remote_fetch_event({**base, "stage": "start"})
    bot._handle_candle_remote_fetch_event(
        {**base, "stage": "not_found", "status": 404, "elapsed_ms": 12}
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    started, skipped = sink.events
    assert skipped.remote_call_id == started.remote_call_id
    for event in (started, skipped):
        assert event.data["url"] == "https://data.example/archive.zip?[redacted]#[redacted]"
        assert event.data["url_hash"]
        assert len(event.data["url_hash"]) == 64
        assert "SECRET" not in event.data["url"]
        assert "signature=abc" not in event.data["url"]


def test_candle_remote_fetch_throttled_stage_emits_throttled_event():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink)
    base = {
        "kind": "ccxt_fetch_ohlcv",
        "exchange": "okx",
        "symbol": "BTC/USDT:USDT",
        "tf": "1m",
        "since_ts": 123000,
    }

    bot._handle_candle_remote_fetch_event({**base, "stage": "start"})
    bot._handle_candle_remote_fetch_event(
        {**base, "stage": "rate_limited", "elapsed_ms": 12}
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    started, throttled = sink.events
    assert throttled.event_type == EventTypes.REMOTE_CALL_THROTTLED
    assert throttled.status == "deferred"
    assert throttled.remote_call_id == started.remote_call_id
    assert throttled.remote_call_group_id == started.remote_call_group_id
    assert bot._live_event_remote_call_ids == {}


def test_archive_prefetch_progress_does_not_consume_remote_call_id():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink)
    base = {"kind": "archive_prefetch", "exchange": "binance", "symbol": "BTC/USDT"}

    bot._handle_candle_remote_fetch_event({**base, "stage": "start", "days_to_fetch": 3})
    bot._handle_candle_remote_fetch_event(
        {**base, "stage": "progress", "completed": 1, "total": 3}
    )
    bot._handle_candle_remote_fetch_event({**base, "stage": "done", "fetched": 3})

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    assert [event.event_type for event in sink.events] == [
        EventTypes.REMOTE_CALL_STARTED,
        EventTypes.REMOTE_CALL_SUCCEEDED,
    ]
    started, done = sink.events
    assert done.remote_call_id == started.remote_call_id
    assert done.remote_call_group_id == started.remote_call_group_id
    assert bot._live_event_remote_call_seq == 1
    assert bot._live_event_remote_call_ids == {}


def test_orphan_remote_fetch_result_is_marked_without_synthetic_id():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink)

    bot._handle_candle_remote_fetch_event(
        {
            "kind": "ccxt_fetch_ohlcv",
            "stage": "ok",
            "exchange": "okx",
            "symbol": "BTC/USDT:USDT",
            "tf": "1m",
            "since_ts": 123000,
            "rows": 1,
        }
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    event = sink.events[0]
    assert event.event_type == EventTypes.REMOTE_CALL_SUCCEEDED
    assert event.remote_call_id is None
    assert event.remote_call_group_id is None
    assert event.data["orphan_result"] is True
    assert bot._live_event_remote_call_seq == 0


def test_remote_fetch_correlation_map_is_bounded():
    sink = ListEventSink()
    bot = _make_remote_fetch_event_bot(sink, map_max=2)

    for idx in range(5):
        bot._handle_candle_remote_fetch_event(
            {
                "kind": "ccxt_fetch_ohlcv",
                "stage": "start",
                "exchange": "okx",
                "symbol": f"SYM{idx}/USDT:USDT",
                "tf": "1m",
                "since_ts": idx,
            }
        )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    assert len(bot._live_event_remote_call_ids) == 2
    assert bot._live_event_remote_call_seq == 5


@pytest.mark.asyncio
async def test_authoritative_timed_fetch_emits_correlated_remote_call_events():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _timed_authoritative_fetch = pb_mod.Passivbot._timed_authoritative_fetch

        def __init__(self):
            self.exchange = "kucoin"
            self.user = "kucoin_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_11"
            self._authoritative_refresh_epoch = 17
            self._authoritative_pending_confirmations = {"open_orders": 18}
            self._live_event_remote_call_seq = 0
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    async def fetch_open_orders():
        return [
            {"id": "a", "symbol": "BTC/USDT:USDT"},
            {"id": "b", "symbol": "ETH/USDT:USDT"},
        ]

    bot = FakeBot()
    timings_ms = {}
    result = await bot._timed_authoritative_fetch(
        "open_orders", fetch_open_orders(), timings_ms
    )

    assert len(result) == 2
    assert "open_orders" in timings_ms
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    started, succeeded = sink.events
    assert [event.event_type for event in sink.events] == [
        EventTypes.REMOTE_CALL_STARTED,
        EventTypes.REMOTE_CALL_SUCCEEDED,
    ]
    assert started.remote_call_id == succeeded.remote_call_id
    assert started.remote_call_group_id == "cy_11:authoritative"
    assert succeeded.remote_call_group_id == "cy_11:authoritative"
    assert started.reason_code == "authoritative_open_orders"
    assert succeeded.data["surface"] == "open_orders"
    assert succeeded.data["count"] == 2
    assert succeeded.data["state_epoch"] == 17
    assert succeeded.data["pending_confirmations"] == ["open_orders"]


@pytest.mark.asyncio
async def test_authoritative_timed_fetch_failure_emits_sanitized_remote_call_event():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _timed_authoritative_fetch = pb_mod.Passivbot._timed_authoritative_fetch

        def __init__(self):
            self.exchange = "kucoin"
            self.user = "kucoin_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = None
            self._authoritative_refresh_epoch = 19
            self._authoritative_pending_confirmations = {}
            self._live_event_remote_call_seq = 0
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    async def fetch_positions():
        raise RuntimeError("apiKey=SECRET token SECRET Bearer abc123")

    bot = FakeBot()
    timings_ms = {}
    with pytest.raises(RuntimeError, match="apiKey=SECRET"):
        await bot._timed_authoritative_fetch("positions", fetch_positions(), timings_ms)

    assert "positions" in timings_ms
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert bot._live_event_pipeline.close(timeout=2.0) is True
    started, failed = sink.events
    assert failed.event_type == EventTypes.REMOTE_CALL_FAILED
    assert failed.remote_call_id == started.remote_call_id
    assert failed.remote_call_group_id == "auth_19:authoritative"
    assert failed.data["surface"] == "positions"
    assert failed.data["error_type"] == "RuntimeError"
    assert "SECRET" not in failed.data["error"]
    assert "abc123" not in failed.data["error"]
    assert "SECRET" not in failed.data["error_repr"]


@pytest.mark.asyncio
async def test_authoritative_timed_fetch_emit_failure_does_not_skip_fetch():
    import passivbot as pb_mod

    class FakeBot:
        _timed_authoritative_fetch = pb_mod.Passivbot._timed_authoritative_fetch

        def __init__(self):
            self.exchange = "kucoin"
            self.user = "kucoin_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_12"
            self._authoritative_refresh_epoch = 20
            self._authoritative_pending_confirmations = {}
            self._live_event_remote_call_seq = 0
            self.fetch_called = False

        def _emit_live_event(self, *args, **kwargs):
            raise RuntimeError("event sink boom")

    async def fetch_ok():
        bot.fetch_called = True
        return ["ok"]

    bot = FakeBot()
    timings_ms = {}
    result = await bot._timed_authoritative_fetch("open_orders", fetch_ok(), timings_ms)

    assert result == ["ok"]
    assert bot.fetch_called is True
    assert "open_orders" in timings_ms


@pytest.mark.asyncio
async def test_authoritative_timed_fetch_emit_failure_preserves_fetch_exception():
    import passivbot as pb_mod

    original_error = RuntimeError("positions failed")

    class FakeBot:
        _timed_authoritative_fetch = pb_mod.Passivbot._timed_authoritative_fetch

        def __init__(self):
            self.exchange = "kucoin"
            self.user = "kucoin_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_13"
            self._authoritative_refresh_epoch = 21
            self._authoritative_pending_confirmations = {}
            self._live_event_remote_call_seq = 0

        def _emit_live_event(self, *args, **kwargs):
            raise RuntimeError("event sink boom")

    async def fetch_fail():
        raise original_error

    bot = FakeBot()
    timings_ms = {}
    with pytest.raises(RuntimeError) as exc_info:
        await bot._timed_authoritative_fetch("positions", fetch_fail(), timings_ms)

    assert exc_info.value is original_error
    assert "positions" in timings_ms


def test_monitor_emit_stop_records_startup_terminal_structured_stopped():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _monitor_emit_stop = pb_mod.Passivbot._monitor_emit_stop
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event

        def __init__(self):
            self.exchange = "gateio"
            self.user = "gateio_01"
            self.bot_id = "bot_1"
            self._monitor_stop_emitted = False
            self.monitor_publisher = RecorderPublisher()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._monitor_emit_stop("startup_error", payload={"stage": "init_markets"})

    assert bot.monitor_publisher.events[-1]["kind"] == "bot.stop"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert [event.event_type for event in sink.events] == [EventTypes.BOT_STOPPED]
    event = sink.events[0]
    assert event.status == "failed"
    assert event.reason_code == "startup_error"
    assert event.data["reason"] == "startup_error"
    assert event.data["stage"] == "init_markets"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_handle_balance_update_records_monitor_balance_event():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_balance_changed_event = pb_mod.Passivbot._emit_balance_changed_event
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event

        def __init__(self):
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_20"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )
            self.monitor_publisher = RecorderPublisher()
            self._previous_balance_raw = 90.0
            self._previous_balance_snapped = 88.0
            self._last_raw_only_log_time = 0.0
            self._monitor_last_equity = 0.0
            self.execution_scheduled = False

        def get_raw_balance(self):
            return 100.0

        def get_hysteresis_snapped_balance(self):
            return 95.0

        async def calc_upnl_sum(self):
            return 7.5

    bot = FakeBot()

    await pb_mod.Passivbot.handle_balance_update(bot, source="REST")

    assert bot.execution_scheduled is True
    assert bot._monitor_last_equity == pytest.approx(107.5)
    assert bot.monitor_publisher.events[-1]["kind"] == "account.balance"
    assert bot.monitor_publisher.events[-1]["payload"]["equity"] == pytest.approx(107.5)
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.BALANCE_CHANGED
    assert event.cycle_id == "cy_20"
    assert event.status == "succeeded"
    assert event.reason_code == "balance_changed"
    assert event.data["balance_raw_delta"] == pytest.approx(10.0)
    assert event.data["balance_snapped_delta"] == pytest.approx(7.0)
    assert event.data["equity"] == pytest.approx(107.5)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_risk_mode_changed_event_emits_structured_summary():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_risk_mode_changed_event = pb_mod.Passivbot._emit_risk_mode_changed_event

        def __init__(self):
            self.exchange = "gateio"
            self.user = "gateio_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_risk_mode"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._emit_risk_mode_changed_event(
        pside="long",
        source="hsl",
        action="replace",
        symbols=["ETH/USDT:USDT", "BTC/USDT:USDT"],
        previous_modes={"BTC/USDT:USDT": "normal"},
        modes={"BTC/USDT:USDT": "panic", "ETH/USDT:USDT": "panic"},
        reason_code="hsl_red_runtime_forced_modes",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.RISK_MODE_CHANGED
    assert event.cycle_id == "cy_risk_mode"
    assert event.pside == "long"
    assert event.reason_code == "hsl_red_runtime_forced_modes"
    assert event.component == "risk.hsl.mode"
    assert event.tags == ("risk", "mode", "hsl")
    assert event.data["action"] == "replace"
    assert event.data["previous_mode_counts"] == {"normal": 1}
    assert event.data["mode_counts"] == {"panic": 2}
    assert event.data["symbols"]["count"] == 2
    assert event.data["symbols"]["sample"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_log_new_fill_events_emits_fill_ingested_event():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_fill_ingested_event = pb_mod.Passivbot._emit_fill_ingested_event
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _log_fill_event = pb_mod.Passivbot._log_fill_event
        _log_new_fill_events = pb_mod.Passivbot._log_new_fill_events
        _monitor_fill_payload = pb_mod.Passivbot._monitor_fill_payload
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_record_fill_history = pb_mod.Passivbot._monitor_record_fill_history

        def __init__(self):
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_21"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )
            self.monitor_publisher = RecorderPublisher()
            self._health_fills = 0
            self._health_pnl = 0.0

    source_ids = ["trade-a", "trade-b"]
    source_derived_fill_id = "+".join(source_ids)
    event = SimpleNamespace(
        id=source_derived_fill_id,
        timestamp=1_782_271_234_000,
        symbol="ETH/USDT:USDT",
        side="sell",
        position_side="long",
        qty=0.5,
        price=2500.0,
        pnl=12.0,
        fee=-0.5,
        fee_paid=-0.5,
        pb_order_type="close_grid_normal_long",
        client_order_id="client-abcdef123456",
        source_ids=source_ids,
        pnl_status="complete",
    )
    bot = FakeBot()

    bot._log_new_fill_events([event])

    assert bot._health_fills == 1
    assert bot._health_pnl == pytest.approx(11.5)
    assert bot.monitor_publisher.events[-1]["kind"] == "order.filled"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    live_event = sink.events[-1]
    assert live_event.event_type == EventTypes.FILL_INGESTED
    assert live_event.cycle_id == "cy_21"
    assert live_event.symbol == "ETH/USDT:USDT"
    assert live_event.pside == "long"
    assert live_event.side == "sell"
    assert live_event.data["qty"] == pytest.approx(0.5)
    assert live_event.data["pnl"] == pytest.approx(12.0)
    assert live_event.data["fill_id_hash"] == hashlib.sha256(
        source_derived_fill_id.encode("utf-8")
    ).hexdigest()
    assert live_event.data["source_ids_count"] == 2
    assert "id_short" not in live_event.data
    assert "source_ids" not in live_event.data
    assert "trade-a" not in str(live_event.data)
    assert "trade-b" not in str(live_event.data)
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_emit_fills_refresh_summary_event_sanitizes_error_and_stays_off_console():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _emit_fills_refresh_summary_event = (
            pb_mod.Passivbot._emit_fills_refresh_summary_event
        )

        def __init__(self):
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_fills"
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()
    bot._emit_fills_refresh_summary_event(
        source="staged_blocking",
        refresh_mode="lookback_bootstrap",
        status="failed",
        reason_code="fill_refresh_failed",
        elapsed_ms=123,
        lookback=30.0,
        history_scope="window",
        event_count_before=3,
        coverage_before={
            "ready": False,
            "reason": "known_gap_overlaps_lookback",
            "history_scope": "window",
            "covered_start_ms": 100,
            "oldest_event_ts": 90,
            "gap_start_ts": 120,
            "gap_end_ts": 180,
            "gap_reason": "fetch_failed",
        },
        next_retry_in_ms=45_000,
        error=RuntimeError("apiKey=secret-token failed"),
        level="error",
    )

    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.FILLS_REFRESH_SUMMARY
    assert event.cycle_id == "cy_fills"
    assert event.status == "failed"
    assert event.reason_code == "fill_refresh_failed"
    assert event.data["source"] == "staged_blocking"
    assert event.data["refresh_mode"] == "lookback_bootstrap"
    assert event.data["coverage_ready_before"] is False
    assert event.data["coverage_reason_before"] == "known_gap_overlaps_lookback"
    assert event.data["coverage_before"]["gap_reason"] == "fetch_failed"
    assert event.data["next_retry_in_ms"] == 45_000
    assert event.data["error_type"] == "RuntimeError"
    assert "secret-token" not in event.data["error"]
    assert "[redacted]" in event.data["error"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execute_orders_parent_records_order_opened_event():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_order_payload = pb_mod.Passivbot._monitor_order_payload
        _is_market_execution_order = staticmethod(
            pb_mod.Passivbot._is_market_execution_order
        )
        _log_market_execution_notice = pb_mod.Passivbot._log_market_execution_notice

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._health_orders_placed = 0
            self.debug_mode = False
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_11"
            self._order_wave_in_progress = {"id": 7, "event_id": "ow_7"}
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

        def live_value(self, key):
            assert key == "max_n_creations_per_batch"
            return 5

        def add_to_recent_order_executions(self, order):
            return None

        def log_order_action(self, *args, **kwargs):
            return None

        def _log_order_action_summary(self, *args, **kwargs):
            return None

        async def execute_orders(self, orders):
            return [{"id": "abc123", **orders[0]}]

        def did_create_order(self, executed):
            return True

        def add_new_order(self, order, source="POST"):
            return None

        def _resolve_pb_order_type(self, order):
            return "entry_grid_normal_long"

    bot = FakeBot()
    order = {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "price": 100000.0,
        "reduce_only": False,
        "custom_id": "cid123",
    }

    res = await pb_mod.Passivbot.execute_orders_parent(bot, [order])

    assert len(res) == 1
    assert bot._health_orders_placed == 1
    assert bot.monitor_publisher.events[-1]["kind"] == "order.opened"
    assert bot.monitor_publisher.events[-1]["symbol"] == "BTC/USDT:USDT"
    assert bot.monitor_publisher.events[-1]["payload"]["pb_order_type"] == "entry_grid_normal_long"
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event_types = [event.event_type for event in sink.events]
    assert event_types == [
        EventTypes.EXECUTION_CREATE_SENT,
        EventTypes.EXECUTION_CREATE_SUCCEEDED,
    ]
    sent, succeeded = sink.events
    assert sent.cycle_id == "cy_11"
    assert sent.order_wave_id == "ow_7"
    assert sent.action_id == "ow_7:create:0"
    assert sent.symbol == "BTC/USDT:USDT"
    assert sent.client_order_id == "cid123"
    assert succeeded.order_id == "abc123"
    assert succeeded.status == "succeeded"
    assert succeeded.data["result_order_id_short"] == "abc123"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_execute_cancellations_parent_emits_ambiguous_confirmation_events():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_confirmation_requested_event = (
            pb_mod.Passivbot._emit_execution_confirmation_requested_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _monitor_order_payload = staticmethod(lambda order, source="POST": dict(order))
        _request_authoritative_confirmation = (
            pb_mod.Passivbot._request_authoritative_confirmation
        )
        _cancel_result_requires_full_authoritative_confirmation = (
            pb_mod.Passivbot._cancel_result_requires_full_authoritative_confirmation
        )

        def __init__(self):
            self.exchange = "hyperliquid"
            self.user = "hyperliquid_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_12"
            self._authoritative_pending_confirmations = {}
            self._authoritative_refresh_epoch = 4
            self._order_wave_in_progress = {"id": 9, "event_id": "ow_9"}
            self._health_orders_cancelled = 0
            self.state_change_detected_by_symbol = set()
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

        def live_value(self, key):
            assert key == "max_n_cancellations_per_batch"
            return 5

        def add_to_recent_order_cancellations(self, order):
            return None

        def log_order_action(self, *args, **kwargs):
            return None

        def _log_order_action_summary(self, *args, **kwargs):
            return None

        async def execute_cancellations(self, orders):
            return [
                {
                    "status": "success",
                    "_passivbot_cancel_requires_full_authoritative_confirmation": True,
                    **orders[0],
                }
            ]

        def did_cancel_order(self, executed, order):
            return True

        def remove_order(self, order, source="POST"):
            return None

        def _monitor_record_event(self, *args, **kwargs):
            return None

        def _authoritative_full_confirmation_surfaces(self):
            return {"balance", "positions", "open_orders", "fills"}

    bot = FakeBot()
    order = {
        "id": "order-1",
        "symbol": "ETH/USDT:USDT",
        "side": "sell",
        "position_side": "long",
        "qty": 0.5,
        "price": 2500.0,
        "reduce_only": True,
        "custom_id": "cid-cancel",
    }

    res = await pb_mod.Passivbot.execute_cancellations_parent(bot, [order])

    assert len(res) == 1
    assert bot._authoritative_pending_confirmations == {
        "balance": 5,
        "positions": 5,
        "open_orders": 5,
        "fills": 5,
    }
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event_types = [event.event_type for event in sink.events]
    assert event_types == [
        EventTypes.EXECUTION_CANCEL_SENT,
        EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
        EventTypes.EXECUTION_CONFIRMATION_REQUESTED,
    ]
    ambiguous = sink.events[1]
    assert ambiguous.status == "degraded"
    assert ambiguous.reason_code == "requires_full_authoritative_confirmation"
    assert ambiguous.order_wave_id == "ow_9"
    requested = sink.events[2]
    assert requested.data["target_epoch"] == 5
    assert requested.data["surfaces"] == ["balance", "fills", "open_orders", "positions"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_order_wave_settlement_emits_confirmation_satisfied_event():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_confirmation_satisfied_event = (
            pb_mod.Passivbot._emit_execution_confirmation_satisfied_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _log_settled_order_waves = pb_mod.Passivbot._log_settled_order_waves

        def __init__(self):
            self.exchange = "okx"
            self.user = "okx_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_13"
            self._pending_order_waves = [
                {
                    "id": 3,
                    "event_id": "ow_3",
                    "started_ms": 1_000,
                    "posted_ms": 2_000,
                    "planned_cancel": 1,
                    "planned_create": 1,
                    "cancel_posted": 1,
                    "create_posted": 1,
                    "symbols": ["BTC/USDT:USDT"],
                    "confirmations": {"open_orders": 6},
                }
            ]
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._log_settled_order_waves(
        current_epoch=6,
        fresh_surfaces={"open_orders"},
        changed_surfaces=["open_orders"],
    )

    assert bot._pending_order_waves == []
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    event = sink.events[-1]
    assert event.event_type == EventTypes.EXECUTION_CONFIRMATION_SATISFIED
    assert event.cycle_id == "cy_13"
    assert event.order_wave_id == "ow_3"
    assert event.status == "succeeded"
    assert event.data["confirmations"] == {"open_orders": 6}
    assert event.data["fresh_surfaces"] == ["open_orders"]
    assert bot._live_event_pipeline.close(timeout=2.0) is True


def test_order_wave_settlement_emits_confirmation_timeout_once():
    import passivbot as pb_mod

    sink = ListEventSink()

    class FakeBot:
        _current_live_event_cycle_id = pb_mod.Passivbot._current_live_event_cycle_id
        _emit_execution_confirmation_satisfied_event = (
            pb_mod.Passivbot._emit_execution_confirmation_satisfied_event
        )
        _emit_execution_confirmation_timeout_event = (
            pb_mod.Passivbot._emit_execution_confirmation_timeout_event
        )
        _emit_live_event = pb_mod.Passivbot._emit_live_event
        _log_settled_order_waves = pb_mod.Passivbot._log_settled_order_waves

        def __init__(self):
            self.exchange = "binance"
            self.user = "binance_01"
            self.bot_id = "bot_1"
            self._live_event_current_cycle_id = "cy_14"
            self._order_wave_confirmation_timeout_ms = 1
            self._pending_order_waves = [
                {
                    "id": 4,
                    "event_id": "ow_4",
                    "started_ms": 1_000,
                    "posted_ms": 2_000,
                    "planned_cancel": 0,
                    "planned_create": 1,
                    "cancel_posted": 0,
                    "create_posted": 1,
                    "symbols": ["ETH/USDT:USDT"],
                    "confirmations": {"open_orders": 9},
                }
            ]
            self._live_event_pipeline = LiveEventPipeline(
                structured_sinks=[sink],
                monitor_sinks=[],
            )

    bot = FakeBot()

    bot._log_settled_order_waves(
        current_epoch=8,
        fresh_surfaces=set(),
        changed_surfaces=[],
    )
    bot._log_settled_order_waves(
        current_epoch=8,
        fresh_surfaces=set(),
        changed_surfaces=[],
    )

    assert len(bot._pending_order_waves) == 1
    assert bot._pending_order_waves[0]["timeout_emitted"] is True
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    timeout_events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.EXECUTION_CONFIRMATION_TIMEOUT
    ]
    assert len(timeout_events) == 1
    timeout_event = timeout_events[0]
    assert timeout_event.cycle_id == "cy_14"
    assert timeout_event.order_wave_id == "ow_4"
    assert timeout_event.status == "degraded"
    assert timeout_event.reason_code == "authoritative_confirmation_timeout"
    assert timeout_event.data["pending_surfaces"] == ["open_orders"]
    assert timeout_event.data["confirmations"] == {"open_orders": 9}

    bot._log_settled_order_waves(
        current_epoch=9,
        fresh_surfaces={"open_orders"},
        changed_surfaces=["open_orders"],
    )

    assert bot._pending_order_waves == []
    assert bot._live_event_pipeline.flush(timeout=2.0) is True
    assert sink.events[-1].event_type == EventTypes.EXECUTION_CONFIRMATION_SATISFIED
    assert sink.events[-1].order_wave_id == "ow_4"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


@pytest.mark.asyncio
async def test_start_bot_records_startup_error_stop_and_early_snapshot(monkeypatch):
    import passivbot as pb_mod

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(pb_mod, "format_approved_ignored_coins", _noop)

    class FakeBot:
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_record_error = pb_mod.Passivbot._monitor_record_error
        _monitor_emit_stop = pb_mod.Passivbot._monitor_emit_stop
        _emit_live_event = staticmethod(lambda *args, **kwargs: None)
        _set_log_silence_watchdog_context = pb_mod.Passivbot._set_log_silence_watchdog_context
        _start_log_silence_watchdog = pb_mod.Passivbot._start_log_silence_watchdog
        _stop_log_silence_watchdog = pb_mod.Passivbot._stop_log_silence_watchdog
        _shutdown_requested = pb_mod.Passivbot._shutdown_requested
        _raise_if_shutdown_requested = pb_mod.Passivbot._raise_if_shutdown_requested
        _sleep_unless_shutdown = pb_mod.Passivbot._sleep_unless_shutdown
        _startup_exception_is_terminal = staticmethod(
            pb_mod.Passivbot._startup_exception_is_terminal
        )

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._monitor_stop_emitted = False
            self.config = {"live": {}, "monitor": {"enabled": True}}
            self.user_info = {"exchange": "bitget"}
            self.exchange = "bitget"
            self.user = "bitget_01"
            self.quote = "USDT"
            self.start_time_ms = 1234567890
            self.debug_mode = False
            self.stop_signal_received = False
            self.snapshot_flushes = []
            self._log_silence_watchdog_seconds = 0.0
            self._log_silence_watchdog_phase = "startup"
            self._log_silence_watchdog_stage = "idle"
            self._log_silence_watchdog_task = None
            self._bot_ready = False

        def _log_startup_banner(self):
            return None

        async def init_markets(self):
            return None

        async def warmup_trading_ready_candles(self):
            return None

        def _equity_hard_stop_enabled(self):
            return False

        def _log_memory_snapshot(self):
            return None

        async def start_data_maintainers(self):
            raise RuntimeError("maintainers failed")

        async def _monitor_flush_snapshot(self, *, force=False, ts=None):
            self.snapshot_flushes.append({"force": force, "ts": ts})
            return True

    bot = FakeBot()

    with pytest.raises(RuntimeError, match="maintainers failed"):
        await pb_mod.Passivbot.start_bot(bot)

    assert bot.snapshot_flushes
    assert bot.snapshot_flushes[0]["force"] is True
    assert bot.monitor_publisher.events[0]["kind"] == "bot.start"
    assert bot.monitor_publisher.events[-1]["kind"] == "bot.stop"
    assert bot.monitor_publisher.events[-1]["payload"]["reason"] == "startup_error"
    assert bot.monitor_publisher.events[-1]["payload"]["stage"] == "start_data_maintainers"
    assert bot.monitor_publisher.errors[-1]["kind"] == "error.bot"
    assert bot.monitor_publisher.errors[-1]["payload"]["source"] == "start_bot"
    assert bot.monitor_publisher.errors[-1]["payload"]["stage"] == "start_data_maintainers"


def test_maybe_log_silence_watchdog_emits_phase_and_stage(monkeypatch, caplog):
    import passivbot as pb_mod

    class FakeBot:
        _maybe_log_silence_watchdog = pb_mod.Passivbot._maybe_log_silence_watchdog
        _format_duration = pb_mod.Passivbot._format_duration

        def __init__(self):
            self._log_silence_watchdog_seconds = 60.0
            self._log_silence_watchdog_phase = "startup"
            self._log_silence_watchdog_stage = "equity_hard_stop_initialize_from_history"
            self._health_start_ms = 0
            self._last_loop_duration_ms = 0

    monkeypatch.setattr(pb_mod, "get_last_log_activity_monotonic", lambda: 0.0)
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: 120_000)
    caplog.set_level(logging.INFO)

    bot = FakeBot()

    assert bot._maybe_log_silence_watchdog(now_monotonic=61.0) is True
    assert any("silence watchdog" in rec.message for rec in caplog.records)
    assert any("phase=startup" in rec.message for rec in caplog.records)
    assert any("stage=equity_hard_stop_initialize_from_history" in rec.message for rec in caplog.records)


def test_log_new_fill_events_records_fill_history():
    import passivbot as pb_mod

    class FakeBot:
        _log_new_fill_events = pb_mod.Passivbot._log_new_fill_events
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_fill_payload = pb_mod.Passivbot._monitor_fill_payload
        _monitor_record_fill_history = pb_mod.Passivbot._monitor_record_fill_history

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._health_fills = 0
            self._health_pnl = 0.0
            self.quote = "USDT"

        def _log_fill_event(self, event):
            return f"fill {event.id}"

    bot = FakeBot()
    event = SimpleNamespace(
        id="fill-1",
        timestamp=1234,
        symbol="BTC/USDT:USDT",
        side="buy",
        position_side="long",
        qty=0.01,
        price=100000.0,
        pnl=1.25,
        fee=0.1,
        pb_order_type="entry_grid_normal_long",
        client_order_id="cid-1",
        source_ids=["src-1"],
        raw={"exchange_fill_id": "ex-1"},
    )

    bot._log_new_fill_events([event])

    assert bot.monitor_publisher.fills[-1]["symbol"] == "BTC/USDT:USDT"
    assert bot.monitor_publisher.fills[-1]["payload"]["id"] == "fill-1"
    assert bot.monitor_publisher.events[-1]["kind"] == "order.filled"
    assert bot._health_fills == 1
    assert bot._health_pnl == pytest.approx(1.25)


def test_monitor_record_price_ticks_delegates_valid_prices_only():
    import passivbot as pb_mod

    class FakeBot:
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()

    bot = FakeBot()

    emitted = bot._monitor_record_price_ticks(
        {
            "BTC/USDT:USDT": 100000.0,
            "ETH/USDT:USDT": 0.0,
            "SOL/USDT:USDT": float("nan"),
            "XRP/USDT:USDT": 2.5,
        },
        ts=1234,
        source="test",
    )

    assert emitted == 2
    assert [entry["symbol"] for entry in bot.monitor_publisher.price_ticks] == [
        "BTC/USDT:USDT",
        "XRP/USDT:USDT",
    ]
    assert all(entry["source"] == "test" for entry in bot.monitor_publisher.price_ticks)


def test_monitor_handle_candlestick_persist_requires_ready_bot():
    import passivbot as pb_mod

    class FakeBot:
        _monitor_handle_candlestick_persist = pb_mod.Passivbot._monitor_handle_candlestick_persist

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._bot_ready = False

    bot = FakeBot()
    batch = np.array(
        [
            (60_000, 1.0, 2.0, 0.5, 1.5, 10.0),
            (120_000, 1.5, 2.5, 1.0, 2.0, 11.0),
        ],
        dtype=pb_mod.CANDLE_DTYPE,
    )

    bot._monitor_handle_candlestick_persist("BTC/USDT:USDT", "1m", batch)
    assert bot.monitor_publisher.completed_candles == []

    bot._bot_ready = True
    bot._monitor_handle_candlestick_persist("BTC/USDT:USDT", "1m", batch)

    assert len(bot.monitor_publisher.completed_candles) == 1
    entry = bot.monitor_publisher.completed_candles[0]
    assert entry["symbol"] == "BTC/USDT:USDT"
    assert entry["timeframe"] == "1m"
    assert entry["candles"][0]["ts"] == 60_000
    assert entry["candles"][1]["bv"] == pytest.approx(11.0)


def test_live_event_pipeline_install_routes_diagnostics_to_monitor_projection():
    import passivbot as pb_mod

    class FakeBot:
        _install_live_event_pipeline = pb_mod.Passivbot._install_live_event_pipeline
        _close_live_event_pipeline = pb_mod.Passivbot._close_live_event_pipeline

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._live_event_pipeline = None
            self.exchange = "bybit"
            self.user = "bybit_01"
            self.bot_id = "bot_1"

        def config_get(self, keys):
            return self.user

    bot = FakeBot()
    pipeline = bot._install_live_event_pipeline()

    emitted = pb_mod.emit_diagnostic_event(
        bot,
        pb_mod.DiagnosticEvent.build(
            "planning_unavailable",
            ("diagnostic", "planning"),
            {"cycle_id": 7, "defer_reason": "stale_ema", "apiKey": "secret"},
            ts_ms=1234,
            symbol="BTC/USDT:USDT",
            pside="long",
        ),
    )

    assert emitted.event_type == EventTypes.PLANNING_UNAVAILABLE
    assert pipeline.flush(timeout=2.0) is True
    event = bot.monitor_publisher.events[-1]
    assert event["kind"] == EventTypes.PLANNING_UNAVAILABLE
    assert event["symbol"] == "BTC/USDT:USDT"
    assert event["pside"] == "long"
    assert event["payload"]["cycle_id"] == 7
    assert event["payload"]["defer_reason"] == "stale_ema"
    assert event["payload"]["apiKey"] == "[redacted]"
    live_event = event["payload"]["_live_event"]
    assert live_event["event_type"] == EventTypes.PLANNING_UNAVAILABLE
    assert live_event["ids"]["bot_id"] == "bot_1"
    assert live_event["data"]["apiKey"] == "[redacted]"
    assert bot._close_live_event_pipeline(timeout=2.0) is True
    assert bot._live_event_pipeline is None


def test_close_live_event_pipeline_is_best_effort_and_clears_reference():
    import passivbot as pb_mod

    class FakePipeline:
        def __init__(self):
            self.closed_with = None

        def close(self, timeout):
            self.closed_with = timeout
            return True

    class FakeBot:
        _close_live_event_pipeline = pb_mod.Passivbot._close_live_event_pipeline

        def __init__(self):
            self._live_event_pipeline = FakePipeline()

    bot = FakeBot()
    pipeline = bot._live_event_pipeline

    assert bot._close_live_event_pipeline(timeout=1.25) is True
    assert pipeline.closed_with == 1.25
    assert bot._live_event_pipeline is None


@pytest.mark.asyncio
async def test_shutdown_gracefully_closes_event_pipeline_before_monitor_publisher():
    import passivbot as pb_mod

    order = []

    class FakePipeline:
        def close(self, timeout):
            order.append(("pipeline", timeout))
            return True

    class FakePublisher:
        def close(self):
            order.append(("publisher", None))

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._emit_live_event = lambda event_type, *args, **kwargs: order.append(
        ("event", event_type)
    )
    bot._monitor_emit_stop = lambda *args, **kwargs: order.append(("stop", None))

    async def flush_snapshot(*, force=False, ts=None):
        order.append(("snapshot", force))
        return True

    bot._monitor_flush_snapshot = flush_snapshot
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot._execution_loop_task = None
    bot._execution_loop_stopped = None
    bot.ccp = None
    bot.cca = None
    bot._live_event_pipeline = FakePipeline()
    bot.monitor_publisher = FakePublisher()

    await bot.shutdown_gracefully()

    assert order[-4:] == [
        ("event", EventTypes.BOT_STOPPED),
        ("event", EventTypes.BOT_SHUTDOWN_STAGE),
        ("pipeline", 2.0),
        ("publisher", None),
    ]
    assert ("snapshot", True) in order
    assert bot._live_event_pipeline is None


@pytest.mark.asyncio
async def test_build_monitor_snapshot_includes_market_forager_unstuck_and_recent():
    import passivbot as pb_mod

    class FakeCM:
        def __init__(self):
            self._current_close_cache = {"BTC/USDT:USDT": (100500.0, 123450)}

        def get_last_refresh_ms(self, symbol):
            return 123400

        def get_last_final_ts(self, symbol):
            return 123000

    class FakeBot:
        _build_monitor_snapshot = pb_mod.Passivbot._build_monitor_snapshot
        _build_health_summary_payload = pb_mod.Passivbot._build_health_summary_payload
        _monitor_hsl_payload = pb_mod.Passivbot._monitor_hsl_payload
        _monitor_order_payload = pb_mod.Passivbot._monitor_order_payload
        _monitor_recent_orders_payload = pb_mod.Passivbot._monitor_recent_orders_payload
        _build_monitor_position_side_payload = pb_mod.Passivbot._build_monitor_position_side_payload
        _build_monitor_positions_section = pb_mod.Passivbot._build_monitor_positions_section
        _build_monitor_market_section = pb_mod.Passivbot._build_monitor_market_section
        _build_monitor_trailing_section = pb_mod.Passivbot._build_monitor_trailing_section
        _build_monitor_forager_section = pb_mod.Passivbot._build_monitor_forager_section
        _build_monitor_unstuck_section = pb_mod.Passivbot._build_monitor_unstuck_section
        _build_monitor_runtime_market_hints = pb_mod.Passivbot._build_monitor_runtime_market_hints
        _build_monitor_runtime_unstuck_hints = pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints
        _build_monitor_recent_section = pb_mod.Passivbot._build_monitor_recent_section
        _resolve_pb_order_type = pb_mod.Passivbot._resolve_pb_order_type

        def __init__(self):
            self.exchange = "bitget"
            self.user = "bitget_01"
            self.quote = "USDT"
            self.start_time_ms = 123000
            self._health_start_ms = 100000
            self._last_loop_duration_ms = 250
            self._health_orders_placed = 2
            self._health_orders_cancelled = 1
            self._health_fills = 3
            self._health_pnl = 1.5
            self._health_ws_reconnects = 1
            self._health_rate_limits = 0
            self._monitor_last_equity = 1005.0
            self.error_counts = [250000]
            self.positions = {
                "BTC/USDT:USDT": {
                    "long": {"size": 0.001, "price": 100000.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            }
            self.open_orders = {
                "BTC/USDT:USDT": [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "side": "buy",
                        "position_side": "long",
                        "qty": 0.001,
                        "price": 99000.0,
                        "custom_id": "entry_grid_normal_long",
                        "pb_order_type": "entry_grid_normal_long",
                    },
                    {
                        "symbol": "BTC/USDT:USDT",
                        "side": "sell",
                        "position_side": "long",
                        "qty": 0.0005,
                        "price": 101000.0,
                        "custom_id": "close_unstuck_long",
                        "pb_order_type": "close_unstuck_long",
                    },
                ]
            }
            self.PB_modes = {
                "long": {"BTC/USDT:USDT": "normal", "ETH/USDT:USDT": "normal"},
                "short": {},
            }
            self._runtime_forced_modes = {
                "long": {"BTC/USDT:USDT": "normal", "ETH/USDT:USDT": "normal"},
                "short": {},
            }
            self.trailing_prices = {
                "BTC/USDT:USDT": {
                    "long": {
                        "min_since_open": 99500.0,
                        "max_since_min": 100800.0,
                        "max_since_open": 100900.0,
                        "min_since_max": 100100.0,
                    }
                }
            }
            self.active_symbols = ["BTC/USDT:USDT"]
            self.approved_coins = {"long": {"BTC/USDT:USDT", "ETH/USDT:USDT"}, "short": set()}
            self.ignored_coins = {"long": set(), "short": {"ETH/USDT:USDT"}}
            self.approved_coins_minus_ignored_coins = {
                "long": {"BTC/USDT:USDT", "ETH/USDT:USDT"},
                "short": set(),
            }
            self.effective_min_cost = {"BTC/USDT:USDT": 5.0, "ETH/USDT:USDT": 5.0}
            self.min_costs = {"BTC/USDT:USDT": 1.0, "ETH/USDT:USDT": 1.0}
            self.min_qtys = {"BTC/USDT:USDT": 0.001, "ETH/USDT:USDT": 0.001}
            self.price_steps = {"BTC/USDT:USDT": 0.1, "ETH/USDT:USDT": 0.1}
            self.qty_steps = {"BTC/USDT:USDT": 0.001, "ETH/USDT:USDT": 0.001}
            self.markets_dict = {
                "BTC/USDT:USDT": {"active": True},
                "ETH/USDT:USDT": {"active": True},
            }
            self.recent_order_executions = [
                {
                        "symbol": "BTC/USDT:USDT",
                        "side": "buy",
                        "position_side": "long",
                        "qty": 0.001,
                        "price": 99000.0,
                    "custom_id": "entry_grid_normal_long",
                    "pb_order_type": "entry_grid_normal_long",
                    "execution_timestamp": 123456,
                    "source": "POST",
                }
            ]
            self.recent_order_cancellations = [
                {
                        "symbol": "BTC/USDT:USDT",
                        "side": "sell",
                        "position_side": "long",
                        "qty": 0.0005,
                        "price": 101000.0,
                    "custom_id": "close_unstuck_long",
                    "pb_order_type": "close_unstuck_long",
                    "execution_timestamp": 123460,
                    "source": "REST",
                }
            ]
            self.cm = FakeCM()
            self.market_snapshot_provider = SimpleNamespace(
                _cache={
                    "BTC/USDT:USDT": SimpleNamespace(
                        last=100500.0,
                        fetched_ms=123450,
                        source="test_snapshot",
                    )
                }
            )
            self.inverse = False
            self.c_mults = {"BTC/USDT:USDT": 1.0}
            self.pside_int_map = {"long": 0, "short": 1}
            self._coin_bot_values = {
                ("long", "forager_score_weights"): {
                    "volume": 1.0,
                    "volatility": 0.5,
                    "ema_readiness": 0.25,
                },
                ("short", "forager_score_weights"): {
                    "volume": 1.0,
                    "volatility": 0.5,
                    "ema_readiness": 0.25,
                },
                ("long", "forager_volume_drop_pct"): 0.1,
                ("short", "forager_volume_drop_pct"): 0.1,
                ("long", "wallet_exposure_limit"): 0.2,
                ("short", "wallet_exposure_limit"): 0.2,
                ("long", "risk_we_excess_allowance_pct"): 0.5,
                ("short", "risk_we_excess_allowance_pct"): 0.5,
                ("long", "risk_we_excess_allowance_mode"): "bounded",
                ("short", "risk_we_excess_allowance_mode"): "bounded",
                ("long", "total_wallet_exposure_limit"): 0.24,
                ("short", "total_wallet_exposure_limit"): 0.0,
                ("long", "ema_span_0"): 10.0,
                ("long", "ema_span_1"): 20.0,
                ("short", "ema_span_0"): 10.0,
                ("short", "ema_span_1"): 20.0,
                ("long", "entry_initial_ema_dist"): 0.01,
                ("short", "entry_initial_ema_dist"): 0.01,
                ("long", "entry_volatility_ema_span_1h"): 24.0,
                ("short", "entry_volatility_ema_span_1h"): 24.0,
                ("long", "unstuck_ema_dist"): 0.02,
                ("short", "unstuck_ema_dist"): 0.02,
                ("long", "unstuck_loss_allowance_pct"): 0.02,
                ("short", "unstuck_loss_allowance_pct"): 0.0,
                ("long", "unstuck_close_pct"): 0.5,
                ("short", "unstuck_close_pct"): 0.5,
                ("long", "unstuck_threshold"): 0.8,
                ("short", "unstuck_threshold"): 0.8,
                ("long", "entry_grid_double_down_factor"): 1.0,
                ("short", "entry_grid_double_down_factor"): 1.0,
                ("long", "entry_grid_spacing_pct"): 0.01,
                ("short", "entry_grid_spacing_pct"): 0.01,
                ("long", "entry_volatility_ema_span_1m"): 60.0,
                ("short", "entry_volatility_ema_span_1m"): 60.0,
                ("long", "entry_weight_volatility_1h"): 0.0,
                ("short", "entry_weight_volatility_1h"): 0.0,
                ("long", "entry_weight_volatility_1m"): 0.0,
                ("short", "entry_weight_volatility_1m"): 0.0,
                ("long", "entry_we_weight"): 0.0,
                ("short", "entry_we_weight"): 0.0,
                ("long", "entry_initial_qty_pct"): 0.1,
                ("short", "entry_initial_qty_pct"): 0.1,
                ("long", "entry_trailing_double_down_factor"): 1.0,
                ("short", "entry_trailing_double_down_factor"): 1.0,
                ("long", "entry_trailing_retracement_pct"): 0.01,
                ("short", "entry_trailing_retracement_pct"): 0.01,
                ("long", "entry_trailing_threshold_pct"): 0.01,
                ("short", "entry_trailing_threshold_pct"): 0.01,
                ("long", "close_grid_qty_pct"): 1.0,
                ("short", "close_grid_qty_pct"): 1.0,
                ("long", "close_trailing_qty_pct"): 1.0,
                ("short", "close_trailing_qty_pct"): 1.0,
                ("long", "close_trailing_retracement_pct"): 0.01,
                ("short", "close_trailing_retracement_pct"): 0.01,
                ("long", "close_trailing_threshold_pct"): 0.01,
                ("short", "close_trailing_threshold_pct"): 0.01,
                ("long", "close_weight_volatility_1h"): 0.0,
                ("short", "close_weight_volatility_1h"): 0.0,
                ("long", "close_weight_volatility_1m"): 0.0,
                ("short", "close_weight_volatility_1m"): 0.0,
                ("long", "risk_wel_enforcer_threshold"): 0.0,
                ("short", "risk_wel_enforcer_threshold"): 0.0,
            }
            span2 = float((10.0 * 20.0) ** 0.5)
            self._monitor_runtime_market_hints = {}
            self._monitor_runtime_unstuck_hints = {}
            self._update_monitor_runtime_hints(
                symbols=["BTC/USDT:USDT"],
                last_prices={"BTC/USDT:USDT": 100500.0},
                m1_close_emas={
                    "BTC/USDT:USDT": {
                        10.0: 100200.0,
                        20.0: 100600.0,
                        span2: 100400.0,
                    }
                },
                m1_log_range_emas={"BTC/USDT:USDT": {60.0: 0.0}},
                h1_log_range_emas={"BTC/USDT:USDT": {24.0: 0.0}},
                idx_to_symbol={0: "BTC/USDT:USDT"},
                orders=[
                    {
                        "symbol_idx": 0,
                        "order_type": "close_unstuck_long",
                        "price": 101000.0,
                    }
                ],
            )

        def get_raw_balance(self):
            return 1000.0

        def get_hysteresis_snapped_balance(self):
            return 995.0

        def _equity_hard_stop_realized_pnl_now(self):
            return 12.5

        def _equity_hard_stop_enabled(self, pside):
            return pside == "long"

        def _hsl_state(self, pside):
            return {
                "halted": False,
                "no_restart_latched": False,
                "last_metrics": {"tier": "green"},
            }

        def has_position(self, pside=None, symbol=None):
            if pside is None:
                return any(self.has_position(side, symbol) for side in ("long", "short"))
            if symbol is None:
                return any(self.has_position(pside, sym) for sym in self.positions)
            return abs(float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0))) > 0.0

        def get_current_n_positions(self, pside):
            return sum(1 for sym in self.positions if self.has_position(pside, sym))

        def get_max_n_positions(self, pside):
            return 5 if pside == "long" else 0

        def is_pside_enabled(self, pside):
            return pside == "long"

        def is_forager_mode(self, pside=None):
            if pside is None:
                return True
            return pside == "long"

        def live_value(self, key):
            if key == "forced_mode_long":
                return ""
            if key == "forced_mode_short":
                return ""
            raise KeyError(key)

        def bot_value(self, pside, key):
            return self._coin_bot_values[(pside, key)]

        def bp(self, pside, key, symbol=None):
            return self._coin_bot_values[(pside, key)]

        def has_open_unstuck_order(self):
            return True

        def _calc_unstuck_allowance_for_logging(self, pside):
            if pside == "long":
                return {
                    "status": "ok",
                    "allowance": -20.0,
                    "peak": 1100.0,
                    "pct_from_peak": -9.1,
                    "loss_allowance_pct": 0.02,
                    "override_loss_allowance_pcts": {"BTC/USDT:USDT": 0.005},
                    "override_allowances": {"BTC/USDT:USDT": -44.75},
                }
            return {"status": "disabled"}

        def _calc_unstuck_allowances_live(self, allow_new_unstuck):
            return {"long": 0.0 if not allow_new_unstuck else 1.0, "short": 0.0}

        async def build_forager_candidate_payload(
            self,
            pside,
            symbols,
            min_cost_flags,
            *,
            max_age_ms,
            max_network_fetches,
        ):
            assert pside == "long"
            assert max_age_ms == 60_000
            assert max_network_fetches == 0
            payloads = []
            for symbol in symbols:
                if symbol == "BTC/USDT:USDT":
                    payloads.append(
                        {
                            "enabled": min_cost_flags[symbol],
                            "volume_score": 100.0,
                            "volatility_score": 0.03,
                            "bid": 100500.0,
                            "ask": 100500.0,
                            "ema_lower": 100200.0,
                            "ema_upper": 100600.0,
                            "entry_initial_ema_dist": 0.01,
                        }
                    )
                else:
                    payloads.append(
                        {
                            "enabled": min_cost_flags[symbol],
                            "volume_score": 120.0,
                            "volatility_score": 0.05,
                            "bid": 2500.0,
                            "ask": 2500.0,
                            "ema_lower": 2550.0,
                            "ema_upper": 2600.0,
                            "entry_initial_ema_dist": 0.01,
                        }
                    )
            return payloads

    bot = FakeBot()

    snapshot = await bot._build_monitor_snapshot(now_ms=300000)

    assert "market" in snapshot
    assert "forager" in snapshot
    assert "trailing" in snapshot
    assert "unstuck" in snapshot
    assert "recent" in snapshot
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["last_price"] == pytest.approx(100500.0)
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["wallet_exposure"] == pytest.approx(0.1)
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["wel_ratio"] == pytest.approx(0.5)
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["wele_ratio"] == pytest.approx(
        0.1 / 0.24
    )
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["twel_ratio"] == pytest.approx(
        0.1 / 0.24
    )
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["price_action_distance"] == pytest.approx(
        -0.005
    )
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["upnl"] == pytest.approx(0.5)
    assert snapshot["market"]["BTC/USDT:USDT"]["last_price"] == pytest.approx(100500.0)
    assert snapshot["market"]["BTC/USDT:USDT"]["c_mult"] == pytest.approx(1.0)
    assert snapshot["market"]["BTC/USDT:USDT"]["entry_volatility_logrange_ema"]["long"] == pytest.approx(
        0.0
    )
    assert snapshot["market"]["BTC/USDT:USDT"]["ema_bands"]["long"]["lower"] == pytest.approx(100200.0)
    assert snapshot["market"]["BTC/USDT:USDT"]["ema_bands"]["long"]["upper"] == pytest.approx(100600.0)
    assert snapshot["market"]["BTC/USDT:USDT"]["trailing"]["long"]["max_since_open"] == pytest.approx(
        100900.0
    )
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["entry"]["order_type"] == "entry_trailing_normal_long"
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["entry"]["limit_cap"] == pytest.approx(
        0.24
    )
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["entry"]["threshold_met"] is False
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["entry"]["retracement_met"] is True
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["close"]["order_type"] == "close_trailing_long"
    assert snapshot["trailing"]["BTC/USDT:USDT"]["long"]["close"]["threshold_met"] is False
    assert snapshot["forager"]["long"]["forager_mode"] is True
    assert snapshot["forager"]["long"]["selected_symbols"] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert snapshot["forager"]["long"]["next_symbol"] == "ETH/USDT:USDT"
    assert snapshot["forager"]["long"]["next_entry_trigger_price"] == pytest.approx(2550.0 * 0.99)
    assert snapshot["forager"]["long"]["next_entry_distance_ratio"] == pytest.approx(
        2500.0 / (2550.0 * 0.99) - 1.0
    )
    assert snapshot["forager"]["long"]["ranking"]["top_volume"]["symbol"] == "ETH/USDT:USDT"
    assert snapshot["forager"]["long"]["ranking"]["top_volatility"]["symbol"] == "ETH/USDT:USDT"
    assert snapshot["forager"]["long"]["ranking"]["top_ema_readiness"]["symbol"] == "ETH/USDT:USDT"
    assert snapshot["unstuck"]["has_open_order"] is True
    assert snapshot["unstuck"]["sides"]["long"]["allowance"] == pytest.approx(-20.0)
    assert snapshot["unstuck"]["sides"]["long"]["override_loss_allowance_pcts"] == {
        "BTC/USDT:USDT": pytest.approx(0.005)
    }
    assert snapshot["unstuck"]["sides"]["long"]["override_allowances"] == {
        "BTC/USDT:USDT": pytest.approx(-44.75)
    }
    assert snapshot["unstuck"]["sides"]["long"]["next_symbol"] == "BTC/USDT:USDT"
    assert snapshot["unstuck"]["sides"]["long"]["next_target_price"] == pytest.approx(101000.0)
    assert snapshot["unstuck"]["sides"]["long"]["next_target_distance_ratio"] == pytest.approx(
        101000.0 / 100500.0 - 1.0
    )
    assert snapshot["unstuck"]["sides"]["long"]["next_unstuck_trigger_distance_ratio"] == pytest.approx(
        (100600.0 * 1.02) / 100500.0 - 1.0
    )
    assert snapshot["recent"]["order_executions"][0]["execution_timestamp"] == 123456
    assert snapshot["recent"]["order_cancellations"][0]["pb_order_type"] == "close_unstuck_long"


def test_monitor_trailing_section_marks_trailing_grid_v7_diagnostics_unsupported():
    import passivbot as pb_mod

    class FakeBot:
        _build_monitor_trailing_section = pb_mod.Passivbot._build_monitor_trailing_section

        def __init__(self):
            self.config = {"live": {"strategy_kind": "trailing_grid_v7"}}

    bot = FakeBot()

    payload = bot._build_monitor_trailing_section(
        balance_raw=1000.0,
        market={
            "BTC/USDT:USDT": {
                "last_price": 100.0,
                "trailing": {
                    "long": {
                        "min_since_open": 90.0,
                        "max_since_min": 95.0,
                    }
                },
            }
        },
    )

    assert payload == {
        "_meta": {
            "diagnostics_supported": False,
            "strategy_kind": "trailing_grid_v7",
            "reason": "monitor trailing diagnostics use trailing_martingale helper formulas",
        }
    }


@pytest.mark.asyncio
async def test_update_positions_and_balance_cancels_balance_task_when_positions_fail():
    import passivbot as pb_mod

    class FakeBot:
        update_positions_and_balance = pb_mod.Passivbot.update_positions_and_balance

        def __init__(self):
            self.balance_cancelled = False

        async def update_balance(self):
            try:
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                self.balance_cancelled = True
                raise

        async def _fetch_and_apply_positions(self):
            raise RuntimeError("positions failed")

        async def log_position_changes(self, fetched_positions_old, fetched_positions_new):
            raise AssertionError("should not be called")

        async def handle_balance_update(self, source="REST"):
            raise AssertionError("should not be called")

    bot = FakeBot()

    with pytest.raises(RuntimeError, match="positions failed"):
        await bot.update_positions_and_balance()

    assert bot.balance_cancelled is True


@pytest.mark.asyncio
async def test_update_pos_oos_pnls_ohlcvs_propagates_open_order_failure():
    import passivbot as pb_mod

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.stop_signal_received = False
    bot.exchange = "bybit"
    bot.config = {"live": {}, "_raw_effective": {"live": {}}}
    bot.refresh_authoritative_state = AsyncMock(
        side_effect=RuntimeError("open orders failed")
    )
    bot.update_ohlcvs_1m_for_actives = AsyncMock()

    with pytest.raises(RuntimeError, match="open orders failed"):
        await bot.update_pos_oos_pnls_ohlcvs()

    bot.update_ohlcvs_1m_for_actives.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_open_orders_propagates_unexpected_fetch_errors():
    import passivbot as pb_mod

    bot = pb_mod.Passivbot.__new__(pb_mod.Passivbot)
    bot.stop_signal_received = False
    bot.open_orders = {}

    async def fake_fetch_open_orders():
        raise RuntimeError("exchange fetch broke")

    bot.fetch_open_orders = fake_fetch_open_orders

    with pytest.raises(RuntimeError, match="exchange fetch broke"):
        await bot.update_open_orders()
