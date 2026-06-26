import json
import logging
import queue
import re
import threading
import time

import pytest

from live.event_bus import (
    DEFAULT_ROUTES,
    EventRoute,
    EventTags,
    EventTypes,
    ListEventSink,
    LiveEvent,
    LiveEventContext,
    LiveEventPipeline,
    MonitorEventSink,
    REDACTED,
    ReasonCodes,
    authoritative_reason_code,
    format_console_event,
    normalize_event_type,
    payload_hash,
    payload_hash_raw,
    redact_payload,
    sink_failed_reason_code,
)
from live.events import DiagnosticEvent
from monitor_publisher import MonitorPublisher


def _make_monitor_publisher(tmp_path, **overrides):
    params = {
        "exchange": "bybit",
        "user": "user01",
        "root_dir": str(tmp_path),
        "snapshot_interval_seconds": 1.0,
        "checkpoint_interval_minutes": 10.0,
        "event_rotation_mb": 128.0,
        "event_rotation_minutes": 60.0,
        "retain_days": 7.0,
        "max_total_bytes": 1_000_000,
        "compress_rotated_segments": False,
        "retain_price_ticks": True,
        "retain_candles": True,
        "retain_fills": True,
        "price_tick_min_interval_ms": 500,
        "emit_completed_candles": True,
        "include_raw_fill_payloads": False,
    }
    params.update(overrides)
    return MonitorPublisher(**params)


def _registry_values(registry: type) -> list[str]:
    return [
        value
        for name, value in vars(registry).items()
        if name.isupper() and isinstance(value, str)
    ]


def test_live_event_tag_registry_values_are_unique_and_query_safe():
    values = _registry_values(EventTags)

    assert EventTags.REMOTE_CALL == "remote_call"
    assert EventTags.CANDLE == "candle"
    assert EventTags.EMA == "ema"
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in values)


def test_live_event_reason_code_registry_values_are_unique_and_query_safe():
    values = _registry_values(ReasonCodes)

    assert ReasonCodes.REQUIRED_EMA_UNAVAILABLE == "required_ema_unavailable"
    assert ReasonCodes.EXCHANGE_ACKNOWLEDGED == "exchange_acknowledged"
    assert ReasonCodes.EXECUTION_LOOP_ERROR_BURST == "execution_loop_error_burst"
    assert ReasonCodes.WARMUP_CACHE_DECISION == "warmup_cache_decision"
    assert authoritative_reason_code("balance") == "authoritative_balance"
    assert sink_failed_reason_code("monitor") == "monitor_sink_failed"
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in values)


def test_live_event_serializes_stable_envelope_and_redacts_sensitive_data():
    event = LiveEvent(
        EventTypes.RUST_ORCHESTRATOR_CALLED,
        level="debug",
        tags=("planning", "rust"),
        exchange="binance",
        user="user01",
        cycle_id="cy_1",
        status="started",
        data={
            "symbol_count": 3,
            "apiKey": "secret",
            "nested": {"password": "pw", "safe": "ok"},
            "rows": [{"signature": "sig", "value": 1}],
        },
    )

    data = event.to_dict()
    assert data["schema_version"] == 1
    assert data["event_type"] == EventTypes.RUST_ORCHESTRATOR_CALLED
    assert data["level"] == "debug"
    assert data["event_id"]
    assert data["ts_ms"] > 0
    assert data["monotonic_ms"] > 0
    assert data["tags"] == ["planning", "rust"]
    assert data["data"]["apiKey"] == REDACTED
    assert data["data"]["nested"]["password"] == REDACTED
    assert data["data"]["nested"]["safe"] == "ok"
    assert data["data"]["rows"][0]["signature"] == REDACTED
    assert json.loads(event.to_json())["event_type"] == event.event_type


def test_live_event_context_propagates_ids_without_overwriting_event_values():
    context = LiveEventContext(
        exchange="gateio",
        user="gateio_01",
        bot_id="bot_a",
        cycle_id="cy_a",
    )

    event = LiveEvent(
        EventTypes.SNAPSHOT_BUILT,
        exchange="kucoin",
        snapshot_id="snap_1",
    ).with_context(context)

    assert event.exchange == "kucoin"
    assert event.user == "gateio_01"
    assert event.bot_id == "bot_a"
    assert event.cycle_id == "cy_a"
    assert event.snapshot_id == "snap_1"

    next_context = context.with_ids(plan_id="plan_1")
    assert next_context.plan_id == "plan_1"
    assert context.plan_id is None
    with pytest.raises(KeyError):
        context.with_ids(unknown_id="x")


def test_pipeline_context_ids_can_be_updated_between_cycles():
    sink = ListEventSink()
    pipeline = LiveEventPipeline(
        context=LiveEventContext(exchange="gateio", user="gateio_01"),
        structured_sinks=[sink],
        routes={EventTypes.CYCLE_STARTED: EventRoute(structured=True, monitor=False)},
    )

    pipeline.with_context_ids(cycle_id="cy_1")
    first = pipeline.emit(LiveEvent(EventTypes.CYCLE_STARTED))
    pipeline.with_context_ids(cycle_id="cy_2", order_wave_id="ow_1")
    second = pipeline.emit(LiveEvent(EventTypes.CYCLE_STARTED))

    assert first.cycle_id == "cy_1"
    assert second.cycle_id == "cy_2"
    assert second.order_wave_id == "ow_1"
    assert pipeline.flush(timeout=2.0) is True
    assert [event.cycle_id for event in sink.events] == ["cy_1", "cy_2"]
    assert pipeline.close(timeout=2.0) is True


def test_route_table_keeps_data_events_off_console_by_default():
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].structured is True
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].monitor is True
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].console is False
    for event_type in (
        EventTypes.FORAGER_SELECTION,
        EventTypes.FORAGER_FEATURE_UNAVAILABLE,
        EventTypes.ACTION_PLANNED,
        EventTypes.EMA_BUNDLE_STARTED,
        EventTypes.EMA_BUNDLE_COMPLETED,
        EventTypes.EMA_FALLBACK_USED,
        EventTypes.EMA_UNAVAILABLE,
        EventTypes.CANDLE_COVERAGE_CHECKED,
        EventTypes.CANDLE_TAIL_PROJECTED,
        EventTypes.CACHE_LOAD_COMPLETED,
        EventTypes.CACHE_FLUSH_COMPLETED,
        EventTypes.CACHE_WARMUP_DECISION,
        EventTypes.FILLS_REFRESH_SUMMARY,
        EventTypes.BOT_STARTUP_TIMING,
        EventTypes.HEALTH_SUMMARY,
        EventTypes.PLANNING_DEFER_SUMMARY,
        EventTypes.PLANNING_SYMBOL_STATE,
        EventTypes.RISK_MODE_CHANGED,
        EventTypes.HSL_TRANSITION,
        EventTypes.HSL_STATUS,
        EventTypes.HSL_RED_TRIGGERED,
        EventTypes.HSL_COOLDOWN_STARTED,
        EventTypes.HSL_COOLDOWN_ENDED,
    ):
        assert DEFAULT_ROUTES[event_type].structured is True
        assert DEFAULT_ROUTES[event_type].monitor is True
        assert DEFAULT_ROUTES[event_type].console is False
        assert DEFAULT_ROUTES[event_type].text is False
    assert DEFAULT_ROUTES[EventTypes.ORDER_WAVE_COMPLETED].console is True
    assert DEFAULT_ROUTES[EventTypes.ORDER_WAVE_COMPLETED].text is True
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CREATE_DEFERRED].console is False
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CREATE_SKIPPED].console is False
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CONFIRMATION_TIMEOUT].console is True
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CONFIRMATION_TIMEOUT].text is True
    assert DEFAULT_ROUTES[EventTypes.BOT_SHUTDOWN_STAGE].console is True
    assert DEFAULT_ROUTES[EventTypes.BOT_SHUTDOWN_STAGE].text is True


def test_redact_payload_recurses_and_payload_hash_is_stable():
    payload = {"safe": 1, "auth": {"token": "secret"}, "items": [{"secret": "x"}]}

    assert redact_payload(payload) == {
        "safe": 1,
        "auth": REDACTED,
        "items": [{"secret": REDACTED}],
    }
    assert payload_hash({"b": 2, "a": 1}) == payload_hash({"a": 1, "b": 2})


def test_payload_hash_raw_hashes_exact_wire_payload():
    assert payload_hash_raw('{"b":2,"a":1}') != payload_hash_raw('{"a":1,"b":2}')
    assert payload_hash_raw("abc") == payload_hash_raw(b"abc")


def test_legacy_event_type_names_normalize_to_phase1_schema():
    assert normalize_event_type("planning_unavailable") == EventTypes.PLANNING_UNAVAILABLE
    assert LiveEvent("planning_unavailable").event_type == EventTypes.PLANNING_UNAVAILABLE


def test_pipeline_routes_console_and_async_sinks_with_context():
    structured = ListEventSink()
    monitor = ListEventSink()
    console = ListEventSink()
    pipeline = LiveEventPipeline(
        context=LiveEventContext(exchange="okx", user="okx_01", cycle_id="cy_1"),
        structured_sinks=[structured],
        monitor_sinks=[monitor],
        console_sink=console,
    )

    event = pipeline.emit(
        LiveEvent(
            EventTypes.ORDER_WAVE_COMPLETED,
            status="succeeded",
            message="create=1 cancel=0",
        )
    )

    assert event.exchange == "okx"
    assert event.user == "okx_01"
    assert event.cycle_id == "cy_1"
    assert console.events == [event]
    assert pipeline.flush(timeout=2.0) is True
    assert structured.events == [event]
    assert monitor.events == [event]
    assert pipeline.close(timeout=2.0) is True


def test_pipeline_queue_overflow_is_observable_without_raising():
    pipeline = LiveEventPipeline(
        queue_maxsize=1,
        start=False,
        routes={
            EventTypes.DATA_PACKET_UPDATED: EventRoute(
                structured=True, monitor=False, console=False
            )
        },
    )

    first = pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED))
    second = pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED))

    assert first.event_type == EventTypes.DATA_PACKET_UPDATED
    assert second.event_type == EventTypes.DATA_PACKET_UPDATED
    assert pipeline.drop_counters[EventTypes.DATA_PACKET_UPDATED] == 1
    assert pipeline.degraded_events[-1].event_type == EventTypes.SINK_DEGRADED
    assert pipeline.degraded_events[-1].reason_code == "queue_full"


def test_pipeline_health_snapshot_reports_queue_and_degraded_counters():
    pipeline = LiveEventPipeline(
        queue_maxsize=1,
        start=False,
        routes={
            EventTypes.DATA_PACKET_UPDATED: EventRoute(
                structured=True, monitor=False, console=False
            )
        },
    )

    assert pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED)) is not None
    assert (
        pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED), require_enqueue=True)
        is None
    )

    snapshot = pipeline.health_snapshot()

    assert snapshot["event_queue_depth"] == 1
    assert snapshot["event_queue_maxsize"] == 1
    assert snapshot["event_queue_unfinished_tasks"] == 1
    assert snapshot["event_dropped_total"] == 1
    assert snapshot["event_drop_counts"] == {EventTypes.DATA_PACKET_UPDATED: 1}
    assert snapshot["event_sink_error_total"] == 0
    assert snapshot["event_sink_error_counts"] == {}
    assert snapshot["event_degraded_count"] == 1
    assert snapshot["event_pipeline_stopping"] is False
    assert snapshot["event_pipeline_worker_alive"] is False

    pipeline._queue.get_nowait()
    pipeline._queue.task_done()


def test_pipeline_queue_overflow_is_logged_and_monitor_visible(caplog):
    monitor = ListEventSink()
    pipeline = LiveEventPipeline(
        queue_maxsize=1,
        start=False,
        monitor_sinks=[monitor],
        routes={
            EventTypes.DATA_PACKET_UPDATED: EventRoute(
                structured=False, monitor=True, console=False
            )
        },
    )

    with caplog.at_level(logging.WARNING):
        assert pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED)) is not None
        assert (
            pipeline.emit(
                LiveEvent(EventTypes.DATA_PACKET_UPDATED),
                require_enqueue=True,
            )
            is None
        )

    assert pipeline.drop_counters[EventTypes.DATA_PACKET_UPDATED] == 1
    assert monitor.events[-1].event_type == EventTypes.SINK_DEGRADED
    assert monitor.events[-1].reason_code == "queue_full"
    assert any("live event queue full" in record.message for record in caplog.records)


def test_sink_failure_degrades_observability_without_raising():
    class FailingSink:
        def write(self, event):
            raise OSError("disk full")

    monitor = ListEventSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[FailingSink()],
        monitor_sinks=[monitor],
        routes={EventTypes.SNAPSHOT_BUILT: EventRoute(structured=True, monitor=False)},
    )

    event = pipeline.emit(LiveEvent(EventTypes.SNAPSHOT_BUILT))

    assert event.event_type == EventTypes.SNAPSHOT_BUILT
    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.sink_error_counters["structured"] == 1
    assert pipeline.degraded_events[-1].reason_code == "structured_sink_failed"
    assert monitor.events[-1].event_type == EventTypes.SINK_DEGRADED
    assert monitor.events[-1].reason_code == "structured_sink_failed"
    assert pipeline.close(timeout=2.0) is True


def test_monitor_sink_none_ack_records_monitor_sink_failure():
    class NonePublisher:
        def record_event(self, *args, **kwargs):
            return None

    pipeline = LiveEventPipeline(
        monitor_sinks=[MonitorEventSink(NonePublisher())],
        routes={EventTypes.SNAPSHOT_BUILT: EventRoute(structured=False, monitor=True)},
    )

    event = pipeline.emit(LiveEvent(EventTypes.SNAPSHOT_BUILT))

    assert event.event_type == EventTypes.SNAPSHOT_BUILT
    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.sink_error_counters["monitor"] == 1
    assert pipeline.degraded_events[-1].reason_code == "monitor_sink_failed"
    assert pipeline.close(timeout=2.0) is True


def test_pipeline_close_drains_queued_events_after_close_starts():
    class SlowSink:
        def __init__(self):
            self.events = []

        def write(self, event):
            time.sleep(0.02)
            self.events.append(event)
            return event

    sink = SlowSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        routes={EventTypes.DATA_PACKET_UPDATED: EventRoute(structured=True, monitor=False)},
    )

    for _ in range(5):
        pipeline.emit(LiveEvent(EventTypes.DATA_PACKET_UPDATED))

    assert pipeline.close(timeout=2.0) is True
    assert [event.event_type for event in sink.events] == [
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.DATA_PACKET_UPDATED,
    ]


def test_pipeline_rejects_new_async_events_after_close_begins():
    pipeline = LiveEventPipeline(
        start=False,
        routes={EventTypes.DATA_PACKET_UPDATED: EventRoute(structured=True, monitor=False)},
    )

    closer = threading.Thread(target=lambda: pipeline.close(timeout=0.1))
    closer.start()
    closer.join(timeout=1.0)

    assert (
        pipeline.emit(
            LiveEvent(EventTypes.DATA_PACKET_UPDATED),
            require_enqueue=True,
        )
        is None
    )
    assert pipeline.degraded_events[-1].reason_code == "pipeline_closing"


def test_pipeline_close_waits_for_in_flight_enqueue_before_sentinel():
    class BlockingQueue(queue.Queue):
        def __init__(self):
            super().__init__(maxsize=10)
            self.entered_put_nowait = threading.Event()
            self.release_put_nowait = threading.Event()

        def put_nowait(self, item):
            if isinstance(item, LiveEvent):
                self.entered_put_nowait.set()
                assert self.release_put_nowait.wait(timeout=2.0)
            return super().put_nowait(item)

    sink = ListEventSink()
    blocking_queue = BlockingQueue()
    pipeline = LiveEventPipeline(
        start=False,
        structured_sinks=[sink],
        routes={EventTypes.DATA_PACKET_UPDATED: EventRoute(structured=True, monitor=False)},
    )
    pipeline._queue = blocking_queue
    pipeline.start()

    emit_result = {}
    close_result = {}

    def emit_event():
        emit_result["event"] = pipeline.emit(
            LiveEvent(EventTypes.DATA_PACKET_UPDATED),
            require_enqueue=True,
        )

    emitter = threading.Thread(target=emit_event)
    emitter.start()
    assert blocking_queue.entered_put_nowait.wait(timeout=2.0)

    closer = threading.Thread(target=lambda: close_result.update(ok=pipeline.close(timeout=2.0)))
    closer.start()
    time.sleep(0.05)
    blocking_queue.release_put_nowait.set()

    emitter.join(timeout=2.0)
    closer.join(timeout=2.0)

    assert close_result["ok"] is True
    assert emit_result["event"].event_type == EventTypes.DATA_PACKET_UPDATED
    assert [event.event_type for event in sink.events] == [EventTypes.DATA_PACKET_UPDATED]
    assert blocking_queue.unfinished_tasks == 0


def test_monitor_sink_preserves_current_monitor_record_event_contract():
    class Publisher:
        def __init__(self):
            self.calls = []

        def record_event(self, kind, tags, payload, *, ts=None, symbol=None, pside=None):
            self.calls.append(
                {
                    "kind": kind,
                    "tags": tags,
                    "payload": payload,
                    "ts": ts,
                    "symbol": symbol,
                    "pside": pside,
                }
            )
            return self.calls[-1]

    publisher = Publisher()
    event = LiveEvent(
        EventTypes.PLANNING_UNAVAILABLE,
        tags=("planning", "gate"),
        exchange="binance",
        user="binance_01",
        bot_id="bot_1",
        symbol="BTC/USDT:USDT",
        pside="long",
        side="buy",
        order_id="order-1",
        client_order_id="client-1",
        data={"reason": "stale_ema"},
        ts_ms=12345,
    )

    result = MonitorEventSink(publisher).write(event)

    assert result["kind"] == EventTypes.PLANNING_UNAVAILABLE
    assert result["tags"] == ("planning", "gate")
    assert result["symbol"] == "BTC/USDT:USDT"
    assert result["pside"] == "long"
    assert result["ts"] == 12345
    assert result["payload"]["reason"] == "stale_ema"
    live_event = result["payload"]["_live_event"]
    assert live_event["event_type"] == EventTypes.PLANNING_UNAVAILABLE
    assert live_event["exchange"] == "binance"
    assert live_event["user"] == "binance_01"
    assert live_event["bot_id"] == "bot_1"
    assert live_event["symbol"] == "BTC/USDT:USDT"
    assert live_event["pside"] == "long"
    assert live_event["side"] == "buy"
    assert live_event["order_id"] == "order-1"
    assert live_event["client_order_id"] == "client-1"
    assert live_event["data"] == {"reason": "stale_ema"}


def test_diagnostic_event_uses_pipeline_when_available_and_legacy_when_absent():
    class BotWithPipeline:
        exchange = "binance"
        bot_id = "bot_1"

        def __init__(self):
            self._live_event_pipeline = LiveEventPipeline(
                context=LiveEventContext(user="binance_01"),
                structured_sinks=[],
                monitor_sinks=[],
            )

        def config_get(self, keys):
            return "binance_01"

    bot = BotWithPipeline()
    event = DiagnosticEvent.build(
        "planning_unavailable",
        ("planning",),
        {"apiKey": "secret", "reason": "stale"},
        ts_ms=1000,
        symbol="ETH/USDT:USDT",
        pside="long",
    )

    emitted = event.emit(bot)

    assert emitted.event_type == EventTypes.PLANNING_UNAVAILABLE
    assert emitted.exchange == "binance"
    assert emitted.user == "binance_01"
    assert emitted.bot_id == "bot_1"
    assert emitted.data["apiKey"] == REDACTED
    assert emitted.data["reason"] == "stale"
    assert bot._live_event_pipeline.close(timeout=2.0) is True

    class LegacyBot:
        def __init__(self):
            self.calls = []

        def _monitor_record_event(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "legacy"

    legacy = LegacyBot()
    assert event.emit(legacy) == "legacy"
    assert legacy.calls[0][0][:3] == (
        "planning_unavailable",
        ("planning",),
        {"apiKey": "secret", "reason": "stale"},
    )


def test_diagnostic_event_falls_back_to_legacy_recorder_when_pipeline_queue_is_full():
    class BotWithFullPipeline:
        exchange = "binance"
        bot_id = "bot_1"

        def __init__(self):
            self.calls = []
            self._live_event_pipeline = LiveEventPipeline(
                queue_maxsize=1,
                start=False,
                structured_sinks=[],
                monitor_sinks=[],
                routes={
                    EventTypes.PLANNING_UNAVAILABLE: EventRoute(
                        structured=True, monitor=False, console=False
                    )
                },
            )

        def config_get(self, keys):
            return "binance_01"

        def _monitor_record_event(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "legacy"

    bot = BotWithFullPipeline()
    first = DiagnosticEvent.build("planning_unavailable", ("planning",), {"n": 1})
    second = DiagnosticEvent.build("planning_unavailable", ("planning",), {"n": 2})

    assert first.emit(bot).event_type == EventTypes.PLANNING_UNAVAILABLE
    assert second.emit(bot) == "legacy"
    assert bot.calls[0][0][:3] == ("planning_unavailable", ("planning",), {"n": 2})


def test_console_format_is_compact_and_operator_facing():
    event = LiveEvent(
        EventTypes.PLANNING_UNAVAILABLE,
        status="deferred",
        cycle_id="cy_1",
        symbol="BTC/USDT:USDT",
        pside="long",
        reason_code="stale_ema",
        message="entries deferred",
    )

    expected = (
        "[gate] deferred cycle=cy_1 symbol=BTC/USDT:USDT "
        "pside=long reason=stale_ema entries deferred"
    )
    assert format_console_event(event) == expected


def test_console_format_summarizes_order_wave_payload():
    event = LiveEvent(
        EventTypes.ORDER_WAVE_COMPLETED,
        status="deferred",
        cycle_id="cy_9",
        order_wave_id="ow_7",
        reason_code="create_deferred",
        data={
            "planned_cancel": 1,
            "cancel_posted": 1,
            "planned_create": 3,
            "create_posted": 2,
            "deferred_create": 1,
            "elapsed_ms": 642,
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
        },
    )

    assert format_console_event(event) == (
        "[execute] deferred cycle=cy_9 wave=ow_7 cancel=1/1 create=2/3 "
        "deferred_create=1 elapsed=642ms "
        "symbols=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT "
        "reason=create_deferred"
    )


def test_console_format_summarizes_order_write_without_raw_payload():
    event = LiveEvent(
        EventTypes.EXECUTION_CREATE_SUCCEEDED,
        status="succeeded",
        cycle_id="cy_3",
        order_wave_id="ow_2",
        symbol="AAVE/USDT:USDT",
        pside="long",
        side="buy",
        data={
            "order_type": "limit",
            "qty": 1.23456789,
            "price": 88.7654321,
            "reduce_only": False,
            "result_status": "open",
            "result_order_id_short": "abc123",
            "result_client_order_id_short": "pbot_456",
            "apiKey": "must_not_leak",
        },
    )

    assert format_console_event(event) == (
        "[order] succeeded cycle=cy_3 wave=ow_2 side=buy type=limit "
        "qty=1.23456789 price=88.7654321 exchange_status=open "
        "order_id=abc123 client_id=pbot_456 symbol=AAVE/USDT:USDT pside=long"
    )


def test_console_format_summarizes_create_filter_payload():
    event = LiveEvent(
        EventTypes.EXECUTION_CREATE_SKIPPED,
        status="skipped",
        cycle_id="cy_8",
        order_wave_id="ow_4",
        reason_code=ReasonCodes.PENDING_EXCHANGE_CONFIG,
        message="create orders skipped while exchange config update is pending",
        data={
            "order_count": 2,
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        },
    )

    assert format_console_event(event) == (
        "[gate] skipped cycle=cy_8 wave=ow_4 orders=2 "
        "symbols=BTC/USDT:USDT,ETH/USDT:USDT "
        "reason=pending_exchange_config "
        "create orders skipped while exchange config update is pending"
    )


def test_console_format_summarizes_confirmation_timeout():
    event = LiveEvent(
        EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
        status="degraded",
        cycle_id="cy_5",
        order_wave_id="ow_3",
        reason_code="authoritative_confirmation_timeout",
        data={
            "fresh_surfaces": ["balance", "positions"],
            "pending_surfaces": ["open_orders"],
            "elapsed_ms": 2200,
            "timeout_ms": 2000,
        },
    )

    assert format_console_event(event) == (
        "[execute] degraded cycle=cy_5 wave=ow_3 fresh=balance,positions "
        "pending=open_orders elapsed=2200ms timeout=2000ms "
        "reason=authoritative_confirmation_timeout"
    )


def test_console_format_summarizes_rust_return():
    event = LiveEvent(
        EventTypes.RUST_ORCHESTRATOR_RETURNED,
        status="succeeded",
        cycle_id="cy_6",
        data={"order_count": 4, "elapsed_ms": 17},
    )

    assert format_console_event(event) == "[rust] succeeded cycle=cy_6 orders=4 elapsed=17ms"


def test_shutdown_stage_console_format_uses_shutdown_tag():
    event = LiveEvent(
        EventTypes.BOT_SHUTDOWN_STAGE,
        status="started",
        reason_code="maintainers_stopping",
        message="waiting for maintainer tasks task_count=2 elapsed=0.01s",
    )

    assert (
        format_console_event(event)
        == "[shutdown] started reason=maintainers_stopping waiting for maintainer "
        "tasks task_count=2 elapsed=0.01s"
    )


def test_route_throttle_applies_only_to_console_and_text_sinks():
    structured = ListEventSink()
    console = ListEventSink()
    text = ListEventSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[structured],
        monitor_sinks=[],
        console_sink=console,
        text_sink=text,
        routes={
            EventTypes.CYCLE_STARTED: EventRoute(
                structured=True,
                monitor=False,
                console=True,
                text=True,
                throttle_interval_ms=60_000,
            )
        },
    )

    for ts_ms in (1_000, 2_000, 61_000):
        pipeline.emit(
            LiveEvent(
                EventTypes.CYCLE_STARTED,
                status="started",
                cycle_id=f"cy_{ts_ms}",
                ts_ms=ts_ms,
            )
        )

    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.close(timeout=2.0) is True
    assert [event.cycle_id for event in structured.events] == [
        "cy_1000",
        "cy_2000",
        "cy_61000",
    ]
    assert [event.cycle_id for event in console.events] == ["cy_1000", "cy_61000"]
    assert [event.cycle_id for event in text.events] == ["cy_1000", "cy_61000"]


def test_monitor_event_sink_writes_real_monitor_event_stream(tmp_path):
    publisher = _make_monitor_publisher(tmp_path)
    pipeline = LiveEventPipeline(
        context=LiveEventContext(exchange="bybit", user="user01", cycle_id="cy_1"),
        structured_sinks=[],
        monitor_sinks=[MonitorEventSink(publisher)],
    )

    event = pipeline.emit(
        LiveEvent(
            "planning_unavailable",
            tags=("planning", "ema"),
            symbol="ETH/USDT:USDT",
            pside="long",
            status="deferred",
            reason_code="stale_ema",
            data={"age_ms": 300_000, "apiKey": "secret"},
            ts_ms=12345,
        )
    )

    assert event.event_type == EventTypes.PLANNING_UNAVAILABLE
    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.close(timeout=2.0) is True

    event_path = tmp_path / "bybit" / "user01" / "events" / "current.ndjson"
    rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["kind"] == EventTypes.PLANNING_UNAVAILABLE
    assert rows[0]["symbol"] == "ETH/USDT:USDT"
    assert rows[0]["pside"] == "long"
    assert rows[0]["payload"]["age_ms"] == 300_000
    assert rows[0]["payload"]["apiKey"] == REDACTED
    assert rows[0]["payload"]["_live_event"]["event_type"] == EventTypes.PLANNING_UNAVAILABLE
    assert rows[0]["payload"]["_live_event"]["exchange"] == "bybit"
    assert rows[0]["payload"]["_live_event"]["user"] == "user01"
    assert rows[0]["payload"]["_live_event"]["symbol"] == "ETH/USDT:USDT"
    assert rows[0]["payload"]["_live_event"]["pside"] == "long"
    assert rows[0]["payload"]["_live_event"]["ids"]["cycle_id"] == "cy_1"
    assert rows[0]["payload"]["_live_event"]["reason_code"] == "stale_ema"
    assert rows[0]["payload"]["_live_event"]["data"]["apiKey"] == REDACTED


def test_cycle_events_are_reconstructable_by_cycle_id():
    structured = ListEventSink()
    pipeline = LiveEventPipeline(
        context=LiveEventContext(exchange="gateio", user="gateio_01", cycle_id="cy_1"),
        structured_sinks=[structured],
        monitor_sinks=[],
    )

    for event_type in (
        EventTypes.CYCLE_STARTED,
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.SNAPSHOT_BUILT,
        EventTypes.RUST_ORCHESTRATOR_CALLED,
        EventTypes.RUST_ORCHESTRATOR_RETURNED,
        EventTypes.ACTION_PLANNED,
        EventTypes.ORDER_WAVE_COMPLETED,
        EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
        EventTypes.CYCLE_COMPLETED,
    ):
        pipeline.emit(LiveEvent(event_type, status="succeeded"))

    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.close(timeout=2.0) is True
    assert [event.event_type for event in structured.events] == [
        EventTypes.CYCLE_STARTED,
        EventTypes.DATA_PACKET_UPDATED,
        EventTypes.SNAPSHOT_BUILT,
        EventTypes.RUST_ORCHESTRATOR_CALLED,
        EventTypes.RUST_ORCHESTRATOR_RETURNED,
        EventTypes.ACTION_PLANNED,
        EventTypes.ORDER_WAVE_COMPLETED,
        EventTypes.EXECUTION_CONFIRMATION_TIMEOUT,
        EventTypes.CYCLE_COMPLETED,
    ]
    assert {event.cycle_id for event in structured.events} == {"cy_1"}
