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
    LIVE_EVENT_DEBUG_PROFILES,
    ListEventSink,
    LiveEvent,
    LiveEventContext,
    LiveEventPipeline,
    MonitorEventSink,
    REDACTED,
    ReasonCodes,
    authoritative_reason_code,
    format_console_event,
    live_event_debug_profile_enabled,
    normalize_live_event_console_enabled,
    normalize_live_event_debug_profiles,
    normalize_event_type,
    payload_hash,
    payload_hash_raw,
    redact_payload,
    resolve_live_event_console_enabled,
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
    assert ReasonCodes.EXCHANGE_CONFIG_REFRESH == "exchange_config_refresh"
    assert ReasonCodes.EXECUTION_LOOP_ERROR_BURST == "execution_loop_error_burst"
    assert ReasonCodes.WARMUP_CACHE_DECISION == "warmup_cache_decision"
    assert authoritative_reason_code("balance") == "authoritative_balance"
    assert sink_failed_reason_code("monitor") == "monitor_sink_failed"
    assert len(values) == len(set(values))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in values)


def test_live_event_debug_profiles_normalize_and_validate():
    assert normalize_live_event_debug_profiles(None) == ()
    assert normalize_live_event_debug_profiles("") == ()
    assert normalize_live_event_debug_profiles(
        "rust,remote-calls candle EMA-readiness warmup"
    ) == (
        "cache",
        "candles",
        "ema",
        "remote_calls",
        "rust",
    )
    assert normalize_live_event_debug_profiles(["rust", "rust", "off", "hsl"]) == (
        "hsl",
        "rust",
    )
    assert normalize_live_event_debug_profiles("all") == tuple(
        sorted(LIVE_EVENT_DEBUG_PROFILES)
    )
    with pytest.raises(ValueError, match="unknown live event debug profile"):
        normalize_live_event_debug_profiles("rust,unknown_profile")

    class Holder:
        live_event_debug_profiles = ("rust",)

    assert live_event_debug_profile_enabled(Holder(), "rust")
    assert live_event_debug_profile_enabled(Holder(), "remote_calls") is False
    assert LiveEventPipeline(debug_profiles="rust", start=False).debug_profiles == (
        "rust",
    )


def test_live_event_console_flag_normalizes_common_config_values():
    assert normalize_live_event_console_enabled(None) is False
    assert normalize_live_event_console_enabled(False) is False
    assert normalize_live_event_console_enabled("") is False
    assert normalize_live_event_console_enabled("off") is False
    assert normalize_live_event_console_enabled("0") is False
    assert normalize_live_event_console_enabled(True) is True
    assert normalize_live_event_console_enabled("on") is True
    assert normalize_live_event_console_enabled("1") is True
    assert normalize_live_event_console_enabled("console") is True


def test_live_event_console_resolver_defaults_on_with_explicit_opt_outs():
    assert resolve_live_event_console_enabled() is True
    assert resolve_live_event_console_enabled(config_value=None, env_value=None) is True
    assert resolve_live_event_console_enabled(config_value=False) is False
    assert resolve_live_event_console_enabled(config_value="off") is False
    assert resolve_live_event_console_enabled(config_value=True) is True
    assert resolve_live_event_console_enabled(config_value=False, env_value="1") is True
    assert resolve_live_event_console_enabled(config_value=True, env_value="0") is False


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
        EventTypes.EXCHANGE_CONFIG_REFRESH,
        EventTypes.WEBSOCKET_RECONNECT,
        EventTypes.FILLS_REFRESH_SUMMARY,
        EventTypes.PLANNING_DEFER_SUMMARY,
        EventTypes.PLANNING_SYMBOL_STATE,
        EventTypes.HSL_RAW_RED_PENDING,
        EventTypes.HSL_RED_TRIGGERED,
        EventTypes.HSL_RED_FINALIZED_WITHOUT_ORDER,
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
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CREATE_SKIPPED].console is True
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CREATE_SKIPPED].text is True
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CONFIRMATION_TIMEOUT].console is True
    assert DEFAULT_ROUTES[EventTypes.EXECUTION_CONFIRMATION_TIMEOUT].text is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED].console is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED].text is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED].console is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED].text is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED].console is True
    assert DEFAULT_ROUTES[EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED].text is True
    assert DEFAULT_ROUTES[EventTypes.FILL_INGESTED].console is True
    assert DEFAULT_ROUTES[EventTypes.FILL_INGESTED].text is True
    assert DEFAULT_ROUTES[EventTypes.POSITION_CHANGED].console is True
    assert DEFAULT_ROUTES[EventTypes.POSITION_CHANGED].text is True
    assert DEFAULT_ROUTES[EventTypes.BALANCE_CHANGED].console is True
    assert DEFAULT_ROUTES[EventTypes.BALANCE_CHANGED].text is True
    assert DEFAULT_ROUTES[EventTypes.RISK_MODE_CHANGED].console is True
    assert DEFAULT_ROUTES[EventTypes.RISK_MODE_CHANGED].text is True
    assert DEFAULT_ROUTES[EventTypes.HSL_TRANSITION].console is True
    assert DEFAULT_ROUTES[EventTypes.HSL_TRANSITION].text is True
    assert DEFAULT_ROUTES[EventTypes.BOT_STARTUP_TIMING].console is True
    assert DEFAULT_ROUTES[EventTypes.BOT_STARTUP_TIMING].text is True
    assert DEFAULT_ROUTES[EventTypes.BOT_SHUTDOWN_STAGE].console is True
    assert DEFAULT_ROUTES[EventTypes.BOT_SHUTDOWN_STAGE].text is True
    assert DEFAULT_ROUTES[EventTypes.HEALTH_SUMMARY].console is True
    assert DEFAULT_ROUTES[EventTypes.HEALTH_SUMMARY].text is True
    assert DEFAULT_ROUTES[EventTypes.HSL_STATUS].console is True
    assert DEFAULT_ROUTES[EventTypes.HSL_STATUS].text is True
    assert DEFAULT_ROUTES[EventTypes.TRAILING_STATUS].console is True
    assert DEFAULT_ROUTES[EventTypes.TRAILING_STATUS].text is True
    assert DEFAULT_ROUTES[EventTypes.UNSTUCK_STATUS].console is True
    assert DEFAULT_ROUTES[EventTypes.UNSTUCK_STATUS].text is True
    assert DEFAULT_ROUTES[EventTypes.UNSTUCK_SELECTION].console is True
    assert DEFAULT_ROUTES[EventTypes.UNSTUCK_SELECTION].text is True
    assert DEFAULT_ROUTES[EventTypes.REALIZED_LOSS_GATE_BLOCKED].console is True
    assert DEFAULT_ROUTES[EventTypes.REALIZED_LOSS_GATE_BLOCKED].text is True


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
    assert emitted.cycle_id is None
    assert emitted.snapshot_id is None
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


def test_diagnostic_event_carries_envelope_ids_to_pipeline():
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
        "snapshot.built",
        ("diagnostic", "snapshot"),
        {"snapshot_id": "snap_1"},
        cycle_id="cy_7",
        snapshot_id="snap_1",
    )

    emitted = event.emit(bot)

    assert emitted.event_type == EventTypes.SNAPSHOT_BUILT
    assert emitted.cycle_id == "cy_7"
    assert emitted.snapshot_id == "snap_1"
    assert emitted.data["snapshot_id"] == "snap_1"
    assert bot._live_event_pipeline.close(timeout=2.0) is True


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


def test_console_format_summarizes_startup_timing():
    event = LiveEvent(
        EventTypes.BOT_STARTUP_TIMING,
        status="succeeded",
        reason_code=ReasonCodes.STARTUP_PHASE_READY,
        data={
            "phase": "hsl",
            "elapsed_ms": 12345,
            "since_previous_ms": 2345,
            "details": "mode=coin",
        },
    )

    assert format_console_event(event) == (
        "[boot] succeeded phase=hsl-ready elapsed=12.35s since_previous=2.35s "
        "details=mode=coin reason=startup_phase_ready"
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


def test_console_format_summarizes_low_balance_create_skip():
    event = LiveEvent(
        EventTypes.EXECUTION_CREATE_SKIPPED,
        status="skipped",
        cycle_id="cy_low_balance",
        order_wave_id="ow_5",
        reason_code=ReasonCodes.LOW_BALANCE,
        message="exposure-increasing creates skipped because balance is below threshold",
        data={
            "order_count": 3,
            "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
            "raw_balance": 0.25,
            "balance_threshold": 1.0,
            "quote": "USDT",
            "allowed_cancel": 2,
            "allowed_protective_create": 1,
        },
    )

    assert format_console_event(event) == (
        "[gate] skipped cycle=cy_low_balance wave=ow_5 orders=3 "
        "symbols=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT "
        "balance=0.25 USDT threshold=1 USDT allow_cancel=2 "
        "allow_protective_create=1 reason=low_balance "
        "exposure-increasing creates skipped because balance is below threshold"
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


def test_console_format_summarizes_forager_selection():
    event = LiveEvent(
        EventTypes.FORAGER_SELECTION,
        status="succeeded",
        cycle_id="cy_forager",
        pside="long",
        reason_code="rust_orchestrator_selection",
        data={
            "candidate_count": 40,
            "eligible_count": 33,
            "selected_count": 3,
            "selected_symbols": [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "SOL/USDT:USDT",
            ],
            "incumbent_symbols": ["BTC/USDT:USDT"],
            "slots_to_fill": 1,
            "max_n_positions": 3,
            "feature_unavailable_count": 7,
            "volatility_dropped_count": 5,
            "max_age_ms": 180_000,
            "fetch_budget": 4,
            "hysteresis_event_count": 2,
            "source": "rust_orchestrator",
        },
    )

    assert format_console_event(event) == (
        "[forager] succeeded cycle=cy_forager selected=3/33/40 slots=1/3 "
        "unavailable=7 volatility_dropped=5 "
        "symbols=BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT "
        "incumbents=BTC/USDT:USDT max_age=180s fetch_budget=4 "
        "hysteresis_events=2 source=rust_orchestrator pside=long "
        "reason=rust_orchestrator_selection"
    )


def test_forager_selection_console_route_is_throttled():
    route = DEFAULT_ROUTES[EventTypes.FORAGER_SELECTION]

    assert route.console is True
    assert route.text is True
    assert route.throttle_interval_ms == 5 * 60 * 1000


def test_console_format_summarizes_hsl_status_distance_to_red():
    event = LiveEvent(
        EventTypes.HSL_STATUS,
        status="succeeded",
        cycle_id="cy_hsl",
        symbol="ZEC/USDT:USDT",
        pside="long",
        reason_code="green",
        data={
            "signal_mode": "coin",
            "tier": "green",
            "dist_to_red": 0.02,
            "drawdown_score": 0.03,
            "red_threshold": 0.05,
            "cooldown_remaining": "none",
            "last_red_ts": 1_600_000,
            "has_open_position": True,
        },
    )

    assert format_console_event(event) == (
        "[risk] succeeded cycle=cy_hsl mode=coin tier=green "
        "dist_to_red=0.020000 drawdown_score=0.030000 red_threshold=0.050000 "
        "cooldown=none last_red_ts=1600000 symbol=ZEC/USDT:USDT "
        "pside=long reason=green"
    )


def test_console_format_summarizes_hsl_cooldown_seconds():
    event = LiveEvent(
        EventTypes.HSL_STATUS,
        status="degraded",
        symbol="NEAR/USDT:USDT",
        pside="long",
        reason_code="cooldown_active",
        data={"tier": "red", "cooldown_remaining_seconds": 300.0},
    )

    assert format_console_event(event) == (
        "[risk] degraded tier=red cooldown=300s symbol=NEAR/USDT:USDT "
        "pside=long reason=cooldown_active"
    )


def test_console_format_summarizes_initial_entry_distance_gate_block():
    event = LiveEvent(
        EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED,
        status="skipped",
        cycle_id="cy_entry_gate",
        symbol="BTC/USDT:USDT",
        pside="long",
        side="buy",
        reason_code="initial_entry_distance_gate",
        data={
            "action": "skip_create",
            "order_type": "entry_initial_normal_long",
            "qty": 0.001,
            "price": 98_750.0,
            "market_price": 101_000.0,
            "distance_pct": 2.278481012658,
            "threshold_pct": 1.25,
            "tolerance_pct": 0.1,
        },
    )

    assert format_console_event(event) == (
        "[entry] skipped cycle=cy_entry_gate action=skip_create "
        "type=entry_initial_normal_long qty=0.001 price=98750 market=101000 "
        "dist=2.2785% threshold=1.2500% tolerance=0.1000% "
        "symbol=BTC/USDT:USDT pside=long reason=initial_entry_distance_gate"
    )


def test_console_format_summarizes_min_effective_cost_block():
    event = LiveEvent(
        EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED,
        status="skipped",
        symbol="SOL/USDT:USDT",
        pside="long",
        reason_code="min_effective_cost_blocked",
        data={
            "action": "skip_create",
            "projected_initial_cost": 3.25,
            "effective_min_cost": 10.0,
            "balance": 250.0,
            "effective_limit": 0.12,
            "entry_initial_qty_pct": 0.05,
        },
    )

    assert format_console_event(event) == (
        "[entry] skipped action=skip_create notional=3.25/10 "
        "effective_limit=12.0000% initial_qty=5.0000% "
        "symbol=SOL/USDT:USDT pside=long reason=min_effective_cost_blocked"
    )


def test_console_format_summarizes_realized_loss_gate_block():
    event = LiveEvent(
        EventTypes.REALIZED_LOSS_GATE_BLOCKED,
        status="deferred",
        symbol="ETH/USDT:USDT",
        pside="short",
        reason_code="realized_loss_gate_blocked",
        data={
            "order_type": "close_grid_short",
            "qty": 0.25,
            "price": 3_100.0,
            "projected_pnl": -12.5,
            "projected_balance_after": 987.5,
            "balance_floor": 990.0,
            "max_realized_loss_pct": 0.01,
        },
    )

    assert format_console_event(event) == (
        "[risk] deferred type=close_grid_short qty=0.25 price=3100 "
        "projected_pnl=-12.5 projected_balance=987.5 floor=990 "
        "max_loss=1.0000% symbol=ETH/USDT:USDT pside=short "
        "reason=realized_loss_gate_blocked"
    )


def test_console_format_summarizes_trailing_status():
    event = LiveEvent(
        EventTypes.TRAILING_STATUS,
        status="succeeded",
        cycle_id="cy_trailing",
        symbol="BTC/USDT:USDT",
        pside="long",
        reason_code="trailing_status",
        data={
            "kind": "entry",
            "trailing_status": "waiting_threshold",
            "selected_mode": "grid",
            "threshold_met": False,
            "threshold_pct": 0.0125,
            "threshold_price": 98_750.0,
            "retracement_met": False,
            "retracement_pct": 0.004,
            "retracement_price": 99_145.0,
            "threshold_projection_retracement_price": 99_145.0,
            "current_price": 101_000.0,
            "current_vs_threshold_ratio": 101_000.0 / 98_750.0 - 1.0,
            "current_vs_retracement_ratio": 101_000.0 / 99_145.0 - 1.0,
        },
    )

    assert format_console_event(event) == (
        "[trailing] succeeded cycle=cy_trailing kind=entry "
        "trailing_status=waiting_threshold mode=grid threshold_met=no threshold=1.2500%@98750 "
        "threshold_dist=2.2785% retracement_met=no retracement=0.4000%@99145 "
        "retracement_dist=1.8710% if_threshold_retrace=0.4000%@99145 current=101000 "
        "symbol=BTC/USDT:USDT pside=long reason=trailing_status"
    )


def test_console_format_summarizes_unsupported_trailing_status():
    event = LiveEvent(
        EventTypes.TRAILING_STATUS,
        status="succeeded",
        symbol="BTC/USDT:USDT",
        pside="long",
        reason_code="trailing_status",
        data={
            "kind": "position",
            "diagnostics_supported": False,
            "strategy_kind": "trailing_grid_v7",
            "unsupported_reason": "monitor trailing diagnostics use trailing_martingale helper formulas",
        },
    )

    assert format_console_event(event) == (
        "[trailing] succeeded kind=position strategy=trailing_grid_v7 "
        "unsupported=monitor trailing diagnostics use trailing_martingale helper "
        "formulas symbol=BTC/USDT:USDT pside=long reason=trailing_status"
    )


def test_console_format_summarizes_unstuck_status():
    event = LiveEvent(
        EventTypes.UNSTUCK_STATUS,
        status="succeeded",
        reason_code="unstuck_status",
        data={
            "changed": True,
            "over_budget_sides": ["long"],
            "sides": {
                "long": {
                    "status": "ok",
                    "allowance": -12.3,
                    "over_budget": True,
                    "next_symbol": "BTC/USDT:USDT",
                    "next_target_price": 101_000.0,
                    "next_target_distance_ratio": 0.005,
                    "next_unstuck_trigger_distance_ratio": 0.0125,
                },
                "short": {"status": "disabled"},
            },
        },
    )

    assert format_console_event(event) == (
        "[unstuck] succeeded long:ok allowance=-12.3 over_budget "
        "candidate=BTC/USDT:USDT target=101000 target_dist=0.5000% ema_gate_dist=1.2500% "
        "short:disabled over_budget=long changed=true reason=unstuck_status"
    )


def test_console_format_summarizes_unstuck_selection():
    event = LiveEvent(
        EventTypes.UNSTUCK_SELECTION,
        status="succeeded",
        symbol="SUI/USDT:USDT",
        pside="long",
        reason_code="unstuck_selection",
        data={
            "changed": True,
            "entry_price": 1.0,
            "current_price": 1.1,
            "price_diff_pct": 10.0,
            "allowance": -12.3,
        },
    )

    assert format_console_event(event) == (
        "[unstuck] succeeded entry=1 current=1.1 pos_pnl_dist=10.0000% "
        "allowance=-12.3 changed=true symbol=SUI/USDT:USDT pside=long "
        "reason=unstuck_selection"
    )


def test_console_format_summarizes_periodic_health():
    event = LiveEvent(
        EventTypes.HEALTH_SUMMARY,
        status="succeeded",
        cycle_id="cy_health",
        reason_code=ReasonCodes.PERIODIC_HEALTH_SUMMARY,
        data={
            "uptime_ms": 123_456,
            "last_loop_duration_ms": 1_250,
            "positions_long": 2,
            "positions_short": 1,
            "balance_raw": 1_005.25,
            "equity": 1_010.75,
            "orders_placed": 3,
            "orders_cancelled": 1,
            "fills": 2,
            "pnl": -1.5,
            "errors_last_hour": 1,
            "ws_reconnects": 2,
            "rate_limits": 3,
            "rss_bytes": 157_286_400,
            "event_queue_depth": 4,
            "event_queue_maxsize": 1000,
            "event_dropped_total": 2,
            "event_sink_error_total": 1,
            "event_pipeline_worker_alive": True,
        },
    )

    assert format_console_event(event) == (
        "[health] succeeded cycle=cy_health uptime=123s loop=1.2s "
        "positions=2L/1S balance=1005.25 equity=1010.75 orders=+3/-1 "
        "fills=2:pnl=-1.5 errors=1/h ws=2 rate_limits=3 rss=150.0MiB "
        "event_q=4/1000 event_dropped=2 sink_errors=1 "
        "reason=periodic_health_summary"
    )


def test_console_format_summarizes_health_error_burst_without_raw_error():
    event = LiveEvent(
        EventTypes.HEALTH_SUMMARY,
        level="warning",
        status="degraded",
        cycle_id="cy_error",
        reason_code=ReasonCodes.EXECUTION_LOOP_ERROR_BURST,
        data={
            "count": 3,
            "window_s": 60,
            "top_endpoints": [
                {"endpoint": "account-overview", "count": 2},
                {"endpoint": "open-orders", "count": 1},
            ],
            "latest_error_type": "RequestTimeout",
            "latest_status": "-",
            "latest_code": "-",
            "latest_error": "kucoinfutures GET https://example.invalid/account?apiKey=SECRET",
        },
    )

    rendered = format_console_event(event)
    assert rendered == (
        "[health] degraded cycle=cy_error errors=3/60s "
        "top=account-overview:2,open-orders:1 latest=RequestTimeout "
        "status=- code=- reason=execution_loop_error_burst"
    )
    assert "SECRET" not in rendered
    assert "example.invalid" not in rendered


def test_console_format_summarizes_fill_ingested():
    event = LiveEvent(
        EventTypes.FILL_INGESTED,
        status="succeeded",
        cycle_id="cy_fill",
        symbol="BTC/USDT:USDT",
        pside="long",
        side="buy",
        reason_code="new_fill",
        data={
            "pb_order_type": "entry_grid_normal_long",
            "qty": 0.001,
            "price": 101_234.5,
            "pnl": -1.25,
            "fee": -0.04,
            "client_order_id_short": "abc123",
            "fill_id_hash": "9f54f33d005de125ca93371eeda0374f039e520574633e8335066351e275c6a2",
        },
    )

    assert format_console_event(event) == (
        "[fill] succeeded cycle=cy_fill side=buy type=entry_grid_normal_long "
        "qty=0.001 price=101234.5 pnl=-1.25 fee=-0.04 client_id=abc123 "
        "symbol=BTC/USDT:USDT pside=long reason=new_fill"
    )


def test_console_format_summarizes_position_changed():
    event = LiveEvent(
        EventTypes.POSITION_CHANGED,
        status="succeeded",
        cycle_id="cy_pos",
        symbol="ETH/USDT:USDT",
        pside="short",
        reason_code="short_increased",
        data={
            "action": "short_increased",
            "old_size": -0.2,
            "new_size": -0.35,
            "size_delta": -0.15,
            "new_price": 2_500.0,
            "last_price": 2_480.0,
            "wallet_exposure": 0.12,
            "wel_ratio": 0.6,
            "twel_ratio": 0.24,
            "upnl": 7.5,
        },
    )

    assert format_console_event(event) == (
        "[pos] succeeded cycle=cy_pos action=short_increased size=-0.2->-0.35 "
        "delta=-0.15 price=2500 last=2480 we=12.0000% wel=60.0000% "
        "twel=24.0000% upnl=7.5 symbol=ETH/USDT:USDT pside=short "
        "reason=short_increased"
    )


def test_console_format_summarizes_balance_changed():
    event = LiveEvent(
        EventTypes.BALANCE_CHANGED,
        status="succeeded",
        cycle_id="cy_balance",
        reason_code="balance_changed",
        data={
            "balance_raw": 1_005.25,
            "balance_raw_delta": 5.25,
            "balance_snapped": 1_004.0,
            "balance_snapped_delta": 4.0,
            "equity": 1_010.75,
            "source": "REST",
        },
    )

    assert format_console_event(event) == (
        "[balance] succeeded cycle=cy_balance balance=1005.25 delta=5.25 "
        "snapped=1004 snapped_delta=4 equity=1010.75 source=REST "
        "reason=balance_changed"
    )


def test_console_format_summarizes_risk_mode_changed():
    event = LiveEvent(
        EventTypes.RISK_MODE_CHANGED,
        status="succeeded",
        cycle_id="cy_risk_mode",
        symbol="BTC/USDT:USDT",
        pside="long",
        reason_code="hsl_red_runtime_forced_modes",
        data={
            "source": "hsl",
            "action": "replace",
            "symbols": {
                "count": 3,
                "sample": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
            },
            "previous_mode_counts": {"normal": 3},
            "mode_counts": {"panic": 2, "manual": 1},
        },
    )

    assert format_console_event(event) == (
        "[risk] succeeded cycle=cy_risk_mode source=hsl action=replace "
        "symbols=3:BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT "
        "previous_counts=normal:3 mode_counts=manual:1,panic:2 "
        "symbol=BTC/USDT:USDT pside=long reason=hsl_red_runtime_forced_modes"
    )


def test_console_format_summarizes_hsl_transition():
    event = LiveEvent(
        EventTypes.HSL_TRANSITION,
        status="succeeded",
        cycle_id="cy_hsl_transition",
        symbol="NEAR/USDT:USDT",
        pside="long",
        reason_code="yellow_to_orange",
        data={
            "signal_mode": "coin",
            "previous_tier": "yellow",
            "tier": "orange",
            "dist_to_red": 0.012345678,
            "drawdown_score": 0.1875,
            "red_threshold": 0.2,
            "balance": 1_000.0,
            "strategy_equity": 950.0,
            "metrics": {"balance": 1_000.0},
            "timestamp_ms": 1_782_946_000_000,
        },
    )

    assert format_console_event(event) == (
        "[risk] succeeded cycle=cy_hsl_transition mode=coin tier=yellow->orange "
        "dist_to_red=0.012346 drawdown_score=0.187500 red_threshold=0.200000 "
        "ts=1782946000000 symbol=NEAR/USDT:USDT pside=long reason=yellow_to_orange"
    )


def test_operator_console_filters_flat_green_coin_hsl_status():
    console = ListEventSink()
    text = ListEventSink()
    pipeline = LiveEventPipeline(
        structured_sinks=[],
        monitor_sinks=[],
        console_sink=console,
        text_sink=text,
    )

    flat_green = LiveEvent(
        EventTypes.HSL_STATUS,
        status="succeeded",
        symbol="ARB/USDT:USDT",
        pside="long",
        reason_code="green",
        data={
            "signal_mode": "coin",
            "tier": "green",
            "dist_to_red": 0.02,
            "has_open_position": False,
        },
    )
    held_green = LiveEvent(
        EventTypes.HSL_STATUS,
        status="succeeded",
        symbol="ZEC/USDT:USDT",
        pside="long",
        reason_code="green",
        data={
            "signal_mode": "coin",
            "tier": "green",
            "dist_to_red": 0.02,
            "has_open_position": True,
        },
    )
    flat_orange = LiveEvent(
        EventTypes.HSL_STATUS,
        status="succeeded",
        symbol="MORPHO/USDT:USDT",
        pside="long",
        reason_code="orange",
        data={
            "signal_mode": "coin",
            "tier": "orange",
            "dist_to_red": 0.01,
            "has_open_position": False,
        },
    )
    cooldown = LiveEvent(
        EventTypes.HSL_STATUS,
        status="degraded",
        symbol="NEAR/USDT:USDT",
        pside="long",
        reason_code="cooldown_active",
        data={"tier": "red", "cooldown_remaining_seconds": 300.0},
    )

    pipeline.emit(flat_green)
    pipeline.emit(held_green)
    pipeline.emit(flat_orange)
    pipeline.emit(cooldown)

    assert console.events == [held_green, cooldown]
    assert text.events == [held_green, cooldown]
    assert pipeline.close(timeout=2.0) is True


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
