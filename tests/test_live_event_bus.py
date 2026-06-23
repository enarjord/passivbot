import json

import pytest

from live.event_bus import (
    DEFAULT_ROUTES,
    EventRoute,
    EventTypes,
    ListEventSink,
    LiveEvent,
    LiveEventContext,
    LiveEventPipeline,
    MonitorEventSink,
    REDACTED,
    format_console_event,
    normalize_event_type,
    payload_hash,
    redact_payload,
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


def test_route_table_keeps_data_events_off_console_by_default():
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].structured is True
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].monitor is True
    assert DEFAULT_ROUTES[EventTypes.DATA_PACKET_UPDATED].console is False
    assert DEFAULT_ROUTES[EventTypes.ORDER_WAVE_COMPLETED].console is True
    assert DEFAULT_ROUTES[EventTypes.ORDER_WAVE_COMPLETED].text is True


def test_redact_payload_recurses_and_payload_hash_is_stable():
    payload = {"safe": 1, "auth": {"token": "secret"}, "items": [{"secret": "x"}]}

    assert redact_payload(payload) == {
        "safe": 1,
        "auth": REDACTED,
        "items": [{"secret": REDACTED}],
    }
    assert payload_hash({"b": 2, "a": 1}) == payload_hash({"a": 1, "b": 2})


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


def test_sink_failure_degrades_observability_without_raising():
    class FailingSink:
        def write(self, event):
            raise OSError("disk full")

    pipeline = LiveEventPipeline(
        structured_sinks=[FailingSink()],
        routes={EventTypes.SNAPSHOT_BUILT: EventRoute(structured=True, monitor=False)},
    )

    event = pipeline.emit(LiveEvent(EventTypes.SNAPSHOT_BUILT))

    assert event.event_type == EventTypes.SNAPSHOT_BUILT
    assert pipeline.flush(timeout=2.0) is True
    assert pipeline.sink_error_counters["structured"] == 1
    assert pipeline.degraded_events[-1].reason_code == "structured_sink_failed"
    assert pipeline.close(timeout=2.0) is True


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
        symbol="BTC/USDT:USDT",
        pside="long",
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
    assert result["payload"]["_live_event"]["event_type"] == EventTypes.PLANNING_UNAVAILABLE
    assert result["payload"]["_live_event"]["data"] == {"reason": "stale_ema"}


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
        "[planning.unavailable] deferred cycle=cy_1 symbol=BTC/USDT:USDT "
        "pside=long reason=stale_ema entries deferred"
    )
    assert format_console_event(event) == expected


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
        EventTypes.ORDER_WAVE_COMPLETED,
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
        EventTypes.ORDER_WAVE_COMPLETED,
        EventTypes.CYCLE_COMPLETED,
    ]
    assert {event.cycle_id for event in structured.events} == {"cy_1"}
