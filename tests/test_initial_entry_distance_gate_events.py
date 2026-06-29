import pytest

from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline
from passivbot import Passivbot


def _make_bot_with_event_sink():
    bot = Passivbot.__new__(Passivbot)
    bot.bot_id = "bot_distance_gate"
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot._live_event_current_cycle_id = "cy_distance_gate"
    bot._initial_entry_distance_gate_log_state = {}
    bot.open_orders = {}
    bot.live_value = lambda key: 0.0005 if key == "order_match_tolerance_pct" else 0.0
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    return bot, sink


def _initial_entry_order():
    return {
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "price": 95_000.0,
        "type": "limit",
        "pb_order_type": "entry_initial_long",
    }


def test_initial_entry_distance_gate_block_emits_structured_event(monkeypatch):
    bot, sink = _make_bot_with_event_sink()
    monkeypatch.setattr("passivbot.logging.info", lambda msg, *args: None)
    monkeypatch.setattr("passivbot.logging.debug", lambda msg, *args: None)

    try:
        bot._log_initial_entry_distance_gate_block(
            _initial_entry_order(),
            market_price=100_000.0,
            signed_dist=0.05,
            threshold=0.01,
        )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "info"
    assert event.status == "skipped"
    assert event.reason_code == "initial_entry_distance_gate"
    assert event.cycle_id == "cy_distance_gate"
    assert event.symbol == "BTC/USDT:USDT"
    assert event.pside == "long"
    assert event.side == "buy"
    assert event.data["action"] == "skip_create"
    assert event.data["order_type"] == "entry_initial_long"
    assert event.data["qty"] == pytest.approx(0.01)
    assert event.data["price"] == pytest.approx(95_000.0)
    assert event.data["market_price"] == pytest.approx(100_000.0)
    assert event.data["distance_pct"] == pytest.approx(5.0)
    assert event.data["threshold_pct"] == pytest.approx(1.0)
    assert event.data["tolerance_pct"] == pytest.approx(0.05)


def test_initial_entry_distance_gate_event_respects_info_throttle(monkeypatch):
    bot, sink = _make_bot_with_event_sink()
    monkeypatch.setattr("passivbot.logging.info", lambda msg, *args: None)
    monkeypatch.setattr("passivbot.logging.debug", lambda msg, *args: None)

    try:
        order = _initial_entry_order()
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100_000.0,
            signed_dist=0.05,
            threshold=0.01,
        )
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100_000.0,
            signed_dist=0.05,
            threshold=0.01,
        )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    assert [
        event.event_type
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED
    ] == [EventTypes.ENTRY_INITIAL_DISTANCE_GATE_BLOCKED]


def test_initial_entry_distance_gate_cleared_emits_structured_event(monkeypatch):
    bot, sink = _make_bot_with_event_sink()
    monkeypatch.setattr("passivbot.logging.info", lambda msg, *args: None)
    monkeypatch.setattr("passivbot.logging.debug", lambda msg, *args: None)
    order = _initial_entry_order()

    try:
        bot._log_initial_entry_distance_gate_block(
            order,
            market_price=100_000.0,
            signed_dist=0.05,
            threshold=0.01,
        )
        bot._log_initial_entry_distance_gate_cleared(
            order,
            market_price=96_000.0,
            signed_dist=0.005,
            threshold=0.01,
        )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    cleared = [
        event
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_INITIAL_DISTANCE_GATE_CLEARED
    ]
    assert len(cleared) == 1
    event = cleared[0]
    assert event.status == "recovered"
    assert event.reason_code == "initial_entry_distance_gate"
    assert event.data["action"] == "allow_create"
    assert event.data["market_price"] == pytest.approx(96_000.0)
    assert event.data["distance_pct"] == pytest.approx(0.5)
    assert event.data["threshold_pct"] == pytest.approx(1.0)
    assert "tolerance_pct" not in event.data
