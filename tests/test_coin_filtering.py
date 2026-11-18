import logging

import pytest

from passivbot import Passivbot
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline


class _FailingConsoleSink:
    def write(self, _event):
        raise OSError("console unavailable")


def _make_min_effective_cost_event_bot(*, console_sink=None):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot._min_effective_cost_summary_last_log_ms = 0
    bot._min_effective_cost_summary_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda pside: pside == "long"
    bot.bot_id = "bot_min_cost"
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot.live_event_console_enabled = True
    bot._live_event_current_cycle_id = "cy_min_cost"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
        console_sink=console_sink,
    )
    return bot, sink


def _min_effective_cost_out(count=1):
    return {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": idx,
                    "pside": "long",
                    "balance": 51.154957,
                    "effective_limit": 1.5,
                    "entry_initial_qty_fraction": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
                for idx in range(count)
            ]
        }
    }


def test_split_forager_budget_by_side_round_robins_remainder():
    bot = Passivbot.__new__(Passivbot)
    first = bot._split_forager_budget_by_side(1, ["long", "short"])
    second = bot._split_forager_budget_by_side(1, ["long", "short"])
    assert first in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert second in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert first != second


def test_log_min_effective_cost_blocks_includes_concrete_numbers(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda pside: pside == "long"

    seen = []
    monkeypatch.setattr(
        "passivbot.logging.info",
        lambda msg, *args: seen.append(msg % args)
        if str(msg).startswith("[entry]")
        else None,
    )

    out = {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": 0,
                    "pside": "long",
                    "balance": 51.154957,
                    "effective_limit": 1.5,
                    "entry_initial_qty_fraction": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
            ]
        }
    }
    bot._log_min_effective_cost_blocks(out, {0: "BTC/USDC:USDC"})
    assert len(seen) == 1
    assert "BTC long" in seen[0]
    assert "notional wanted/required=1.473263/10.100000" in seen[0]
    assert "action=skip_create" in seen[0]
    assert "docs/configuration.md#filter_by_min_effective_cost" in seen[0]


def test_log_min_effective_cost_blocks_debug_includes_full_context(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda pside: pside == "long"

    seen = []
    monkeypatch.setattr(
        "passivbot.logging.debug",
        lambda msg, *args: seen.append(msg % args)
        if str(msg).startswith("[entry]")
        else None,
    )

    out = {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": 0,
                    "pside": "long",
                    "balance": 51.154957,
                    "effective_limit": 1.5,
                    "entry_initial_qty_fraction": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
            ]
        }
    }
    bot._log_min_effective_cost_blocks(out, {0: "BTC/USDC:USDC"})
    assert len(seen) == 1
    assert "symbol=BTC pside=long" in seen[0]
    assert "projected_initial_cost=1.473263" in seen[0]
    assert "required_effective_min_cost=10.100000" in seen[0]
    assert "balance=51.154957" in seen[0]
    assert "bot.long.universe.filter_by_min_effective_cost=false" in seen[0]
    assert "override_may_create_exchange-min-sized_entries" in seen[0]


def test_log_min_effective_cost_blocks_emits_structured_event(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda pside: pside == "long"
    bot.bot_id = "bot_min_cost"
    bot.exchange = "binance"
    bot.user = "binance_01"
    bot._live_event_current_cycle_id = "cy_min_cost"
    sink = ListEventSink()
    bot._live_event_pipeline = LiveEventPipeline(
        structured_sinks=[sink],
        monitor_sinks=[],
    )
    monkeypatch.setattr(
        "passivbot.logging.info",
        lambda msg, *args: None,
    )
    monkeypatch.setattr(
        "passivbot.logging.debug",
        lambda msg, *args: None,
    )

    out = {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": 0,
                    "pside": "long",
                    "balance": 51.154957,
                    "effective_limit": 1.5,
                    "entry_initial_qty_fraction": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
            ]
        }
    }
    try:
        bot._log_min_effective_cost_blocks(out, {0: "BTC/USDC:USDC"})
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED
    ]
    assert len(events) == 1
    event = events[0]
    assert event.level == "info"
    assert event.status == "skipped"
    assert event.reason_code == "min_effective_cost_blocked"
    assert event.cycle_id == "cy_min_cost"
    assert event.symbol == "BTC/USDC:USDC"
    assert event.pside == "long"
    assert event.data["projected_initial_cost"] == pytest.approx(1.4732627616)
    assert event.data["effective_min_cost"] == pytest.approx(10.1)
    assert event.data["balance"] == pytest.approx(51.154957)
    assert event.data["effective_limit"] == pytest.approx(1.5)
    assert event.data["entry_initial_qty_fraction"] == pytest.approx(0.0192)
    assert event.data["action"] == "skip_create"


@pytest.mark.parametrize("console_sink_fails", [False, True])
def test_min_effective_cost_structured_console_owns_detail_line(
    caplog, console_sink_fails
):
    console_sink = _FailingConsoleSink() if console_sink_fails else ListEventSink()
    bot, sink = _make_min_effective_cost_event_bot(console_sink=console_sink)

    try:
        with caplog.at_level(logging.DEBUG):
            bot._log_min_effective_cost_blocks(
                _min_effective_cost_out(), {0: "BTC/USDC:USDC"}
            )
            bot._log_min_effective_cost_blocks(
                _min_effective_cost_out(), {0: "BTC/USDC:USDC"}
            )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    assert not [
        record
        for record in caplog.records
        if record.levelno == logging.INFO
        and "initial entry blocked by min effective cost |" in record.message
    ]
    assert sum(
        "initial entry min effective cost detail" in record.message
        for record in caplog.records
    ) == 1
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED
    ]
    assert len(events) == 1
    if console_sink_fails:
        assert bot._live_event_pipeline.sink_error_counters["console"] >= 1
    else:
        assert console_sink.events == events


def test_min_effective_cost_uses_legacy_detail_without_pipeline(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot._min_effective_cost_summary_last_log_ms = 0
    bot._min_effective_cost_summary_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda pside: pside == "long"
    bot.live_event_console_enabled = True
    emitted = []
    bot._emit_entry_min_effective_cost_blocked_event = lambda **kwargs: emitted.append(
        kwargs
    )

    with caplog.at_level(logging.INFO):
        bot._log_min_effective_cost_blocks(
            _min_effective_cost_out(), {0: "BTC/USDC:USDC"}
        )

    assert sum(
        "initial entry blocked by min effective cost | BTC long" in record.message
        for record in caplog.records
    ) == 1
    assert len(emitted) == 1


def test_min_effective_cost_uses_legacy_detail_without_console_sink(caplog):
    bot, sink = _make_min_effective_cost_event_bot(console_sink=None)

    try:
        with caplog.at_level(logging.INFO):
            bot._log_min_effective_cost_blocks(
                _min_effective_cost_out(), {0: "BTC/USDC:USDC"}
            )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    assert sum(
        "initial entry blocked by min effective cost | BTC long" in record.message
        for record in caplog.records
    ) == 1
    assert [
        event.event_type
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED
    ] == [EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED]


def test_min_effective_cost_uses_legacy_detail_when_emitter_unavailable(caplog):
    console_sink = ListEventSink()
    bot, sink = _make_min_effective_cost_event_bot(console_sink=console_sink)
    bot._emit_entry_min_effective_cost_blocked_event = None

    try:
        with caplog.at_level(logging.INFO):
            bot._log_min_effective_cost_blocks(
                _min_effective_cost_out(), {0: "BTC/USDC:USDC"}
            )
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    assert sum(
        "initial entry blocked by min effective cost | BTC long" in record.message
        for record in caplog.records
    ) == 1
    assert sink.events == []
    assert console_sink.events == []


def test_min_effective_cost_structured_console_keeps_aggregate_summary(caplog):
    console_sink = ListEventSink()
    bot, sink = _make_min_effective_cost_event_bot(console_sink=console_sink)
    idx_to_symbol = {idx: f"SYM{idx}/USDT:USDT" for idx in range(5)}

    try:
        with caplog.at_level(logging.INFO):
            bot._log_min_effective_cost_blocks(
                _min_effective_cost_out(count=5), idx_to_symbol
            )
        assert bot._live_event_pipeline.flush(timeout=2.0) is True
    finally:
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    info_messages = [
        record.message for record in caplog.records if record.levelno == logging.INFO
    ]
    assert not [
        message
        for message in info_messages
        if "initial entry blocked by min effective cost | SYM" in message
    ]
    summaries = [message for message in info_messages if "[entry] min-cost gate" in message]
    assert len(summaries) == 1
    assert "blocked=5 detailed=3 suppressed=0" in summaries[0]
    assert "examples=SYM0:long,SYM1:long,SYM2:long" in summaries[0]
    assert "SYM3" not in summaries[0] and "SYM4" not in summaries[0]
    events = [
        event
        for event in sink.events
        if event.event_type == EventTypes.ENTRY_MIN_EFFECTIVE_COST_BLOCKED
    ]
    assert len(events) == 3
    assert console_sink.events == events


def test_log_min_effective_cost_blocks_is_throttled(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda _pside: True

    seen = []
    monkeypatch.setattr(
        "passivbot.logging.info",
        lambda msg, *args: seen.append(msg % args)
        if str(msg).startswith("[entry]")
        else None,
    )

    out = {
        "diagnostics": {
            "min_effective_cost_blocks": [
                {
                    "symbol_idx": 0,
                    "pside": "long",
                    "balance": 51.154957,
                    "effective_limit": 1.5,
                    "entry_initial_qty_fraction": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
            ]
        }
    }
    idx_to_symbol = {0: "BTC/USDC:USDC"}
    bot._log_min_effective_cost_blocks(out, idx_to_symbol)
    bot._log_min_effective_cost_blocks(out, idx_to_symbol)
    assert len(seen) == 1


def test_log_min_effective_cost_blocks_throttles_symbol_across_set_changes(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot._min_effective_cost_last_log_ms = {}
    bot._min_effective_cost_log_interval_ms = 900_000
    bot.is_pside_enabled = lambda _pside: True

    seen = []
    monkeypatch.setattr("passivbot.utc_ms", lambda: 1_000_000)
    monkeypatch.setattr(
        "passivbot.logging.info",
        lambda msg, *args: seen.append(msg % args)
        if str(msg).startswith("[entry]")
        else None,
    )

    def block(symbol_idx):
        return {
            "symbol_idx": symbol_idx,
            "pside": "long",
            "balance": 51.154957,
            "effective_limit": 1.5,
            "entry_initial_qty_fraction": 0.0192,
            "projected_initial_cost": 1.4732627616,
            "effective_min_cost": 10.1,
        }

    idx_to_symbol = {
        0: "BTC/USDC:USDC",
        1: "LINK/USDC:USDC",
        2: "BCH/USDC:USDC",
    }
    bot._log_min_effective_cost_blocks(
        {"diagnostics": {"min_effective_cost_blocks": [block(0), block(1)]}},
        idx_to_symbol,
    )
    bot._log_min_effective_cost_blocks(
        {"diagnostics": {"min_effective_cost_blocks": [block(2), block(0)]}},
        idx_to_symbol,
    )

    assert sum("BTC long" in line for line in seen) == 1
    assert sum("LINK long" in line for line in seen) == 1
    assert sum("BCH long" in line for line in seen) == 1


@pytest.mark.parametrize("filtered_side", ["long", "short"])
def test_live_minimum_cost_uses_each_side_flag_even_with_simulation_override(filtered_side):
    bot = Passivbot.__new__(Passivbot)
    bot.config = {
        "bot": {
            side: {"universe": {"filter_by_min_effective_cost": side == filtered_side}}
            for side in ("long", "short")
        },
        "backtest": {"filter_by_min_effective_cost": False},
    }
    bot.get_wallet_exposure_limit_ratio = lambda *args: 1.0
    bot.bot_value = lambda *args: 1.0
    bot.bp = lambda *args: 0.0
    bot._strategy_initial_sizing_fraction = lambda *args: 0.001
    bot.get_hysteresis_snapped_balance = lambda: 1000.0
    bot.effective_min_cost = {"AAA": 5.0, "BBB": 0.5}

    for side in ("long", "short"):
        eligible = [
            symbol for symbol in ("AAA", "BBB")
            if bot.effective_min_cost_is_low_enough(side, symbol)
        ]
        assert eligible == (["BBB"] if side == filtered_side else ["AAA", "BBB"])
