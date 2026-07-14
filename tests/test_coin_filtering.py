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
                    "entry_initial_qty_pct": 0.0192,
                    "projected_initial_cost": 1.4732627616,
                    "effective_min_cost": 10.1,
                }
                for idx in range(count)
            ]
        }
    }


class CoinFilterHarness(Passivbot):
    def __init__(
        self,
        *,
        approved_by_side,
        volumes,
        log_ranges,
        max_positions,
        clip_pct=0.0,
        volatility_drop_pct=0.0,
        forager=True,
        min_cost_ok=None,
        age_ok=None,
        filter_min_cost_flag=True,
    ):
        self.PB_modes = {"long": {}, "short": {}}
        self.approved_coins_minus_ignored_coins = approved_by_side
        self._volumes = volumes
        self._log_ranges = log_ranges
        self._max_positions = max_positions
        self._clip_pct = clip_pct
        self._volatility_drop_pct = volatility_drop_pct
        self._forager = forager
        self._min_cost_ok = min_cost_ok or {}
        self._age_ok = age_ok or {}
        self._live_flags = {"filter_by_min_effective_cost": filter_min_cost_flag}
        self.warn_triggered = False
        self.positions = {}
        self.config = {"live": {}}

    # ---- overrides for get_filtered_coins dependencies ----

    def get_forced_PB_mode(self, pside, symbol=None):
        return None

    def is_old_enough(self, pside, symbol):
        return self._age_ok.get(symbol, True)

    def effective_min_cost_is_low_enough(self, pside, symbol):
        return self._min_cost_ok.get(symbol, True)

    def warn_on_high_effective_min_cost(self, _pside):
        self.warn_triggered = True

    def is_forager_mode(self, _pside):
        return self._forager

    def bot_value(self, pside, key):
        if key == "filter_volume_drop_pct":
            return self._clip_pct
        if key == "filter_volatility_drop_pct":
            return self._volatility_drop_pct
        if key == "n_positions":
            return float(self._max_positions[pside])
        return 0.0

    def get_max_n_positions(self, pside):
        return self._max_positions[pside]

    def live_value(self, key):
        return self._live_flags.get(key, False)

    async def calc_volumes(self, _pside, symbols):
        return {sym: self._volumes[sym] for sym in symbols}

    async def calc_volumes_and_log_ranges(
        self, _pside, symbols, max_age_ms=None, max_network_fetches=None
    ):
        return (
            {sym: self._volumes[sym] for sym in symbols},
            {sym: self._log_ranges[sym] for sym in symbols},
        )

    async def calc_log_range(self, _pside, eligible_symbols, max_age_ms=None, max_network_fetches=None):
        return {sym: self._log_ranges[sym] for sym in eligible_symbols}

    def is_pside_enabled(self, _pside):
        return True


@pytest.mark.asyncio
async def test_forager_prefers_highest_volatility():
    approved = {"long": ["AAA", "BBB", "CCC"], "short": []}
    volumes = {"AAA": 10.0, "BBB": 8.0, "CCC": 3.0}
    log_ranges = {"AAA": 0.2, "BBB": 0.6, "CCC": 0.4}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == ["BBB", "CCC"]


@pytest.mark.asyncio
async def test_volume_clip_applied_before_volatility_sort():
    approved = {"long": ["AAA", "BBB", "CCC", "DDD"], "short": []}
    volumes = {"AAA": 100.0, "BBB": 80.0, "CCC": 10.0, "DDD": 5.0}
    log_ranges = {"AAA": 0.3, "BBB": 0.2, "CCC": 0.9, "DDD": 0.8}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
        clip_pct=0.5,
    )
    # clip removes bottom 50% by volume -> keep AAA,BBB
    coins = await bot.get_filtered_coins("long")
    assert coins == ["AAA", "BBB"]


@pytest.mark.asyncio
async def test_volatility_drop_pct_discards_top_tail():
    approved = {"long": ["AAA", "BBB", "CCC", "DDD"], "short": []}
    volumes = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0, "DDD": 10.0}
    log_ranges = {"AAA": 0.9, "BBB": 0.8, "CCC": 0.7, "DDD": 0.6}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 2, "short": 0},
        volatility_drop_pct=0.5,
    )
    coins = await bot.get_filtered_coins("long")
    # drops top 50% => keep CCC,DDD and pick highest among them => [CCC, DDD]
    assert coins == ["CCC", "DDD"]


@pytest.mark.asyncio
async def test_min_cost_filter_warns_when_all_removed():
    approved = {"long": ["AAA", "BBB"], "short": []}
    volumes = {"AAA": 1.0, "BBB": 1.0}
    log_ranges = {"AAA": 0.1, "BBB": 0.2}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 1, "short": 0},
        min_cost_ok={"AAA": False, "BBB": False},
        filter_min_cost_flag=True,
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == []
    assert bot.warn_triggered is True


@pytest.mark.asyncio
async def test_non_forager_returns_sorted_candidates():
    approved = {"long": ["CCC", "AAA", "BBB"], "short": []}
    volumes = {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0}
    log_ranges = {"AAA": 0.1, "BBB": 0.2, "CCC": 0.3}
    bot = CoinFilterHarness(
        approved_by_side=approved,
        volumes=volumes,
        log_ranges=log_ranges,
        max_positions={"long": 3, "short": 0},
        forager=False,
    )
    coins = await bot.get_filtered_coins("long")
    assert coins == ["AAA", "BBB", "CCC"]


def test_split_forager_budget_by_side_round_robins_remainder():
    bot = Passivbot.__new__(Passivbot)
    first = bot._split_forager_budget_by_side(1, ["long", "short"])
    second = bot._split_forager_budget_by_side(1, ["long", "short"])
    assert first in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert second in ({"long": 1, "short": 0}, {"long": 0, "short": 1})
    assert first != second


class _CMColdCacheOnlyStub:
    def __init__(self):
        self.calls = 0

    def get_last_refresh_ms(self, _symbol):
        return 0

    async def get_latest_ema_metrics(self, _symbol, spans_by_metric, **_kwargs):
        self.calls += 1
        return {k: 1.0 for k in spans_by_metric}


class _CMStaleCacheStub:
    def __init__(self):
        self.current_calls = 0
        self.cached_calls = 0

    def get_last_refresh_ms(self, _symbol):
        return 1

    def get_last_final_ts(self, _symbol):
        return 1

    async def get_latest_ema_metrics(self, _symbol, spans_by_metric, **_kwargs):
        self.current_calls += 1
        return {k: float("nan") for k in spans_by_metric}

    async def get_latest_cached_ema_metrics(
        self,
        _symbol,
        spans_by_metric,
        *,
        max_staleness_ms=None,
        window_candles=None,
        timeframe="1m",
    ):
        self.cached_calls += 1
        out = {}
        if "qv" in spans_by_metric:
            out["qv"] = 123.0
        if "log_range" in spans_by_metric:
            out["log_range"] = 0.0042
        return out


@pytest.mark.asyncio
async def test_calc_log_range_respects_cache_only_budget_for_cold_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMColdCacheOnlyStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = (
        lambda _pside, key: 12.0
        if key in ("filter_volatility_ema_span_1m", "filter_volume_ema_span_1m")
        else 0.0
    )
    bot.has_position = lambda *_args, **_kwargs: False
    out = await bot.calc_log_range(
        "long",
        eligible_symbols=["AAA", "BBB", "CCC"],
        max_age_ms=60_000,
        max_network_fetches=0,
    )
    assert out == {}
    assert bot.cm.calls == 0


@pytest.mark.asyncio
async def test_calc_volumes_and_log_ranges_respects_cache_only_budget_for_cold_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMColdCacheOnlyStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = (
        lambda _pside, key: 12.0
        if key in ("filter_volatility_ema_span_1m", "filter_volume_ema_span_1m")
        else 0.0
    )
    bot.has_position = lambda *_args, **_kwargs: False
    volumes, log_ranges = await bot.calc_volumes_and_log_ranges(
        "long",
        symbols=["AAA", "BBB", "CCC"],
        max_age_ms=60_000,
        max_network_fetches=0,
    )
    assert volumes == {}
    assert log_ranges == {}
    assert bot.cm.calls == 0


@pytest.mark.asyncio
async def test_calc_volumes_and_log_ranges_carries_cached_values_for_stale_symbols():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {}}
    bot.cm = _CMStaleCacheStub()
    bot.open_orders = {}
    bot.positions = {}
    bot.bot_value = lambda _pside, key: 12.0
    bot.has_position = lambda *_args, **_kwargs: False

    volumes, log_ranges = await bot.calc_volumes_and_log_ranges(
        "long",
        symbols=["AAA"],
        max_age_ms=300_000,
        max_network_fetches=0,
    )

    assert volumes == {"AAA": 123.0}
    assert log_ranges == {"AAA": 0.0042}
    assert bot.cm.current_calls == 1
    assert bot.cm.cached_calls == 1


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
                    "entry_initial_qty_pct": 0.0192,
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
                    "entry_initial_qty_pct": 0.0192,
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
    assert "live.filter_by_min_effective_cost=false" in seen[0]
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
                    "entry_initial_qty_pct": 0.0192,
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
    assert event.data["entry_initial_qty_pct"] == pytest.approx(0.0192)
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
    assert sum(
        "initial entries blocked by min effective cost summary" in message
        for message in info_messages
    ) == 1
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
                    "entry_initial_qty_pct": 0.0192,
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
            "entry_initial_qty_pct": 0.0192,
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
