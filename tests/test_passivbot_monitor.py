import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

import pytest


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


@pytest.mark.asyncio
async def test_handle_balance_update_records_monitor_balance_event():
    import passivbot as pb_mod

    class FakeBot:
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event

        def __init__(self):
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


@pytest.mark.asyncio
async def test_execute_orders_parent_records_order_opened_event():
    import passivbot as pb_mod

    class FakeBot:
        _monitor_record_event = pb_mod.Passivbot._monitor_record_event
        _monitor_order_payload = pb_mod.Passivbot._monitor_order_payload

        def __init__(self):
            self.monitor_publisher = RecorderPublisher()
            self._health_orders_placed = 0
            self.debug_mode = False

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
        _set_log_silence_watchdog_context = pb_mod.Passivbot._set_log_silence_watchdog_context
        _start_log_silence_watchdog = pb_mod.Passivbot._start_log_silence_watchdog
        _stop_log_silence_watchdog = pb_mod.Passivbot._stop_log_silence_watchdog

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

        async def warmup_candles_staggered(self):
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
                ("long", "total_wallet_exposure_limit"): 1.7,
                ("short", "total_wallet_exposure_limit"): 0.0,
                ("long", "ema_span_0"): 10.0,
                ("long", "ema_span_1"): 20.0,
                ("short", "ema_span_0"): 10.0,
                ("short", "ema_span_1"): 20.0,
                ("long", "entry_initial_ema_dist"): 0.01,
                ("short", "entry_initial_ema_dist"): 0.01,
                ("long", "entry_volatility_ema_span_hours"): 24.0,
                ("short", "entry_volatility_ema_span_hours"): 24.0,
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
                ("long", "entry_volatility_ema_span_minutes"): 60.0,
                ("short", "entry_volatility_ema_span_minutes"): 60.0,
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
                ("long", "entry_trailing_grid_ratio"): 1.0,
                ("short", "entry_trailing_grid_ratio"): 0.0,
                ("long", "entry_trailing_retracement_pct"): 0.01,
                ("short", "entry_trailing_retracement_pct"): 0.01,
                ("long", "entry_trailing_threshold_pct"): 0.01,
                ("short", "entry_trailing_threshold_pct"): 0.01,
                ("long", "close_grid_markup_end"): 0.02,
                ("short", "close_grid_markup_end"): 0.02,
                ("long", "close_grid_markup_start"): 0.01,
                ("short", "close_grid_markup_start"): 0.01,
                ("long", "close_grid_qty_pct"): 1.0,
                ("short", "close_grid_qty_pct"): 1.0,
                ("long", "close_trailing_grid_ratio"): 1.0,
                ("short", "close_trailing_grid_ratio"): 0.0,
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
                return {"status": "ok", "allowance": -20.0, "peak": 1100.0, "pct_from_peak": -9.1}
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
        0.1 / 0.3
    )
    assert snapshot["positions"]["BTC/USDT:USDT"]["long"]["twel_ratio"] == pytest.approx(
        0.1 / 1.7
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
    bot.update_positions_and_balance = AsyncMock(return_value=(True, True))
    bot.update_open_orders = AsyncMock(side_effect=RuntimeError("open orders failed"))
    bot.update_pnls = AsyncMock(return_value=True)
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
