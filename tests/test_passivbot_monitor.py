import logging

import pytest


class RecorderPublisher:
    def __init__(self):
        self.events = []
        self.errors = []
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
