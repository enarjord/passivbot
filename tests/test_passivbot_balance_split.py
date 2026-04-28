import asyncio
import json
import logging
import sys
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from passivbot_exceptions import FatalBotException

# Stub passivbot_rust before importing passivbot to avoid native dependency during unit test.
sys.modules.setdefault(
    "passivbot_rust",
    types.SimpleNamespace(
        qty_to_cost=lambda *args, **kwargs: 0.0,
        round_dynamic=lambda x, y=None: x,
        calc_order_price_diff=lambda *args, **kwargs: 0.0,
        hysteresis=lambda x, y, z: x,
    ),
)

from passivbot import Passivbot
import passivbot as passivbot_module
from freshness_ledger import ACCOUNT_SURFACES, LIVE_STATE_SURFACES, FreshnessLedger


def test_authoritative_refresh_mode_defaults_to_staged_without_explicit_choice():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "binance"
    bot.config = {"live": {"authoritative_refresh_mode": "legacy"}, "_raw_effective": {"live": {}}}

    assert bot._authoritative_refresh_mode() == "staged"


def test_authoritative_refresh_mode_respects_explicit_legacy_opt_out_for_hyperliquid():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.config = {
        "live": {"authoritative_refresh_mode": "legacy"},
        "_raw_effective": {"live": {"authoritative_refresh_mode": "legacy"}},
    }

    assert bot._authoritative_refresh_mode() == "legacy"


def test_market_snapshot_ticker_strategy_defaults_to_symbols_for_bitget():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {"live": {}}

    assert bot._market_snapshot_ticker_strategy() == "symbols"


def test_market_snapshot_ticker_strategy_keeps_hyperliquid_on_bulk_all_mids():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "hyperliquid"
    bot.config = {"live": {}}

    assert bot._market_snapshot_ticker_strategy() == "bulk"


def test_market_snapshot_ticker_strategy_respects_explicit_override():
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bitget"
    bot.config = {"live": {"market_snapshot_ticker_strategy": "bulk"}}

    assert bot._market_snapshot_ticker_strategy() == "bulk"


@pytest.mark.asyncio
async def test_log_position_changes_batches_market_snapshot_request(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.inverse = False
    bot.c_mults = {
        "BTC/USDT:USDT": 1.0,
        "ETH/USDT:USDT": 1.0,
    }
    bot.pside_int_map = {"long": 1, "short": -1}
    bot.get_raw_balance = lambda: 1_000.0
    bot.bp = lambda pside, key, symbol: 1.0 if key == "wallet_exposure_limit" else 0.0
    bot.bot_value = lambda pside, key: 10.0

    calls = []

    async def _get_live_last_prices(symbols, **kwargs):
        calls.append((list(symbols), kwargs))
        return {
            "BTC/USDT:USDT": 50_000.0,
            "ETH/USDT:USDT": 3_000.0,
        }

    bot._get_live_last_prices = _get_live_last_prices
    monkeypatch.setattr(
        passivbot_module.pbr,
        "qty_to_cost",
        lambda qty, price, c_mult: abs(qty) * price * c_mult,
    )
    monkeypatch.setattr(
        passivbot_module.pbr, "calc_pprice_diff_int", lambda *args: 0.0, raising=False
    )

    await bot.log_position_changes(
        [],
        [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 0.01,
                "price": 49_000.0,
            },
            {
                "symbol": "ETH/USDT:USDT",
                "position_side": "long",
                "size": 0.2,
                "price": 2_900.0,
            },
        ],
    )

    assert len(calls) == 1
    assert calls[0][0] == ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    assert calls[0][1]["context"] == "position_change_log"


@pytest.mark.asyncio
async def test_shutdown_gracefully_awaits_cancelled_maintainers():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = {"maintainer_cancelled": False, "ccp_closed": False, "cca_closed": False}

    async def _maintainer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["maintainer_cancelled"] = True
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen[self.key] = True

    maintainer_task = asyncio.create_task(_maintainer())
    bot.maintainers = {"watch_orders": maintainer_task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await asyncio.sleep(0)
    await bot.shutdown_gracefully()

    assert seen == {
        "maintainer_cancelled": True,
        "ccp_closed": True,
        "cca_closed": True,
    }
    assert maintainer_task.done() is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_awaits_cancelled_maintainers():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = {"maintainer_cancelled": False, "ccp_closed": False, "cca_closed": False}

    async def _maintainer():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen["maintainer_cancelled"] = True
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen[self.key] = True

    maintainer_task = asyncio.create_task(_maintainer())
    bot.maintainers = {"watch_orders": maintainer_task}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await asyncio.sleep(0)
    await bot.shutdown_gracefully()

    assert seen == {
        "maintainer_cancelled": True,
        "ccp_closed": True,
        "cca_closed": True,
    }
    assert maintainer_task.done() is True


@pytest.mark.asyncio
async def test_shutdown_gracefully_waits_for_execution_loop_before_closing_sessions():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _active_execution_loop():
        await asyncio.sleep(0.01)
        seen.append("execution_loop_stopped")
        execution_loop_stopped.set()

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_active_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = False
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()
    await execution_task

    assert seen == ["execution_loop_stopped", "ccp_closed", "cca_closed"]


@pytest.mark.asyncio
async def test_shutdown_gracefully_cancels_stuck_execution_loop_before_closing_sessions():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_execution_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _stuck_execution_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("execution_loop_cancelled")
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_stuck_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = False
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()

    assert execution_task.cancelled() is True
    assert execution_loop_stopped.is_set() is True
    assert seen == ["execution_loop_cancelled", "ccp_closed", "cca_closed"]


@pytest.mark.asyncio
async def test_shutdown_gracefully_does_not_cancel_inline_execution_task_on_timeout():
    bot = Passivbot.__new__(Passivbot)
    bot._shutdown_in_progress = False
    bot.stop_signal_received = False
    bot._shutdown_execution_grace_seconds = 0.01
    bot._monitor_emit_stop = lambda *args, **kwargs: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.monitor_publisher = None

    seen = []
    execution_loop_stopped = asyncio.Event()

    async def _inline_like_execution_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("unexpected_cancel")
            raise

    class _Closer:
        def __init__(self, key):
            self.key = key

        async def close(self):
            seen.append(self.key)

    execution_task = asyncio.create_task(_inline_like_execution_loop())
    bot._execution_loop_task = execution_task
    bot._execution_loop_task_is_inline = True
    bot._execution_loop_stopped = execution_loop_stopped
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.ccp = _Closer("ccp_closed")
    bot.cca = _Closer("cca_closed")

    await bot.shutdown_gracefully()

    assert execution_task.cancelled() is False
    assert execution_loop_stopped.is_set() is True
    assert seen == ["ccp_closed", "cca_closed"]
    execution_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution_task


@pytest.mark.asyncio
async def test_start_bot_treats_shutdown_cancelled_warmup_as_clean_stop(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.exchange = "bybit"
    bot.user = "test_user"
    bot.quote = "USDT"
    bot.start_time_ms = 1_000_000
    bot.config = {"live": {}}
    bot.debug_mode = False
    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    bot._bot_ready = False
    bot.user_info = {"exchange": "bybit"}
    bot._log_startup_banner = lambda: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shutdown cancellation should not be recorded as startup error")
    )
    bot._monitor_flush_snapshot = AsyncMock()
    stop_events = []
    bot._monitor_emit_stop = lambda *args, **kwargs: stop_events.append((args, kwargs))
    bot.init_markets = AsyncMock()
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False

    async def _format(*args, **kwargs):
        return None

    async def _warmup():
        bot.stop_signal_received = True
        raise asyncio.CancelledError("shutdown during warmup")

    monkeypatch.setattr(passivbot_module, "format_approved_ignored_coins", _format)
    bot.warmup_trading_ready_candles = _warmup

    await bot.start_bot()

    assert bot.stop_signal_received is True
    assert bot._bot_ready is False
    assert stop_events
    assert stop_events[-1][0][0] == "startup_aborted"
    assert stop_events[-1][1]["payload"]["stage"] == "warmup_trading_ready_candles"


def _set_pnl_lookback(bot, *, lookback_days: float, now_ms: int) -> None:
    bot.config = {"live": {"pnls_max_lookback_days": float(lookback_days)}}
    bot.get_exchange_time = lambda: now_ms


def test_handle_order_update_logs_summary_and_dedupes(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    now = [1000.0]

    monkeypatch.setattr(time, "time", lambda: now[0])

    batch = [
        {"symbol": "SOL/USDC:USDC", "status": "open"},
        {"symbol": "BTC/USDC:USDC", "status": "canceled"},
    ]

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(batch)
        bot.handle_order_update(batch)
        now[0] += 5.1
        bot.handle_order_update(batch)

    ws_logs = [record.message for record in caplog.records if "[ws]" in record.message]
    assert ws_logs == [
        "[ws] order update detected | cause=replace_hint | events=2 | symbols=BTC/USDC:USDC,SOL/USDC:USDC | statuses=canceled,open | scheduling refresh",
        "[ws] order update detected | cause=replace_hint | events=2 | symbols=BTC/USDC:USDC,SOL/USDC:USDC | statuses=canceled,open | scheduling refresh",
    ]
    assert bot.execution_scheduled is True


def test_handle_order_update_ignores_empty_batches(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False

    with caplog.at_level(logging.INFO):
        bot.handle_order_update([])

    assert bot.execution_scheduled is False
    assert not [record.message for record in caplog.records if "[ws]" in record.message]


def test_handle_order_update_suppresses_recent_self_echoes(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    now_ms = 1_000_000
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms)
    bot.recent_order_executions = [
        {
            "symbol": "BTC/USDC:USDC",
            "side": "buy",
            "position_side": "long",
            "price": 70000.0,
            "qty": 0.001,
            "execution_timestamp": now_ms,
        }
    ]
    bot.recent_order_cancellations = [
        {
            "symbol": "ETH/USDC:USDC",
            "side": "sell",
            "position_side": "long",
            "price": 2500.0,
            "qty": 0.01,
            "execution_timestamp": now_ms,
        }
    ]
    batch = [
        {
            "symbol": "BTC/USDC:USDC",
            "side": "buy",
            "position_side": "long",
            "price": 70000.0,
            "qty": 0.001,
            "status": "open",
        },
        {
            "symbol": "ETH/USDC:USDC",
            "side": "sell",
            "position_side": "long",
            "price": 2500.0,
            "qty": 0.01,
            "status": "canceled",
        },
    ]

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(batch)

    assert bot.execution_scheduled is True
    assert not [record.message for record in caplog.records if "[ws]" in record.message]


def test_handle_order_update_fill_hint_requests_full_confirmation(caplog, monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.execution_scheduled = False
    bot._authoritative_pending_confirmations = {"open_orders": 1}
    bot._authoritative_refresh_epoch = 4
    now = [1000.0]

    monkeypatch.setattr(time, "time", lambda: now[0])

    with caplog.at_level(logging.INFO):
        bot.handle_order_update(
            [{"symbol": "PARA-TOTAL2/USDC:USDC", "status": "closed"}]
        )

    assert bot.execution_scheduled is True
    assert bot._authoritative_pending_confirmations == {
        "open_orders": 5,
        "balance": 5,
        "positions": 5,
        "fills": 5,
    }
    assert any("cause=fill_hint" in record.message for record in caplog.records)


def test_log_staged_refresh_timings_logs_only_for_slow_refreshes(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._authoritative_pending_confirmations = {}
    bot._authoritative_refresh_epoch_changed = set()

    with caplog.at_level(logging.DEBUG):
        bot._log_staged_refresh_timings({"open_orders"}, {"open_orders": 250}, 250)
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 250, "positions": 300, "open_orders": 200, "fills": 400},
            650,
        )
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders"},
            {"balance": 500, "positions": 600, "open_orders": 700},
            1700,
        )
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 2500, "positions": 3000, "open_orders": 2000, "fills": 4000},
            8500,
        )
        bot._authoritative_refresh_epoch_changed = {"positions"}
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 1200, "positions": 1700, "open_orders": 900, "fills": 1300},
            4100,
        )
        bot._authoritative_refresh_epoch_changed = set()
        bot._log_staged_refresh_timings(
            {"balance", "positions", "open_orders", "fills"},
            {"balance": 2500, "positions": 3000, "open_orders": 2000, "fills": 4000},
            10500,
        )

    state_logs = [
        (record.levelname, record.message) for record in caplog.records if "[state]" in record.message
    ]
    assert state_logs == [
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=650ms | sum=1150ms | balance=250ms fills=400ms open_orders=200ms positions=300ms",
        ),
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,open_orders,positions | wall=1700ms | sum=1800ms | balance=500ms open_orders=700ms positions=600ms",
        ),
        (
            "DEBUG",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=8500ms | sum=11500ms | balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms",
        ),
        (
            "INFO",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=4100ms | sum=5100ms | balance=1200ms fills=1300ms open_orders=900ms positions=1700ms",
        ),
        (
            "INFO",
            "[state] staged refresh timings | plan=balance,fills,open_orders,positions | wall=10500ms | sum=11500ms | balance=2500ms fills=4000ms open_orders=2000ms positions=3000ms",
        ),
    ]


def test_order_plan_summary_is_interesting_only_for_large_or_clipped_waves():
    bot = Passivbot.__new__(Passivbot)

    assert (
        bot._order_plan_summary_is_interesting(
            total_pre_cancel=1,
            total_cancel=1,
            total_pre_create=1,
            total_create=1,
            total_skipped=0,
        )
        is False
    )
    assert (
        bot._order_plan_summary_is_interesting(
            total_pre_cancel=6,
            total_cancel=3,
            total_pre_create=5,
            total_create=2,
            total_skipped=4,
        )
        is True
    )


def test_candle_health_missing_trailing_1m_gap_is_not_actionable_during_grace():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"candle_health_trailing_grace_ms": 60_000}}

    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            1,
            {"last_cached_age_ms": 60_000, "refresh_age_ms": 30_000},
        )
        is False
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            1,
            {"last_cached_age_ms": 60_000, "refresh_age_ms": 61_000},
        )
        is True
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1m",
            True,
            2,
            {"last_cached_age_ms": 120_000, "refresh_age_ms": 30_000},
        )
        is True
    )
    assert (
        bot._candle_health_missing_is_actionable(
            "1h",
            True,
            1,
            {"last_cached_age_ms": 3_600_000, "refresh_age_ms": 30_000},
        )
        is True
    )


def test_memory_snapshot_is_interesting_only_initially_or_on_large_change():
    bot = Passivbot.__new__(Passivbot)

    assert bot._memory_snapshot_is_interesting(prev=None, pct_change=None) is True
    assert (
        bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=10.0) is False
    )
    assert bot._memory_snapshot_is_interesting(prev={"rss": 100}, pct_change=30.0) is True


def test_unstuck_status_logs_info_on_change_then_hourly(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_last_log_ms = 0
    bot._unstuck_log_interval_ms = 5 * 60 * 1000
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_allowance_log_hyst_snap_pct = 0.002
    bot._unstuck_allowance_log_snap_by_pside = {}
    bot._unstuck_last_status_signature = None
    bot._unstuck_last_status_info_ms = 0
    now = [5 * 60 * 1000]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])

    state_by_pside = {
        "long": {"status": "ok", "allowance": -41.01, "peak": 16400.0, "pct_from_peak": -0.3},
        "short": {"status": "disabled"},
    }

    def _calc(pside):
        return state_by_pside[pside]

    bot._calc_unstuck_allowance_for_logging = _calc

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_status()
        state_by_pside["long"] = {
            "status": "ok",
            "allowance": -41.03,
            "peak": 16550.0,
            "pct_from_peak": -0.2,
        }
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_status()
        state_by_pside["long"] = {
            "status": "ok",
            "allowance": -41.20,
            "peak": 16580.0,
            "pct_from_peak": -0.2,
        }
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_status()
        now[0] += 60 * 60 * 1000
        bot._maybe_log_unstuck_status()

    unstuck_logs = [record.message for record in caplog.records if "[unstuck]" in record.message]
    assert len(unstuck_logs) == 3


def test_hysteresis_snapped_unstuck_allowance_updates_only_after_threshold():
    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_allowance_log_hyst_snap_pct = 0.002
    bot._unstuck_allowance_log_snap_by_pside = {}

    first = bot._get_hysteresis_snapped_unstuck_allowance("long", -41.00)
    small = bot._get_hysteresis_snapped_unstuck_allowance("long", -41.03)
    large = bot._get_hysteresis_snapped_unstuck_allowance("long", -41.20)

    assert first == pytest.approx(-41.00)
    assert small == pytest.approx(-41.00)
    assert large == pytest.approx(-41.20)


def test_unstuck_selection_logs_on_change_then_hourly(monkeypatch, caplog):
    bot = Passivbot.__new__(Passivbot)
    bot._unstuck_unchanged_info_log_interval_ms = 60 * 60 * 1000
    bot._unstuck_last_selection_signature = None
    bot._unstuck_last_selection_info_ms = 0
    now = [0]
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now[0])

    with caplog.at_level(logging.INFO):
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 5 * 60 * 1000
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 60 * 60 * 1000
        bot._maybe_log_unstuck_selection(
            symbol="SUI/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.1,
            allowance=-41.0,
        )
        now[0] += 1
        bot._maybe_log_unstuck_selection(
            symbol="ADA/USDT:USDT",
            pside="long",
            entry_price=1.0,
            current_price=1.05,
            allowance=-41.0,
        )

    unstuck_logs = [record.message for record in caplog.records if "[unstuck] selecting" in record.message]
    assert len(unstuck_logs) == 3


@pytest.mark.asyncio
async def test_update_pnls_all_lookback_backfills_when_cache_scope_is_narrower():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._pnls_manager = _Manager(cached_events, history_scope="window")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh.assert_awaited_once_with(start_ms=None, end_ms=None)
    bot._pnls_manager.refresh_latest.assert_not_awaited()
    assert bot._pnls_manager.history_scope == "all"


@pytest.mark.asyncio
async def test_update_pnls_all_lookback_uses_incremental_refresh_when_cache_is_full_history():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock()
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    result = await bot.update_pnls()

    assert result is True
    bot._pnls_manager.refresh.assert_not_awaited()
    bot._pnls_manager.refresh_latest.assert_awaited_once_with(
        overlap=20,
        last_refresh_overlap_ms=60 * 60 * 1000,
    )
    assert bot._pnls_manager.history_scope == "all"


@pytest.mark.asyncio
async def test_update_pnls_propagates_unexpected_refresh_errors():
    bot = Passivbot.__new__(Passivbot)
    cached_events = [SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])]

    class _Manager:
        def __init__(self, events, *, history_scope="unknown"):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.refresh_latest = AsyncMock(side_effect=RuntimeError("fill refresh failed"))
            self.history_scope = history_scope

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._pnls_manager = _Manager(cached_events, history_scope="all")
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: None
    bot.logging_level = 0
    bot._health_rate_limits = 0

    with pytest.raises(RuntimeError, match="fill refresh failed"):
        await bot.update_pnls()


@pytest.mark.asyncio
async def test_update_pnls_suppresses_inflight_shutdown_refresh_error(caplog):
    bot = Passivbot.__new__(Passivbot)
    cached_events = [SimpleNamespace(timestamp=1_700_000_000_000, id="fill-1", source_ids=["fill-1"])]
    monitor_errors = []

    class _Manager:
        def __init__(self, events):
            self._events = list(events)
            self.refresh = AsyncMock()
            self.history_scope = "all"

        async def refresh_latest(self, overlap=20, last_refresh_overlap_ms=None):
            bot.stop_signal_received = True
            raise RuntimeError("connector is closed")

        def get_events(self):
            return list(self._events)

        def get_history_scope(self):
            return self.history_scope

        def set_history_scope(self, scope):
            self.history_scope = scope

    bot.stop_signal_received = False
    bot._shutdown_in_progress = False
    bot._pnls_manager = _Manager(cached_events)
    bot.init_pnls = AsyncMock()
    bot.live_value = lambda key: "all" if key == "pnls_max_lookback_days" else None
    bot.get_exchange_time = lambda: 1_700_000_060_000
    bot._log_new_fill_events = lambda new_events: None
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._monitor_record_error = lambda *args, **kwargs: monitor_errors.append((args, kwargs))
    bot.logging_level = 2
    bot._health_rate_limits = 0

    with caplog.at_level(logging.DEBUG):
        result = await bot.update_pnls()

    assert result is False
    assert monitor_errors == []
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any("fill refresh stopped during in-flight request" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_applies_fake_snapshots():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "fake"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot.recent_order_cancellations = []
    bot.fetch_balance = AsyncMock(return_value=123.45)
    bot.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 0.01,
                "price": 100000.0,
            }
        ]
    )
    bot.fetch_open_orders = AsyncMock(
        return_value=[
            {
                "id": "1",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "amount": 0.01,
                "price": 99000.0,
                "timestamp": 1,
                "reduce_only": False,
            }
        ]
    )
    bot.update_pnls = AsyncMock(return_value=True)
    seen = {}

    async def _log_position_changes(*args, **kwargs):
        del args, kwargs
        seen["balance_raw_when_logged"] = bot.balance_raw

    bot.log_position_changes = AsyncMock(side_effect=_log_position_changes)
    bot.handle_balance_update = AsyncMock()
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: False
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()

    assert result is True
    bot.fetch_balance.assert_awaited_once()
    bot.fetch_positions.assert_awaited_once()
    bot.fetch_open_orders.assert_awaited_once()
    bot.update_pnls.assert_awaited_once()
    assert bot.balance_raw == pytest.approx(123.45)
    assert seen["balance_raw_when_logged"] == pytest.approx(123.45)
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(0.01)
    assert bot.open_orders["BTC/USDT:USDT"][0]["id"] == "1"
    bot.handle_balance_update.assert_awaited_once_with(source="REST")


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_applies_bybit_snapshots():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "bybit"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot.recent_order_cancellations = []
    seen = {}

    async def fake_capture_balance_snapshot():
        return {"raw": "balance"}, 456.78

    async def fake_capture_positions_snapshot():
        return (
            {"raw": "positions"},
            [
                {
                    "symbol": "ETH/USDT:USDT",
                    "position_side": "long",
                    "size": 0.25,
                    "price": 2500.0,
                }
            ],
        )

    async def fake_fetch_open_orders():
        return [
            {
                "id": "2",
                "symbol": "ETH/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.25,
                "amount": 0.25,
                "price": 2400.0,
                "timestamp": 2,
                "reduce_only": False,
            }
        ]

    async def _log_position_changes(*args, **kwargs):
        del args, kwargs
        seen["balance_raw_when_logged"] = bot.balance_raw

    bot.capture_balance_snapshot = fake_capture_balance_snapshot
    bot.capture_positions_snapshot = fake_capture_positions_snapshot
    bot.fetch_open_orders = fake_fetch_open_orders
    bot.update_pnls = AsyncMock(return_value=True)
    bot.log_position_changes = AsyncMock(side_effect=_log_position_changes)
    bot.handle_balance_update = AsyncMock()
    bot.order_matches_bot_cancellation = lambda order: False
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: False
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()

    assert result is True
    assert bot.balance_raw == pytest.approx(456.78)
    assert seen["balance_raw_when_logged"] == pytest.approx(456.78)
    assert bot.positions["ETH/USDT:USDT"]["long"]["size"] == pytest.approx(0.25)
    assert bot.open_orders["ETH/USDT:USDT"][0]["id"] == "2"
    bot.handle_balance_update.assert_awaited_once_with(source="REST")


@pytest.mark.asyncio
async def test_fetch_authoritative_state_staged_snapshot_uses_exchange_cohort_hook():
    bot = Passivbot.__new__(Passivbot)
    seen = {}

    async def fake_capture(plan, timings_ms):
        seen["plan"] = set(plan)
        timings_ms["positions_balance"] = 321
        return {"plan": set(plan), "balance": 12.34, "positions": [], "pnls_ok": True}

    bot.capture_authoritative_state_staged_snapshot = fake_capture

    snapshot = await bot._fetch_authoritative_state_staged_snapshot(
        {"balance", "positions", "open_orders", "fills"}
    )

    assert seen["plan"] == {"balance", "positions", "open_orders", "fills"}
    assert snapshot["balance"] == pytest.approx(12.34)
    assert snapshot["positions"] == []
    assert snapshot["pnls_ok"] is True


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_uses_generic_staged_fetch_for_any_exchange():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "binance"
    bot.stop_signal_received = False
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot.fetch_balance = AsyncMock(return_value=100.0)
    bot.fetch_positions = AsyncMock(return_value=[])
    bot.fetch_open_orders = AsyncMock(return_value=[])
    bot.update_pnls = AsyncMock(return_value=True)
    bot._apply_positions_snapshot = lambda positions: ({}, {})
    bot._apply_balance_snapshot = lambda balance: True
    bot._record_authoritative_surface = lambda surface, signature: None
    bot._staged_defer_balance_publication = lambda: False
    bot._reconcile_balance_after_positions_and_balance_refresh = lambda: True
    bot.get_hysteresis_snapped_balance = lambda: 100.0
    bot.log_position_changes = AsyncMock()
    bot.handle_balance_update = AsyncMock()
    bot._apply_open_orders_snapshot = AsyncMock(return_value=True)
    finalized = []
    bot._finalize_authoritative_refresh_consistency = lambda plan: finalized.append(set(plan))

    result = await bot.refresh_authoritative_state()

    assert result is True
    bot.fetch_balance.assert_awaited_once()
    bot.fetch_positions.assert_awaited_once()
    bot.fetch_open_orders.assert_awaited_once()
    bot.update_pnls.assert_awaited_once()
    assert finalized == [{"balance", "positions", "open_orders", "fills"}]


def test_get_exchange_time_uses_direct_utc_ms(monkeypatch):
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    monkeypatch.setattr(pb_mod, "utc_ms", lambda: 123_456.0)

    assert bot.get_exchange_time() == pytest.approx(123_456.0)


@pytest.mark.asyncio
async def test_update_positions_only_updates_positions(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.positions = {}
    bot.active_symbols = []
    bot.fetched_positions = []
    bot.quote = "USDT"

    async def fake_fetch_positions():
        return [
            {
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "size": 1.0,
                "price": 100.0,
            }
        ]

    async def fake_log_position_changes(old, new):
        return

    async def fail_handle_balance_update(source="REST"):
        raise AssertionError("handle_balance_update should not be called by update_positions")

    bot.fetch_positions = fake_fetch_positions
    bot.log_position_changes = fake_log_position_changes
    bot.handle_balance_update = fail_handle_balance_update

    ok = await bot.update_positions()
    assert ok is True
    assert bot.fetched_positions[0]["symbol"] == "BTC/USDT:USDT"
    assert bot.positions["BTC/USDT:USDT"]["long"]["size"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_update_balance_uses_fetch_balance_when_no_cache():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    calls = {"balance": 0}

    async def fake_handle_balance_update(source="REST"):
        calls["balance"] += 1
        assert bot.balance == pytest.approx(123.0)

    async def fake_fetch_balance():
        return 123.0

    bot.fetch_balance = fake_fetch_balance

    ok = await bot.update_balance()
    assert ok is True
    # handle_balance_update should not be called automatically
    assert calls["balance"] == 0
    # manual invocation uses latest balance/positions
    bot.handle_balance_update = fake_handle_balance_update
    await bot.handle_balance_update()
    assert calls["balance"] == 1


@pytest.mark.asyncio
async def test_update_balance_failure_keeps_previous():
    """Test that when fetch_balance raises an exception, previous balance is preserved.

    Per CLAUDE.md: exchange fetch methods should raise exceptions on failure,
    not return False. The exception propagates to the caller who handles it.
    """
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        raise Exception("network error")  # simulate failed fetch

    bot.fetch_balance = fake_fetch_balance

    with pytest.raises(Exception, match="network error"):
        await bot.update_balance()
    # Balance should remain unchanged because assignment was never reached
    assert bot.balance == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_update_balance_override_does_not_reset_hysteresis_anchor():
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot.balance_override = 250.0
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 133.0
    bot.balance_hysteresis_snap_pct = 0.02

    ok = await bot.update_balance()
    assert ok is True
    assert bot.balance == pytest.approx(250.0)
    assert bot.balance_raw == pytest.approx(250.0)
    # Keep hysteresis anchor unchanged while override is active.
    assert bot.previous_hysteresis_balance == pytest.approx(133.0)


def test_apply_balance_snapshot_honors_override_and_retains_exchange_raw(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 100.0
    bot.balance_raw = 100.0
    bot._exchange_reported_balance_raw = 100.0
    bot.balance_override = 250.0
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = 133.0
    bot.balance_hysteresis_snap_pct = 0.02

    with caplog.at_level(logging.INFO):
        ok = bot._apply_balance_snapshot(123.45)

    assert ok is True
    assert bot.balance == pytest.approx(250.0)
    assert bot.balance_raw == pytest.approx(250.0)
    assert bot._exchange_reported_balance_raw == pytest.approx(123.45)
    assert bot.previous_hysteresis_balance == pytest.approx(133.0)
    assert "Using balance override: 250.000000" in caplog.text


@pytest.mark.asyncio
async def test_update_balance_nan_keeps_previous_and_logs_warning(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        return float("nan")

    bot.fetch_balance = fake_fetch_balance

    with caplog.at_level(logging.WARNING):
        ok = await bot.update_balance()
    assert ok is False
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)
    assert "non-finite balance fetch result; keeping previous balance" in caplog.text


@pytest.mark.asyncio
async def test_update_balance_inf_keeps_previous_and_logs_warning(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"
    bot.balance = 50.0
    bot.balance_raw = 50.0
    bot.previous_hysteresis_balance = 50.0

    async def fake_fetch_balance():
        return float("inf")

    bot.fetch_balance = fake_fetch_balance

    with caplog.at_level(logging.WARNING):
        ok = await bot.update_balance()
    assert ok is False
    assert bot.balance == pytest.approx(50.0)
    assert bot.balance_raw == pytest.approx(50.0)
    assert bot.previous_hysteresis_balance == pytest.approx(50.0)
    assert "non-finite balance fetch result; keeping previous balance" in caplog.text


def test_accessor_fallback_when_balance_raw_absent():
    """get_raw_balance() falls back to snapped balance when balance_raw is absent."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 500.0
    # Intentionally do NOT set balance_raw
    if hasattr(bot, "balance_raw"):
        del bot.balance_raw
    assert bot.get_raw_balance() == pytest.approx(500.0)


def test_accessor_returns_raw_when_present():
    """get_raw_balance() returns balance_raw when it exists."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0
    bot.balance_raw = 1010.0
    assert bot.get_raw_balance() == pytest.approx(1010.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


def test_get_wallet_exposure_limit_uses_configured_n_positions_only():
    bot = Passivbot.__new__(Passivbot)
    bot.coin_overrides = {}

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.4
        if key == "n_positions":
            return 7
        raise KeyError(key)

    bot.bot_value = bot_value
    bot.get_max_n_positions = lambda pside: 1
    bot.get_current_n_positions = lambda pside: 99

    wel = bot.get_wallet_exposure_limit("long")
    assert wel == pytest.approx(0.2)


def test_balance_peak_uses_raw_not_snapped():
    """Core bug regression: balance_peak must be derived from raw balance, not snapped.

    After a profit fill, raw balance advances (e.g. 1010) but snapped balance
    may remain stale (e.g. 1000) due to hysteresis. The peak calculation must
    use raw balance so that balance_peak is correct.

    balance_peak = balance_raw + (pnls_cumsum_max - pnls_cumsum_last)

    With raw=1010, cumsum_max=100, cumsum_last=100:
        peak_from_raw = 1010 + (100 - 100) = 1010  ← correct
        peak_from_snap = 1000 + (100 - 100) = 1000  ← WRONG (stale)
    """
    import types
    from unittest.mock import MagicMock

    import numpy as np

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0  # snapped (stale)
    bot.balance_raw = 1010.0  # raw (advanced after profit fill)
    bot._pnls_manager = MagicMock()
    bot._pnls_manager.get_events.return_value = [
        types.SimpleNamespace(pnl=100.0, timestamp=1.0),
    ]

    # Simulate the calc_auto_unstuck_allowance_from_scratch peak calculation
    events = bot._pnls_manager.get_events()
    pnls_cumsum = np.array([ev.pnl for ev in events]).cumsum()
    pnls_cumsum_max = float(pnls_cumsum.max())
    pnls_cumsum_last = float(pnls_cumsum[-1])

    # Using raw balance (correct)
    balance_peak_raw = bot.get_raw_balance() + (pnls_cumsum_max - pnls_cumsum_last)
    assert balance_peak_raw == pytest.approx(1010.0)

    # Using snapped balance (would be wrong)
    balance_peak_snapped = bot.get_hysteresis_snapped_balance() + (pnls_cumsum_max - pnls_cumsum_last)
    assert balance_peak_snapped == pytest.approx(1000.0)

    # The raw-based peak is correct and higher
    assert balance_peak_raw > balance_peak_snapped


def test_effective_min_cost_filter_uses_snapped_balance():
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 10.0
    bot.effective_min_cost = {"BTC/USDT:USDT": 40.0}
    bot.live_value = lambda key: key == "filter_by_min_effective_cost"
    bot.get_wallet_exposure_limit = lambda pside, symbol=None: 1.0
    bot.bp = lambda pside, key, symbol=None: 0.5 if key == "entry_initial_qty_pct" else 0.0

    # Passes only when snapped balance is used:
    # 100 * 1.0 * 0.5 = 50 >= 40; raw path would fail (10 * 1.0 * 0.5 = 5).
    assert bot.effective_min_cost_is_low_enough("long", "BTC/USDT:USDT") is True


@pytest.mark.asyncio
async def test_update_effective_min_cost_uses_executable_min_qty():
    symbol = "SOL/USDT:USDT"
    bot = Passivbot.__new__(Passivbot)
    bot.effective_min_cost = {}
    bot.min_qtys = {symbol: 0.0}
    bot.qty_steps = {symbol: 1.0}
    bot.min_costs = {symbol: 0.1}
    bot.c_mults = {symbol: 1.0}
    bot.get_symbols_approved_or_has_pos = lambda: [symbol]

    async def fake_get_live_last_prices(symbols, **kwargs):
        return {symbol: 88.165}

    bot._get_live_last_prices = fake_get_live_last_prices

    await bot.update_effective_min_cost()

    assert bot.effective_min_cost[symbol] == pytest.approx(88.165)


def test_unstuck_allowance_routes_raw_balance_to_rust(monkeypatch):
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0
    bot.balance_raw = 200.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=10.0), types.SimpleNamespace(pnl=-4.0)]
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.2 if pside == "long" else 0.0
        if key == "total_wallet_exposure_limit":
            return 0.5
        return 0.0

    bot.bot_value = bot_value

    calls = []

    def fake_calc_auto_unstuck_allowance(
        balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last
    ):
        calls.append((balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last))
        return 123.45

    monkeypatch.setattr(pb_mod.pbr, "calc_auto_unstuck_allowance", fake_calc_auto_unstuck_allowance)

    out = bot._calc_unstuck_allowances(allow_new_unstuck=True)

    assert out["long"] == pytest.approx(123.45)
    assert out["short"] == pytest.approx(0.0)
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(200.0)  # raw balance


def test_unstuck_allowance_uses_only_configured_pnl_lookback(monkeypatch):
    import passivbot as pb_mod

    now_ms = 10 * 86_400_000
    bot = Passivbot.__new__(Passivbot)
    bot.balance_raw = 1000.0
    bot.get_raw_balance = lambda: float(bot.balance_raw)
    _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda start_ms=None, end_ms=None, symbol=None: [
            ev
            for ev in [
                types.SimpleNamespace(pnl=100.0, timestamp=now_ms - 3 * 86_400_000),
                types.SimpleNamespace(pnl=-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
                types.SimpleNamespace(pnl=10.0, timestamp=now_ms - 60_000),
            ]
            if start_ms is None or ev.timestamp >= start_ms
        ]
    )

    def bot_value(pside, key):
        if key == "unstuck_loss_allowance_pct":
            return 0.01 if pside == "long" else 0.0
        if key == "total_wallet_exposure_limit":
            return 1.0
        return 0.0

    bot.bot_value = bot_value

    calls = []

    def fake_calc_auto_unstuck_allowance(
        balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last
    ):
        calls.append((balance, loss_allowance_pct, pnl_cumsum_max, pnl_cumsum_last))
        return 10.0

    monkeypatch.setattr(pb_mod.pbr, "calc_auto_unstuck_allowance", fake_calc_auto_unstuck_allowance)

    out = bot._calc_unstuck_allowances(allow_new_unstuck=True)

    assert out["long"] == pytest.approx(10.0)
    assert len(calls) == 1
    assert calls[0] == pytest.approx((1000.0, 0.01, 10.0, 10.0))


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_routes_split_balances(monkeypatch):
    import passivbot as pb_mod

    class FakeBot:
        positions = {}
        balance = 120.0  # snapped
        balance_raw = 175.0  # raw
        PB_modes = {}
        effective_min_cost = {}
        _config_hedge_mode = False
        hedge_mode = False
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = pb_mod.Passivbot._build_monitor_runtime_market_hints
        _build_monitor_runtime_unstuck_hints = pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

    snapshot = {
        "symbols": [],
        "last_prices": {},
        "m1_close_emas": {},
        "m1_volume_emas": {},
        "m1_log_range_emas": {},
        "h1_log_range_emas": {},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    bot = FakeBot()
    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(bot, snapshot, return_snapshot=False)

    assert captured["input"]["balance"] == pytest.approx(120.0)
    assert captured["input"]["balance_raw"] == pytest.approx(175.0)


@pytest.mark.asyncio
async def test_orchestrator_snapshot_payload_includes_exchange_fees(monkeypatch):
    import passivbot as pb_mod

    symbol = "BTC/USDT:USDT"

    class FakeBot:
        positions = {}
        balance = 120.0
        balance_raw = 175.0
        PB_modes = {"long": {symbol: "normal"}, "short": {symbol: "manual"}}
        effective_min_cost = {symbol: 1.0}
        _config_hedge_mode = False
        hedge_mode = False
        qty_steps = {symbol: 0.001}
        price_steps = {symbol: 0.1}
        min_qtys = {symbol: 0.001}
        min_costs = {symbol: 5.0}
        c_mults = {symbol: 1.0}
        markets_dict = {symbol: {"maker": 0.0001, "taker": 0.0004}}
        equity_hard_stop_loss = {"panic_close_order_type": "limit"}
        trailing_prices = {}
        _monitor_record_price_ticks = pb_mod.Passivbot._monitor_record_price_ticks
        _build_monitor_runtime_market_hints = pb_mod.Passivbot._build_monitor_runtime_market_hints
        _build_monitor_runtime_unstuck_hints = pb_mod.Passivbot._build_monitor_runtime_unstuck_hints
        _update_monitor_runtime_hints = pb_mod.Passivbot._update_monitor_runtime_hints

        def config_get(self, keys):
            return None

        def _bot_params_to_rust_dict(self, pside, symbol):
            return {}

        def live_value(self, key):
            return False

        def _log_realized_loss_gate_blocks(self, out, idx_to_symbol):
            return None

        def _log_ema_gating(self, ideal_orders, m1_close_emas, last_prices, symbols):
            return None

        def _to_executable_orders(self, ideal_orders, last_prices):
            return ideal_orders, []

        def _finalize_reduce_only_orders(self, ideal_orders_f, last_prices):
            return ideal_orders_f

        def get_raw_balance(self):
            return float(self.balance_raw)

        def get_hysteresis_snapped_balance(self):
            return float(self.balance)

        _orchestrator_exchange_params = pb_mod.Passivbot._orchestrator_exchange_params
        _get_exchange_fee_rates = pb_mod.Passivbot._get_exchange_fee_rates

        def _pb_mode_to_orchestrator_mode(self, mode):
            return pb_mod.Passivbot._pb_mode_to_orchestrator_mode(self, mode)

    snapshot = {
        "symbols": [symbol],
        "last_prices": {symbol: 100.0},
        "m1_close_emas": {symbol: {10.0: 100.0}},
        "m1_volume_emas": {symbol: {10.0: 1000.0}},
        "m1_log_range_emas": {symbol: {10.0: 0.01}},
        "h1_log_range_emas": {symbol: {10.0: 0.01}},
        "unstuck_allowances": {"long": 0.0, "short": 0.0},
        "realized_pnl_cumsum": {"max": 0.0, "last": 0.0},
    }

    captured = {}

    def fake_compute(json_str):
        captured["input"] = json.loads(json_str)
        return json.dumps({"orders": [], "diagnostics": {"loss_gate_blocks": []}})

    monkeypatch.setattr(pb_mod.pbr, "compute_ideal_orders_json", fake_compute)

    method = pb_mod.Passivbot.calc_ideal_orders_orchestrator_from_snapshot
    await method(FakeBot(), snapshot, return_snapshot=False)

    exchange = captured["input"]["symbols"][0]["exchange"]
    assert exchange["maker_fee"] == pytest.approx(0.0001)
    assert exchange["taker_fee"] == pytest.approx(0.0004)


def test_unstuck_logging_peak_stays_stable_when_profit_updates_both_balance_and_pnl():
    """Regression for peak drift: peak must not decay when profits increase pnl_last and balance_raw."""
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 100.0  # stale snapped value to simulate hysteresis lag

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.0
        if key == "unstuck_loss_allowance_pct":
            return 0.1
        return 0.0

    bot.bot_value = bot_value

    # State A: peak = 100 + (100 - 0) = 200
    bot.balance_raw = 100.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=100.0), types.SimpleNamespace(pnl=-100.0)]
    )
    info_a = bot._calc_unstuck_allowance_for_logging("long")

    # State B (after net +50 realized since trough): peak should still be 200
    # If snapped balance were incorrectly used here, peak would drift down to 150.
    bot.balance_raw = 150.0
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda: [types.SimpleNamespace(pnl=100.0), types.SimpleNamespace(pnl=-50.0)]
    )
    info_b = bot._calc_unstuck_allowance_for_logging("long")

    assert info_a["status"] == "ok"
    assert info_b["status"] == "ok"
    assert info_a["peak"] == pytest.approx(200.0)
    assert info_b["peak"] == pytest.approx(200.0)


def test_unstuck_logging_uses_only_configured_pnl_lookback():
    now_ms = 10 * 86_400_000
    bot = Passivbot.__new__(Passivbot)
    bot.balance = 1000.0
    bot.balance_raw = 1000.0
    _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

    def bot_value(pside, key):
        if key == "total_wallet_exposure_limit":
            return 1.0
        if key == "unstuck_loss_allowance_pct":
            return 0.01
        return 0.0

    bot.bot_value = bot_value
    bot._pnls_manager = types.SimpleNamespace(
        get_events=lambda start_ms=None, end_ms=None, symbol=None: [
            ev
            for ev in [
                types.SimpleNamespace(pnl=100.0, timestamp=now_ms - 3 * 86_400_000),
                types.SimpleNamespace(pnl=-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
                types.SimpleNamespace(pnl=10.0, timestamp=now_ms - 60_000),
            ]
            if start_ms is None or ev.timestamp >= start_ms
        ]
    )

    info = bot._calc_unstuck_allowance_for_logging("long")

    assert info["status"] == "ok"
    assert info["peak"] == pytest.approx(1000.0)
    assert info["pct_from_peak"] == pytest.approx(0.0)
    assert info["allowance"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_update_balance_hysteresis_divergence():
    """Calls update_balance() twice with values within snap threshold, verifies balance_raw != balance."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        # First call: 100.0, second call: 100.5 (within 2% snap threshold)
        return 100.0 if call_count[0] == 1 else 100.5

    bot.fetch_balance = fake_fetch_balance

    ok1 = await bot.update_balance()
    assert ok1 is True
    # After first call, both should be the same
    assert bot.balance_raw == pytest.approx(100.0)
    assert bot.balance == pytest.approx(100.0)

    ok2 = await bot.update_balance()
    assert ok2 is True
    # Raw should update to 100.5, but snapped stays at 100.0 (within 2% threshold)
    assert bot.balance_raw == pytest.approx(100.5)
    assert bot.balance == pytest.approx(100.0)
    # Verify divergence
    assert bot.balance_raw != bot.balance


@pytest.mark.asyncio
async def test_update_balance_divergence_routes_to_orchestrator_input():
    """After creating divergence, builds orchestrator input dict and asserts both values pass through."""
    import passivbot as pb_mod

    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0 if call_count[0] == 1 else 1005.0

    bot.fetch_balance = fake_fetch_balance

    await bot.update_balance()
    await bot.update_balance()

    # Verify divergence exists
    assert bot.balance_raw == pytest.approx(1005.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now verify get_raw_balance / get_hysteresis_snapped_balance route correctly
    assert bot.get_raw_balance() == pytest.approx(1005.0)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_ws_balance_update_sets_balance_raw_correctly():
    """Simulate a WS-triggered balance update and verify balance_raw is set."""
    bot = Passivbot.__new__(Passivbot)
    bot.quote = "USDT"

    # First establish a baseline via REST update_balance
    call_count = [0]

    async def fake_fetch_balance():
        call_count[0] += 1
        return 1000.0

    bot.fetch_balance = fake_fetch_balance
    await bot.update_balance()

    assert bot.balance_raw == pytest.approx(1000.0)
    assert bot.balance == pytest.approx(1000.0)

    # Now simulate a WS balance update: exchange pushes a new raw balance
    # directly onto the bot attributes (as exchange-specific subclasses do).
    bot.balance_raw = 1002.0
    # Snapped balance stays at 1000.0 since WS doesn't re-run hysteresis

    # Verify handle_balance_update sees the divergence and schedules execution
    async def fake_calc_upnl_sum():
        return 50.0

    bot.calc_upnl_sum = fake_calc_upnl_sum
    bot.execution_scheduled = False

    await bot.handle_balance_update(source="WS")

    # After WS update, the raw accessor should return the new value
    assert bot.get_raw_balance() == pytest.approx(1002.0)
    # Snapped balance hasn't changed (no hysteresis re-snap via WS path)
    assert bot.get_hysteresis_snapped_balance() == pytest.approx(1000.0)
    # Execution should have been scheduled due to balance_raw change
    assert bot.execution_scheduled is True


def test_authoritative_barrier_allows_coherent_changed_cycle():
    bot = Passivbot.__new__(Passivbot)

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["changed"] == ["fills", "open_orders", "positions"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["changed"] == []


def test_authoritative_barrier_waits_for_next_epoch_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()
    for surface, sig in (
        ("balance", ("b", 1)),
        ("positions", ("p", 1)),
        ("open_orders", ("o", 1)),
        ("fills", ("f", 1)),
    ):
        bot._record_authoritative_surface(surface, sig)
    bot._authoritative_execution_barrier_state()

    bot._request_authoritative_confirmation({"open_orders"})
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is True
    assert details["missing"] == ["open_orders"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("open_orders", ("o", 1))
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["missing"] == []
    assert getattr(bot, "_authoritative_pending_confirmations", {}) == {}
    assert bot.freshness_ledger.surface_epoch("open_orders") == 2


def test_staged_refresh_plan_defers_fills_until_next_minute(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)
    plan = bot._authoritative_staged_refresh_plan()

    assert plan == {"balance", "positions", "open_orders"}
    assert bot._authoritative_refresh_plan_surfaces == plan

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 180_000)
    plan = bot._authoritative_staged_refresh_plan()

    assert plan == {"balance", "positions", "open_orders", "fills"}


def test_staged_refresh_plan_keeps_pending_fills_even_with_recent_fills(monkeypatch):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {"fills": 2}
    bot.freshness_ledger.stamp("fills", ("fills", "fresh"), now_ms=120_010, epoch=1)

    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: 120_500)

    assert bot._authoritative_staged_refresh_plan() == {
        "balance",
        "positions",
        "open_orders",
        "fills",
    }


def test_authoritative_barrier_uses_current_staged_plan_when_no_pending():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    bot._authoritative_refresh_plan_surfaces = {"balance", "positions", "open_orders"}
    for surface in ("balance", "positions", "open_orders"):
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    blocked, details = bot._authoritative_execution_barrier_state()

    assert blocked is False
    assert details["required"] == ["balance", "open_orders", "positions"]


def test_positions_change_without_fills_requests_full_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 1))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))

    bot._finalize_authoritative_refresh_consistency({"balance", "positions", "open_orders"})

    assert bot._authoritative_pending_confirmations == {
        "balance": 2,
        "positions": 2,
        "open_orders": 2,
        "fills": 2,
    }


def test_staged_planner_preconditions_require_current_epoch_surfaces():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is False
    assert details["missing"] == sorted(ACCOUNT_SURFACES | {"completed_candles"})

    for surface in ACCOUNT_SURFACES | {"completed_candles"}:
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is True
    assert details["missing"] == []

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is False
    assert details["missing"] == ["market_snapshot"]

    bot._record_authoritative_surface("market_snapshot", ("market", "fresh"))
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is True
    assert set(details["required"]) == set(LIVE_STATE_SURFACES)


def test_staged_planner_preconditions_allow_open_orders_only_confirmation_epoch():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_pending_confirmations = {}

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES | {"completed_candles"}:
        bot._record_authoritative_surface(surface, (surface, "baseline"))

    bot._request_authoritative_confirmation({"open_orders"})
    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is True
    assert details["missing"] == ["open_orders"]

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("open_orders", ("open_orders", "confirmed"))
    bot._record_authoritative_surface("completed_candles", ("completed_candles", "current"))

    blocked, details = bot._authoritative_execution_barrier_state()
    assert blocked is False
    assert details["missing"] == []

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=False)
    assert ok is True
    assert details["missing"] == []
    assert bot.freshness_ledger.surface_epoch("balance") == 1
    assert bot.freshness_ledger.surface_epoch("positions") == 1
    assert bot.freshness_ledger.surface_epoch("fills") == 1
    assert bot.freshness_ledger.surface_epoch("open_orders") == 2

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is False
    assert details["missing"] == ["market_snapshot"]

    bot._record_authoritative_surface("market_snapshot", ("market_snapshot", "current"))
    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)
    assert ok is True
    assert details["missing"] == []


def test_staged_planner_preconditions_raise_before_rust_planning():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "bybit"
    bot.freshness_ledger = FreshnessLedger(now_ms=0)

    bot._begin_authoritative_refresh_epoch()

    with pytest.raises(RuntimeError, match="staged planner precondition failed"):
        bot._assert_staged_planner_preconditions(
            include_market_snapshot=False, context="market snapshot refresh"
        )


def test_legacy_planner_preconditions_do_not_require_freshness_ledger():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "legacy"}}
    bot.exchange = "bybit"

    ok, details = bot._staged_planner_precondition_state(include_market_snapshot=True)

    assert ok is True
    assert details["missing"] == []


@pytest.mark.asyncio
async def test_disappeared_self_order_blocks_creations_until_full_freshness():
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 3
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = {}
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.recent_order_cancellations = []

    order = {
        "id": "created-1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "amount": 0.01,
        "price": 99_000.0,
        "reduce_only": False,
    }
    bot.open_orders = {"BTC/USDT:USDT": [order]}
    bot.fetched_open_orders = [order]
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: True
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    ok = await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )

    assert ok is True
    assert bot.execution_scheduled is True
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}
    assert set(bot.freshness_ledger.blocked_symbols()) == {"BTC/USDT:USDT"}
    assert bot.freshness_ledger.blocked_symbols()["BTC/USDT:USDT"].min_epoch == 4
    assert bot._authoritative_pending_confirmations == {
        surface: 4 for surface in ACCOUNT_SURFACES
    }

    to_create, skipped = Passivbot._apply_freshness_creation_guardrails(
        bot,
        [
            {"symbol": "BTC/USDT:USDT", "side": "buy", "position_side": "long"},
            {"symbol": "ETH/USDT:USDT", "side": "buy", "position_side": "long"},
        ],
    )

    assert skipped == 1
    assert [order["symbol"] for order in to_create] == ["ETH/USDT:USDT"]

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    assert bot.freshness_ledger.blocked_symbols() == {}


@pytest.mark.asyncio
async def test_disappeared_self_order_guardrail_blocks_real_plan_create_until_refresh(caplog):
    bot = Passivbot.__new__(Passivbot)
    bot.freshness_ledger = FreshnessLedger(now_ms=0)
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 7
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = {}
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.recent_order_cancellations = []
    bot._last_plan_detail = {}
    bot._order_plan_summary_is_interesting = lambda **kwargs: False
    bot.PB_modes = {"long": {}, "short": {}}
    bot.active_symbols = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    class _CM:
        def get_current_close(self, symbol, max_age_ms=None):
            return 100.0

    bot.cm = _CM()
    async def fake_get_live_last_prices(symbols, **kwargs):
        return {symbol: 100.0 for symbol in symbols}

    bot._get_live_last_prices = fake_get_live_last_prices
    bot.live_value = lambda key: 0.0 if key == "order_match_tolerance_pct" else 0.0

    disappeared_order = {
        "id": "created-1",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.01,
        "amount": 0.01,
        "price": 99_000.0,
        "reduce_only": False,
    }
    bot.open_orders = {"BTC/USDT:USDT": [disappeared_order], "ETH/USDT:USDT": []}
    bot.fetched_open_orders = [disappeared_order]
    bot.order_matches_recent_execution = lambda _order, max_age_ms=None: True
    bot.order_matches_bot_cancellation = lambda _order: False
    bot.order_was_recently_cancelled = lambda _order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    await Passivbot._apply_open_orders_snapshot(
        bot,
        [],
        allow_followup_positions_refresh=False,
    )
    bot.state_change_detected_by_symbol = set()

    ideal_orders = {
        "BTC/USDT:USDT": [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "price": 99_000.0,
                "reduce_only": False,
            }
        ],
        "ETH/USDT:USDT": [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.1,
                "price": 3_000.0,
                "reduce_only": False,
            }
        ],
    }

    async def fake_calc_ideal_orders():
        return ideal_orders

    bot.calc_ideal_orders = fake_calc_ideal_orders

    with caplog.at_level(logging.INFO):
        _to_cancel, to_create = await Passivbot.calc_orders_to_cancel_and_create(bot)

    assert [order["symbol"] for order in to_create] == ["ETH/USDT:USDT"]
    assert "freshness guardrail blocking order creation" in caplog.text

    bot._begin_authoritative_refresh_epoch()
    for surface in ACCOUNT_SURFACES:
        bot._record_authoritative_surface(surface, (surface, "fresh"))

    _to_cancel, to_create = await Passivbot.calc_orders_to_cancel_and_create(bot)

    assert sorted(order["symbol"] for order in to_create) == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
    ]


@pytest.mark.asyncio
async def test_ambiguous_cancel_forces_full_authoritative_confirmation():
    bot = Passivbot.__new__(Passivbot)
    bot.debug_mode = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot._health_orders_cancelled = 0
    confirmations = []

    order = {
        "id": "abc123",
        "symbol": "BTC/USDT:USDT",
        "side": "buy",
        "position_side": "long",
        "qty": 0.001,
        "price": 100_000.0,
        "reduce_only": False,
    }

    async def fake_execute_cancellations(orders):
        assert orders == [order]
        return [
            {
                "status": "success",
                "_passivbot_cancel_requires_full_authoritative_confirmation": True,
            }
        ]

    bot.live_value = lambda key: 10 if key == "max_n_cancellations_per_batch" else None
    bot.add_to_recent_order_cancellations = lambda _order: None
    bot.log_order_action = lambda *args, **kwargs: None
    bot._log_order_action_summary = lambda *args, **kwargs: None
    bot.execute_cancellations = fake_execute_cancellations
    bot.did_cancel_order = lambda executed, _order: executed.get("status") == "success"
    bot.remove_order = lambda *args, **kwargs: None
    bot._monitor_order_payload = lambda *args, **kwargs: {}
    bot._monitor_record_event = lambda *args, **kwargs: None
    bot._request_authoritative_confirmation = lambda surfaces: confirmations.append(set(surfaces))

    res = await Passivbot.execute_cancellations_parent(bot, [order])

    assert len(res) == 1
    assert confirmations == [{"balance", "positions", "open_orders", "fills"}]
    assert bot.state_change_detected_by_symbol == {"BTC/USDT:USDT"}


def test_positions_signature_ignores_margin_used_and_margin_mode_noise():
    bot = Passivbot.__new__(Passivbot)
    positions_a = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.1,
            "price": 100000.0,
            "margin_mode": "cross",
            "margin_used": 50.0,
        }
    ]
    positions_b = [
        {
            "symbol": "BTC/USDT:USDT",
            "position_side": "long",
            "size": 0.1,
            "price": 100000.0,
            "margin_mode": "isolated",
            "margin_used": 55.0,
        }
    ]

    assert bot._positions_signature(positions_a) == bot._positions_signature(positions_b)


def test_authoritative_barrier_does_not_block_on_balance_only_change():
    bot = Passivbot.__new__(Passivbot)

    bot._begin_authoritative_refresh_epoch()
    for surface, sig in (
        ("balance", ("b", 1)),
        ("positions", ("p", 1)),
        ("open_orders", ("o", 1)),
        ("fills", ("f", 1)),
    ):
        bot._record_authoritative_surface(surface, sig)
    bot._authoritative_execution_barrier_state()

    bot._begin_authoritative_refresh_epoch()
    bot._record_authoritative_surface("balance", ("b", 2))
    bot._record_authoritative_surface("positions", ("p", 1))
    bot._record_authoritative_surface("open_orders", ("o", 1))
    bot._record_authoritative_surface("fills", ("f", 1))
    blocked, details = bot._authoritative_execution_barrier_state()

    assert blocked is False
    assert details["changed"] == []


@pytest.mark.asyncio
async def test_run_execution_loop_waits_for_clean_authoritative_cycle_before_execute():
    bot = Passivbot.__new__(Passivbot)
    cycle = {"n": 0}
    executes = []

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = True
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        cycle["n"] += 1
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
        ):
            bot._record_authoritative_surface(surface, sig)
        return True

    async def fake_refresh_market_state_if_needed():
        return True

    async def fake_execute_to_exchange():
        executes.append(cycle["n"])
        return {"executed_cycle": cycle["n"]}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result == {"executed_cycle": 1}
    assert executes == [1]


@pytest.mark.asyncio
async def test_run_execution_loop_stops_before_execute_when_signal_arrives_after_refresh():
    bot = Passivbot.__new__(Passivbot)
    executes = []

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot._begin_authoritative_refresh_epoch()
        for surface, sig in (
            ("balance", ("b", 1)),
            ("positions", ("p", 1)),
            ("open_orders", ("o", 1)),
            ("fills", ("f", 1)),
        ):
            bot._record_authoritative_surface(surface, sig)
        bot.stop_signal_received = True
        return True

    async def fake_refresh_market_state_if_needed():
        return True

    async def fake_execute_to_exchange():
        executes.append(True)
        return {"unexpected": True}

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.refresh_market_state_if_needed = fake_refresh_market_state_if_needed
    bot.execute_to_exchange = fake_execute_to_exchange

    result = await bot.run_execution_loop()

    assert result is None
    assert executes == []


@pytest.mark.asyncio
async def test_run_execution_loop_suppresses_inflight_shutdown_refresh_error(caplog):
    bot = Passivbot.__new__(Passivbot)

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._health_errors = 0
    bot._health_rate_limits = 0
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot._monitor_record_error = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("shutdown errors should not be recorded as runtime errors")
    )
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        bot.stop_signal_received = True
        raise RuntimeError("connector is closed")

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with caplog.at_level(logging.DEBUG):
        result = await bot.run_execution_loop()

    assert result is None
    assert bot._health_errors == 0
    assert bot._health_rate_limits == 0
    bot.restart_bot_on_too_many_errors.assert_not_awaited()
    bot.execute_to_exchange.assert_not_awaited()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any("execution loop stopped during in-flight refresh" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_refresh_authoritative_state_staged_uses_open_orders_only_confirmation_plan():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {"live": {"authoritative_refresh_mode": "staged"}}
    bot.exchange = "fake"
    bot.stop_signal_received = False
    bot.balance_override = None
    bot._balance_override_logged = False
    bot.previous_hysteresis_balance = None
    bot.balance_hysteresis_snap_pct = 0.02
    bot.balance_raw = 0.0
    bot.balance = 0.0
    bot._exchange_reported_balance_raw = 0.0
    bot.fetched_positions = []
    bot.positions = {}
    bot.open_orders = {}
    bot.fetched_open_orders = []
    bot.active_symbols = []
    bot.state_change_detected_by_symbol = set()
    bot.execution_scheduled = False
    bot._authoritative_surface_signatures = {}
    bot._authoritative_surface_generations = {}
    bot._authoritative_refresh_epoch = 0
    bot._authoritative_refresh_epoch_fresh = set()
    bot._authoritative_refresh_epoch_changed = set()
    bot._authoritative_pending_confirmations = {"open_orders": 1}
    bot.recent_order_cancellations = []
    bot.recent_order_executions = [
        {
            "symbol": "BTC/USDT:USDT",
            "side": "buy",
            "position_side": "long",
            "qty": 0.01,
            "price": 99000.0,
            "execution_timestamp": 10**15,
        }
    ]
    bot.fetch_balance = AsyncMock(side_effect=AssertionError("balance fetch should not run"))
    bot.fetch_positions = AsyncMock(side_effect=AssertionError("positions fetch should not run"))
    bot.update_pnls = AsyncMock(side_effect=AssertionError("fills refresh should not run"))
    bot.fetch_open_orders = AsyncMock(
        return_value=[
            {
                "id": "1",
                "symbol": "BTC/USDT:USDT",
                "side": "buy",
                "position_side": "long",
                "qty": 0.01,
                "amount": 0.01,
                "price": 99000.0,
                "timestamp": 1,
                "reduce_only": False,
            }
        ]
    )
    bot.handle_balance_update = AsyncMock()
    bot.order_was_recently_cancelled = lambda order: 0.0
    bot.log_order_action = lambda *args, **kwargs: None
    bot._reconcile_balance_after_open_orders_refresh = lambda: False

    result = await bot.refresh_authoritative_state()
    blocked, details = bot._authoritative_execution_barrier_state()

    assert result is True
    bot.fetch_open_orders.assert_awaited_once()
    assert bot.fetch_balance.await_count == 0
    assert bot.fetch_positions.await_count == 0
    assert bot.update_pnls.await_count == 0
    assert blocked is False
    assert details["missing"] == []


@pytest.mark.asyncio
async def test_run_execution_loop_propagates_fatal_bot_exception():
    bot = Passivbot.__new__(Passivbot)

    bot.stop_signal_received = False
    bot.execution_scheduled = False
    bot.state_change_detected_by_symbol = set()
    bot.debug_mode = False
    bot._equity_hard_stop_enabled = lambda *args, **kwargs: False
    bot._set_log_silence_watchdog_context = lambda *args, **kwargs: None
    bot._maybe_log_health_summary = lambda: None
    bot._maybe_log_unstuck_status = lambda: None
    bot._monitor_flush_snapshot = AsyncMock()
    bot.restart_bot_on_too_many_errors = AsyncMock()
    bot.live_value = lambda key: 0.0 if key == "execution_delay_seconds" else False

    async def fake_refresh_authoritative_state():
        raise FatalBotException("fatal")

    bot.refresh_authoritative_state = fake_refresh_authoritative_state
    bot.execute_to_exchange = AsyncMock()

    with pytest.raises(FatalBotException, match="fatal"):
        await bot.run_execution_loop()
