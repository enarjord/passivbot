"""Tests for the realized-loss gate feature (live.max_realized_loss_pct)."""

import logging
import types
from copy import deepcopy
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from passivbot import Passivbot
from backtest import prep_backtest_args
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill_event(
    pnl: float,
    timestamp: float = 0.0,
    pnl_status: str = "complete",
    fee_paid: float = 0.0,
    pnl_source: str = "authoritative",
) -> types.SimpleNamespace:
    """Create a minimal fill-event namespace with a .pnl attribute."""
    return types.SimpleNamespace(
        pnl=pnl,
        fee_paid=fee_paid,
        timestamp=timestamp,
        pnl_status=pnl_status,
        pnl_source=pnl_source,
        id="test-fill",
        symbol="BTC/USDT:USDT",
        position_side="long",
        pb_order_type="close_grid_long",
    )


class _RiskCache:
    def __init__(
        self,
        *,
        known_gaps=None,
        covered_start_ms=0,
        history_scope="unknown",
        oldest_event_ts=0,
        newest_event_ts=0,
    ):
        self._known_gaps = list(known_gaps or [])
        self._covered_start_ms = covered_start_ms
        self._history_scope = history_scope
        self._oldest_event_ts = oldest_event_ts
        self._newest_event_ts = newest_event_ts

    def get_known_gaps(self):
        return list(self._known_gaps)

    def should_retry_gap(self, gap):
        return int(gap.get("retry_count", 0) or 0) < 3

    def get_covered_start_ms(self):
        return self._covered_start_ms

    def get_history_scope(self):
        return self._history_scope

    def load_metadata(self):
        return {
            "known_gaps": list(self._known_gaps),
            "covered_start_ms": self._covered_start_ms,
            "history_scope": self._history_scope,
            "oldest_event_ts": self._oldest_event_ts,
            "newest_event_ts": self._newest_event_ts,
        }


_DEFAULT_RISK_CACHE = object()


def _make_bot_with_events(events, balance=10000.0, cache=_DEFAULT_RISK_CACHE):
    """Return a Passivbot instance with a mocked FillEventsManager."""
    bot = object.__new__(Passivbot)
    bot._pnls_manager = MagicMock()
    def _get_events(start_ms=None, end_ms=None, symbol=None):
        out = list(events)
        if start_ms is not None:
            out = [ev for ev in out if getattr(ev, "timestamp", 0.0) >= start_ms]
        if end_ms is not None:
            out = [ev for ev in out if getattr(ev, "timestamp", 0.0) <= end_ms]
        if symbol is not None:
            out = [ev for ev in out if getattr(ev, "symbol", None) == symbol]
        return out

    bot._pnls_manager.get_events.side_effect = _get_events
    if cache is _DEFAULT_RISK_CACHE:
        oldest_event_ts = min(
            (int(getattr(ev, "timestamp", 0) or 0) for ev in events),
            default=1,
        )
        cache = _RiskCache(
            covered_start_ms=1,
            history_scope="all",
            oldest_event_ts=max(1, oldest_event_ts),
        )
    bot._pnls_manager.cache = cache
    if cache is not None:
        bot._pnls_manager.cache = cache
        bot._pnls_manager.get_history_scope.side_effect = cache.get_history_scope
    bot.balance = balance
    return bot


def _set_pnl_lookback(bot, *, lookback_days: float | str, now_ms: int) -> None:
    bot.config = {"live": {"pnls_max_lookback_days": lookback_days}}
    bot.get_exchange_time = lambda: now_ms


def _make_replay_fill(
    *,
    timestamp: int,
    pnl: float,
    symbol: str = "BTC/USDT:USDT",
    position_side: str = "long",
    side: str = "sell",
    qty: float = 1.0,
    price: float = 100.0,
    fee_paid: float = 0.0,
    pb_order_type: str = "close_grid_long",
) -> dict:
    return {
        "timestamp": int(timestamp),
        "symbol": symbol,
        "position_side": position_side,
        "side": side,
        "qty": qty,
        "price": price,
        "pnl": pnl,
        "fee_paid": fee_paid,
        "pb_order_type": pb_order_type,
    }


def _rust_effective_pnl_cumsum_reference(events, *, start_ms=None):
    active = [
        ev for ev in events if start_ms is None or getattr(ev, "timestamp", 0.0) >= start_ms
    ]
    if not active:
        return {"max": 0.0, "last": 0.0}
    cumsum = 0.0
    peak = 0.0
    for ev in active:
        cumsum += float(getattr(ev, "pnl", 0.0) or 0.0) + float(
            getattr(ev, "fee_paid", 0.0) or 0.0
        )
        peak = max(peak, cumsum)
    return {"max": peak, "last": cumsum}


def _make_bot_for_logging():
    """Return a Passivbot instance with throttle state initialized."""
    bot = object.__new__(Passivbot)
    bot._loss_gate_last_log_ms = {}
    bot._loss_gate_log_interval_ms = 5 * 60 * 1000
    return bot


# ---------------------------------------------------------------------------
# _get_realized_pnl_cumsum_stats
# ---------------------------------------------------------------------------


class TestGetRealizedPnlCumsumStats:
    def test_no_manager_returns_zeros(self):
        bot = object.__new__(Passivbot)
        bot._pnls_manager = None
        result = bot._get_realized_pnl_cumsum_stats()
        assert result == {"max": 0.0, "last": 0.0}

    def test_empty_events_returns_zeros(self):
        bot = _make_bot_with_events([])
        result = bot._get_realized_pnl_cumsum_stats()
        assert result == {"max": 0.0, "last": 0.0}

    def test_single_positive_event(self):
        bot = _make_bot_with_events([_make_fill_event(50.0)])
        result = bot._get_realized_pnl_cumsum_stats()
        assert result["max"] == pytest.approx(50.0)
        assert result["last"] == pytest.approx(50.0)

    def test_realized_loss_gate_uses_net_pnl_with_fee_paid(self):
        bot = _make_bot_with_events([_make_fill_event(50.0, fee_paid=-5.0)])
        result = bot._get_realized_pnl_cumsum_stats()
        assert result["max"] == pytest.approx(45.0)
        assert result["last"] == pytest.approx(45.0)

    def test_cumsum_peak_differs_from_last(self):
        events = [
            _make_fill_event(100.0),
            _make_fill_event(-60.0),
            _make_fill_event(10.0),
        ]
        # cumsum: [100, 40, 50] → max=100, last=50
        bot = _make_bot_with_events(events)
        result = bot._get_realized_pnl_cumsum_stats()
        assert result["max"] == pytest.approx(100.0)
        assert result["last"] == pytest.approx(50.0)

    def test_all_negative_events(self):
        events = [_make_fill_event(-10.0), _make_fill_event(-20.0)]
        # cumsum: [-10, -30] with an explicit zero starting-balance peak.
        bot = _make_bot_with_events(events)
        result = bot._get_realized_pnl_cumsum_stats()
        assert result["max"] == pytest.approx(0.0)
        assert result["last"] == pytest.approx(-30.0)

    def test_uses_only_events_inside_configured_lookback_window(self):
        now_ms = 10 * 86_400_000
        events = [
            _make_fill_event(100.0, timestamp=now_ms - 3 * 86_400_000),
            _make_fill_event(-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
            _make_fill_event(10.0, timestamp=now_ms - 60_000),
            _make_fill_event(-5.0, timestamp=now_ms - 30_000),
        ]
        bot = _make_bot_with_events(events)
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        result = bot._get_realized_pnl_cumsum_stats()

        assert result["max"] == pytest.approx(10.0)
        assert result["last"] == pytest.approx(5.0)

    def test_zero_lookback_uses_minimal_history(self):
        now_ms = 10 * 86_400_000
        events = [
            _make_fill_event(100.0, timestamp=now_ms - 3 * 86_400_000),
            _make_fill_event(-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
            _make_fill_event(10.0, timestamp=now_ms - 60_000),
        ]
        bot = _make_bot_with_events(events)
        _set_pnl_lookback(bot, lookback_days=0.0, now_ms=now_ms)

        result = bot._get_realized_pnl_cumsum_stats()

        assert result["max"] == pytest.approx(0.0)
        assert result["last"] == pytest.approx(0.0)

    def test_all_lookback_uses_full_history(self):
        now_ms = 10 * 86_400_000
        events = [
            _make_fill_event(100.0, timestamp=now_ms - 3 * 86_400_000),
            _make_fill_event(-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
            _make_fill_event(10.0, timestamp=now_ms - 60_000),
        ]
        bot = _make_bot_with_events(events)
        _set_pnl_lookback(bot, lookback_days="all", now_ms=now_ms)

        result = bot._get_realized_pnl_cumsum_stats()

        assert result["max"] == pytest.approx(100.0)
        assert result["last"] == pytest.approx(30.0)

    def test_live_cumsum_matches_rust_effective_contract_reference(self):
        now_ms = 10 * 86_400_000
        events = [
            _make_fill_event(100.0, timestamp=now_ms - 4 * 86_400_000, fee_paid=-2.0),
            _make_fill_event(50.0, timestamp=now_ms - 2 * 86_400_000),
            _make_fill_event(-120.0, timestamp=now_ms - 86_400_000, fee_paid=-1.5),
            _make_fill_event(190.0, timestamp=now_ms - 60_000, fee_paid=-0.5),
        ]
        bot = _make_bot_with_events(events)
        _set_pnl_lookback(bot, lookback_days=2.0, now_ms=now_ms)
        start_ms = now_ms - 2 * 86_400_000

        result = bot._get_realized_pnl_cumsum_stats()
        expected = _rust_effective_pnl_cumsum_reference(events, start_ms=start_ms)

        assert result["max"] == pytest.approx(expected["max"])
        assert result["last"] == pytest.approx(expected["last"])

    def test_pending_close_pnl_fails_loudly(self):
        bot = _make_bot_with_events([_make_fill_event(0.0, pnl_status="pending")])

        with pytest.raises(RuntimeError, match="realized PnL pending"):
            bot._get_realized_pnl_cumsum_stats()

    def test_degraded_synthetic_pnl_fails_loudly(self):
        bot = _make_bot_with_events(
            [
                _make_fill_event(
                    0.0,
                    pnl_source="synthetic_fill_reconstruction_degraded",
                )
            ]
        )

        with pytest.raises(RuntimeError, match="degraded realized PnL"):
            bot._get_realized_pnl_cumsum_stats()

    def test_missing_cache_blocks_risk_history_coverage(self):
        now_ms = 10 * 86_400_000
        bot = _make_bot_with_events(
            [_make_fill_event(10.0, timestamp=now_ms - 60_000)],
            cache=None,
        )
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="missing FillEventsManager cache"):
            bot._get_realized_pnl_cumsum_stats()

    def test_known_gap_overlapping_lookback_fails_loudly(self):
        now_ms = 10 * 86_400_000
        start_ms = now_ms - 86_400_000
        cache = _RiskCache(
            covered_start_ms=start_ms,
            oldest_event_ts=start_ms,
            known_gaps=[
                {
                    "start_ts": start_ms + 60_000,
                    "end_ts": start_ms + 120_000,
                    "retry_count": 3,
                    "reason": "fetch_failed",
                    "confidence": 0.0,
                }
            ],
        )
        bot = _make_bot_with_events(
            [_make_fill_event(10.0, timestamp=start_ms + 180_000)],
            cache=cache,
        )
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="fill history gap overlaps risk lookback"):
            bot._get_realized_pnl_cumsum_stats()

    def test_uncovered_empty_history_fails_loudly(self):
        now_ms = 10 * 86_400_000
        bot = _make_bot_with_events([], cache=_RiskCache(history_scope="window"))
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="fill history coverage unknown"):
            bot._get_realized_pnl_cumsum_stats()

    def test_oldest_event_without_coverage_does_not_prove_risk_lookback(self):
        now_ms = 10 * 86_400_000
        start_ms = now_ms - 86_400_000
        bot = _make_bot_with_events(
            [_make_fill_event(10.0, timestamp=start_ms - 60_000)],
            cache=_RiskCache(
                covered_start_ms=0,
                history_scope="window",
                oldest_event_ts=start_ms - 60_000,
            ),
        )
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="fill history coverage unknown"):
            bot._get_realized_pnl_cumsum_stats()

    def test_covered_empty_history_returns_zero_cumsum(self):
        now_ms = 10 * 86_400_000
        start_ms = now_ms - 86_400_000
        bot = _make_bot_with_events(
            [],
            cache=_RiskCache(
                covered_start_ms=start_ms,
                history_scope="window",
            ),
        )
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        assert bot._get_realized_pnl_cumsum_stats() == {"max": 0.0, "last": 0.0}

    def test_coin_hsl_uses_risk_history_gate_for_degraded_pnl(self):
        hsl = pytest.importorskip(
            "passivbot_hsl", reason="live HSL dependencies not available"
        )
        now_ms = 10 * 86_400_000
        start_ms = now_ms - 86_400_000
        bot = _make_bot_with_events(
            [
                _make_fill_event(
                    -10.0,
                    timestamp=start_ms + 60_000,
                    pnl_source="synthetic_fill_reconstruction_degraded",
                )
            ],
            cache=_RiskCache(
                covered_start_ms=start_ms,
                history_scope="window",
            ),
        )
        _set_pnl_lookback(bot, lookback_days=1.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="degraded realized PnL"):
            hsl._equity_hard_stop_coin_realized_pnl_peak_last(
                bot,
                "long",
                "BTC/USDT:USDT",
                now_ms,
            )

    def test_equity_hard_stop_uses_net_pnl_with_fee_paid(self):
        bot = _make_bot_with_events([_make_fill_event(50.0, fee_paid=-5.0)])
        result = bot._equity_hard_stop_realized_pnl_now()
        assert result == pytest.approx(45.0)

    def test_equity_hard_stop_blocks_unified_replay_with_balance_override(self):
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"hsl_signal_mode": "unified"}}
        bot.balance_override = 1000.0
        bot.hsl = {
            "long": {"enabled": True},
            "short": {"enabled": False},
        }

        with pytest.raises(RuntimeError, match="unsafe with balance_override"):
            bot._equity_hard_stop_validate_balance_source_for_history_replay()

    def test_equity_hard_stop_balance_override_guard_emits_replay_failed_event(self):
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"hsl_signal_mode": "unified"}}
        bot.balance_override = 1000.0
        bot.hsl = {
            "long": {"enabled": True},
            "short": {"enabled": False},
        }
        sink = ListEventSink()
        bot._live_event_current_cycle_id = "cy_hsl_guard"
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[sink],
            monitor_sinks=[],
        )
        bot._emit_live_event = MethodType(Passivbot._emit_live_event, bot)

        with pytest.raises(RuntimeError, match="unsafe with balance_override"):
            bot._equity_hard_stop_validate_balance_source_for_history_replay()

        assert bot._live_event_pipeline.flush(timeout=2.0) is True
        events = [event for event in sink.events if event.event_type.startswith("hsl.replay.")]
        assert [event.event_type for event in events] == [EventTypes.HSL_REPLAY_FAILED]
        assert events[0].level == "critical"
        assert events[0].status == "failed"
        assert (
            events[0].reason_code
            == "hsl_balance_override_account_level_replay_unsafe"
        )
        assert events[0].cycle_id == "cy_hsl_guard"
        assert events[0].data == {
            "signal_mode": "unified",
            "balance_override_active": True,
            "enabled_psides": ["long"],
        }
        assert bot._live_event_pipeline.close(timeout=2.0) is True

    def test_equity_hard_stop_allows_coin_replay_with_balance_override(self):
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"hsl_signal_mode": "coin"}}
        bot.balance_override = 1000.0
        bot.hsl = {
            "long": {"enabled": True},
            "short": {"enabled": True},
        }

        bot._equity_hard_stop_validate_balance_source_for_history_replay()

    @pytest.mark.asyncio
    async def test_equity_hard_stop_startup_guard_runs_before_history_replay(self):
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"hsl_signal_mode": "pside"}}
        bot.balance_override = 1000.0
        bot.hsl = {
            "long": {"enabled": True},
            "short": {"enabled": False},
        }

        async def fail_if_called(*_args, **_kwargs):
            raise AssertionError("history replay should not start")

        bot.get_balance_equity_history = fail_if_called

        with pytest.raises(RuntimeError, match="false RED panic"):
            await bot._equity_hard_stop_initialize_from_history()

    @pytest.mark.asyncio
    async def test_balance_equity_history_realized_pnl_is_lookback_anchored(self):
        day_ms = 86_400_000
        now_ms = 1_782_700_043_210
        lookback_start_ms = now_ms - day_ms
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
        bot.init_pnls = AsyncMock()
        bot.get_exchange_time = lambda: now_ms
        bot.get_raw_balance = lambda: 1000.0
        bot.live_value = lambda key: bot.config["live"][key]
        bot.c_mults = {"BTC/USDT:USDT": 1.0}
        bot.positions = {}
        bot.cm = None
        bot.inverse = False
        bot.exchange = "binance"
        bot.user = "test_user"
        bot.get_symbol_id_inv = lambda symbol: symbol
        bot._pnls_manager = None

        history = await bot.get_balance_equity_history(
            fill_events=[
                _make_replay_fill(timestamp=lookback_start_ms - 60_000, pnl=200.0),
                _make_replay_fill(timestamp=lookback_start_ms + 60_000, pnl=10.0),
            ],
            current_balance=1000.0,
        )

        assert history["timeline"]
        last = history["timeline"][-1]
        assert last["balance"] == pytest.approx(1000.0)
        assert last["realized_pnl"] == pytest.approx(10.0)
        assert last["realized_pnl_long"] == pytest.approx(10.0)
        assert last["realized_pnl_short"] == pytest.approx(0.0)
        assert last["realized_pnl_by_coin_pside"]["BTC/USDT:USDT"]["long"] == pytest.approx(
            10.0
        )

    @pytest.mark.asyncio
    async def test_balance_equity_history_excludes_same_minute_pre_lookback_pnl(self):
        day_ms = 86_400_000
        now_ms = 1_782_700_043_210
        lookback_start_ms = now_ms - day_ms
        pre_window_ts = lookback_start_ms - 10_000
        in_window_ts = lookback_start_ms + 10_000
        assert pre_window_ts // 60_000 == in_window_ts // 60_000
        bot = object.__new__(Passivbot)
        bot.config = {"live": {"pnls_max_lookback_days": 1.0}}
        bot.init_pnls = AsyncMock()
        bot.get_exchange_time = lambda: now_ms
        bot.get_raw_balance = lambda: 1000.0
        bot.live_value = lambda key: bot.config["live"][key]
        bot.c_mults = {"BTC/USDT:USDT": 1.0}
        bot.positions = {}
        bot.cm = None
        bot.inverse = False
        bot.exchange = "binance"
        bot.user = "test_user"
        bot.get_symbol_id_inv = lambda symbol: symbol
        bot._pnls_manager = None

        history = await bot.get_balance_equity_history(
            fill_events=[
                _make_replay_fill(timestamp=pre_window_ts, pnl=200.0),
                _make_replay_fill(timestamp=in_window_ts, pnl=10.0),
            ],
            current_balance=1000.0,
        )

        last = history["timeline"][-1]
        assert last["balance"] == pytest.approx(1000.0)
        assert last["realized_pnl"] == pytest.approx(10.0)
        assert last["realized_pnl_long"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _log_realized_loss_gate_blocks
# ---------------------------------------------------------------------------


class TestLogRealizedLossGateBlocks:
    def test_no_diagnostics_is_silent(self, caplog):
        bot = _make_bot_for_logging()
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks({}, {})
        assert caplog.text == ""

    def test_empty_blocks_is_silent(self, caplog):
        bot = _make_bot_for_logging()
        out = {"diagnostics": {"loss_gate_blocks": []}}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, {})
        assert caplog.text == ""

    def test_block_emits_risk_warning(self, caplog):
        bot = _make_bot_for_logging()
        block = {
            "symbol_idx": 0,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
            "qty": -1.5,
            "price": 80.0,
            "projected_pnl": -200.0,
            "projected_balance_after": 9800.0,
            "balance_floor": 9900.0,
            "max_realized_loss_pct": 0.01,
        }
        out = {"diagnostics": {"loss_gate_blocks": [block]}}
        idx_to_symbol = {0: "BTCUSDT"}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, idx_to_symbol)
        assert "[risk] order blocked by realized-loss gate" in caplog.text
        assert " BTC long " in caplog.text
        assert "close_auto_reduce_wel_long" in caplog.text

    def test_block_emits_structured_event(self, caplog):
        bot = _make_bot_for_logging()
        sink = ListEventSink()
        bot.bot_id = "bot_loss_gate"
        bot.exchange = "binance"
        bot.user = "binance_01"
        bot._live_event_current_cycle_id = "cy_loss_gate"
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[sink],
            monitor_sinks=[],
        )
        block = {
            "symbol_idx": 0,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
            "qty": -1.5,
            "price": 80.0,
            "projected_pnl": -200.0,
            "projected_balance_after": 9800.0,
            "balance_floor": 9900.0,
            "max_realized_loss_pct": 0.01,
        }
        try:
            with caplog.at_level(logging.WARNING):
                bot._log_realized_loss_gate_blocks(
                    {"diagnostics": {"loss_gate_blocks": [block]}},
                    {0: "BTCUSDT"},
                )
            assert bot._live_event_pipeline.flush(timeout=2.0) is True
        finally:
            assert bot._live_event_pipeline.close(timeout=2.0) is True

        assert "[risk] order blocked by realized-loss gate" in caplog.text
        events = [
            event
            for event in sink.events
            if event.event_type == EventTypes.REALIZED_LOSS_GATE_BLOCKED
        ]
        assert len(events) == 1
        event = events[0]
        assert event.level == "warning"
        assert event.status == "deferred"
        assert event.reason_code == "realized_loss_gate_blocked"
        assert event.cycle_id == "cy_loss_gate"
        assert event.symbol == "BTCUSDT"
        assert event.pside == "long"
        assert event.data["order_type"] == "close_auto_reduce_wel_long"
        assert event.data["projected_pnl"] == pytest.approx(-200.0)
        assert event.data["projected_balance_after"] == pytest.approx(9800.0)
        assert event.data["balance_floor"] == pytest.approx(9900.0)
        assert event.data["max_realized_loss_pct"] == pytest.approx(0.01)

    def test_unknown_symbol_idx_logs_unknown(self, caplog):
        bot = _make_bot_for_logging()
        block = {
            "symbol_idx": 99,
            "pside": "short",
            "order_type": "close_unstuck_short",
            "qty": 2.0,
            "price": 50.0,
            "projected_pnl": -100.0,
            "projected_balance_after": 9900.0,
            "balance_floor": 9950.0,
            "max_realized_loss_pct": 0.005,
        }
        out = {"diagnostics": {"loss_gate_blocks": [block]}}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, {})
        assert "unknown" in caplog.text

    def test_non_dict_blocks_skipped(self, caplog):
        bot = _make_bot_for_logging()
        out = {"diagnostics": {"loss_gate_blocks": ["not_a_dict"]}}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, {})
        assert "[risk]" not in caplog.text

    def test_non_dict_output_is_silent(self, caplog):
        bot = _make_bot_for_logging()
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks("not_a_dict", {})
        assert caplog.text == ""

    def test_throttle_suppresses_repeated_logs(self, caplog):
        bot = _make_bot_for_logging()
        block = {
            "symbol_idx": 0,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
            "qty": -1.0,
            "price": 80.0,
            "projected_pnl": -100.0,
            "projected_balance_after": 9900.0,
            "balance_floor": 9950.0,
            "max_realized_loss_pct": 0.01,
        }
        out = {"diagnostics": {"loss_gate_blocks": [block]}}
        idx_to_symbol = {0: "BTCUSDT"}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, idx_to_symbol)
            bot._log_realized_loss_gate_blocks(out, idx_to_symbol)
            bot._log_realized_loss_gate_blocks(out, idx_to_symbol)
        assert caplog.text.count("[risk] order blocked by realized-loss gate") == 1

    def test_throttle_allows_different_symbols(self, caplog):
        bot = _make_bot_for_logging()
        block_btc = {
            "symbol_idx": 0,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
            "qty": -1.0,
            "price": 80.0,
            "projected_pnl": -100.0,
            "projected_balance_after": 9900.0,
            "balance_floor": 9950.0,
            "max_realized_loss_pct": 0.01,
        }
        block_sui = {
            "symbol_idx": 1,
            "pside": "long",
            "order_type": "close_auto_reduce_wel_long",
            "qty": -100.0,
            "price": 0.9,
            "projected_pnl": -50.0,
            "projected_balance_after": 9950.0,
            "balance_floor": 9950.0,
            "max_realized_loss_pct": 0.01,
        }
        out = {"diagnostics": {"loss_gate_blocks": [block_btc, block_sui]}}
        idx_to_symbol = {0: "BTCUSDT", 1: "SUIUSDT"}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, idx_to_symbol)
        assert " BTC long " in caplog.text
        assert " SUI long " in caplog.text


class TestEntryCooldownDeltaGuardEvents:
    def _make_bot_with_event_sink(self):
        bot = _make_bot_for_logging()
        sink = ListEventSink()
        bot.bot_id = "bot_entry_cooldown"
        bot.exchange = "binance"
        bot.user = "binance_01"
        bot._live_event_current_cycle_id = "cy_entry_cooldown"
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[sink],
            monitor_sinks=[],
        )
        return bot, sink

    def test_position_delta_anchor_emits_structured_event(self, caplog):
        bot, sink = self._make_bot_with_event_sink()
        try:
            with caplog.at_level(logging.WARNING):
                bot._log_entry_cooldown_delta_guard(
                    symbol="BTCUSDT",
                    pside="long",
                    previous_abs_size=1.0,
                    current_abs_size=1.5,
                    qty_step=0.001,
                    epsilon=0.0005,
                    now_ms=123456,
                )
            assert bot._live_event_pipeline.flush(timeout=2.0) is True
        finally:
            assert bot._live_event_pipeline.close(timeout=2.0) is True

        assert "entry cooldown position-delta guard anchored" in caplog.text
        events = [
            event
            for event in sink.events
            if event.event_type == EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED
        ]
        assert len(events) == 1
        event = events[0]
        assert event.level == "warning"
        assert event.status == "deferred"
        assert event.reason_code == "entry_cooldown_position_delta"
        assert event.cycle_id == "cy_entry_cooldown"
        assert event.symbol == "BTCUSDT"
        assert event.pside == "long"
        assert event.data["previous_abs_size"] == pytest.approx(1.0)
        assert event.data["current_abs_size"] == pytest.approx(1.5)
        assert event.data["qty_step"] == pytest.approx(0.001)
        assert event.data["epsilon"] == pytest.approx(0.0005)
        assert event.data["anchor_ts_ms"] == 123456
        assert event.data["fallback_source"] == "exchange_position_delta"
        assert event.data["text_log_emitted"] is True

    def test_position_delta_anchor_event_emits_when_text_warning_throttled(self, caplog):
        bot, sink = self._make_bot_with_event_sink()
        try:
            with caplog.at_level(logging.WARNING):
                bot._log_entry_cooldown_delta_guard(
                    symbol="BTCUSDT",
                    pside="long",
                    previous_abs_size=1.0,
                    current_abs_size=1.5,
                    qty_step=0.001,
                    epsilon=0.0005,
                    now_ms=123456,
                )
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                bot._log_entry_cooldown_delta_guard(
                    symbol="BTCUSDT",
                    pside="long",
                    previous_abs_size=1.5,
                    current_abs_size=2.0,
                    qty_step=0.001,
                    epsilon=0.0005,
                    now_ms=124456,
                )
            assert bot._live_event_pipeline.flush(timeout=2.0) is True
        finally:
            assert bot._live_event_pipeline.close(timeout=2.0) is True

        assert "entry cooldown position-delta guard anchored" not in caplog.text
        events = [
            event
            for event in sink.events
            if event.event_type == EventTypes.RISK_ENTRY_COOLDOWN_DELTA_ANCHORED
        ]
        assert len(events) == 2
        assert events[0].data["text_log_emitted"] is True
        assert events[1].data["text_log_emitted"] is False
        assert events[1].data["previous_abs_size"] == pytest.approx(1.5)
        assert events[1].data["current_abs_size"] == pytest.approx(2.0)
        assert events[1].data["anchor_ts_ms"] == 124456


# ---------------------------------------------------------------------------
# prep_backtest_args passthrough
# ---------------------------------------------------------------------------


class TestPrepBacktestArgsMaxRealizedLossPct:
    def _make_config(self, max_realized_loss_pct=None):
        hsl_long = {
            "hsl_enabled": False,
            "hsl_red_threshold": 0.25,
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 0.0,
            "hsl_no_restart_drawdown_threshold": 1.0,
            "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "hsl_panic_close_order_type": "market",
        }
        hsl_short = deepcopy(hsl_long)
        config = {
            "backtest": {
                "coins": {"binance": ["BTC"]},
                "starting_balance": 10000,
                "btc_collateral_cap": 0.5,
                "btc_collateral_ltv_cap": None,
                "filter_by_min_effective_cost": False,
                "dynamic_wel_by_tradability": True,
            },
            "bot": {
                "long": {
                    **hsl_long,
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.5,
                },
                "short": {
                    **hsl_short,
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 0.0,
                    "wallet_exposure_limit": 0.0,
                },
            },
            "live": {
                "approved_coins": {"long": ["BTC"], "short": []},
                "hedge_mode": True,
                "max_realized_loss_pct": 1.0,
                "pnls_max_lookback_days": 30.0,
            },
            "coin_overrides": {},
        }
        if max_realized_loss_pct is not None:
            config["live"]["max_realized_loss_pct"] = max_realized_loss_pct
        return config

    def _make_mss(self):
        return {
            "BTC": {
                "qty_step": 0.001,
                "price_step": 0.01,
                "min_qty": 0.001,
                "min_cost": 10.0,
                "c_mult": 1.0,
                "maker": 0.0002,
                "taker": 0.0005,
            }
        }

    def test_default_value_is_1(self):
        config = self._make_config()
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(1.0)
        assert bp["pnls_max_lookback_days"] == pytest.approx(30.0)

    def test_explicit_value_passthrough(self):
        config = self._make_config(max_realized_loss_pct=0.05)
        config["live"]["pnls_max_lookback_days"] = 14
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.05)
        assert bp["pnls_max_lookback_days"] == pytest.approx(14.0)

    def test_zero_disables_lossy_closes(self):
        config = self._make_config(max_realized_loss_pct=0.0)
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.0)


class TestPrepBacktestArgsEquityHardStopLoss:
    def _make_config(self, hard_stop_block=None):
        hsl_long = {
            "hsl_enabled": False,
            "hsl_red_threshold": 0.25,
            "hsl_ema_span_minutes": 60.0,
            "hsl_cooldown_minutes_after_red": 0.0,
            "hsl_no_restart_drawdown_threshold": 1.0,
            "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
            "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
            "hsl_panic_close_order_type": "market",
        }
        hsl_short = deepcopy(hsl_long)
        config = {
            "backtest": {
                "coins": {"binance": ["BTC"]},
                "starting_balance": 10000,
                "btc_collateral_cap": 0.5,
                "btc_collateral_ltv_cap": None,
                "filter_by_min_effective_cost": False,
                "dynamic_wel_by_tradability": True,
            },
            "bot": {
                "long": {
                    **hsl_long,
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.5,
                },
                "short": {
                    **hsl_short,
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 0.0,
                    "wallet_exposure_limit": 0.0,
                },
            },
            "live": {
                "approved_coins": {"long": ["BTC"], "short": []},
                "hedge_mode": True,
                "max_realized_loss_pct": 1.0,
                "pnls_max_lookback_days": 30.0,
            },
            "coin_overrides": {},
        }
        if hard_stop_block is not None:
            merged = {
                "enabled": bool(config["bot"]["long"]["hsl_enabled"]),
                "red_threshold": float(config["bot"]["long"]["hsl_red_threshold"]),
                "ema_span_minutes": float(config["bot"]["long"]["hsl_ema_span_minutes"]),
                "cooldown_minutes_after_red": float(
                    config["bot"]["long"]["hsl_cooldown_minutes_after_red"]
                ),
                "no_restart_drawdown_threshold": float(
                    config["bot"]["long"]["hsl_no_restart_drawdown_threshold"]
                ),
                "tier_ratios": deepcopy(config["bot"]["long"]["hsl_tier_ratios"]),
                "orange_tier_mode": str(config["bot"]["long"]["hsl_orange_tier_mode"]),
                "panic_close_order_type": str(config["bot"]["long"]["hsl_panic_close_order_type"]),
            }
            for key, value in hard_stop_block.items():
                if key == "tier_ratios" and isinstance(value, dict):
                    merged["tier_ratios"].update(value)
                else:
                    merged[key] = value
            config["bot"]["long"]["hsl_enabled"] = merged["enabled"]
            config["bot"]["long"]["hsl_red_threshold"] = merged["red_threshold"]
            config["bot"]["long"]["hsl_ema_span_minutes"] = merged["ema_span_minutes"]
            config["bot"]["long"]["hsl_cooldown_minutes_after_red"] = merged[
                "cooldown_minutes_after_red"
            ]
            config["bot"]["long"]["hsl_no_restart_drawdown_threshold"] = merged[
                "no_restart_drawdown_threshold"
            ]
            config["bot"]["long"]["hsl_tier_ratios"] = merged["tier_ratios"]
            config["bot"]["long"]["hsl_orange_tier_mode"] = merged["orange_tier_mode"]
            config["bot"]["long"]["hsl_panic_close_order_type"] = merged["panic_close_order_type"]
        return config

    def _make_mss(self):
        return {
            "BTC": {
                "qty_step": 0.001,
                "price_step": 0.01,
                "min_qty": 0.001,
                "min_cost": 10.0,
                "c_mult": 1.0,
                "maker": 0.0002,
                "taker": 0.0005,
            }
        }

    def test_defaults_passthrough(self):
        config = self._make_config()
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        hs = bp["equity_hard_stop_loss"]
        assert hs["enabled"] is False
        assert hs["red_threshold"] == pytest.approx(0.25)
        assert hs["ema_span_minutes"] == pytest.approx(60.0)
        assert hs["cooldown_minutes_after_red"] == pytest.approx(0.0)
        assert hs["no_restart_drawdown_threshold"] == pytest.approx(1.0)
        assert hs["tier_ratios"]["yellow"] == pytest.approx(0.5)
        assert hs["tier_ratios"]["orange"] == pytest.approx(0.75)
        assert hs["orange_tier_mode"] == "tp_only_with_active_entry_cancellation"
        assert hs["panic_close_order_type"] == "market"

    def test_custom_passthrough(self):
        config = self._make_config(
            {
                "enabled": True,
                "red_threshold": 0.3,
                "ema_span_minutes": 45.0,
                "cooldown_minutes_after_red": 30.0,
                "no_restart_drawdown_threshold": 0.6,
                "tier_ratios": {"yellow": 0.55, "orange": 0.8},
                "orange_tier_mode": "graceful_stop",
                "panic_close_order_type": "limit",
            }
        )
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        hs = bp["equity_hard_stop_loss"]
        assert hs["enabled"] is True
        assert hs["red_threshold"] == pytest.approx(0.3)
        assert hs["ema_span_minutes"] == pytest.approx(45.0)
        assert hs["cooldown_minutes_after_red"] == pytest.approx(30.0)
        assert hs["no_restart_drawdown_threshold"] == pytest.approx(0.6)
        assert hs["tier_ratios"]["yellow"] == pytest.approx(0.55)
        assert hs["tier_ratios"]["orange"] == pytest.approx(0.8)
        assert hs["orange_tier_mode"] == "graceful_stop"
        assert hs["panic_close_order_type"] == "limit"

    def test_invalid_tier_ratios_raise(self):
        config = self._make_config(
            {
                "enabled": True,
                "red_threshold": 0.3,
                "ema_span_minutes": 30.0,
                "tier_ratios": {"yellow": 0.9, "orange": 0.8},
            }
        )
        with pytest.raises(ValueError, match="tier_ratios"):
            prep_backtest_args(config, self._make_mss(), "binance")

    def test_negative_cooldown_raises(self):
        config = self._make_config(
            {
                "enabled": True,
                "red_threshold": 0.3,
                "ema_span_minutes": 30.0,
                "cooldown_minutes_after_red": -1.0,
            }
        )
        with pytest.raises(ValueError, match="cooldown_minutes_after_red"):
            prep_backtest_args(config, self._make_mss(), "binance")

    def test_no_restart_drawdown_threshold_below_red_clamps_to_red(self):
        config = self._make_config(
            {
                "enabled": True,
                "red_threshold": 0.3,
                "ema_span_minutes": 30.0,
                "no_restart_drawdown_threshold": 0.2,
            }
        )
        _, _, _, backtest_params = prep_backtest_args(config, self._make_mss(), "binance")
        assert backtest_params["equity_hard_stop_loss"]["no_restart_drawdown_threshold"] == pytest.approx(
            0.3
        )
