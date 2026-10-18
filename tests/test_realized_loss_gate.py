"""Tests for the realized-loss gate feature (live.max_realized_loss_pct)."""

import json
import logging
import types
from copy import deepcopy
from types import MethodType
from unittest.mock import AsyncMock, MagicMock

import pytest

from passivbot import Passivbot
from backtest import prep_backtest_args
from fill_events_manager import FillEventsManager
from live.event_bus import EventTypes, ListEventSink, LiveEventPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill_event(
    pnl: float,
    timestamp: int = 0,
    pnl_status: str = "complete",
    fee_paid: float = 0.0,
    pnl_source: str = "authoritative",
    event_id: str | None = None,
) -> types.SimpleNamespace:
    """Create a minimal fill-event namespace with a .pnl attribute."""
    return types.SimpleNamespace(
        pnl=pnl,
        fee_paid=fee_paid,
        timestamp=timestamp,
        pnl_status=pnl_status,
        pnl_source=pnl_source,
        id=event_id or f"fill-{timestamp}-{pnl}-{fee_paid}",
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
    bot._orchestrator_uses_realized_pnl = lambda: True
    bot._unstuck_uses_realized_pnl = lambda: False
    bot._live_max_realized_loss_pct = lambda: 0.99
    bot.config = {"live": {
        "lookback_ema_span_minutes": 3.0,
        "risk_bootstrap_max_lookback_minutes": 129600.0,
    }}
    bot.get_exchange_time = lambda: 0
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
    bot._pnls_manager._events = list(events)
    if cache is _DEFAULT_RISK_CACHE:
        oldest_event_ts = min(
            (int(getattr(ev, "timestamp", 0) or 0) for ev in events),
            default=0,
        )
        cache = _RiskCache(
            covered_start_ms=1,
            history_scope="all",
            oldest_event_ts=oldest_event_ts,
        )
    bot._pnls_manager.cache = cache
    if cache is not None:
        bot._pnls_manager.cache = cache
        bot._pnls_manager.get_history_scope.side_effect = cache.get_history_scope
        bot._pnls_manager.get_coverage_status.side_effect = (
            lambda **kwargs: FillEventsManager.get_coverage_status(
                bot._pnls_manager,
                **kwargs,
            )
        )
    else:
        bot._pnls_manager.get_coverage_status.return_value = {
            "ready": False,
            "reason": "missing_cache",
            "history_scope": "unknown",
            "covered_start_ms": 0,
            "oldest_event_ts": 0,
        }
    bot.balance = balance
    return bot


def _set_memory_horizon(bot, *, minutes: float, now_ms: int) -> None:
    bot.config = {"live": {
        "lookback_ema_span_minutes": minutes / 5.0,
        "risk_bootstrap_max_lookback_minutes": minutes,
    }}
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


def _make_bot_for_logging():
    """Return a Passivbot instance with throttle state initialized."""
    bot = object.__new__(Passivbot)
    bot._loss_gate_last_log_ms = {}
    bot._loss_gate_log_interval_ms = 5 * 60 * 1000
    return bot


class _FailingConsoleSink:
    def write(self, _event):
        raise OSError("console unavailable")


# ---------------------------------------------------------------------------
# Shared realized-memory consumers
# ---------------------------------------------------------------------------


class TestRealizedPnlConsumers:
    @staticmethod
    def _make_bot(
        *,
        max_realized_loss_pct=1.0,
        hsl_enabled=False,
        total_wallet_exposure_limit=1.0,
        values=None,
    ):
        bot = object.__new__(Passivbot)
        bot.coin_overrides = {}
        bot._live_max_realized_loss_pct = lambda: max_realized_loss_pct
        bot._equity_hard_stop_enabled = lambda: hsl_enabled
        bot.bot_value = lambda pside, key: (
            total_wallet_exposure_limit if key == "total_wallet_exposure_limit" else 0.0
        )
        values = values or {}
        bot.bp = lambda pside, key, symbol=None: values.get(
            (symbol, pside, key),
            values.get((None, pside, key), False if key == "unstuck_enabled" else 0.0),
        )
        return bot

    def test_disabled_risk_features_do_not_require_authoritative_pnl(self):
        bot = self._make_bot()

        assert bot._live_risk_uses_authoritative_pnl() is False

    @pytest.mark.parametrize(
        "max_realized_loss_pct,hsl_enabled",
        [(0.99, False), (1.0, True)],
    )
    def test_only_realized_loss_gate_requires_authoritative_pnl(
        self, max_realized_loss_pct, hsl_enabled
    ):
        bot = self._make_bot(
            max_realized_loss_pct=max_realized_loss_pct,
            hsl_enabled=hsl_enabled,
        )

        assert bot._live_risk_uses_authoritative_pnl() is (max_realized_loss_pct < 1.0)

    @pytest.mark.parametrize("symbol", [None, "SOL/USDT:USDT"])
    def test_enabled_unstuck_requires_authoritative_pnl(self, symbol):
        values = {
            (symbol, "long", "unstuck_enabled"): True,
            (symbol, "long", "unstuck_loss_allowance_pct"): 0.01,
            (symbol, "long", "unstuck_close_pct"): 0.1,
            (symbol, "long", "unstuck_threshold"): 0.9,
        }
        bot = self._make_bot(values=values)
        if symbol is not None:
            bot.coin_overrides = {symbol: {}}

        assert bot._live_risk_uses_authoritative_pnl() is True

    def test_unstuck_with_zero_total_exposure_does_not_require_pnl(self):
        values = {
            (None, "long", "unstuck_enabled"): True,
            (None, "long", "unstuck_loss_allowance_pct"): 0.01,
            (None, "long", "unstuck_close_pct"): 0.1,
            (None, "long", "unstuck_threshold"): 0.9,
        }
        bot = self._make_bot(
            total_wallet_exposure_limit=0.0,
            values=values,
        )

        assert bot._live_risk_uses_authoritative_pnl() is False


class TestGetRealizedMemory:
    def test_no_manager_with_disabled_consumers_returns_unused_zero(self):
        bot = object.__new__(Passivbot)
        bot._pnls_manager = None
        bot._orchestrator_uses_realized_pnl = lambda: False
        assert bot._get_realized_memory() == {"drawdown": 0.0}

    def test_enabled_realized_risk_requires_manager(self):
        bot = object.__new__(Passivbot)
        bot._pnls_manager = None
        bot._orchestrator_uses_realized_pnl = lambda: True
        with pytest.raises(RuntimeError, match="canonical fill manager"):
            bot._get_realized_memory()

    @pytest.mark.parametrize(
        "pnl_status,pnl_source",
        [("pending", "unknown"), ("complete", "synthetic_degraded")],
    )
    def test_disabled_consumers_do_not_read_unsafe_pnl(self, pnl_status, pnl_source):
        bot = _make_bot_with_events([
            _make_fill_event(-10.0, pnl_status=pnl_status, pnl_source=pnl_source)
        ])
        bot._orchestrator_uses_realized_pnl = lambda: False
        assert bot._get_realized_memory() == {"drawdown": 0.0}
        bot._pnls_manager.get_events.assert_not_called()

    @pytest.mark.parametrize("rows,expected", [
        ([], 0.0),
        ([(50.0, 0.0)], 0.0),
        ([(-20.0, 0.0)], 20.0),
        ([(0.0, -2.0)], 2.0),
        ([(-20.0, 0.0), (5.0, 2.0)], 13.0),
        ([(-20.0, 0.0), (30.0, 0.0), (-4.0, 0.0)], 4.0),
    ])
    def test_gate_only_uses_full_ordered_net_fill_debits(self, rows, expected):
        events = [_make_fill_event(pnl, fee_paid=fee, event_id=f"{i:02d}")
                  for i, (pnl, fee) in enumerate(rows)]
        bot = _make_bot_with_events(events)
        assert bot._unstuck_uses_realized_pnl() is False
        assert bot._get_realized_memory() == {"drawdown": expected}
        diagnostic = bot._realized_memory_diagnostics
        assert diagnostic["unknown_prefix"] is False
        assert diagnostic["initialization"] == "known_zero_origin"
        assert diagnostic["initial_drawdown"] == 0.0

    def test_profit_then_loss_at_same_time_is_not_netted(self):
        bot = _make_bot_with_events([
            _make_fill_event(-15.0, event_id="z-loss"),
            _make_fill_event(10.0, event_id="a-profit"),
        ])
        assert bot._get_realized_memory() == {"drawdown": 15.0}
        assert bot._realized_memory_diagnostics["ambiguous_cohort_count"] == 1
        assert bot._realized_memory_diagnostics["ordering_reasons"] == [
            "ambiguous_same_time_realized_order"
        ]

    def test_idle_decay_once_per_minute_and_repeated_queries_are_inert(self):
        bot = _make_bot_with_events([_make_fill_event(-20.0)])
        for now, expected in [(0, 20.0), (59_999, 20.0), (60_000, 10.0),
                              (60_000, 10.0), (120_000, 5.0)]:
            bot.get_exchange_time = lambda: now
            assert bot._get_realized_memory() == {"drawdown": expected}

    def test_slow_span_is_shared_by_gate_and_unstuck(self, monkeypatch):
        import passivbot as module
        native = module.pbr.realized_memory_v9
        calls = []

        def replay(payload):
            calls.append(json.loads(payload))
            return native(payload)

        monkeypatch.setattr(module.pbr, "realized_memory_v9", replay)
        bot = _make_bot_with_events([_make_fill_event(-15.0)])
        bot._unstuck_uses_realized_pnl = lambda: True
        bot.get_exchange_time = lambda: 60_000
        assert bot._get_realized_memory() == {"drawdown": 7.5}
        assert len(calls) == 1
        bot.config["live"]["lookback_ema_span_minutes"] = 7.0
        assert bot._get_realized_memory() == {"drawdown": 11.25}
        assert len(calls) == 2
        assert [row["span_minutes"] for row in calls] == [3.0, 7.0]

    def test_known_origin_restart_with_same_facts_and_clock_reproduces_memory(self):
        rows = [_make_fill_event(-20.0), _make_fill_event(5.0, timestamp=60_000)]
        first, restarted = _make_bot_with_events(rows), _make_bot_with_events(rows)
        first.get_exchange_time = lambda: 60_000
        assert first._get_realized_memory() == {"drawdown": 5.0}
        for value in (first, restarted):
            value.get_exchange_time = lambda: 120_000
        assert first._get_realized_memory() == restarted._get_realized_memory() == {
            "drawdown": 2.5
        }
        assert first._realized_memory_diagnostics == restarted._realized_memory_diagnostics

    def test_unknown_prefix_is_declared_and_not_a_rolling_expiry(self, caplog):
        rows = [_make_fill_event(-15.0, timestamp=540_001)]
        bot = _make_bot_with_events(rows)
        bot.config["live"] = {
            "lookback_ema_span_minutes": 60.0,
            "risk_bootstrap_max_lookback_minutes": 1.0,
        }
        bot.get_exchange_time = lambda: 600_000
        with caplog.at_level(logging.WARNING):
            first = bot._get_realized_memory()
            assert bot._get_realized_memory() == first
            assert caplog.text.count("approximate zero initializer") == 1
        assert bot._realized_memory_diagnostics["initialization"] == "unknown_prefix_zero"
        assert bot._realized_memory_diagnostics["unknown_prefix"] is True
        assert 0 < bot._realized_memory_diagnostics["prefix_attenuation"] < 1
        assert "not an economic error bound" in caplog.text

        bot.get_exchange_time = lambda: 720_000
        continuous = bot._get_realized_memory()
        assert continuous["drawdown"] > 0
        assert bot._realized_memory_diagnostics["start"] == 540_000
        restarted = _make_bot_with_events(rows)
        restarted.config = deepcopy(bot.config)
        restarted.get_exchange_time = lambda: 720_000
        assert restarted._get_realized_memory() == {"drawdown": 0.0}
        assert restarted._realized_memory_diagnostics["start"] == 660_000
        assert restarted._realized_memory_diagnostics["unknown_prefix"] is True
        # A new suffix is a new declared initializer, not exact infinite history.

    def test_derived_diagnostics_loss_preserves_origin_and_decision(self):
        bot = _make_bot_with_events([_make_fill_event(-15.0)])
        bot.get_exchange_time = lambda: 60_000
        before = bot._get_realized_memory()
        origin = bot._realized_memory_origin_ms
        del bot._realized_memory_diagnostics
        assert bot._get_realized_memory() == before
        assert bot._realized_memory_origin_ms == origin

    @pytest.mark.parametrize("change", ["backward_clock", "larger_bootstrap_cap"])
    def test_initialized_origin_does_not_admit_an_older_prefix(self, change, caplog):
        bot = _make_bot_with_events([
            _make_fill_event(-50.0, timestamp=539_999),
            _make_fill_event(-15.0, timestamp=540_001),
        ])
        bot.config["live"] = {
            "lookback_ema_span_minutes": 60.0,
            "risk_bootstrap_max_lookback_minutes": 1.0,
        }
        now = [600_000]
        bot.get_exchange_time = lambda: now[0]
        with caplog.at_level(logging.WARNING):
            first = bot._get_realized_memory()
            assert bot._realized_memory_origin_ms == 540_000
            if change == "backward_clock":
                now[0] = 599_999
                expected = {"drawdown": 15.0}  # No elapsed minute after this fill.
            else:
                bot.config["live"]["risk_bootstrap_max_lookback_minutes"] = 2.0
                expected = first
            assert bot._get_realized_memory() == expected
            assert bot._get_realized_memory() == expected
        bot._pnls_manager.get_events.assert_called_with(
            start_ms=540_000, end_ms=now[0]
        )
        assert bot._realized_memory_origin_ms == 540_000
        assert bot._realized_memory_diagnostics["start"] == 540_000
        assert bot._realized_memory_diagnostics["fills"] == 1
        assert bot._realized_memory_diagnostics["unknown_prefix"] is True
        assert caplog.text.count("approximate zero initializer") == 1

    def test_unknown_prefix_restart_with_retained_initializer_reproduces_memory(self):
        rows = [_make_fill_event(-15.0, timestamp=540_001)]
        first, restarted = _make_bot_with_events(rows), _make_bot_with_events(rows)
        for bot in (first, restarted):
            bot.config["live"] = {
                "lookback_ema_span_minutes": 60.0,
                "risk_bootstrap_max_lookback_minutes": 1.0,
            }
        first.get_exchange_time = lambda: 600_000
        first._get_realized_memory()
        # Supplying the same factual initializer is different from inventing
        # the unknown prefix or persisting a prior allowance/permission.
        restarted._realized_memory_origin_ms = first._realized_memory_origin_ms
        for bot in (first, restarted):
            bot.get_exchange_time = lambda: 720_000
        assert first._get_realized_memory() == restarted._get_realized_memory()
        assert first._realized_memory_diagnostics == restarted._realized_memory_diagnostics

    @pytest.mark.parametrize("status,source", [
        ("pending", "unknown"), ("complete", "synthetic_degraded")
    ])
    def test_structural_cooldown_history_does_not_rebase_realized_memory(self, status, source):
        bot = _make_bot_with_events([
            _make_fill_event(-50.0, timestamp=1_000, pnl_status=status, pnl_source=source),
            _make_fill_event(-15.0, timestamp=600_000),  # No elapsed minute after this fill.
        ])
        bot.config["live"] = {
            "lookback_ema_span_minutes": 60.0,
            "risk_bootstrap_max_lookback_minutes": 1.0,
        }
        bot.get_exchange_time = lambda: 600_000
        bot._max_configured_entry_cooldown_minutes = lambda: 720.0
        assert bot._required_fill_history_start_ms(600_000, pnl_start_ms=540_000) == (True, 0)
        assert bot._get_realized_memory() == {"drawdown": 15.0}
        assert bot._realized_memory_origin_ms == 540_000
        assert bot._realized_memory_diagnostics["start"] == 540_000
        assert bot._realized_memory_diagnostics["fills"] == 1
        assert bot._required_fill_history_start_ms(600_000, pnl_start_ms=0) == (True, 0)
        assert bot._get_realized_memory() == {"drawdown": 15.0}

    def test_current_fact_query_excludes_future_pending_rows(self):
        bot = _make_bot_with_events([
            _make_fill_event(-15.0),
            _make_fill_event(0.0, timestamp=60_001, pnl_status="pending"),
        ])
        bot.get_exchange_time = lambda: 60_000
        assert bot._get_realized_memory() == {"drawdown": 7.5}
        bot._pnls_manager.get_events.assert_called_once_with(start_ms=0, end_ms=60_000)

    def test_corrections_and_exact_duplicates_replay_without_double_debit(self):
        loss = _make_fill_event(-20.0, event_id="loss")
        rows = [loss, deepcopy(loss), _make_fill_event(5.0, event_id="profit")]
        bot = _make_bot_with_events(rows)
        assert bot._get_realized_memory() == {"drawdown": 15.0}
        rows[0].pnl = rows[1].pnl = -12.0
        assert bot._get_realized_memory() == {"drawdown": 7.0}
        assert bot._get_realized_memory() == {"drawdown": 7.0}

    def test_conflicting_identity_cannot_choose_a_revision(self):
        from live.realized_facts import RealizedFactsError
        bot = _make_bot_with_events([
            _make_fill_event(-20.0, event_id="same"),
            _make_fill_event(-12.0, event_id="same"),
        ])
        with pytest.raises(RealizedFactsError, match="conflicting realized fill identity"):
            bot._get_realized_memory()

    @pytest.mark.parametrize("field,value", [
        ("unknown_prefix", True), ("initialization", "exact"),
        ("initial_drawdown", 1.0), ("fills", True), ("fills", 0),
        ("prefix_attenuation", float("nan")), ("prefix_attenuation", 1.01),
        ("drawdown", -1.0), ("drawdown", True), ("start", True), ("now", -1),
    ])
    def test_malformed_producer_initialization_is_fatal(self, monkeypatch, field, value):
        import passivbot as module
        from passivbot import FatalBotException
        native = module.pbr.realized_memory_v9

        def corrupt(payload):
            result = json.loads(native(payload))
            result[field] = value
            return json.dumps(result)

        monkeypatch.setattr(module.pbr, "realized_memory_v9", corrupt)
        bot = _make_bot_with_events([_make_fill_event(-20.0)])
        with pytest.raises(FatalBotException, match="invalid v9 realized-memory producer"):
            bot._get_realized_memory()

    def test_pending_close_pnl_fails_loudly(self):
        bot = _make_bot_with_events([_make_fill_event(0.0, pnl_status="pending")])

        with pytest.raises(RuntimeError, match="realized PnL pending"):
            bot._get_realized_memory()

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
            bot._get_realized_memory()

    def test_missing_cache_blocks_risk_history_coverage(self):
        now_ms = 10 * 86_400_000
        bot = _make_bot_with_events(
            [_make_fill_event(10.0, timestamp=now_ms - 60_000)],
            cache=None,
        )
        _set_memory_horizon(bot, minutes=1440.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="missing FillEventsManager cache"):
            bot._get_realized_memory()

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
        _set_memory_horizon(bot, minutes=1440.0, now_ms=now_ms)

        with pytest.raises(
            RuntimeError, match="fill history gap overlaps risk lookback"
        ):
            bot._get_realized_memory()

    def test_uncovered_empty_history_fails_loudly(self):
        now_ms = 10 * 86_400_000
        bot = _make_bot_with_events([], cache=_RiskCache(history_scope="window"))
        _set_memory_horizon(bot, minutes=1440.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="fill history coverage unknown"):
            bot._get_realized_memory()

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
        _set_memory_horizon(bot, minutes=1440.0, now_ms=now_ms)

        with pytest.raises(RuntimeError, match="fill history coverage unknown"):
            bot._get_realized_memory()

    def test_covered_empty_history_returns_declared_zero_memory(self):
        now_ms = 10 * 86_400_000
        start_ms = now_ms - 86_400_000
        bot = _make_bot_with_events(
            [],
            cache=_RiskCache(
                covered_start_ms=start_ms,
                history_scope="window",
            ),
        )
        _set_memory_horizon(bot, minutes=1440.0, now_ms=now_ms)

        assert bot._get_realized_memory() == {"drawdown": 0.0}


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

    def test_no_console_sink_uses_legacy_warning_and_emits_event(self, caplog):
        bot = _make_bot_for_logging()
        sink = ListEventSink()
        bot.bot_id = "bot_loss_gate"
        bot.exchange = "binance"
        bot.user = "binance_01"
        bot.live_event_console_enabled = True
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

    def test_missing_pipeline_uses_legacy_warning(self, caplog):
        bot = _make_bot_for_logging()
        bot.live_event_console_enabled = True
        emitted = []
        bot._emit_realized_loss_gate_blocked_event = lambda **kwargs: emitted.append(
            kwargs
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

        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(
                {"diagnostics": {"loss_gate_blocks": [block]}},
                {0: "BTCUSDT"},
            )

        assert "[risk] order blocked by realized-loss gate" in caplog.text
        assert len(emitted) == 1

    @pytest.mark.parametrize("console_sink_fails", [False, True])
    def test_structured_console_owns_warning(self, caplog, console_sink_fails):
        bot = _make_bot_for_logging()
        structured_sink = ListEventSink()
        console_sink = _FailingConsoleSink() if console_sink_fails else ListEventSink()
        bot.bot_id = "bot_loss_gate"
        bot.exchange = "binance"
        bot.user = "binance_01"
        bot.live_event_console_enabled = True
        bot._live_event_current_cycle_id = "cy_loss_gate"
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[structured_sink],
            monitor_sinks=[],
            console_sink=console_sink,
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
                bot._log_realized_loss_gate_blocks(
                    {"diagnostics": {"loss_gate_blocks": [block]}},
                    {0: "BTCUSDT"},
                )
            assert bot._live_event_pipeline.flush(timeout=2.0) is True
        finally:
            assert bot._live_event_pipeline.close(timeout=2.0) is True

        assert "[risk] order blocked by realized-loss gate" not in caplog.text
        events = [
            event
            for event in structured_sink.events
            if event.event_type == EventTypes.REALIZED_LOSS_GATE_BLOCKED
        ]
        assert len(events) == 1
        if console_sink_fails:
            assert bot._live_event_pipeline.sink_error_counters["console"] >= 1
        else:
            assert console_sink.events == events

    def test_missing_emitter_uses_legacy_warning(self, caplog):
        bot = _make_bot_for_logging()
        bot.live_event_console_enabled = True
        console_sink = ListEventSink()
        bot._live_event_pipeline = LiveEventPipeline(
            structured_sinks=[ListEventSink()],
            monitor_sinks=[],
            console_sink=console_sink,
        )
        bot._emit_realized_loss_gate_blocked_event = None
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
        finally:
            assert bot._live_event_pipeline.close(timeout=2.0) is True

        assert "[risk] order blocked by realized-loss gate" in caplog.text
        assert console_sink.events == []

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

        assert "add-entry cooldown anchored" in caplog.text
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

    def test_position_delta_anchor_event_emits_when_text_warning_throttled(
        self, caplog
    ):
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

        assert "add-entry cooldown anchored" not in caplog.text
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
            "hsl_panic_close_order_type": "market",
        }
        hsl_short = deepcopy(hsl_long)
        config = {
            "config_version": "v9.1.0",
            "backtest": {
                "coins": {"binance": ["BTC"]},
                "starting_balance": 10000,
                "btc_collateral_cap": 0.5,
                "btc_collateral_ltv_cap": None,
                "filter_by_min_effective_cost": False,
                "hsl_detailed_report": False,
                "dynamic_wel_by_tradability": True,
                "limit_order_fill_buffer_pct": 0.0,
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
                "hsl_signal_mode": "coin",
                "max_realized_loss_pct": 1.0,
                "lookback_ema_span_minutes": 43200.0,
                "risk_bootstrap_max_lookback_minutes": 129600.0,
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
        assert bp["lookback_ema_span_minutes"] == pytest.approx(43200.0)
        assert "realized_loss_gate_lookback_minutes" not in bp

    def test_explicit_value_passthrough(self):
        config = self._make_config(max_realized_loss_pct=0.05)
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.05)
        assert "realized_loss_gate_lookback_minutes" not in bp
        assert bp["lookback_ema_span_minutes"] == pytest.approx(43200.0)

    def test_zero_disables_lossy_closes(self):
        config = self._make_config(max_realized_loss_pct=0.0)
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.0)


class TestPrepBacktestArgsEquityHardStopLoss:
    def _make_config(self, policy=None):
        from config.hsl import generated_template
        from config.schema import get_template_config

        config = generated_template(get_template_config())
        config["backtest"].update(coins={"binance": ["BTC"]}, starting_balance=10000)
        config["live"]["approved_coins"] = {"long": ["BTC"], "short": []}
        config["bot"]["long"]["hsl"].update(enabled=False)
        config["bot"]["short"]["hsl"].update(enabled=False)
        if policy:
            config["bot"]["long"]["hsl"].update(policy)
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
        assert hs["mode"] == "coin"
        assert hs["portfolio"] is None
        assert hs["sides"] == [config["bot"][s]["hsl"] for s in ("long", "short")]
        assert not any(s["enabled"] for s in hs["sides"])

    def test_custom_passthrough(self):
        policy = dict(
            enabled=True,
            red_threshold=0.3,
            ema_span_minutes=45.5,
            cooldown_minutes_after_red=30.0,
            panic_close_order_type="limit",
        )
        config = self._make_config(policy)
        _, _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["equity_hard_stop_loss"]["sides"][0] == {
            **policy, "scale_budget_with_excess_allowance": False
        }

    @pytest.mark.parametrize(
        "policy,field",
        [
            ({"cooldown_minutes_after_red": -1.0}, "cooldown_minutes_after_red"),
            ({"red_threshold": 0.0}, "red_threshold"),
            ({"ema_span_minutes": 0.5}, "ema_span_minutes"),
            ({"restart_after_red_policy": "threshold"}, "restart_after_red_policy"),
        ],
    )
    def test_invalid_policy_raises(self, policy, field):
        config = self._make_config(dict(enabled=True, **policy))
        with pytest.raises(ValueError, match=field):
            prep_backtest_args(config, self._make_mss(), "binance")
