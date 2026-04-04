"""Tests for the realized-loss gate feature (live.max_realized_loss_pct)."""

import logging
import types
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest

from passivbot import Passivbot
from backtest import prep_backtest_args

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fill_event(pnl: float, timestamp: float = 0.0) -> types.SimpleNamespace:
    """Create a minimal fill-event namespace with a .pnl attribute."""
    return types.SimpleNamespace(pnl=pnl, timestamp=timestamp)


def _make_bot_with_events(events, balance=10000.0):
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
    bot.balance = balance
    return bot


def _set_pnl_lookback(bot, *, lookback_days: float, now_ms: int) -> None:
    bot.config = {"live": {"pnls_max_lookback_days": float(lookback_days)}}
    bot.get_exchange_time = lambda: now_ms


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
        # cumsum: [-10, -30] → max=-10, last=-30
        bot = _make_bot_with_events(events)
        result = bot._get_realized_pnl_cumsum_stats()
        assert result["max"] == pytest.approx(-10.0)
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

    def test_zero_lookback_uses_full_history_like_backtest(self):
        now_ms = 10 * 86_400_000
        events = [
            _make_fill_event(100.0, timestamp=now_ms - 3 * 86_400_000),
            _make_fill_event(-80.0, timestamp=now_ms - 3 * 86_400_000 + 1),
            _make_fill_event(10.0, timestamp=now_ms - 60_000),
        ]
        bot = _make_bot_with_events(events)
        _set_pnl_lookback(bot, lookback_days=0.0, now_ms=now_ms)

        result = bot._get_realized_pnl_cumsum_stats()

        assert result["max"] == pytest.approx(100.0)
        assert result["last"] == pytest.approx(30.0)


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
        assert "BTCUSDT" in caplog.text
        assert "close_auto_reduce_wel_long" in caplog.text

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
        assert "BTCUSDT" in caplog.text
        assert "SUIUSDT" in caplog.text


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
