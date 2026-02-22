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
    bot._pnls_manager.get_events.return_value = events
    bot.balance = balance
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


# ---------------------------------------------------------------------------
# _log_realized_loss_gate_blocks
# ---------------------------------------------------------------------------


class TestLogRealizedLossGateBlocks:
    def test_no_diagnostics_is_silent(self, caplog):
        bot = object.__new__(Passivbot)
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks({}, {})
        assert caplog.text == ""

    def test_empty_blocks_is_silent(self, caplog):
        bot = object.__new__(Passivbot)
        out = {"diagnostics": {"loss_gate_blocks": []}}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, {})
        assert caplog.text == ""

    def test_block_emits_risk_warning(self, caplog):
        bot = object.__new__(Passivbot)
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
        bot = object.__new__(Passivbot)
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
        bot = object.__new__(Passivbot)
        out = {"diagnostics": {"loss_gate_blocks": ["not_a_dict"]}}
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks(out, {})
        assert "[risk]" not in caplog.text

    def test_non_dict_output_is_silent(self, caplog):
        bot = object.__new__(Passivbot)
        with caplog.at_level(logging.WARNING):
            bot._log_realized_loss_gate_blocks("not_a_dict", {})
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# prep_backtest_args passthrough
# ---------------------------------------------------------------------------


class TestPrepBacktestArgsMaxRealizedLossPct:
    def _make_config(self, max_realized_loss_pct=None):
        config = {
            "backtest": {
                "coins": {"binance": ["BTC"]},
                "starting_balance": 10000,
                "btc_collateral_cap": 0.5,
                "btc_collateral_ltv_cap": None,
                "filter_by_min_effective_cost": False,
            },
            "bot": {
                "long": {
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 1.0,
                    "wallet_exposure_limit": 0.5,
                },
                "short": {
                    "n_positions": 1,
                    "total_wallet_exposure_limit": 0.0,
                    "wallet_exposure_limit": 0.0,
                },
            },
            "live": {
                "hedge_mode": True,
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
        _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(1.0)

    def test_explicit_value_passthrough(self):
        config = self._make_config(max_realized_loss_pct=0.05)
        _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.05)

    def test_zero_disables_lossy_closes(self):
        config = self._make_config(max_realized_loss_pct=0.0)
        _, _, bp = prep_backtest_args(config, self._make_mss(), "binance")
        assert bp["max_realized_loss_pct"] == pytest.approx(0.0)
