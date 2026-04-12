"""Tests for _log_inactive_symbol_reasons diagnostic logging."""

import logging

import pytest


@pytest.fixture
def make_bot():
    """Create a minimal mock with the attributes _log_inactive_symbol_reasons needs."""

    class FakeBot:
        def __init__(self, balance, twel, n_pos, qty_pct, emc, filter_enabled=True):
            self.effective_min_cost = {}
            self.config = {
                "bot": {
                    "long": {
                        "total_wallet_exposure_limit": twel,
                        "n_positions": n_pos,
                        "entry_initial_qty_pct": qty_pct,
                    },
                    "short": {
                        "total_wallet_exposure_limit": 0,
                        "n_positions": 0,
                        "entry_initial_qty_pct": 0,
                    },
                },
                "live": {},
            }
            self.coin_overrides = {}
            self._filter_enabled = filter_enabled
            self._emc = emc

        def is_pside_enabled(self, pside):
            if pside == "long":
                twel = self.config["bot"]["long"]["total_wallet_exposure_limit"]
                n = self.config["bot"]["long"]["n_positions"]
                return twel > 0 and n > 0
            return False

        def bot_value(self, pside, key):
            return self.config["bot"].get(pside, {}).get(key, 0.0)

        def bp(self, pside, key, symbol=None):
            return self.config["bot"].get(pside, {}).get(key, 0.0)

        def _calc_effective_min_cost_at_price(self, symbol, price):
            return self._emc

    def _factory(balance=100.0, twel=2.0, n_pos=1, qty_pct=0.05, emc=20.0):
        bot = FakeBot(balance, twel, n_pos, qty_pct, emc)
        # Bind the real method
        from passivbot import Passivbot

        import types

        bot._log_inactive_symbol_reasons = types.MethodType(
            Passivbot._log_inactive_symbol_reasons, bot
        )
        return bot

    return _factory


class TestLogInactiveSymbolReasons:
    def test_min_cost_too_high_logs_info(self, make_bot, caplog):
        """When budget < effective_min_cost, an INFO log explains the numbers."""
        bot = make_bot(balance=100.0, twel=2.0, n_pos=1, qty_pct=0.05, emc=20.0)
        # budget = 100 * 2.0 * 0.05 = 10 < 20 → should log

        diagnostics = {
            "symbol_states": [
                {
                    "symbol_idx": 0,
                    "long": {"active": False, "allow_initial": False},
                    "short": {"active": False, "allow_initial": False},
                }
            ]
        }
        idx_to_symbol = {0: "HYPE/USDC:USDC"}
        input_dict = {
            "balance": 100.0,
            "global": {"filter_by_min_effective_cost": True},
            "symbols": [
                {
                    "symbol_idx": 0,
                    "order_book": {"bid": 41.0, "ask": 41.0},
                }
            ],
        }

        with caplog.at_level(logging.INFO):
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)

        assert any("[orchestrator] HYPE long inactive: min_cost:" in r.message for r in caplog.records)
        # Verify it includes the budget and emc numbers
        msg = [r.message for r in caplog.records if "min_cost:" in r.message][0]
        assert "budget=10.00" in msg
        assert "effective_min_cost=20.00" in msg

    def test_active_symbol_not_logged(self, make_bot, caplog):
        """Active symbols should not produce any log."""
        bot = make_bot()
        diagnostics = {
            "symbol_states": [
                {
                    "symbol_idx": 0,
                    "long": {"active": True, "allow_initial": True},
                    "short": {"active": False, "allow_initial": False},
                }
            ]
        }
        idx_to_symbol = {0: "HYPE/USDC:USDC"}
        input_dict = {
            "balance": 100.0,
            "global": {"filter_by_min_effective_cost": True},
            "symbols": [],
        }

        with caplog.at_level(logging.INFO):
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)

        assert not any("inactive" in r.message for r in caplog.records)

    def test_budget_sufficient_shows_generic_reason(self, make_bot, caplog):
        """When budget >= emc but still inactive, show generic orchestrator message."""
        bot = make_bot(balance=1000.0, twel=2.0, n_pos=1, qty_pct=0.05, emc=5.0)
        # budget = 1000 * 2.0 * 0.05 = 100 > 5 → min cost is not the reason

        diagnostics = {
            "symbol_states": [
                {
                    "symbol_idx": 0,
                    "long": {"active": False, "allow_initial": False},
                    "short": {"active": False, "allow_initial": False},
                }
            ]
        }
        idx_to_symbol = {0: "HYPE/USDC:USDC"}
        input_dict = {
            "balance": 1000.0,
            "global": {"filter_by_min_effective_cost": True},
            "symbols": [{"symbol_idx": 0, "order_book": {"bid": 41.0, "ask": 41.0}}],
        }

        with caplog.at_level(logging.INFO):
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)

        msg = [r.message for r in caplog.records if "inactive" in r.message][0]
        assert "not selected by orchestrator" in msg

    def test_throttle_prevents_spam(self, make_bot, caplog):
        """Second call within 5 minutes should not log again."""
        bot = make_bot(balance=100.0, twel=2.0, n_pos=1, qty_pct=0.05, emc=20.0)

        diagnostics = {
            "symbol_states": [
                {
                    "symbol_idx": 0,
                    "long": {"active": False, "allow_initial": False},
                    "short": {"active": False, "allow_initial": False},
                }
            ]
        }
        idx_to_symbol = {0: "HYPE/USDC:USDC"}
        input_dict = {
            "balance": 100.0,
            "global": {"filter_by_min_effective_cost": True},
            "symbols": [{"symbol_idx": 0, "order_book": {"bid": 41.0, "ask": 41.0}}],
        }

        with caplog.at_level(logging.INFO):
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)
            count_first = sum(1 for r in caplog.records if "inactive" in r.message)
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)
            count_second = sum(1 for r in caplog.records if "inactive" in r.message)

        assert count_first == 1
        assert count_second == count_first  # throttled, no new log

    def test_disabled_pside_not_logged(self, make_bot, caplog):
        """Short side disabled (twel=0) should not produce inactive log."""
        bot = make_bot()
        diagnostics = {
            "symbol_states": [
                {
                    "symbol_idx": 0,
                    "long": {"active": True, "allow_initial": True},
                    "short": {"active": False, "allow_initial": False},
                }
            ]
        }
        idx_to_symbol = {0: "HYPE/USDC:USDC"}
        input_dict = {
            "balance": 100.0,
            "global": {"filter_by_min_effective_cost": True},
            "symbols": [],
        }

        with caplog.at_level(logging.INFO):
            bot._log_inactive_symbol_reasons(diagnostics, idx_to_symbol, input_dict)

        # Short is disabled (twel=0, n_positions=0), should not log
        assert not any("short" in r.message and "inactive" in r.message for r in caplog.records)
