"""Tests for CCXTBot._get_position_side_for_order() — side + reduceOnly derivation."""

import pytest


class TestGetPositionSideForOrder:
    """Test position_side derivation from CCXT unified side + reduceOnly."""

    def _make_bot(self):
        from exchanges.ccxt_bot import CCXTBot

        return CCXTBot.__new__(CCXTBot)

    def test_buy_entry_is_long(self):
        bot = self._make_bot()
        order = {"side": "buy", "reduceOnly": False}
        assert bot._get_position_side_for_order(order) == "long"

    def test_buy_reduce_only_is_short(self):
        bot = self._make_bot()
        order = {"side": "buy", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "short"

    def test_sell_entry_is_short(self):
        bot = self._make_bot()
        order = {"side": "sell", "reduceOnly": False}
        assert bot._get_position_side_for_order(order) == "short"

    def test_sell_reduce_only_is_long(self):
        bot = self._make_bot()
        order = {"side": "sell", "reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "long"

    def test_missing_reduce_only_defaults_to_false(self):
        bot = self._make_bot()
        order = {"side": "buy"}
        assert bot._get_position_side_for_order(order) == "long"

        order = {"side": "sell"}
        assert bot._get_position_side_for_order(order) == "short"

    def test_missing_side_returns_both(self):
        bot = self._make_bot()
        order = {}
        assert bot._get_position_side_for_order(order) == "both"

        order = {"reduceOnly": True}
        assert bot._get_position_side_for_order(order) == "both"
