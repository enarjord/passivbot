"""Tests for CCXTBot._get_position_side_for_order() hook."""

import pytest


class TestGetPositionSideForOrder:
    """Test the position_side derivation hook."""

    def test_uses_position_side_from_info_when_present(self):
        """When exchange provides positionSide in info, use it."""
        from exchanges.ccxt_bot import CCXTBot

        # Create minimal instance (no real exchange connection needed)
        bot = CCXTBot.__new__(CCXTBot)

        order = {"info": {"positionSide": "LONG"}}
        assert bot._get_position_side_for_order(order) == "long"

        order = {"info": {"positionSide": "SHORT"}}
        assert bot._get_position_side_for_order(order) == "short"

    def test_derives_from_client_order_id_when_no_position_side(self):
        """When no positionSide, derive from clientOrderId order type suffix."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)

        # clientOrderId encodes order type - 0x0000 = entry_initial_normal_long (ID 0)
        # The hex pattern "0x0000" decodes to type ID 0 which is "entry_initial_normal_long"
        order = {"info": {}, "clientOrderId": "pb-0x0000-abc123"}
        result = bot._get_position_side_for_order(order)
        assert result == "long"

        # 0x0012 = close_grid_short (ID 18)
        order = {"info": {}, "clientOrderId": "pb-0x0012-xyz789"}
        result = bot._get_position_side_for_order(order)
        assert result == "short"

    def test_returns_both_when_no_position_side_and_no_client_order_id(self):
        """When neither positionSide nor clientOrderId available, return 'both'."""
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot.__new__(CCXTBot)

        order = {"info": {}}
        assert bot._get_position_side_for_order(order) == "both"

        order = {"info": {}, "clientOrderId": ""}
        assert bot._get_position_side_for_order(order) == "both"

        # Also test with invalid clientOrderId that doesn't decode to a valid type
        order = {"info": {}, "clientOrderId": "invalid-no-hex"}
        assert bot._get_position_side_for_order(order) == "both"
