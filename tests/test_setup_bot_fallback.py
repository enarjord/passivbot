"""Tests for setup_bot() CCXTBot fallback behavior."""

import pytest
from unittest.mock import patch, MagicMock


def test_order_churn_rollout_allowlist_matches_supported_connector_contract():
    from types import SimpleNamespace

    from live.order_churn_gate import (
        ORDER_CHURN_GATE_SUPPORTED_EXCHANGES,
        connector_supports_order_churn_gate,
    )

    assert ORDER_CHURN_GATE_SUPPORTED_EXCHANGES == {
        "binance",
        "bitget",
        "bybit",
        "fake",
        "gateio",
        "hyperliquid",
        "kucoin",
        "okx",
        "weex",
    }
    assert connector_supports_order_churn_gate(SimpleNamespace(exchange="weex")) is True
    assert connector_supports_order_churn_gate(SimpleNamespace(exchange="kraken")) is False


def test_setup_bot_unknown_exchange_uses_ccxt_bot():
    """setup_bot() returns CCXTBot for unknown exchanges."""
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}

    # Mock load_user_info to return an unknown exchange
    mock_user_info = {
        "exchange": "kraken",  # Not in the if/elif chain
        "key": "test_key",
        "secret": "test_secret",
    }

    with patch("passivbot.load_user_info", return_value=mock_user_info):
        # Mock CCXTBot to avoid actual initialization
        with patch("exchanges.ccxt_bot.CCXTBot") as mock_ccxt_bot:
            mock_bot = MagicMock()
            mock_ccxt_bot.return_value = mock_bot

            result = setup_bot(config)

            mock_ccxt_bot.assert_called_once_with(config)
            assert result == mock_bot
            assert result._order_churn_gate_enabled_for_connector is False


def test_setup_bot_known_exchange_uses_specific_bot():
    """setup_bot() still uses specific bots for known exchanges."""
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}

    mock_user_info = {
        "exchange": "binance",
        "key": "test_key",
        "secret": "test_secret",
    }

    with patch("passivbot.load_user_info", return_value=mock_user_info):
        with patch("exchanges.binance.BinanceBot") as mock_binance_bot:
            mock_bot = MagicMock()
            mock_binance_bot.return_value = mock_bot

            result = setup_bot(config)

            mock_binance_bot.assert_called_once_with(config)
            assert result == mock_bot
            assert result._order_churn_gate_enabled_for_connector is True


@pytest.mark.parametrize(
    ("exchange", "target"),
    [
        ("defx", "exchanges.defx.DefxBot"),
        ("paradex", "exchanges.paradex.ParadexBot"),
    ],
)
def test_setup_bot_unsupported_legacy_connectors_do_not_enable_churn_gate(
    exchange, target
):
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}
    mock_user_info = {
        "exchange": exchange,
        "key": "test_key",
        "secret": "test_secret",
    }
    with patch("passivbot.load_user_info", return_value=mock_user_info):
        with patch(target) as mock_bot_cls:
            mock_bot = MagicMock()
            mock_bot_cls.return_value = mock_bot

            result = setup_bot(config)

    assert result is mock_bot
    assert result._order_churn_gate_enabled_for_connector is False
