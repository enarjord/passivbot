"""Tests for setup_bot() CCXTBot fallback behavior."""

import pytest
from unittest.mock import patch, MagicMock


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
