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
        "bitunix",
        "bybit",
        "fake",
        "gateio",
        "hyperliquid",
        "kucoin",
        "okx",
        "weex",
        "lighter",
    }
    assert connector_supports_order_churn_gate(SimpleNamespace(exchange="weex")) is True
    assert connector_supports_order_churn_gate(SimpleNamespace(exchange="bitunix")) is True
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


@pytest.mark.parametrize("account_type", ["portfolio_margin", "pm", "PM"])
def test_setup_bot_binance_portfolio_margin_account_type(account_type):
    """setup_bot() routes to BinancePortfolioBot when account_type is portfolio_margin."""
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}

    mock_user_info = {
        "exchange": "binance",
        "account_type": account_type,
        "key": "test_key",
        "secret": "test_secret",
    }

    with patch("passivbot.load_user_info", return_value=mock_user_info):
        with patch("exchanges.binance_portfolio.BinancePortfolioBot") as mock_pm_bot:
            mock_bot = MagicMock()
            mock_pm_bot.return_value = mock_bot

            result = setup_bot(config)

            mock_pm_bot.assert_called_once_with(config)
            assert result == mock_bot
            # account_type is a credentials-level detail, not a different exchange --
            # the order-churn-gate allowlist is still keyed on user_info["exchange"].
            assert result._order_churn_gate_enabled_for_connector is True


@pytest.mark.parametrize("exchange", ["defx", "paradex"])
def test_setup_bot_rejects_retired_connectors_before_generic_fallback(exchange):
    from passivbot import setup_bot

    config = {"live": {"user": "test_user"}}
    with patch("passivbot.load_user_info", return_value={"exchange": exchange}):
        with patch("exchanges.ccxt_bot.CCXTBot") as generic:
            with pytest.raises(ValueError, match="retired in v9"):
                setup_bot(config)
    generic.assert_not_called()


@pytest.mark.parametrize("exchange", ["defx", "paradex"])
def test_offline_connector_inspection_rejects_retired_connectors(exchange):
    from ccxt_contracts import get_bot_class

    with pytest.raises(ValueError, match="retired in v9"):
        get_bot_class(exchange)
