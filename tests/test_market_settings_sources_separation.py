"""Test that market_settings_sources doesn't affect OHLCV data selection."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_ohlcv_source_extraction():
    """Verify that ohlcv_source is correctly extracted from mss dict."""
    from collections import defaultdict
    
    # Simulate what _prepare_hlcvs_combined_impl does
    chosen_mss_per_coin = {
        "BTC": {
            "exchange": "bybit",  # market settings from bybit
            "ohlcv_source": "binance",  # OHLCV from binance
            "min_cost": 10.0,
        },
        "ETH": {
            "exchange": "binance",  # market settings from binance (same as OHLCV)
            "min_cost": 5.0,
        },
        "DOGE": {
            "exchange": "hyperliquid",  # market settings from hyperliquid
            "ohlcv_source": "binance",  # OHLCV from binance
            "min_cost": 1.0,
        },
    }
    
    valid_coins = ["BTC", "ETH", "DOGE"]
    
    # Test 1: exchanges_with_data should use OHLCV sources
    exchanges_with_data = sorted(set([
        chosen_mss_per_coin[coin].get("ohlcv_source", chosen_mss_per_coin[coin]["exchange"]) 
        for coin in valid_coins
    ]))
    
    # Should only have "binance" since all OHLCV comes from there
    assert exchanges_with_data == ["binance"], (
        f"Expected only 'binance' for OHLCV, got {exchanges_with_data}"
    )
    
    # Test 2: exchanges_counts should count OHLCV sources, not market settings
    exchanges_counts = defaultdict(int)
    for coin in chosen_mss_per_coin:
        ohlcv_exchange = chosen_mss_per_coin[coin].get(
            "ohlcv_source", 
            chosen_mss_per_coin[coin]["exchange"]
        )
        exchanges_counts[ohlcv_exchange] += 1
    
    # All 3 coins should count towards "binance" for OHLCV
    assert exchanges_counts["binance"] == 3, (
        f"Expected 3 coins from binance, got {exchanges_counts['binance']}"
    )
    assert "bybit" not in exchanges_counts, "bybit should not be in OHLCV counts"
    assert "hyperliquid" not in exchanges_counts, "hyperliquid should not be in OHLCV counts"
    
    # Test 3: reference_exchange should be determined by OHLCV sources
    reference_exchange = sorted(exchanges_counts.items(), key=lambda x: x[1])[-1][0]
    assert reference_exchange == "binance", (
        f"Expected reference_exchange='binance', got '{reference_exchange}'"
    )
    
    # Test 4: verify market settings are preserved separately
    assert chosen_mss_per_coin["BTC"]["exchange"] == "bybit"
    assert chosen_mss_per_coin["DOGE"]["exchange"] == "hyperliquid"
    assert chosen_mss_per_coin["ETH"]["exchange"] == "binance"


def test_market_settings_vs_ohlcv_separation():
    """Verify that market settings exchange doesn't leak into OHLCV logic."""
    
    # Scenario: User wants bybit market settings but binance OHLCV data
    mss_entry = {
        "exchange": "bybit",  # market settings source
        "ohlcv_source": "binance",  # OHLCV data source
        "min_cost": 0.01,
        "min_qty": 0.1,
        "symbol": "BTC/USDT:USDT",
    }
    
    # Extract OHLCV source (what should be used for volume normalization)
    ohlcv_exchange = mss_entry.get("ohlcv_source", mss_entry["exchange"])
    
    assert ohlcv_exchange == "binance", (
        f"OHLCV should come from binance, got {ohlcv_exchange}"
    )
    
    # Extract market settings source (what should be used for min_cost, etc.)
    market_settings_exchange = mss_entry["exchange"]
    
    assert market_settings_exchange == "bybit", (
        f"Market settings should come from bybit, got {market_settings_exchange}"
    )
    
    # Verify they are different
    assert ohlcv_exchange != market_settings_exchange, (
        "OHLCV and market settings should be able to differ"
    )


def test_coins_by_exchange_grouping():
    """Verify logging groups coins by OHLCV source, not market settings."""
    from collections import defaultdict
    
    chosen_mss_per_coin = {
        "BTC": {"exchange": "bybit", "ohlcv_source": "binance"},
        "ETH": {"exchange": "bybit", "ohlcv_source": "binance"},
        "DOGE": {"exchange": "hyperliquid", "ohlcv_source": "binance"},
        "SOL": {"exchange": "binance"},  # No ohlcv_source, use exchange
    }
    
    valid_coins = ["BTC", "ETH", "DOGE", "SOL"]
    
    # Simulate the grouping logic for logging
    coins_by_exchange = defaultdict(list)
    for coin in valid_coins:
        ohlcv_ex = chosen_mss_per_coin[coin].get(
            "ohlcv_source", 
            chosen_mss_per_coin[coin]["exchange"]
        )
        coins_by_exchange[ohlcv_ex].append(coin)
    
    # All coins should be grouped under "binance" for OHLCV
    assert "binance" in coins_by_exchange
    assert sorted(coins_by_exchange["binance"]) == ["BTC", "DOGE", "ETH", "SOL"]
    
    # No coins should be grouped under bybit/hyperliquid for OHLCV
    assert "bybit" not in coins_by_exchange
    assert "hyperliquid" not in coins_by_exchange
