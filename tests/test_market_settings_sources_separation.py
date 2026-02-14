"""Test that market_settings_sources doesn't affect OHLCV data selection."""
import sys
import os

import pandas as pd
import pytest

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


@pytest.mark.asyncio
async def test_prepare_hlcvs_combined_impl_uses_ohlcv_source_for_volume_normalization(monkeypatch):
    """Verify production combined flow uses ohlcv_source for volume normalization inputs."""
    import hlcv_preparation as hp

    start_ts = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
    candle_df = pd.DataFrame(
        {
            "timestamp": [start_ts, start_ts + 60_000, start_ts + 120_000],
            "open": [1.0, 1.0, 1.0],
            "high": [2.0, 2.0, 2.0],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )

    class DummyOM:
        def __init__(self, exchange_id: str):
            self.exchange_id = exchange_id

        async def load_markets(self):
            return None

        def get_symbol(self, coin):
            return coin

        def get_market_specific_settings(self, _coin):
            return {"exchange": self.exchange_id, "min_cost": 1.0}

    om_dict = {"binanceusdm": DummyOM("binanceusdm"), "bybit": DummyOM("bybit")}

    async def fake_get_first_timestamps_unified(_coins):
        return {"BTC": start_ts}

    async def fake_fetch_data_for_coin_and_exchange(coin, ex, *_args, **_kwargs):
        if coin != "BTC":
            return None
        if ex == "binanceusdm":
            return ex, candle_df.copy(), 3, 0, 1_000.0
        if ex == "bybit":
            return ex, candle_df.copy(), 2, 0, 500.0
        return None

    async def fake_compute_exchange_volume_ratios(
        exchanges_with_data,
        _valid_coins,
        _start_date,
        _end_date,
        om_map,
    ):
        # This is the key behavior under test:
        # market_settings_sources should not force bybit into normalization exchange set.
        assert exchanges_with_data == ["binance"]
        assert set(om_map.keys()) == {"binance"}
        return {}

    monkeypatch.setattr(hp, "get_first_timestamps_unified", fake_get_first_timestamps_unified)
    monkeypatch.setattr(hp, "fetch_data_for_coin_and_exchange", fake_fetch_data_for_coin_and_exchange)
    monkeypatch.setattr(hp, "compute_exchange_volume_ratios", fake_compute_exchange_volume_ratios)

    config = {
        "backtest": {"gap_tolerance_ohlcvs_minutes": 120},
        "bot": {"long": {}, "short": {}},
        "live": {
            "approved_coins": {"long": ["BTC/USDT:USDT"], "short": []},
            "minimum_coin_age_days": 0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
    }

    mss, _timestamps, unified_array = await hp._prepare_hlcvs_combined_impl(
        config=config,
        om_dict=om_dict,
        base_start_ts=start_ts,
        _requested_start_ts=start_ts,
        end_ts=start_ts + 180_000,
        forced_sources={},
        market_settings_sources={"BTC": "bybit"},
    )

    assert mss["BTC"]["exchange"] == "bybit"
    assert mss["BTC"]["ohlcv_source"] == "binance"
    assert unified_array[:, 0, 3].sum() == pytest.approx(candle_df["volume"].sum())
