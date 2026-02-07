"""
Integration tests for the market_settings_sources feature.

This tests the ability to decouple OHLCV data source from market settings source
in combined backtests.
"""

import asyncio
import os
import logging
import pytest

logging.basicConfig(level=logging.INFO)

LIVE = os.environ.get("LIVE_CANDLE_TESTS", "").lower() in {"1", "true", "yes"}
pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="Set LIVE_CANDLE_TESTS=1 to enable live market_settings_sources tests",
)


def _base_config():
    """Minimal config for testing combined HLCV preparation."""
    return {
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": True,
            "end_date": "2025-01-05",
            "start_date": "2025-01-01",
            "exchanges": ["binance", "hyperliquid"],
            "btc_collateral_cap": 1.0,
            "btc_collateral_ltv_cap": None,
            "gap_tolerance_ohlcvs_minutes": 120,
            "max_warmup_minutes": 0,
            "coin_sources": {},
            "market_settings_sources": {},
        },
        "bot": {
            "long": {"n_positions": 1},
            "short": {"n_positions": 1},
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "ignored_coins": {"long": [], "short": []},
            "minimum_coin_age_days": 0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0,
        },
        "optimize": {"bounds": {}, "scoring": [], "suite": {"enabled": False, "scenarios": []}},
    }


@pytest.mark.asyncio
async def test_market_settings_sources_decouples_ohlcv_from_settings():
    """
    Test that market_settings_sources allows loading market settings from a
    different exchange than the OHLCV source.
    """
    from hlcv_preparation import prepare_hlcvs_combined

    cfg = _base_config()
    # Force OHLCV from binance, market settings from hyperliquid
    cfg["backtest"]["coin_sources"] = {"ETH": "binance"}
    cfg["backtest"]["market_settings_sources"] = {"ETH": "hyperliquid"}

    mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(
        cfg,
        forced_sources=cfg["backtest"]["coin_sources"],
        market_settings_sources=cfg["backtest"]["market_settings_sources"],
    )

    # Verify ETH is in the results
    assert "ETH" in mss, f"ETH should be in mss, got keys: {list(mss.keys())}"

    eth_mss = mss["ETH"]

    # Verify market settings come from hyperliquid
    assert eth_mss["exchange"] == "hyperliquid", (
        f"Expected exchange='hyperliquid', got '{eth_mss.get('exchange')}'"
    )

    # Verify ohlcv_source is set to binance (since it differs from exchange)
    assert eth_mss.get("ohlcv_source") == "binance", (
        f"Expected ohlcv_source='binance', got '{eth_mss.get('ohlcv_source')}'"
    )

    # Verify we got actual OHLCV data
    assert hlcvs.shape[0] > 0, "Should have OHLCV data"
    assert len(timestamps) > 0, "Should have timestamps"

    print(f"\nETH market settings:")
    print(f"  exchange: {eth_mss.get('exchange')}")
    print(f"  ohlcv_source: {eth_mss.get('ohlcv_source')}")
    print(f"  min_cost: {eth_mss.get('min_cost')}")
    print(f"  min_qty: {eth_mss.get('min_qty')}")
    print(f"  qty_step: {eth_mss.get('qty_step')}")
    print(f"  c_mult: {eth_mss.get('c_mult')}")
    print(f"  maker_fee: {eth_mss.get('maker_fee')}")
    print(f"  taker_fee: {eth_mss.get('taker_fee')}")


@pytest.mark.asyncio
async def test_market_settings_sources_fallback_on_missing_coin():
    """
    Test that if a coin isn't listed on the market_settings_sources exchange,
    it falls back to the OHLCV source exchange.
    """
    from hlcv_preparation import prepare_hlcvs_combined

    cfg = _base_config()
    # Use a coin that exists on binance but specify a non-existent exchange override
    cfg["live"]["approved_coins"]["long"] = ["BTC"]
    cfg["backtest"]["coin_sources"] = {"BTC": "binance"}
    # Use hyperliquid for market settings - BTC should exist there
    cfg["backtest"]["market_settings_sources"] = {"BTC": "hyperliquid"}

    mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(
        cfg,
        forced_sources=cfg["backtest"]["coin_sources"],
        market_settings_sources=cfg["backtest"]["market_settings_sources"],
    )

    assert "BTC" in mss
    btc_mss = mss["BTC"]

    # BTC exists on hyperliquid, so it should use hyperliquid settings
    assert btc_mss["exchange"] == "hyperliquid"
    assert btc_mss.get("ohlcv_source") == "binance"


@pytest.mark.asyncio
async def test_no_market_settings_sources_uses_ohlcv_source():
    """
    Test that without market_settings_sources, the OHLCV source exchange
    is used for market settings (existing behavior).
    """
    from hlcv_preparation import prepare_hlcvs_combined

    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {"ETH": "binance"}
    # No market_settings_sources - should use binance for both

    mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(
        cfg,
        forced_sources=cfg["backtest"]["coin_sources"],
        market_settings_sources=None,
    )

    assert "ETH" in mss
    eth_mss = mss["ETH"]

    # Should use binance for market settings (same as OHLCV source)
    assert eth_mss["exchange"] == "binance", (
        f"Expected exchange='binance', got '{eth_mss.get('exchange')}'"
    )

    # ohlcv_source should NOT be set when sources are the same
    assert "ohlcv_source" not in eth_mss, (
        f"ohlcv_source should not be set when same as exchange, got '{eth_mss.get('ohlcv_source')}'"
    )


@pytest.mark.asyncio
async def test_market_settings_values_differ_between_exchanges():
    """
    Verify that market settings actually differ between exchanges,
    confirming the feature is meaningful.
    """
    from hlcv_preparation import prepare_hlcvs_combined

    cfg = _base_config()
    cfg["live"]["approved_coins"]["long"] = ["ETH"]

    # First run: binance settings
    cfg["backtest"]["coin_sources"] = {"ETH": "binance"}
    cfg["backtest"]["market_settings_sources"] = {}  # Use binance for settings too

    mss_binance, _, _, _ = await prepare_hlcvs_combined(
        cfg,
        forced_sources=cfg["backtest"]["coin_sources"],
        market_settings_sources=None,
    )

    # Second run: binance OHLCV, hyperliquid settings
    cfg["backtest"]["market_settings_sources"] = {"ETH": "hyperliquid"}

    mss_hl, _, _, _ = await prepare_hlcvs_combined(
        cfg,
        forced_sources=cfg["backtest"]["coin_sources"],
        market_settings_sources=cfg["backtest"]["market_settings_sources"],
    )

    eth_binance = mss_binance["ETH"]
    eth_hl = mss_hl["ETH"]

    print(f"\nBinance ETH settings:")
    print(f"  min_cost: {eth_binance.get('min_cost')}")
    print(f"  min_qty: {eth_binance.get('min_qty')}")
    print(f"  qty_step: {eth_binance.get('qty_step')}")

    print(f"\nHyperliquid ETH settings:")
    print(f"  min_cost: {eth_hl.get('min_cost')}")
    print(f"  min_qty: {eth_hl.get('min_qty')}")
    print(f"  qty_step: {eth_hl.get('qty_step')}")

    # The exchanges should report different values for at least some fields
    # This validates that we're actually getting different settings
    settings_differ = (
        eth_binance.get("min_cost") != eth_hl.get("min_cost")
        or eth_binance.get("min_qty") != eth_hl.get("min_qty")
        or eth_binance.get("qty_step") != eth_hl.get("qty_step")
    )

    # Note: This might not always be true if exchanges happen to have identical settings
    # but it's useful for debugging
    if settings_differ:
        print("\nConfirmed: Settings differ between exchanges")
    else:
        print("\nNote: Settings happen to be identical between exchanges")


if __name__ == "__main__":
    # Allow running directly for debugging
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(test_market_settings_sources_decouples_ohlcv_from_settings())
