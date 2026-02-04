"""
End-to-end test: Run an actual backtest with market_settings_sources.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_test_config():
    """Full config for a minimal backtest."""
    from config_utils import get_template_config

    cfg = get_template_config()

    # Minimal date range for fast test
    cfg["backtest"]["start_date"] = "2025-01-01"
    cfg["backtest"]["end_date"] = "2025-01-03"
    cfg["backtest"]["exchanges"] = ["binance", "hyperliquid"]

    # The key feature: use binance OHLCV with hyperliquid market settings
    cfg["backtest"]["coin_sources"] = {"ETH": "binance"}
    cfg["backtest"]["market_settings_sources"] = {"ETH": "hyperliquid"}

    # Single coin for simplicity
    cfg["live"]["approved_coins"] = {"long": ["ETH"], "short": []}
    cfg["live"]["ignored_coins"] = {"long": [], "short": []}
    cfg["live"]["minimum_coin_age_days"] = 0

    # Minimal bot params
    cfg["bot"]["long"]["n_positions"] = 1
    cfg["bot"]["short"]["n_positions"] = 0

    return cfg


async def run_test():
    from backtest import prepare_hlcvs_mss

    cfg = get_test_config()

    logging.info("=" * 60)
    logging.info("Testing market_settings_sources end-to-end")
    logging.info("=" * 60)
    logging.info(f"coin_sources: {cfg['backtest']['coin_sources']}")
    logging.info(f"market_settings_sources: {cfg['backtest']['market_settings_sources']}")
    logging.info("=" * 60)

    # This is the main backtest preparation function
    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = (
        await prepare_hlcvs_mss(cfg, "combined")
    )

    logging.info(f"Coins loaded: {coins}")
    logging.info(f"HLCVS shape: {hlcvs.shape}")
    logging.info(f"Timestamps: {len(timestamps)} points")

    # Verify the market settings
    for coin in coins:
        coin_mss = mss[coin]
        logging.info(f"\n{coin} market settings:")
        logging.info(f"  exchange (settings source): {coin_mss.get('exchange')}")
        logging.info(f"  ohlcv_source: {coin_mss.get('ohlcv_source', 'same as exchange')}")
        logging.info(f"  min_cost: {coin_mss.get('min_cost')}")
        logging.info(f"  min_qty: {coin_mss.get('min_qty')}")
        logging.info(f"  qty_step: {coin_mss.get('qty_step')}")
        logging.info(f"  c_mult: {coin_mss.get('c_mult')}")
        logging.info(f"  maker_fee: {coin_mss.get('maker_fee')}")
        logging.info(f"  taker_fee: {coin_mss.get('taker_fee')}")

    # Verify ETH specifically
    if "ETH" in mss:
        eth = mss["ETH"]
        assert eth["exchange"] == "hyperliquid", f"Expected hyperliquid, got {eth['exchange']}"
        assert eth.get("ohlcv_source") == "binance", f"Expected binance ohlcv_source"
        logging.info("\n" + "=" * 60)
        logging.info("SUCCESS: ETH has hyperliquid settings with binance OHLCV source")
        logging.info("=" * 60)
    else:
        logging.error("ETH not found in mss!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_test())
