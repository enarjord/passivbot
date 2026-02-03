#!/usr/bin/env python3
"""
Test script for TradFi data providers.

Tests multiple data sources for historical 1m stock data:
- Alpaca (5+ years free, recommended)
- Polygon.io (2 years free)
- yfinance (7 days, no API key)

Usage:
    python test_tradfi_providers.py [--provider NAME] [--api-key KEY] [--api-secret SECRET]

Examples:
    # Test yfinance (no API key needed)
    python test_tradfi_providers.py --provider yfinance

    # Test Alpaca
    python test_tradfi_providers.py --provider alpaca --api-key KEY --api-secret SECRET

    # Test Polygon
    python test_tradfi_providers.py --provider polygon --api-key KEY
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tradfi_data import (
    AlpacaProvider,
    PolygonProvider,
    YFinanceProvider,
    TradFiDataFetcher,
    get_provider,
    candles_to_array,
)


async def test_provider(provider, symbol: str = "TSLA", days_ago: int = 7):
    """Test a single provider."""
    print(f"\n{'=' * 60}")
    print(f"Testing {provider.name.upper()} Provider")
    print(f"{'=' * 60}")

    async with provider:
        # Calculate date range
        end_dt = datetime.now(UTC) - timedelta(days=1)
        start_dt = end_dt - timedelta(days=days_ago)

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        print(f"\nFetching {symbol} 1m candles:")
        print(f"  From: {start_dt.strftime('%Y-%m-%d')}")
        print(f"  To:   {end_dt.strftime('%Y-%m-%d')}")

        try:
            candles = await provider.fetch_1m_candles(symbol, start_ts, end_ts)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            return False

        if not candles:
            print("\n  No candles returned (possibly weekend/holiday)")
            return False

        print(f"\n  SUCCESS: Got {len(candles)} candles")

        # Show sample candles
        print("\n  First 3 candles:")
        for c in candles[:3]:
            dt = datetime.fromtimestamp(c.timestamp_ms / 1000, tz=UTC)
            print(
                f"    {dt.strftime('%Y-%m-%d %H:%M')} "
                f"O:{c.open:.2f} H:{c.high:.2f} L:{c.low:.2f} C:{c.close:.2f} V:{c.volume:.0f}"
            )

        print("\n  Last 3 candles:")
        for c in candles[-3:]:
            dt = datetime.fromtimestamp(c.timestamp_ms / 1000, tz=UTC)
            print(
                f"    {dt.strftime('%Y-%m-%d %H:%M')} "
                f"O:{c.open:.2f} H:{c.high:.2f} L:{c.low:.2f} C:{c.close:.2f} V:{c.volume:.0f}"
            )

        # Verify array conversion
        arr = candles_to_array(candles)
        print(f"\n  Array dtype: {arr.dtype}")
        print(f"  Array shape: {arr.shape}")

        return True


async def test_historical_depth(provider, symbol: str = "TSLA"):
    """Test how far back we can fetch data."""
    print(f"\n{'=' * 60}")
    print(f"Testing Historical Depth for {provider.name.upper()}")
    print(f"{'=' * 60}")

    test_points = [
        ("1 week ago", 7),
        ("1 month ago", 30),
        ("3 months ago", 90),
        ("6 months ago", 180),
        ("1 year ago", 365),
        ("2 years ago", 730),
        ("3 years ago", 1095),
        ("5 years ago", 1825),
    ]

    async with provider:
        for label, days_ago in test_points:
            target_dt = datetime.now(UTC) - timedelta(days=days_ago)
            # Test a single weekday
            while target_dt.weekday() >= 5:  # Skip weekends
                target_dt -= timedelta(days=1)

            start_ts = int(target_dt.timestamp() * 1000)
            end_ts = start_ts + 86_400_000  # 1 day

            try:
                candles = await provider.fetch_1m_candles(symbol, start_ts, end_ts)
                status = f"{len(candles)} candles" if candles else "NO DATA"
            except Exception as e:
                status = f"ERROR: {e}"

            print(f"  {label:20s}: {status}")

            # Rate limit delay
            await asyncio.sleep(provider.rate_limit_delay)


async def test_multiple_symbols(provider, symbols: list = ["TSLA", "NVDA", "AAPL"]):
    """Test fetching multiple symbols."""
    print(f"\n{'=' * 60}")
    print(f"Testing Multiple Symbols for {provider.name.upper()}")
    print(f"{'=' * 60}")

    end_dt = datetime.now(UTC) - timedelta(days=1)
    start_dt = end_dt - timedelta(days=1)

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    async with provider:
        for symbol in symbols:
            try:
                candles = await provider.fetch_1m_candles(symbol, start_ts, end_ts)
                status = f"{len(candles)} candles" if candles else "NO DATA"
            except Exception as e:
                status = f"ERROR: {e}"

            print(f"  {symbol}: {status}")

            # Rate limit delay
            await asyncio.sleep(provider.rate_limit_delay)


def load_config():
    """Load tradfi config from api-keys.json."""
    try:
        with open("api-keys.json") as f:
            config = json.load(f)
        return config.get("tradfi", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


async def main():
    parser = argparse.ArgumentParser(description="Test TradFi data providers")
    parser.add_argument(
        "--provider",
        choices=["alpaca", "polygon", "yfinance", "finnhub", "alphavantage"],
        help="Provider to test (default: from api-keys.json or yfinance)",
    )
    parser.add_argument("--api-key", help="API key for provider")
    parser.add_argument("--api-secret", help="API secret (for Alpaca)")
    parser.add_argument("--symbol", default="TSLA", help="Stock symbol to test")
    parser.add_argument(
        "--depth-test", action="store_true", help="Test historical depth"
    )
    parser.add_argument(
        "--multi-symbol", action="store_true", help="Test multiple symbols"
    )
    args = parser.parse_args()

    # Load config
    config = load_config()
    provider_name = args.provider or config.get("provider", "yfinance")
    api_key = args.api_key or config.get("api_key")
    api_secret = args.api_secret or config.get("api_secret")

    print(f"Using provider: {provider_name}")

    # Validate requirements
    if provider_name == "alpaca" and (not api_key or not api_secret):
        print("\nERROR: Alpaca requires both --api-key and --api-secret")
        print("\nTo get free Alpaca API keys:")
        print("  1. Go to https://alpaca.markets/")
        print("  2. Sign up for free (no payment required)")
        print("  3. Go to Paper Trading API Keys")
        print("  4. Generate new keys")
        print("\nAdd to api-keys.json:")
        print('  "tradfi": {')
        print('    "provider": "alpaca",')
        print('    "api_key": "YOUR_KEY",')
        print('    "api_secret": "YOUR_SECRET"')
        print("  }")
        return

    if provider_name == "polygon" and not api_key:
        print("\nERROR: Polygon requires --api-key")
        print("\nTo get free Polygon API key:")
        print("  1. Go to https://polygon.io/ (now massive.com)")
        print("  2. Sign up for free")
        print("  3. Copy API key from dashboard")
        return

    # Create provider
    if provider_name == "alpaca":
        provider = AlpacaProvider(api_key=api_key, api_secret=api_secret)
    elif provider_name == "polygon":
        provider = PolygonProvider(api_key=api_key)
    elif provider_name == "yfinance":
        provider = YFinanceProvider()
    else:
        provider = get_provider(provider_name, api_key)

    # Run tests
    success = await test_provider(provider, args.symbol)

    if not success:
        print("\nBasic test failed. Check your API credentials.")
        return

    if args.depth_test:
        # Recreate provider for depth test
        if provider_name == "alpaca":
            provider = AlpacaProvider(api_key=api_key, api_secret=api_secret)
        elif provider_name == "polygon":
            provider = PolygonProvider(api_key=api_key)
        else:
            provider = YFinanceProvider()

        await test_historical_depth(provider, args.symbol)

    if args.multi_symbol:
        # Recreate provider for multi-symbol test
        if provider_name == "alpaca":
            provider = AlpacaProvider(api_key=api_key, api_secret=api_secret)
        elif provider_name == "polygon":
            provider = PolygonProvider(api_key=api_key)
        else:
            provider = YFinanceProvider()

        await test_multiple_symbols(provider)

    print(f"\n{'=' * 60}")
    print("Tests completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
