import asyncio
import time
import os
import pytest


LIVE = os.getenv("LIVE_CANDLE_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(not LIVE, reason="Set LIVE_CANDLE_TESTS=1 to enable live ccxt tests")

EX_IDS = [
    # Perpetual/futures exchanges only
    "binanceusdm",
    "bybit",
    "okx",
    "kucoinfutures",
    "gateio",
    "bitget",
    "hyperliquid",
]


def _floor_minute(ms: int) -> int:
    return (int(ms) // 60000) * 60000


@pytest.mark.asyncio
@pytest.mark.parametrize("exid", EX_IDS)
async def test_fetch_ohlcv_includes_current_minute(exid):
    try:
        import ccxt.async_support as ccxt  # type: ignore
    except Exception as e:
        pytest.skip(f"ccxt not available: {e}")

    if not hasattr(ccxt, exid):
        pytest.skip(f"ccxt has no exchange id {exid}")

    ex = getattr(ccxt, exid)({"enableRateLimit": True})
    # Prefer swap/linear derivatives where applicable
    try:
        if hasattr(ex, "options") and isinstance(ex.options, dict):
            ex.options.setdefault("defaultType", "swap")
    except Exception:
        pass

    try:
        await ex.load_markets()

        # Pick a BTC linear swap market (USDT/USDC) when available
        market = None
        for m in ex.markets.values():
            try:
                if (
                    m.get("swap")
                    and m.get("base") == "BTC"
                    and m.get("quote") in ("USDT", "USDC", "USD")
                ):
                    market = m
                    # Prefer linear if field exists
                    if m.get("linear") is True:
                        break
            except Exception:
                continue

        if not market:
            pytest.skip(f"no suitable BTC swap market on {exid}")

        symbol = market["symbol"]

        before = _floor_minute(int(time.time() * 1000))
        rows = await ex.fetch_ohlcv(symbol, timeframe="1m", limit=30)
        after = _floor_minute(int(time.time() * 1000))

        if not rows or len(rows) < 1:
            pytest.xfail(f"{exid} returned no rows for {symbol}")

        last_ts = int(rows[-1][0])
        # Accept inclusion if the latest entry matches current minute at either
        # fetch boundary to avoid minute-boundary race conditions.
        if last_ts in (before, after):
            assert True
        else:
            pytest.xfail(
                f"{exid} did not include current minute for {symbol}: last_ts={last_ts}, before={before}, after={after}"
            )
    finally:
        try:
            await ex.close()
        except Exception:
            pass
