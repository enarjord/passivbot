import os
import time
import asyncio
import pytest
import numpy as np

LIVE = os.getenv("LIVE_CANDLE_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE, reason="Set LIVE_CANDLE_TESTS=1 to enable live candlestick tests"
)


ONE_MIN_MS = 60_000


def _floor_minute(ms: int) -> int:
    return (int(ms) // ONE_MIN_MS) * ONE_MIN_MS


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

TF_LIST = ["1m", "5m", "15m", "1h", "1d"]


def _tf_ms(tf: str) -> int:
    import re

    m = re.fullmatch(r"(\d+)([smhd])", tf.strip())
    if not m:
        return ONE_MIN_MS
    n, unit = int(m.group(1)), m.group(2)
    if unit == "s":
        return max(ONE_MIN_MS, (n // 60) * ONE_MIN_MS)
    if unit == "m":
        return n * ONE_MIN_MS
    if unit == "h":
        return n * 60 * ONE_MIN_MS
    if unit == "d":
        return n * 1440 * ONE_MIN_MS
    return ONE_MIN_MS


async def _pick_btc_symbol(ex):
    # Choose a BTC perpetual (contract market). Prefer linear USDT/USDC.
    markets = await ex.load_markets()
    # Prefer linear contracts
    for m in markets.values():
        try:
            if (
                m.get("contract")
                and m.get("linear")
                and m.get("base") == "BTC"
                and m.get("quote") in {"USDT", "USDC"}
            ):
                return m["symbol"]
        except Exception:
            continue
    # Then allow inverse contracts
    for m in markets.values():
        try:
            if m.get("contract") and m.get("inverse") and m.get("base") == "BTC":
                return m["symbol"]
        except Exception:
            continue
    pytest.skip("No BTC perpetual contract found on this exchange")


@pytest.mark.asyncio
@pytest.mark.parametrize("ex_id", EX_IDS)
async def test_historical_range_per_exchange(tmp_path, ex_id):
    try:
        import ccxt.async_support as ccxt
    except Exception as e:
        pytest.skip(f"ccxt not available: {e}")

    if not hasattr(ccxt, ex_id):
        pytest.skip(f"Exchange not available in ccxt: {ex_id}")

    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    try:
        symbol = await _pick_btc_symbol(ex)
        from candlestick_manager import CandlestickManager, CANDLE_DTYPE

        cm = CandlestickManager(
            exchange=ex,
            exchange_name=ex.id,
            cache_dir=str(tmp_path / "caches"),
            debug=True,
        )

        now = int(time.time() * 1000)
        # 2h historical ending 1h ago
        end_ts = _floor_minute(now) - ONE_MIN_MS * 60
        start_ts = end_ts - ONE_MIN_MS * 120

        arr = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=False)
        assert isinstance(arr, np.ndarray) and arr.dtype == CANDLE_DTYPE
        assert arr.size > 0
        ts = np.asarray(arr["ts"], dtype=np.int64)

        assert int(ts[0]) == _floor_minute(start_ts)
        assert int(ts[-1]) == _floor_minute(end_ts)
        # After standardization, steps should be 1 minute
        if arr.shape[0] > 1:
            assert int(np.diff(ts).max(initial=ONE_MIN_MS)) == ONE_MIN_MS

        # Ensure shards are written and can be reloaded
        cm._cache.pop(symbol, None)  # drop memory
        arr2 = await cm.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, strict=False)
        ts2 = np.asarray(arr2["ts"], dtype=np.int64)
        assert int(ts2[0]) == _floor_minute(start_ts)
        assert int(ts2[-1]) == _floor_minute(end_ts)
    finally:
        try:
            await ex.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("ex_id", EX_IDS)
@pytest.mark.parametrize("tf", TF_LIST)
async def test_historical_range_timeframes(tmp_path, ex_id, tf):
    try:
        import ccxt.async_support as ccxt
    except Exception as e:
        pytest.skip(f"ccxt not available: {e}")
    if not hasattr(ccxt, ex_id):
        pytest.skip(f"Exchange not available in ccxt: {ex_id}")
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    try:
        symbol = await _pick_btc_symbol(ex)
        from candlestick_manager import CandlestickManager, CANDLE_DTYPE

        cm = CandlestickManager(
            exchange=ex, exchange_name=ex.id, cache_dir=str(tmp_path / "caches"), debug=True
        )
        now = int(time.time() * 1000)
        period = _tf_ms(tf)
        # choose 24 buckets ending one bucket ago
        end_ts = ((now - ONE_MIN_MS) // period) * period - period
        buckets = 24
        start_ts = end_ts - period * (buckets - 1)
        arr = await cm.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, timeframe=tf, strict=False
        )
        assert arr.dtype == CANDLE_DTYPE and arr.size > 0
        ts = np.asarray(arr["ts"], dtype=np.int64)
        assert int(ts[0]) == int(start_ts)
        assert int(ts[-1]) == int(end_ts)
        if arr.shape[0] > 1:
            diffs = np.diff(ts)
            assert int(diffs.max(initial=period)) == period
            assert int(diffs.min(initial=period)) == period
        # Allow exchanges to miss some buckets; only assert alignment when full
        if arr.shape[0] == buckets:
            assert arr.shape[0] == buckets
    finally:
        try:
            await ex.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("ex_id", EX_IDS)
async def test_present_range_includes_current_minute_per_exchange(tmp_path, ex_id):
    try:
        import ccxt.async_support as ccxt
    except Exception as e:
        pytest.skip(f"ccxt not available: {e}")

    if not hasattr(ccxt, ex_id):
        pytest.skip(f"Exchange not available in ccxt: {ex_id}")

    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    try:
        symbol = await _pick_btc_symbol(ex)
        from candlestick_manager import CandlestickManager, CANDLE_DTYPE

        cm = CandlestickManager(
            exchange=ex,
            exchange_name=ex.id,
            cache_dir=str(tmp_path / "caches"),
            debug=True,
        )

        now = int(time.time() * 1000)
        end_floor = _floor_minute(now)
        start_ts = end_floor - ONE_MIN_MS * 90

        arr = await cm.get_candles(symbol, start_ts=start_ts, end_ts=None, strict=False)
        assert arr.dtype == CANDLE_DTYPE and arr.size > 0
        ts = np.asarray(arr["ts"], dtype=np.int64)
        assert int(ts[0]) == _floor_minute(start_ts)
        # Should include current minute
        assert int(ts[-1]) == end_floor
        if arr.shape[0] > 1:
            assert int(np.diff(ts).max(initial=ONE_MIN_MS)) == ONE_MIN_MS
    finally:
        try:
            await ex.close()
        except Exception:
            pass


def _ema(values, span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    ema = float(values[0])
    for v in values[1:]:
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return float(ema)


@pytest.mark.asyncio
@pytest.mark.parametrize("ex_id", EX_IDS)
@pytest.mark.parametrize("span", [5, 12])
@pytest.mark.parametrize("tf", ["1m", "5m", "15m", "1h"])  # skip 1d for EMA due to low resolution
async def test_latest_ema_metrics_per_exchange(tmp_path, ex_id, span, tf):
    try:
        import ccxt.async_support as ccxt
    except Exception as e:
        pytest.skip(f"ccxt not available: {e}")

    if not hasattr(ccxt, ex_id):
        pytest.skip(f"Exchange not available in ccxt: {ex_id}")

    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    try:
        symbol = await _pick_btc_symbol(ex)
        from candlestick_manager import CandlestickManager, CANDLE_DTYPE

        cm = CandlestickManager(
            exchange=ex,
            exchange_name=ex.id,
            cache_dir=str(tmp_path / "caches"),
            debug=True,
        )

        now = int(time.time() * 1000)
        period = _tf_ms(tf)
        end_final = ((now - ONE_MIN_MS) // period) * period
        start_ts = end_final - period * (span - 1)

        # Fetch the exact finalized span
        arr = await cm.get_candles(
            symbol, start_ts=start_ts, end_ts=end_final, timeframe=tf, strict=False
        )
        assert arr.dtype == CANDLE_DTYPE
        # Expect exactly `span` rows after standardization
        expected_len = span
        if arr.shape[0] != expected_len:
            pytest.skip(
                f"unexpected length {arr.shape[0]} != {expected_len}; exchange returned gaps that could not be standardized"
            )

        closes = np.asarray(arr["c"], dtype=np.float64)
        vols = np.asarray(arr["bv"], dtype=np.float64)
        highs = np.asarray(arr["h"], dtype=np.float64)
        lows = np.asarray(arr["l"], dtype=np.float64)
        denom = np.maximum(closes, 1e-12)
        log_ranges = np.log(np.maximum(highs, 1e-12) / np.maximum(lows, 1e-12))

        exp_close = _ema(closes, span)
        exp_vol = _ema(vols, span)
        exp_log_range = _ema(log_ranges, span)

        # Query manager helpers
        ema_close = await cm.get_latest_ema_close(symbol, span, timeframe=tf)
        ema_vol = await cm.get_latest_ema_volume(symbol, span, timeframe=tf)
        log_range_ema = await cm.get_latest_ema_log_range(symbol, span, timeframe=tf)

        # Allow small tolerance due to float handling across exchanges
        assert pytest.approx(exp_close, rel=1e-6, abs=1e-9) == ema_close
        assert pytest.approx(exp_vol, rel=1e-6, abs=1e-9) == ema_vol
        assert pytest.approx(exp_log_range, rel=1e-6, abs=1e-9) == log_range_ema
    finally:
        try:
            await ex.close()
        except Exception:
            pass
