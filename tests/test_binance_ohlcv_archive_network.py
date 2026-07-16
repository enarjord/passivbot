import os

import pytest

from binance_ohlcv_archive import BinanceArchiveRequest, BinanceOhlcvArchiveClient
from hlcv_preparation import HLCVManager, _fetch_coin_range_into_v2_store
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts


RUN_NETWORK_TESTS = os.environ.get("PASSIVBOT_RUN_PUBLIC_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="set PASSIVBOT_RUN_PUBLIC_NETWORK_TESTS=1 for public Binance download tests",
)


def archive_request(kind, period_key, start_ts, end_ts, missing_candles):
    return BinanceArchiveRequest(
        kind=kind,
        symbol_code="BTCUSDT",
        period_key=period_key,
        start_ts=start_ts,
        end_ts=end_ts,
        missing_candles=missing_candles,
    )


@pytest.mark.asyncio
async def test_real_binance_monthly_and_daily_archives_verify_and_parse_in_parallel():
    january_start = month_start_ts(2024, 1)
    february_start = month_start_ts(2024, 2)
    requests = [
        archive_request(
            "monthly",
            "2024-01",
            january_start,
            february_start - 60_000,
            31 * 1440,
        ),
        archive_request(
            "daily",
            "2024-02-01",
            february_start,
            february_start + 1439 * 60_000,
            1440,
        ),
    ]
    client = BinanceOhlcvArchiveClient(max_concurrency=2)
    try:
        monthly, daily = await client.fetch_many(requests)
    finally:
        await client.aclose()

    assert monthly.status == "ok", monthly.error
    assert daily.status == "ok", daily.error
    assert monthly.frame is not None and len(monthly.frame) == 31 * 1440
    assert daily.frame is not None and len(daily.frame) == 1440
    assert monthly.frame["timestamp"].is_monotonic_increasing
    assert daily.frame["timestamp"].is_monotonic_increasing


@pytest.mark.asyncio
async def test_real_v2_downloader_uses_monthly_then_daily_archives(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "ohlcvs", catalog)
    manager = HLCVManager("binance", verbose=False)
    january_start = month_start_ts(2024, 1)
    january_ninth = january_start + 8 * 1440 * 60_000
    february_start = month_start_ts(2024, 2)
    february_end = february_start + 1439 * 60_000

    try:
        monthly = await _fetch_coin_range_into_v2_store(
            om=manager,
            catalog=catalog,
            store=store,
            exchange="binance",
            coin="BTC",
            symbol="BTC/USDT:USDT",
            start_ts=january_start,
            end_ts=january_ninth,
        )
        daily = await _fetch_coin_range_into_v2_store(
            om=manager,
            catalog=catalog,
            store=store,
            exchange="binance",
            coin="BTC",
            symbol="BTC/USDT:USDT",
            start_ts=february_start,
            end_ts=february_end,
        )
    finally:
        await manager.aclose()

    assert monthly.ok and monthly.reason == "binance_archive_ok"
    assert daily.ok and daily.reason == "binance_archive_ok"
    assert store.read_range(
        "binance", "1m", "BTC/USDT:USDT", january_start, january_ninth
    ).valid.all()
    assert store.read_range(
        "binance", "1m", "BTC/USDT:USDT", february_start, february_end
    ).valid.all()
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "BTC/USDT:USDT", january_start, february_end
    )
    assert [attempt.outcome for attempt in attempts] == [
        "archive_monthly_ok",
        "archive_daily_ok",
    ]


@pytest.mark.asyncio
async def test_real_v2_downloader_uses_ccxt_for_small_gap(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "ohlcvs", catalog)
    manager = HLCVManager("binance", verbose=False, gap_tolerance_ohlcvs_minutes=0.0)
    start_ts = month_start_ts(2024, 2) + 12 * 60 * 60_000
    end_ts = start_ts + 2 * 60_000

    try:
        result = await _fetch_coin_range_into_v2_store(
            om=manager,
            catalog=catalog,
            store=store,
            exchange="binance",
            coin="BTC",
            symbol="BTC/USDT:USDT",
            start_ts=start_ts,
            end_ts=end_ts,
        )
    finally:
        await manager.aclose()
        if manager.cc is not None:
            await manager.cc.close()

    assert result.ok and result.reason == "ok"
    assert result.wrote_rows == 3
    assert store.read_range(
        "binance", "1m", "BTC/USDT:USDT", start_ts, end_ts
    ).valid.all()
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "BTC/USDT:USDT", start_ts, end_ts
    )
    assert [attempt.outcome for attempt in attempts] == ["ok"]
