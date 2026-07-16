import asyncio
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import pytest

from binance_ohlcv_archive import (
    BinanceArchiveIntegrityError,
    BinanceArchiveRequest,
    BinanceOhlcvArchiveClient,
    _parse_archive_zip,
    _parse_checksum,
    daily_archive_eligible,
    first_monday_after_month,
    monthly_archive_eligible,
    plan_binance_archive_requests,
    symbol_to_archive_code,
)


def ts(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def minute_grid(start: str, minutes: int) -> np.ndarray:
    return np.arange(ts(start), ts(start) + minutes * 60_000, 60_000, dtype=np.int64)


def make_zip(rows: list[list[object]], filename: str = "BTCUSDT-1m.csv") -> bytes:
    csv_payload = "\n".join(",".join(str(value) for value in row) for row in rows).encode()
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, csv_payload)
    return output.getvalue()


def request(kind: str = "daily") -> BinanceArchiveRequest:
    if kind == "monthly":
        return BinanceArchiveRequest(
            kind="monthly",
            symbol_code="BTCUSDT",
            period_key="2026-04",
            start_ts=ts("2026-04-01"),
            end_ts=ts("2026-04-30T23:59:00"),
            missing_candles=20_000,
        )
    return BinanceArchiveRequest(
        kind="daily",
        symbol_code="BTCUSDT",
        period_key="2026-04-01",
        start_ts=ts("2026-04-01"),
        end_ts=ts("2026-04-01T23:59:00"),
        missing_candles=1440,
    )


def valid_row(timestamp: int, *, close: float = 100.0) -> list[object]:
    return [
        timestamp,
        close,
        close + 1.0,
        close - 1.0,
        close + 0.5,
        12.0,
        timestamp + 59_999,
        0,
        1,
        0,
        0,
        0,
    ]


def test_archive_urls_use_usdm_monthly_and_daily_paths():
    monthly = request("monthly")
    daily = request("daily")

    assert monthly.url.endswith(
        "/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-04.zip"
    )
    assert daily.url.endswith(
        "/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2026-04-01.zip"
    )


def test_first_monday_and_monthly_publication_buffer():
    assert first_monday_after_month(2026, 4) == datetime(2026, 5, 4, tzinfo=timezone.utc)
    before_buffer = ts("2026-05-04T23:59:59")
    after_buffer = ts("2026-05-05")

    assert not monthly_archive_eligible(2026, 4, now_ms=before_buffer)
    assert monthly_archive_eligible(2026, 4, now_ms=after_buffer)
    assert not monthly_archive_eligible(2026, 5, now_ms=ts("2026-05-31T23:59:00"))


def test_daily_archive_requires_two_complete_lag_days():
    now_ms = ts("2026-07-16T12:00:00")
    assert daily_archive_eligible(2026, 7, 13, now_ms=now_ms)
    assert not daily_archive_eligible(2026, 7, 14, now_ms=now_ms)
    assert not daily_archive_eligible(2026, 7, 16, now_ms=now_ms)


def test_monthly_threshold_counts_missing_candles_not_affected_dates():
    timestamps = minute_grid("2026-04-01", 8 * 24 * 60)
    valid = np.ones(len(timestamps), dtype=np.bool_)
    valid[::1440] = False

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16"),
    )

    assert monthly == []
    assert daily == []


def test_more_than_seven_missing_days_selects_monthly_archive():
    timestamps = minute_grid("2026-04-01", 8 * 24 * 60 + 1)
    valid = np.zeros(len(timestamps), dtype=np.bool_)

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16"),
    )

    assert [item.period_key for item in monthly] == ["2026-04"]
    assert monthly[0].missing_candles == 8 * 24 * 60 + 1
    assert daily == []


def test_exactly_seven_missing_days_does_not_select_monthly():
    timestamps = minute_grid("2026-04-01", 7 * 24 * 60)
    valid = np.zeros(len(timestamps), dtype=np.bool_)

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16"),
    )

    assert monthly == []
    assert len(daily) == 7


def test_current_month_never_selects_monthly_but_old_days_select_daily():
    timestamps = minute_grid("2026-07-01", 13 * 24 * 60)
    valid = np.zeros(len(timestamps), dtype=np.bool_)

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16T12:00:00"),
    )

    assert monthly == []
    assert daily[0].period_key == "2026-07-01"
    assert daily[-1].period_key == "2026-07-13"


def test_small_fragment_uses_neither_archive():
    timestamps = minute_grid("2026-04-01", 1000)
    valid = np.zeros(len(timestamps), dtype=np.bool_)

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16"),
    )

    assert monthly == []
    assert daily == []


def test_more_than_one_thousand_missing_bars_selects_daily_archive():
    timestamps = minute_grid("2026-04-10", 1001)
    valid = np.zeros(len(timestamps), dtype=np.bool_)

    monthly, daily = plan_binance_archive_requests(
        timestamps,
        valid,
        symbol_code="BTCUSDT",
        now_ms=ts("2026-07-16"),
    )

    assert monthly == []
    assert [item.period_key for item in daily] == ["2026-04-10"]
    assert daily[0].missing_candles == 1001


def test_checksum_parser_validates_digest_and_filename():
    digest = "a" * 64
    assert _parse_checksum(f"{digest}  BTCUSDT-1m-2026-04.zip\n".encode(), "BTCUSDT-1m-2026-04.zip") == digest

    with pytest.raises(BinanceArchiveIntegrityError, match="expected"):
        _parse_checksum(f"{digest}  ETHUSDT.zip\n".encode(), "BTCUSDT.zip")
    with pytest.raises(BinanceArchiveIntegrityError, match="invalid SHA-256"):
        _parse_checksum(b"not-a-digest file.zip", "file.zip")


def test_zip_parser_preserves_real_internal_gaps():
    req = request("daily")
    raw = make_zip([valid_row(req.start_ts), valid_row(req.start_ts + 2 * 60_000)])

    frame = _parse_archive_zip(raw, req)

    assert frame["timestamp"].tolist() == [req.start_ts, req.start_ts + 2 * 60_000]


def test_zip_parser_accepts_header_and_identical_duplicate():
    req = request("daily")
    header = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    row = valid_row(req.start_ts)
    raw = make_zip([header, row, row])

    frame = _parse_archive_zip(raw, req)

    assert len(frame) == 1


def test_zip_parser_rejects_conflicting_duplicates_and_bad_price_bounds():
    req = request("daily")
    row = valid_row(req.start_ts)
    conflicting = valid_row(req.start_ts, close=200.0)
    with pytest.raises(BinanceArchiveIntegrityError, match="conflicting"):
        _parse_archive_zip(make_zip([row, conflicting]), req)

    bad = valid_row(req.start_ts)
    bad[2] = 90.0
    with pytest.raises(BinanceArchiveIntegrityError, match="price bounds"):
        _parse_archive_zip(make_zip([bad]), req)


def test_symbol_to_archive_code_handles_ccxt_perpetual_symbols():
    assert symbol_to_archive_code("BTC/USDT:USDT") == "BTCUSDT"
    assert symbol_to_archive_code("1000SHIB/USDT:USDT") == "1000SHIBUSDT"


@pytest.mark.asyncio
async def test_client_verifies_checksum_and_parses_archive():
    req = request("daily")
    raw = make_zip([valid_row(req.start_ts)])
    digest = hashlib.sha256(raw).hexdigest()

    class FakeClient(BinanceOhlcvArchiveClient):
        async def _get(self, url):
            if url.endswith(".CHECKSUM"):
                return 200, f"{digest}  {req.filename}\n".encode()
            return 200, raw

    client = FakeClient()
    result = await client.fetch(req)

    assert result.status == "ok"
    assert result.frame is not None
    assert result.frame["timestamp"].tolist() == [req.start_ts]


@pytest.mark.asyncio
async def test_client_checksum_mismatch_and_404_are_fallback_results():
    req = request("daily")
    raw = make_zip([valid_row(req.start_ts)])

    class MismatchClient(BinanceOhlcvArchiveClient):
        async def _get(self, url):
            if url.endswith(".CHECKSUM"):
                return 200, f"{'0' * 64}  {req.filename}\n".encode()
            return 200, raw

    mismatch = await MismatchClient().fetch(req)
    assert mismatch.status == "error"
    assert "SHA-256 mismatch" in (mismatch.error or "")

    class MissingClient(BinanceOhlcvArchiveClient):
        async def _get(self, _url):
            return 404, b""

    missing = await MissingClient().fetch(req)
    assert missing.status == "not_found"


@pytest.mark.asyncio
async def test_fetch_many_propagates_cancellation_and_cancels_siblings():
    requests = [request("daily"), request("monthly")]
    started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    started_count = 0

    class SlowClient(BinanceOhlcvArchiveClient):
        async def fetch(self, archive_request):
            nonlocal started_count
            started_count += 1
            if started_count == len(requests):
                started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    task = asyncio.create_task(SlowClient().fetch_many(requests))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sibling_cancelled.is_set()
