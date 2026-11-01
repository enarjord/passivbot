from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from binance_ohlcv_archive import BinanceArchiveResult
from hlcv_preparation import (
    V2FetchResult,
    _fetch_binance_archives_into_v2_store,
    _fetch_coin_range_into_v2_store,
)
from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts


SYMBOL = "BTC/USDT:USDT"
EXCHANGE = "binance"
MINUTE_MS = 60_000


def ts(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def archive_frame(timestamps, *, close=100.0):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    closes = np.full(len(timestamps), close, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(timestamps), 10.0, dtype=np.float64),
        }
    )


def write_rows(store, timestamps, *, close=50.0):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    values = np.column_stack(
        [
            np.full(len(timestamps), close + 1.0, dtype=np.float32),
            np.full(len(timestamps), close - 1.0, dtype=np.float32),
            np.full(len(timestamps), close, dtype=np.float32),
            np.full(len(timestamps), 5.0, dtype=np.float32),
        ]
    )
    store.write_rows(EXCHANGE, "1m", SYMBOL, timestamps, values)


class RecordingArchiveClient:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.batches = []

    async def fetch_many(self, requests):
        self.batches.append(list(requests))
        return [self.result_factory(request) for request in requests]


class ArchiveManager:
    gap_tolerance_ohlcvs_minutes = 0.0

    def __init__(self, client):
        self.client = client

    def get_binance_archive_client(self):
        return self.client

    def update_timestamp_range(self, start_ts, end_ts):
        self.start_ts = int(start_ts)
        self.end_ts = int(end_ts)


def make_store(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "ohlcvs" / "catalog.sqlite")
    return catalog, OhlcvStore(tmp_path / "ohlcvs", catalog)


@pytest.mark.asyncio
async def test_monthly_archive_wins_and_does_not_overwrite_valid_rows(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start = ts("2026-04-01")
    end = ts("2026-04-09")
    timestamps = np.arange(start, end + MINUTE_MS, MINUTE_MS, dtype=np.int64)
    existing_ts = timestamps[123]
    write_rows(store, [existing_ts], close=777.0)

    def result_factory(request):
        return BinanceArchiveResult(request, "ok", archive_frame(timestamps, close=100.0))

    client = RecordingArchiveClient(result_factory)
    manager = ArchiveManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))

    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    written = await _fetch_binance_archives_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange=EXCHANGE,
        coin="BTC",
        symbol=SYMBOL,
        timestamps=before.timestamps,
        valid=before.valid,
    )

    assert len(client.batches) == 1
    assert [request.kind for request in client.batches[0]] == ["monthly"]
    assert written == len(timestamps) - 1
    after = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    assert after.valid.all()
    assert after.values[123, 2] == pytest.approx(777.0)
    attempts = catalog.list_fetch_attempts(EXCHANGE, "1m", SYMBOL, start, end)
    assert [attempt.outcome for attempt in attempts] == ["archive_monthly_ok"]
    assert "source=binance_monthly_archive" in attempts[0].note


@pytest.mark.asyncio
async def test_unavailable_monthly_falls_back_to_parallel_daily_archives(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start = ts("2026-04-01")
    end = ts("2026-04-09T23:59:00")
    timestamps = np.arange(start, end + MINUTE_MS, MINUTE_MS, dtype=np.int64)

    def result_factory(request):
        if request.kind == "monthly":
            return BinanceArchiveResult(request, "not_found")
        day_ts = np.arange(request.start_ts, request.end_ts + MINUTE_MS, MINUTE_MS)
        return BinanceArchiveResult(request, "ok", archive_frame(day_ts))

    client = RecordingArchiveClient(result_factory)
    manager = ArchiveManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)

    written = await _fetch_binance_archives_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange=EXCHANGE,
        coin="BTC",
        symbol=SYMBOL,
        timestamps=before.timestamps,
        valid=before.valid,
    )

    assert [[request.kind for request in batch] for batch in client.batches] == [
        ["monthly"],
        ["daily"] * 9,
    ]
    assert written == len(timestamps)
    assert store.read_range(EXCHANGE, "1m", SYMBOL, start, end).valid.all()


@pytest.mark.asyncio
async def test_successful_monthly_real_gap_goes_directly_to_ccxt(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start = ts("2026-04-01")
    end = ts("2026-04-09")
    timestamps = np.arange(start, end + MINUTE_MS, MINUTE_MS, dtype=np.int64)
    gap_ts = timestamps[5000]

    def result_factory(request):
        archive_ts = timestamps[timestamps != gap_ts]
        return BinanceArchiveResult(request, "ok", archive_frame(archive_ts))

    class RepairManager(ArchiveManager):
        def __init__(self, client):
            super().__init__(client)
            self.ccxt_ranges = []

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            self.ccxt_ranges.append((start_ts, end_ts))
            requested = np.arange(start_ts, end_ts + MINUTE_MS, MINUTE_MS)
            return archive_frame(requested)

    client = RecordingArchiveClient(result_factory)
    manager = RepairManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))

    result = await _fetch_coin_range_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange=EXCHANGE,
        coin="BTC",
        symbol=SYMBOL,
        start_ts=start,
        end_ts=end,
    )

    assert result.ok
    assert store.read_range(EXCHANGE, "1m", SYMBOL, start, end).valid.all()
    assert manager.ccxt_ranges == [(gap_ts, gap_ts)]
    assert len(client.batches) == 1
    assert client.batches[0][0].kind == "monthly"


@pytest.mark.asyncio
async def test_small_old_gap_and_recent_tail_skip_archives_and_use_ccxt(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16T12:00:00"))

    class CcxtOnlyManager(ArchiveManager):
        def __init__(self, client):
            super().__init__(client)
            self.ccxt_ranges = []

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            self.ccxt_ranges.append((start_ts, end_ts))
            requested = np.arange(start_ts, end_ts + MINUTE_MS, MINUTE_MS)
            return archive_frame(requested)

    client = RecordingArchiveClient(lambda request: pytest.fail("archive must not be fetched"))
    manager = CcxtOnlyManager(client)
    ranges = [
        (ts("2026-04-10"), ts("2026-04-10T00:09:00")),
        (ts("2026-07-14"), ts("2026-07-16T12:00:00")),
    ]
    for start, end in ranges:
        result = await _fetch_coin_range_into_v2_store(
            om=manager,
            catalog=catalog,
            store=store,
            exchange=EXCHANGE,
            coin="BTC",
            symbol=SYMBOL,
            start_ts=start,
            end_ts=end,
        )
        assert result.ok

    assert client.batches == []
    assert manager.ccxt_ranges == ranges


@pytest.mark.asyncio
async def test_daily_archives_are_written_once_per_month(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start = ts("2026-04-01")
    end = ts("2026-04-02T23:59:00")
    timestamps = np.arange(start, end + MINUTE_MS, MINUTE_MS, dtype=np.int64)

    def result_factory(request):
        requested = np.arange(request.start_ts, request.end_ts + MINUTE_MS, MINUTE_MS)
        return BinanceArchiveResult(request, "ok", archive_frame(requested))

    client = RecordingArchiveClient(result_factory)
    manager = ArchiveManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    original_write_rows = store.write_rows
    calls = []

    def recording_write_rows(*args, **kwargs):
        calls.append(len(args[3]))
        return original_write_rows(*args, **kwargs)

    monkeypatch.setattr(store, "write_rows", recording_write_rows)
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    written = await _fetch_binance_archives_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange=EXCHANGE,
        coin="BTC",
        symbol=SYMBOL,
        timestamps=before.timestamps,
        valid=before.valid,
    )

    assert written == 2880
    assert calls == [2880]
    assert len(client.batches) == 1
    assert len(client.batches[0]) == 2


@pytest.mark.asyncio
async def test_archive_errors_are_logged_and_ccxt_repairs_range(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start = ts("2026-04-01")
    end = ts("2026-04-09")

    def result_factory(request):
        return BinanceArchiveResult(request, "error", error="checksum failed")

    class RepairManager(ArchiveManager):
        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            requested = np.arange(start_ts, end_ts + MINUTE_MS, MINUTE_MS)
            return archive_frame(requested)

    manager = RepairManager(RecordingArchiveClient(result_factory))
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))

    result = await _fetch_coin_range_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange=EXCHANGE,
        coin="BTC",
        symbol=SYMBOL,
        start_ts=start,
        end_ts=end,
    )

    assert result == V2FetchResult(
        True,
        "ok",
        wrote_rows=((end - start) // MINUTE_MS) + 1,
        first_ts=start,
        last_ts=end,
        max_internal_missing_bars=0,
    )
    attempts = catalog.list_fetch_attempts(EXCHANGE, "1m", SYMBOL, start, end)
    outcomes = [attempt.outcome for attempt in attempts]
    assert "archive_monthly_error" in outcomes
    assert "archive_daily_error" in outcomes
    monthly_attempt = next(
        attempt for attempt in attempts if attempt.outcome == "archive_monthly_error"
    )
    assert "checksum failed" in monthly_attempt.note


@pytest.mark.asyncio
async def test_confirmed_source_boundary_skips_older_archive_periods(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start, boundary, end = ts("2024-01-01"), ts("2026-04-01"), ts("2026-04-15")
    write_rows(store, np.arange(ts("2026-04-10"), end + MINUTE_MS, MINUTE_MS))
    client = RecordingArchiveClient(lambda request: BinanceArchiveResult(request, "not_found"))

    from types import SimpleNamespace
    class SourceManager(ArchiveManager):
        exchange = "binanceusdm"
        cm = SimpleNamespace(_get_authoritative_start_ts=lambda symbol: boundary)
        markets = {SYMBOL: {"id": "BTCUSDT", "symbol": SYMBOL,
            "info": {"symbol": "BTCUSDT", "onboardDate": boundary}}}
        def get_symbol(self, coin):
            assert coin == "BTC"
            return SYMBOL
        def load_first_timestamp(self, coin):
            return boundary

    manager = SourceManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    await _fetch_binance_archives_into_v2_store(
        om=manager, catalog=catalog, store=store, exchange=EXCHANGE, coin="BTC", symbol=SYMBOL,
        timestamps=before.timestamps, valid=before.valid,
    )
    requests = [request for batch in client.batches for request in batch]
    assert requests
    assert all(request.start_ts >= boundary for request in requests)
    after = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    np.testing.assert_array_equal(after.valid, before.valid)
    assert catalog.get_persistent_gaps(EXCHANGE, "1m", SYMBOL, start, end) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,error", [("not_found", None), ("error", "HTTP 451"), ("error", "timeout")])
async def test_unknown_leading_prefix_has_finite_probes_without_inventing_coverage(tmp_path, monkeypatch, caplog, outcome, error):
    from hlcv_preparation import _BINANCE_UNKNOWN_PREFIX_ARCHIVE_PROBES
    catalog, store = make_store(tmp_path)
    start, first, end = ts("2024-01-01"), ts("2026-04-10"), ts("2026-04-15")
    write_rows(store, np.arange(first, end + MINUTE_MS, MINUTE_MS))
    client = RecordingArchiveClient(lambda request: BinanceArchiveResult(request, outcome, error=error))
    manager = ArchiveManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    caplog.set_level("INFO")
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    for _ in range(2):
        await _fetch_binance_archives_into_v2_store(
            om=manager, catalog=catalog, store=store, exchange=EXCHANGE, coin="BTC", symbol=SYMBOL,
            timestamps=before.timestamps, valid=before.valid,
        )
    prefix_requests = [request for batch in client.batches for request in batch if request.end_ts < first]
    assert len(prefix_requests) == _BINANCE_UNKNOWN_PREFIX_ARCHIVE_PROBES
    assert "bounded archive discovery" in caplog.text
    after = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    np.testing.assert_array_equal(after.valid, before.valid)
    assert catalog.get_persistent_gaps(EXCHANGE, "1m", SYMBOL, start, end) == []
    attempts = catalog.list_fetch_attempts(EXCHANGE, "1m", SYMBOL, start, end)
    assert all(item.outcome.endswith(outcome) for item in attempts)


@pytest.mark.asyncio
async def test_source_alias_mismatch_cannot_clip_archive_history(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start, end = ts("2026-04-01"), ts("2026-04-15")
    client = RecordingArchiveClient(lambda request: BinanceArchiveResult(request, "not_found"))
    class WrongSource(ArchiveManager):
        exchange = "bybit"
        def get_symbol(self, coin):
            return SYMBOL
        def load_first_timestamp(self, coin):
            return ts("2026-04-10")
    manager = WrongSource(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    await _fetch_binance_archives_into_v2_store(
        om=manager, catalog=catalog, store=store, exchange=EXCHANGE, coin="BTC", symbol=SYMBOL,
        timestamps=before.timestamps, valid=before.valid,
    )
    assert client.batches[0][0].start_ts == start


@pytest.mark.asyncio
async def test_confirmed_catalog_prefix_skips_archives_without_timestamp_lookup(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start, boundary, end = ts("2024-01-01"), ts("2026-04-01"), ts("2026-04-15")
    catalog.mark_gap(exchange=EXCHANGE, timeframe="1m", symbol=SYMBOL, start_ts=start,
                     end_ts=boundary-MINUTE_MS, reason="pre_inception", persistent=True,
                     retry_count=0, note="verified_source_history_boundary_v2")
    client = RecordingArchiveClient(lambda request: BinanceArchiveResult(request, "not_found"))
    manager = ArchiveManager(client)
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    await _fetch_binance_archives_into_v2_store(
        om=manager, catalog=catalog, store=store, exchange=EXCHANGE, coin="BTC", symbol=SYMBOL,
        timestamps=before.timestamps, valid=before.valid,
    )
    assert all(request.start_ts >= boundary for batch in client.batches for request in batch)


@pytest.mark.asyncio
async def test_coin_only_listing_timestamp_is_not_source_bound_archive_proof(tmp_path, monkeypatch):
    catalog, store = make_store(tmp_path)
    start, end = ts("2026-04-01"), ts("2026-04-15")
    client = RecordingArchiveClient(lambda request: BinanceArchiveResult(request, "not_found"))
    class CoinOnlyManager(ArchiveManager):
        exchange = "binanceusdm"
        cm = None
        def get_symbol(self, coin):
            return SYMBOL
        def load_first_timestamp(self, coin):
            return ts("2026-04-10")
    monkeypatch.setattr("hlcv_preparation.utc_ms", lambda: ts("2026-07-16"))
    before = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    await _fetch_binance_archives_into_v2_store(
        om=CoinOnlyManager(client), catalog=catalog, store=store, exchange=EXCHANGE,
        coin="BTC", symbol=SYMBOL, timestamps=before.timestamps, valid=before.valid,
    )
    assert client.batches[0][0].start_ts == start


def test_older_real_candles_cannot_be_hidden_by_later_archive_boundary(tmp_path):
    from types import SimpleNamespace
    from hlcv_preparation import _binance_archive_source_boundary
    catalog, store = make_store(tmp_path)
    start, first, listing, end = ts("2026-04-01"), ts("2026-04-05"), ts("2026-04-10"), ts("2026-04-15")
    write_rows(store, np.arange(first, end + MINUTE_MS, MINUTE_MS))
    manager = SimpleNamespace(exchange="binanceusdm", get_symbol=lambda coin: SYMBOL,
        markets={SYMBOL: {"id": "BTCUSDT", "symbol": SYMBOL,
            "info": {"symbol": "BTCUSDT", "onboardDate": listing}}})
    rng = store.read_range(EXCHANGE, "1m", SYMBOL, start, end)
    assert _binance_archive_source_boundary(manager, catalog, EXCHANGE, "BTC", SYMBOL,
        rng.timestamps, rng.valid) is None
