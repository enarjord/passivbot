import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from hlcv_preparation import (
    _fetch_coin_range_into_v2_store,
    _resolve_v2_store_range,
    prepare_hlcvs,
    try_prepare_hlcvs_v2_local,
)
from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import resolve_legacy_symbol_dir
from ohlcv_store import OhlcvStore, month_end_ts, month_start_ts


LEGACY_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("o", "float32"),
        ("h", "float32"),
        ("l", "float32"),
        ("c", "float32"),
        ("bv", "float32"),
    ]
)


def _write_day(root, exchange, symbol, day, rows):
    symbol_dir = resolve_legacy_symbol_dir(root, exchange, "1m", symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    np.save(symbol_dir / f"{day}.npy", np.array(rows, dtype=LEGACY_DTYPE))


def _minimal_bot_config():
    return {
        "long": {
            "n_positions": 1,
            "total_wallet_exposure_limit": 1.0,
            "wallet_exposure_limit": 1.0,
        },
        "short": {
            "n_positions": 0,
            "total_wallet_exposure_limit": 0.0,
            "wallet_exposure_limit": 0.0,
        },
    }


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_preserves_intraday_end(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4) + 24 * 24 * 60 * 60_000 + 60_000
    end_ts = start_ts + 2 * 60_000
    seen_ranges = []

    class FakeOhlcvManager:
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)
            seen_ranges.append((self.start_ts, self.end_ts))

        async def get_ohlcvs(self, coin):
            timestamps = np.array(
                [self.start_ts + i * 60_000 for i in range(3)],
                dtype=np.int64,
            )
            return pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "high": np.array([101.0, 102.0, 103.0], dtype=np.float32),
                    "low": np.array([99.0, 100.0, 101.0], dtype=np.float32),
                    "close": np.array([100.0, 101.0, 102.0], dtype=np.float32),
                    "volume": np.array([10.0, 11.0, 12.0], dtype=np.float32),
                }
            )

    ok = await _fetch_coin_range_into_v2_store(
        om=FakeOhlcvManager(),
        catalog=catalog,
        store=store,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert ok
    assert seen_ranges == [(start_ts, end_ts)]
    rng = store.read_range("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert rng.valid.all()
    np.testing.assert_array_equal(rng.timestamps, np.array([start_ts, start_ts + 60_000, end_ts]))


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_prefers_v2_fetcher(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4)
    end_ts = start_ts + 60_000
    calls = []

    class FakeOhlcvManager:
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            calls.append((coin, int(start_ts), int(end_ts)))
            return pd.DataFrame(
                {
                    "timestamp": np.array([start_ts, end_ts], dtype=np.int64),
                    "high": np.array([101.0, 102.0], dtype=np.float32),
                    "low": np.array([99.0, 100.0], dtype=np.float32),
                    "close": np.array([100.0, 101.0], dtype=np.float32),
                    "volume": np.array([10.0, 11.0], dtype=np.float32),
                }
            )

        async def get_ohlcvs(self, coin):
            raise AssertionError("v2 fetch must not use legacy-persisting get_ohlcvs")

    ok = await _fetch_coin_range_into_v2_store(
        om=FakeOhlcvManager(),
        catalog=catalog,
        store=store,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert ok
    assert calls == [("ETH", start_ts, end_ts)]
    rng = store.read_range("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert rng.valid.all()


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_accepts_sparse_within_tolerance(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4)
    end_ts = start_ts + 2 * 60_000

    class FakeOhlcvManager:
        gap_tolerance_ohlcvs_minutes = 1.0
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": np.array([start_ts, end_ts], dtype=np.int64),
                    "high": np.array([101.0, 103.0], dtype=np.float32),
                    "low": np.array([99.0, 101.0], dtype=np.float32),
                    "close": np.array([100.0, 102.0], dtype=np.float32),
                    "volume": np.array([10.0, 12.0], dtype=np.float32),
                }
            )

    manager = FakeOhlcvManager()
    ok = await _fetch_coin_range_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert ok
    attempts = catalog.list_fetch_attempts("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert attempts[0].outcome == "sparse_ok"
    rng = store.read_range("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    np.testing.assert_array_equal(rng.valid, np.array([True, False, True]))

    resolved = await _resolve_v2_store_range(
        om=manager,
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
        allow_remote_fetch=False,
        local_hit_log_label="test local hit",
        remote_fetch_log_label="test remote fetch",
    )
    assert resolved is not None
    np.testing.assert_array_equal(resolved.valid, np.array([True, False, True]))


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_accepts_edge_sparse_within_tolerance(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4)
    end_ts = start_ts + 2 * 60_000

    class FakeOhlcvManager:
        gap_tolerance_ohlcvs_minutes = 2.0
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": np.array([start_ts + 60_000, end_ts], dtype=np.int64),
                    "high": np.array([102.0, 103.0], dtype=np.float32),
                    "low": np.array([100.0, 101.0], dtype=np.float32),
                    "close": np.array([101.0, 102.0], dtype=np.float32),
                    "volume": np.array([11.0, 12.0], dtype=np.float32),
                }
            )

    manager = FakeOhlcvManager()
    ok = await _fetch_coin_range_into_v2_store(
        om=manager,
        catalog=catalog,
        store=store,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert ok
    attempts = catalog.list_fetch_attempts("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert attempts[0].outcome == "sparse_ok"
    assert "missing_bars=1" in attempts[0].note
    rng = store.read_range("binance", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    np.testing.assert_array_equal(rng.valid, np.array([False, True, True]))

    resolved = await _resolve_v2_store_range(
        om=manager,
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
        allow_remote_fetch=False,
        local_hit_log_label="test local hit",
        remote_fetch_log_label="test remote fetch",
    )
    assert resolved is not None
    np.testing.assert_array_equal(resolved.valid, np.array([False, True, True]))


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_normalizes_real_pagination_overlap(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4)
    end_ts = start_ts + 3 * 60_000

    class FakeOhlcvManager:
        gap_tolerance_ohlcvs_minutes = 1.0
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": np.array(
                        [
                            start_ts - 60_000,
                            start_ts + 60_000,
                            start_ts,
                            start_ts + 60_000,
                            end_ts,
                            end_ts + 60_000,
                        ],
                        dtype=np.int64,
                    ),
                    "high": np.array([90.0, 102.0, 101.0, 102.0, 104.0, 999.0]),
                    "low": np.array([89.0, 100.0, 99.0, 100.0, 102.0, 998.0]),
                    "close": np.array([89.5, 101.0, 100.0, 101.0, 103.0, 998.5]),
                    "volume": np.array([9.0, 11.0, 10.0, 11.0, 13.0, 99.0]),
                }
            )

    ok = await _fetch_coin_range_into_v2_store(
        om=FakeOhlcvManager(),
        catalog=catalog,
        store=store,
        exchange="bitget",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert ok
    attempts = catalog.list_fetch_attempts("bitget", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert attempts[0].outcome == "sparse_ok"
    assert "normalized_duplicates=1" in attempts[0].note
    assert "clipped_rows=2" in attempts[0].note

    rng = store.read_range("bitget", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    np.testing.assert_array_equal(rng.valid, np.array([True, True, False, True]))
    assert float(rng.values[1, 0]) == 102.0


@pytest.mark.asyncio
async def test_fetch_coin_range_into_v2_store_rejects_conflicting_pagination_duplicates(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start_ts = month_start_ts(2026, 4)
    end_ts = start_ts + 2 * 60_000

    class FakeOhlcvManager:
        gap_tolerance_ohlcvs_minutes = 1.0
        cm = None

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": np.array(
                        [start_ts, start_ts + 60_000, start_ts + 60_000, end_ts],
                        dtype=np.int64,
                    ),
                    "high": np.array([101.0, 102.0, 202.0, 103.0]),
                    "low": np.array([99.0, 100.0, 200.0, 101.0]),
                    "close": np.array([100.0, 101.0, 201.0, 102.0]),
                    "volume": np.array([10.0, 11.0, 21.0, 12.0]),
                }
            )

    result = await _fetch_coin_range_into_v2_store(
        om=FakeOhlcvManager(),
        catalog=catalog,
        store=store,
        exchange="bitget",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=start_ts,
        end_ts=end_ts,
    )

    assert not result
    assert result.reason == "conflicting_duplicate_timestamps"
    attempts = catalog.list_fetch_attempts("bitget", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert attempts[0].outcome == "gaps"
    assert "conflicting_duplicate_timestamps" in attempts[0].note
    rng = store.read_range("bitget", "1m", "ETH/USDT:USDT", start_ts, end_ts)
    assert not rng.valid.any()


@pytest.mark.asyncio
async def test_resolve_v2_store_range_repairs_invalid_windows_from_partial_legacy(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    legacy_root = tmp_path / "caches" / "ohlcv"
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "ETH/USDT:USDT"
    start = month_start_ts(2026, 4) + 1438 * 60_000
    timestamps = np.array([start + i * 60_000 for i in range(3)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 104.0, dtype=np.float32),
            np.arange(99.0, 102.0, dtype=np.float32),
            np.arange(100.0, 103.0, dtype=np.float32),
            np.arange(10.0, 13.0, dtype=np.float32),
        ]
    )

    # V2 has bounded local data but one invalid middle minute. The requested
    # range spans into 2026-04-02, which is absent from legacy, so full-range
    # legacy inspection is intentionally partial even though the missing v2
    # minute is repairable from the 2026-04-01 legacy shard.
    store.write_rows("binance", "1m", symbol, timestamps[[0, 2]], values[[0, 2]])
    _write_day(
        legacy_root,
        "binance",
        symbol,
        "2026-04-01",
        [(int(timestamps[1]), 0.0, 203.0, 201.0, 202.0, 22.0)],
    )

    async def fail_remote_fetch(*args, **kwargs):
        raise AssertionError("legacy repair should happen before remote fetch")

    monkeypatch.setattr("hlcv_preparation._fetch_coin_range_into_v2_store", fail_remote_fetch)

    rng = await _resolve_v2_store_range(
        om=object(),
        catalog=catalog,
        store=store,
        legacy_root=legacy_root,
        exchange="binance",
        coin="ETH",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert rng is not None
    assert rng.valid.all()
    np.testing.assert_array_equal(rng.timestamps, timestamps)
    assert float(rng.values[1, 2]) == 202.0


@pytest.mark.asyncio
async def test_resolve_v2_store_range_fetches_invalid_windows_with_context(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "ETH/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(3)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 104.0, dtype=np.float32),
            np.arange(99.0, 102.0, dtype=np.float32),
            np.arange(100.0, 103.0, dtype=np.float32),
            np.arange(10.0, 13.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[0, 2]], values[[0, 2]])
    calls = []

    async def fake_remote_fetch(**kwargs):
        calls.append((kwargs["start_ts"], kwargs["end_ts"]))
        assert kwargs["start_ts"] == int(timestamps[0])
        assert kwargs["end_ts"] == int(timestamps[-1])
        kwargs["store"].write_rows(
            kwargs["exchange"],
            "1m",
            kwargs["symbol"],
            timestamps[[1]],
            values[[1]],
        )
        return True

    monkeypatch.setattr("hlcv_preparation._fetch_coin_range_into_v2_store", fake_remote_fetch)

    class StrictGapManager:
        gap_tolerance_ohlcvs_minutes = 0.0

    rng = await _resolve_v2_store_range(
        om=StrictGapManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert calls == [(int(timestamps[0]), int(timestamps[-1]))]
    assert rng is not None
    assert rng.valid.all()
    np.testing.assert_array_equal(rng.timestamps, timestamps)


@pytest.mark.asyncio
async def test_resolve_v2_store_range_ignores_persistent_gap_after_local_repair(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "ETH/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(4)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 105.0, dtype=np.float32),
            np.arange(99.0, 103.0, dtype=np.float32),
            np.arange(100.0, 104.0, dtype=np.float32),
            np.arange(10.0, 14.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[0, 1, 3]], values[[0, 1, 3]])
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[0]),
        reason="stale_repaired_gap",
        persistent=True,
    )
    calls = []

    async def fake_remote_fetch(**kwargs):
        calls.append((kwargs["start_ts"], kwargs["end_ts"]))
        assert kwargs["end_ts"] == int(timestamps[-1])
        kwargs["store"].write_rows(
            kwargs["exchange"],
            "1m",
            kwargs["symbol"],
            timestamps[[2]],
            values[[2]],
        )
        return True

    monkeypatch.setattr("hlcv_preparation._fetch_coin_range_into_v2_store", fake_remote_fetch)

    class StrictGapManager:
        gap_tolerance_ohlcvs_minutes = 0.0

    rng = await _resolve_v2_store_range(
        om=StrictGapManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert calls == [(int(timestamps[1]), int(timestamps[-1]))]
    assert rng is not None
    assert rng.valid.all()


@pytest.mark.asyncio
async def test_resolve_v2_store_range_retries_unrepaired_persistent_gap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "ETH/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(3)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 104.0, dtype=np.float32),
            np.arange(99.0, 102.0, dtype=np.float32),
            np.arange(100.0, 103.0, dtype=np.float32),
            np.arange(10.0, 13.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[0, 2]], values[[0, 2]])
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(timestamps[1]),
        end_ts=int(timestamps[1]),
        reason="no_archive",
        persistent=True,
    )

    calls = []

    async def remote_fetch_miss(*args, **kwargs):
        calls.append((kwargs["start_ts"], kwargs["end_ts"]))
        return False

    monkeypatch.setattr("hlcv_preparation._fetch_coin_range_into_v2_store", remote_fetch_miss)

    rng = await _resolve_v2_store_range(
        om=object(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert rng is None
    assert calls == [(int(timestamps[0]), int(timestamps[-1]))]


@pytest.mark.asyncio
async def test_resolve_v2_store_range_repairs_stale_pre_inception_gap(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "CC/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(5)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 106.0, dtype=np.float32),
            np.arange(99.0, 104.0, dtype=np.float32),
            np.arange(100.0, 105.0, dtype=np.float32),
            np.arange(10.0, 15.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[3, 4]], values[[3, 4]])
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[2]),
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="mirrored_from_candlestick_manager",
    )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(timestamps[0] - 30 * 24 * 60 * 60_000) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    calls = []

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 0.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(timestamps[0] - 30 * 24 * 60 * 60_000)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            calls.append((int(start_ts), int(end_ts)))
            idx = (timestamps >= int(start_ts)) & (timestamps <= int(end_ts))
            return pd.DataFrame(
                {
                    "timestamp": timestamps[idx],
                    "high": values[idx, 0],
                    "low": values[idx, 1],
                    "close": values[idx, 2],
                    "volume": values[idx, 3],
                }
            )

    rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="CC",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert calls == [(int(timestamps[0]), int(timestamps[3]))]
    assert rng is not None
    assert rng.valid.all()
    np.testing.assert_array_equal(rng.timestamps, timestamps)
    repaired_gaps = catalog.get_persistent_gaps(
        "binance", "1m", symbol, int(timestamps[0]), int(timestamps[2])
    )
    assert repaired_gaps == []


@pytest.mark.asyncio
async def test_resolve_v2_store_range_fails_loudly_on_unrepaired_stale_pre_inception_gap(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "CC/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(5)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 106.0, dtype=np.float32),
            np.arange(99.0, 104.0, dtype=np.float32),
            np.arange(100.0, 105.0, dtype=np.float32),
            np.arange(10.0, 15.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[3, 4]], values[[3, 4]])
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(timestamps[0]),
        end_ts=int(timestamps[2]),
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="mirrored_from_candlestick_manager",
    )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(timestamps[0] - 30 * 24 * 60 * 60_000) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 0.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(timestamps[0] - 30 * 24 * 60 * 60_000)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": np.array([], dtype=np.int64),
                    "high": np.array([], dtype=np.float32),
                    "low": np.array([], dtype=np.float32),
                    "close": np.array([], dtype=np.float32),
                    "volume": np.array([], dtype=np.float32),
                }
            )

    with pytest.raises(ValueError) as exc_info:
        await _resolve_v2_store_range(
            om=FakeManager(),
            catalog=catalog,
            store=store,
            legacy_root=None,
            exchange="binance",
            coin="CC",
            symbol=symbol,
            start_ts=int(timestamps[0]),
            end_ts=int(timestamps[-1]),
            allow_remote_fetch=True,
            local_hit_log_label="v2 local hit",
            remote_fetch_log_label="v2 fetching missing range",
        )

    message = str(exc_info.value)
    assert "strict v2 HLCV repair failed for stale pre-inception gap" in message
    assert "coin=CC" in message
    assert "requested=" in message
    assert "persistent_gaps=" in message
    assert "local_first_timestamp=" in message
    assert "recent_fetch_attempts=" in message
    remaining_gaps = catalog.get_persistent_gaps(
        "binance", "1m", symbol, int(timestamps[0]), int(timestamps[2])
    )
    assert len(remaining_gaps) == 1
    assert remaining_gaps[0].reason == "pre_inception"


@pytest.mark.asyncio
async def test_resolve_v2_store_range_keeps_unrepaired_stale_pre_inception_gap(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "CC/USDT:USDT"
    start = month_start_ts(2026, 4)
    timestamps = np.array([start + i * 60_000 for i in range(7)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 108.0, dtype=np.float32),
            np.arange(99.0, 106.0, dtype=np.float32),
            np.arange(100.0, 107.0, dtype=np.float32),
            np.arange(10.0, 17.0, dtype=np.float32),
        ]
    )
    store.write_rows("binance", "1m", symbol, timestamps[[0, 2, 4, 5, 6]], values[[0, 2, 4, 5, 6]])
    first_gap_start = int(timestamps[1])
    first_gap_end = int(timestamps[1])
    second_gap_start = int(timestamps[3])
    second_gap_end = int(timestamps[3])
    for gap_start, gap_end in (
        (first_gap_start, first_gap_end),
        (second_gap_start, second_gap_end),
    ):
        catalog.mark_gap(
            exchange="binance",
            timeframe="1m",
            symbol=symbol,
            start_ts=gap_start,
            end_ts=gap_end,
            reason="pre_inception",
            persistent=True,
            retry_count=3,
            note="mirrored_from_candlestick_manager",
        )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(timestamps[0] - 30 * 24 * 60 * 60_000) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 1.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(timestamps[0] - 30 * 24 * 60 * 60_000)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            if int(start_ts) <= first_gap_start <= int(end_ts) and not (
                int(start_ts) <= second_gap_start <= int(end_ts)
            ):
                idx = np.array([1], dtype=np.int64)
                return pd.DataFrame(
                    {
                        "timestamp": timestamps[idx],
                        "high": values[idx, 0],
                        "low": values[idx, 1],
                        "close": values[idx, 2],
                        "volume": values[idx, 3],
                    }
                )
            return pd.DataFrame(
                {
                    "timestamp": np.array([], dtype=np.int64),
                    "high": np.array([], dtype=np.float32),
                    "low": np.array([], dtype=np.float32),
                    "close": np.array([], dtype=np.float32),
                    "volume": np.array([], dtype=np.float32),
                }
            )

    with pytest.raises(ValueError) as exc_info:
        await _resolve_v2_store_range(
            om=FakeManager(),
            catalog=catalog,
            store=store,
            legacy_root=None,
            exchange="binance",
            coin="CC",
            symbol=symbol,
            start_ts=int(timestamps[0]),
            end_ts=int(timestamps[-1]),
            allow_remote_fetch=True,
            local_hit_log_label="v2 local hit",
            remote_fetch_log_label="v2 fetching missing range",
        )

    assert "strict v2 HLCV repair failed for stale pre-inception gap" in str(exc_info.value)
    first_remaining = catalog.get_persistent_gaps(
        "binance", "1m", symbol, first_gap_start, first_gap_end
    )
    second_remaining = catalog.get_persistent_gaps(
        "binance", "1m", symbol, second_gap_start, second_gap_end
    )
    assert first_remaining == []
    assert len(second_remaining) == 1
    assert second_remaining[0].reason == "pre_inception"
    rng = store.read_range("binance", "1m", symbol, int(timestamps[0]), int(timestamps[-1]))
    np.testing.assert_array_equal(
        rng.valid,
        np.array([True, True, True, False, True, True, True]),
    )


@pytest.mark.asyncio
async def test_resolve_v2_store_range_accepts_discovered_pre_inception_boundary(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "HYPE/USDT:USDT"
    start = month_start_ts(2026, 4)
    first = start + 630 * 60_000
    end = first + 9 * 60_000
    timestamps = np.array([first + i * 60_000 for i in range(10)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 111.0, dtype=np.float32),
            np.arange(99.0, 109.0, dtype=np.float32),
            np.arange(100.0, 110.0, dtype=np.float32),
            np.arange(10.0, 20.0, dtype=np.float32),
        ]
    )
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(start),
        end_ts=int(end),
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="mirrored_from_candlestick_manager",
    )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(start) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 120.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(start)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            idx = (timestamps >= int(start_ts)) & (timestamps <= int(end_ts))
            return pd.DataFrame(
                {
                    "timestamp": timestamps[idx],
                    "high": values[idx, 0],
                    "low": values[idx, 1],
                    "close": values[idx, 2],
                    "volume": values[idx, 3],
                }
            )

    rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="HYPE",
        symbol=symbol,
        start_ts=int(start),
        end_ts=int(end),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert rng is not None
    np.testing.assert_array_equal(rng.timestamps, timestamps)
    assert rng.valid.all()
    attempts = catalog.list_fetch_attempts("binance", "1m", symbol, int(start), int(end))
    assert attempts[-1].outcome == "pre_inception_boundary"
    assert "edge_missing_bars=630,0" in attempts[-1].note
    prefix_gaps = catalog.get_persistent_gaps("binance", "1m", symbol, int(start), int(first - 60_000))
    assert len(prefix_gaps) == 1
    assert prefix_gaps[0].reason == "pre_inception"
    assert prefix_gaps[0].note == "discovered_first_candle_during_stale_repair"
    repaired_gaps = catalog.get_persistent_gaps("binance", "1m", symbol, int(first), int(end))
    assert repaired_gaps == []


@pytest.mark.asyncio
async def test_discovered_pre_inception_boundary_replaces_overlapping_stale_gap(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "HYPE/USDT:USDT"
    requested_start = month_start_ts(2026, 4)
    stale_gap_start = requested_start - 30 * 24 * 60 * 60_000
    first = requested_start + 630 * 60_000
    end = first + 9 * 60_000
    timestamps = np.array([first + i * 60_000 for i in range(10)], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 111.0, dtype=np.float32),
            np.arange(99.0, 109.0, dtype=np.float32),
            np.arange(100.0, 110.0, dtype=np.float32),
            np.arange(10.0, 20.0, dtype=np.float32),
        ]
    )
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(stale_gap_start),
        end_ts=int(first - 60_000),
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="mirrored_from_candlestick_manager",
    )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(requested_start) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    calls = []

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 120.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(requested_start)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            calls.append((int(start_ts), int(end_ts)))
            idx = (timestamps >= int(start_ts)) & (timestamps <= int(end_ts))
            return pd.DataFrame(
                {
                    "timestamp": timestamps[idx],
                    "high": values[idx, 0],
                    "low": values[idx, 1],
                    "close": values[idx, 2],
                    "volume": values[idx, 3],
                }
            )

    first_rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="HYPE",
        symbol=symbol,
        start_ts=int(requested_start),
        end_ts=int(end),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert first_rng is not None
    np.testing.assert_array_equal(first_rng.timestamps, timestamps)
    assert len(calls) == 1
    prefix_gaps = catalog.get_persistent_gaps(
        "binance", "1m", symbol, int(requested_start), int(first - 60_000)
    )
    assert len(prefix_gaps) == 1
    assert prefix_gaps[0].start_ts == int(requested_start)
    assert prefix_gaps[0].end_ts == int(first - 60_000)
    assert prefix_gaps[0].note == "discovered_first_candle_during_stale_repair"
    prior_gaps = catalog.get_persistent_gaps(
        "binance", "1m", symbol, int(stale_gap_start), int(requested_start - 60_000)
    )
    assert len(prior_gaps) == 1
    assert prior_gaps[0].note == "mirrored_from_candlestick_manager"

    second_rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="HYPE",
        symbol=symbol,
        start_ts=int(requested_start),
        end_ts=int(end),
        allow_remote_fetch=True,
        local_hit_log_label="v2 local hit",
        remote_fetch_log_label="v2 fetching missing range",
    )

    assert second_rng is not None
    np.testing.assert_array_equal(second_rng.timestamps, timestamps)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_resolve_v2_store_range_keeps_internal_gap_failure_during_boundary_repair(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    symbol = "HYPE/USDT:USDT"
    start = month_start_ts(2026, 4)
    first = start + 630 * 60_000
    end = first + 4 * 60_000
    returned_ts = np.array([first, first + 2 * 60_000, first + 3 * 60_000, end], dtype=np.int64)
    values = np.column_stack(
        [
            np.arange(101.0, 105.0, dtype=np.float32),
            np.arange(99.0, 103.0, dtype=np.float32),
            np.arange(100.0, 104.0, dtype=np.float32),
            np.arange(10.0, 14.0, dtype=np.float32),
        ]
    )
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=int(start),
        end_ts=int(end),
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="mirrored_from_candlestick_manager",
    )

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(start) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 0.0
        cm = None

        def load_first_timestamp(self, coin):
            return int(start)

        def update_timestamp_range(self, new_start_ts, new_end_ts):
            self.start_ts = int(new_start_ts)
            self.end_ts = int(new_end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            return pd.DataFrame(
                {
                    "timestamp": returned_ts,
                    "high": values[:, 0],
                    "low": values[:, 1],
                    "close": values[:, 2],
                    "volume": values[:, 3],
                }
            )

    with pytest.raises(ValueError) as exc_info:
        await _resolve_v2_store_range(
            om=FakeManager(),
            catalog=catalog,
            store=store,
            legacy_root=None,
            exchange="binance",
            coin="HYPE",
            symbol=symbol,
            start_ts=int(start),
            end_ts=int(end),
            allow_remote_fetch=True,
            local_hit_log_label="v2 local hit",
            remote_fetch_log_label="v2 fetching missing range",
        )

    assert "strict v2 HLCV repair failed for stale pre-inception gap" in str(exc_info.value)
    attempts = catalog.list_fetch_attempts("binance", "1m", symbol, int(start), int(end))
    assert attempts[-1].outcome == "gaps"
    assert "max_missing_bars=1" in attempts[-1].note


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_uses_local_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    legacy_root = tmp_path / "caches" / "ohlcv"
    (tmp_path / "caches" / "binance").mkdir(parents=True, exist_ok=True)

    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000, start + 2 * 60_000], dtype=np.int64)
    _write_day(
        legacy_root,
        "binance",
        "ETH/USDT:USDT",
        "2026-04-01",
        [
            (int(ts[0]), 0.0, 101.0, 99.0, 100.0, 10.0),
            (int(ts[1]), 0.0, 102.0, 100.0, 101.0, 11.0),
            (int(ts[2]), 0.0, 103.0, 101.0, 102.0, 12.0),
        ],
    )
    _write_day(
        legacy_root,
        "binance",
        "BTC/USDT:USDT",
        "2026-04-01",
        [
            (int(ts[0]), 0.0, 50001.0, 49999.0, 50000.0, 100.0),
            (int(ts[1]), 0.0, 50011.0, 50009.0, 50010.0, 101.0),
            (int(ts[2]), 0.0, 50021.0, 50019.0, 50020.0, 102.0),
        ],
    )

    with open(tmp_path / "caches" / "binance" / "first_timestamps.json", "w", encoding="utf-8") as f:
        json.dump({"ETH": int(ts[0]), "BTC": int(ts[0])}, f)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
            "BTC/USDT:USDT": {
                "base": "BTC",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is not None
    mss, timestamps, hlcvs, btc_prices = prepared
    np.testing.assert_array_equal(timestamps, np.array([ts[0]], dtype=np.int64))
    assert hlcvs.shape == (1, 1, 4)
    np.testing.assert_allclose(hlcvs[:, 0, 2], np.array([100.0]))
    np.testing.assert_allclose(btc_prices, np.array([50000.0]))
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0
    assert mss["__meta__"]["btc_source_exchange"] == "binance"


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_prefers_local_v2_before_full_prepare(monkeypatch, tmp_path):
    import rust_utils

    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )

    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "hlcvs_cache_permissive": True,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)

    async def fake_try_prepare(*args, **kwargs):
        return prepared

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("full prepare_hlcvs should not be called when local v2 succeeds")

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", fake_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fail_prepare)
    monkeypatch.setattr(backtest, "save_coins_hlcvs_to_cache", lambda *args, **kwargs: None)

    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = (
        await backtest.prepare_hlcvs_mss(config, "binance")
    )

    assert coins == ["ETH"]
    assert cache_dir is None
    assert results_path.endswith("/binance/")
    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_skips_inner_v2_after_outer_v2_miss(monkeypatch, tmp_path):
    import rust_utils

    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )

    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "hlcvs_cache_permissive": True,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)
    calls = {"outer_v2": 0, "legacy_prepare": 0}

    async def fake_try_prepare(*args, **kwargs):
        calls["outer_v2"] += 1
        return None

    async def fake_prepare_hlcvs(*args, **kwargs):
        calls["legacy_prepare"] += 1
        assert kwargs.get("skip_v2_local") is True
        return prepared

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", fake_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fake_prepare_hlcvs)
    monkeypatch.setattr(backtest, "save_coins_hlcvs_to_cache", lambda *args, **kwargs: None)

    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = await backtest.prepare_hlcvs_mss(
        config, "binance"
    )

    assert calls == {"outer_v2": 1, "legacy_prepare": 1}
    assert coins == ["ETH"]
    assert cache_dir is None
    assert results_path.endswith("/binance/")
    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_strict_reraises_outer_v2_exception(monkeypatch, tmp_path):
    import rust_utils

    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "hlcvs_cache_permissive": False,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)

    async def fail_try_prepare(*args, **kwargs):
        raise RuntimeError("v2 failed with useful details")

    async def fail_prepare_hlcvs(*args, **kwargs):
        raise AssertionError("strict mode must not call legacy prepare_hlcvs")

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", fail_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fail_prepare_hlcvs)

    with pytest.raises(ValueError) as exc_info:
        await backtest.prepare_hlcvs_mss(config, "binance")

    message = str(exc_info.value)
    assert "deterministic HLCV materialization failed before legacy fallback was allowed" in message
    assert "v2 failed with useful details" in message


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_permissive_falls_back_after_outer_v2_exception(
    monkeypatch, tmp_path
):
    import rust_utils

    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )
    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "hlcvs_cache_permissive": True,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)
    calls = {"legacy_prepare": 0}

    async def fail_try_prepare(*args, **kwargs):
        raise RuntimeError("v2 failed")

    async def fake_prepare_hlcvs(*args, **kwargs):
        calls["legacy_prepare"] += 1
        assert kwargs.get("skip_v2_local") is True
        return prepared

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", fail_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fake_prepare_hlcvs)
    monkeypatch.setattr(backtest, "save_coins_hlcvs_to_cache", lambda *args, **kwargs: None)

    coins, hlcvs, mss, _results_path, cache_dir, btc_usd_prices, timestamps = (
        await backtest.prepare_hlcvs_mss(config, "binance")
    )

    assert calls == {"legacy_prepare": 1}
    assert coins == ["ETH"]
    assert cache_dir is None
    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0


@pytest.mark.asyncio
async def test_prepare_hlcvs_mss_strict_rejects_outer_v2_none(monkeypatch, tmp_path):
    import rust_utils

    config = {
        "backtest": {
            "base_dir": str(tmp_path / "results"),
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "hlcvs_cache_permissive": False,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    monkeypatch.setattr(rust_utils, "check_and_maybe_compile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rust_utils,
        "verify_loaded_runtime_extension",
        lambda *args, **kwargs: {"skipped": "test"},
    )
    sys.modules.pop("backtest", None)
    backtest = importlib.import_module("backtest")

    monkeypatch.setattr(backtest, "load_coins_hlcvs_from_cache", lambda *args, **kwargs: None)

    async def miss_try_prepare(*args, **kwargs):
        return None

    async def fail_prepare_hlcvs(*args, **kwargs):
        raise AssertionError("strict mode must not call legacy prepare_hlcvs")

    monkeypatch.setattr(backtest, "try_prepare_hlcvs_v2_local", miss_try_prepare)
    monkeypatch.setattr(backtest, "prepare_hlcvs", fail_prepare_hlcvs)

    with pytest.raises(ValueError) as exc_info:
        await backtest.prepare_hlcvs_mss(config, "binance")

    assert "deterministic HLCV materialization could not build the requested range" in str(exc_info.value)


@pytest.mark.asyncio
async def test_prepare_hlcvs_prefers_local_v2_before_legacy_prepare(monkeypatch):
    prepared = (
        {
            "ETH": {"first_valid_index": 0, "last_valid_index": 0},
            "__meta__": {"btc_source_exchange": "binance"},
        },
        np.array([month_start_ts(2026, 4)], dtype=np.int64),
        np.array([[[101.0, 99.0, 100.0, 10.0]]], dtype=np.float64),
        np.array([50_000.0], dtype=np.float64),
    )

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    async def fake_try_prepare(*args, **kwargs):
        return prepared

    async def fail_prepare_internal(*args, **kwargs):
        raise AssertionError("legacy prepare_hlcvs_internal should not run when local v2 succeeds")

    monkeypatch.setattr("hlcv_preparation.try_prepare_hlcvs_v2_local", fake_try_prepare)
    monkeypatch.setattr("hlcv_preparation.prepare_hlcvs_internal", fail_prepare_internal)

    mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs(config, "binance")

    np.testing.assert_allclose(hlcvs, prepared[2])
    np.testing.assert_allclose(btc_usd_prices, prepared[3])
    np.testing.assert_array_equal(timestamps, prepared[1])
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_fetches_missing_remote_range_into_store(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    start = month_start_ts(2026, 4)
    ts = np.array([start], dtype=np.int64)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
            "BTC/USDT:USDT": {
                "base": "BTC",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    async def fake_fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
        if coin == "ETH":
            closes = np.array([100.0], dtype=np.float64)
        elif coin == "BTC":
            closes = np.array([50_000.0], dtype=np.float64)
        else:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.array([10.0], dtype=np.float64),
            }
        )

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)
    monkeypatch.setattr(
        "hlcv_preparation.HLCVManager.fetch_ohlcvs_for_v2_store",
        fake_fetch_ohlcvs_for_v2_store,
    )

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is not None
    mss, timestamps, hlcvs, btc_prices = prepared

    np.testing.assert_array_equal(timestamps, ts)
    assert hlcvs.shape == (1, 1, 4)
    np.testing.assert_allclose(hlcvs[:, 0, 2], np.array([100.0]))
    np.testing.assert_allclose(btc_prices, np.array([50_000.0]))
    assert mss["ETH"]["first_valid_index"] == 0
    assert mss["ETH"]["last_valid_index"] == 0

    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0])
    )
    assert len(attempts) == 1
    assert attempts[0].outcome == "ok"


@pytest.mark.asyncio
async def test_try_prepare_hlcvs_v2_local_persists_persistent_cm_gap_after_empty_fetch(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    start = month_start_ts(2026, 4)
    ts = np.array([start], dtype=np.int64)

    async def fake_load_markets(exchange, verbose=False, **kwargs):
        return {
            "ETH/USDT:USDT": {
                "base": "ETH",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            },
            "BTC/USDT:USDT": {
                "base": "BTC",
                "maker": 0.0002,
                "taker": 0.00055,
                "contractSize": 1.0,
                "limits": {"cost": {"min": 0.01}, "amount": {"min": 0.001}},
                "precision": {"price": 0.1, "amount": 0.001},
            }
        }

    async def fake_first_timestamps_unified(coins, exchange=None):
        return {coin: int(ts[0]) for coin in coins}

    class FakeCM:
        def get_gap_summary(self, symbol):
            return {
                "gaps": [
                    {
                        "start_ts": int(ts[0]),
                        "end_ts": int(ts[0]),
                        "retry_count": 3,
                        "reason": "no_archive",
                        "persistent": True,
                    }
                ]
            }

    async def fake_fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
        self.cm = FakeCM()
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    monkeypatch.setattr("hlcv_preparation.load_markets", fake_load_markets)
    monkeypatch.setattr("hlcv_preparation.get_first_timestamps_unified", fake_first_timestamps_unified)
    monkeypatch.setattr(
        "hlcv_preparation.HLCVManager.fetch_ohlcvs_for_v2_store",
        fake_fetch_ohlcvs_for_v2_store,
    )

    config = {
        "backtest": {
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "gap_tolerance_ohlcvs_minutes": 120.0,
            "cm_debug_level": 0,
            "cm_progress_log_interval_seconds": 0.0,
        },
        "live": {
            "approved_coins": {"long": ["ETH"], "short": []},
            "minimum_coin_age_days": 0.0,
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0.0,
        },
        "bot": _minimal_bot_config(),
    }

    prepared = await try_prepare_hlcvs_v2_local(config, "binance")
    assert prepared is None

    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    gaps = catalog.get_persistent_gaps("binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0]))
    assert len(gaps) == 1
    assert gaps[0].reason == "no_archive"
    attempts = catalog.list_fetch_attempts(
        "binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[0])
    )
    assert len(attempts) == 1
    assert attempts[0].outcome == "empty"

    async def repaired_fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
        close = 50_000.0 if coin == "BTC" else 100.0
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": np.array([close]),
                "high": np.array([close + 1.0]),
                "low": np.array([close - 1.0]),
                "close": np.array([close]),
                "volume": np.array([10.0]),
            }
        )

    monkeypatch.setattr(
        "hlcv_preparation.HLCVManager.fetch_ohlcvs_for_v2_store",
        repaired_fetch_ohlcvs_for_v2_store,
    )
    second = await try_prepare_hlcvs_v2_local(config, "binance")
    assert second is not None


@pytest.mark.asyncio
async def test_resolve_v2_store_range_refetches_corrupt_chunk(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start = month_start_ts(2026, 4)
    month_end = month_end_ts(2026, 4, "1m")
    ts = np.array([start, start + 60_000], dtype=np.int64)
    initial = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    repair_ts = ts.copy()
    repair_close = 200.0 + np.arange(repair_ts.size, dtype=np.float32)
    repaired = np.column_stack(
        [repair_close + 1.0, repair_close - 1.0, repair_close, repair_close * 0.1]
    ).astype(np.float32)
    store.write_rows("binance", "1m", "ETH/USDT:USDT", ts, initial)
    chunk = catalog.list_chunks("binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[-1]))[0]
    body = np.load(chunk.body_path, mmap_mode="r+")
    body[0, 0] = 999.0
    body.flush()
    del body

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 120.0
        cm = None

        def update_timestamp_range(self, start_ts, end_ts):
            self.start_ts = int(start_ts)
            self.end_ts = int(end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            assert int(start_ts) == int(start)
            assert int(end_ts) == int(month_end)
            return pd.DataFrame(
                {
                    "timestamp": repair_ts,
                    "open": repaired[:, 2],
                    "high": repaired[:, 0],
                    "low": repaired[:, 1],
                    "close": repaired[:, 2],
                    "volume": repaired[:, 3],
                }
            )

    rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=int(ts[0]),
        end_ts=int(ts[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="test local hit",
        remote_fetch_log_label="test remote fetch",
    )

    assert rng is not None
    np.testing.assert_allclose(rng.values, repaired)
    gaps = catalog.get_gaps("binance", "1m", "ETH/USDT:USDT", int(start), int(month_end))
    assert any(gap.reason == "local_corrupt_chunk" for gap in gaps)


@pytest.mark.asyncio
async def test_resolve_v2_store_range_refetches_missing_checksum_chunk(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start = month_start_ts(2026, 4)
    month_end = month_end_ts(2026, 4, "1m")
    ts = np.array([start, start + 60_000], dtype=np.int64)
    initial = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    repair_close = 200.0 + np.arange(ts.size, dtype=np.float32)
    repaired = np.column_stack(
        [repair_close + 1.0, repair_close - 1.0, repair_close, repair_close * 0.1]
    ).astype(np.float32)
    store.write_rows("binance", "1m", "ETH/USDT:USDT", ts, initial)
    with catalog._connect() as conn:
        conn.execute("UPDATE chunks SET checksum = NULL")

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 120.0
        cm = None

        def update_timestamp_range(self, start_ts, end_ts):
            self.start_ts = int(start_ts)
            self.end_ts = int(end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            assert int(start_ts) == int(start)
            assert int(end_ts) == int(month_end)
            return pd.DataFrame(
                {
                    "timestamp": ts,
                    "open": repaired[:, 2],
                    "high": repaired[:, 0],
                    "low": repaired[:, 1],
                    "close": repaired[:, 2],
                    "volume": repaired[:, 3],
                }
            )

    rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=int(ts[0]),
        end_ts=int(ts[-1]),
        allow_remote_fetch=True,
        local_hit_log_label="test local hit",
        remote_fetch_log_label="test remote fetch",
    )

    assert rng is not None
    np.testing.assert_allclose(rng.values, repaired)
    chunk = catalog.list_chunks("binance", "1m", "ETH/USDT:USDT", int(ts[0]), int(ts[-1]))[0]
    assert chunk.checksum
    gaps = catalog.get_gaps("binance", "1m", "ETH/USDT:USDT", int(start), int(month_end))
    assert any(gap.reason == "local_corrupt_chunk" for gap in gaps)


@pytest.mark.asyncio
async def test_corrupt_chunk_repair_rebuilds_full_chunk_not_only_requested_slice(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)
    start = month_start_ts(2026, 4)
    month_end = month_end_ts(2026, 4, "1m")
    requested_ts = np.array([start], dtype=np.int64)
    preserved_ts = np.array([start + 60_000], dtype=np.int64)
    seed_ts = np.array([requested_ts[0], preserved_ts[0]], dtype=np.int64)
    initial = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    repair = np.array([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "ETH/USDT:USDT", seed_ts, initial)
    chunk = catalog.list_chunks("binance", "1m", "ETH/USDT:USDT", int(start), int(start))[0]
    body = np.load(chunk.body_path, mmap_mode="r+")
    body[0, 0] = 999.0
    body.flush()
    del body
    repair_calls = []

    class FakeManager:
        gap_tolerance_ohlcvs_minutes = 120.0
        cm = None

        def update_timestamp_range(self, start_ts, end_ts):
            self.start_ts = int(start_ts)
            self.end_ts = int(end_ts)

        async def fetch_ohlcvs_for_v2_store(self, coin, *, start_ts, end_ts):
            repair_calls.append((int(start_ts), int(end_ts)))
            return pd.DataFrame(
                {
                    "timestamp": seed_ts,
                    "open": repair[:, 2],
                    "high": repair[:, 0],
                    "low": repair[:, 1],
                    "close": repair[:, 2],
                    "volume": repair[:, 3],
                }
            )

    rng = await _resolve_v2_store_range(
        om=FakeManager(),
        catalog=catalog,
        store=store,
        legacy_root=None,
        exchange="binance",
        coin="ETH",
        symbol="ETH/USDT:USDT",
        start_ts=int(requested_ts[0]),
        end_ts=int(requested_ts[0]),
        allow_remote_fetch=True,
        local_hit_log_label="test local hit",
        remote_fetch_log_label="test remote fetch",
    )

    assert repair_calls == [(int(start), int(month_end))]
    assert rng is not None
    np.testing.assert_array_equal(rng.valid, np.array([True]))
    np.testing.assert_allclose(rng.values, repair[:1])
    preserved = store.read_range(
        "binance", "1m", "ETH/USDT:USDT", int(preserved_ts[0]), int(preserved_ts[0])
    )
    np.testing.assert_array_equal(preserved.valid, np.array([True]))
    np.testing.assert_allclose(preserved.values, repair[1:])
