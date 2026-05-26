import numpy as np
import pytest

from backtest_dataset_materializer import BacktestDatasetMaterializer, materialize_frames
from ohlcv_catalog import OhlcvCatalog
from ohlcv_planner import plan_local_symbol_range
from ohlcv_store import OhlcvStore, month_offset, month_start_ts, rows_in_month


def test_month_offset_and_rows_for_april_2026():
    start_ts = month_start_ts(2026, 4)
    assert rows_in_month(2026, 4, "1m") == 30 * 24 * 60
    assert month_offset(start_ts, 2026, 4, "1m") == 0
    assert month_offset(start_ts + 60_000, 2026, 4, "1m") == 1
    assert rows_in_month(2026, 4, "1h") == 30 * 24


def test_store_writes_and_reads_partial_month_range(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    base = month_start_ts(2026, 4) + 12 * 24 * 60 * 60_000
    ts = np.array([base, base + 60_000, base + 2 * 60_000], dtype=np.int64)
    vals = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [102.0, 100.0, 101.0, 11.0],
            [103.0, 101.0, 102.0, 12.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "BTC/USDT", ts, vals)

    out = store.read_range("binance", "1m", "BTC/USDT", ts[0], ts[-1])
    np.testing.assert_array_equal(out.timestamps, ts)
    np.testing.assert_array_equal(out.valid, np.array([True, True, True]))
    np.testing.assert_allclose(out.values, vals)


def test_due_persistent_gap_remains_visible_to_planner(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    start = month_start_ts(2026, 4)
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
        reason="internal_gap",
        persistent=True,
        retry_count=3,
        last_attempt_at=0,
        next_retry_at=1,
        note="test_expired_retry",
    )

    all_gaps = catalog.get_gaps("binance", "1m", "ETH/USDT:USDT", int(start), int(start + 60_000))
    active_gaps = catalog.get_persistent_gaps(
        "binance", "1m", "ETH/USDT:USDT", int(start), int(start + 60_000)
    )
    plan = plan_local_symbol_range(
        catalog=catalog,
        legacy_root=None,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=int(start),
        end_ts=int(start + 60_000),
    )

    assert len(all_gaps) == 1
    assert len(active_gaps) == 1
    assert plan.blocked_by_persistent_gap


def test_store_detects_chunk_checksum_mismatch(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000], dtype=np.int64)
    vals = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "BTC/USDT", ts, vals)
    chunk = catalog.list_chunks("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))[0]
    assert chunk.checksum

    body = np.load(chunk.body_path, mmap_mode="r+")
    body[0, 0] = 999.0
    body.flush()
    del body

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.read_range("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))


def test_store_detects_same_process_mutation_after_verified_read(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000], dtype=np.int64)
    vals = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "BTC/USDT", ts, vals)
    first = store.read_range("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))
    assert first.valid.all()

    chunk = catalog.list_chunks("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))[0]
    body = np.load(chunk.body_path, mmap_mode="r+")
    body[0, 0] = 999.0
    body.flush()
    del body

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.read_range("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))


def test_store_rejects_missing_chunk_checksum_on_read(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    ts = np.array([start, start + 60_000], dtype=np.int64)
    vals = np.array([[101.0, 99.0, 100.0, 10.0], [102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "BTC/USDT", ts, vals)
    with catalog._connect() as conn:
        conn.execute("UPDATE chunks SET checksum = NULL")

    chunk_before = catalog.list_chunks("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))[0]
    assert chunk_before.checksum is None

    with pytest.raises(ValueError, match="checksum missing"):
        store.read_range("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))
    chunk_after = catalog.list_chunks("binance", "1m", "BTC/USDT", int(ts[0]), int(ts[-1]))[0]
    assert chunk_after.checksum is None


def test_open_month_patch_extends_existing_chunk_and_symbol_bounds(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4) + 13 * 24 * 60 * 60_000
    first_ts = np.array([start], dtype=np.int64)
    second_ts = np.array([start + 24 * 60 * 60_000], dtype=np.int64)
    first_vals = np.array([[10.0, 9.0, 9.5, 1.0]], dtype=np.float32)
    second_vals = np.array([[11.0, 10.0, 10.5, 2.0]], dtype=np.float32)

    store.write_rows("hyperliquid", "1m", "ETH/USDC:USDC", first_ts, first_vals)
    store.write_rows("hyperliquid", "1m", "ETH/USDC:USDC", second_ts, second_vals)

    bounds = catalog.get_symbol_bounds("hyperliquid", "1m", "ETH/USDC:USDC")
    assert bounds == (int(first_ts[0]), int(second_ts[0]))

    chunks = catalog.list_chunks(
        "hyperliquid", "1m", "ETH/USDC:USDC", int(first_ts[0]), int(second_ts[0])
    )
    assert len(chunks) == 1
    assert chunks[0].status == "open"

    out = store.read_range(
        "hyperliquid", "1m", "ETH/USDC:USDC", int(first_ts[0]), int(second_ts[0])
    )
    assert out.valid.sum() == 2
    np.testing.assert_allclose(out.values[0], first_vals[0])
    np.testing.assert_allclose(out.values[-1], second_vals[0])


def test_catalog_gap_persistence(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT",
        start_ts=1_000,
        end_ts=2_000,
        reason="exchange_outage",
        persistent=True,
        retry_count=3,
        note="verified on source",
    )
    gaps = catalog.get_gaps("binance", "1m", "BTC/USDT", 0, 5_000)
    assert len(gaps) == 1
    assert gaps[0].persistent is True
    assert gaps[0].reason == "exchange_outage"
    assert gaps[0].retry_count == 3


def test_catalog_clear_gap_range_handles_duplicate_remainders(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    base = month_start_ts(2026, 4)
    symbol = "HYPE/USDT:USDT"
    broad_start = int(base)
    broad_end = int(base + 9 * 60_000)
    clear_end = int(base + 4 * 60_000)
    remainder_start = int(base + 5 * 60_000)

    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=broad_start,
        end_ts=broad_end,
        reason="pre_inception",
        persistent=True,
        retry_count=3,
        note="broad_gap",
    )
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=remainder_start,
        end_ts=broad_end,
        reason="pre_inception",
        persistent=True,
        retry_count=1,
        note="existing_remainder",
    )

    changed = catalog.clear_gap_range(
        exchange="binance",
        timeframe="1m",
        symbol=symbol,
        start_ts=broad_start,
        end_ts=clear_end,
        reason="pre_inception",
    )

    assert changed == 1
    gaps = catalog.get_persistent_gaps("binance", "1m", symbol, broad_start, broad_end)
    assert len(gaps) == 1
    assert gaps[0].start_ts == remainder_start
    assert gaps[0].end_ts == broad_end
    assert gaps[0].reason == "pre_inception"
    assert gaps[0].retry_count == 3
    assert gaps[0].note == "existing_remainder"


def test_materializer_creates_shared_memmap_payload(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4) + 10 * 60_000
    end = start + 4 * 60_000
    ts_full = np.arange(start, end + 60_000, 60_000, dtype=np.int64)

    btc_vals = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [102.0, 100.0, 101.0, 11.0],
            [103.0, 101.0, 102.0, 12.0],
            [104.0, 102.0, 103.0, 13.0],
            [105.0, 103.0, 104.0, 14.0],
        ],
        dtype=np.float32,
    )
    eth_ts = ts_full[:3]
    eth_vals = np.array(
        [
            [201.0, 199.0, 200.0, 20.0],
            [202.0, 200.0, 201.0, 21.0],
            [203.0, 201.0, 202.0, 22.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "BTC/USDT", ts_full, btc_vals)
    store.write_rows("binance", "1m", "ETH/USDT", eth_ts, eth_vals)

    materializer = BacktestDatasetMaterializer(store, tmp_path / "caches" / "ohlcvs" / "materialized")
    handle = materializer.materialize(
        exchange="binance",
        coins=["BTC/USDT", "ETH/USDT"],
        start_ts=int(start),
        end_ts=int(end),
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0, 30_300.0, 30_400.0]),
        mss={"BTC/USDT": {}, "ETH/USDT": {}},
        run_id="test_run",
    )

    hlcvs = handle.open_hlcvs()
    timestamps = handle.open_timestamps()
    btc = handle.open_btc_usd_prices()

    assert hlcvs.shape == (5, 2, 4)
    np.testing.assert_array_equal(timestamps, ts_full)
    np.testing.assert_allclose(btc, np.array([30_000.0, 30_100.0, 30_200.0, 30_300.0, 30_400.0]))
    np.testing.assert_allclose(hlcvs[:, 0, :], btc_vals.astype(np.float64))
    np.testing.assert_allclose(hlcvs[:3, 1, :], eth_vals.astype(np.float64))
    assert np.isnan(hlcvs[3:, 1, :]).all()

    assert handle.mss["BTC/USDT"]["first_valid_index"] == 0
    assert handle.mss["BTC/USDT"]["last_valid_index"] == 4
    assert handle.mss["ETH/USDT"]["first_valid_index"] == 0
    assert handle.mss["ETH/USDT"]["last_valid_index"] == 2


def test_materializer_supports_distinct_coin_keys_and_store_symbols(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    end = start + 60_000
    ts = np.array([start, end], dtype=np.int64)
    vals = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [102.0, 100.0, 101.0, 11.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "ETH/USDT:USDT", ts, vals)

    handle = BacktestDatasetMaterializer(store, tmp_path / "caches" / "ohlcvs" / "materialized").materialize(
        exchange="binance",
        coins=["ETH"],
        symbols_by_coin={"ETH": "ETH/USDT:USDT"},
        start_ts=int(start),
        end_ts=int(end),
        btc_usd_prices=np.array([30_000.0, 30_100.0]),
        mss={"ETH": {}},
        run_id="coin_symbol_mapping",
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_allclose(hlcvs[:, 0, :], vals.astype(np.float64))
    assert handle.mss["ETH"]["first_valid_index"] == 0
    assert handle.mss["ETH"]["last_valid_index"] == 1


def test_materializer_fills_internal_sparse_gaps_without_extending_edges(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    end = start + 2 * 60_000
    ts = np.array([start, end], dtype=np.int64)
    vals = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [103.0, 101.0, 102.0, 12.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "ETH/USDT", ts, vals)

    handle = BacktestDatasetMaterializer(store, tmp_path / "caches" / "ohlcvs" / "materialized").materialize(
        exchange="binance",
        coins=["ETH/USDT"],
        start_ts=int(start),
        end_ts=int(end),
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0]),
        mss={"ETH/USDT": {}},
        run_id="sparse_fill",
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_allclose(hlcvs[0, 0, :], vals[0].astype(np.float64))
    np.testing.assert_allclose(hlcvs[1, 0, :], np.array([100.0, 100.0, 100.0, 0.0]))
    np.testing.assert_allclose(hlcvs[2, 0, :], vals[1].astype(np.float64))
    assert handle.mss["ETH/USDT"]["first_valid_index"] == 0
    assert handle.mss["ETH/USDT"]["last_valid_index"] == 2
    assert handle.mss["ETH/USDT"]["coverage_internal_gap_count"] == 1
    assert handle.mss["ETH/USDT"]["coverage_internal_gap_minutes"] == 1
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_count"] == 1
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_source"] == "previous_or_edge_close"


def test_materializer_can_fill_accepted_edge_sparse_gaps(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    end = start + 2 * 60_000
    ts = np.array([start + 60_000, end], dtype=np.int64)
    vals = np.array(
        [
            [102.0, 100.0, 101.0, 11.0],
            [103.0, 101.0, 102.0, 12.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "ETH/USDT", ts, vals)

    handle = BacktestDatasetMaterializer(store, tmp_path / "caches" / "ohlcvs" / "materialized").materialize(
        exchange="binance",
        coins=["ETH/USDT"],
        start_ts=int(start),
        end_ts=int(end),
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0]),
        mss={"ETH/USDT": {}},
        run_id="edge_sparse_fill",
        fill_edge_gaps=True,
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_allclose(hlcvs[0, 0, :], np.array([101.0, 101.0, 101.0, 0.0]))
    np.testing.assert_allclose(hlcvs[1:, 0, :], vals.astype(np.float64))
    assert handle.mss["ETH/USDT"]["first_valid_index"] == 1
    assert handle.mss["ETH/USDT"]["last_valid_index"] == 2
    assert handle.mss["ETH/USDT"]["coverage_leading_missing_minutes"] == 1
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_count"] == 1
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_source"] == "previous_or_edge_close"


def test_materializer_records_trailing_unavailable_coverage(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 4)
    end = start + 3 * 60_000
    ts = np.array([start, start + 60_000], dtype=np.int64)
    vals = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [102.0, 100.0, 101.0, 11.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "ETH/USDT", ts, vals)

    handle = BacktestDatasetMaterializer(
        store, tmp_path / "caches" / "ohlcvs" / "materialized"
    ).materialize(
        exchange="binance",
        coins=["ETH/USDT"],
        start_ts=int(start),
        end_ts=int(end),
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0, 30_300.0]),
        mss={"ETH/USDT": {}},
        run_id="trailing_unavailable",
        fill_edge_gaps=False,
    )

    assert handle.mss["ETH/USDT"]["first_valid_index"] == 0
    assert handle.mss["ETH/USDT"]["last_valid_index"] == 1
    assert handle.mss["ETH/USDT"]["coverage_trailing_missing_minutes"] == 2
    assert handle.mss["ETH/USDT"]["coverage_invalid_rows"] == 2
    assert "synthetic_gap_fill_count" not in handle.mss["ETH/USDT"]


def test_materializer_fills_large_leading_edge_gap_linearly(tmp_path):
    catalog = OhlcvCatalog(tmp_path / "caches" / "ohlcvs" / "catalog.sqlite")
    store = OhlcvStore(tmp_path / "caches" / "ohlcvs", catalog)

    start = month_start_ts(2026, 1)
    leading_gap_rows = 120_000
    valid_ts = start + leading_gap_rows * 60_000
    vals = np.array([[102.0, 100.0, 101.0, 11.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "ETH/USDT", np.array([valid_ts], dtype=np.int64), vals)

    handle = BacktestDatasetMaterializer(store, tmp_path / "caches" / "ohlcvs" / "materialized").materialize(
        exchange="binance",
        coins=["ETH/USDT"],
        start_ts=int(start),
        end_ts=int(valid_ts),
        btc_usd_prices=np.full(leading_gap_rows + 1, 30_000.0, dtype=np.float64),
        mss={"ETH/USDT": {}},
        run_id="large_edge_sparse_fill",
        fill_edge_gaps=True,
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_allclose(hlcvs[0, 0, :], np.array([101.0, 101.0, 101.0, 0.0]))
    np.testing.assert_allclose(
        hlcvs[leading_gap_rows - 1, 0, :],
        np.array([101.0, 101.0, 101.0, 0.0]),
    )
    np.testing.assert_allclose(hlcvs[leading_gap_rows, 0, :], vals[0].astype(np.float64))
    assert handle.mss["ETH/USDT"]["first_valid_index"] == leading_gap_rows
    assert handle.mss["ETH/USDT"]["last_valid_index"] == leading_gap_rows
    assert handle.mss["ETH/USDT"]["coverage_leading_missing_minutes"] == leading_gap_rows
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_count"] == leading_gap_rows
    assert handle.mss["ETH/USDT"]["synthetic_gap_fill_source"] == "previous_or_edge_close"


def test_materialize_frames_creates_shared_memmap_payload(tmp_path):
    start = month_start_ts(2026, 4)
    timestamps = np.array([start, start + 60_000, start + 120_000], dtype=np.int64)
    aligned_values_by_coin = {
        "BTC": np.array(
            [
                [101.0, 99.0, 100.0, 10.0],
                [102.0, 100.0, 101.0, 11.0],
                [103.0, 101.0, 102.0, 12.0],
            ],
            dtype=np.float64,
        ),
        "ETH": np.array(
            [
                [201.0, 199.0, 200.0, 20.0],
                [202.0, 200.0, 201.0, 21.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float64,
        ),
    }
    handle = materialize_frames(
        output_root=tmp_path / "caches" / "ohlcvs" / "materialized",
        exchange="combined",
        coins=["BTC", "ETH"],
        timestamps=timestamps,
        aligned_values_by_coin=aligned_values_by_coin,
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0]),
        mss={"BTC": {}, "ETH": {}},
        run_id="combined_frames",
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_array_equal(handle.open_timestamps(), timestamps)
    np.testing.assert_allclose(hlcvs[:, 0, :], aligned_values_by_coin["BTC"])
    np.testing.assert_allclose(hlcvs[:2, 1, :], aligned_values_by_coin["ETH"][:2])
    assert np.isnan(hlcvs[2, 1, :]).all()
    assert handle.mss["BTC"]["first_valid_index"] == 0
    assert handle.mss["BTC"]["last_valid_index"] == 2
    assert handle.mss["ETH"]["first_valid_index"] == 0
    assert handle.mss["ETH"]["last_valid_index"] == 1


def test_materialize_frames_fills_internal_sparse_gaps(tmp_path):
    start = month_start_ts(2026, 4)
    timestamps = np.array([start, start + 60_000, start + 120_000], dtype=np.int64)
    aligned_values_by_coin = {
        "ETH": np.array(
            [
                [201.0, 199.0, 200.0, 20.0],
                [np.nan, np.nan, np.nan, np.nan],
                [203.0, 201.0, 202.0, 22.0],
            ],
            dtype=np.float64,
        ),
    }

    handle = materialize_frames(
        output_root=tmp_path / "caches" / "ohlcvs" / "materialized",
        exchange="combined",
        coins=["ETH"],
        timestamps=timestamps,
        aligned_values_by_coin=aligned_values_by_coin,
        btc_usd_prices=np.array([30_000.0, 30_100.0, 30_200.0]),
        mss={"ETH": {}},
        run_id="combined_sparse_fill",
    )

    hlcvs = handle.open_hlcvs()
    np.testing.assert_allclose(hlcvs[1, 0, :], np.array([200.0, 200.0, 200.0, 0.0]))
    assert handle.mss["ETH"]["first_valid_index"] == 0
    assert handle.mss["ETH"]["last_valid_index"] == 2
    assert handle.mss["ETH"]["synthetic_gap_fill_count"] == 1
