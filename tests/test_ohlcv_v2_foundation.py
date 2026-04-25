import numpy as np

from backtest_dataset_materializer import BacktestDatasetMaterializer, materialize_frames
from ohlcv_catalog import OhlcvCatalog
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
