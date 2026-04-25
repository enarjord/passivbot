from pathlib import Path

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_legacy_import import (
    import_legacy_range_into_store,
    inspect_legacy_range,
    resolve_legacy_symbol_dir,
)
from ohlcv_store import OhlcvStore, month_start_ts


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


def _write_legacy_day(root: Path, exchange: str, timeframe: str, symbol: str, day: str, rows: list[tuple]):
    symbol_dir = resolve_legacy_symbol_dir(root, exchange, timeframe, symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    np.save(symbol_dir / f"{day}.npy", np.array(rows, dtype=LEGACY_DTYPE))


def _write_legacy_day_npz(
    root: Path, exchange: str, timeframe: str, symbol: str, day: str, rows: list[tuple]
):
    symbol_dir = resolve_legacy_symbol_dir(root, exchange, timeframe, symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)
    np.savez(symbol_dir / f"{day}.npz", candles=np.array(rows, dtype=LEGACY_DTYPE))


def test_import_legacy_range_into_store_imports_requested_days_only(tmp_path):
    legacy_root = tmp_path / "legacy"
    store_root = tmp_path / "new"
    catalog = OhlcvCatalog(store_root / "catalog.sqlite")
    store = OhlcvStore(store_root, catalog)

    day1 = month_start_ts(2026, 4)
    day2 = day1 + 24 * 60 * 60_000
    _write_legacy_day(
        legacy_root,
        "binance",
        "1m",
        "ETH/USDT:USDT",
        "2026-04-01",
        [
            (day1, 0.0, 101.0, 99.0, 100.0, 10.0),
            (day1 + 60_000, 0.0, 102.0, 100.0, 101.0, 11.0),
        ],
    )
    _write_legacy_day(
        legacy_root,
        "binance",
        "1m",
        "ETH/USDT:USDT",
        "2026-04-02",
        [
            (day2, 0.0, 201.0, 199.0, 200.0, 20.0),
            (day2 + 60_000, 0.0, 202.0, 200.0, 201.0, 21.0),
        ],
    )

    imported = import_legacy_range_into_store(
        store=store,
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=day1 + 60_000,
        end_ts=day2,
    )
    assert imported == 2

    out = store.read_range("binance", "1m", "ETH/USDT:USDT", day1 + 60_000, day2)
    assert out.valid.sum() == 2
    np.testing.assert_allclose(
        out.values[out.valid],
        np.array(
            [
                [102.0, 100.0, 101.0, 11.0],
                [201.0, 199.0, 200.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )


def test_import_legacy_range_into_store_returns_zero_when_symbol_missing(tmp_path):
    store_root = tmp_path / "new"
    catalog = OhlcvCatalog(store_root / "catalog.sqlite")
    store = OhlcvStore(store_root, catalog)

    imported = import_legacy_range_into_store(
        store=store,
        legacy_root=tmp_path / "legacy",
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=month_start_ts(2026, 4),
        end_ts=month_start_ts(2026, 4) + 60_000,
    )
    assert imported == 0


def test_inspect_legacy_range_reports_missing_days(tmp_path):
    legacy_root = tmp_path / "legacy"
    day1 = month_start_ts(2026, 4)
    _write_legacy_day(
        legacy_root,
        "binance",
        "1m",
        "ETH/USDT:USDT",
        "2026-04-01",
        [(day1, 0.0, 101.0, 99.0, 100.0, 10.0)],
    )

    inspection = inspect_legacy_range(
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=day1,
        end_ts=day1 + 24 * 60 * 60_000,
    )

    assert inspection.present_days == ("2026-04-01",)
    assert inspection.missing_days == ("2026-04-02",)
    assert inspection.all_days_present is False


def test_import_legacy_range_into_store_supports_npz_shards(tmp_path):
    legacy_root = tmp_path / "legacy"
    store_root = tmp_path / "new"
    catalog = OhlcvCatalog(store_root / "catalog.sqlite")
    store = OhlcvStore(store_root, catalog)

    day1 = month_start_ts(2026, 4)
    _write_legacy_day_npz(
        legacy_root,
        "binance",
        "1m",
        "ETH/USDT:USDT",
        "2026-04-01",
        [
            (day1, 100.0, 101.0, 99.0, 100.0, 10.0),
            (day1 + 60_000, 101.0, 102.0, 100.0, 101.0, 11.0),
        ],
    )

    inspection = inspect_legacy_range(
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=day1,
        end_ts=day1 + 60_000,
    )
    assert inspection.present_days == ("2026-04-01",)
    assert inspection.missing_days == ()

    imported = import_legacy_range_into_store(
        store=store,
        legacy_root=legacy_root,
        exchange="binance",
        timeframe="1m",
        symbol="ETH/USDT:USDT",
        start_ts=day1,
        end_ts=day1 + 60_000,
    )
    assert imported == 2

    out = store.read_range("binance", "1m", "ETH/USDT:USDT", day1, day1 + 60_000)
    assert out.valid.sum() == 2
    np.testing.assert_allclose(
        out.values[out.valid],
        np.array(
            [
                [101.0, 99.0, 100.0, 10.0],
                [102.0, 100.0, 101.0, 11.0],
            ],
            dtype=np.float32,
        ),
    )
