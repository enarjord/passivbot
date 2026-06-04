from __future__ import annotations

import json

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts, rows_in_month
from tools import ohlcvs_doctor


def _write_month(root, exchange="binance", symbol_dir="XMR_USDT_USDT", year=2021, month=1):
    rows = rows_in_month(year, month, "1m")
    base = root / "data" / exchange / "1m" / symbol_dir / f"{year:04d}"
    base.mkdir(parents=True)
    body = np.full((rows, 4), np.nan, dtype=np.float32)
    valid = np.zeros(rows, dtype=np.bool_)
    body[:3] = np.array(
        [
            [101.0, 99.0, 100.0, 10.0],
            [102.0, 100.0, 101.0, 11.0],
            [103.0, 101.0, 102.0, 12.0],
        ],
        dtype=np.float32,
    )
    valid[:3] = True
    np.save(base / f"{month:02d}.npy", body)
    np.save(base / f"{month:02d}.valid.npy", valid)
    return base / f"{month:02d}.npy"


def test_ohlcvs_doctor_dry_run_does_not_create_missing_catalog(tmp_path):
    root = tmp_path / "ohlcvs"
    _write_month(root)

    report = ohlcvs_doctor.run_doctor(root=root, exchanges={"binance"})

    assert report.chunks_scanned == 1
    assert not (root / "catalog.sqlite").exists()
    assert report.by_check["missing_catalog"] == 1


def test_ohlcvs_doctor_repair_catalog_from_copied_data_chunks(tmp_path):
    root = tmp_path / "ohlcvs"
    _write_month(root)

    report = ohlcvs_doctor.run_doctor(
        root=root,
        exchanges={"binance"},
        repair_catalog=True,
    )

    assert report.chunks_scanned == 1
    assert report.chunks_indexed == 1
    assert not report.by_severity.get("error")

    catalog = OhlcvCatalog(root / "catalog.sqlite")
    first_ts, last_ts = catalog.get_symbol_bounds("binance", "1m", "XMR/USDT:USDT")
    start = month_start_ts(2021, 1)
    assert first_ts == start
    assert last_ts == start + 2 * 60_000

    chunk = catalog.list_chunks("binance", "1m", "XMR/USDT:USDT", start, start)[0]
    assert chunk.body_path.startswith(str(root.resolve()))
    assert chunk.checksum

    store = OhlcvStore(root, catalog)
    rng = store.read_range("binance", "1m", "XMR/USDT:USDT", start, start + 2 * 60_000)
    assert rng.valid.tolist() == [True, True, True]
    np.testing.assert_allclose(rng.values[:, 2], np.array([100.0, 101.0, 102.0], dtype=np.float32))


def test_ohlcvs_doctor_rewrites_copied_absolute_catalog_paths(tmp_path):
    root = tmp_path / "ohlcvs"
    _write_month(root)
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    catalog.register_chunk(
        exchange="binance",
        timeframe="1m",
        symbol="XMR/USDT:USDT",
        year=2021,
        month=1,
        body_path="/old/source/caches/ohlcvs/data/binance/1m/XMR_USDT_USDT/2021/01.npy",
        valid_path="/old/source/caches/ohlcvs/data/binance/1m/XMR_USDT_USDT/2021/01.valid.npy",
        start_ts=month_start_ts(2021, 1),
        end_ts=month_start_ts(2021, 1) + rows_in_month(2021, 1, "1m") * 60_000 - 60_000,
        rows=rows_in_month(2021, 1, "1m"),
        status="open",
        checksum="stale",
    )

    report = ohlcvs_doctor.run_doctor(root=root, exchanges={"binance"}, repair_catalog=True)

    assert report.chunks_indexed == 1
    chunk = catalog.list_chunks(
        "binance", "1m", "XMR/USDT:USDT", month_start_ts(2021, 1), month_start_ts(2021, 1)
    )[0]
    assert chunk.body_path.startswith(str(root.resolve()))
    assert chunk.checksum != "stale"


def test_ohlcvs_doctor_reports_missing_valid_mask(tmp_path):
    root = tmp_path / "ohlcvs"
    rows = rows_in_month(2021, 1, "1m")
    base = root / "data" / "binance" / "1m" / "XMR_USDT_USDT" / "2021"
    base.mkdir(parents=True)
    np.save(base / "01.npy", np.zeros((rows, 4), dtype=np.float32))

    report = ohlcvs_doctor.run_doctor(root=root, exchanges={"binance"})

    assert report.by_check["missing_valid_mask"] == 1
    assert report.by_severity["error"] == 1


def test_ohlcvs_doctor_prunes_missing_catalog_rows_and_symbol_bounds(tmp_path):
    root = tmp_path / "ohlcvs"
    (root / "data" / "binance" / "1m").mkdir(parents=True)
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    start = month_start_ts(2021, 1)
    catalog.register_chunk(
        exchange="binance",
        timeframe="1m",
        symbol="XMR/USDT:USDT",
        year=2021,
        month=1,
        body_path="/old/source/missing/01.npy",
        valid_path="/old/source/missing/01.valid.npy",
        start_ts=start,
        end_ts=start + rows_in_month(2021, 1, "1m") * 60_000 - 60_000,
        rows=rows_in_month(2021, 1, "1m"),
        status="open",
        checksum="stale",
    )
    catalog.upsert_symbol_bounds("binance", "1m", "XMR/USDT:USDT", start, start + 60_000)

    report = ohlcvs_doctor.run_doctor(
        root=root,
        exchanges={"binance"},
        prune_missing_catalog=True,
    )

    assert report.catalog_rows_pruned == 1
    assert catalog.get_symbol_bounds("binance", "1m", "XMR/USDT:USDT") == (None, None)
    assert catalog.list_chunks("binance", "1m", "XMR/USDT:USDT", start, start) == []


def test_ohlcvs_doctor_json_cli(tmp_path, capsys):
    root = tmp_path / "ohlcvs"
    _write_month(root)

    assert ohlcvs_doctor.main(["--root", str(root), "--repair-catalog", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["chunks_scanned"] == 1
    assert payload["chunks_indexed"] == 1
