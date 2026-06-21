from __future__ import annotations

import json

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore, month_start_ts
from tools import crash_finder


def _write_hour(store: OhlcvStore, exchange: str, symbol: str, hour_ts: int, lows: list[float]) -> None:
    timestamps = np.array([hour_ts + idx * 60_000 for idx in range(len(lows))], dtype=np.int64)
    values = []
    for idx, low in enumerate(lows):
        high = 100.0 + idx
        close = max(float(low), 1.0)
        values.append([high, float(low), close, 1.0])
    store.write_rows(exchange, "1m", symbol, timestamps, np.asarray(values, dtype=np.float32))


def test_crash_finder_identifies_clusters_and_writes_outputs(tmp_path, capsys):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    base = month_start_ts(2025, 4) + 12 * 24 * 60 * 60_000
    market_crash = base + 10 * HOUR_MS
    isolated_crash = base + 3 * 24 * HOUR_MS

    _write_hour(store, "binance", "BTC/USDT:USDT", market_crash, [99, 97, 90, 80, 70])
    _write_hour(store, "binance", "ETH/USDT:USDT", market_crash, [99, 96, 88, 77, 66])
    _write_hour(store, "binance", "LTC/USDT:USDT", market_crash, [99, 95, 86, 76, 65])
    _write_hour(store, "binance", "OM/USDT:USDT", isolated_crash, [99, 70, 40, 22, 18])
    _write_hour(store, "binance", "XRP/USDT:USDT", base, [99, 98, 97, 96, 95])

    out_dir = tmp_path / "crashes"
    assert (
        crash_finder.main(
            [
                "--root",
                str(root),
                "--exchange",
                "binance",
                "--threshold",
                "-0.20",
                "--min-valid-minutes",
                "2",
                "--top-clusters",
                "10",
                "--output-dir",
                str(out_dir),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbols_scanned"] == 5
    assert payload["events_selected"] >= 4
    assert payload["clusters_selected"] == 2
    assert any(cluster["market_wide"] for cluster in payload["clusters"])
    assert any(cluster["affected_coins"] == ["OM"] for cluster in payload["clusters"])
    assert len(payload["scanned_ranges"]) == 5
    for item in payload["scanned_ranges"]:
        assert item["first_iso"]
        assert item["last_iso"]
        assert item["valid_rows"] > 0

    assert (out_dir / "crash_events.csv").exists()
    assert (out_dir / "crash_clusters.csv").exists()
    assert (out_dir / "scanned_ranges.csv").exists()
    assert (out_dir / "scan_errors.csv").exists()
    suite_payload = json.loads((out_dir / "crash_scenarios.hjson").read_text(encoding="utf-8"))
    scenarios = suite_payload["backtest"]["scenarios"]
    assert len(scenarios) == 2
    assert all("start_date" in scenario and "end_date" in scenario for scenario in scenarios)


def test_ordered_metric_does_not_treat_low_before_high_as_crash(tmp_path):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    hour_ts = month_start_ts(2025, 5) + 5 * HOUR_MS
    timestamps = np.array([hour_ts + idx * 60_000 for idx in range(4)], dtype=np.int64)
    values = np.asarray(
        [
            [20.0, 20.0, 20.0, 1.0],
            [30.0, 30.0, 30.0, 1.0],
            [60.0, 60.0, 60.0, 1.0],
            [100.0, 100.0, 100.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "PUMP/USDT:USDT", timestamps, values)

    payload = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--root",
                str(root),
                "--threshold",
                "-0.50",
                "--min-valid-minutes",
                "2",
                "--rank-metric",
                "ordered",
            ]
        )
    )

    assert payload["events_selected"] == 0


def test_crash_finder_skips_symbol_with_missing_checksum(tmp_path):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    base = month_start_ts(2025, 6) + 4 * HOUR_MS
    _write_hour(store, "binance", "BTC/USDT:USDT", base, [99, 97, 90, 80, 70])
    _write_hour(store, "binance", "ETH/USDT:USDT", base, [99, 96, 88, 77, 66])
    with catalog._connect() as conn:
        conn.execute(
            "UPDATE chunks SET checksum = NULL WHERE exchange = ? AND timeframe = ? AND symbol = ?",
            ("binance", "1m", "BTC/USDT:USDT"),
        )

    out_dir = tmp_path / "crashes"
    payload = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--root",
                str(root),
                "--threshold",
                "-0.20",
                "--min-valid-minutes",
                "2",
                "--output-dir",
                str(out_dir),
            ]
        )
    )

    assert payload["symbols_attempted"] == 2
    assert payload["symbols_scanned"] == 1
    assert payload["symbols_failed"] == 1
    assert payload["scan_errors"][0]["symbol"] == "BTC/USDT:USDT"
    assert payload["scan_errors"][0]["error_type"] == "ValueError"
    assert "checksum missing" in payload["scan_errors"][0]["message"]
    assert payload["events_selected"] == 1
    assert (out_dir / "scan_errors.csv").exists()


HOUR_MS = 60 * 60_000
