from __future__ import annotations

import json

import numpy as np

from ohlcv_catalog import OhlcvCatalog
from ohlcv_store import OhlcvStore
from tools import inspect_ohlcvs


def test_inspect_ohlcvs_overview_json(tmp_path, capsys):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    timestamps = np.array([1_700_000_040_000, 1_700_000_100_000], dtype=np.int64)
    values = np.array([[10.0, 9.0, 9.5, 100.0], [11.0, 10.0, 10.5, 110.0]], dtype=np.float32)
    store.write_rows("binance", "1m", "BTC/USDT:USDT", timestamps, values, status="sealed")
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=1_699_999_940_000,
        end_ts=1_699_999_940_000,
        reason="pre_inception",
        persistent=True,
        retry_count=1,
        note="seeded_for_test",
    )
    catalog.record_fetch_attempt(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=1_700_000_000_000,
        end_ts=1_700_000_060_000,
        attempt=1,
        outcome="ok",
        latency_ms=12,
        note="seeded_for_test",
    )

    assert inspect_ohlcvs.main(["--root", str(root), "--exchange", "binance", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"]["symbols"] == 1
    assert payload["counts"]["chunks"] == 1
    assert payload["counts"]["persistent_gaps"] == 1
    assert len(payload["symbols"]) == 1
    assert payload["symbols"][0]["symbol"] == "BTC/USDT:USDT"
    assert payload["symbols"][0]["chunk_count"] == 1
    assert payload["symbols"][0]["persistent_gap_count"] == 1


def test_inspect_ohlcvs_symbol_details_json(tmp_path, capsys):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    timestamps = np.array(
        [1_700_000_040_000, 1_700_000_100_000, 1_700_000_160_000],
        dtype=np.int64,
    )
    values = np.array(
        [
            [10.0, 9.0, 9.5, 100.0],
            [11.0, 10.0, 10.5, 110.0],
            [12.0, 11.0, 11.5, 120.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "BTC/USDT:USDT", timestamps, values, status="open")
    catalog.mark_gap(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=1_699_999_940_000,
        end_ts=1_699_999_940_000,
        reason="pre_inception",
        persistent=True,
        retry_count=2,
        note="mirrored_from_candlestick_manager",
    )
    catalog.record_fetch_attempt(
        exchange="binance",
        timeframe="1m",
        symbol="BTC/USDT:USDT",
        start_ts=1_699_999_940_000,
        end_ts=1_700_000_120_000,
        attempt=1,
        outcome="range_mismatch",
        latency_ms=55,
        note="got later start",
    )

    assert (
        inspect_ohlcvs.main(
            [
                "--root",
                str(root),
                "--exchange",
                "binance",
                "--symbol",
                "BTC/USDT:USDT",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"] == "BTC/USDT:USDT"
    assert payload["bounds"]["first_ts"] == int(timestamps[0])
    assert payload["bounds"]["last_ts"] == int(timestamps[-1])
    assert len(payload["chunks"]) == 1
    assert payload["chunks"][0]["valid_rows"] == 3
    assert payload["chunks"][0]["first_valid_ts"] == int(timestamps[0])
    assert payload["chunks"][0]["last_valid_ts"] == int(timestamps[-1])
    assert len(payload["gaps"]) == 1
    assert payload["gaps"][0]["reason"] == "pre_inception"
    assert payload["gaps"][0]["persistent"] is True
    assert len(payload["fetch_attempts"]) == 1
    assert payload["fetch_attempts"][0]["outcome"] == "range_mismatch"
