from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

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


def test_crash_finder_defaults_to_hourly_discovery_from_minute_candles():
    args = crash_finder.build_parser().parse_args(
        ["--exchange", "binance", "--exchange", "bybit"]
    )

    assert args.source_timeframe == "1m"
    assert args.timeframe == "1h"
    assert args.direction == "down"
    assert args.pump_threshold == 0.10
    assert args.exchange == ["binance", "bybit"]
    assert crash_finder._duration_timeframe_to_ms("4h") == 4 * HOUR_MS
    assert crash_finder._duration_timeframe_to_ms("12h") == 12 * HOUR_MS


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
                "--write-filtered-suites",
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
        assert item["scan_timeframe"] == "1h"
        assert item["buckets_scanned"] > 0

    assert (out_dir / "crash_events.csv").exists()
    assert (out_dir / "crash_clusters.csv").exists()
    assert (out_dir / "scanned_ranges.csv").exists()
    assert (out_dir / "scan_errors.csv").exists()
    for suffix in ["market_wide", "coin_focused", "single_coin"]:
        assert (out_dir / f"crash_scenarios_{suffix}.hjson").exists()
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
                "--timeframe",
                "4h",
            ]
        )
    )

    assert payload["events_selected"] == 0


def test_ordered_metric_discovers_low_to_later_high_as_pump(tmp_path):
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
                "--direction",
                "both",
                "--threshold",
                "-0.50",
                "--pump-threshold",
                "0.50",
                "--min-valid-minutes",
                "2",
                "--rank-metric",
                "ordered",
                "--timeframe",
                "4h",
            ]
        )
    )

    assert payload["events_selected"] == 1
    assert payload["events"][0]["direction"] == "up"
    assert payload["events"][0]["ordered_low_to_later_high_log"] > 1.0
    assert payload["clusters"][0]["label"].startswith("pump_")


def test_adverse_force_mode_is_directional():
    timestamp = month_start_ts(2025, 5) + 5 * HOUR_MS
    common = {
        "timestamp": timestamp,
        "timestamp_iso": crash_finder._ts_to_iso(timestamp),
        "start_ts": timestamp,
        "end_ts": timestamp,
        "start_iso": crash_finder._ts_to_iso(timestamp),
        "end_iso": crash_finder._ts_to_iso(timestamp),
        "severity": -1.0,
        "event_count": 1,
        "affected_coin_count": 1,
        "affected_coins": ["TEST"],
        "exchanges": ["binance"],
        "market_wide": False,
    }
    clusters = [
        crash_finder.CrashCluster(label="crash_test", direction="down", **common),
        crash_finder.CrashCluster(label="pump_test", direction="up", **common),
    ]

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=1,
        post_days=1,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="all",
        force_normal="adverse",
        merge_overlaps=True,
        all_scanned_coins=["TEST"],
    )

    scenarios = {item["label"]: item for item in payload["backtest"]["scenarios"]}
    assert scenarios["crash_test"]["overrides"]["coin_overrides"]["TEST"]["live"] == {
        "forced_mode_long": "normal",
        "forced_mode_short": "manual",
    }
    assert scenarios["pump_test"]["overrides"]["coin_overrides"]["TEST"]["live"] == {
        "forced_mode_long": "manual",
        "forced_mode_short": "normal",
    }


def test_adverse_force_mode_isolates_market_wide_side():
    timestamp = month_start_ts(2025, 5) + 5 * HOUR_MS
    common = {
        "timestamp": timestamp,
        "timestamp_iso": crash_finder._ts_to_iso(timestamp),
        "start_ts": timestamp,
        "end_ts": timestamp,
        "start_iso": crash_finder._ts_to_iso(timestamp),
        "end_iso": crash_finder._ts_to_iso(timestamp),
        "severity": -1.0,
        "event_count": 3,
        "affected_coin_count": 3,
        "affected_coins": ["BTC", "ETH", "SOL"],
        "exchanges": ["binance"],
        "market_wide": True,
    }
    clusters = [
        crash_finder.CrashCluster(label="crash_market", direction="down", **common),
        crash_finder.CrashCluster(label="pump_market", direction="up", **common),
    ]

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=1,
        post_days=1,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="all",
        force_normal="adverse",
        merge_overlaps=True,
        all_scanned_coins=["BTC", "ETH", "SOL"],
    )

    scenarios = {item["label"]: item for item in payload["backtest"]["scenarios"]}
    assert scenarios["crash_market"]["overrides"] == {
        "bot.short.risk.total_wallet_exposure_limit": 0.0,
    }
    assert scenarios["pump_market"]["overrides"] == {
        "bot.long.risk.total_wallet_exposure_limit": 0.0,
    }


def test_larger_scan_timeframe_combines_source_minutes_across_hour_boundary(tmp_path):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    four_hour_start = month_start_ts(2025, 5) + 8 * HOUR_MS
    timestamps = np.asarray(
        [four_hour_start + 59 * 60_000, four_hour_start + 60 * 60_000],
        dtype=np.int64,
    )
    values = np.asarray(
        [
            [100.0, 100.0, 100.0, 1.0],
            [50.0, 50.0, 50.0, 1.0],
        ],
        dtype=np.float32,
    )
    store.write_rows("binance", "1m", "CROSS/USDT:USDT", timestamps, values)

    one_hour = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--root",
                str(root),
                "--timeframe",
                "1h",
                "--threshold",
                "-0.50",
                "--min-valid-minutes",
                "1",
            ]
        )
    )
    four_hours = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--root",
                str(root),
                "--timeframe",
                "4h",
                "--threshold",
                "-0.50",
                "--min-valid-minutes",
                "1",
            ]
        )
    )

    assert one_hour["events_selected"] == 0
    assert one_hour["scanned_ranges"][0]["buckets_scanned"] == 2
    assert four_hours["events_selected"] == 1
    assert four_hours["scanned_ranges"][0]["buckets_scanned"] == 1
    event = four_hours["events"][0]
    assert event["timeframe"] == "4h"
    assert event["timestamp"] == four_hour_start
    assert event["valid_minutes"] == 2
    assert event["bucket_high"] == 100.0
    assert event["bucket_low"] == 50.0


def test_full_scan_all_scanned_coin_mode_keeps_non_crashing_scanned_coins(tmp_path):
    root = tmp_path / "ohlcvs"
    catalog = OhlcvCatalog(root / "catalog.sqlite")
    store = OhlcvStore(root, catalog)
    base = month_start_ts(2025, 6) + 4 * HOUR_MS
    _write_hour(store, "binance", "BTC/USDT:USDT", base, [99, 97, 90, 80, 70])
    _write_hour(store, "binance", "XRP/USDT:USDT", base, [99, 98, 97, 96, 95])

    payload = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--root",
                str(root),
                "--threshold",
                "-0.20",
                "--min-valid-minutes",
                "2",
                "--scenario-coin-mode",
                "all-scanned",
            ]
        )
    )

    scenario = payload["suite"]["backtest"]["scenarios"][0]
    assert scenario["coins"] == ["BTC", "XRP"]


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


def test_suite_builder_merged_market_wide_scenario_suppresses_forced_normal():
    market_ts = int(datetime(2025, 10, 10, 21, tzinfo=UTC).timestamp() * 1000)
    coin_ts = int(datetime(2025, 9, 29, 17, tzinfo=UTC).timestamp() * 1000)
    clusters = [
        crash_finder.CrashCluster(
            label="crash_2025_10_10_21_market_wide",
            timestamp=market_ts,
            timestamp_iso="2025-10-10T21:00:00Z",
            start_ts=market_ts,
            end_ts=market_ts,
            start_iso="2025-10-10T21:00:00Z",
            end_iso="2025-10-10T21:00:00Z",
            severity=-1.0,
            event_count=4,
            affected_coin_count=3,
            affected_coins=["BTC", "M", "OM"],
            exchanges=["binance", "bybit"],
            market_wide=True,
        ),
        crash_finder.CrashCluster(
            label="crash_2025_09_29_17_dexe_m",
            timestamp=coin_ts,
            timestamp_iso="2025-09-29T17:00:00Z",
            start_ts=coin_ts,
            end_ts=coin_ts,
            start_iso="2025-09-29T17:00:00Z",
            end_iso="2025-09-29T17:00:00Z",
            severity=-0.5,
            event_count=2,
            affected_coin_count=2,
            affected_coins=["DEXE", "M"],
            exchanges=["binance"],
            market_wide=False,
        ),
    ]

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="all",
        force_normal="long",
        merge_overlaps=True,
        all_scanned_coins=[],
    )

    scenarios = payload["backtest"]["scenarios"]
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["label"] == "crash_2025_10_10_21_market_wide_combined_2"
    assert scenario["start_date"] == "2025-09-15"
    assert scenario["end_date"] == "2025-12-09"
    assert scenario["coins"] == ["BTC", "DEXE", "M", "OM"]
    assert "overrides" not in scenario


def test_suite_builder_can_filter_market_wide_clusters():
    base = month_start_ts(2025, 4)
    clusters = [
        crash_finder.CrashCluster(
            label="crash_market",
            timestamp=base,
            timestamp_iso="2025-04-01T00:00:00Z",
            start_ts=base,
            end_ts=base,
            start_iso="2025-04-01T00:00:00Z",
            end_iso="2025-04-01T00:00:00Z",
            severity=-0.4,
            event_count=4,
            affected_coin_count=4,
            affected_coins=["BTC", "ETH", "SOL", "XRP"],
            exchanges=["binance"],
            market_wide=True,
        ),
        crash_finder.CrashCluster(
            label="crash_om",
            timestamp=base + HOUR_MS,
            timestamp_iso="2025-04-01T01:00:00Z",
            start_ts=base + HOUR_MS,
            end_ts=base + HOUR_MS,
            start_iso="2025-04-01T01:00:00Z",
            end_iso="2025-04-01T01:00:00Z",
            severity=-0.9,
            event_count=1,
            affected_coin_count=1,
            affected_coins=["OM"],
            exchanges=["binance"],
            market_wide=False,
        ),
    ]

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="market-wide",
        force_normal="none",
        merge_overlaps=False,
        all_scanned_coins=[],
    )

    scenarios = payload["backtest"]["scenarios"]
    assert [scenario["label"] for scenario in scenarios] == ["crash_market"]


def test_suite_builder_can_filter_strict_single_coin_clusters():
    base = month_start_ts(2025, 4)
    clusters = [
        crash_finder.CrashCluster(
            label="crash_om",
            timestamp=base,
            timestamp_iso="2025-04-01T00:00:00Z",
            start_ts=base,
            end_ts=base,
            start_iso="2025-04-01T00:00:00Z",
            end_iso="2025-04-01T00:00:00Z",
            severity=-0.9,
            event_count=1,
            affected_coin_count=1,
            affected_coins=["OM"],
            exchanges=["binance"],
            market_wide=False,
        ),
        crash_finder.CrashCluster(
            label="crash_dexe_m",
            timestamp=base + HOUR_MS,
            timestamp_iso="2025-04-01T01:00:00Z",
            start_ts=base + HOUR_MS,
            end_ts=base + HOUR_MS,
            start_iso="2025-04-01T01:00:00Z",
            end_iso="2025-04-01T01:00:00Z",
            severity=-0.8,
            event_count=2,
            affected_coin_count=2,
            affected_coins=["DEXE", "M"],
            exchanges=["binance"],
            market_wide=False,
        ),
    ]

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="single-coin",
        force_normal="none",
        merge_overlaps=False,
        all_scanned_coins=[],
    )

    scenarios = payload["backtest"]["scenarios"]
    assert [scenario["label"] for scenario in scenarios] == ["crash_om"]
    assert scenarios[0]["coins"] == ["OM"]


def test_suite_builder_splits_merged_idiosyncratic_force_targets_into_pairs():
    base_ts = int(datetime(2025, 4, 13, 18, tzinfo=UTC).timestamp() * 1000)
    clusters = []
    for idx, coin in enumerate(["OM", "DEXE", "M"]):
        ts = base_ts + idx * HOUR_MS
        clusters.append(
            crash_finder.CrashCluster(
                label=f"crash_2025_04_13_{18 + idx:02d}_{coin.lower()}",
                timestamp=ts,
                timestamp_iso="2025-04-13T18:00:00Z",
                start_ts=ts,
                end_ts=ts,
                start_iso="2025-04-13T18:00:00Z",
                end_iso="2025-04-13T18:00:00Z",
                severity=-1.0 + idx * 0.1,
                event_count=1,
                affected_coin_count=1,
                affected_coins=[coin],
                exchanges=["binance"],
                market_wide=False,
            )
        )

    payload = crash_finder.build_suite_payload(
        clusters,
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="coin-focused",
        force_normal="both",
        merge_overlaps=True,
        all_scanned_coins=[],
    )

    scenarios = payload["backtest"]["scenarios"]
    assert len(scenarios) == 2
    forced_groups = [
        sorted(scenario["overrides"]["coin_overrides"])
        for scenario in scenarios
        if scenario.get("overrides")
    ]
    assert forced_groups == [["DEXE", "M"], ["OM"]]
    assert all(len(group) <= 2 for group in forced_groups)
    assert all(scenario["coins"] == ["DEXE", "M", "OM"] for scenario in scenarios)


def test_suite_builder_filters_coins_without_data_in_scenario_window():
    event_ts = int(datetime(2026, 6, 25, tzinfo=UTC).timestamp() * 1000)
    cluster = crash_finder.CrashCluster(
        label="crash_2026_06_25_00_m_legacy",
        timestamp=event_ts,
        timestamp_iso="2026-06-25T00:00:00Z",
        start_ts=event_ts,
        end_ts=event_ts,
        start_iso="2026-06-25T00:00:00Z",
        end_iso="2026-06-25T00:00:00Z",
        severity=-1.0,
        event_count=2,
        affected_coin_count=2,
        affected_coins=["LEGACY", "M"],
        exchanges=["binance"],
        market_wide=False,
    )

    payload = crash_finder.build_suite_payload(
        [cluster],
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="coin-focused",
        force_normal="both",
        merge_overlaps=False,
        all_scanned_coins=[],
        coin_data_ranges=[
            crash_finder.CoinDataRange(
                exchange="binance",
                coin="LEGACY",
                first_ts=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000),
                last_ts=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp() * 1000),
            ),
            crash_finder.CoinDataRange(
                exchange="binance",
                coin="M",
                first_ts=int(datetime(2026, 6, 1, tzinfo=UTC).timestamp() * 1000),
                last_ts=int(datetime(2026, 6, 29, tzinfo=UTC).timestamp() * 1000),
            ),
        ],
    )

    scenario = payload["backtest"]["scenarios"][0]
    assert scenario["coins"] == ["M"]
    assert sorted(scenario["overrides"]["coin_overrides"]) == ["M"]


def test_suite_builder_skips_targeted_scenario_when_all_coins_are_filtered(caplog):
    event_ts = int(datetime(2026, 6, 25, tzinfo=UTC).timestamp() * 1000)
    cluster = crash_finder.CrashCluster(
        label="crash_2026_06_25_00_legacy",
        timestamp=event_ts,
        timestamp_iso="2026-06-25T00:00:00Z",
        start_ts=event_ts,
        end_ts=event_ts,
        start_iso="2026-06-25T00:00:00Z",
        end_iso="2026-06-25T00:00:00Z",
        severity=-1.0,
        event_count=1,
        affected_coin_count=1,
        affected_coins=["LEGACY"],
        exchanges=["binance"],
        market_wide=False,
    )

    with caplog.at_level(logging.WARNING):
        payload = crash_finder.build_suite_payload(
            [cluster],
            pre_days=14,
            post_days=60,
            top_clusters=0,
            coin_mode="affected",
            scenario_kind="coin-focused",
            force_normal="both",
            merge_overlaps=False,
            all_scanned_coins=[],
            coin_data_ranges=[
                crash_finder.CoinDataRange(
                    exchange="binance",
                    coin="LEGACY",
                    first_ts=int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1000),
                    last_ts=int(datetime(2025, 2, 1, tzinfo=UTC).timestamp() * 1000),
                )
            ],
        )

    assert payload["backtest"]["scenarios"] == []
    assert "omitted targeted scenario crash_2026_06_25_00_legacy" in caplog.text


def test_suite_builder_none_coin_mode_keeps_inherit_base_scenario():
    event_ts = int(datetime(2026, 6, 25, tzinfo=UTC).timestamp() * 1000)
    cluster = crash_finder.CrashCluster(
        label="crash_2026_06_25_00_m",
        timestamp=event_ts,
        timestamp_iso="2026-06-25T00:00:00Z",
        start_ts=event_ts,
        end_ts=event_ts,
        start_iso="2026-06-25T00:00:00Z",
        end_iso="2026-06-25T00:00:00Z",
        severity=-1.0,
        event_count=1,
        affected_coin_count=1,
        affected_coins=["M"],
        exchanges=["binance"],
        market_wide=False,
    )

    payload = crash_finder.build_suite_payload(
        [cluster],
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="none",
        scenario_kind="coin-focused",
        force_normal="both",
        merge_overlaps=False,
        all_scanned_coins=[],
        coin_data_ranges=[],
    )

    scenario = payload["backtest"]["scenarios"][0]
    assert "coins" not in scenario
    assert "overrides" not in scenario


def test_suite_builder_includes_coin_with_data_exactly_on_end_date():
    event_ts = int(datetime(2026, 6, 25, tzinfo=UTC).timestamp() * 1000)
    end_ts = int(datetime(2026, 8, 24, tzinfo=UTC).timestamp() * 1000)
    cluster = crash_finder.CrashCluster(
        label="crash_2026_06_25_00_end",
        timestamp=event_ts,
        timestamp_iso="2026-06-25T00:00:00Z",
        start_ts=event_ts,
        end_ts=event_ts,
        start_iso="2026-06-25T00:00:00Z",
        end_iso="2026-06-25T00:00:00Z",
        severity=-1.0,
        event_count=1,
        affected_coin_count=1,
        affected_coins=["END"],
        exchanges=["binance"],
        market_wide=False,
    )

    payload = crash_finder.build_suite_payload(
        [cluster],
        pre_days=14,
        post_days=60,
        top_clusters=0,
        coin_mode="affected",
        scenario_kind="coin-focused",
        force_normal="none",
        merge_overlaps=False,
        all_scanned_coins=[],
        coin_data_ranges=[
            crash_finder.CoinDataRange(
                exchange="binance",
                coin="END",
                first_ts=end_ts,
                last_ts=end_ts,
            )
        ],
    )

    assert payload["backtest"]["scenarios"][0]["coins"] == ["END"]


def test_clusters_csv_fast_path_copies_source_scan_artifacts(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    clusters_csv = source_dir / "crash_clusters.csv"
    clusters_csv.write_text(
        "\n".join(
            [
                "label,timestamp,timestamp_iso,start_ts,end_ts,start_iso,end_iso,severity,event_count,affected_coin_count,affected_coins,exchanges,market_wide",
                "crash_2025_04_13_18_om,1744567200000,2025-04-13T18:00:00Z,1744567200000,1744567200000,2025-04-13T18:00:00Z,2025-04-13T18:00:00Z,-1.0,1,1,OM,binance,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ["crash_events.csv", "scan_errors.csv"]:
        (source_dir / name).write_text(f"source {name}\n", encoding="utf-8")
    (source_dir / "scanned_ranges.csv").write_text(
        "\n".join(
            [
                "exchange,timeframe,symbol,first_ts,last_ts,first_iso,last_iso,valid_rows,hours_scanned,events_found",
                "binance,1m,OM/USDT:USDT,1740000000000,1750000000000,2025-02-19T21:20:00Z,2025-06-15T15:06:40Z,1,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    payload = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            [
                "--clusters-csv",
                str(clusters_csv),
                "--pre-days",
                "14",
                "--post-days",
                "60",
                "--scenario-force-normal",
                "both",
                "--output-dir",
                str(out_dir),
            ]
        )
    )

    assert payload["clusters_loaded"] == 1
    assert (out_dir / "crash_clusters.csv").exists()
    assert (out_dir / "crash_scenarios.hjson").exists()
    for name in ["crash_events.csv", "scan_errors.csv"]:
        assert (out_dir / name).read_text(encoding="utf-8") == f"source {name}\n"
    assert (out_dir / "scanned_ranges.csv").read_text(encoding="utf-8").startswith(
        "exchange,timeframe,symbol"
    )
    suite_payload = json.loads((out_dir / "crash_scenarios.hjson").read_text(encoding="utf-8"))
    scenario = suite_payload["backtest"]["scenarios"][0]
    assert scenario["start_date"] == "2025-03-30"
    assert scenario["end_date"] == "2025-06-12"
    assert scenario["overrides"]["coin_overrides"]["OM"]["live"] == {
        "forced_mode_long": "normal",
        "forced_mode_short": "normal",
    }


def test_clusters_csv_fast_path_does_not_create_empty_scan_artifacts(tmp_path):
    clusters_csv = tmp_path / "crash_clusters.csv"
    clusters_csv.write_text(
        "\n".join(
            [
                "label,timestamp,timestamp_iso,start_ts,end_ts,start_iso,end_iso,severity,event_count,affected_coin_count,affected_coins,exchanges,market_wide",
                "crash_2025_04_13_18_om,1744567200000,2025-04-13T18:00:00Z,1744567200000,1744567200000,2025-04-13T18:00:00Z,2025-04-13T18:00:00Z,-1.0,1,1,OM,binance,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    payload = crash_finder.run_scan(
        crash_finder.build_parser().parse_args(
            ["--clusters-csv", str(clusters_csv), "--output-dir", str(out_dir)]
        )
    )

    assert "events_csv" not in (payload["output_paths"] or {})
    assert not (out_dir / "crash_events.csv").exists()
    assert not (out_dir / "scanned_ranges.csv").exists()
    assert not (out_dir / "scan_errors.csv").exists()


def test_logging_configuration_honors_requested_level_with_existing_handlers():
    logging.basicConfig(level=logging.INFO, force=True)

    crash_finder._configure_logging("warning")

    assert logging.getLogger().level == logging.WARNING


HOUR_MS = 60 * 60_000
