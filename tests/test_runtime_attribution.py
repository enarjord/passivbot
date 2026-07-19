import json
from pathlib import Path

import pytest

from live.runtime_attribution import (
    AttributionScanLimitError,
    build_runtime_attribution_report,
)
from tools import runtime_attribution as runtime_attribution_tool


def _identity(run_id: str, started_at_ms: int, marker: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_ms": started_at_ms,
        "passivbot_version": "8.0.0",
        "python_git_commit": marker * 40,
        "python_git_dirty": False,
        "config_sha256": marker * 64,
        "rust_crate_version": "0.1.0",
        "rust_source_sha256": marker * 64,
        "rust_artifact_sha256": marker * 64,
    }


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _roots(tmp_path: Path) -> dict:
    return {
        "fill_roots": [tmp_path / "fills"],
        "runtime_roots": [tmp_path / "runtime"],
        "monitor_roots": [tmp_path / "monitor"],
        "log_roots": [tmp_path / "logs"],
    }


def test_report_separates_recorded_ingestion_from_producer_candidate(tmp_path: Path):
    roots = _roots(tmp_path)
    run_1 = _identity("run-1", 1_000, "a")
    run_2 = _identity("run-2", 2_000, "b")
    _write_json(tmp_path / "runtime" / "bybit" / "alice" / "run-1.json", run_1)
    _write_json(tmp_path / "runtime" / "bybit" / "alice" / "run-2.json", run_2)
    _write_json(
        tmp_path / "fills" / "bybit" / "alice" / "1970-01-01.json",
        [
            {
                "id": "legacy-trailing",
                "timestamp": 1_500,
                "symbol": "BTC/USDT:USDT",
                "position_side": "long",
                "side": "buy",
                "pb_order_type": "entry_trailing_normal_long",
                "client_order_id": "cid-1",
            },
            {
                "id": "recorded-grid",
                "timestamp": 2_500,
                "symbol": "ETH/USDT:USDT",
                "position_side": "short",
                "side": "sell",
                "pb_order_type": "entry_grid_normal_short",
                "client_order_id": "cid-2",
                "provenance": {
                    "schema_version": 1,
                    "attribution": "first_ingested_by_runtime",
                    "first_ingested_at_ms": 3_000,
                    "runtime": run_2,
                },
            },
        ],
    )

    report = build_runtime_attribution_report(**roots)

    assert report["summary"]["runtime_count"] == 2
    assert report["summary"]["fill_count"] == 2
    legacy, recorded = report["fills"]
    assert legacy["is_trailing"] is True
    assert legacy["first_ingestion"]["status"] == "unattributed"
    assert legacy["producer_attribution"] == {
        "status": "single_runtime_window_candidate",
        "candidate_run_ids": ["run-1"],
        "reason": "fill_timestamp_within_one_observed_runtime_window",
        "proven": False,
        "caveat": (
            "Runtime-window correlation does not prove which binary submitted the order; "
            "use client-order IDs and contemporaneous order/execution logs when available."
        ),
    }
    assert recorded["first_ingestion"]["status"] == "recorded"
    assert recorded["first_ingestion"]["run_id"] == "run-2"
    assert recorded["producer_attribution"]["candidate_run_ids"] == ["run-2"]


def test_trailing_filter_and_monitor_fill_deduplication(tmp_path: Path):
    roots = _roots(tmp_path)
    fill = {
        "id": "fill-1",
        "timestamp": 1_500,
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": "buy",
        "pb_order_type": "entry_trailing_normal_long",
        "client_order_id": "cid-1",
    }
    _write_json(tmp_path / "fills" / "bybit" / "alice" / "day.json", [fill])
    monitor_path = tmp_path / "monitor" / "bybit" / "alice" / "history" / "fills.ndjson"
    monitor_path.parent.mkdir(parents=True)
    monitor_path.write_text(
        json.dumps(
            {
                "ts": 1_500,
                "kind": "fill",
                "stream": "fills",
                "exchange": "bybit",
                "user": "alice",
                "payload": fill,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_runtime_attribution_report(**roots, trailing_only=True)

    assert report["summary"]["fill_count"] == 1
    assert report["summary"]["trailing_fill_count"] == 1
    assert {source["kind"] for source in report["fills"][0]["sources"]} == {
        "fill_cache",
        "monitor_fill_history",
    }


def test_runtime_log_prefix_merges_with_full_manifest_identity(tmp_path: Path):
    roots = _roots(tmp_path)
    identity = _identity("1234567890abcdef", 1_000, "c")
    _write_json(tmp_path / "runtime" / "bybit" / "alice" / "run.json", identity)
    log_path = tmp_path / "logs" / "alice.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "1970-01-01T00:00:01Z INFO     ║  PASSIVBOT  │  bybit:alice  │  now  ║\n"
        "1970-01-01T00:00:01Z INFO     [runtime] run=1234567890ab pb=8.0.0 "
        "python=cccccccccccc config=cccccccccccc rust_source=cccccccccccc "
        "rust_artifact=cccccccccccc\n",
        encoding="utf-8",
    )

    report = build_runtime_attribution_report(**roots)

    assert report["summary"]["runtime_count"] == 1
    runtime = report["runtimes"][0]
    assert runtime["run_id"] == identity["run_id"]
    assert {source["kind"] for source in runtime["sources"]} == {
        "runtime_log",
        "runtime_manifest",
        "legacy_startup_log",
    }


def test_monitor_runtime_event_is_accepted_without_runtime_manifest(tmp_path: Path):
    roots = _roots(tmp_path)
    identity = _identity("run-monitor", 4_000, "d")
    path = tmp_path / "monitor" / "bybit" / "alice" / "events" / "current.ndjson"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "kind": "runtime.started",
                "payload": {
                    "live_event": {
                        "exchange": "bybit",
                        "user": "alice",
                        "data": identity,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_runtime_attribution_report(**roots)

    assert report["runtimes"][0]["run_id"] == "run-monitor"
    assert report["runtimes"][0]["sources"][0]["kind"] == "monitor_event"


def test_scan_limits_fail_closed(tmp_path: Path):
    roots = _roots(tmp_path)
    _write_json(tmp_path / "fills" / "bybit" / "alice" / "day.json", [{"x": "y" * 100}])
    with pytest.raises(AttributionScanLimitError, match="max_bytes_per_file"):
        build_runtime_attribution_report(**roots, max_bytes_per_file=10)


def test_fill_limit_fails_closed_before_unbounded_report(tmp_path: Path):
    roots = _roots(tmp_path)
    _write_json(
        tmp_path / "fills" / "bybit" / "alice" / "day.json",
        [
            {"id": "fill-1", "timestamp": 1_000},
            {"id": "fill-2", "timestamp": 2_000},
        ],
    )
    with pytest.raises(AttributionScanLimitError, match="max_fills=1"):
        build_runtime_attribution_report(**roots, max_fills=1)


def test_total_byte_limit_fails_closed(tmp_path: Path):
    roots = _roots(tmp_path)
    _write_json(
        tmp_path / "fills" / "bybit" / "alice" / "day.json",
        [{"id": "fill-1", "timestamp": 1_000}],
    )
    with pytest.raises(AttributionScanLimitError, match="max_total_bytes=10"):
        build_runtime_attribution_report(
            **roots,
            max_bytes_per_file=1_000,
            max_total_bytes=10,
        )


def test_account_specific_fill_root_uses_singleton_scope_filters(tmp_path: Path):
    account_root = tmp_path / "account-cache"
    _write_json(
        account_root / "day.json",
        [{"id": "fill-1", "timestamp": 1_000, "symbol": "BTC/USDT:USDT"}],
    )
    report = build_runtime_attribution_report(
        fill_roots=[account_root],
        runtime_roots=[],
        monitor_roots=[],
        log_roots=[],
        exchanges=["bybit"],
        users=["alice"],
    )
    assert report["fills"][0]["exchange"] == "bybit"
    assert report["fills"][0]["user"] == "alice"


def test_cli_fail_on_unattributed_returns_one(tmp_path: Path, capsys):
    fill_root = tmp_path / "fills"
    _write_json(
        fill_root / "bybit" / "alice" / "day.json",
        [{"id": "legacy", "timestamp": 1_500, "symbol": "BTC/USDT:USDT"}],
    )
    empty = tmp_path / "empty"
    assert (
        runtime_attribution_tool.main(
            [
                "--fill-root",
                str(fill_root),
                "--runtime-root",
                str(empty),
                "--monitor-root",
                str(empty),
                "--log-root",
                str(empty),
                "--fail-on-unattributed",
                "--compact",
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["first_ingestion_status_counts"] == {"unattributed": 1}
