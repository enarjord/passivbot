from __future__ import annotations

import json
import os

import pytest

import live.event_file_rows as event_file_rows_module
import live.smoke_report as smoke_report_module
from live.smoke_report import (
    build_live_smoke_report,
    default_logs_root_for_monitor,
    project_live_smoke_report_sections,
    summarize_live_smoke_report,
    summarize_live_smoke_report_brief,
)
from tools import live_smoke_report


def _monitor_row(
    *,
    event_type: str,
    seq: int,
    ts: int,
    exchange: str = "binance",
    user: str = "binance_01",
    status: str = "succeeded",
    level: str = "info",
    reason_code: str = "test",
    symbol: str | None = None,
    pside: str | None = None,
    side: str | None = None,
    ids: dict | None = None,
    data: dict | None = None,
    message: str | None = None,
) -> dict:
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": level,
        "source": "live",
        "component": "test",
        "exchange": exchange,
        "user": user,
        "symbol": symbol,
        "pside": pside,
        "side": side,
        "status": status,
        "reason_code": reason_code,
        "data": dict(data or {"seq": seq}),
        "ids": dict(ids or {}),
    }
    if message is not None:
        live_event["message"] = message
    return {
        "exchange": exchange,
        "user": user,
        "kind": event_type,
        "tags": ["test"],
        "payload": {"_live_event": live_event},
        "seq": seq,
        "ts": ts,
    }


def _write_ndjson(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _set_mtime(path, seconds: int):
    os.utime(path, (seconds, seconds))


def _write_minimal_monitor_event(monitor_root):
    _write_ndjson(
        monitor_root / "binance" / "binance_01" / "events" / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )


def test_live_smoke_report_scans_monitor_segments_once(tmp_path, monkeypatch):
    events_path = (
        tmp_path / "monitor" / "binance" / "binance_01" / "events" / "current.ndjson"
    )
    rotated_path = (
        tmp_path / "monitor" / "binance" / "binance_01" / "events" / "20260629.ndjson.gz"
    )
    _write_ndjson(
        events_path,
        [
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_1"},
            ),
        ],
    )
    rotated_path.write_bytes(b"")
    opened = []
    original_open_text = event_file_rows_module._open_text

    def spy_open_text(path):
        opened.append(path)
        return original_open_text(path)

    monkeypatch.setattr(event_file_rows_module, "_open_text", spy_open_text)

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        since_ms=1500,
    )

    assert report["ok"] is True
    assert opened == [events_path]
    assert report["monitor"]["live_events"] == 2
    assert report["monitor"]["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 1,
        "scope_pruned": 0,
    }
    assert report["event_window"]["events_considered"] == 1


def test_live_smoke_report_event_tail_lines_bounds_monitor_scan(tmp_path):
    events_path = (
        tmp_path / "monitor" / "binance" / "binance_01" / "events" / "current.ndjson"
    )
    _write_ndjson(
        events_path,
        [
            _monitor_row(
                event_type="bot.stopped",
                seq=1,
                ts=1000,
                status="failed",
                level="critical",
                reason_code="old_failure_outside_tail",
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_old"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=3000,
                ids={"cycle_id": "cy_old"},
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=4,
                ts=4000,
                ids={"cycle_id": "cy_fresh"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=5,
                ts=5000,
                ids={"cycle_id": "cy_fresh"},
            ),
        ],
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        since_ms=3500,
        event_tail_lines=2,
    )

    assert report["ok"] is True
    assert report["monitor"]["records_total"] == 2
    assert report["monitor"]["live_events"] == 2
    assert report["hard_problem_event_count"] == 0
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 3500,
        "until_ms": None,
        "events_considered": 2,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "event_tail_lines": 2,
        "event_tail_limited_files": 1,
        "event_tail_skipped_lines": 3,
        "event_tail_skipped_lines_exact": True,
        "event_tail_skipped_bytes": 0,
        "event_tail_line_numbers_exact": True,
        "event_tail_methods": {"seek_tail": 1},
    }
    assert report["bots"][0]["events"] == 2


def test_live_smoke_report_max_event_files_per_bot_is_fair(tmp_path):
    binance_events = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    okx_events = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    files = [
        (
            binance_events / "current.ndjson",
            10,
            "binance",
            "binance_01",
            "binance_current",
        ),
        (
            binance_events / "new.ndjson",
            11,
            "binance",
            "binance_01",
            "binance_new",
        ),
        (
            binance_events / "old.ndjson",
            12,
            "binance",
            "binance_01",
            "binance_old",
        ),
        (okx_events / "current.ndjson", 20, "okx", "okx_01", "okx_current"),
        (okx_events / "new.ndjson", 21, "okx", "okx_01", "okx_new"),
        (okx_events / "old.ndjson", 22, "okx", "okx_01", "okx_old"),
    ]
    for path, seq, exchange, user, cycle_id in files:
        _write_ndjson(
            path,
            [
                _monitor_row(
                    event_type="cycle.completed",
                    seq=seq,
                    ts=seq * 100,
                    exchange=exchange,
                    user=user,
                    ids={"cycle_id": cycle_id},
                )
            ],
        )
    for path in (binance_events / "old.ndjson", okx_events / "old.ndjson"):
        _set_mtime(path, 100)
    for path in (binance_events / "new.ndjson", okx_events / "new.ndjson"):
        _set_mtime(path, 200)
    for path in (binance_events / "current.ndjson", okx_events / "current.ndjson"):
        _set_mtime(path, 50)

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        include_rotated=True,
        max_event_files_per_bot=2,
    )

    assert report["ok"] is True
    assert report["monitor"]["files_scanned"] == 4
    assert report["monitor"]["live_events"] == 4
    assert report["event_window"] == {
        "enabled": False,
        "since_ms": None,
        "until_ms": None,
        "events_considered": 4,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "max_event_files_per_bot": 2,
        "event_file_limit_scope": "per_bot",
        "event_file_limit_groups": 2,
        "event_files_before_limit": 6,
        "event_files_skipped_by_limit": 2,
        "event_file_limit_order": "current_then_recent_mtime",
    }
    cycle_ids = {item["cycle_id"] for item in report["monitor"]["cycle_ids_sample"]}
    assert cycle_ids == {
        "binance_current",
        "binance_new",
        "okx_current",
        "okx_new",
    }
    assert {bot["bot"]: bot["events"] for bot in report["bots"]} == {
        "binance/binance_01": 2,
        "okx/okx_01": 2,
    }

    brief = summarize_live_smoke_report_brief(report)
    assert brief["event_window"]["max_event_files_per_bot"] == 2
    assert brief["event_window"]["event_file_limit_scope"] == "per_bot"
    assert brief["event_window"]["event_files_skipped_by_limit"] == 2


def test_live_smoke_report_summarizes_monitor_events_and_log_attention(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            ),
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=1100,
                status="failed",
                level="warning",
                reason_code="request_timeout",
                symbol="BTC/USDT:USDT",
                ids={"remote_call_id": "rc_1"},
                data={
                    "surface": "balance",
                    "error_type": "RequestTimeout",
                    "elapsed_ms": 12345,
                    "error": (
                        "binance GET https://example.test/account"
                        "?api_key=AKIA123"
                    ),
                },
            ),
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "bot.log").write_text(
        "\n".join(
            [
                "2026-06-25T00:00:00Z INFO ok",
                "2026-06-25T00:00:01Z ERROR exchange timeout",
                "2026-06-25T00:00:02Z CRITICAL terminal startup failure",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        log_tail_lines=10,
        max_log_files=2,
    )

    assert report["ok"] is False
    assert report["attention"] is True
    assert report["monitor"]["files_scanned"] == 1
    assert report["monitor"]["live_events"] == 2
    assert report["bots"] == [
        {
            "bot": "binance/binance_01",
            "events": 2,
            "event_types": {"cycle.completed": 1, "remote_call.failed": 1},
            "invalid_ts": 0,
            "last_ts": 1100,
            "levels": {"info": 1, "warning": 1},
            "hard_problem_events": 0,
            "problem_events": 1,
            "statuses": {"failed": 1, "succeeded": 1},
        }
    ]
    assert report["problem_events"][0]["event_type"] == "remote_call.failed"
    assert report["problem_events"][0]["hard"] is False
    assert report["problem_events"][0]["ids"] == {"remote_call_id": "rc_1"}
    assert report["hard_failure_sources"] == {
        "monitor_errors": 0,
        "invalid_event_rows": 0,
        "hard_problem_events": 0,
        "log_hard_matches": 1,
        "process_hard_failures": 0,
        "total": 1,
    }
    assert report["attention_sources"] == {
        "problem_events": 1,
        "log_attention_matches": 2,
        "dropped_unparsed_attention_matches": 0,
        "total": 3,
    }
    assert report["remote_call_failures"] == {
        "total": 1,
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "reason_code": "request_timeout",
                "surface": "balance",
                "error_type": "RequestTimeout",
                "component": "test",
                "count": 1,
                "latest_ts": 1100,
                "latest_elapsed_ms": 12345,
                "latest_error": (
                    "binance GET https://example.test/account?api_key=[redacted]"
                ),
                "latest_ids": {"remote_call_id": "rc_1"},
            }
        ],
    }
    assert report["remote_call_health"] == {
        "total": 1,
        "succeeded": 0,
        "failed": 1,
        "throttled": 0,
        "failure_pct": 100,
        "throttled_pct": 0,
        "failed_reason_codes": {"request_timeout": 1},
        "failed_error_types": {"RequestTimeout": 1},
        "failed_surfaces": {"balance": 1},
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "component": "test",
                "surface": "balance",
                "count": 1,
                "succeeded": 0,
                "failed": 1,
                "throttled": 0,
                "failure_pct": 100,
                "throttled_pct": 0,
                "statuses": {"failed": 1},
                "reason_codes": {"request_timeout": 1},
                "error_types": {"RequestTimeout": 1},
                "symbols": {
                    "count": 1,
                    "sample": ["BTC/USDT:USDT"],
                    "truncated": 0,
                },
                "elapsed_ms": {
                    "median_ms": 12345,
                    "p95_ms": 12345,
                    "min_ms": 12345,
                    "max_ms": 12345,
                },
                "latest_ts": 1100,
                "latest_event_type": "remote_call.failed",
                "latest_status": "failed",
                "latest_elapsed_ms": 12345,
                "latest_symbol": "BTC/USDT:USDT",
                "latest_error_type": "RequestTimeout",
                "latest_ids": {"remote_call_id": "rc_1"},
            }
        ],
    }
    assert report["remote_call_timings"] == {
        "total": 1,
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "event_type": "remote_call.failed",
                "reason_code": "request_timeout",
                "surface": "balance",
                "error_type": "RequestTimeout",
                "component": "test",
                "status": "failed",
                "symbol": "BTC/USDT:USDT",
                "count": 1,
                "latest_ts": 1100,
                "latest_elapsed_ms": 12345,
                "latest_ids": {"remote_call_id": "rc_1"},
                "elapsed_ms": {
                    "median_ms": 12345,
                    "p95_ms": 12345,
                    "min_ms": 12345,
                    "max_ms": 12345,
                },
            }
        ],
    }
    assert report["hard_problem_event_count"] == 0
    assert report["logs"]["attention_matches"] == 2
    assert report["logs"]["hard_matches"] == 1
    assert [match["hard"] for match in report["logs"]["matches"]] == [False, True]


def test_live_smoke_report_summarizes_remote_call_timings(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=1,
                ts=1000,
                level="debug",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rca_1",
                    "remote_call_group_id": "cy_1:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 5000,
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=2,
                ts=2000,
                level="debug",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 15000,
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=3,
                ts=3000,
                level="debug",
                reason_code="authoritative_open_orders",
                ids={
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_3",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "open_orders",
                    "elapsed_ms": 90000,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is True
    assert report["attention"] is False
    assert report["remote_call_failures"] == {
        "total": 0,
        "groups_truncated": False,
        "groups": [],
    }
    assert report["remote_call_health"] == {
        "total": 3,
        "succeeded": 3,
        "failed": 0,
        "throttled": 0,
        "failure_pct": 0,
        "throttled_pct": 0,
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "component": "test",
                "kind": "authoritative_state_fetch",
                "surface": "open_orders",
                "count": 1,
                "succeeded": 1,
                "failed": 0,
                "throttled": 0,
                "failure_pct": 0,
                "throttled_pct": 0,
                "statuses": {"succeeded": 1},
                "reason_codes": {"authoritative_open_orders": 1},
                "elapsed_ms": {
                    "median_ms": 90000,
                    "p95_ms": 90000,
                    "min_ms": 90000,
                    "max_ms": 90000,
                },
                "latest_ts": 3000,
                "latest_event_type": "remote_call.succeeded",
                "latest_status": "succeeded",
                "latest_elapsed_ms": 90000,
                "latest_ids": {
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_3",
                    "remote_call_group_id": "cy_2:authoritative",
                },
            },
            {
                "bot": "binance/binance_01",
                "component": "test",
                "kind": "authoritative_state_fetch",
                "surface": "balance",
                "count": 2,
                "succeeded": 2,
                "failed": 0,
                "throttled": 0,
                "failure_pct": 0,
                "throttled_pct": 0,
                "statuses": {"succeeded": 2},
                "reason_codes": {"authoritative_balance": 2},
                "elapsed_ms": {
                    "median_ms": 10000,
                    "p95_ms": 15000,
                    "min_ms": 5000,
                    "max_ms": 15000,
                },
                "latest_ts": 2000,
                "latest_event_type": "remote_call.succeeded",
                "latest_status": "succeeded",
                "latest_elapsed_ms": 15000,
                "latest_ids": {
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
            },
        ],
    }
    assert report["remote_call_timings"] == {
        "total": 3,
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "event_type": "remote_call.succeeded",
                "reason_code": "authoritative_open_orders",
                "surface": "open_orders",
                "kind": "authoritative_state_fetch",
                "component": "test",
                "status": "succeeded",
                "count": 1,
                "latest_ts": 3000,
                "latest_elapsed_ms": 90000,
                "latest_ids": {
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_3",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                "elapsed_ms": {
                    "median_ms": 90000,
                    "p95_ms": 90000,
                    "min_ms": 90000,
                    "max_ms": 90000,
                },
            },
            {
                "bot": "binance/binance_01",
                "event_type": "remote_call.succeeded",
                "reason_code": "authoritative_balance",
                "surface": "balance",
                "kind": "authoritative_state_fetch",
                "component": "test",
                "status": "succeeded",
                "count": 2,
                "latest_ts": 2000,
                "latest_elapsed_ms": 15000,
                "latest_ids": {
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                "elapsed_ms": {
                    "median_ms": 10000,
                    "p95_ms": 15000,
                    "min_ms": 5000,
                    "max_ms": 15000,
                },
            },
        ],
    }


def test_live_smoke_report_remote_call_health_totals_mixed_groups(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rca_1",
                    "remote_call_group_id": "cy_1:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=2000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 7000,
                    "error_type": "RequestTimeout",
                },
            ),
            *[
                _monitor_row(
                    event_type="remote_call.succeeded",
                    seq=3 + idx,
                    ts=3000 + idx,
                    level="debug",
                    reason_code="authoritative_open_orders",
                    ids={
                        "cycle_id": f"cy_{3 + idx}",
                        "remote_call_id": f"rca_{3 + idx}",
                        "remote_call_group_id": f"cy_{3 + idx}:authoritative",
                    },
                    data={
                        "kind": "authoritative_state_fetch",
                        "surface": "open_orders",
                        "elapsed_ms": 100 + idx,
                    },
                )
                for idx in range(8)
            ],
            _monitor_row(
                event_type="remote_call.throttled",
                seq=11,
                ts=4000,
                status="deferred",
                level="debug",
                reason_code="rate_limited",
                ids={
                    "cycle_id": "cy_11",
                    "remote_call_id": "rct_11",
                    "remote_call_group_id": "cy_11:candles",
                },
                data={
                    "kind": "ccxt_fetch_ohlcv",
                    "elapsed_ms": 250,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    health = report["remote_call_health"]
    assert health["total"] == 11
    assert health["succeeded"] == 8
    assert health["failed"] == 2
    assert health["throttled"] == 1
    assert health["failure_pct"] == 18
    assert health["throttled_pct"] == 9
    assert health["failed_reason_codes"] == {"authoritative_balance": 2}
    assert health["failed_error_types"] == {"RequestTimeout": 2}
    assert health["failed_kinds"] == {"authoritative_state_fetch": 2}
    assert health["failed_surfaces"] == {"balance": 2}
    groups_by_surface = {
        (group.get("kind"), group.get("surface")): group for group in health["groups"]
    }
    assert groups_by_surface[
        ("authoritative_state_fetch", "balance")
    ]["failed"] == 2
    assert groups_by_surface[
        ("authoritative_state_fetch", "open_orders")
    ]["succeeded"] == 8
    assert groups_by_surface[("ccxt_fetch_ohlcv", None)]["throttled"] == 1


def test_live_smoke_report_account_critical_remote_call_health_subset(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rca_1",
                    "remote_call_group_id": "cy_1:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=2,
                ts=2000,
                status="succeeded",
                level="debug",
                reason_code="authoritative_positions",
                ids={
                    "cycle_id": "cy_2",
                    "remote_call_id": "rca_2",
                    "remote_call_group_id": "cy_2:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "positions",
                    "elapsed_ms": 600,
                },
            ),
            _monitor_row(
                event_type="remote_call.succeeded",
                seq=3,
                ts=3000,
                status="succeeded",
                level="debug",
                reason_code="authoritative_open_orders",
                ids={
                    "cycle_id": "cy_3",
                    "remote_call_id": "rca_3",
                    "remote_call_group_id": "cy_3:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "open_orders",
                    "elapsed_ms": 300,
                },
            ),
            _monitor_row(
                event_type="remote_call.failed",
                seq=4,
                ts=4000,
                status="failed",
                level="warning",
                reason_code="authoritative_fills",
                ids={
                    "cycle_id": "cy_4",
                    "remote_call_id": "rca_4",
                    "remote_call_group_id": "cy_4:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "fills",
                    "elapsed_ms": 1200,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="remote_call.throttled",
                seq=5,
                ts=5000,
                status="deferred",
                level="debug",
                reason_code="rate_limited",
                ids={
                    "cycle_id": "cy_5",
                    "remote_call_id": "rcc_5",
                    "remote_call_group_id": "cy_5:candles",
                },
                data={
                    "kind": "ccxt_fetch_ohlcv",
                    "elapsed_ms": 250,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["remote_call_health"]["total"] == 5
    account_health = report["account_critical_remote_call_health"]
    assert account_health["total"] == 3
    assert account_health["succeeded"] == 2
    assert account_health["failed"] == 1
    assert account_health["throttled"] == 0
    assert account_health["failure_pct"] == 33
    assert account_health["throttled_pct"] == 0
    assert account_health["failed_reason_codes"] == {"authoritative_balance": 1}
    assert account_health["failed_error_types"] == {"RequestTimeout": 1}
    assert account_health["failed_kinds"] == {"authoritative_state_fetch": 1}
    assert account_health["failed_surfaces"] == {"balance": 1}
    group_surfaces = {group.get("surface") for group in account_health["groups"]}
    assert group_surfaces == {"balance", "positions", "open_orders"}


def test_live_smoke_report_summarizes_execution_health(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    rows = [
        _monitor_row(
            event_type="order_wave.started",
            seq=1,
            ts=1000,
            status="started",
            ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
            data={
                "id": 1,
                "planned_cancel": 1,
                "planned_create": 2,
                "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            },
        ),
        _monitor_row(
            event_type="execution.create_sent",
            seq=2,
            ts=1100,
            status="started",
            symbol="BTC/USDT:USDT",
            pside="long",
            ids={
                "cycle_id": "cy_1",
                "order_wave_id": "ow_1",
                "action_id": "ow_1:create:0",
            },
            data={
                "index": 0,
                "order_type": "limit",
                "context": "entry",
                "qty": 1.0,
                "price": 10.0,
                "client_order_id_short": "pb_secret_create",
            },
        ),
        _monitor_row(
            event_type="execution.create_failed",
            seq=3,
            ts=1200,
            status="failed",
            level="warning",
            reason_code="exchange_exception",
            symbol="BTC/USDT:USDT",
            pside="long",
            ids={
                "cycle_id": "cy_1",
                "order_wave_id": "ow_1",
                "action_id": "ow_1:create:0",
            },
            data={
                "index": 0,
                "order_type": "limit",
                "error_type": "RequestTimeout",
                "error": "raw exchange error should stay out",
                "result_order_id_short": "exchange_order_secret",
                "result_client_order_id_short": "client_secret",
            },
        ),
        _monitor_row(
            event_type="execution.cancel_ambiguous_terminal",
            seq=4,
            ts=1300,
            status="degraded",
            level="warning",
            reason_code="exchange_exception",
            symbol="ETH/USDT:USDT",
            pside="short",
            side="sell",
            ids={
                "cycle_id": "cy_1",
                "order_wave_id": "ow_1",
                "action_id": "ow_1:cancel:0",
            },
            data={
                "index": 0,
                "order_type": "limit",
                "error_type": "NetworkError",
                "order_id_short": "cancel_order_secret",
            },
        ),
        _monitor_row(
            event_type="execution.confirmation_timeout",
            seq=5,
            ts=1400,
            status="degraded",
            level="warning",
            reason_code="authoritative_confirmation_timeout",
            ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
            data={
                "elapsed_ms": 5000,
                "confirm_ms": 3000,
                "timeout_ms": 2500,
                "pending_surfaces": ["open_orders"],
                "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            },
        ),
        _monitor_row(
            event_type="order_wave.completed",
            seq=6,
            ts=1500,
            status="degraded",
            reason_code="order_filtered",
            ids={"cycle_id": "cy_1", "order_wave_id": "ow_1"},
            data={
                "id": 1,
                "elapsed_ms": 6000,
                "planned_cancel": 1,
                "planned_create": 2,
                "cancel_posted": 0,
                "create_posted": 0,
                "skipped_cancel": 0,
                "deferred_create": 1,
                "skipped_create": 1,
                "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
            },
        ),
    ]
    rows[2]["payload"]["_live_event"]["order_id"] = "raw_exchange_order_id"
    rows[2]["payload"]["_live_event"]["client_order_id"] = "raw_client_order_id"
    _write_ndjson(events_dir / "current.ndjson", rows)

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    execution = report["execution_health"]

    assert execution["total"] == 6
    assert execution["bots"] == 1
    assert execution["failed"] == 1
    assert execution["rejected"] == 0
    assert execution["ambiguous"] == 1
    assert execution["confirmation_timeout"] == 1
    assert execution["event_types"] == {
        "execution.cancel_ambiguous_terminal": 1,
        "execution.confirmation_timeout": 1,
        "execution.create_failed": 1,
        "execution.create_sent": 1,
        "order_wave.completed": 1,
        "order_wave.started": 1,
    }
    assert execution["outcomes"]["create_failed"] == 1
    assert execution["outcomes"]["cancel_ambiguous_terminal"] == 1
    assert execution["outcomes"]["confirmation_timeout"] == 1
    assert execution["statuses"] == {
        "degraded": 3,
        "failed": 1,
        "started": 2,
    }
    assert execution["groups_truncated"] is False
    rendered_execution = json.dumps(execution, sort_keys=True)
    assert "raw_exchange_order_id" not in rendered_execution
    assert "raw_client_order_id" not in rendered_execution
    assert "exchange_order_secret" not in rendered_execution
    assert "client_secret" not in rendered_execution
    assert "cancel_order_secret" not in rendered_execution
    assert "raw exchange error" not in rendered_execution
    assert '"price"' not in rendered_execution
    assert '"qty"' not in rendered_execution
    assert "RequestTimeout" in rendered_execution
    assert "ow_1:create:0" in rendered_execution

    summary = summarize_live_smoke_report(report, max_groups=2)
    assert summary["execution_health"]["total"] == 6
    assert summary["execution_health"]["failed"] == 1
    assert summary["execution_health"]["ambiguous"] == 1
    assert summary["execution_health"]["confirmation_timeout"] == 1
    assert summary["execution_health"]["groups_truncated"] is True

    brief = summarize_live_smoke_report_brief(report)
    assert brief["execution"] == {
        "total": 6,
        "bots": 1,
        "failed": 1,
        "rejected": 0,
        "ambiguous": 1,
        "confirmation_timeout": 1,
        "event_types": {
            "execution.cancel_ambiguous_terminal": 1,
            "execution.confirmation_timeout": 1,
            "execution.create_failed": 1,
            "execution.create_sent": 1,
            "order_wave.completed": 1,
            "order_wave.started": 1,
        },
        "statuses": {
            "degraded": 3,
            "failed": 1,
            "started": 2,
        },
        "outcomes": {
            "cancel_ambiguous_terminal": 1,
            "confirmation_timeout": 1,
            "create_failed": 1,
            "create_sent": 1,
            "wave_completed": 1,
            "wave_started": 1,
        },
    }

    projected = project_live_smoke_report_sections(
        summarize_live_smoke_report_brief(report),
        ["execution"],
    )
    assert projected["execution"]["total"] == 6
    assert "remote_calls" not in projected


def test_live_smoke_report_summary_projects_high_signal_fields(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rca_1",
                    "remote_call_group_id": "cy_1:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=2,
                ts=2000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_2"},
                data={
                    "candidate_unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                },
            ),
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kucoin_01.log").write_text(
        "2026-06-25T00:00:00Z ERROR exchange timeout\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        log_tail_lines=10,
    )
    summary = summarize_live_smoke_report(report, max_groups=1)

    assert summary["ok"] is True
    assert summary["attention"] is True
    assert summary["hard_failures"] == 0
    assert summary["attention_count"] == 3
    assert summary["hard_failure_sources"] == {
        "monitor_errors": 0,
        "invalid_event_rows": 0,
        "hard_problem_events": 0,
        "log_hard_matches": 0,
        "process_hard_failures": 0,
        "total": 0,
    }
    assert summary["attention_sources"] == {
        "problem_events": 2,
        "log_attention_matches": 1,
        "dropped_unparsed_attention_matches": 0,
        "total": 3,
    }
    assert summary["monitor"]["live_events"] == 2
    assert summary["logs"]["attention_matches"] == 1
    assert summary["logs"]["matches_truncated"] is False
    assert len(summary["logs"]["matches"]) == 1
    assert summary["problem_events"]["total"] == 2
    assert summary["problem_events"]["hard"] == 0
    assert summary["problem_events"]["groups_truncated"] is True
    assert len(summary["problem_events"]["groups"]) == 1
    assert summary["problem_events"]["groups"][0]["event_type"] in {
        "ema.unavailable",
        "remote_call.failed",
    }
    assert summary["account_critical_remote_calls"]["total"] == 1
    assert summary["account_critical_remote_calls"]["failed"] == 1
    assert summary["remote_calls"]["groups"][0]["surface"] == "balance"
    assert "bots" not in summary
    assert "problem_event_groups" not in summary


def test_live_smoke_report_brief_summary_projects_top_level_counters(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rca_1",
                    "remote_call_group_id": "cy_1:authoritative",
                },
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "elapsed_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=2,
                ts=2000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_2"},
                data={
                    "candidate_unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    }
                },
            ),
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kucoin_01.log").write_text(
        "2026-06-25T00:00:00Z ERROR exchange timeout\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        log_tail_lines=10,
    )
    brief = summarize_live_smoke_report_brief(report)

    assert brief["ok"] is True
    assert brief["attention"] is True
    assert brief["hard_failures"] == 0
    assert brief["attention_count"] == 3
    assert brief["hard_failure_sources"] == {
        "monitor_errors": 0,
        "invalid_event_rows": 0,
        "hard_problem_events": 0,
        "log_hard_matches": 0,
        "process_hard_failures": 0,
        "total": 0,
    }
    assert brief["attention_sources"] == {
        "problem_events": 2,
        "log_attention_matches": 1,
        "dropped_unparsed_attention_matches": 0,
        "total": 3,
    }
    assert brief["monitor"]["live_events"] == 2
    assert brief["event_window"]["enabled"] is False
    assert brief["logs"] == {
        "max_files": 8,
        "tail_lines": 10,
        "max_matches": 50,
        "files_scanned": 1,
        "hard_matches": 0,
        "attention_matches": 1,
        "risk_attention_matches": 0,
        "risk_hard_matches": 0,
        "non_risk_attention_matches": 1,
        "non_risk_hard_matches": 0,
        "dropped_unparsed_attention_matches": 0,
        "dropped_unparsed_hard_matches": 0,
        "window": {
            "enabled": False,
            "since_ms": None,
            "until_ms": None,
            "lines_considered": 1,
            "lines_skipped_before": 0,
            "lines_skipped_after": 0,
            "unparsed_ts": 0,
            "unparsed_policy": "keep",
            "lines_skipped_unparsed": 0,
            "dropped_unparsed_attention_matches": 0,
            "dropped_unparsed_hard_matches": 0,
        },
    }
    assert brief["problem_events"]["total"] == 2
    assert brief["problem_events"]["hard"] == 0
    assert brief["problem_events"]["groups_truncated"] is False
    assert brief["problem_events"]["event_types_truncated"] is False
    assert brief["problem_events"]["event_types"] == {
        "ema.unavailable": 1,
        "remote_call.failed": 1,
    }
    problem_groups = brief["problem_events"]["groups"]
    assert len(problem_groups) == 2
    assert {group["event_type"] for group in problem_groups} == {
        "ema.unavailable",
        "remote_call.failed",
    }
    for group in problem_groups:
        assert "latest_data" not in group
        assert "latest_ids" not in group
        assert "latest_path" not in group
        assert "latest_line" not in group
    assert brief["remote_calls"]["total"] == 1
    assert brief["remote_calls"]["failed"] == 1
    assert brief["remote_calls"]["failed_reason_codes"] == {
        "authoritative_balance": 1
    }
    assert brief["remote_calls"]["failed_error_types"] == {"RequestTimeout": 1}
    assert brief["remote_calls"]["failed_kinds"] == {
        "authoritative_state_fetch": 1
    }
    assert brief["remote_calls"]["failed_surfaces"] == {"balance": 1}
    assert brief["account_critical_remote_calls"]["total"] == 1
    assert brief["account_critical_remote_calls"]["failed"] == 1
    assert brief["account_critical_remote_calls"]["failed_reason_codes"] == {
        "authoritative_balance": 1
    }
    assert brief["account_critical_remote_calls"]["failed_error_types"] == {
        "RequestTimeout": 1
    }
    assert brief["account_critical_remote_calls"]["failed_kinds"] == {
        "authoritative_state_fetch": 1
    }
    assert brief["account_critical_remote_calls"]["failed_surfaces"] == {
        "balance": 1
    }
    assert brief["ema_readiness"] == {
        "total": 1,
        "bots": 1,
        "latest_candidate_unavailable_total": 1,
        "latest_unavailable_total": 0,
        "latest_optional_drop_total": 0,
        "event_types": {"ema.unavailable": 1},
    }
    assert brief["exchange_config_refresh"] == {
        "total": 0,
        "bots": 0,
        "succeeded": 0,
        "failed": 0,
        "failure_pct": 0,
        "failed_bots": 0,
        "event_types": {},
    }
    assert brief["staged_readiness"] == {
        "total": 0,
        "bots": 0,
        "latest_missing_surface_total": 0,
        "latest_invalid_surface_total": 0,
        "event_types": {},
    }
    assert "bots" not in brief
    assert "matches" not in brief["logs"]
    assert "groups" not in brief["remote_calls"]
    assert "groups" not in brief["account_critical_remote_calls"]


def test_live_smoke_report_brief_bounds_problem_event_types(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    limit = smoke_report_module.SMOKE_REPORT_BRIEF_PROBLEM_GROUP_LIMIT
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type=f"problem.type_{idx}",
                seq=idx + 1,
                ts=1000 + idx,
                status="degraded",
                level="warning",
                reason_code=f"reason_{idx}",
            )
            for idx in range(limit + 1)
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    brief = summarize_live_smoke_report_brief(report)

    assert brief["problem_events"]["total"] == limit + 1
    assert brief["problem_events"]["hard"] == 0
    assert brief["problem_events"]["groups_truncated"] is True
    assert brief["problem_events"]["event_types_truncated"] is True
    assert len(brief["problem_events"]["groups"]) == limit
    assert len(brief["problem_events"]["event_types"]) == limit
    assert f"problem.type_{limit}" not in brief["problem_events"]["event_types"]


def test_live_smoke_report_remote_call_health_counts_throttled_events(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.throttled",
                seq=1,
                ts=1000,
                status="deferred",
                level="debug",
                reason_code="rate_limited",
                symbol="TRX/USDT:USDT",
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rct_1",
                    "remote_call_group_id": "cy_1:candles",
                },
                data={
                    "kind": "ccxt_fetch_ohlcv",
                    "elapsed_ms": 250,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is True
    assert report["remote_call_health"] == {
        "total": 1,
        "succeeded": 0,
        "failed": 0,
        "throttled": 1,
        "failure_pct": 0,
        "throttled_pct": 100,
        "groups_truncated": False,
        "groups": [
            {
                "bot": "binance/binance_01",
                "component": "test",
                "kind": "ccxt_fetch_ohlcv",
                "count": 1,
                "succeeded": 0,
                "failed": 0,
                "throttled": 1,
                "failure_pct": 0,
                "throttled_pct": 100,
                "statuses": {"throttled": 1},
                "raw_statuses": {"deferred": 1},
                "reason_codes": {"rate_limited": 1},
                "symbols": {
                    "count": 1,
                    "sample": ["TRX/USDT:USDT"],
                    "truncated": 0,
                },
                "elapsed_ms": {
                    "median_ms": 250,
                    "p95_ms": 250,
                    "min_ms": 250,
                    "max_ms": 250,
                },
                "latest_ts": 1000,
                "latest_event_type": "remote_call.throttled",
                "latest_status": "throttled",
                "latest_raw_status": "deferred",
                "latest_elapsed_ms": 250,
                "latest_symbol": "TRX/USDT:USDT",
                "latest_ids": {
                    "cycle_id": "cy_1",
                    "remote_call_id": "rct_1",
                    "remote_call_group_id": "cy_1:candles",
                },
            }
        ],
    }


def test_live_smoke_report_summarizes_fill_refresh_health(tmp_path):
    hyperliquid_dir = (
        tmp_path / "monitor" / "hyperliquid" / "hyperliquid_tradfi" / "events"
    )
    binance_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        hyperliquid_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=1,
                ts=1000,
                exchange="hyperliquid",
                user="hyperliquid_tradfi",
                status="failed",
                level="error",
                reason_code="fill_refresh_failed",
                ids={"cycle_id": "cy_fill_1"},
                data={
                    "source": "exchange",
                    "refresh_mode": "periodic",
                    "elapsed_ms": 12244,
                    "history_scope": "window",
                    "event_count_after": 2704,
                    "coverage_ready_after": False,
                    "coverage_reason_after": "window_uncovered",
                    "retry_count": 1,
                    "next_retry_in_ms": 5000,
                    "error_type": "RequestTimeout",
                },
            ),
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=2,
                ts=2000,
                exchange="hyperliquid",
                user="hyperliquid_tradfi",
                status="ready",
                level="debug",
                reason_code="fill_cache_ready",
                ids={"cycle_id": "cy_fill_2"},
                data={
                    "source": "exchange",
                    "refresh_mode": "periodic",
                    "elapsed_ms": 420,
                    "history_scope": "all",
                    "event_count_after": 2705,
                    "new_count": 1,
                    "enriched_count": 1,
                    "pending_pnl_count": 0,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )
    _write_ndjson(
        binance_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=3,
                ts=3000,
                exchange="binance",
                user="binance_01",
                status="ready",
                level="debug",
                reason_code="fill_cache_ready",
                ids={"cycle_id": "cy_fill_3"},
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "elapsed_ms": 40,
                    "history_scope": "all",
                    "event_count_after": 100,
                    "coverage_ready_after": True,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report, max_groups=1)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is False
    assert report["hard_failure_sources"]["hard_problem_events"] == 1
    health = report["fill_refresh_health"]
    assert health["total"] == 3
    assert health["bots"] == 2
    assert health["failed"] == 1
    assert health["failure_pct"] == 33
    assert health["failed_bots"] == 1
    assert health["latest_failed_bots"] == 0
    assert health["recovered_groups"] == 1
    assert health["statuses"] == {"ready": 2, "failed": 1}
    assert health["groups_truncated"] is False
    recovered = health["groups"][0]
    assert recovered["bot"] == "hyperliquid/hyperliquid_tradfi"
    assert recovered["source"] == "exchange"
    assert recovered["refresh_mode"] == "periodic"
    assert recovered["count"] == 2
    assert recovered["failed"] == 1
    assert recovered["recovered"] is True
    assert recovered["latest_status"] == "ready"
    assert recovered["latest_history_scope"] == "all"
    assert recovered["latest_event_count_after"] == 2705
    assert recovered["latest_new_count"] == 1
    assert recovered["latest_coverage_ready_after"] is True
    assert recovered["error_types"] == {"RequestTimeout": 1}
    assert recovered["latest_ids"] == {"cycle_id": "cy_fill_2"}
    assert summary["fill_refresh_health"]["total"] == 3
    assert summary["fill_refresh_health"]["groups_truncated"] is True
    assert summary["fill_refresh_health"]["groups"][0]["recovered"] is True
    assert brief["fill_refresh"] == {
        "total": 3,
        "bots": 2,
        "failed": 1,
        "failure_pct": 33,
        "failed_bots": 1,
        "latest_failed_bots": 0,
        "recovered_groups": 1,
        "statuses": {"ready": 2, "failed": 1},
    }


def test_live_smoke_report_problem_events_include_allowlisted_ema_data(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="ema.unavailable",
                seq=1,
                ts=1000,
                status="degraded",
                level="debug",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_ema"},
                data={
                    "candidate_unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "candidate_unavailable_groups": [
                        {
                            "reason": "candidate_required_ema_unavailable",
                            "symbols": {
                                "count": 1,
                                "sample": ["ZEC/USDT:USDT"],
                                "truncated": 0,
                            },
                            "error_types": ["MissingCloseEma"],
                            "example_error": (
                                "GET https://example.test/candles"
                                "?api_key=SECRET&signature=SIG"
                            ),
                        }
                    ],
                    "unavailable_reasons": [
                        {
                            "reason": "candidate_required_ema_unavailable",
                            "symbols": {
                                "count": 1,
                                "sample": ["ZEC/USDT:USDT"],
                                "truncated": 0,
                            },
                        }
                    ],
                    "ignored_extra": "do not surface",
                },
            )
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is True
    event = report["problem_events"][0]
    assert event["event_type"] == "ema.unavailable"
    assert event["ids"] == {"cycle_id": "cy_ema"}
    latest_data = event["latest_data"]
    assert latest_data["candidate_unavailable"]["sample"] == ["ZEC/USDT:USDT"]
    assert latest_data["candidate_unavailable_groups"][0]["reason"] == (
        "candidate_required_ema_unavailable"
    )
    assert (
        latest_data["candidate_unavailable_groups"][0]["example_error"]
        == "GET https://example.test/candles?api_key=[redacted]&signature=[redacted]"
    )
    assert "ignored_extra" not in latest_data


def test_live_smoke_report_summarizes_ema_readiness_health(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="ema.unavailable",
                seq=1,
                ts=1000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_ema_1"},
                data={
                    "candidate_unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 3,
                        "sample": ["ZEC/USDT:USDT", "NEAR/USDT:USDT"],
                        "truncated": 1,
                    },
                    "candidate_unavailable_groups": [
                        {
                            "reason": "cache_only_fetch_failed",
                            "symbols": {
                                "count": 1,
                                "sample": ["ZEC/USDT:USDT"],
                                "truncated": 0,
                            },
                            "error_types": ["MissingCloseEma"],
                        }
                    ],
                    "unavailable_reasons": [
                        {
                            "reason": "cache_only_fetch_failed",
                            "symbols": {
                                "count": 1,
                                "sample": ["ZEC/USDT:USDT"],
                                "truncated": 0,
                            },
                        },
                        {
                            "reason": "never_fetched_cache_only",
                            "symbols": {
                                "count": 2,
                                "sample": ["NEAR/USDT:USDT", "AVAX/USDT:USDT"],
                                "truncated": 0,
                            },
                        },
                    ],
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=2,
                ts=2000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_ema_2"},
                data={
                    "optional_drop_count": 2,
                    "candidate_unavailable": {
                        "count": 2,
                        "sample": ["AVAX/USDT:USDT", "WLD/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 5,
                        "sample": ["AVAX/USDT:USDT", "WLD/USDT:USDT"],
                        "truncated": 3,
                    },
                    "candidate_unavailable_groups": [
                        {
                            "reason": "cache_only_fetch_failed",
                            "symbols": {
                                "count": 2,
                                "sample": ["AVAX/USDT:USDT", "WLD/USDT:USDT"],
                                "truncated": 0,
                            },
                            "error_types": ["MissingLogRangeEma"],
                        }
                    ],
                    "unavailable_reasons": [
                        {
                            "reason": "cache_only_fetch_failed",
                            "symbols": {
                                "count": 2,
                                "sample": ["AVAX/USDT:USDT", "WLD/USDT:USDT"],
                                "truncated": 0,
                            },
                        },
                        {
                            "reason": "never_fetched_cache_only",
                            "symbols": {
                                "count": 3,
                                "sample": ["SUI/USDT:USDT"],
                                "truncated": 2,
                            },
                        },
                    ],
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    health = report["ema_readiness_health"]
    assert health["total"] == 2
    assert health["bots"] == 1
    assert health["latest_candidate_unavailable_total"] == 2
    assert health["latest_unavailable_total"] == 5
    assert health["latest_optional_drop_total"] == 2
    assert health["event_types"] == {"ema.unavailable": 2}
    assert health["latest_candidate_reason_counts"] == {
        "cache_only_fetch_failed": 2
    }
    assert health["latest_unavailable_reason_counts"] == {
        "never_fetched_cache_only": 3,
        "cache_only_fetch_failed": 2,
    }
    assert health["latest_candidate_error_type_counts"] == {
        "MissingLogRangeEma": 1
    }
    assert health["groups"][0]["count"] == 2
    assert health["groups"][0]["latest_ids"] == {"cycle_id": "cy_ema_2"}
    assert health["groups"][0]["latest_candidate_unavailable_count"] == 2
    assert health["groups"][0]["latest_unavailable_count"] == 5
    assert health["groups"][0]["candidate_reason_counts"] == {
        "cache_only_fetch_failed": 2
    }
    assert health["groups"][0]["unavailable_reason_counts"] == {
        "never_fetched_cache_only": 3,
        "cache_only_fetch_failed": 2,
    }
    assert health["groups"][0]["candidate_error_type_counts"] == {
        "MissingLogRangeEma": 1
    }
    assert summary["ema_readiness_health"]["total"] == 2
    assert summary["ema_readiness_health"]["groups"][0]["latest_ids"] == {
        "cycle_id": "cy_ema_2"
    }
    assert brief["ema_readiness"] == {
        "total": 2,
        "bots": 1,
        "latest_candidate_unavailable_total": 2,
        "latest_unavailable_total": 5,
        "latest_optional_drop_total": 2,
        "latest_candidate_reason_counts": {"cache_only_fetch_failed": 2},
        "latest_unavailable_reason_counts": {
            "never_fetched_cache_only": 3,
            "cache_only_fetch_failed": 2,
        },
        "latest_candidate_error_type_counts": {"MissingLogRangeEma": 1},
        "event_types": {"ema.unavailable": 2},
    }


def test_live_smoke_report_summarizes_exchange_config_refresh_health(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=1,
                ts=1000,
                status="succeeded",
                level="debug",
                reason_code="exchange_config_refresh",
                ids={"cycle_id": "cy_cfg_1"},
                data={
                    "context": "maintain_hourly_cycle",
                    "operation": "init_markets",
                    "started_ms": 900,
                    "elapsed_ms": 100,
                },
            ),
            _monitor_row(
                event_type="exchange.config_refresh",
                seq=2,
                ts=2000,
                status="failed",
                level="warning",
                reason_code="exchange_config_refresh_failed",
                ids={"cycle_id": "cy_cfg_2"},
                data={
                    "context": "maintain_hourly_cycle",
                    "operation": "init_markets",
                    "started_ms": 1800,
                    "elapsed_ms": 200,
                    "error_type": "ExchangeError",
                    "error": "binanceusdm apiKey=supersecret code=-4084",
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    health = report["exchange_config_refresh_health"]
    assert health["total"] == 2
    assert health["succeeded"] == 1
    assert health["failed"] == 1
    assert health["failure_pct"] == 50.0
    assert health["bots"] == 1
    assert health["failed_bots"] == 1
    assert health["event_types"] == {"exchange.config_refresh": 2}
    assert health["statuses"] == {"succeeded": 1, "failed": 1}
    assert health["groups"][0]["status"] == "failed"
    assert health["groups"][0]["latest_data"] == {
        "context": "maintain_hourly_cycle",
        "operation": "init_markets",
        "error_type": "ExchangeError",
        "elapsed_ms": 200,
        "started_ms": 1800,
    }
    rendered = json.dumps(health, sort_keys=True)
    assert "supersecret" not in rendered
    assert "apiKey" not in rendered
    assert "code=-4084" not in rendered

    assert summary["exchange_config_refresh_health"]["total"] == 2
    assert summary["exchange_config_refresh_health"]["statuses"] == {
        "succeeded": 1,
        "failed": 1,
    }
    assert brief["exchange_config_refresh"] == {
        "total": 2,
        "bots": 1,
        "succeeded": 1,
        "failed": 1,
        "failure_pct": 50.0,
        "failed_bots": 1,
        "event_types": {"exchange.config_refresh": 2},
    }


def test_live_smoke_report_summarizes_event_pipeline_health(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                seq=1,
                ts=1000,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "cy_health_1"},
                data={
                    "event_queue_depth": 1,
                    "event_queue_maxsize": 5000,
                    "event_queue_unfinished_tasks": 2,
                    "event_dropped_total": 0,
                    "event_drop_counts": {},
                    "event_sink_error_total": 0,
                    "event_sink_error_counts": {},
                    "event_degraded_count": 0,
                    "event_pipeline_stopping": False,
                    "event_pipeline_worker_alive": True,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=2,
                ts=2000,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "cy_health_2"},
                data={
                    "event_queue_depth": 3,
                    "event_queue_maxsize": 5000,
                    "event_queue_unfinished_tasks": 4,
                    "event_dropped_total": 2,
                    "event_drop_counts": {"health.summary": 2},
                    "event_sink_error_total": 1,
                    "event_sink_error_counts": {"monitor": 1},
                    "event_degraded_count": 3,
                    "event_pipeline_stopping": True,
                    "event_pipeline_worker_alive": False,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                seq=3,
                ts=3000,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "cy_ignored"},
                data={"process_rss_mb": 125},
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    health = report["event_pipeline_health"]
    assert health["total"] == 2
    assert health["bots"] == 1
    assert health["latest_queue_depth_total"] == 3
    assert health["latest_queue_unfinished_total"] == 4
    assert health["latest_dropped_total"] == 2
    assert health["latest_sink_error_total"] == 1
    assert health["latest_degraded_total"] == 3
    assert health["latest_worker_not_alive_count"] == 1
    assert health["latest_stopping_count"] == 1
    assert health["event_types"] == {"health.summary": 2}
    assert health["groups"][0]["latest_ids"] == {"cycle_id": "cy_health_2"}
    assert health["groups"][0]["latest_drop_counts"] == {"health.summary": 2}
    assert health["groups"][0]["latest_sink_error_counts"] == {"monitor": 1}
    assert health["groups"][0]["latest_worker_alive"] is False
    assert health["groups"][0]["latest_pipeline_stopping"] is True
    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert summary["event_pipeline_health"]["total"] == 2
    assert summary["event_pipeline_health"]["groups"][0]["latest_ids"] == {
        "cycle_id": "cy_health_2"
    }
    assert brief["event_pipeline"] == {
        "total": 2,
        "bots": 1,
        "latest_queue_depth_total": 3,
        "latest_queue_unfinished_total": 4,
        "latest_dropped_total": 2,
        "latest_sink_error_total": 1,
        "latest_degraded_total": 3,
        "latest_worker_not_alive_count": 1,
        "latest_stopping_count": 1,
        "event_types": {"health.summary": 2},
    }


def test_live_smoke_report_event_pipeline_health_aggregates_multi_bot_queue_overflow(tmp_path):
    okx_events = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    gateio_events = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        okx_events / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                exchange="okx",
                user="okx_01",
                seq=1,
                ts=1000,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "okx_old"},
                data={
                    "event_queue_depth": 4,
                    "event_queue_unfinished_tasks": 5,
                    "event_dropped_total": 1,
                    "event_drop_counts": {"order.created": 1},
                    "event_sink_error_total": 0,
                    "event_degraded_count": 1,
                    "event_pipeline_stopping": False,
                    "event_pipeline_worker_alive": True,
                },
            ),
            _monitor_row(
                event_type="health.summary",
                exchange="okx",
                user="okx_01",
                seq=2,
                ts=2000,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "okx_latest"},
                data={
                    "event_queue_depth": 8,
                    "event_queue_unfinished_tasks": 9,
                    "event_dropped_total": 7,
                    "event_drop_counts": {"order.created": 5, "health.summary": 2},
                    "event_sink_error_total": 1,
                    "event_sink_error_counts": {"monitor": 1},
                    "event_degraded_count": 2,
                    "event_pipeline_stopping": False,
                    "event_pipeline_worker_alive": True,
                },
            ),
        ],
    )
    _write_ndjson(
        gateio_events / "current.ndjson",
        [
            _monitor_row(
                event_type="health.summary",
                exchange="gateio",
                user="gateio_01",
                seq=3,
                ts=1500,
                level="debug",
                reason_code="periodic_health_summary",
                ids={"cycle_id": "gateio_latest"},
                data={
                    "event_queue_depth": 3,
                    "event_queue_unfinished_tasks": 4,
                    "event_dropped_total": 0,
                    "event_sink_error_total": 2,
                    "event_sink_error_counts": {"structured": 2},
                    "event_degraded_count": 4,
                    "event_pipeline_stopping": True,
                    "event_pipeline_worker_alive": False,
                },
            )
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report, max_groups=1)
    brief = summarize_live_smoke_report_brief(report)

    health = report["event_pipeline_health"]
    assert health["total"] == 3
    assert health["bots"] == 2
    assert health["latest_queue_depth_total"] == 11
    assert health["latest_queue_unfinished_total"] == 13
    assert health["latest_dropped_total"] == 7
    assert health["latest_sink_error_total"] == 3
    assert health["latest_degraded_total"] == 6
    assert health["latest_worker_not_alive_count"] == 1
    assert health["latest_stopping_count"] == 1
    assert [group["bot"] for group in health["groups"]] == [
        "okx/okx_01",
        "gateio/gateio_01",
    ]
    assert health["groups"][0]["latest_ids"] == {"cycle_id": "okx_latest"}
    assert health["groups"][0]["latest_drop_counts"] == {
        "health.summary": 2,
        "order.created": 5,
    }
    assert health["groups"][1]["latest_sink_error_counts"] == {"structured": 2}
    assert report["ok"] is True
    assert report["attention"] is False
    assert summary["event_pipeline_health"]["total"] == 3
    assert summary["event_pipeline_health"]["groups_truncated"] is True
    assert summary["event_pipeline_health"]["groups"][0]["bot"] == "okx/okx_01"
    assert brief["event_pipeline"] == {
        "total": 3,
        "bots": 2,
        "latest_queue_depth_total": 11,
        "latest_queue_unfinished_total": 13,
        "latest_dropped_total": 7,
        "latest_sink_error_total": 3,
        "latest_degraded_total": 6,
        "latest_worker_not_alive_count": 1,
        "latest_stopping_count": 1,
        "event_types": {"health.summary": 3},
    }


def test_live_smoke_report_problem_events_include_cycle_degraded_details(tmp_path):
    events_dir = tmp_path / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.degraded",
                seq=1,
                ts=1000,
                status="degraded",
                level="debug",
                reason_code="staged_execution_not_ready",
                ids={"cycle_id": "cy_blocked"},
                data={
                    "details": {
                        "context": "market snapshot refresh",
                        "missing": ["completed_candles"],
                        "required": ["positions", "balance"],
                        "invalid": {
                            "auth": "api_key=SECRET&signature=SIG",
                        },
                    },
                    "timings_ms": {"market_state": 112},
                    "authoritative_epoch": 9,
                },
            )
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    event = report["problem_events"][0]
    assert event["event_type"] == "cycle.degraded"
    assert event["latest_data"] == {
        "authoritative_epoch": 9,
        "details": {
            "context": "market snapshot refresh",
            "invalid": {"auth": "api_key=[redacted]&signature=[redacted]"},
            "missing": ["completed_candles"],
            "required": ["positions", "balance"],
        },
    }


def test_live_smoke_report_recovers_time_sync_cycle_degraded(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.degraded",
                seq=1,
                ts=1000,
                exchange="kucoin",
                user="kucoin_01",
                status="degraded",
                level="error",
                reason_code="InvalidNonce",
                ids={"cycle_id": "cy_1"},
                data={
                    "error_type": "InvalidNonce",
                    "error": 'kucoinfutures {"code":"400002","msg":"Invalid KC-API-TIMESTAMP"}',
                },
            ),
            _monitor_row(
                event_type="exchange.time_sync",
                seq=2,
                ts=1100,
                exchange="kucoin",
                user="kucoin_01",
                status="succeeded",
                level="debug",
                reason_code="exchange_time_sync",
                ids={"cycle_id": "cy_1"},
                data={
                    "error_type": "InvalidNonce",
                    "recovered": True,
                    "synced_count": 2,
                    "failed_count": 0,
                },
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=1200,
                exchange="kucoin",
                user="kucoin_01",
                status="succeeded",
                level="info",
                reason_code="cycle_completed",
                ids={"cycle_id": "cy_2"},
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["hard_problem_event_count"] == 0
    assert report["recovered_problem_events"] == {
        "total": 1,
        "hard": 1,
        "event_types": {"cycle.degraded": 1},
    }
    assert report["problem_events"][0]["hard"] is False
    assert report["problem_events"][0]["recovered"] is True
    assert report["problem_events"][0]["recovery"] == {
        "event_type": "exchange.time_sync",
        "reason_code": "exchange_time_sync",
        "status": "succeeded",
        "ts": 1100,
        "cycle_id": "cy_1",
    }
    assert report["problem_event_groups"]["groups"][0]["recovered"] is True
    assert summary["recovered_problem_events"]["hard"] == 1
    assert brief["recovered_problem_events"]["hard"] == 1


def test_live_smoke_report_recovers_time_sync_after_cycle_id_cleared(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.degraded",
                seq=1,
                ts=1000,
                exchange="kucoin",
                user="kucoin_01",
                status="degraded",
                level="error",
                reason_code="InvalidNonce",
                ids={"cycle_id": "cy_1"},
                data={"error_type": "InvalidNonce"},
            ),
            _monitor_row(
                event_type="exchange.time_sync",
                seq=2,
                ts=1100,
                exchange="kucoin",
                user="kucoin_01",
                status="succeeded",
                level="debug",
                reason_code="exchange_time_sync",
                ids={},
                data={
                    "error_type": "InvalidNonce",
                    "recovered": True,
                    "synced_count": 2,
                    "failed_count": 0,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is True
    assert report["hard_problem_event_count"] == 0
    assert report["recovered_problem_events"] == {
        "total": 1,
        "hard": 1,
        "event_types": {"cycle.degraded": 1},
    }
    assert report["problem_events"][0]["recovery"] == {
        "event_type": "exchange.time_sync",
        "reason_code": "exchange_time_sync",
        "status": "succeeded",
        "ts": 1100,
        "cycle_id": None,
    }


def test_live_smoke_report_keeps_unrecovered_time_sync_cycle_degraded_hard(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.degraded",
                seq=1,
                ts=1000,
                exchange="kucoin",
                user="kucoin_01",
                status="degraded",
                level="error",
                reason_code="InvalidNonce",
                ids={"cycle_id": "cy_1"},
                data={"error_type": "InvalidNonce"},
            ),
            _monitor_row(
                event_type="exchange.time_sync",
                seq=2,
                ts=1100,
                exchange="kucoin",
                user="kucoin_01",
                status="degraded",
                level="warning",
                reason_code="exchange_time_sync",
                ids={"cycle_id": "cy_1"},
                data={
                    "error_type": "InvalidNonce",
                    "recovered": False,
                    "synced_count": 0,
                    "failed_count": 1,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["hard_problem_event_count"] == 1
    assert report["recovered_problem_events"] == {
        "total": 0,
        "hard": 0,
        "event_types": {},
    }
    assert report["problem_events"][0]["hard"] is True
    assert "recovered" not in report["problem_events"][0]


def test_live_smoke_report_summarizes_staged_readiness_health(tmp_path):
    events_dir = tmp_path / "monitor" / "hyperliquid" / "tradfi" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.degraded",
                seq=1,
                ts=1000,
                status="degraded",
                level="debug",
                reason_code="staged_execution_not_ready",
                ids={"cycle_id": "cy_stage_1"},
                data={
                    "details": {
                        "context": "market snapshot refresh",
                        "defer_reason": "staged_planner_inputs_not_fresh",
                        "missing": ["completed_candles"],
                        "invalid": {
                            "completed_candles": [
                                {
                                    "mismatch_type": "completed_candle_target_changed",
                                    "changed_count": 2,
                                    "changed_symbols": [
                                        "XYZ-NVDA/USDC:USDC",
                                        "XYZ-SPCX/USDC:USDC",
                                    ],
                                }
                            ]
                        },
                    }
                },
            ),
            _monitor_row(
                event_type="cycle.degraded",
                seq=2,
                ts=2000,
                status="degraded",
                level="debug",
                reason_code="staged_execution_not_ready",
                ids={"cycle_id": "cy_stage_2"},
                data={
                    "details": {
                        "context": "rust order calculation",
                        "defer_reason": "staged_planner_inputs_not_fresh",
                        "missing": ["completed_candles", "market_prices"],
                        "invalid": {
                            "completed_candles": [
                                {
                                    "mismatch_type": "completed_candle_target_changed",
                                    "changed_count": 1,
                                    "changed_symbols": ["WLD/USDT:USDT"],
                                },
                                {
                                    "mismatch_type": "completed_candle_target_changed",
                                    "changed_count": 1,
                                    "changed_symbols": ["AAVE/USDT:USDT"],
                                },
                            ],
                            "market_prices": [
                                {"mismatch_type": "epoch_stale", "missing_count": 1}
                            ],
                        },
                    }
                },
            ),
            _monitor_row(
                event_type="cycle.degraded",
                seq=3,
                ts=3000,
                status="degraded",
                level="error",
                reason_code="InvalidNonce",
                ids={"cycle_id": "cy_nonce"},
                data={"details": {"missing": ["positions"]}},
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    health = report["staged_readiness_health"]
    assert health["total"] == 2
    assert health["bots"] == 1
    assert health["latest_missing_surface_total"] == 2
    assert health["latest_invalid_surface_total"] == 3
    assert health["event_types"] == {"cycle.degraded": 2}
    assert health["groups"][0]["count"] == 2
    assert health["groups"][0]["latest_ids"] == {"cycle_id": "cy_stage_2"}
    assert health["groups"][0]["latest_context"] == "rust order calculation"
    assert health["groups"][0]["latest_missing_surfaces"] == {
        "completed_candles": 1,
        "market_prices": 1,
    }
    assert health["groups"][0]["latest_invalid_surfaces"] == {
        "completed_candles": 2,
        "market_prices": 1,
    }
    assert health["groups"][0]["latest_completed_candle_mismatch_counts"] == {
        "completed_candle_target_changed": 2
    }
    assert summary["staged_readiness_health"]["total"] == 2
    assert summary["staged_readiness_health"]["groups"][0]["latest_ids"] == {
        "cycle_id": "cy_stage_2"
    }
    assert brief["staged_readiness"] == {
        "total": 2,
        "bots": 1,
        "latest_missing_surface_total": 2,
        "latest_invalid_surface_total": 3,
        "event_types": {"cycle.degraded": 2},
    }


def test_live_smoke_report_problem_events_include_state_refresh_progress(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="state.refresh_progress",
                seq=1,
                ts=1000,
                status="degraded",
                level="info",
                reason_code="staged_refresh_progress",
                ids={"cycle_id": "cy_refresh"},
                data={
                    "plan": ["balance", "open_orders", "positions"],
                    "pending": ["positions"],
                    "elapsed_ms": 12_500,
                    "completed_timings_ms": {
                        "balance": 120,
                        "open_orders": 300,
                    },
                    "threshold_s": 10.0,
                    "repeated": False,
                    "secret": "api_key=SECRET",
                },
            )
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    event = report["problem_events"][0]
    assert event["event_type"] == "state.refresh_progress"
    assert event["latest_data"] == {
        "completed_timings_ms": {"balance": 120, "open_orders": 300},
        "elapsed_ms": 12_500,
        "pending": ["positions"],
        "plan": ["balance", "open_orders", "positions"],
        "repeated": False,
        "threshold_s": 10.0,
    }


def test_live_smoke_report_summarizes_startup_phase_baselines(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.startup_timing",
                seq=1,
                ts=1000,
                level="debug",
                reason_code="startup_phase_ready",
                data={
                    "phase": "account",
                    "elapsed_ms": 2000,
                    "since_previous_ms": 2000,
                },
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=2,
                ts=2000,
                level="debug",
                reason_code="startup_phase_ready",
                data={
                    "phase": "startup",
                    "elapsed_ms": 8000,
                    "since_previous_ms": 6000,
                },
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=3,
                ts=3000,
                level="debug",
                reason_code="startup_phase_ready",
                data={
                    "phase": "account",
                    "elapsed_ms": 3000,
                    "since_previous_ms": 3000,
                },
            ),
            _monitor_row(
                event_type="bot.startup_timing",
                seq=4,
                ts=4000,
                level="debug",
                reason_code="startup_phase_ready",
                data={
                    "phase": "startup",
                    "elapsed_ms": 10000,
                    "since_previous_ms": 7000,
                    "details": "ready api_key=AKIA123 Authorization: Bearer TOKEN123",
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    assert report["startup_timings"] == [
        {
            "bot": "binance/binance_01",
            "baseline_window": 20,
            "phases": {
                "account": {
                    "samples": 2,
                    "latest_ts": 3000,
                    "latest_elapsed_ms": 3000,
                    "latest_since_previous_ms": 3000,
                    "elapsed_baseline": {
                        "median_ms": 2500,
                        "p95_ms": 3000,
                        "min_ms": 2000,
                        "max_ms": 3000,
                    },
                    "phase_baseline": {
                        "median_ms": 2500,
                        "p95_ms": 3000,
                        "min_ms": 2000,
                        "max_ms": 3000,
                    },
                    "elapsed_budget": {
                        "status": "over_budget",
                        "latest_ms": 3000,
                        "budget_ms": 2000,
                        "baseline_samples": 1,
                        "usage_pct": 150,
                        "over_budget_by_ms": 1000,
                        "source": "prior_p95_ms",
                    },
                    "phase_budget": {
                        "status": "over_budget",
                        "latest_ms": 3000,
                        "budget_ms": 2000,
                        "baseline_samples": 1,
                        "usage_pct": 150,
                        "over_budget_by_ms": 1000,
                        "source": "prior_p95_ms",
                    },
                    "latest_elapsed_vs_p95_pct": 100,
                    "latest_phase_vs_p95_pct": 100,
                },
                "startup": {
                    "samples": 2,
                    "latest_ts": 4000,
                    "latest_elapsed_ms": 10000,
                    "latest_since_previous_ms": 7000,
                    "elapsed_baseline": {
                        "median_ms": 9000,
                        "p95_ms": 10000,
                        "min_ms": 8000,
                        "max_ms": 10000,
                    },
                    "phase_baseline": {
                        "median_ms": 6500,
                        "p95_ms": 7000,
                        "min_ms": 6000,
                        "max_ms": 7000,
                    },
                    "elapsed_budget": {
                        "status": "over_budget",
                        "latest_ms": 10000,
                        "budget_ms": 8000,
                        "baseline_samples": 1,
                        "usage_pct": 125,
                        "over_budget_by_ms": 2000,
                        "source": "prior_p95_ms",
                    },
                    "phase_budget": {
                        "status": "over_budget",
                        "latest_ms": 7000,
                        "budget_ms": 6000,
                        "baseline_samples": 1,
                        "usage_pct": 117,
                        "over_budget_by_ms": 1000,
                        "source": "prior_p95_ms",
                    },
                    "latest_elapsed_vs_p95_pct": 100,
                    "latest_phase_vs_p95_pct": 100,
                    "latest_details": (
                        "ready api_key=[redacted] Authorization: [redacted]"
                    ),
                },
            },
        }
    ]
    latest_details = report["startup_timings"][0]["phases"]["startup"]["latest_details"]
    assert "AKIA123" not in latest_details
    assert "TOKEN123" not in latest_details
    assert summary["startup_timings"]["bots"] == 1
    assert summary["startup_timings"]["groups"][0]["bot"] == "binance/binance_01"
    assert (
        summary["startup_timings"]["groups"][0]["phases"]["startup"]["latest_details"]
        == "ready api_key=[redacted] Authorization: [redacted]"
    )
    assert brief["startup_timings"] == {
        "bots": 1,
        "phases": 2,
        "over_budget_phases": 2,
        "startup_phase_bots": 1,
        "max_latest_elapsed_ms": 10000,
        "max_latest_phase_ms": 7000,
        "max_startup_elapsed_ms": 10000,
    }


def test_live_smoke_report_startup_budget_no_baseline(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.startup_timing",
                seq=1,
                ts=1000,
                level="debug",
                reason_code="startup_phase_ready",
                data={
                    "phase": "account",
                    "elapsed_ms": 2000,
                    "since_previous_ms": 2000,
                },
            )
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    phase = report["startup_timings"][0]["phases"]["account"]

    assert phase["elapsed_budget"] == {
        "status": "no_baseline",
        "latest_ms": 2000,
        "budget_ms": None,
        "baseline_samples": 0,
        "usage_pct": None,
        "over_budget_by_ms": None,
        "source": "prior_p95_ms",
    }
    assert phase["phase_budget"]["status"] == "no_baseline"


def test_live_smoke_report_summarizes_shutdown_events(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.stopping",
                seq=1,
                ts=1000,
                status="started",
                reason_code="shutdown_gracefully",
                data={"reason": "shutdown_gracefully"},
            ),
            _monitor_row(
                event_type="bot.shutdown.stage",
                seq=2,
                ts=2000,
                status="degraded",
                level="warning",
                reason_code="maintainers_timeout",
                message="shutdown delayed in /Users/alice/passivbot for alice_01",
                data={
                    "stage": "maintainers_timeout",
                    "elapsed_s": 6.25,
                    "task_count": 4,
                    "timeout_s": 5.0,
                    "error": "api_key=AKIA123 Authorization: Bearer TOKEN123",
                },
            ),
            _monitor_row(
                event_type="bot.stopped",
                seq=3,
                ts=3000,
                status="succeeded",
                reason_code="shutdown_gracefully",
                data={"reason": "shutdown_gracefully", "elapsed_s": 7.5},
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report, max_groups=2)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is True
    assert report["attention"] is True
    assert report["shutdown_events"]["total"] == 3
    assert report["shutdown_events"]["event_types"] == {
        "bot.stopping": 1,
        "bot.shutdown.stage": 1,
        "bot.stopped": 1,
    }
    assert [group["event_type"] for group in report["shutdown_events"]["groups"]] == [
        "bot.stopped",
        "bot.shutdown.stage",
        "bot.stopping",
    ]
    stage_group = report["shutdown_events"]["groups"][1]
    assert stage_group["reason_code"] == "maintainers_timeout"
    assert stage_group["status"] == "degraded"
    assert stage_group["latest_data"]["elapsed_s"] == 6.25
    assert stage_group["latest_data"]["task_count"] == 4
    assert "AKIA123" not in stage_group["latest_data"]["error"]
    assert "TOKEN123" not in stage_group["latest_data"]["error"]
    assert "latest_message" not in stage_group
    assert "alice" not in json.dumps(report["shutdown_events"], sort_keys=True)
    assert summary["shutdown_events"]["total"] == 3
    assert summary["shutdown_events"]["groups_truncated"] is True
    assert len(summary["shutdown_events"]["groups"]) == 2
    assert brief["shutdown_events"] == {
        "total": 3,
        "event_types": {
            "bot.stopping": 1,
            "bot.shutdown.stage": 1,
            "bot.stopped": 1,
        },
    }


def test_live_smoke_report_summarizes_recent_risk_events(tmp_path):
    events_dir = tmp_path / "monitor" / "bybit" / "bybit_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.status",
                seq=1,
                ts=1000,
                symbol="ZEC/USDT:USDT",
                pside="long",
                reason_code="green",
                data={
                    "signal_mode": "coin",
                    "tier": "green",
                    "dist_to_red": 0.09,
                    "drawdown_score": 0.01,
                    "red_threshold": 0.10,
                },
            ),
            _monitor_row(
                event_type="hsl.status",
                seq=2,
                ts=2000,
                symbol="ZEC/USDT:USDT",
                pside="long",
                reason_code="yellow",
                ids={"cycle_id": "cy_risk_1"},
                data={
                    "signal_mode": "coin",
                    "tier": "yellow",
                    "dist_to_red": 0.04,
                    "drawdown_score": 0.06,
                    "red_threshold": 0.10,
                },
            ),
            _monitor_row(
                event_type="hsl.status",
                seq=3,
                ts=3000,
                symbol="ZEC/USDT:USDT",
                pside="long",
                reason_code="yellow",
                ids={"cycle_id": "cy_risk_2"},
                data={
                    "signal_mode": "coin",
                    "tier": "yellow",
                    "dist_to_red": 0.02,
                    "drawdown_score": 0.08,
                    "red_threshold": 0.10,
                },
            ),
            _monitor_row(
                event_type="risk.mode_changed",
                seq=4,
                ts=4000,
                pside="long",
                reason_code="hsl_runtime_forced_modes",
                ids={"cycle_id": "cy_risk_3"},
                data={
                    "action": "forced_modes",
                    "previous_mode": "normal",
                    "mode": "panic",
                },
            ),
            _monitor_row(
                event_type="hsl.red_finalized_without_order",
                seq=5,
                ts=4500,
                symbol="ZEC/USDT:USDT",
                pside="long",
                reason_code="hsl_red_finalized_without_exchange_order",
                ids={"cycle_id": "cy_risk_4"},
                data={
                    "no_exchange_close_needed": True,
                    "exchange_close_order_submitted": False,
                    "panic_order_submitted_count": 0,
                    "symbol_position_open": False,
                    "position_count": 0,
                    "entry_orders": 0,
                    "nonpanic_close_orders": 0,
                    "flat_confirmations": 2,
                    "stop_event_timestamp_ms": 4300,
                    "stop_event_anchor_source": "current_time_fallback",
                    "stop_event_anchor_timestamp_ms": 4300,
                    "stop_event_anchor_fallback_used": True,
                    "cooldown_until_ms": 999999,
                    "drawdown_raw": 0.42,
                    "balance": 12345.67,
                    "secret_marker": "flat-secret",
                },
            ),
            _monitor_row(
                event_type="unstuck.selection",
                seq=6,
                ts=5000,
                symbol="SUI/USDT:USDT",
                pside="long",
                reason_code="unstuck_selection",
                ids={"cycle_id": "cy_risk_5"},
                data={
                    "allowance": -12.3,
                    "price_diff_pct": 10.0,
                    "changed": True,
                },
            ),
        ],
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        since_ms=1500,
    )

    assert report["ok"] is True
    assert report["attention"] is False
    assert report["risk_events"] == {
        "total": 5,
        "groups_truncated": False,
        "event_types": {
            "hsl.red_finalized_without_order": 1,
            "hsl.status": 2,
            "risk.mode_changed": 1,
            "unstuck.selection": 1,
        },
        "hsl_flat_finalization_anchors": {
            "total": 1,
            "source_counts": {"current_time_fallback": 1},
            "fallback_used": 1,
            "fallback_used_pct": 100.0,
            "bots": 1,
            "symbols": {
                "count": 1,
                "sample": ["ZEC/USDT:USDT"],
                "truncated": 0,
            },
        },
        "hsl_status": {
            "total": 2,
            "bots": 1,
            "symbols": {
                "count": 1,
                "sample": ["ZEC/USDT:USDT"],
                "truncated": 0,
            },
            "tier_counts": {"yellow": 2},
            "signal_mode_counts": {"coin": 2},
            "closest_to_red": [
                {
                    "bot": "binance/binance_01",
                    "symbol": "ZEC/USDT:USDT",
                    "pside": "long",
                    "tier": "yellow",
                    "signal_mode": "coin",
                    "dist_to_red": 0.02,
                    "red_threshold": 0.1,
                    "latest_ts": 3000,
                },
            ],
            "closest_to_red_truncated": 0,
        },
        "groups": [
            {
                "bot": "binance/binance_01",
                "event_type": "unstuck.selection",
                "reason_code": "unstuck_selection",
                "status": "succeeded",
                "level": "info",
                "symbol": "SUI/USDT:USDT",
                "pside": "long",
                "component": "test",
                "count": 1,
                "latest_ts": 5000,
                "latest_data": {
                    "changed": True,
                    "price_diff_pct": 10.0,
                },
                "latest_ids": {"cycle_id": "cy_risk_5"},
            },
            {
                "bot": "binance/binance_01",
                "event_type": "hsl.red_finalized_without_order",
                "reason_code": "hsl_red_finalized_without_exchange_order",
                "status": "succeeded",
                "level": "info",
                "symbol": "ZEC/USDT:USDT",
                "pside": "long",
                "component": "test",
                "count": 1,
                "latest_ts": 4500,
                "latest_data": {
                    "cooldown_until_ms": 999999,
                    "stop_event_timestamp_ms": 4300,
                    "stop_event_anchor_source": "current_time_fallback",
                    "stop_event_anchor_timestamp_ms": 4300,
                    "stop_event_anchor_fallback_used": True,
                    "no_exchange_close_needed": True,
                    "exchange_close_order_submitted": False,
                    "panic_order_submitted_count": 0,
                    "symbol_position_open": False,
                    "position_count": 0,
                    "entry_orders": 0,
                    "nonpanic_close_orders": 0,
                    "flat_confirmations": 2,
                },
                "latest_ids": {"cycle_id": "cy_risk_4"},
            },
            {
                "bot": "binance/binance_01",
                "event_type": "risk.mode_changed",
                "reason_code": "hsl_runtime_forced_modes",
                "status": "succeeded",
                "level": "info",
                "pside": "long",
                "component": "test",
                "count": 1,
                "latest_ts": 4000,
                "latest_data": {
                    "action": "forced_modes",
                    "mode": "panic",
                    "previous_mode": "normal",
                },
                "latest_ids": {"cycle_id": "cy_risk_3"},
            },
            {
                "bot": "binance/binance_01",
                "event_type": "hsl.status",
                "reason_code": "yellow",
                "status": "succeeded",
                "level": "info",
                "symbol": "ZEC/USDT:USDT",
                "pside": "long",
                "component": "test",
                "count": 2,
                "latest_ts": 3000,
                "latest_data": {
                    "signal_mode": "coin",
                    "tier": "yellow",
                    "drawdown_score": 0.08,
                    "dist_to_red": 0.02,
                    "red_threshold": 0.10,
                },
                "latest_ids": {"cycle_id": "cy_risk_2"},
            },
        ],
    }
    rendered = json.dumps(report["risk_events"], sort_keys=True)
    assert "12345.67" not in rendered
    assert "0.42" not in rendered
    assert "flat-secret" not in rendered
    summary = summarize_live_smoke_report(report)
    summary_risk_json = json.dumps(summary["risk_events"], sort_keys=True)
    assert "dist_to_red" not in summary_risk_json
    assert "red_threshold" not in summary_risk_json
    assert "drawdown_raw" not in summary_risk_json
    assert "price_diff_pct" not in summary_risk_json
    assert summary["risk_events"]["hsl_flat_finalization_anchors"] == {
        "total": 1,
        "source_counts": {"current_time_fallback": 1},
        "fallback_used": 1,
        "fallback_used_pct": 100.0,
        "bots": 1,
        "symbols": {
            "count": 1,
            "sample": ["ZEC/USDT:USDT"],
            "truncated": 0,
        },
    }
    assert summary["risk_events"]["hsl_status"]["closest_to_red"] == [
        {
            "bot": "binance/binance_01",
            "symbol": "ZEC/USDT:USDT",
            "pside": "long",
            "tier": "yellow",
            "signal_mode": "coin",
            "latest_ts": 3000,
        }
    ]
    brief = summarize_live_smoke_report_brief(report)
    assert brief["risk_events"]["hsl_flat_finalization_anchors"] == {
        "total": 1,
        "source_counts": {"current_time_fallback": 1},
        "fallback_used": 1,
        "fallback_used_pct": 100.0,
        "bots": 1,
        "symbols": {
            "count": 1,
            "sample": ["ZEC/USDT:USDT"],
            "truncated": 0,
        },
    }
    assert brief["risk_events"]["hsl_status"] == {
        "total": 2,
        "bots": 1,
        "symbols": {
            "count": 1,
            "sample": ["ZEC/USDT:USDT"],
            "truncated": 0,
        },
        "tier_counts": {"yellow": 2},
        "signal_mode_counts": {"coin": 2},
        "closest_to_red": [
            {
                "bot": "binance/binance_01",
                "symbol": "ZEC/USDT:USDT",
                "pside": "long",
                "tier": "yellow",
                "signal_mode": "coin",
                "latest_ts": 3000,
            }
        ],
        "closest_to_red_truncated": 0,
    }
    assert "dist_to_red" not in json.dumps(summary["risk_events"]["hsl_status"])
    assert "red_threshold" not in json.dumps(brief["risk_events"]["hsl_status"])


def test_live_smoke_report_summarizes_hsl_replay_health(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke_report_module, "utc_ms", lambda: 365000)
    _write_ndjson(
        tmp_path
        / "monitor"
        / "binance"
        / "binance_01"
        / "events"
        / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=1,
                ts=1000,
                reason_code="coin_history_replay",
                level="debug",
                status="started",
                data={"signal_mode": "coin", "lookback_days": 30.0},
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=2,
                ts=2000,
                reason_code="history_loaded",
                level="debug",
                status="started",
                data={
                    "signal_mode": "coin",
                    "stage": "loaded",
                    "symbols": 26,
                    "pairs": 26,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 20,
                    "timeline_rows": 43201,
                    "fill_events": 2700,
                    "secret": "must-not-render",
                },
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=3,
                ts=3000,
                reason_code="coin_history_replay_completed",
                level="debug",
                status="succeeded",
                data={
                    "signal_mode": "coin",
                    "stage": "full_replay",
                    "rows": 985965,
                    "pairs": 26,
                    "timeline_rows": 43201,
                    "full_elapsed_s": 1623.4,
                    "startup_blocking_elapsed_s": 1623.4,
                },
            ),
        ],
    )
    _write_ndjson(
        tmp_path
        / "monitor"
        / "gateio"
        / "gateio_01"
        / "events"
        / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=4,
                ts=4000,
                exchange="gateio",
                user="gateio_01",
                reason_code="coin_history_replay",
                level="debug",
                status="started",
                data={"signal_mode": "coin", "lookback_days": 30.0},
            ),
            _monitor_row(
                event_type="hsl.replay.progress",
                seq=5,
                ts=5000,
                exchange="gateio",
                user="gateio_01",
                reason_code="pair_replay_progress",
                level="debug",
                status="started",
                symbol="ZEC/USDT:USDT",
                pside="long",
                data={
                    "signal_mode": "coin",
                    "stage": "pair_replay",
                    "pair_idx": 3,
                    "pairs": 29,
                    "held_pairs": 1,
                    "cooldown_pairs": 1,
                    "required_pairs": 20,
                    "current_position_pairs": 1,
                    "timeline_rows": 43201,
                    "start_ts": 1782492000000,
                    "end_ts": 1782492600000,
                    "record_start_ts": 1782492000000,
                    "applied_rows": 12000,
                    "total_applied_rows": 64000,
                    "skipped_price_symbols": 1,
                    "missing_price_symbols": 2,
                    "rows_per_second": 318.415,
                    "elapsed_s": 701.2,
                    "history_build_elapsed_s": 755.75,
                    "price_history_fetch_elapsed_s": 210.5,
                    "timeline_replay_elapsed_s": 12.25,
                    "timeframe": "1m",
                    "history_minutes": 43201,
                    "price_replay_symbols": 29,
                    "is_held_pair": True,
                    "is_cooldown_pair": False,
                    "balance": 1234.56,
                    "equity": 1200.0,
                    "drawdown_score": 0.42,
                    "error": "api_key=AKIA123",
                },
            ),
        ],
    )
    _write_ndjson(
        tmp_path
        / "monitor"
        / "okx"
        / "okx_01"
        / "events"
        / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=6,
                ts=6000,
                exchange="okx",
                user="okx_01",
                reason_code="coin_history_replay",
                level="debug",
                status="started",
                data={"signal_mode": "coin", "lookback_days": 30.0},
            ),
            _monitor_row(
                event_type="hsl.replay.failed",
                seq=7,
                ts=7000,
                exchange="okx",
                user="okx_01",
                reason_code="coin_history_replay_failed",
                level="warning",
                status="failed",
                data={
                    "signal_mode": "coin",
                    "error_type": "RuntimeError",
                    "elapsed_s": 12.3,
                    "secret": "must-not-render",
                },
            ),
        ],
    )
    _write_ndjson(
        tmp_path
        / "monitor"
        / "kucoin"
        / "kucoin_01"
        / "events"
        / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=8,
                ts=8000,
                exchange="kucoin",
                user="kucoin_01",
                reason_code="coin_history_replay",
                level="debug",
                status="started",
                data={"signal_mode": "coin", "lookback_days": 30.0},
            ),
            _monitor_row(
                event_type="hsl.replay.failed",
                seq=9,
                ts=9000,
                exchange="kucoin",
                user="kucoin_01",
                reason_code="shutdown_cancelled",
                level="debug",
                status="failed",
                data={
                    "signal_mode": "coin",
                    "elapsed_s": 1.0,
                },
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is True
    assert report["attention"] is True
    assert report["hard_failures"] == 0
    assert report["attention_sources"]["problem_events"] == 1
    assert report["attention_sources"]["hsl_replay_active_bots"] == 1
    assert report["attention_sources"]["hsl_replay_failed_bots"] == 1
    assert report["attention_sources"]["total"] == 3
    assert report["hsl_replay_health"]["active_bots"] == 1
    assert report["hsl_replay_health"]["stale_active_bots"] == 1
    assert report["hsl_replay_health"]["long_running_active_bots"] == 1
    assert report["hsl_replay_health"]["completed_bots"] == 1
    assert report["hsl_replay_health"]["failed_bots"] == 2
    assert report["hsl_replay_health"]["failed_attention_bots"] == 1
    assert report["hsl_replay_health"]["event_types"] == {
        "hsl.replay.progress": 2,
        "hsl.replay.started": 4,
        "hsl.replay.completed": 1,
        "hsl.replay.failed": 2,
    }
    active_group = report["hsl_replay_health"]["groups"][0]
    assert active_group["bot"] == "gateio/gateio_01"
    assert active_group["active"] is True
    assert active_group["active_latest_event_age_ms"] == 360000
    assert active_group["active_stale"] is True
    assert active_group["active_stale_threshold_ms"] == 300000
    assert active_group["active_long_running"] is True
    assert active_group["active_long_running_threshold_ms"] == 600000
    assert active_group["latest"]["symbol"] == "ZEC/USDT:USDT"
    assert active_group["latest"]["data"]["timeframe"] == "1m"
    assert active_group["latest"]["data"]["history_minutes"] == 43201
    assert active_group["latest"]["data"]["price_replay_symbols"] == 29
    assert active_group["latest"]["data"]["current_position_pairs"] == 1
    assert active_group["latest"]["data"]["skipped_price_symbols"] == 1
    assert active_group["latest"]["data"]["missing_price_symbols"] == 2
    assert active_group["latest"]["data"]["start_ts"] == 1782492000000
    assert active_group["latest"]["data"]["end_ts"] == 1782492600000
    assert active_group["latest"]["data"]["record_start_ts"] == 1782492000000
    assert active_group["latest"]["derived"]["history_build_elapsed_ms"] == 755750
    assert active_group["latest"]["derived"]["latest_event_age_ms"] == 360000
    assert active_group["latest"]["derived"]["price_history_fetch_elapsed_ms"] == 210500
    assert active_group["latest"]["derived"]["timeline_replay_elapsed_ms"] == 12250
    assert active_group["latest"]["derived"]["estimated_dense_pair_row_work"] == (
        43201 * 29
    )
    assert active_group["latest"]["derived"]["estimated_held_pair_row_work"] == 43201
    assert active_group["latest"]["derived"]["estimated_cooldown_pair_row_work"] == 43201
    assert active_group["latest"]["derived"]["estimated_required_pair_row_work"] == (
        43201 * 20
    )
    assert active_group["latest"]["derived"]["estimated_dense_remaining_rows"] == (
        43201 * 29 - 64000
    )
    assert active_group["latest"]["derived"]["estimated_required_remaining_rows"] == (
        43201 * 20 - 64000
    )
    assert active_group["latest"]["derived"]["estimated_remaining_rows"] == (
        43201 * 20 - 64000
    )
    assert active_group["latest"]["derived"]["estimated_dense_remaining_ms"] == 3733584
    assert active_group["latest"]["derived"]["estimated_required_remaining_ms"] == 2512507
    assert active_group["latest"]["derived"]["estimated_remaining_ms"] == 2512507
    assert active_group["latest"]["derived"]["observed_work_pct"] == pytest.approx(
        5.108
    )
    assert active_group["latest"]["derived"]["observed_required_work_pct"] == pytest.approx(
        7.407
    )
    assert "AKIA123" not in json.dumps(report["hsl_replay_health"], sort_keys=True)
    assert "must-not-render" not in json.dumps(
        report["hsl_replay_health"],
        sort_keys=True,
    )
    assert "balance" not in json.dumps(report["hsl_replay_health"], sort_keys=True)
    assert "equity" not in json.dumps(report["hsl_replay_health"], sort_keys=True)
    assert "drawdown_score" not in json.dumps(
        report["hsl_replay_health"],
        sort_keys=True,
    )
    assert summary["hsl_replay_health"]["active_bots"] == 1
    assert summary["hsl_replay_health"]["stale_active_bots"] == 1
    assert summary["hsl_replay_health"]["long_running_active_bots"] == 1
    assert summary["hsl_replay_health"]["groups"][0]["active"] is True
    failed_group = next(
        group
        for group in report["hsl_replay_health"]["groups"]
        if group["bot"] == "okx/okx_01"
    )
    assert failed_group["active"] is False
    assert failed_group["failed"]["event_type"] == "hsl.replay.failed"
    assert failed_group["failed"]["status"] == "failed"
    shutdown_group = next(
        group
        for group in report["hsl_replay_health"]["groups"]
        if group["bot"] == "kucoin/kucoin_01"
    )
    assert shutdown_group["active"] is False
    assert shutdown_group["failed"]["reason_code"] == "shutdown_cancelled"
    assert brief["hsl_replay"] == {
        "total": 9,
        "bots": 4,
        "active_bots": 1,
        "stale_active_bots": 1,
        "long_running_active_bots": 1,
        "completed_bots": 1,
        "failed_bots": 2,
        "failed_attention_bots": 1,
        "max_active_latest_elapsed_ms": 755750,
        "max_active_latest_event_age_ms": 360000,
        "max_active_estimated_remaining_rows": 800020,
        "max_active_estimated_remaining_ms": 2512507,
        "max_completed_elapsed_ms": 1623400,
        "active_stage_counts": {"pair_replay": 1},
        "event_types": {
            "hsl.replay.progress": 2,
            "hsl.replay.started": 4,
            "hsl.replay.completed": 1,
            "hsl.replay.failed": 2,
        },
    }


def test_live_smoke_report_hsl_replay_latest_failure_overrides_stale_completion(tmp_path):
    _write_ndjson(
        tmp_path
        / "monitor"
        / "binance"
        / "binance_01"
        / "events"
        / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.replay.started",
                seq=1,
                ts=1000,
                reason_code="coin_history_replay",
                level="debug",
                status="started",
                data={"signal_mode": "coin"},
            ),
            _monitor_row(
                event_type="hsl.replay.completed",
                seq=2,
                ts=2000,
                reason_code="coin_history_replay_completed",
                level="debug",
                status="succeeded",
                data={"signal_mode": "coin", "stage": "full_replay"},
            ),
            _monitor_row(
                event_type="hsl.replay.failed",
                seq=3,
                ts=4000,
                reason_code="coin_history_replay_failed",
                level="warning",
                status="failed",
                data={"signal_mode": "coin", "error_type": "RuntimeError"},
            ),
        ],
    )

    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    health = report["hsl_replay_health"]

    assert health["active_bots"] == 0
    assert health["completed_bots"] == 0
    assert health["failed_bots"] == 1
    assert health["failed_attention_bots"] == 1
    assert report["attention_sources"]["hsl_replay_failed_bots"] == 1
    assert health["groups"][0]["latest"]["event_type"] == "hsl.replay.failed"
    assert health["groups"][0]["completed"]["event_type"] == "hsl.replay.completed"


def test_live_smoke_report_distinguishes_attention_and_hard_structured_events(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "bybit" / "bybit_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="request_timeout",
            ),
            _monitor_row(
                event_type="bot.stopped",
                seq=2,
                ts=1100,
                status="failed",
                level="critical",
                reason_code="terminal_startup_failure",
            ),
            _monitor_row(
                event_type="hsl.red_triggered",
                seq=3,
                ts=1200,
                status="succeeded",
                level="info",
                reason_code="coin_red_stop_finalized",
                symbol="ZEC/USDT:USDT",
                pside="long",
                data={
                    "no_exchange_close_needed": True,
                    "exchange_close_order_submitted": False,
                    "panic_order_submitted_count": 0,
                    "symbol_position_open": False,
                },
            ),
        ],
    )

    attention_only = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        max_problem_events=1,
    )

    assert attention_only["ok"] is False
    assert attention_only["hard_problem_event_count"] == 1
    assert attention_only["problem_events"] == [
        {
            "event_type": "bot.stopped",
            "exchange": "binance",
            "hard": True,
            "level": "critical",
            "line": 2,
            "path": str(events_dir / "current.ndjson"),
            "reason_code": "terminal_startup_failure",
            "seq": 2,
            "status": "failed",
            "ts": 1100,
            "user": "binance_01",
        }
    ]

    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="request_timeout",
            )
        ],
    )
    report = build_live_smoke_report(tmp_path / "monitor", logs_root=None)

    assert report["ok"] is True
    assert report["attention"] is True
    assert report["hard_failures"] == 0
    assert report["hard_problem_event_count"] == 0
    assert report["problem_events"][0]["hard"] is False


def test_live_smoke_report_summarizes_problem_event_groups(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="ema.unavailable",
                seq=1,
                ts=1000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_ema_1"},
                data={
                    "candidate_unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 1,
                        "sample": ["ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                },
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=2,
                ts=2000,
                status="degraded",
                level="warning",
                reason_code="required_ema_unavailable",
                ids={"cycle_id": "cy_ema_2"},
                data={
                    "candidate_unavailable": {
                        "count": 2,
                        "sample": ["XRP/USDT:USDT", "ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 2,
                        "sample": ["XRP/USDT:USDT", "ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                },
            ),
            _monitor_row(
                event_type="bot.stopped",
                seq=3,
                ts=3000,
                status="failed",
                level="critical",
                reason_code="terminal_startup_failure",
                ids={"cycle_id": "cy_terminal"},
            ),
        ],
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        max_problem_events=1,
    )

    assert report["problem_event_count"] == 3
    assert len(report["problem_events"]) == 1
    assert report["problem_event_groups"] == {
        "total": 3,
        "groups_truncated": False,
        "event_types": {"ema.unavailable": 2, "bot.stopped": 1},
        "groups": [
            {
                "bot": "binance/binance_01",
                "event_type": "bot.stopped",
                "reason_code": "terminal_startup_failure",
                "status": "failed",
                "level": "critical",
                "hard": True,
                "component": "test",
                "count": 1,
                "latest_ts": 3000,
                "latest_ids": {"cycle_id": "cy_terminal"},
            },
            {
                "bot": "binance/binance_01",
                "event_type": "ema.unavailable",
                "reason_code": "required_ema_unavailable",
                "status": "degraded",
                "level": "warning",
                "hard": False,
                "component": "test",
                "count": 2,
                "latest_ts": 2000,
                "latest_data": {
                    "candidate_unavailable": {
                        "count": 2,
                        "sample": ["XRP/USDT:USDT", "ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                    "unavailable": {
                        "count": 2,
                        "sample": ["XRP/USDT:USDT", "ZEC/USDT:USDT"],
                        "truncated": 0,
                    },
                },
                "latest_ids": {"cycle_id": "cy_ema_2"},
            },
        ],
    }


def test_live_smoke_report_time_window_filters_structured_problem_events(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="bot.stopped",
                seq=1,
                ts=1000,
                status="failed",
                level="critical",
                reason_code="old_terminal_failure",
            ),
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=2000,
                status="failed",
                level="warning",
                reason_code="old_timeout",
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=3,
                ts=3000,
                ids={"cycle_id": "cy_fresh"},
            ),
        ],
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        since_ms=2500,
    )

    assert report["ok"] is True
    assert report["monitor"]["live_events"] == 3
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 2500,
        "until_ms": None,
        "events_considered": 1,
        "events_skipped_before": 2,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
    }
    assert report["bots"] == [
        {
            "bot": "binance/binance_01",
            "events": 1,
            "event_types": {"cycle.completed": 1},
            "invalid_ts": 0,
            "last_ts": 3000,
            "levels": {"info": 1},
            "hard_problem_events": 0,
            "problem_events": 0,
            "statuses": {"succeeded": 1},
        }
    ]
    assert report["hard_problem_event_count"] == 0
    assert report["problem_events"] == []
    assert report["remote_call_failures"] == {
        "total": 0,
        "groups_truncated": False,
        "groups": [],
    }
    assert report["remote_call_health"] == {
        "total": 0,
        "succeeded": 0,
        "failed": 0,
        "throttled": 0,
        "failure_pct": None,
        "throttled_pct": None,
        "groups_truncated": False,
        "groups": [],
    }
    assert report["remote_call_timings"] == {
        "total": 0,
        "groups_truncated": False,
        "groups": [],
    }


def test_live_smoke_report_log_scan_deduplicates_aliases_and_matches_levels(
    tmp_path,
):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    log_file = logs_dir / "timestamped.log"
    log_file.write_text(
        "\n".join(
            [
                "WARNING error=bybit should not match as log level",
                "ERROR:root: boom",
                "[ERROR] bracketed boom",
                "level=error lower-case field",
                (
                    "ERROR api_key=AKIA123 secret=hunter2 "
                    "Authorization: Bearer TOKEN123"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "okx_faisal.log").symlink_to(log_file)

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        max_log_files=10,
        log_tail_lines=10,
    )

    assert report["ok"] is True
    assert report["logs"]["max_files"] == 10
    assert report["logs"]["tail_lines"] == 10
    assert report["logs"]["max_matches"] == 50
    assert report["logs"]["files_scanned"] == 1
    assert report["logs"]["attention_matches"] == 4
    assert report["logs"]["hard_matches"] == 0
    assert [match["text"] for match in report["logs"]["matches"]] == [
        "ERROR:root: boom",
        "[ERROR] bracketed boom",
        "level=error lower-case field",
        "ERROR api_key=[redacted] secret=[redacted] Authorization: [redacted]",
    ]
    emitted = json.dumps(report["logs"]["matches"])
    assert "AKIA123" not in emitted
    assert "hunter2" not in emitted
    assert "TOKEN123" not in emitted

    summary = summarize_live_smoke_report(report)
    assert summary["logs"]["max_files"] == 10
    assert summary["logs"]["tail_lines"] == 10
    assert summary["logs"]["max_matches"] == 50
    brief = summarize_live_smoke_report_brief(report)
    assert brief["logs"]["max_files"] == 10
    assert brief["logs"]["tail_lines"] == 10
    assert brief["logs"]["max_matches"] == 50


def test_live_smoke_report_log_scan_classifies_risk_matches(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "binance_01.log").write_text(
        "\n".join(
            [
                (
                    "2026-06-27T16:06:24Z CRITICAL [binance] [risk] "
                    "HSL[long:XLM/USDT:USDT] reconstructed active coin RED cooldown"
                ),
                "2026-06-27T16:06:25Z CRITICAL uncaught task failure",
                "2026-06-27T16:06:26Z ERROR [fills] temporary fetch failed",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        log_tail_lines=10,
    )
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is False
    assert report["hard_failures"] == 2
    assert report["logs"]["attention_matches"] == 3
    assert report["logs"]["hard_matches"] == 2
    assert report["logs"]["risk_attention_matches"] == 1
    assert report["logs"]["risk_hard_matches"] == 1
    assert report["logs"]["non_risk_attention_matches"] == 2
    assert report["logs"]["non_risk_hard_matches"] == 1
    assert [match["category"] for match in report["logs"]["matches"]] == [
        "risk",
        "general",
        "general",
    ]
    assert summary["logs"]["risk_hard_matches"] == 1
    assert summary["logs"]["non_risk_hard_matches"] == 1
    assert brief["logs"]["risk_hard_matches"] == 1
    assert brief["logs"]["non_risk_hard_matches"] == 1


def test_live_smoke_report_log_window_filters_parseable_timestamps(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "okx_faisal.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:01Z ERROR stale before window",
                "1970-01-01T00:00:03Z ERROR fresh in window",
                "ERROR unparseable timestamp kept visible",
                "unparseable timestamp noise",
                "1970-01-01T00:00:05Z CRITICAL future hard skipped",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        since_ms=2000,
        until_ms=4000,
        log_tail_lines=10,
    )

    assert report["ok"] is True
    assert report["logs"]["attention_matches"] == 2
    assert report["logs"]["hard_matches"] == 0
    assert report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 3,
        "lines_skipped_before": 1,
        "lines_skipped_after": 1,
        "unparsed_ts": 2,
        "unparsed_policy": "keep",
        "lines_skipped_unparsed": 0,
        "dropped_unparsed_attention_matches": 0,
        "dropped_unparsed_hard_matches": 0,
    }
    assert [match["text"] for match in report["logs"]["matches"]] == [
        "1970-01-01T00:00:03Z ERROR fresh in window",
        "ERROR unparseable timestamp kept visible",
    ]
    assert report["logs"]["matches"][0]["ts"] == 3000
    assert "future hard skipped" not in json.dumps(report["logs"]["matches"])
    brief = summarize_live_smoke_report_brief(report)
    assert brief["logs"]["window"] == report["logs"]["window"]

    drop_report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        since_ms=2000,
        until_ms=4000,
        log_tail_lines=10,
        log_window_unparsed_policy="drop",
    )

    assert drop_report["ok"] is True
    assert drop_report["logs"]["attention_matches"] == 2
    assert drop_report["logs"]["hard_matches"] == 0
    assert drop_report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 2,
        "lines_skipped_before": 1,
        "lines_skipped_after": 1,
        "unparsed_ts": 2,
        "unparsed_policy": "drop",
        "lines_skipped_unparsed": 1,
        "dropped_unparsed_attention_matches": 0,
        "dropped_unparsed_hard_matches": 0,
    }
    assert [match["text"] for match in drop_report["logs"]["matches"]] == [
        "1970-01-01T00:00:03Z ERROR fresh in window",
        "ERROR unparseable timestamp kept visible",
    ]


def test_live_smoke_report_log_window_drop_preserves_unparseable_hard_signal(tmp_path):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "okx_faisal.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:03Z ERROR exchange call failed",
                "Traceback (most recent call last):",
                "  File \"/tmp/passivbot.py\", line 1, in run",
                "old unparseable non-signal noise",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        since_ms=2000,
        until_ms=4000,
        log_tail_lines=10,
        log_window_unparsed_policy="drop",
    )

    assert report["ok"] is False
    assert report["logs"]["attention_matches"] == 2
    assert report["logs"]["hard_matches"] == 1
    assert report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 2,
        "lines_skipped_before": 0,
        "lines_skipped_after": 0,
        "unparsed_ts": 3,
        "unparsed_policy": "drop",
        "lines_skipped_unparsed": 2,
        "dropped_unparsed_attention_matches": 0,
        "dropped_unparsed_hard_matches": 0,
    }
    assert [match["text"] for match in report["logs"]["matches"]] == [
        "1970-01-01T00:00:03Z ERROR exchange call failed",
        "Traceback (most recent call last):",
    ]
    assert report["logs"]["matches"][1]["context_ts"] == 3000
    assert report["logs"]["matches"][1]["context_line"] == 1
    assert (
        report["logs"]["matches"][1]["context_text"]
        == "1970-01-01T00:00:03Z ERROR exchange call failed"
    )


def test_live_smoke_report_log_window_drops_stale_traceback_with_context(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kucoin_01.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:01Z ERROR old exchange call failed",
                "Traceback (most recent call last):",
                "1970-01-01T00:00:03Z INFO recovered",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        since_ms=2000,
        until_ms=4000,
        log_tail_lines=10,
        log_window_unparsed_policy="drop",
    )

    assert report["ok"] is True
    assert report["logs"]["attention_matches"] == 0
    assert report["logs"]["hard_matches"] == 0
    assert report["logs"]["matches"] == []
    assert report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 1,
        "lines_skipped_before": 2,
        "lines_skipped_after": 0,
        "unparsed_ts": 1,
        "unparsed_policy": "drop",
        "lines_skipped_unparsed": 0,
        "dropped_unparsed_attention_matches": 0,
        "dropped_unparsed_hard_matches": 0,
    }


def test_live_smoke_report_log_window_drops_contextless_tailed_traceback(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kucoin_01.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:01Z ERROR old exchange call failed",
                "old stack line before tail",
                "Traceback (most recent call last):",
                "  File \"/tmp/passivbot.py\", line 1, in run",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        since_ms=2000,
        until_ms=4000,
        log_tail_lines=2,
        log_window_unparsed_policy="drop",
    )

    assert report["ok"] is True
    assert report["attention"] is True
    assert report["attention_count"] == 1
    assert report["logs"]["attention_matches"] == 0
    assert report["logs"]["hard_matches"] == 0
    assert report["logs"]["dropped_unparsed_attention_matches"] == 1
    assert report["logs"]["dropped_unparsed_hard_matches"] == 1
    assert report["logs"]["matches"] == []
    assert report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 0,
        "lines_skipped_before": 0,
        "lines_skipped_after": 0,
        "unparsed_ts": 2,
        "unparsed_policy": "drop",
        "lines_skipped_unparsed": 2,
        "dropped_unparsed_attention_matches": 1,
        "dropped_unparsed_hard_matches": 1,
    }


def test_live_smoke_report_log_scan_ignores_traceback_prose(tmp_path):
    events_dir = tmp_path / "monitor" / "kucoin" / "kucoin_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "kucoin_01.log").write_text(
        "\n".join(
            [
                (
                    "2026-06-26T02:31:46Z WARNING [kucoin] [ws] "
                    "websocket callback ping timeout; suppressing callback "
                    "traceback and relying on reconnect"
                ),
                "Traceback (most recent call last):",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=logs_dir,
        log_tail_lines=10,
    )

    assert report["ok"] is False
    assert report["logs"]["attention_matches"] == 1
    assert report["logs"]["hard_matches"] == 1
    assert [match["text"] for match in report["logs"]["matches"]] == [
        "Traceback (most recent call last):"
    ]


def test_live_smoke_report_default_logs_root_follows_monitor_root(tmp_path, capsys):
    bot_root = tmp_path / "bot"
    events_dir = bot_root / "monitor" / "gateio" / "gateio_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = bot_root / "logs"
    logs_dir.mkdir()
    (logs_dir / "gateio_01.log").write_text(
        "2026-06-25T00:00:00Z ERROR account fetch failed\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / "logs"
    unrelated.mkdir()
    (unrelated / "wrong.log").write_text(
        "2026-06-25T00:00:00Z CRITICAL wrong cwd log\n",
        encoding="utf-8",
    )

    assert default_logs_root_for_monitor(bot_root / "monitor") == logs_dir
    assert (
        live_smoke_report.main([str(bot_root / "monitor"), "--compact"])
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["logs"]["root"] == str(logs_dir)
    assert report["logs"]["attention_matches"] == 1
    assert report["logs"]["matches"][0]["path"] == str(logs_dir / "gateio_01.log")


def test_live_smoke_report_cli_outputs_json_and_can_skip_logs(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--since-ms",
                "1001",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["event_window"] == {
        "enabled": True,
        "since_ms": 1001,
        "until_ms": None,
        "events_considered": 0,
        "events_skipped_before": 1,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
    }
    assert report["bots"] == []
    assert report["logs"]["files_scanned"] == 0
    assert report["logs"]["max_files"] == 8
    assert report["logs"]["tail_lines"] == 300
    assert report["logs"]["max_matches"] == 50
    assert report["logs"]["root"] is None
    assert report["monitor"]["live_events"] == 1


def test_live_smoke_report_cli_can_emit_concise_summary(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "okx_faisal.log").write_text(
        "2026-06-25T00:00:00Z ERROR exchange timeout\n",
        encoding="utf-8",
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                str(logs_dir),
                "--log-tail-lines",
                "10",
                "--summary",
                "--compact",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["attention"] is True
    assert summary["logs"]["attention_matches"] == 1
    assert summary["monitor"]["live_events"] == 1
    assert "remote_calls" in summary
    assert "account_critical_remote_calls" in summary
    assert "bots" not in summary
    assert "problem_events" in summary


def test_live_smoke_report_cli_can_emit_brief_summary(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="authoritative_balance",
                data={
                    "kind": "authoritative_state_fetch",
                    "surface": "balance",
                    "error_type": "RequestTimeout",
                },
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "okx_faisal.log").write_text(
        "2026-06-25T00:00:00Z ERROR exchange timeout\n",
        encoding="utf-8",
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                str(logs_dir),
                "--log-tail-lines",
                "10",
                "--brief",
                "--compact",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["attention"] is True
    assert summary["attention_count"] == 2
    assert summary["logs"]["attention_matches"] == 1
    assert summary["problem_events"]["hard"] == 0
    assert summary["problem_events"]["total"] == 1
    assert summary["problem_events"]["event_types"] == {"remote_call.failed": 1}
    assert summary["problem_events"]["event_types_truncated"] is False
    assert summary["problem_events"]["groups"] == [
        {
            "bot": "binance/binance_01",
            "event_type": "remote_call.failed",
            "reason_code": "authoritative_balance",
            "status": "failed",
            "level": "warning",
            "hard": False,
            "component": "test",
            "count": 1,
            "latest_ts": 1000,
        }
    ]
    assert summary["account_critical_remote_calls"]["failed"] == 1
    assert "bots" not in summary
    assert "matches" not in summary["logs"]
    assert "groups" not in summary["remote_calls"]
    assert "groups" not in summary["account_critical_remote_calls"]


def test_live_smoke_report_section_projection_keeps_common_metadata(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=1,
                ts=1000,
                status="succeeded",
                reason_code="fills_refresh_succeeded",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "history_scope": "all",
                    "coverage_ready_after": True,
                    "elapsed_ms": 25,
                },
            )
        ],
    )
    report = summarize_live_smoke_report_brief(
        build_live_smoke_report(tmp_path / "monitor", logs_root=None)
    )

    projected = project_live_smoke_report_sections(report, ["fill_refresh"])

    assert projected["ok"] is True
    assert projected["attention"] is False
    assert projected["monitor"]["live_events"] == 1
    assert projected["fill_refresh"]["total"] == 1
    assert "logs" not in projected
    assert "remote_calls" not in projected
    assert "processes" not in projected


def test_live_smoke_report_cli_brief_section_filter(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="fills.refresh_summary",
                seq=1,
                ts=1000,
                status="succeeded",
                reason_code="fills_refresh_succeeded",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "history_scope": "all",
                    "coverage_ready_after": True,
                    "elapsed_ms": 25,
                },
            )
        ],
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--brief",
                "--section",
                "fill_refresh",
                "--compact",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["ok"] is True
    assert summary["fill_refresh"]["total"] == 1
    assert "logs" not in summary
    assert "remote_calls" not in summary
    assert "processes" not in summary


def test_live_smoke_report_cli_rejects_unknown_section(capsys):
    with pytest.raises(SystemExit):
        live_smoke_report.main(["monitor", "--brief", "--section", "not_a_section"])

    assert "unknown --section value" in capsys.readouterr().err


def test_live_smoke_report_cli_brief_projects_event_tail_metadata(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_old"},
            ),
            _monitor_row(
                event_type="cycle.completed",
                seq=2,
                ts=2000,
                ids={"cycle_id": "cy_fresh"},
            ),
        ],
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--event-tail-lines",
                "1",
                "--brief",
                "--compact",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["monitor"]["live_events"] == 1
    assert summary["monitor"]["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 1,
        "event_segments": 1,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 0,
        "scope_pruned": 0,
    }
    assert summary["event_window"]["enabled"] is False
    assert summary["event_window"]["event_tail_lines"] == 1
    assert summary["event_window"]["event_tail_limited_files"] == 1
    assert summary["event_window"]["event_tail_skipped_lines"] == 1


def test_live_smoke_report_cli_can_drop_unparseable_window_log_lines(
    tmp_path,
    capsys,
):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=3000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "okx_faisal.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:03Z ERROR fresh in window",
                "stale unparseable noise dropped",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        live_smoke_report.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                str(logs_dir),
                "--since-ms",
                "2000",
                "--until-ms",
                "4000",
                "--log-tail-lines",
                "10",
                "--log-window-unparsed-policy",
                "drop",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["logs"]["attention_matches"] == 1
    assert report["logs"]["window"]["unparsed_policy"] == "drop"
    assert report["logs"]["window"]["lines_skipped_unparsed"] == 1
    assert [match["text"] for match in report["logs"]["matches"]] == [
        "1970-01-01T00:00:03Z ERROR fresh in window"
    ]


def test_live_smoke_report_cli_rejects_invalid_window_timestamp(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_smoke_report.main(["monitor", "--since-ms", "not-an-int"])

    assert exc_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_live_smoke_report_cli_rejects_negative_event_tail_lines(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_smoke_report.main(["monitor", "--event-tail-lines", "-1"])

    assert exc_info.value.code == 2
    assert "--event-tail-lines must be non-negative" in capsys.readouterr().err


def test_live_smoke_report_cli_rejects_negative_max_event_files_per_bot(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_smoke_report.main(["monitor", "--max-event-files-per-bot", "-1"])

    assert exc_info.value.code == 2
    assert "--max-event-files-per-bot must be non-negative" in capsys.readouterr().err


def test_live_smoke_report_process_status_matches_supervisor_config(
    tmp_path,
    monkeypatch,
):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    supervisor_config = tmp_path / "bots_vps5.yaml"
    supervisor_config.write_text(
        "\n".join(
            [
                "session_name: passivbot",
                "windows:",
                "  - window_name: binance_01",
                "    panes:",
                "      - passivbot live configs/forager.json -u binance_01",
                "  - window_name: gateio_01",
                "    panes:",
                "      - passivbot live configs/forager.json -u gateio_01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "123 1 42 S 3.5 7.0 "
                    "/root/passivbot/venv/bin/python3 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/forager.json -u binance_01"
                )
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        supervisor_config=supervisor_config,
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["processes"]["enabled"] is True
    assert report["processes"]["ok"] is False
    assert report["processes"]["expected_total"] == 2
    assert report["processes"]["matched_expected"] == 1
    assert report["processes"]["running_live_total"] == 1
    assert report["processes"]["missing_expected"] == [
        {
            "name": "gateio_01",
            "account": "gateio_01",
            "config_path": "configs/forager.json",
            "config_key": "gateio_01:configs/forager.json",
            "command": "passivbot live configs/forager.json -u gateio_01",
            "command_key": "passivbot live configs/forager.json -u gateio_01",
            "match_count": 0,
        }
    ]
    assert report["processes"]["running"][0]["pid"] == 123
    assert report["processes"]["running"][0]["age_s"] == 42
    assert report["processes"]["running"][0]["account"] == "binance_01"
    assert report["processes"]["running"][0]["config_path"] == "configs/forager.json"


def test_live_smoke_report_process_status_parses_no_rss_etimes_passivbot_rows(
    tmp_path,
    monkeypatch,
):
    _write_minimal_monitor_event(tmp_path / "monitor")
    supervisor_config = tmp_path / "bots.yaml"
    supervisor_config.write_text(
        "\n".join(
            [
                "session_name: passivbot",
                "windows:",
                "  - window_name: binance_01",
                "    panes:",
                "      - passivbot live configs/forager.json -u binance_01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "123 1 42 S 3.5 7.0 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/forager.json -u binance_01"
                )
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        supervisor_config=supervisor_config,
    )

    assert report["ok"] is True
    assert report["processes"]["running_live_total"] == 1
    assert report["processes"]["matched_expected"] == 1
    assert report["processes"]["running"][0]["pid"] == 123
    assert report["processes"]["running"][0]["age_s"] == 42
    assert "rss_kb" not in report["processes"]["running"][0]
    assert report["processes"]["running"][0]["command_key"] == (
        "passivbot live configs/forager.json -u binance_01"
    )


def test_live_smoke_report_process_status_reports_duplicates_and_extra_live_processes(
    tmp_path,
    monkeypatch,
):
    _write_minimal_monitor_event(tmp_path / "monitor")
    supervisor_config = tmp_path / "bots.yaml"
    supervisor_config.write_text(
        "\n".join(
            [
                "session_name: passivbot",
                "windows:",
                "  - window_name: binance_01",
                "    panes:",
                "      - passivbot live configs/forager.json -u binance_01",
                "  - window_name: gateio_01",
                "    panes:",
                "      - passivbot live configs/forager.json -u gateio_01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "123 1 42 S 3.5 7.0 100000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/forager.json -u binance_01"
                ),
                (
                    "124 1 44 S 1.5 3.0 90000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/forager.json -u binance_01"
                ),
                (
                    "125 1 45 S 0.5 2.0 80000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/forager.json -u gateio_01"
                ),
                (
                    "126 1 46 S 0.1 1.0 70000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/old.json -u okx_old"
                ),
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        supervisor_config=supervisor_config,
    )
    summary = summarize_live_smoke_report(report, max_groups=1)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is False
    assert report["processes"]["ok"] is False
    assert report["processes"]["hard_failures"] == 2
    assert report["processes"]["expected_total"] == 2
    assert report["processes"]["matched_expected"] == 2
    assert report["processes"]["missing_expected"] == []
    assert report["processes"]["classification_source"] == (
        "local_process_table_command_match"
    )
    assert report["processes"]["tmux_pane_ownership"] == (
        "not_available_from_process_table"
    )
    assert report["processes"]["duplicate_configured_command_matches"][0][
        "account"
    ] == "binance_01"
    assert report["processes"]["duplicate_configured_command_matches"][0][
        "match_count"
    ] == 2
    assert [
        process["pid"]
        for process in report["processes"]["duplicate_configured_command_matches"][0][
            "matched_processes"
        ]
    ] == [123, 124]
    assert report["processes"]["running"][0]["rss_kb"] == 100000
    assert report["processes"]["extra_passivbot_live_processes"] == [
        {
            "pid": 126,
            "ppid": 1,
            "age_s": 46,
            "stat": "S",
            "cpu_pct": 0.1,
            "mem_pct": 1.0,
            "rss_kb": 70000,
            "account": "okx_old",
            "config_path": "configs/old.json",
            "config_key": "okx_old:configs/old.json",
            "command": (
                "/root/passivbot/venv/bin/passivbot live configs/old.json -u okx_old"
            ),
            "command_key": "passivbot live configs/old.json -u okx_old",
        }
    ]
    assert report["processes"]["unexpected_running"] == report["processes"][
        "extra_passivbot_live_processes"
    ]
    assert summary["processes"]["duplicate_configured_command_matches_count"] == 1
    assert summary["processes"]["extra_passivbot_live_processes_count"] == 1
    assert len(summary["processes"]["duplicate_configured_command_matches"]) == 1
    assert len(summary["processes"]["extra_passivbot_live_processes"]) == 1
    assert brief["processes"]["duplicate_configured_command_matches_count"] == 1
    assert brief["processes"]["extra_passivbot_live_processes_count"] == 1


def test_live_smoke_report_process_scan_without_config_is_observational(
    tmp_path,
    monkeypatch,
):
    events_dir = tmp_path / "monitor" / "okx" / "okx_faisal" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(
                event_type="cycle.completed",
                seq=1,
                ts=1000,
                ids={"cycle_id": "cy_1"},
            )
        ],
    )
    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "321 1 S 0.5 2.0 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/v8.json -u okx_faisal"
                )
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        include_processes=True,
    )

    assert report["ok"] is True
    assert report["processes"]["enabled"] is True
    assert report["processes"]["expected_total"] == 0
    assert report["processes"]["running_live_total"] == 1
    assert report["processes"]["unexpected_running"] == []
    assert report["processes"]["running"][0]["command_key"] == (
        "passivbot live configs/v8.json -u okx_faisal"
    )


def test_live_smoke_report_flags_account_hsl_with_launch_balance_override(
    tmp_path,
    monkeypatch,
):
    _write_minimal_monitor_event(tmp_path / "monitor")
    config_path = tmp_path / "configs" / "xmr_migrated.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "live": {"hsl_signal_mode": "unified"},
                "bot": {
                    "long": {"hsl": {"enabled": True}},
                    "short": {"hsl": {"enabled": False}},
                },
            }
        ),
        encoding="utf-8",
    )
    supervisor_config = tmp_path / "bots.yaml"
    supervisor_config.write_text(
        "\n".join(
            [
                "session_name: passivbot",
                "windows:",
                "  - window_name: ebybitsub03",
                "    panes:",
                "      - passivbot live -u ebybitsub03 -bo 1000 configs/xmr_migrated.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "123 1 42 S 2.5 6.0 120000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "-u ebybitsub03 -bo 1000 configs/xmr_migrated.json"
                )
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        supervisor_config=supervisor_config,
    )
    summary = summarize_live_smoke_report(report)
    brief = summarize_live_smoke_report_brief(report)

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["processes"]["hard_failures"] == 1
    running = report["processes"]["running"][0]
    assert running["config_path"] == "configs/xmr_migrated.json"
    assert "balance_override" not in running
    assert running["command_key"] == (
        "passivbot live -u ebybitsub03 -bo [redacted] configs/xmr_migrated.json"
    )
    assert "1000" not in running["command"]
    assert "1000" not in running["command_key"]
    checks = report["processes"]["config_checks"]
    assert checks["ok"] is False
    assert checks["hard_failures"] == 1
    assert checks["checked"] == 1
    issue = checks["issues"][0]
    assert issue["code"] == "hsl_balance_override_account_level_replay_unsafe"
    assert issue["severity"] == "error"
    assert issue["account"] == "ebybitsub03"
    assert issue["hsl_signal_mode"] == "unified"
    assert issue["enabled_psides"] == ["long"]
    assert issue["balance_override_active"] is True
    assert issue["balance_override_source"] == "argument"
    assert "balance_override" not in issue
    assert "1000" not in issue["command_key"]
    assert summary["processes"]["config_checks"]["hard_failures"] == 1
    assert len(summary["processes"]["config_checks"]["issues"]) == 1
    assert brief["processes"]["config_checks"]["hard_failures"] == 1
    assert brief["processes"]["config_checks"]["issues_count"] == 1


def test_live_smoke_report_allows_coin_hsl_with_balance_override(
    tmp_path,
    monkeypatch,
):
    _write_minimal_monitor_event(tmp_path / "monitor")
    config_path = tmp_path / "configs" / "coin_hsl.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "live": {
                    "hsl_signal_mode": "coin",
                    "balance_override": 1000,
                },
                "bot": {
                    "long": {"hsl": {"enabled": True}},
                    "short": {"hsl": {"enabled": True}},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        smoke_report_module,
        "_ps_process_rows",
        lambda: (
            [
                (
                    "123 1 42 S 2.5 6.0 120000 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/coin_hsl.json -u bybit_01"
                )
            ],
            None,
        ),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        include_processes=True,
    )

    assert report["ok"] is True
    assert report["processes"]["config_checks"] == {
        "enabled": True,
        "ok": True,
        "checked": 1,
        "skipped": 0,
        "hard_failures": 0,
        "issues": [],
    }


def test_live_smoke_report_includes_repository_metadata(tmp_path, monkeypatch):
    _write_minimal_monitor_event(tmp_path / "monitor")
    calls = []
    responses = {
        ("rev-parse", "--is-inside-work-tree"): ("true", None),
        ("rev-parse", "HEAD"): ("abcdef1234567890", None),
        ("rev-parse", "--short", "HEAD"): ("abcdef1", None),
        ("rev-parse", "--abbrev-ref", "HEAD"): ("v8", None),
        ("status", "--porcelain", "--untracked-files=no"): (
            " M src/live/smoke_report.py\nM  tests/test_live_smoke_report.py",
            None,
        ),
    }

    def fake_git_output(repo_root, args, *, timeout=2.0):
        calls.append((repo_root, tuple(args), timeout))
        return responses[tuple(args)]

    monkeypatch.setattr(smoke_report_module, "_git_output", fake_git_output)

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        repo_root=tmp_path,
    )

    repository = report["repository"]
    assert report["ok"] is True
    assert repository == {
        "root": str(tmp_path.resolve()),
        "is_git_repo": True,
        "branch": "v8",
        "head": "abcdef1",
        "head_full": "abcdef1234567890",
        "dirty": True,
        "tracked_changes": 2,
        "error": None,
    }
    assert calls[-1][1] == ("status", "--porcelain", "--untracked-files=no")


def test_live_smoke_report_repository_root_redacts_home_prefix(tmp_path, monkeypatch):
    home = tmp_path / "home" / "operator"
    repo_root = home / "passivbot"
    _write_minimal_monitor_event(tmp_path / "monitor")
    calls = []

    def fake_git_output(root, args, *, timeout=2.0):
        calls.append(root)
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return "true", None
        if args == ["rev-parse", "HEAD"]:
            return "1234567890abcdef", None
        if args == ["rev-parse", "--short", "HEAD"]:
            return "1234567", None
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "v8", None
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return "", None
        raise AssertionError(args)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(smoke_report_module, "_git_output", fake_git_output)

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        repo_root=repo_root,
    )

    assert report["repository"]["root"] == "~/passivbot"
    assert set(calls) == {repo_root.resolve()}


def test_live_smoke_report_repository_root_redacts_common_user_dirs():
    assert smoke_report_module._user_safe_display_path("/root/passivbot") == "~/passivbot"
    assert (
        smoke_report_module._user_safe_display_path("/home/deploy/passivbot")
        == "~/passivbot"
    )
    assert (
        smoke_report_module._user_safe_display_path("/Users/operator/passivbot")
        == "~/passivbot"
    )


def test_live_smoke_report_discovers_repository_from_monitor_root(tmp_path, monkeypatch):
    repo_root = tmp_path / "passivbot"
    (repo_root / ".git").mkdir(parents=True)
    _write_minimal_monitor_event(repo_root / "monitor")
    roots = []

    def fake_git_output(repo_root, args, *, timeout=2.0):
        roots.append(repo_root)
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return "true", None
        if args == ["rev-parse", "HEAD"]:
            return "1234567890abcdef", None
        if args == ["rev-parse", "--short", "HEAD"]:
            return "1234567", None
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "v8", None
        if args == ["status", "--porcelain", "--untracked-files=no"]:
            return "", None
        raise AssertionError(args)

    monkeypatch.setattr(smoke_report_module, "_git_output", fake_git_output)

    report = build_live_smoke_report(repo_root / "monitor", logs_root=None)

    assert set(roots) == {repo_root.resolve()}
    assert report["repository"]["root"] == str(repo_root.resolve())
    assert report["repository"]["dirty"] is False


def test_live_smoke_report_repository_metadata_is_observational_on_git_error(
    tmp_path,
    monkeypatch,
):
    _write_minimal_monitor_event(tmp_path / "monitor")
    monkeypatch.setattr(
        smoke_report_module,
        "_git_output",
        lambda repo_root, args, *, timeout=2.0: (None, "not_a_git_repo"),
    )

    report = build_live_smoke_report(
        tmp_path / "monitor",
        logs_root=None,
        repo_root=tmp_path,
    )

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["repository"] == {
        "root": str(tmp_path.resolve()),
        "is_git_repo": False,
        "branch": None,
        "head": None,
        "head_full": None,
        "dirty": None,
        "tracked_changes": None,
        "error": "not_a_git_repo",
    }
