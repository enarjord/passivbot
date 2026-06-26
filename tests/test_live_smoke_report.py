from __future__ import annotations

import json

import pytest

import live.smoke_report as smoke_report_module
from live.smoke_report import (
    build_live_smoke_report,
    default_logs_root_for_monitor,
    summarize_live_smoke_report,
)
from tools import live_smoke_report


def _monitor_row(
    *,
    event_type: str,
    seq: int,
    ts: int,
    status: str = "succeeded",
    level: str = "info",
    reason_code: str = "test",
    symbol: str | None = None,
    pside: str | None = None,
    ids: dict | None = None,
    data: dict | None = None,
) -> dict:
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": level,
        "source": "live",
        "component": "test",
        "exchange": "binance",
        "user": "binance_01",
        "symbol": symbol,
        "pside": pside,
        "status": status,
        "reason_code": reason_code,
        "data": dict(data or {"seq": seq}),
        "ids": dict(ids or {}),
    }
    return {
        "exchange": "binance",
        "user": "binance_01",
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
    group_surfaces = {group.get("surface") for group in account_health["groups"]}
    assert group_surfaces == {"balance", "positions", "open_orders"}


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
        "total": 3,
        "groups_truncated": False,
        "event_types": {
            "hsl.status": 2,
            "risk.mode_changed": 1,
        },
        "groups": [
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
            "command": "passivbot live configs/forager.json -u gateio_01",
            "command_key": "passivbot live configs/forager.json -u gateio_01",
        }
    ]
    assert report["processes"]["running"][0]["pid"] == 123
    assert report["processes"]["running"][0]["age_s"] == 42


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
