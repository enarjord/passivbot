from __future__ import annotations

import json

import pytest

import live.smoke_report as smoke_report_module
from live.smoke_report import build_live_smoke_report, default_logs_root_for_monitor
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
    assert report["hard_problem_event_count"] == 0
    assert report["logs"]["attention_matches"] == 2
    assert report["logs"]["hard_matches"] == 1
    assert [match["hard"] for match in report["logs"]["matches"]] == [False, True]


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
        "lines_considered": 2,
        "lines_skipped_before": 1,
        "lines_skipped_after": 1,
        "unparsed_ts": 1,
        "unparsed_policy": "keep",
        "lines_skipped_unparsed": 0,
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
    assert drop_report["logs"]["attention_matches"] == 1
    assert drop_report["logs"]["hard_matches"] == 0
    assert drop_report["logs"]["window"] == {
        "enabled": True,
        "since_ms": 2000,
        "until_ms": 4000,
        "lines_considered": 1,
        "lines_skipped_before": 1,
        "lines_skipped_after": 1,
        "unparsed_ts": 1,
        "unparsed_policy": "drop",
        "lines_skipped_unparsed": 1,
    }
    assert [match["text"] for match in drop_report["logs"]["matches"]] == [
        "1970-01-01T00:00:03Z ERROR fresh in window"
    ]


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
                "ERROR stale unparseable line dropped",
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
