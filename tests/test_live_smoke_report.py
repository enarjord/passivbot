from __future__ import annotations

import json

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
    ids: dict | None = None,
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
        "status": status,
        "reason_code": reason_code,
        "data": {"seq": seq},
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
    assert report["hard_problem_event_count"] == 0
    assert report["logs"]["attention_matches"] == 2
    assert report["logs"]["hard_matches"] == 1
    assert [match["hard"] for match in report["logs"]["matches"]] == [False, True]


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
            [str(tmp_path / "monitor"), "--logs-root", "", "--compact"]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["logs"]["files_scanned"] == 0
    assert report["logs"]["root"] is None
    assert report["monitor"]["live_events"] == 1
