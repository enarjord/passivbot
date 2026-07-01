from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import pytest

import live.smoke_report as smoke_report_module
from live.incident_bundle import (
    _copy_event_segments,
    _redact_url_userinfo,
    build_live_incident_bundle,
)
from tools import live_incident_bundle


def _monitor_row(
    *,
    event_type: str,
    seq: int,
    ts: int,
    status: str = "succeeded",
    level: str = "info",
    reason_code: str = "test",
    exchange: str = "binance",
    user: str = "binance_01",
    source: str = "live",
    component: str = "test",
    symbol: str | None = None,
    side: str | None = None,
    ids: dict | None = None,
    tags: list[str] | None = None,
    data: dict | None = None,
) -> dict:
    event_data = {"seq": seq, "detail": "kept only when include_data is enabled"}
    if data is not None:
        event_data.update(data)
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": level,
        "source": source,
        "component": component,
        "exchange": exchange,
        "user": user,
        "symbol": symbol,
        "side": side,
        "status": status,
        "reason_code": reason_code,
        "data": event_data,
        "ids": dict(ids or {}),
    }
    return {
        "exchange": exchange,
        "user": user,
        "kind": event_type,
        "tags": list(tags or ["test"]),
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


def _read_tar_json(tar, name: str):
    member = tar.extractfile(name)
    assert member is not None
    return json.loads(member.read().decode())


def test_live_incident_bundle_collects_hashes_snapshots_events_and_window(tmp_path):
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
                event_type="order_wave.started",
                seq=3,
                ts=1100,
                ids={"cycle_id": "cy_1", "order_wave_id": "wave_1"},
            ),
            _monitor_row(
                event_type="execution.create_sent",
                seq=4,
                ts=1200,
                ids={
                    "cycle_id": "cy_1",
                    "order_wave_id": "wave_1",
                    "action_id": "wave_1:create:0",
                },
            ),
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=2000,
                status="failed",
                level="warning",
                reason_code="request_timeout",
                symbol="BTC/USDT:USDT",
                ids={"cycle_id": "cy_2", "remote_call_id": "rc_1"},
            ),
        ],
    )
    (events_dir / "20260629.ndjson.gz").write_bytes(b"")
    snapshot = tmp_path / "monitor" / "binance" / "binance_01" / "state.latest.json"
    snapshot.write_text(
        json.dumps(
            {
                "ok": True,
                "api_key": "SNAPSHOT_KEY",
                "nested": {
                    "authorization": "Bearer SNAPSHOT_TOKEN",
                    "url": "https://user:pass@example.com/path",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    unexpected_snapshot = (
        tmp_path / "monitor" / "binance" / "binance_01" / "debug_dump.json"
    )
    unexpected_snapshot.write_text('{"secret": "do-not-copy-snapshot"}\n', encoding="utf-8")
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "bot.log").write_text(
        "1970-01-01T00:00:02Z ERROR request timeout\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text('{"api_key": "do-not-copy"}\n', encoding="utf-8")
    output = tmp_path / "incident.tar.gz"

    report = build_live_incident_bundle(
        tmp_path / "monitor",
        output_path=output,
        logs_root=logs_dir,
        config_paths=[config_path],
        cycle_id="cy_1",
        since_ms=1500,
        until_ms=2500,
        include_data=False,
        max_event_segment_bytes=100_000,
        cwd=tmp_path,
    )

    assert report["ok"] is True
    assert report["bundle_path"] == str(output)
    assert report["event_report"]["cycle_matched_events"] == 3
    assert report["event_report"]["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 1,
        "scope_pruned": 0,
    }
    assert report["event_report"]["trace_summary_matched_events"] == 3
    assert report["event_report"]["order_trace_matched_events"] == 2
    assert report["event_report"]["cycle_trace_matched_events"] == 3
    assert report["problem_event_report"]["enabled"] is True
    assert report["problem_event_report"]["matched_events"] == 0
    assert report["time_window"]["files_scanned"] == 1
    assert report["time_window"]["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 1,
        "scope_pruned": 0,
    }
    assert report["time_window"]["matched_events"] == 1
    assert report["smoke_report"]["event_window"] == {
        "enabled": True,
        "since_ms": 1500,
        "until_ms": 2500,
        "events_considered": 1,
        "events_skipped_before": 3,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
    }
    assert report["smoke_report"]["logs"] == {
        "max_files": 8,
        "tail_lines": 500,
        "max_matches": 100,
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
            "enabled": True,
            "since_ms": 1500,
            "until_ms": 2500,
            "lines_considered": 1,
            "lines_skipped_before": 0,
            "lines_skipped_after": 0,
            "lines_skipped_unparsed": 0,
            "unparsed_policy": "keep",
            "unparsed_ts": 0,
            "dropped_unparsed_attention_matches": 0,
            "dropped_unparsed_hard_matches": 0,
        },
    }
    assert report["config_hashes"] == 1
    assert report["monitor_snapshots"] == 1
    assert report["event_segments"]["included"] == 1
    assert report["event_segments"]["file_discovery"] == {
        "bot_path_pruning_applied": False,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 1,
        "scope_pruned": 0,
    }

    with tarfile.open(output, "r:gz") as tar:
        tar_names = tar.getnames()
        names = set(tar_names)
        assert len(tar_names) == len(names)
        assert "manifest.json" in names
        assert "event_report.json" in names
        assert "problem_event_report.json" in names
        assert "time_window_report.json" in names
        assert "smoke_report.json" in names
        assert "timeline.txt" in names
        assert "config_hashes.json" in names
        assert "binance/binance_01/state.latest.json" not in names
        assert "monitor_snapshots/binance/binance_01/state.latest.json" in names
        assert "monitor_snapshots/binance/binance_01/debug_dump.json" not in names
        assert any(name.startswith("event_segments/") for name in names)

        manifest = _read_tar_json(tar, "manifest.json")
        event_report = _read_tar_json(tar, "event_report.json")
        problem_event_report = _read_tar_json(tar, "problem_event_report.json")
        event_segments_manifest = _read_tar_json(tar, "event_segments_manifest.json")
        window_report = _read_tar_json(tar, "time_window_report.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
        config_hashes = _read_tar_json(tar, "config_hashes.json")
        redacted_snapshot = _read_tar_json(
            tar,
            "monitor_snapshots/binance/binance_01/state.latest.json",
        )

    assert manifest["config_hashes"][0]["sha256"] == config_hashes[0]["sha256"]
    assert manifest["filters"]["max_log_files"] == 8
    assert manifest["filters"]["log_tail_lines"] == 500
    assert manifest["filters"]["max_log_matches"] == 100
    assert manifest["filters"]["log_window_unparsed_policy"] == "keep"
    assert manifest["event_segments"]["file_discovery"] == report["event_segments"][
        "file_discovery"
    ]
    assert event_segments_manifest["file_discovery"] == report["event_segments"][
        "file_discovery"
    ]
    assert manifest["monitor_snapshots"][0]["redacted"] is True
    assert "do-not-copy" not in json.dumps(manifest)
    assert "do-not-copy" not in json.dumps(config_hashes)
    snapshot_dump = json.dumps(redacted_snapshot)
    assert "SNAPSHOT_KEY" not in snapshot_dump
    assert "SNAPSHOT_TOKEN" not in snapshot_dump
    assert "user:pass" not in snapshot_dump
    assert "do-not-copy-snapshot" not in snapshot_dump
    assert redacted_snapshot["api_key"] == "[redacted]"
    assert redacted_snapshot["nested"]["authorization"] == "[redacted]"
    assert redacted_snapshot["nested"]["url"] == "https://[redacted]@example.com/path"
    assert event_report["cycle"]["events"][0]["seq"] == 1
    assert event_report["file_discovery"] == report["event_report"]["file_discovery"]
    assert problem_event_report["query"]["filters"] == {
        "cycle_id": "cy_1",
        "problem_events": True,
        "since_ms": 1500,
        "until_ms": 2500,
    }
    assert problem_event_report["query"]["matched_events"] == 0
    assert "data" not in event_report["cycle"]["events"][0]
    assert event_report["cycle"]["trace_summary"]["matched_events"] == 3
    assert event_report["cycle"]["order_trace"]["matched_order_events"] == 2
    assert event_report["cycle"]["cycle_trace"]["matched_cycle_events"] == 3
    assert (
        event_report["cycle"]["cycle_trace"]["cycles"][0]["order_trace"][
            "matched_order_events"
        ]
        == 2
    )
    assert window_report["events"][0]["seq"] == 2
    assert "remote_call.failed" in window_report["timeline"][0]
    assert smoke_report["event_window"] == report["smoke_report"]["event_window"]
    assert smoke_report["logs"]["max_files"] == report["smoke_report"]["logs"][
        "max_files"
    ]
    assert smoke_report["logs"]["tail_lines"] == report["smoke_report"]["logs"][
        "tail_lines"
    ]
    assert smoke_report["logs"]["max_matches"] == report["smoke_report"]["logs"][
        "max_matches"
    ]
    assert smoke_report["logs"]["window"] == report["smoke_report"]["logs"]["window"]
    assert smoke_report["remote_call_failures"]["total"] == 1


def test_live_incident_bundle_embeds_problem_event_report(tmp_path):
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
                ids={"cycle_id": "cy_1", "remote_call_id": "rc_1"},
            ),
            _monitor_row(
                event_type="ema.unavailable",
                seq=3,
                ts=1200,
                status="degraded",
                level="warning",
                reason_code="stale_ema",
                symbol="ETH/USDT:USDT",
                ids={"cycle_id": "cy_2"},
            ),
        ],
    )
    output = tmp_path / "incident.tar.gz"

    report = build_live_incident_bundle(
        tmp_path / "monitor",
        output_path=output,
        logs_root="",
        cycle_id="cy_1",
        include_data=True,
        max_problem_events=10,
    )

    assert report["ok"] is True
    assert report["problem_event_report"] == {
        "enabled": True,
        "files_scanned": 1,
        "file_discovery": {
            "bot_path_pruning_applied": False,
            "candidate_files": 1,
            "event_segments": 1,
            "opaque_bot_id_full_scan": False,
            "rotated_skipped": 0,
            "scope_pruned": 0,
        },
        "live_events": 3,
        "error_count": 0,
        "warning_count": 0,
        "event_window": None,
        "matched_events": 1,
        "events_truncated": False,
        "trace_summary_matched_events": 1,
    }

    with tarfile.open(output, "r:gz") as tar:
        problem_event_report = _read_tar_json(tar, "problem_event_report.json")

    assert problem_event_report["query"]["filters"] == {
        "cycle_id": "cy_1",
        "problem_events": True,
    }
    assert problem_event_report["query"]["matched_events"] == 1
    assert (
        problem_event_report["query"]["events"][0]["event_type"]
        == "remote_call.failed"
    )
    assert problem_event_report["query"]["events"][0]["data"] == {
        "detail": "kept only when include_data is enabled",
        "seq": 2,
    }
    assert problem_event_report["query"]["trace_summary"]["matched_events"] == 1


def test_live_incident_bundle_cli_filters_event_reports_by_query_scopes(
    tmp_path, capsys
):
    binance_events = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    okx_events = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    _write_ndjson(
        binance_events / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=1,
                ts=1000,
                status="failed",
                level="warning",
                reason_code="request_timeout",
                exchange="binance",
                user="binance_01",
                source="live",
                component="ema.bundle",
                side="buy",
                tags=["ema"],
                data={"scope": "other"},
                ids={
                    "cycle_id": "cy_1",
                    "remote_call_id": "rc_binance",
                    "remote_call_group_id": "group_other",
                },
            ),
        ],
    )
    _write_ndjson(
        okx_events / "current.ndjson",
        [
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=1100,
                status="failed",
                level="warning",
                reason_code="request_timeout",
                exchange="okx",
                user="okx_01",
                source="live",
                component="execution",
                symbol="ZEC/USDT:USDT",
                side="sell",
                tags=["order", "execution"],
                data={"scope": "target"},
                ids={
                    "cycle_id": "cy_2",
                    "remote_call_id": "rc_okx",
                    "remote_call_group_id": "group_target",
                },
            ),
        ],
    )
    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--exchange",
                "okx",
                "--user",
                "okx_01",
                "--bot-id",
                "okx/okx_01",
                "--level",
                "warning",
                "--source",
                "live",
                "--component",
                "execution",
                "--side",
                "sell",
                "--tag",
                "order",
                "--data-eq",
                "scope=target",
                "--remote-call-group-id",
                "group_target",
                "--since-ms",
                "0",
                "--until-ms",
                "2000",
                "--include-data",
                "--no-event-segments",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["event_report"]["query_matched_events"] == 1
    assert report["problem_event_report"]["matched_events"] == 1
    assert report["event_report"]["file_discovery"] == {
        "bot_path_pruning_applied": True,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 0,
        "scope_pruned": 1,
    }
    assert report["problem_event_report"]["file_discovery"] == {
        "bot_path_pruning_applied": True,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 0,
        "scope_pruned": 1,
    }
    assert report["time_window"]["files_scanned"] == 1
    assert report["time_window"]["file_discovery"] == {
        "bot_path_pruning_applied": True,
        "candidate_files": 2,
        "event_segments": 2,
        "opaque_bot_id_full_scan": False,
        "rotated_skipped": 0,
        "scope_pruned": 1,
    }

    with tarfile.open(output, "r:gz") as tar:
        event_report = _read_tar_json(tar, "event_report.json")
        problem_event_report = _read_tar_json(tar, "problem_event_report.json")
        time_window_report = _read_tar_json(tar, "time_window_report.json")
        timeline_text = tar.extractfile("timeline.txt").read().decode("utf-8")
        event_segments_manifest = _read_tar_json(tar, "event_segments_manifest.json")
        manifest = _read_tar_json(tar, "manifest.json")

    expected_filters = {
        "bot_ids": ["okx/okx_01"],
        "components": ["execution"],
        "data_eq": {"scope": ["target"]},
        "exchanges": ["okx"],
        "levels": ["warning"],
        "remote_call_group_ids": ["group_target"],
        "sides": ["sell"],
        "sources": ["live"],
        "tags": ["order"],
        "users": ["okx_01"],
    }
    assert event_report["query"]["filters"] == expected_filters
    assert problem_event_report["query"]["filters"] == {
        **expected_filters,
        "problem_events": True,
        "since_ms": 0,
        "until_ms": 2000,
    }
    assert time_window_report["filters"] == {
        **expected_filters,
        "since_ms": 0,
        "until_ms": 2000,
    }
    assert time_window_report["matched_events"] == 1
    assert event_report["query"]["events"][0]["exchange"] == "okx"
    assert event_report["query"]["events"][0]["user"] == "okx_01"
    assert event_report["query"]["events"][0]["side"] == "sell"
    assert event_report["query"]["events"][0]["data"]["scope"] == "target"
    assert time_window_report["events"][0]["exchange"] == "okx"
    assert "seq=2" in timeline_text
    assert "seq=1" not in timeline_text
    assert len(event_segments_manifest["files"]) == 1
    assert event_segments_manifest["files"][0]["path"].endswith(
        "okx/okx_01/events/current.ndjson"
    )
    assert manifest["filters"]["exchange"] == ["okx"]
    assert manifest["filters"]["bot_id"] == ["okx/okx_01"]
    assert manifest["filters"]["data_eq"] == ["scope=target"]
    assert manifest["filters"]["since_ms"] == 0
    assert manifest["filters"]["until_ms"] == 2000


def test_live_incident_bundle_can_skip_logs_and_segments_from_cli(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_01" / "events"
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
    (logs_dir / "bot.log").write_text(
        "2026-06-25T00:00:00Z CRITICAL skipped log\n",
        encoding="utf-8",
    )
    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--no-event-segments",
                "--no-trace-report",
                "--no-problem-report",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["bundle_path"] == str(output)
    assert report["problem_event_report"] == {
        "enabled": False,
        "files_scanned": None,
        "file_discovery": {},
        "live_events": None,
        "error_count": None,
        "warning_count": None,
        "event_window": None,
        "matched_events": None,
        "events_truncated": None,
        "trace_summary_matched_events": None,
    }
    assert report["smoke_report"]["logs"]["max_files"] == 8
    assert report["smoke_report"]["logs"]["tail_lines"] == 500
    assert report["smoke_report"]["logs"]["max_matches"] == 100
    assert report["smoke_report"]["logs"]["files_scanned"] == 0
    assert report["event_segments"]["included"] == 0
    assert output.exists()
    with tarfile.open(output, "r:gz") as tar:
        assert "problem_event_report.json" not in set(tar.getnames())

    with tarfile.open(output, "r:gz") as tar:
        tar_names = tar.getnames()
        names = set(tar_names)
        assert len(tar_names) == len(names)
        assert "event_segments_manifest.json" in names
        assert not any(name.startswith("event_segments/") for name in names)
        event_segments_manifest = _read_tar_json(tar, "event_segments_manifest.json")
        event_report = _read_tar_json(tar, "event_report.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
    assert event_segments_manifest["files"]
    assert event_segments_manifest["files"][0]["included"] is False
    assert event_segments_manifest["files"][0]["reason"] == "disabled"
    assert "sha256" not in event_segments_manifest["files"][0]
    assert "trace_summary" not in event_report["query"]
    assert "order_trace" not in event_report["query"]
    assert smoke_report["logs"]["root"] is None
    assert smoke_report["logs"]["max_files"] == report["smoke_report"]["logs"][
        "max_files"
    ]
    assert smoke_report["logs"]["hard_matches"] == 0


def test_live_incident_bundle_cli_can_focus_embedded_smoke_report(tmp_path, capsys):
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
                component="fills.refresh",
                data={
                    "source": "cache",
                    "refresh_mode": "startup",
                    "history_scope": "all",
                    "coverage_ready_after": True,
                    "elapsed_ms": 30,
                },
            )
        ],
    )
    output = tmp_path / "incident.tar.gz"
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "bot.log").write_text(
        "1970-01-01T00:00:01Z ERROR kept in compact summary\n",
        encoding="utf-8",
    )

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                str(logs_dir),
                "--no-event-segments",
                "--no-trace-report",
                "--no-problem-report",
                "--smoke-section",
                "fill_refresh_health",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["smoke_report"]["logs"]["files_scanned"] == 1
    assert report["smoke_report"]["logs"]["attention_matches"] == 1
    with tarfile.open(output, "r:gz") as tar:
        smoke_report = _read_tar_json(tar, "smoke_report.json")
        manifest = _read_tar_json(tar, "manifest.json")
    assert smoke_report["ok"] is True
    assert smoke_report["fill_refresh_health"]["total"] == 1
    assert "logs" not in smoke_report
    assert "remote_call_health" not in smoke_report
    assert manifest["filters"]["smoke_sections"] == ["fill_refresh_health"]


def test_live_incident_bundle_cli_rejects_malformed_data_eq_filter(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--data-eq",
                "missing_equals",
            ]
        )

    assert exc_info.value.code == 2
    assert "key=value" in capsys.readouterr().err


def test_live_incident_bundle_cli_passes_log_window_unparsed_policy(
    tmp_path,
    capsys,
):
    events_dir = tmp_path / "monitor" / "okx" / "okx_01" / "events"
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
    (logs_dir / "okx_01.log").write_text(
        "\n".join(
            [
                "1970-01-01T00:00:03Z ERROR fresh in window",
                "old unparseable noise dropped",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                str(logs_dir),
                "--since-ms",
                "2000",
                "--until-ms",
                "4000",
                "--log-window-unparsed-policy",
                "drop",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    with tarfile.open(output, "r:gz") as tar:
        smoke_report = _read_tar_json(tar, "smoke_report.json")
    assert smoke_report["logs"]["attention_matches"] == 1
    assert smoke_report["logs"]["window"]["unparsed_policy"] == "drop"
    assert smoke_report["logs"]["window"]["lines_skipped_unparsed"] == 1


def test_live_incident_bundle_cli_recent_minutes_sets_since_ms(
    tmp_path,
    capsys,
    monkeypatch,
):
    events_dir = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="cycle.completed", seq=1, ts=1000),
            _monitor_row(event_type="cycle.completed", seq=2, ts=2500),
        ],
    )
    monkeypatch.setattr(live_incident_bundle.time, "time", lambda: 5.0)
    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--recent-minutes",
                "0.05",
                "--no-event-segments",
                "--no-trace-report",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["time_window"]["matched_events"] == 1
    with tarfile.open(output, "r:gz") as tar:
        manifest = _read_tar_json(tar, "manifest.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
        window_report = _read_tar_json(tar, "time_window_report.json")
    assert manifest["filters"]["since_ms"] == 2000
    assert manifest["filters"]["include_rotated"] is False
    assert smoke_report["event_window"]["since_ms"] == 2000
    assert window_report["filters"] == {"since_ms": 2000}


def test_live_incident_bundle_cli_event_tail_lines_bounds_event_scans(tmp_path, capsys):
    events_dir = tmp_path / "monitor" / "okx" / "okx_01" / "events"
    _write_ndjson(
        events_dir / "current.ndjson",
        [
            _monitor_row(event_type="cycle.completed", seq=1, ts=1000),
            _monitor_row(
                event_type="remote_call.failed",
                seq=2,
                ts=2000,
                status="failed",
                level="warning",
            ),
            _monitor_row(event_type="cycle.completed", seq=3, ts=3000),
        ],
    )
    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--since-ms",
                "1000",
                "--event-type",
                "cycle.completed",
                "--event-tail-lines",
                "1",
                "--no-event-segments",
                "--no-trace-report",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["event_report"]["query_matched_events"] == 1
    assert report["event_report"]["event_window"] == {
        "enabled": False,
        "since_ms": None,
        "until_ms": None,
        "events_considered": 1,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "files_skipped_before_window": 0,
        "event_tail_lines": 1,
        "event_tail_limited_files": 1,
        "event_tail_skipped_lines": 2,
        "event_tail_skipped_lines_exact": True,
        "event_tail_skipped_bytes": 0,
        "event_tail_line_numbers_exact": True,
        "event_tail_methods": {"seek_tail": 1},
    }
    assert report["time_window"]["matched_events"] == 1
    assert report["time_window"]["event_tail_lines"] == 1
    assert report["time_window"]["event_tail_limited_files"] == 1
    assert report["time_window"]["event_tail_skipped_lines"] == 2
    assert report["smoke_report"]["event_window"]["event_tail_lines"] == 1
    assert report["smoke_report"]["event_window"]["event_tail_limited_files"] == 1
    assert report["smoke_report"]["event_window"]["event_tail_skipped_lines"] == 2

    with tarfile.open(output, "r:gz") as tar:
        manifest = _read_tar_json(tar, "manifest.json")
        event_report = _read_tar_json(tar, "event_report.json")
        window_report = _read_tar_json(tar, "time_window_report.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
    assert manifest["filters"]["event_tail_lines"] == 1
    assert event_report["query"]["events"][0]["seq"] == 3
    assert event_report["event_window"]["event_tail_skipped_lines"] == 2
    assert window_report["events"][0]["seq"] == 3
    assert smoke_report["event_window"]["event_tail_skipped_lines"] == 2


def test_live_incident_bundle_cli_max_event_files_per_bot_is_fair(tmp_path, capsys):
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
    for path, seq, exchange, user, label in files:
        _write_ndjson(
            path,
            [
                _monitor_row(
                    event_type="remote_call.failed",
                    seq=seq,
                    ts=seq * 100,
                    status="failed",
                    level="warning",
                    reason_code="request_timeout",
                    exchange=exchange,
                    user=user,
                    data={"label": label},
                )
            ],
        )
    for path in (binance_events / "old.ndjson", okx_events / "old.ndjson"):
        _set_mtime(path, 100)
    for path in (binance_events / "new.ndjson", okx_events / "new.ndjson"):
        _set_mtime(path, 200)
    for path in (binance_events / "current.ndjson", okx_events / "current.ndjson"):
        _set_mtime(path, 50)

    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--event-type",
                "remote_call.failed",
                "--since-ms",
                "0",
                "--include-rotated",
                "--include-data",
                "--max-event-files-per-bot",
                "2",
                "--no-event-segments",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["event_report"]["query_matched_events"] == 4
    assert report["event_report"]["event_window"] == {
        "enabled": False,
        "since_ms": None,
        "until_ms": None,
        "events_considered": 4,
        "events_skipped_before": 0,
        "events_skipped_after": 0,
        "invalid_window_ts": 0,
        "files_skipped_before_window": 0,
        "max_event_files_per_bot": 2,
        "event_file_limit_scope": "per_bot",
        "event_file_limit_groups": 2,
        "event_files_before_limit": 6,
        "event_files_skipped_by_limit": 2,
        "event_file_limit_order": "current_then_recent_mtime",
    }
    assert report["problem_event_report"]["matched_events"] == 4
    assert (
        report["problem_event_report"]["event_window"]["event_file_limit_scope"]
        == "per_bot"
    )
    assert report["time_window"]["matched_events"] == 4
    assert report["time_window"]["max_event_files_per_bot"] == 2
    assert report["time_window"]["event_file_limit_groups"] == 2
    assert report["time_window"]["event_files_before_limit"] == 6
    assert report["time_window"]["event_files_skipped_by_limit"] == 2

    with tarfile.open(output, "r:gz") as tar:
        manifest = _read_tar_json(tar, "manifest.json")
        event_report = _read_tar_json(tar, "event_report.json")
        problem_event_report = _read_tar_json(tar, "problem_event_report.json")
        window_report = _read_tar_json(tar, "time_window_report.json")

    assert manifest["filters"]["max_event_files_per_bot"] == 2
    kept_labels = {event["data"]["label"] for event in event_report["query"]["events"]}
    assert kept_labels == {
        "binance_current",
        "binance_new",
        "okx_current",
        "okx_new",
    }
    assert {
        event["data"]["label"]
        for event in problem_event_report["query"]["events"]
    } == kept_labels
    assert {event["data"]["label"] for event in window_report["events"]} == kept_labels
    assert window_report["event_file_limit_order"] == "current_then_recent_mtime"


def test_live_incident_bundle_cli_caps_fallback_event_segments_per_bot(tmp_path, capsys):
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
    for path, seq, exchange, user, label in files:
        _write_ndjson(
            path,
            [
                _monitor_row(
                    event_type="cycle.completed",
                    seq=seq,
                    ts=seq * 100,
                    exchange=exchange,
                    user=user,
                    data={"label": label},
                )
            ],
        )
    for path in (binance_events / "old.ndjson", okx_events / "old.ndjson"):
        _set_mtime(path, 100)
    for path in (binance_events / "new.ndjson", okx_events / "new.ndjson"):
        _set_mtime(path, 200)
    for path in (binance_events / "current.ndjson", okx_events / "current.ndjson"):
        _set_mtime(path, 50)

    output = tmp_path / "incident.tar.gz"

    assert (
        live_incident_bundle.main(
            [
                str(tmp_path / "monitor"),
                "--logs-root",
                "",
                "--event-type",
                "no.such.event",
                "--include-rotated",
                "--max-event-files-per-bot",
                "2",
                "--max-event-segment-bytes",
                "1000000",
                "--output",
                str(output),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["event_report"]["query_matched_events"] == 0
    assert report["event_segments"]["files"] == 4
    assert report["event_segments"]["included"] == 4
    assert report["event_segments"]["selection"] == "fallback_discovered_paths"
    assert report["event_segments"]["max_event_files_per_bot"] == 2
    assert report["event_segments"]["event_file_limit_applies_to"] == (
        "fallback_discovered_segments"
    )
    assert report["event_segments"]["event_file_limit_groups"] == 2
    assert report["event_segments"]["event_files_before_limit"] == 6
    assert report["event_segments"]["event_files_skipped_by_limit"] == 2

    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())
        event_segments_manifest = _read_tar_json(tar, "event_segments_manifest.json")
        manifest = _read_tar_json(tar, "manifest.json")

    included_paths = {
        item["path"]
        for item in event_segments_manifest["files"]
        if item.get("included")
    }
    assert {Path(path).name for path in included_paths} == {
        "current.ndjson",
        "new.ndjson",
    }
    assert len(included_paths) == 4
    assert not any(path.endswith("old.ndjson") for path in included_paths)
    assert sum(1 for name in names if name.startswith("event_segments/")) == 4
    assert event_segments_manifest["event_file_limit_order"] == (
        "current_then_recent_mtime"
    )
    assert manifest["event_segments"]["event_files_skipped_by_limit"] == 2


def test_live_incident_bundle_preserves_matched_event_segments_under_cap(tmp_path):
    events_dir = tmp_path / "monitor" / "binance" / "binance_01" / "events"
    files = [
        events_dir / "current.ndjson",
        events_dir / "new.ndjson",
        events_dir / "old.ndjson",
    ]
    for seq, path in enumerate(files, start=1):
        _write_ndjson(
            path,
            [
                _monitor_row(
                    event_type="cycle.completed",
                    seq=seq,
                    ts=seq * 100,
                    exchange="binance",
                    user="binance_01",
                )
            ],
        )
    _set_mtime(events_dir / "old.ndjson", 100)
    _set_mtime(events_dir / "new.ndjson", 200)
    _set_mtime(events_dir / "current.ndjson", 50)

    bundle_root = tmp_path / "bundle"
    manifest = _copy_event_segments(
        monitor_root=tmp_path / "monitor",
        bundle_root=bundle_root,
        event_report={
            "query": {
                "events": [
                    {"path": str(events_dir / "current.ndjson")},
                    {"path": str(events_dir / "new.ndjson")},
                    {"path": str(events_dir / "old.ndjson")},
                ]
            }
        },
        window_report={},
        problem_report={},
        include_rotated=True,
        include_segments=True,
        max_total_bytes=1_000_000,
        max_event_files_per_bot=1,
    )

    included_paths = {
        item["path"]
        for item in manifest["files"]
        if item.get("included")
    }
    assert included_paths == {str(path) for path in files}
    assert manifest["selection"] == "matched_report_paths"
    assert manifest["max_event_files_per_bot"] == 1
    assert manifest["event_file_limit_applies_to"] == "matched_report_paths_preserved"
    assert manifest["event_file_limit_groups"] == 0
    assert manifest["event_files_before_limit"] == 3
    assert manifest["event_files_skipped_by_limit"] == 0
    assert len(list((bundle_root / "event_segments").iterdir())) == 3


def test_live_incident_bundle_cli_rejects_invalid_window_timestamp(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(["monitor", "--since-ms", "not-an-int"])

    assert exc_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_live_incident_bundle_cli_rejects_conflicting_recent_window(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(
            ["monitor", "--since-ms", "1000", "--recent-minutes", "1"]
        )

    assert exc_info.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_live_incident_bundle_cli_rejects_non_positive_recent_minutes(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(["monitor", "--recent-minutes", "0"])

    assert exc_info.value.code == 2
    assert "must be greater than 0" in capsys.readouterr().err


def test_live_incident_bundle_cli_rejects_negative_event_tail_lines(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(["monitor", "--event-tail-lines", "-1"])

    assert exc_info.value.code == 2
    assert "--event-tail-lines must be >= 0" in capsys.readouterr().err


def test_live_incident_bundle_cli_rejects_negative_max_event_files_per_bot(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(["monitor", "--max-event-files-per-bot", "-1"])

    assert exc_info.value.code == 2
    assert "--max-event-files-per-bot must be >= 0" in capsys.readouterr().err


def test_live_incident_bundle_cli_rejects_recent_window_after_until(capsys, monkeypatch):
    monkeypatch.setattr(live_incident_bundle.time, "time", lambda: 5.0)

    with pytest.raises(SystemExit) as exc_info:
        live_incident_bundle.main(
            ["monitor", "--recent-minutes", "0.05", "--until-ms", "1500"]
        )

    assert exc_info.value.code == 2
    assert "--since-ms/--recent-minutes must be <= --until-ms" in capsys.readouterr().err


def test_live_incident_bundle_includes_process_status_when_requested(
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
                "      - passivbot live configs/v8.json -u binance_01",
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
                    "123 1 99 S 1.0 5.0 "
                    "/root/passivbot/venv/bin/passivbot live "
                    "configs/v8.json -u binance_01"
                )
            ],
            None,
        ),
    )
    output = tmp_path / "incident.tar.gz"

    report = build_live_incident_bundle(
        tmp_path / "monitor",
        output_path=output,
        logs_root=None,
        supervisor_config=supervisor_config,
        include_event_segments=False,
    )

    assert report["ok"] is True
    assert report["smoke_report"]["processes"] == {
        "enabled": True,
        "ok": True,
        "expected_total": 1,
        "running_live_total": 1,
        "missing_expected": 0,
    }
    with tarfile.open(output, "r:gz") as tar:
        smoke_report = _read_tar_json(tar, "smoke_report.json")
    assert smoke_report["processes"]["expected_total"] == 1
    assert smoke_report["processes"]["matched_expected"] == 1
    assert smoke_report["processes"]["missing_expected"] == []


def test_live_incident_bundle_redacts_git_remote_url_userinfo():
    assert (
        _redact_url_userinfo("https://token:secret@example.com/org/repo.git")
        == "https://[redacted]@example.com/org/repo.git"
    )
    assert (
        _redact_url_userinfo("git@github.com:enarjord/passivbot.git")
        == "git@github.com:enarjord/passivbot.git"
    )
