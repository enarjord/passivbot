from __future__ import annotations

import json
import tarfile

import pytest

import live.smoke_report as smoke_report_module
from live.incident_bundle import _redact_url_userinfo, build_live_incident_bundle
from tools import live_incident_bundle


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
        "data": {"seq": seq, "detail": "kept only when include_data is enabled"},
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
        "2026-06-25T00:00:00Z ERROR request timeout\n",
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
        event_segments_manifest = _read_tar_json(tar, "event_segments_manifest.json")
        window_report = _read_tar_json(tar, "time_window_report.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
        config_hashes = _read_tar_json(tar, "config_hashes.json")
        redacted_snapshot = _read_tar_json(
            tar,
            "monitor_snapshots/binance/binance_01/state.latest.json",
        )

    assert manifest["config_hashes"][0]["sha256"] == config_hashes[0]["sha256"]
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
    assert smoke_report["remote_call_failures"]["total"] == 1


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
    assert report["event_segments"]["included"] == 0
    assert output.exists()

    with tarfile.open(output, "r:gz") as tar:
        tar_names = tar.getnames()
        names = set(tar_names)
        assert len(tar_names) == len(names)
        assert "event_segments_manifest.json" in names
        assert not any(name.startswith("event_segments/") for name in names)
        event_report = _read_tar_json(tar, "event_report.json")
        smoke_report = _read_tar_json(tar, "smoke_report.json")
    assert "trace_summary" not in event_report["query"]
    assert "order_trace" not in event_report["query"]
    assert smoke_report["logs"]["root"] is None
    assert smoke_report["logs"]["hard_matches"] == 0


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
