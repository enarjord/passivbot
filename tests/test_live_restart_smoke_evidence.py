from __future__ import annotations

import json

import pytest

from live.restart_smoke_evidence import build_live_restart_smoke_evidence
from tools import live_restart_smoke_evidence


HEAD = "a" * 40
FINGERPRINT = "b" * 64


def _target_report(*, targets: int = 2) -> dict:
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": True,
        "hard_failures": 0,
        "issues": [],
        "expected_targets": targets,
        "resolved_targets": targets,
        "relaunch_ready_targets": targets,
        "extra_panes": [],
        "supervisor_contract": {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": FINGERPRINT,
            "target_count": targets,
            "command_content_exposed": False,
        },
        "sampling": {
            "requested_samples": 3,
            "collected_samples": 3,
            "successful_samples": 3,
            "failed_samples": 0,
            "stable": True,
            "supervisor_contract_stable": True,
            "supervisor_contract_changed": False,
            "stable_targets": targets,
            "changed_target_count": 0,
        },
        "targets": [
            {
                "window_name": f"private_bot_{index}",
                "current_command": "secret command must not project",
            }
            for index in range(targets)
        ],
    }


def _window() -> dict:
    return {"enabled": True, "since_ms": 1000, "until_ms": 2000}


def _smoke_report(*, targets: int = 2) -> dict:
    return {
        "tool": "live-smoke-report",
        "schema_version": 1,
        "ok": True,
        "attention": False,
        "hard_failures": 0,
        "attention_count": 0,
        "repository": {
            "is_git_repo": True,
            "head_full": HEAD,
            "dirty": False,
            "tracked_changes": 0,
            "error": None,
        },
        "event_window": _window(),
        "monitor": {"files_scanned": 2, "error_count": 0, "warning_count": 0},
        "logs": {
            "files_scanned": 2,
            "hard_matches": 0,
            "attention_matches": 0,
            "window": _window(),
        },
        "shutdown_events": {
            "event_types": {"bot.stopping": targets, "bot.stopped": targets}
        },
        "startup_timings": [
            {"bot": f"venue/bot_{index}", "phases": {"startup": {}}}
            for index in range(targets)
        ],
    }


def _evaluate(target_report: dict | None = None, smoke_report: dict | None = None) -> dict:
    return build_live_restart_smoke_evidence(
        target_report or _target_report(),
        smoke_report or _smoke_report(),
        expected_repository_head=HEAD,
        expected_supervisor_fingerprint=FINGERPRINT,
        expected_targets=2,
    )


def _codes(report: dict) -> set[str]:
    return {row["code"] for row in report["issues"]}


def test_evaluator_accepts_complete_bounded_evidence():
    report = _evaluate()

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["issues"] == []
    assert report["gates"]["target_contract"]["sampling_ok"] is True
    assert report["gates"]["repository"]["head_matches_expected"] is True
    assert report["safety"] == {
        "local_only": True,
        "read_only": True,
        "subprocess_execution": False,
        "signals_processes": False,
        "starts_processes": False,
        "network": False,
        "exchange_access": False,
        "writes_files": False,
    }


@pytest.mark.parametrize(
    ("mutate", "expected_count"),
    [
        (lambda report: report.update({"resolved_targets": 1}), 1),
        (
            lambda report: report["supervisor_contract"].update({"fingerprint": "c" * 64}),
            1,
        ),
        (
            lambda report: report["sampling"].update(
                {"stable": False, "changed_target_count": 1}
            ),
            1,
        ),
    ],
)
def test_evaluator_rejects_target_count_fingerprint_and_sampling_failures(
    mutate, expected_count
):
    target = _target_report()
    mutate(target)

    report = _evaluate(target)

    assert report["ok"] is False
    assert report["hard_failures"] == expected_count
    assert _codes(report) == {"target_contract_invalid"}


def test_evaluator_rejects_wrong_or_dirty_repository():
    smoke = _smoke_report()
    smoke["repository"].update(
        {"head_full": "c" * 40, "dirty": True, "tracked_changes": 1}
    )

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is False
    assert _codes(report) == {"repository_mismatch"}
    assert report["gates"]["repository"]["tracked_change_count"] == 1


@pytest.mark.parametrize(
    "window",
    [
        {"enabled": False, "since_ms": None, "until_ms": None},
        {"enabled": True, "since_ms": 2001, "until_ms": 2000},
        {"enabled": True, "since_ms": 2000, "until_ms": 2000},
    ],
)
def test_evaluator_rejects_unbounded_or_inverted_event_window(window):
    smoke = _smoke_report()
    smoke["event_window"] = window

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is False
    assert "event_window_invalid" in _codes(report)


def test_evaluator_rejects_missing_monitor_and_log_scans():
    smoke = _smoke_report()
    smoke["monitor"].update({"files_scanned": 0, "error_count": 1})
    smoke["logs"].update({"files_scanned": 0, "hard_matches": 1})

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is False
    assert {"monitor_scan_invalid", "log_scan_invalid"} <= _codes(report)


def test_evaluator_rejects_smoke_hard_failures_and_invalid_optional_schema():
    smoke = _smoke_report()
    smoke.update({"ok": False, "hard_failures": 1, "schema_version": 2})

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is False
    assert {"smoke_contract_invalid", "smoke_hard_failures"} <= _codes(report)
    assert report["gates"]["smoke_contract"]["hard_failures"] == 1


def test_evaluator_accepts_current_untagged_smoke_report_shape():
    smoke = _smoke_report()
    smoke.pop("tool")
    smoke.pop("schema_version")

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is True


@pytest.mark.parametrize(
    ("mutate_target", "mutate_smoke", "expected_code"),
    [
        (
            lambda target: target.update({"schema_version": True}),
            None,
            "target_contract_invalid",
        ),
        (
            lambda target: target.update({"hard_failures": False}),
            None,
            "target_contract_invalid",
        ),
        (
            lambda target: target["sampling"].update({"failed_samples": False}),
            None,
            "target_contract_invalid",
        ),
        (None, lambda smoke: smoke.update({"schema_version": True}), "smoke_contract_invalid"),
        (None, lambda smoke: smoke.update({"hard_failures": False}), "smoke_hard_failures"),
        (
            None,
            lambda smoke: smoke["monitor"].update({"error_count": False}),
            "monitor_scan_invalid",
        ),
        (
            None,
            lambda smoke: smoke["logs"].update({"hard_matches": False}),
            "log_scan_invalid",
        ),
    ],
)
def test_evaluator_rejects_malformed_required_numeric_and_schema_fields(
    mutate_target, mutate_smoke, expected_code
):
    target = _target_report()
    smoke = _smoke_report()
    if mutate_target is not None:
        mutate_target(target)
    if mutate_smoke is not None:
        mutate_smoke(smoke)

    report = _evaluate(target, smoke)

    assert report["ok"] is False
    assert expected_code in _codes(report)


def test_evaluator_rejects_bool_target_count_and_supervisor_count_for_one_target():
    target = _target_report(targets=1)
    target["expected_targets"] = True
    target["resolved_targets"] = True
    target["relaunch_ready_targets"] = True
    target["supervisor_contract"]["target_count"] = True
    smoke = _smoke_report(targets=1)

    report = build_live_restart_smoke_evidence(
        target,
        smoke,
        expected_repository_head=HEAD,
        expected_supervisor_fingerprint=FINGERPRINT,
        expected_targets=1,
    )

    assert report["ok"] is False
    assert _codes(report) == {"target_contract_invalid"}


def test_evaluator_rejects_missing_shutdown_and_startup_evidence():
    smoke = _smoke_report()
    smoke["shutdown_events"] = {"event_types": {"bot.stopping": 1, "bot.stopped": 1}}
    smoke["startup_timings"] = [{"bot": "venue/bot_0", "phases": {"startup": {}}}]

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is False
    assert {"shutdown_evidence_missing", "startup_evidence_missing"} <= _codes(report)


def test_attention_evidence_remains_green():
    smoke = _smoke_report()
    smoke.update({"attention": True, "attention_count": 3})
    smoke["monitor"]["warning_count"] = 2
    smoke["logs"]["attention_matches"] = 4

    report = _evaluate(smoke_report=smoke)

    assert report["ok"] is True
    assert report["evidence"]["attention"] == {
        "reported": True,
        "reported_count": 3,
        "monitor_warning_count": 2,
        "log_attention_matches": 4,
    }


def test_evaluator_redacts_input_paths_commands_samples_and_bot_names():
    target = _target_report()
    target["extra_payload"] = {
        "path": "/private/secret-target.json",
        "command": "dangerous raw command",
        "sample": "raw target sample",
    }
    smoke = _smoke_report()
    smoke["monitor"].update(
        {
            "root": "/private/secret-monitor",
            "issues": [{"message": "raw monitor message"}],
        }
    )
    smoke["logs"].update(
        {
            "root": "/private/secret-logs",
            "matches": [{"text": "raw log sample"}],
        }
    )
    smoke["startup_timings"][0]["bot"] = "sensitive-operator-bot"

    output = json.dumps(_evaluate(target, smoke), sort_keys=True)

    for forbidden in (
        "/private/secret-target.json",
        "dangerous raw command",
        "raw target sample",
        "/private/secret-monitor",
        "/private/secret-logs",
        "raw monitor message",
        "raw log sample",
        "sensitive-operator-bot",
    ):
        assert forbidden not in output


def test_cli_emits_compact_report_and_red_exit(tmp_path, capsys):
    target_path = tmp_path / "target.json"
    smoke_path = tmp_path / "smoke.json"
    target_path.write_text(json.dumps(_target_report()), encoding="utf-8")
    smoke = _smoke_report()
    smoke["monitor"]["files_scanned"] = 0
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")

    result = live_restart_smoke_evidence.main(
        [
            str(target_path),
            str(smoke_path),
            "--expected-repository-head",
            HEAD,
            "--expected-supervisor-fingerprint",
            FINGERPRINT,
            "--expected-targets",
            "2",
            "--compact",
        ]
    )

    assert result == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_emits_green_exit_for_complete_evidence(tmp_path, capsys):
    target_path = tmp_path / "target.json"
    smoke_path = tmp_path / "smoke.json"
    target_path.write_text(json.dumps(_target_report()), encoding="utf-8")
    smoke_path.write_text(json.dumps(_smoke_report()), encoding="utf-8")

    result = live_restart_smoke_evidence.main(
        [
            str(target_path),
            str(smoke_path),
            "--expected-repository-head",
            HEAD,
            "--expected-supervisor-fingerprint",
            FINGERPRINT,
            "--expected-targets",
            "2",
            "--compact",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_rejects_invalid_json_and_expected_arguments(tmp_path, capsys):
    target_path = tmp_path / "target.json"
    smoke_path = tmp_path / "smoke.json"
    target_path.write_text("[]", encoding="utf-8")
    smoke_path.write_text(json.dumps(_smoke_report()), encoding="utf-8")

    with pytest.raises(SystemExit) as invalid_json:
        live_restart_smoke_evidence.main(
            [
                str(target_path),
                str(smoke_path),
                "--expected-repository-head",
                HEAD,
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
                "--expected-targets",
                "2",
            ]
        )
    assert invalid_json.value.code == 2
    assert "target report must be a JSON object" in capsys.readouterr().err

    target_path.write_text(json.dumps(_target_report()), encoding="utf-8")
    with pytest.raises(SystemExit) as invalid_expected:
        live_restart_smoke_evidence.main(
            [
                str(target_path),
                str(smoke_path),
                "--expected-repository-head",
                "ABC",
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
                "--expected-targets",
                "0",
            ]
        )
    assert invalid_expected.value.code == 2
    assert "expected_repository_head" in capsys.readouterr().err


def test_cli_rejects_oversized_json_without_reading_beyond_the_limit(
    tmp_path, capsys, monkeypatch
):
    target_path = tmp_path / "target.json"
    smoke_path = tmp_path / "smoke.json"
    target_path.write_bytes(b"x" * 9)
    smoke_path.write_text(json.dumps(_smoke_report()), encoding="utf-8")
    monkeypatch.setattr(live_restart_smoke_evidence, "MAX_INPUT_JSON_BYTES", 8)

    with pytest.raises(SystemExit) as oversized:
        live_restart_smoke_evidence.main(
            [
                str(target_path),
                str(smoke_path),
                "--expected-repository-head",
                HEAD,
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
                "--expected-targets",
                "2",
            ]
        )

    assert oversized.value.code == 2
    assert "within the input limit" in capsys.readouterr().err
