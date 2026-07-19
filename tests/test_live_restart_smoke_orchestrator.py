from __future__ import annotations

import json
import os
import sys

import pytest

import live.restart_smoke_orchestrator as orchestrator_module
from live.restart_smoke_orchestrator import execute_live_restart_smoke
from passivbot_cli import main as cli_main
from tools import live_restart_smoke_run

HEAD = "a" * 40
RUST_FINGERPRINT = "b" * 64
SUPERVISOR_FINGERPRINT = "c" * 64
TARGETS = 2


def _target_preflight() -> dict:
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": True,
        "hard_failures": 0,
        "issues": [],
        "extra_panes": [],
        "expected_targets": TARGETS,
        "resolved_targets": TARGETS,
        "relaunch_ready_targets": TARGETS,
        "relaunch_unready_targets": 0,
        "supervisor_contract": {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": SUPERVISOR_FINGERPRINT,
            "target_count": TARGETS,
            "command_content_exposed": False,
        },
        "sampling": {
            "requested_samples": 3,
            "collected_samples": 3,
            "successful_samples": 3,
            "failed_samples": 0,
            "failed_sample_issues": [],
            "stable": True,
            "supervisor_contract_stable": True,
            "supervisor_contract_changed": False,
            "stable_targets": TARGETS,
            "changed_target_count": 0,
            "changed_targets_truncated": 0,
            "changed_targets": [],
        },
        "targets": [
            {
                "window_name": "private_bot_one",
                "pane_id": "%100",
                "process_pid": 1000,
            },
            {
                "window_name": "private_bot_two",
                "pane_id": "%101",
                "process_pid": 1001,
            },
        ],
    }


def _restart_report(*, ok: bool = True, action_started: bool = True) -> dict:
    issues = [] if ok else [{"code": "restart_problem", "severity": "error"}]
    rows = [
        {
            "window_name": f"private_bot_{index}",
            "pane_id": f"%{100 + index}",
            "old_process_pid": 1000 + index,
            "stop_requested": True,
            "exited": True,
            "relaunch_requested": True,
            "relaunch_succeeded": True,
        }
        for index in range(TARGETS)
    ]
    return {
        "tool": "live-restart-executor",
        "schema_version": 3,
        "ok": ok,
        "outcome": (
            "completed"
            if ok
            else (
                "manual_recovery_required"
                if action_started
                else "preflight_failed"
            )
        ),
        "action_started": action_started,
        "hard_failures": 0 if ok else 1,
        "issues": issues,
        "targets": rows if action_started else [],
        "verification": (
            {
                "ok": True,
                "hard_failures": 0,
                "expected_targets": TARGETS,
                "resolved_targets": TARGETS,
                "relaunch_ready_targets": TARGETS,
                "supervisor_contract": {
                    "fingerprint": SUPERVISOR_FINGERPRINT,
                    "target_count": TARGETS,
                    "command_content_exposed": False,
                },
            }
            if ok
            else {}
        ),
        "private_command": "passivbot live private-account",
    }


def _smoke_report(*, ok: bool = True) -> dict:
    return {
        "tool": "live-restart-smoke-collect",
        "schema_version": 1,
        "ok": ok,
        "hard_failures": 0 if ok else 1,
        "issues": [] if ok else [{"code": "startup_event_missing"}],
        "gates": {"repository": {"ok": ok}},
        "evidence": {"startup": {"targets": TARGETS if ok else 1}},
    }


def _kwargs() -> dict:
    return {
        "supervisor_config": "private-supervisor.yaml",
        "session_name": "private-session",
        "monitor_root": "private-monitor",
        "logs_root": "private-logs",
        "expected_repository_head": HEAD,
        "expected_rust_source_fingerprint": RUST_FINGERPRINT,
        "expected_supervisor_fingerprint": SUPERVISOR_FINGERPRINT,
        "expected_targets": TARGETS,
        "smoke_wait_s": 10.0,
        "preflight_samples": 3,
        "preflight_interval_s": 0.5,
        "shutdown_timeout_s": 20.0,
        "startup_timeout_s": 30.0,
        "poll_interval_s": 0.25,
        "verification_samples": 3,
        "verification_interval_s": 0.5,
        "smoke_target_samples": 3,
        "smoke_target_interval_s": 0.25,
        "execute": True,
    }


def _install_happy_dependencies(monkeypatch) -> dict[str, list]:
    calls: dict[str, list] = {
        "target": [],
        "restart": [],
        "sleep": [],
        "smoke": [],
    }
    clock = iter([1_000_000, 1_010_000])
    monkeypatch.setattr(orchestrator_module, "_epoch_ms", lambda: next(clock))
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_target_report",
        lambda *args, **kwargs: (
            calls["target"].append((args, kwargs)) or _target_preflight()
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "execute_live_restart",
        lambda *args, **kwargs: (
            calls["restart"].append((args, kwargs)) or _restart_report()
        ),
    )
    monkeypatch.setattr(
        orchestrator_module.time,
        "sleep",
        lambda value: calls["sleep"].append(value),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_smoke_collection",
        lambda *args, **kwargs: (
            calls["smoke"].append((args, kwargs)) or _smoke_report()
        ),
    )
    return calls


def test_restart_smoke_runs_exact_sequence_and_sanitizes_restart(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)

    report = execute_live_restart_smoke(**_kwargs())

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["outcome"] == "completed"
    assert report["action_started"] is True
    assert report["window"] == {
        "since_ms": 1_000_000,
        "until_ms": 1_010_000,
        "observation_wait_s": 10.0,
    }
    assert report["restart"] == {
        "ok": True,
        "contract_valid": True,
        "outcome": "completed",
        "action_started": True,
        "hard_failures": 0,
        "targets": TARGETS,
        "stop_requested": TARGETS,
        "exited": TARGETS,
        "relaunch_requested": TARGETS,
        "relaunch_succeeded": TARGETS,
        "verified_targets": TARGETS,
        "verification_ok": True,
    }
    assert calls["sleep"] == [10.0]
    assert calls["target"][0][1] == {
        "session_name": "private-session",
        "config_base_dir": orchestrator_module.Path.cwd(),
        "samples": 3,
        "sample_interval_s": 0.5,
    }
    assert calls["restart"][0][1]["execute"] is True
    assert calls["smoke"][0][1]["since_ms"] == 1_000_000
    assert calls["smoke"][0][1]["until_ms"] == 1_010_000
    serialized = json.dumps(report, sort_keys=True)
    for private_value in (
        "private-supervisor.yaml",
        "private-session",
        "private-monitor",
        "private-logs",
        "private_bot_one",
        "%100",
        "old_process_pid",
        "passivbot live private-account",
        SUPERVISOR_FINGERPRINT,
    ):
        assert private_value not in serialized


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        (("resolved_targets",), 1),
        (("issues",), [{"code": "hidden_failure"}]),
        (("sampling", "failed_samples"), 1),
        (("sampling", "changed_target_count"), 1),
        (("supervisor_contract", "fingerprint"), "d" * 64),
    ],
)
def test_outer_preflight_failure_prevents_action(monkeypatch, mutation, value):
    calls = _install_happy_dependencies(monkeypatch)
    failed = _target_preflight()
    row = failed
    for key in mutation[:-1]:
        row = row[key]
    row[mutation[-1]] = value
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_target_report",
        lambda *_args, **_kwargs: failed,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["outcome"] == "preflight_failed"
    assert report["issues"] == [
        {"code": "target_preflight_failed", "severity": "error"}
    ]
    assert calls["restart"] == []
    assert calls["sleep"] == []
    assert calls["smoke"] == []


@pytest.mark.parametrize("action_started", [False, True])
def test_executor_failure_skips_sleep_and_collection(
    monkeypatch, action_started
):
    calls = _install_happy_dependencies(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "execute_live_restart",
        lambda *_args, **_kwargs: _restart_report(
            ok=False, action_started=action_started
        ),
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == (
        "manual_recovery_required" if action_started else "preflight_failed"
    )
    assert report["issues"] == [
        {"code": "restart_failed", "severity": "error"}
    ]
    assert calls["sleep"] == []
    assert calls["smoke"] == []


def test_malformed_green_executor_report_requires_manual_recovery(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    malformed = _restart_report()
    malformed["targets"][0]["relaunch_succeeded"] = False
    monkeypatch.setattr(
        orchestrator_module,
        "execute_live_restart",
        lambda *_args, **_kwargs: malformed,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "manual_recovery_required"
    assert report["restart"]["contract_valid"] is False
    assert report["issues"] == [
        {"code": "restart_contract_invalid", "severity": "error"}
    ]
    assert calls["sleep"] == []
    assert calls["smoke"] == []


def test_malformed_executor_outcome_is_not_projected(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    malformed = _restart_report()
    malformed["outcome"] = "private-account-name"
    monkeypatch.setattr(
        orchestrator_module,
        "execute_live_restart",
        lambda *_args, **_kwargs: malformed,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["restart"]["outcome"] is None
    assert report["restart"]["contract_valid"] is False
    assert "private-account-name" not in json.dumps(report, sort_keys=True)
    assert calls["sleep"] == []
    assert calls["smoke"] == []


@pytest.mark.parametrize("malformed", [None, [], {"ok": True}])
def test_malformed_executor_shape_requires_manual_recovery(
    monkeypatch, malformed
):
    calls = _install_happy_dependencies(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "execute_live_restart",
        lambda *_args, **_kwargs: malformed,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "manual_recovery_required"
    assert report["action_started"] is True
    assert report["restart"]["contract_valid"] is False
    assert report["issues"] == [
        {"code": "restart_contract_invalid", "severity": "error"}
    ]
    assert calls["sleep"] == []
    assert calls["smoke"] == []


def test_red_smoke_leaves_restart_complete_and_returns_red(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_smoke_collection",
        lambda *_args, **_kwargs: _smoke_report(ok=False),
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["ok"] is False
    assert report["action_started"] is True
    assert report["outcome"] == "restart_completed_smoke_failed"
    assert report["issues"] == [{"code": "smoke_failed", "severity": "error"}]
    assert calls["sleep"] == [10.0]


def test_malformed_smoke_report_is_not_projected(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    malformed = _smoke_report()
    malformed["schema_version"] = True
    malformed["private_value"] = "private-monitor-path"
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_smoke_collection",
        lambda *_args, **_kwargs: malformed,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "restart_completed_smoke_failed"
    assert report["smoke"] is None
    assert report["issues"] == [
        {"code": "smoke_contract_invalid", "severity": "error"}
    ]
    assert "private-monitor-path" not in json.dumps(report, sort_keys=True)
    assert calls["sleep"] == [10.0]


def test_non_mapping_smoke_report_is_not_projected(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_smoke_collection",
        lambda *_args, **_kwargs: ["private-monitor-path"],
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "restart_completed_smoke_failed"
    assert report["smoke"] is None
    assert "private-monitor-path" not in json.dumps(report, sort_keys=True)
    assert calls["sleep"] == [10.0]


def test_smoke_collection_error_is_sanitized_after_restart(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)

    def fail_collection(*_args, **_kwargs):
        raise OSError("private-monitor-path")

    monkeypatch.setattr(
        orchestrator_module,
        "build_live_restart_smoke_collection",
        fail_collection,
    )

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "restart_completed_smoke_failed"
    assert report["issues"] == [
        {
            "code": "smoke_collection_failed",
            "severity": "error",
            "error_class": "OSError",
        }
    ]
    assert "private-monitor-path" not in json.dumps(report, sort_keys=True)
    assert calls["sleep"] == [10.0]


def test_nonadvancing_clock_skips_collection(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    clock = iter([1_000_000, 1_000_000])
    monkeypatch.setattr(orchestrator_module, "_epoch_ms", lambda: next(clock))

    report = execute_live_restart_smoke(**_kwargs())

    assert report["outcome"] == "restart_completed_smoke_failed"
    assert report["issues"] == [
        {"code": "smoke_window_clock_invalid", "severity": "error"}
    ]
    assert calls["smoke"] == []


@pytest.mark.parametrize(
    "override",
    [
        {"expected_repository_head": "bad"},
        {"expected_rust_source_fingerprint": "bad"},
        {"expected_supervisor_fingerprint": "bad"},
        {"expected_targets": True},
        {"monitor_root": ""},
        {"logs_root": ""},
        {"smoke_wait_s": 0.0},
        {"smoke_wait_s": 1800.1},
        {"smoke_target_samples": 1},
        {"smoke_target_samples": True},
        {"smoke_target_interval_s": -0.1},
        {"smoke_target_interval_s": True},
    ],
)
def test_invalid_confirmation_inputs_fail_before_preflight(
    monkeypatch, override
):
    calls = _install_happy_dependencies(monkeypatch)
    kwargs = _kwargs()
    kwargs.update(override)

    with pytest.raises(ValueError):
        execute_live_restart_smoke(**kwargs)

    assert calls["target"] == []
    assert calls["restart"] == []


def test_execute_confirmation_is_required_before_preflight(monkeypatch):
    calls = _install_happy_dependencies(monkeypatch)
    kwargs = _kwargs()
    kwargs["execute"] = False

    with pytest.raises(ValueError, match="execute must be true"):
        execute_live_restart_smoke(**kwargs)

    assert calls["target"] == []


def test_cli_requires_execute(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_run.main(
            [
                "supervisor.yaml",
                "monitor",
                "--session-name",
                "passivbot",
                "--expected-repository-head",
                HEAD,
                "--expected-rust-source-fingerprint",
                RUST_FINGERPRINT,
                "--expected-supervisor-fingerprint",
                SUPERVISOR_FINGERPRINT,
                "--expected-targets",
                str(TARGETS),
            ]
        )

    assert exc_info.value.code == 2
    assert "--execute is required" in capsys.readouterr().err


def test_cli_outputs_compact_report(monkeypatch, capsys):
    green = {"tool": "live-restart-smoke-run", "ok": True}
    monkeypatch.setattr(
        live_restart_smoke_run,
        "execute_live_restart_smoke",
        lambda *_args, **_kwargs: green,
    )

    exit_code = live_restart_smoke_run.main(
        [
            "supervisor.yaml",
            "monitor",
            "--session-name",
            "passivbot",
            "--expected-repository-head",
            HEAD,
            "--expected-rust-source-fingerprint",
            RUST_FINGERPRINT,
            "--expected-supervisor-fingerprint",
            SUPERVISOR_FINGERPRINT,
            "--expected-targets",
            str(TARGETS),
            "--execute",
            "--compact",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == json.dumps(green, sort_keys=True) + "\n"


def test_unified_cli_dispatches_restart_smoke_run(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(
        cli_main, "_invoke_module_main", fake_invoke_module_main
    )
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])
    argv = [
        "tool",
        "live-restart-smoke-run",
        "supervisor.yaml",
        "monitor",
        "--session-name",
        "passivbot",
        "--expected-repository-head",
        HEAD,
        "--expected-rust-source-fingerprint",
        RUST_FINGERPRINT,
        "--expected-supervisor-fingerprint",
        SUPERVISOR_FINGERPRINT,
        "--expected-targets",
        str(TARGETS),
        "--execute",
    ]

    assert cli_main.main(argv) == 0
    assert captured["module_name"] == "tools.live_restart_smoke_run"
    assert captured["argv"][0] == "passivbot tool live-restart-smoke-run"
    assert captured["argv"][1:] == argv[2:]
    assert captured["prog_env"] == "passivbot tool live-restart-smoke-run"


def test_safety_contract_keeps_force_and_broad_signals_disabled():
    safety = orchestrator_module.SAFETY_CONTRACT

    assert safety["signals_exact_tmux_panes_only"] is True
    assert safety["automatic_force_signal"] is False
    assert safety["broad_process_pattern_signals"] is False
    assert safety["direct_exchange_access"] is False
    assert safety["configured_live_processes_may_contact_exchanges"] is True
    assert safety["writes_report_files"] is False
