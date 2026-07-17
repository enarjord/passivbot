from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

import live.restart_executor as executor_module
from live.restart_executor import execute_live_restart
from live.smoke_report import (
    _parse_tmuxp_live_commands,
    _supervisor_command_contract,
    parse_tmuxp_live_commands,
)
from passivbot_cli import main as cli_main
from tools import live_restart_executor


FINGERPRINT = "a" * 64


def _target(
    name: str,
    pane_id: str,
    pane_pid: int,
    process_pid: int,
) -> dict:
    return {
        "window_name": name,
        "target": pane_id,
        "pane_id": pane_id,
        "pane_index": 0,
        "pane_pid": pane_pid,
        "process_pid": process_pid,
        "ownership_proof": "matched_process_ppid_equals_pane_pid",
        "relaunch": {
            "ready": True,
            "method": "exact_pane_input_after_verified_exit",
            "reason": None,
            "command_source": "supervisor_config",
            "requires_process_exit": True,
            "requires_post_stop_pane_recheck": True,
        },
    }


def _report(*targets: dict, ok: bool = True) -> dict:
    target_rows = deepcopy(list(targets)) or [
        _target("binance_01", "%10", 100, 200)
    ]
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": ok,
        "hard_failures": 0 if ok else 1,
        "session_name": "passivbot",
        "expected_targets": len(target_rows),
        "resolved_targets": len(target_rows),
        "relaunch_ready_targets": len(target_rows),
        "supervisor_contract": {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": FINGERPRINT,
            "target_count": len(target_rows),
            "command_content_exposed": False,
        },
        "sampling": {
            "stable": ok,
            "supervisor_contract_stable": ok,
            "requested_samples": 3,
            "collected_samples": 3,
        },
        "targets": target_rows,
    }


def _install_happy_dependencies(monkeypatch, targets: list[dict]) -> dict:
    report = _report(*targets)
    calls: dict[str, list] = {"stop": [], "start": [], "reports": []}

    def build_report(*_args, **kwargs):
        calls["reports"].append(kwargs)
        return deepcopy(report)

    monkeypatch.setattr(executor_module, "build_live_restart_target_report", build_report)
    monkeypatch.setattr(
        executor_module,
        "_load_launch_snapshot",
        lambda _path: (
            {
                target["window_name"]: {
                    "command": f"private launch {target['window_name']}",
                    "match_key": f"passivbot live {target['window_name']}",
                }
                for target in targets
            },
            FINGERPRINT,
            None,
        ),
    )
    monkeypatch.setattr(
        executor_module,
        "_send_graceful_interrupt",
        lambda pane_id: calls["stop"].append(pane_id),
    )
    monkeypatch.setattr(
        executor_module,
        "_send_launch_command",
        lambda pane_id, command: calls["start"].append((pane_id, command)),
    )
    monkeypatch.setattr(
        executor_module,
        "_wait_for_process_exits",
        lambda rows, **_kwargs: (
            {int(row["process_pid"]): float(index + 1) for index, row in enumerate(rows)},
            [],
        ),
    )
    monkeypatch.setattr(
        executor_module,
        "_post_stop_process_recheck",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        executor_module,
        "_wait_for_post_stop_pane_recheck",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        executor_module,
        "_wait_for_startup_report",
        lambda *_args, **_kwargs: deepcopy(report),
    )
    return calls


def _execute(**kwargs) -> dict:
    return execute_live_restart(
        "bots.yaml",
        session_name="passivbot",
        expected_supervisor_fingerprint=FINGERPRINT,
        preflight_samples=2,
        preflight_interval_s=0.1,
        shutdown_timeout_s=1.0,
        startup_timeout_s=1.0,
        poll_interval_s=0.1,
        verification_samples=2,
        verification_interval_s=0.1,
        execute=True,
        **kwargs,
    )


def test_live_restart_executor_restarts_only_exact_verified_panes(monkeypatch):
    targets = [
        _target("binance_01", "%10", 100, 200),
        _target("gateio_01", "%11", 101, 201),
    ]
    calls = _install_happy_dependencies(monkeypatch, targets)

    report = _execute()

    assert report["ok"] is True
    assert report["outcome"] == "completed"
    assert calls["stop"] == ["%10", "%11"]
    assert calls["start"] == [
        ("%10", "private launch binance_01"),
        ("%11", "private launch gateio_01"),
    ]
    assert [row["shutdown_elapsed_s"] for row in report["targets"]] == [1.0, 2.0]
    assert all(row["relaunch_succeeded"] for row in report["targets"])
    serialized = json.dumps(report, sort_keys=True)
    assert "private launch" not in serialized
    assert report["safety"]["automatic_force_signal"] is False
    assert report["safety"]["broad_process_pattern_signals"] is False
    assert report["safety"]["direct_file_writes"] is False
    assert report["safety"]["configured_live_processes_may_write_files"] is True
    assert report["safety"]["configured_live_processes_may_contact_exchanges"] is True


def test_live_restart_executor_fails_closed_before_action_on_fingerprint_mismatch(
    monkeypatch,
):
    calls = _install_happy_dependencies(
        monkeypatch, [_target("binance_01", "%10", 100, 200)]
    )

    report = execute_live_restart(
        "bots.yaml",
        session_name="passivbot",
        expected_supervisor_fingerprint="b" * 64,
        preflight_samples=2,
        preflight_interval_s=0.1,
        execute=True,
    )

    assert report["ok"] is False
    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "supervisor_fingerprint_mismatch"
    }
    assert calls["stop"] == []
    assert calls["start"] == []


def test_live_restart_executor_fails_closed_when_snapshot_changes_after_preflight(
    monkeypatch,
):
    calls = _install_happy_dependencies(
        monkeypatch, [_target("binance_01", "%10", 100, 200)]
    )
    monkeypatch.setattr(
        executor_module,
        "_load_launch_snapshot",
        lambda _path: (
            {
                "binance_01": {
                    "command": "private",
                    "match_key": "passivbot live binance_01",
                }
            },
            "b" * 64,
            None,
        ),
    )

    report = _execute()

    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "supervisor_changed_after_preflight"
    }
    assert calls["stop"] == []


def test_live_restart_executor_relaunches_only_exited_targets_after_timeout(
    monkeypatch,
):
    targets = [
        _target("binance_01", "%10", 100, 200),
        _target("gateio_01", "%11", 101, 201),
    ]
    calls = _install_happy_dependencies(monkeypatch, targets)
    monkeypatch.setattr(
        executor_module,
        "_wait_for_process_exits",
        lambda _rows, **_kwargs: ({200: 0.5}, [201]),
    )

    report = _execute()

    assert report["ok"] is False
    assert report["outcome"] == "recovered_with_errors"
    assert {issue["code"] for issue in report["issues"]} == {"shutdown_timeout"}
    assert calls["start"] == [("%10", "private launch binance_01")]
    by_name = {row["window_name"]: row for row in report["targets"]}
    assert by_name["binance_01"]["relaunch_succeeded"] is True
    assert by_name["gateio_01"]["relaunch_requested"] is False


def test_live_restart_executor_halts_relaunch_when_pane_identity_changes(monkeypatch):
    calls = _install_happy_dependencies(
        monkeypatch, [_target("binance_01", "%10", 100, 200)]
    )
    monkeypatch.setattr(
        executor_module,
        "_wait_for_post_stop_pane_recheck",
        lambda *_args, **_kwargs: (False, "pane_identity_changed"),
    )

    report = _execute()

    assert report["ok"] is False
    assert report["outcome"] == "manual_recovery_required"
    assert {issue["code"] for issue in report["issues"]} == {
        "post_stop_pane_recheck_failed"
    }
    assert calls["start"] == []


def test_live_restart_executor_halts_relaunch_when_process_reappears(monkeypatch):
    calls = _install_happy_dependencies(
        monkeypatch, [_target("binance_01", "%10", 100, 200)]
    )
    monkeypatch.setattr(
        executor_module,
        "_post_stop_process_recheck",
        lambda *_args, **_kwargs: (False, "configured_process_count_changed"),
    )

    report = _execute()

    assert report["outcome"] == "manual_recovery_required"
    assert {issue["code"] for issue in report["issues"]} == {
        "post_stop_process_recheck_failed"
    }
    assert calls["start"] == []


def test_live_restart_executor_rechecks_processes_immediately_before_relaunch(
    monkeypatch,
):
    calls = _install_happy_dependencies(
        monkeypatch, [_target("binance_01", "%10", 100, 200)]
    )
    rechecks = iter(
        [
            (True, None),
            (False, "configured_process_count_changed"),
        ]
    )
    monkeypatch.setattr(
        executor_module,
        "_post_stop_process_recheck",
        lambda *_args, **_kwargs: next(rechecks),
    )

    report = _execute()

    assert report["outcome"] == "manual_recovery_required"
    assert {issue["code"] for issue in report["issues"]} == {
        "process_changed_before_relaunch"
    }
    assert calls["start"] == []


def test_live_restart_executor_rechecks_supervisor_before_relaunch(monkeypatch):
    targets = [_target("binance_01", "%10", 100, 200)]
    calls = _install_happy_dependencies(monkeypatch, targets)
    snapshots = iter(
        [
            (
                {
                    "binance_01": {
                        "command": "private launch binance_01",
                        "match_key": "passivbot live binance_01",
                    }
                },
                FINGERPRINT,
                None,
            ),
            (
                {
                    "binance_01": {
                        "command": "changed private launch",
                        "match_key": "passivbot live binance_01",
                    }
                },
                "b" * 64,
                None,
            ),
        ]
    )
    monkeypatch.setattr(
        executor_module,
        "_load_launch_snapshot",
        lambda _path: next(snapshots),
    )

    report = _execute()

    assert report["outcome"] == "manual_recovery_required"
    assert {issue["code"] for issue in report["issues"]} == {
        "supervisor_changed_before_relaunch"
    }
    assert calls["start"] == []


def test_live_restart_executor_rejects_target_change_after_preflight(monkeypatch):
    targets = [_target("binance_01", "%10", 100, 200)]
    calls = _install_happy_dependencies(monkeypatch, targets)
    report_calls = 0

    def build_report(*_args, **_kwargs):
        nonlocal report_calls
        report_calls += 1
        report = _report(*targets)
        if report_calls == 2:
            report["targets"][0]["process_pid"] = 999
        return report

    monkeypatch.setattr(executor_module, "build_live_restart_target_report", build_report)

    report = _execute()

    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "target_changed_after_preflight"
    }
    assert calls["stop"] == []


def test_tmux_actions_use_exact_pane_without_shell_or_broad_patterns(monkeypatch):
    calls: list[list[str]] = []

    def run(args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(executor_module.subprocess, "run", run)

    assert executor_module._send_graceful_interrupt("%17") is None
    assert executor_module._send_launch_command("%17", "private command") is None

    assert calls == [
        ["tmux", "send-keys", "-t", "%17", "C-c"],
        ["tmux", "send-keys", "-t", "%17", "-l", "--", "private command"],
        ["tmux", "send-keys", "-t", "%17", "Enter"],
    ]
    serialized = json.dumps(calls)
    assert "pkill" not in serialized
    assert "pgrep" not in serialized
    assert "SIGTERM" not in serialized


def test_post_stop_recheck_requires_shell_only_for_relaunch_targets(monkeypatch):
    targets = [
        _target("binance_01", "%10", 100, 200),
        _target("gateio_01", "%11", 101, 201),
    ]
    panes = [
        {
            "pane_id": "%10",
            "session_name": "passivbot",
            "window_name": "binance_01",
            "pane_index": 0,
            "pane_pid": 100,
            "current_command": "bash",
        },
        {
            "pane_id": "%11",
            "session_name": "passivbot",
            "window_name": "gateio_01",
            "pane_index": 0,
            "pane_pid": 101,
            "current_command": "python3",
        },
    ]
    monkeypatch.setattr(executor_module, "_tmux_pane_inventory", lambda: (panes, None))

    assert executor_module._post_stop_pane_recheck(
        targets,
        session_name="passivbot",
        relaunch_pane_ids={"%10"},
    ) == (True, None)
    assert executor_module._post_stop_pane_recheck(
        targets,
        session_name="passivbot",
        relaunch_pane_ids={"%10", "%11"},
    ) == (False, "pane_not_at_shell_prompt")


def test_post_stop_process_recheck_prevents_duplicate_relaunch(monkeypatch):
    snapshot = {
        "binance_01": {
            "command": "private binance",
            "match_key": "passivbot live binance",
        },
        "gateio_01": {
            "command": "private gateio",
            "match_key": "passivbot live gateio",
        },
    }
    monkeypatch.setattr(
        executor_module,
        "_running_live_processes",
        lambda **_kwargs: {
            "scan_error": None,
            "running": [{"_match_key": "passivbot live gateio"}],
        },
    )
    assert executor_module._post_stop_process_recheck(
        snapshot,
        relaunch_window_names={"binance_01"},
    ) == (True, None)

    monkeypatch.setattr(
        executor_module,
        "_running_live_processes",
        lambda **_kwargs: {
            "scan_error": None,
            "running": [
                {"_match_key": "passivbot live binance"},
                {"_match_key": "passivbot live gateio"},
            ],
        },
    )
    assert executor_module._post_stop_process_recheck(
        snapshot,
        relaunch_window_names={"binance_01"},
    ) == (False, "configured_process_count_changed")


def test_live_restart_executor_rejects_final_fingerprint_drift(monkeypatch):
    targets = [_target("binance_01", "%10", 100, 200)]
    _install_happy_dependencies(monkeypatch, targets)
    changed = _report(*targets)
    changed["supervisor_contract"]["fingerprint"] = "b" * 64
    sampled_calls = 0

    def build_report(*_args, **kwargs):
        nonlocal sampled_calls
        if int(kwargs.get("samples") or 1) > 1:
            sampled_calls += 1
            return _report(*targets) if sampled_calls == 1 else deepcopy(changed)
        return _report(*targets)

    monkeypatch.setattr(executor_module, "build_live_restart_target_report", build_report)

    report = _execute()

    assert report["ok"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "supervisor_changed_during_restart"
    }


def test_private_launch_source_changes_fingerprint_without_public_exposure(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        "window_name: binance_01\n"
        "- cd /one && passivbot live bot.json -u binance_01\n",
        encoding="utf-8",
    )
    second.write_text(
        "window_name: binance_01\n"
        "- cd /two && passivbot live bot.json -u binance_01\n",
        encoding="utf-8",
    )

    first_private = _parse_tmuxp_live_commands(first)
    second_private = _parse_tmuxp_live_commands(second)
    first_public = parse_tmuxp_live_commands(first)

    assert first_private["expected"][0]["_match_key"] == second_private["expected"][0][
        "_match_key"
    ]
    assert _supervisor_command_contract(first_private["expected"])[
        "fingerprint"
    ] != _supervisor_command_contract(second_private["expected"])["fingerprint"]
    assert "_launch_command" not in first_public["expected"][0]
    assert "/one" not in json.dumps(
        _supervisor_command_contract(first_private["expected"]), sort_keys=True
    )


def test_live_restart_executor_cli_requires_explicit_execute(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_restart_executor.main(
            [
                "bots.yaml",
                "--session-name",
                "passivbot",
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
            ]
        )
    assert exc_info.value.code == 2
    assert "--execute is required" in capsys.readouterr().err


def test_live_restart_executor_cli_outputs_sanitized_report(monkeypatch, capsys):
    monkeypatch.setattr(
        live_restart_executor,
        "execute_live_restart",
        lambda *_args, **_kwargs: {"tool": "live-restart-executor", "ok": True},
    )

    exit_code = live_restart_executor.main(
        [
            "bots.yaml",
            "--session-name",
            "passivbot",
            "--expected-supervisor-fingerprint",
            FINGERPRINT,
            "--execute",
            "--compact",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "tool": "live-restart-executor",
    }


def test_live_restart_executor_tool_dispatch_forwards_module(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert (
        cli_main.main(
            [
                "tool",
                "live-restart-executor",
                "bots.yaml",
                "--session-name",
                "passivbot",
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
                "--execute",
            ]
        )
        == 0
    )
    assert captured == {
        "module_name": "tools.live_restart_executor",
        "argv": [
            "passivbot tool live-restart-executor",
            "bots.yaml",
            "--session-name",
            "passivbot",
            "--expected-supervisor-fingerprint",
            FINGERPRINT,
            "--execute",
        ],
        "prog_env": "passivbot tool live-restart-executor",
    }
