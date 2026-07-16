from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

import live.restart_smoke_targets as target_module
from live.restart_smoke_targets import build_live_restart_target_report
from passivbot_cli import main as cli_main
from tools import live_restart_target_report


def _process_report(*names: str) -> dict:
    expected = [
        {
            "name": name,
            "command": f"passivbot live configs/private.json -u {name}",
            "config_path": "configs/private.json",
            "match_count": 1,
            "matched_processes": [
                {
                    "pid": index + 100,
                    "ppid": 20 + index * 10,
                }
            ],
        }
        for index, name in enumerate(names)
    ]
    return {
        "enabled": True,
        "ok": True,
        "hard_failures": 0,
        "expected_total": len(expected),
        "matched_expected": len(expected),
        "running_live_total": len(expected),
        "expected": expected,
        "missing_expected": [],
        "duplicate_configured_command_matches": [],
        "extra_passivbot_live_processes": [],
        "unexpected_running": [],
        "config_checks": {
            "enabled": True,
            "ok": True,
            "checked": len(expected),
            "skipped": 0,
            "hard_failures": 0,
            "issues": [],
        },
    }


def _pane(session: str, window: str, index: int, pid: int) -> dict:
    return {
        "pane_id": f"%{pid}",
        "session_name": session,
        "window_name": window,
        "pane_index": index,
        "pane_pid": pid,
        "current_command": "python3",
    }


def test_live_restart_target_report_resolves_exact_panes_and_ignores_other_sessions(
    monkeypatch,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01", "gateio_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (
            [
                _pane("misc", "bash", 0, 10),
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "gateio_01", 0, 30),
            ],
            None,
        ),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["expected_targets"] == 2
    assert report["resolved_targets"] == 2
    assert report["session_panes"] == 2
    assert [target["target"] for target in report["targets"]] == [
        "%20",
        "%30",
    ]
    assert [target["pane_pid"] for target in report["targets"]] == [20, 30]
    assert [target["process_pid"] for target in report["targets"]] == [100, 101]
    assert {
        target["ownership_proof"] for target in report["targets"]
    } == {"matched_process_ppid_equals_pane_pid"}
    assert report["extra_panes"] == []
    assert report["issues"] == []
    assert report["safety"]["process_control"] is False
    assert report["safety"]["signals_processes"] is False
    assert report["safety"]["starts_processes"] is False
    assert "passivbot live" not in json.dumps(report, sort_keys=True)
    assert "private.json" not in json.dumps(report, sort_keys=True)


def test_tmux_pane_inventory_only_lists_and_bounds_metadata(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="%12\tpassivbot\tbinance_01\t0\t856294\tpython3\n",
            stderr="",
        )

    monkeypatch.setattr(target_module.subprocess, "run", fake_run)

    panes, error = target_module._tmux_pane_inventory()

    assert error is None
    assert captured["command"] == [
        "tmux",
        "list-panes",
        "-a",
        "-F",
        target_module.TMUX_PANE_FORMAT,
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "timeout": 5.0,
        "check": False,
    }
    assert panes == [
        {
            "pane_id": "%12",
            "session_name": "passivbot",
            "window_name": "binance_01",
            "pane_index": 0,
            "pane_pid": 856294,
            "current_command": "python3",
        }
    ]


def test_tmux_pane_inventory_rejects_malformed_rows(monkeypatch):
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-a-valid-row\n",
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_malformed_row",
    )


def test_tmux_pane_inventory_rejects_oversized_identifiers(monkeypatch):
    long_name = "x" * (target_module.MAX_TMUX_IDENTIFIER_CHARS + 1)
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"%1\tpassivbot\t{long_name}\t0\t20\tpython3\n",
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_invalid_identifier",
    )


def test_tmux_pane_inventory_rejects_duplicate_pane_ids(monkeypatch):
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "%1\tpassivbot\tbinance_01\t0\t20\tpython3\n"
                "%1\tpassivbot\tgateio_01\t0\t30\tpython3\n"
            ),
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_duplicate_id",
    )


@pytest.mark.parametrize(
    ("panes", "expected_code"),
    [
        ([], "tmux_target_missing"),
        (
            [
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "binance_01", 1, 21),
            ],
            "tmux_target_duplicated",
        ),
    ],
)
def test_live_restart_target_report_fails_missing_or_duplicate_target(
    monkeypatch,
    panes,
    expected_code,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (panes, None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["resolved_targets"] == 0
    assert {issue["code"] for issue in report["issues"]} == {expected_code}


def test_live_restart_target_report_fails_unconfigured_pane_in_target_session(
    monkeypatch,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (
            [
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "unexpected", 0, 30),
            ],
            None,
        ),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["resolved_targets"] == 1
    assert report["extra_panes"][0]["window_name"] == "unexpected"
    assert report["issues"] == [
        {
            "code": "unconfigured_session_panes",
            "severity": "error",
            "count": 1,
        }
    ]


def test_live_restart_target_report_fails_process_parent_mismatch(monkeypatch):
    processes = _process_report("binance_01")
    processes["expected"][0]["matched_processes"][0]["ppid"] = 999
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([_pane("passivbot", "binance_01", 0, 20)], None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["resolved_targets"] == 0
    assert report["issues"] == [
        {
            "code": "tmux_process_parent_mismatch",
            "severity": "error",
            "window_name": "binance_01",
            "pane_pid": 20,
            "process_pid": 100,
            "process_ppid": 999,
        }
    ]


def test_live_restart_target_report_accepts_direct_pane_process(monkeypatch):
    processes = _process_report("binance_01")
    processes["expected"][0]["matched_processes"][0] = {
        "pid": 20,
        "ppid": 1,
    }
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([_pane("passivbot", "binance_01", 0, 20)], None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is True
    assert report["targets"][0]["ownership_proof"] == (
        "matched_process_pid_equals_pane_pid"
    )


def test_live_restart_target_report_propagates_process_and_tmux_failures(
    monkeypatch,
):
    processes = _process_report("binance_01")
    processes["ok"] = False
    processes["hard_failures"] = 2
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([], "tmux_list_panes_failed"),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 4
    assert report["processes"]["hard_failures"] == 2
    assert {issue["code"] for issue in report["issues"]} == {
        "tmux_list_panes_failed",
        "tmux_target_missing",
    }


def test_live_restart_target_report_requires_explicit_session(monkeypatch):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report(),
    )

    with pytest.raises(ValueError, match="session_name must be non-empty"):
        build_live_restart_target_report("bots.yaml", session_name="")

    with pytest.raises(ValueError, match="session_name must not exceed"):
        build_live_restart_target_report(
            "bots.yaml",
            session_name="x" * (target_module.MAX_TMUX_IDENTIFIER_CHARS + 1),
        )


def test_live_restart_target_report_cli_preserves_verdict(monkeypatch, capsys):
    captured = {}

    def fake_report(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ok": False, "hard_failures": 3, "targets": []}

    monkeypatch.setattr(
        live_restart_target_report,
        "build_live_restart_target_report",
        fake_report,
    )

    assert (
        live_restart_target_report.main(
            ["bots.yaml", "--session-name", "passivbot", "--compact"]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["hard_failures"] == 3
    assert captured["args"] == ("bots.yaml",)
    assert captured["kwargs"]["session_name"] == "passivbot"


def test_live_restart_target_report_tool_dispatch_forwards_module(monkeypatch):
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
                "live-restart-target-report",
                "bots.yaml",
                "--session-name",
                "passivbot",
            ]
        )
        == 0
    )
    assert captured == {
        "module_name": "tools.live_restart_target_report",
        "argv": [
            "passivbot tool live-restart-target-report",
            "bots.yaml",
            "--session-name",
            "passivbot",
        ],
        "prog_env": "passivbot tool live-restart-target-report",
    }
