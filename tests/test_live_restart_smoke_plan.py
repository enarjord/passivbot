from __future__ import annotations

import json
import os
import sys

import pytest

from live.restart_smoke_plan import _display_path, build_live_restart_smoke_plan
from passivbot_cli import main as cli_main
from tools import live_restart_smoke_plan


def _write_supervisor(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_live_restart_smoke_plan_builds_plan_from_supervisor_config(tmp_path):
    supervisor_config = tmp_path / "bots_vps5.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
            "  - window_name: gateio_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u gateio_01",
        ],
    )

    report = build_live_restart_smoke_plan(
        supervisor_config,
        repo_path="/srv/passivbot",
        monitor_root="/srv/passivbot/monitor",
        logs_root="/srv/passivbot/logs",
        shutdown_timeout_s=45,
        startup_wait_s=120,
        smoke_window_minutes=15,
    )

    assert report["ok"] is True
    assert report["metadata"] == {
        "dry_run": True,
        "execute": False,
        "execution_available": False,
        "plan_only": True,
    }
    assert report["supervisor_config"]["expected_live_commands"] == 2
    assert [bot["account"] for bot in report["bots"]] == ["binance_01", "gateio_01"]
    assert report["bots"][0]["phases"][0]["planned_actions"][1]["execute"] is False
    assert "passivbot tool live-smoke-report" in report["smoke_report"]["command"]
    assert "--supervisor-config" in report["smoke_report"]["command"]
    assert "--recent-minutes 15" in report["smoke_report"]["command"]
    assert "--event-tail-lines 2000" in report["smoke_report"]["command"]
    assert "--max-event-files-per-bot 2" in report["smoke_report"]["command"]
    assert "--max-log-files 8" in report["smoke_report"]["command"]
    assert "--log-tail-lines 1200" in report["smoke_report"]["command"]
    assert "--max-log-matches 20" in report["smoke_report"]["command"]
    assert "--brief" in report["smoke_report"]["command"]
    assert "--summary" not in report["smoke_report"]["command"]
    assert report["process_signal_safety"]["strategy"] == (
        "exact_tmux_pane_or_exact_pid_only"
    )
    assert report["process_signal_safety"]["forbid_broad_process_pattern_signals"] is True
    assert "pkill -f 'passivbot live'" in report["process_signal_safety"][
        "unsafe_patterns"
    ]
    assert "exclude the controller process and its ancestors" in report[
        "process_signal_safety"
    ]["required_guards"]
    assert all(
        item["execute"] is False for item in report["timeout_escalation_ladder"]
    )
    assert "ssh" in report["execution_policy"]["rejected_operations"]
    assert "broad process-pattern kill/signal" in report["execution_policy"][
        "rejected_operations"
    ]
    assert "does_not_start_passivbot_live" in report["warnings"]


def test_live_restart_smoke_plan_redacts_and_bounds_configured_commands(tmp_path):
    supervisor_config = tmp_path / "bots.yaml"
    long_secret = "S" * 600
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            (
                "      - passivbot live configs/forager.json -u binance_01 "
                f"api_key=ABC123 secret={long_secret}"
            ),
        ],
    )

    report = build_live_restart_smoke_plan(supervisor_config)
    command = report["bots"][0]["command"]
    command_key = report["bots"][0]["command_key"]

    assert "ABC123" not in command
    assert long_secret not in command
    assert "[redacted]" in command
    assert len(command) <= len("...<truncated>") + 400
    assert len(command_key) <= len("...<truncated>") + 400


def test_live_restart_smoke_plan_display_path_collapses_user_prefixes():
    assert _display_path("/root/bots_vps5.yaml") == "~/bots_vps5.yaml"
    assert _display_path("/Users/alice/passivbot/monitor") == "~/passivbot/monitor"
    assert _display_path(None) is None


def test_live_restart_smoke_plan_reports_missing_supervisor_config(tmp_path):
    report = build_live_restart_smoke_plan(tmp_path / "missing.yaml")

    assert report["ok"] is False
    assert report["metadata"]["execute"] is False
    assert report["bots"] == []
    assert report["supervisor_config"]["exists"] is False
    assert {issue["code"] for issue in report["issues"]} == {"config_not_found"}


def test_live_restart_smoke_plan_reports_malformed_supervisor_config(tmp_path):
    supervisor_config = tmp_path / "bots.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: not_a_bot",
            "    panes:",
            "      - echo no live command here",
        ],
    )

    report = build_live_restart_smoke_plan(supervisor_config)

    assert report["ok"] is False
    assert report["bots"] == []
    assert report["supervisor_config"] == {
        "path": report["inputs"]["supervisor_config"],
        "exists": True,
        "error": None,
        "expected_live_commands": 0,
    }
    assert report["issues"] == [
        {
            "severity": "error",
            "code": "no_expected_live_commands",
            "message": "Supervisor config contains no parseable passivbot live commands.",
        }
    ]


def test_live_restart_smoke_plan_rejects_execution():
    with pytest.raises(
        NotImplementedError,
        match="execution is intentionally unavailable",
    ):
        build_live_restart_smoke_plan("bots.yaml", execute=True)


def test_live_restart_smoke_plan_cli_outputs_json(tmp_path, capsys):
    supervisor_config = tmp_path / "bots.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
        ],
    )

    assert (
        live_restart_smoke_plan.main(
            [str(supervisor_config), "--compact", "--shutdown-timeout-s", "30"]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["inputs"]["shutdown_timeout_s"] == 30
    assert report["inputs"]["smoke_event_tail_lines"] == 2000
    assert report["inputs"]["smoke_max_event_files_per_bot"] == 2
    assert report["inputs"]["smoke_max_log_files"] == 8
    assert report["inputs"]["smoke_log_tail_lines"] == 1200
    assert report["inputs"]["smoke_max_log_matches"] == 20
    assert "--event-tail-lines 2000" in report["smoke_report"]["command"]
    assert "--max-event-files-per-bot 2" in report["smoke_report"]["command"]
    assert "--max-log-files 8" in report["smoke_report"]["command"]
    assert "--log-tail-lines 1200" in report["smoke_report"]["command"]
    assert "--max-log-matches 20" in report["smoke_report"]["command"]
    assert "--brief" in report["smoke_report"]["command"]


def test_live_restart_smoke_plan_can_disable_planned_event_scan_bounds(
    tmp_path, capsys
):
    supervisor_config = tmp_path / "bots.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
        ],
    )

    assert (
        live_restart_smoke_plan.main(
            [
                str(supervisor_config),
                "--event-tail-lines",
                "0",
                "--max-event-files-per-bot",
                "0",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["smoke_event_tail_lines"] == 0
    assert report["inputs"]["smoke_max_event_files_per_bot"] == 0
    assert "--event-tail-lines" not in report["smoke_report"]["command"]
    assert "--max-event-files-per-bot" not in report["smoke_report"]["command"]


def test_live_restart_smoke_plan_can_disable_planned_log_scan_bounds(tmp_path, capsys):
    supervisor_config = tmp_path / "bots.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
        ],
    )

    assert (
        live_restart_smoke_plan.main(
            [
                str(supervisor_config),
                "--max-log-files",
                "0",
                "--log-tail-lines",
                "0",
                "--max-log-matches",
                "0",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["smoke_max_log_files"] == 0
    assert report["inputs"]["smoke_log_tail_lines"] == 0
    assert report["inputs"]["smoke_max_log_matches"] == 0
    assert "--max-log-files" not in report["smoke_report"]["command"]
    assert "--log-tail-lines" not in report["smoke_report"]["command"]
    assert "--max-log-matches" not in report["smoke_report"]["command"]


def test_live_restart_smoke_plan_can_plan_summary_or_full_smoke_projection(
    tmp_path, capsys
):
    supervisor_config = tmp_path / "bots.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
        ],
    )

    assert (
        live_restart_smoke_plan.main(
            [str(supervisor_config), "--summary-smoke-report", "--compact"]
        )
        == 0
    )
    summary_report = json.loads(capsys.readouterr().out)
    assert "--summary" in summary_report["smoke_report"]["command"]
    assert "--brief" not in summary_report["smoke_report"]["command"]

    assert (
        live_restart_smoke_plan.main(
            [str(supervisor_config), "--full-smoke-report", "--compact"]
        )
        == 0
    )
    full_report = json.loads(capsys.readouterr().out)
    assert "--summary" not in full_report["smoke_report"]["command"]
    assert "--brief" not in full_report["smoke_report"]["command"]


def test_live_restart_smoke_plan_cli_rejects_execute(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--execute"])

    assert exc_info.value.code == 2
    assert "--execute is not implemented" in capsys.readouterr().err


def test_live_restart_smoke_plan_cli_rejects_negative_event_scan_bounds(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--event-tail-lines", "-1"])

    assert exc_info.value.code == 2
    assert "smoke_event_tail_lines must be non-negative" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--max-event-files-per-bot", "-1"])

    assert exc_info.value.code == 2
    assert (
        "smoke_max_event_files_per_bot must be non-negative"
        in capsys.readouterr().err
    )


def test_live_restart_smoke_plan_cli_rejects_negative_log_scan_bounds(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--max-log-files", "-1"])

    assert exc_info.value.code == 2
    assert "smoke_max_log_files must be non-negative" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--log-tail-lines", "-1"])

    assert exc_info.value.code == 2
    assert "smoke_log_tail_lines must be non-negative" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        live_restart_smoke_plan.main(["bots.yaml", "--max-log-matches", "-1"])

    assert exc_info.value.code == 2
    assert "smoke_max_log_matches must be non-negative" in capsys.readouterr().err


def test_live_restart_smoke_plan_tool_dispatch_forwards_module_and_prog(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert (
        cli_main.main(["tool", "live-restart-smoke-plan", "bots.yaml", "--compact"])
        == 0
    )

    assert captured["module_name"] == "tools.live_restart_smoke_plan"
    assert captured["argv"] == [
        "passivbot tool live-restart-smoke-plan",
        "bots.yaml",
        "--compact",
    ]
    assert captured["prog_env"] == "passivbot tool live-restart-smoke-plan"
