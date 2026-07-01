from __future__ import annotations

import json
import os
import re
import sys

import pytest

from live.restart_smoke_plan import (
    _display_path,
    build_live_restart_smoke_plan,
    summarize_live_restart_smoke_plan,
)
from passivbot_cli import main as cli_main
from tools import live_restart_smoke_plan


INCIDENT_BUNDLE_OUTPUT_PATTERN = (
    r"/passivbot_incident_bundle_restart_smoke_\d{8}_\d{6}_\d{6}\.tar\.gz$"
)


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
    assert "--log-window-unparsed-policy keep" in report["smoke_report"]["command"]
    assert "--brief" in report["smoke_report"]["command"]
    assert "--summary" not in report["smoke_report"]["command"]
    assert report["config_preflight"] == {
        "command_count": 1,
        "commands": ["passivbot tool live-config-preflight configs/forager.json --compact"],
        "execute": False,
        "skipped_without_config_path_count": 0,
        "expected_fields": [
            "identity hints",
            "HSL signal mode and enabled sides",
            "approved and ignored universe counts",
            "forager slot and staleness settings",
            "cache-related live settings and readiness attention",
        ],
    }
    assert (
        report["phases"][0]["planned_commands"][-1]["command"]
        == "passivbot tool live-config-preflight configs/forager.json --compact"
    )
    assert report["phases"][0]["planned_commands"][-1]["execute"] is False
    assert report["incident_bundle"]["execute"] is False
    assert re.search(
        INCIDENT_BUNDLE_OUTPUT_PATTERN,
        report["incident_bundle"]["output_path"],
    )
    incident_command = report["incident_bundle"]["command"]
    assert "passivbot tool live-incident-bundle" in incident_command
    assert "--output /tmp/passivbot_incident_bundle_restart_smoke_" in incident_command
    assert ".tar.gz" in incident_command
    assert "--supervisor-config" in incident_command
    assert "--processes" in incident_command
    assert "--recent-minutes 15" in incident_command
    assert "--no-event-segments" in incident_command
    assert "--restart-smoke-plan" in incident_command
    assert "--restart-smoke-window-minutes 15" in incident_command
    assert "--event-tail-lines 2000" in incident_command
    assert "--max-event-files-per-bot 2" in incident_command
    assert "--max-log-files 8" in incident_command
    assert "--log-tail-lines 1200" in incident_command
    assert "--max-log-matches 20" in incident_command
    assert "--log-window-unparsed-policy keep" in incident_command
    assert "--compact" in incident_command
    assert report["phases"][-1]["name"] == "post_failure_incident_bundle"
    assert report["phases"][-1]["planned_commands"][0]["command"] == incident_command
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


def test_live_restart_smoke_plan_can_focus_smoke_sections(tmp_path):
    supervisor_config = tmp_path / "bots_vps5.yaml"
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

    report = build_live_restart_smoke_plan(
        supervisor_config,
        smoke_sections=["fill_refresh", "risk_events"],
    )

    assert report["inputs"]["smoke_sections"] == ["fill_refresh", "risk_events"]
    assert "--section fill_refresh" in report["smoke_report"]["command"]
    assert "--section risk_events" in report["smoke_report"]["command"]
    assert "--smoke-section fill_refresh" in report["incident_bundle"]["command"]
    assert "--smoke-section risk_events" in report["incident_bundle"]["command"]
    assert report["smoke_report"]["execute"] is False


def test_live_restart_smoke_plan_deduplicates_config_preflight_commands(tmp_path):
    supervisor_config = tmp_path / "bots_vps5.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u binance_01",
            "  - window_name: kucoin_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u kucoin_01",
            "  - window_name: gateio_01",
            "    panes:",
            "      - passivbot live configs/tradfi.json -u gateio_01",
        ],
    )

    report = build_live_restart_smoke_plan(supervisor_config)

    assert report["config_preflight"]["commands"] == [
        "passivbot tool live-config-preflight configs/forager.json --compact",
        "passivbot tool live-config-preflight configs/tradfi.json --compact",
    ]
    assert report["config_preflight"]["command_count"] == 2
    assert report["config_preflight"]["skipped_without_config_path_count"] == 0


def test_live_restart_smoke_plan_reports_config_preflight_skips(tmp_path):
    supervisor_config = tmp_path / "bots_vps5.yaml"
    _write_supervisor(
        supervisor_config,
        [
            "session_name: passivbot",
            "windows:",
            "  - window_name: binance_01",
            "    panes:",
            "      - passivbot live -u binance_01",
            "  - window_name: gateio_01",
            "    panes:",
            "      - passivbot live configs/forager.json -u gateio_01",
        ],
    )

    report = build_live_restart_smoke_plan(supervisor_config)
    summary = summarize_live_restart_smoke_plan(report)

    assert report["config_preflight"]["command_count"] == 1
    assert report["config_preflight"]["skipped_without_config_path_count"] == 1
    assert summary["config_preflight"]["skipped_without_config_path_count"] == 1


def test_live_restart_smoke_plan_can_set_log_window_unparsed_policy(tmp_path):
    supervisor_config = tmp_path / "bots_vps5.yaml"
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

    report = build_live_restart_smoke_plan(
        supervisor_config,
        log_window_unparsed_policy="drop",
    )

    assert report["inputs"]["log_window_unparsed_policy"] == "drop"
    assert "--log-window-unparsed-policy drop" in report["smoke_report"]["command"]
    assert (
        "--log-window-unparsed-policy drop"
        in report["incident_bundle"]["command"]
    )


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
    assert re.search(
        INCIDENT_BUNDLE_OUTPUT_PATTERN,
        report["inputs"]["incident_bundle_output"],
    )
    assert "--event-tail-lines 2000" in report["smoke_report"]["command"]
    assert "--max-event-files-per-bot 2" in report["smoke_report"]["command"]
    assert "--max-log-files 8" in report["smoke_report"]["command"]
    assert "--log-tail-lines 1200" in report["smoke_report"]["command"]
    assert "--max-log-matches 20" in report["smoke_report"]["command"]
    assert "--log-window-unparsed-policy keep" in report["smoke_report"]["command"]
    assert "--brief" in report["smoke_report"]["command"]
    assert "--no-event-segments" in report["incident_bundle"]["command"]
    assert (
        "--log-window-unparsed-policy keep"
        in report["incident_bundle"]["command"]
    )
    assert "--compact" in report["incident_bundle"]["command"]


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
    assert "--event-tail-lines" not in report["incident_bundle"]["command"]
    assert "--max-event-files-per-bot" not in report["incident_bundle"]["command"]


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
    assert "--max-log-files" not in report["incident_bundle"]["command"]
    assert "--log-tail-lines" not in report["incident_bundle"]["command"]
    assert "--max-log-matches" not in report["incident_bundle"]["command"]


def test_live_restart_smoke_plan_can_override_incident_bundle_output(tmp_path, capsys):
    supervisor_config = tmp_path / "bots.yaml"
    output_path = tmp_path / "incident.tar.gz"
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
                "--incident-bundle-output",
                str(output_path),
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["incident_bundle_output"] == str(output_path)
    assert report["incident_bundle"]["output_path"] == str(output_path)
    assert f"--output {output_path}" in report["incident_bundle"]["command"]


def test_live_restart_smoke_plan_summary_projects_concise_commands(tmp_path, capsys):
    supervisor_config = tmp_path / "bots.yaml"
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

    full_report = build_live_restart_smoke_plan(
        supervisor_config,
        repo_path="/srv/passivbot",
        smoke_window_minutes=7,
    )
    summary = summarize_live_restart_smoke_plan(full_report)

    assert summary["ok"] is True
    assert "bots" not in summary["bots"]
    assert summary["bots"] == {
        "count": 2,
        "names": ["binance_01", "gateio_01"],
        "truncated": 0,
    }
    assert summary["phases"]["names"] == [
        "pre_restart_readiness",
        "graceful_stop_all",
        "orphan_duplicate_check",
        "supervisor_reload_start",
        "post_start_smoke_report",
        "post_failure_incident_bundle",
    ]
    assert "passivbot tool live-smoke-report" in summary["smoke_report"]["command"]
    assert "--recent-minutes 7" in summary["smoke_report"]["command"]
    assert "passivbot tool live-incident-bundle" in summary["incident_bundle"][
        "command"
    ]
    assert summary["incident_bundle"]["execute"] is False
    assert summary["config_preflight"] == {
        "command_count": 1,
        "commands": ["passivbot tool live-config-preflight configs/forager.json --compact"],
        "execute": False,
        "skipped_without_config_path_count": 0,
    }
    assert summary["incident_bundle"]["event_segments"] == (
        "disabled_by_default_for_fast_restart_smoke_bundle"
    )
    assert summary["timeout_escalation_ladder"][1]["planned_command_count"] == 2
    assert summary["execution_policy"]["execute_flag"] == "not_implemented"
    assert summary["execution_policy"]["rejected_operation_count"] >= 1
    assert summary["warnings"]["count"] == len(full_report["warnings"])
    assert summary["issues"] == {"count": 0, "items": []}

    assert (
        live_restart_smoke_plan.main(
            [str(supervisor_config), "--summary", "--compact"]
        )
        == 0
    )
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["bots"]["count"] == 2
    assert "phases" in cli_summary
    assert "passivbot tool live-incident-bundle" in cli_summary["incident_bundle"][
        "command"
    ]
    assert cli_summary["config_preflight"]["command_count"] == 1
    assert "command_key" not in json.dumps(cli_summary["bots"])


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


def test_live_restart_smoke_plan_cli_can_plan_smoke_sections(tmp_path, capsys):
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
                "--smoke-section",
                "fill_refresh",
                "--smoke-section",
                "hsl_replay",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["smoke_sections"] == ["fill_refresh", "hsl_replay"]
    assert "--section fill_refresh" in report["smoke_report"]["command"]
    assert "--section hsl_replay" in report["smoke_report"]["command"]
    assert "--smoke-section fill_refresh" in report["incident_bundle"]["command"]
    assert "--smoke-section hsl_replay" in report["incident_bundle"]["command"]
    assert "--restart-smoke-plan" in report["incident_bundle"]["command"]
    assert "--brief" in report["smoke_report"]["command"]


def test_live_restart_smoke_plan_cli_can_plan_log_window_unparsed_policy(
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
                "--log-window-unparsed-policy",
                "drop",
                "--compact",
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["inputs"]["log_window_unparsed_policy"] == "drop"
    assert "--log-window-unparsed-policy drop" in report["smoke_report"]["command"]
    assert (
        "--log-window-unparsed-policy drop"
        in report["incident_bundle"]["command"]
    )


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
