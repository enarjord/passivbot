from __future__ import annotations

from pathlib import Path
from typing import Any

from live.smoke_report import parse_tmuxp_live_commands, _user_safe_display_path


DEFAULT_SHUTDOWN_TIMEOUT_S = 90
DEFAULT_STARTUP_WAIT_S = 180
DEFAULT_SMOKE_WINDOW_MINUTES = 30
DEFAULT_SMOKE_EVENT_TAIL_LINES = 2000
DEFAULT_SMOKE_MAX_EVENT_FILES_PER_BOT = 2
DEFAULT_SMOKE_MAX_LOG_FILES = 8
DEFAULT_SMOKE_LOG_TAIL_LINES = 1200
DEFAULT_SMOKE_MAX_LOG_MATCHES = 20
DEFAULT_INCIDENT_BUNDLE_OUTPUT = "/tmp/passivbot_incident_bundle_restart_smoke.tar.gz"
DEFAULT_MONITOR_ROOT = "monitor"
DEFAULT_LOGS_ROOT = "logs"
UNSAFE_PROCESS_SIGNAL_PATTERNS = (
    "pkill -f 'passivbot live'",
    "pgrep -f 'passivbot live' | xargs kill",
    "kill $(pgrep -f 'passivbot live')",
)


def _positive_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _non_negative_int(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _display_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return _user_safe_display_path(value)


def _shell_join(args: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(arg)) for arg in args if str(arg) != "")


def _smoke_report_command(
    *,
    monitor_root: str | Path,
    logs_root: str | Path | None,
    supervisor_config: str | Path,
    smoke_window_minutes: int,
    event_tail_lines: int,
    max_event_files_per_bot: int,
    max_log_files: int,
    log_tail_lines: int,
    max_log_matches: int,
    compact: bool,
    brief: bool,
    summary: bool,
) -> str:
    args = [
        "passivbot",
        "tool",
        "live-smoke-report",
        str(monitor_root),
        "--supervisor-config",
        str(supervisor_config),
        "--processes",
        "--recent-minutes",
        str(smoke_window_minutes),
    ]
    if logs_root is not None:
        args.extend(["--logs-root", str(logs_root)])
    if int(event_tail_lines) > 0:
        args.extend(["--event-tail-lines", str(event_tail_lines)])
    if int(max_event_files_per_bot) > 0:
        args.extend(["--max-event-files-per-bot", str(max_event_files_per_bot)])
    if int(max_log_files) > 0:
        args.extend(["--max-log-files", str(max_log_files)])
    if int(log_tail_lines) > 0:
        args.extend(["--log-tail-lines", str(log_tail_lines)])
    if int(max_log_matches) > 0:
        args.extend(["--max-log-matches", str(max_log_matches)])
    if summary:
        args.append("--summary")
    elif brief:
        args.append("--brief")
    if compact:
        args.append("--compact")
    return _shell_join(args)


def _incident_bundle_command(
    *,
    monitor_root: str | Path,
    logs_root: str | Path | None,
    supervisor_config: str | Path,
    output_path: str | Path,
    smoke_window_minutes: int,
    event_tail_lines: int,
    max_event_files_per_bot: int,
    max_log_files: int,
    log_tail_lines: int,
    max_log_matches: int,
) -> str:
    args = [
        "passivbot",
        "tool",
        "live-incident-bundle",
        str(monitor_root),
        "--output",
        str(output_path),
        "--supervisor-config",
        str(supervisor_config),
        "--processes",
        "--recent-minutes",
        str(smoke_window_minutes),
        "--no-event-segments",
    ]
    if logs_root is not None:
        args.extend(["--logs-root", str(logs_root)])
    if int(event_tail_lines) > 0:
        args.extend(["--event-tail-lines", str(event_tail_lines)])
    if int(max_event_files_per_bot) > 0:
        args.extend(["--max-event-files-per-bot", str(max_event_files_per_bot)])
    if int(max_log_files) > 0:
        args.extend(["--max-log-files", str(max_log_files)])
    if int(log_tail_lines) > 0:
        args.extend(["--log-tail-lines", str(log_tail_lines)])
    if int(max_log_matches) > 0:
        args.extend(["--max-log-matches", str(max_log_matches)])
    args.append("--compact")
    return _shell_join(args)


def _repo_check_commands(repo_path: str | Path | None) -> list[str]:
    if repo_path is None:
        return [
            _shell_join(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            _shell_join(["git", "rev-parse", "--short", "HEAD"]),
            _shell_join(["git", "status", "--porcelain", "--untracked-files=no"]),
        ]
    repo = str(repo_path)
    return [
        _shell_join(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"]),
        _shell_join(["git", "-C", repo, "rev-parse", "--short", "HEAD"]),
        _shell_join(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"]
        ),
    ]


def _process_signal_safety_contract() -> dict[str, Any]:
    return {
        "strategy": "exact_tmux_pane_or_exact_pid_only",
        "forbid_broad_process_pattern_signals": True,
        "unsafe_patterns": list(UNSAFE_PROCESS_SIGNAL_PATTERNS),
        "why": (
            "Broad process-pattern signals can match the controller shell or "
            "smoke command when the pattern appears in its arguments."
        ),
        "required_guards": [
            "derive candidate processes from canonical live-smoke-report process rows",
            "match configured command keys exactly before signaling",
            "exclude the controller process and its ancestors",
            "signal one matched bot at a time and re-scan after each signal",
            "halt reload when duplicate, extra, or unowned matching live processes remain",
        ],
    }


def _bot_phase_plan(
    bot: dict[str, Any],
    *,
    shutdown_timeout_s: int,
    startup_wait_s: int,
    smoke_command: str,
) -> list[dict[str, Any]]:
    command = str(bot.get("command") or bot.get("command_key") or "")
    return [
        {
            "name": "stop_requested",
            "goal": "operator requests graceful stop for the configured pane/process",
            "planned_actions": [
                {
                    "description": (
                        "identify the tmux pane or process matching the configured "
                        "command"
                    ),
                    "command": command,
                    "execute": False,
                },
                {
                    "description": "send graceful interrupt to the matched live bot",
                    "command": "operator_action: graceful interrupt only",
                    "execute": False,
                },
            ],
            "timeout_s": shutdown_timeout_s,
            "measure": "elapsed time until process exits and shutdown events settle",
        },
        {
            "name": "stop_verified",
            "goal": "confirm no duplicate, stale, or orphaned matching live processes remain",
            "planned_actions": [
                {
                    "description": "compare local process table against supervisor config",
                    "command": smoke_command,
                    "execute": False,
                }
            ],
            "timeout_s": shutdown_timeout_s,
        },
        {
            "name": "start_requested",
            "goal": "operator reloads the configured tmuxp supervisor entry",
            "planned_actions": [
                {
                    "description": (
                        "start the configured command exactly as listed by the "
                        "supervisor config"
                    ),
                    "command": command,
                    "execute": False,
                }
            ],
            "timeout_s": startup_wait_s,
        },
        {
            "name": "startup_smoke",
            "goal": "wait for startup, then collect read-only smoke evidence",
            "planned_actions": [
                {
                    "description": (
                        "collect process, monitor, log, shutdown, risk, remote-call, "
                        "and repository summaries"
                    ),
                    "command": smoke_command,
                    "execute": False,
                }
            ],
            "wait_s": startup_wait_s,
        },
    ]


def build_live_restart_smoke_plan(
    supervisor_config: str | Path,
    *,
    repo_path: str | Path | None = None,
    monitor_root: str | Path = DEFAULT_MONITOR_ROOT,
    logs_root: str | Path | None = DEFAULT_LOGS_ROOT,
    shutdown_timeout_s: int = DEFAULT_SHUTDOWN_TIMEOUT_S,
    startup_wait_s: int = DEFAULT_STARTUP_WAIT_S,
    smoke_window_minutes: int = DEFAULT_SMOKE_WINDOW_MINUTES,
    smoke_event_tail_lines: int = DEFAULT_SMOKE_EVENT_TAIL_LINES,
    smoke_max_event_files_per_bot: int = DEFAULT_SMOKE_MAX_EVENT_FILES_PER_BOT,
    smoke_max_log_files: int = DEFAULT_SMOKE_MAX_LOG_FILES,
    smoke_log_tail_lines: int = DEFAULT_SMOKE_LOG_TAIL_LINES,
    smoke_max_log_matches: int = DEFAULT_SMOKE_MAX_LOG_MATCHES,
    incident_bundle_output: str | Path = DEFAULT_INCIDENT_BUNDLE_OUTPUT,
    compact_smoke_report: bool = True,
    brief_smoke_report: bool = True,
    summary_smoke_report: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    if execute:
        raise NotImplementedError("execution is intentionally unavailable for this tool")
    shutdown_timeout_s = _positive_int(shutdown_timeout_s, field="shutdown_timeout_s")
    startup_wait_s = _positive_int(startup_wait_s, field="startup_wait_s")
    smoke_window_minutes = _positive_int(smoke_window_minutes, field="smoke_window_minutes")
    smoke_event_tail_lines = _non_negative_int(
        smoke_event_tail_lines, field="smoke_event_tail_lines"
    )
    smoke_max_event_files_per_bot = _non_negative_int(
        smoke_max_event_files_per_bot, field="smoke_max_event_files_per_bot"
    )
    smoke_max_log_files = _non_negative_int(
        smoke_max_log_files, field="smoke_max_log_files"
    )
    smoke_log_tail_lines = _non_negative_int(
        smoke_log_tail_lines, field="smoke_log_tail_lines"
    )
    smoke_max_log_matches = _non_negative_int(
        smoke_max_log_matches, field="smoke_max_log_matches"
    )

    supervisor = parse_tmuxp_live_commands(supervisor_config)
    bots = list(supervisor.get("expected") or [])
    warnings = [
        "plan_only_no_execution",
        "does_not_signal_processes",
        "does_not_invoke_tmux",
        "does_not_run_ssh",
        "does_not_run_git_pull",
        "does_not_start_passivbot_live",
        "does_not_contact_exchanges",
    ]
    issues: list[dict[str, str]] = []
    if supervisor.get("error"):
        issues.append(
            {
                "severity": "error",
                "code": str(supervisor["error"]),
                "message": "Supervisor config could not be parsed into expected live commands.",
            }
        )
    if supervisor_config and supervisor.get("exists") and not bots:
        issues.append(
            {
                "severity": "error",
                "code": "no_expected_live_commands",
                "message": "Supervisor config contains no parseable passivbot live commands.",
            }
        )
    if not supervisor.get("exists") and not any(
        issue.get("code") == "config_not_found" for issue in issues
    ):
        issues.append(
            {
                "severity": "error",
                "code": "config_not_found",
                "message": "Supervisor config is required for a restart smoke plan.",
            }
        )

    smoke_command = _smoke_report_command(
        monitor_root=monitor_root,
        logs_root=logs_root,
        supervisor_config=supervisor_config,
        smoke_window_minutes=smoke_window_minutes,
        event_tail_lines=smoke_event_tail_lines,
        max_event_files_per_bot=smoke_max_event_files_per_bot,
        max_log_files=smoke_max_log_files,
        log_tail_lines=smoke_log_tail_lines,
        max_log_matches=smoke_max_log_matches,
        compact=compact_smoke_report,
        brief=brief_smoke_report,
        summary=summary_smoke_report,
    )
    incident_bundle_command = _incident_bundle_command(
        monitor_root=monitor_root,
        logs_root=logs_root,
        supervisor_config=supervisor_config,
        output_path=incident_bundle_output,
        smoke_window_minutes=smoke_window_minutes,
        event_tail_lines=smoke_event_tail_lines,
        max_event_files_per_bot=smoke_max_event_files_per_bot,
        max_log_files=smoke_max_log_files,
        log_tail_lines=smoke_log_tail_lines,
        max_log_matches=smoke_max_log_matches,
    )
    planned_bots = []
    for index, bot in enumerate(bots, start=1):
        planned_bots.append(
            {
                "index": index,
                **bot,
                "phases": _bot_phase_plan(
                    bot,
                    shutdown_timeout_s=shutdown_timeout_s,
                    startup_wait_s=startup_wait_s,
                    smoke_command=smoke_command,
                ),
            }
        )

    return {
        "tool": "live-restart-smoke-plan",
        "schema_version": 1,
        "ok": not issues,
        "metadata": {
            "dry_run": True,
            "execute": False,
            "execution_available": False,
            "plan_only": True,
        },
        "inputs": {
            "supervisor_config": _display_path(supervisor_config),
            "repo_path": _display_path(repo_path),
            "monitor_root": _display_path(monitor_root),
            "logs_root": _display_path(logs_root),
            "shutdown_timeout_s": shutdown_timeout_s,
            "startup_wait_s": startup_wait_s,
            "smoke_window_minutes": smoke_window_minutes,
            "smoke_event_tail_lines": smoke_event_tail_lines,
            "smoke_max_event_files_per_bot": smoke_max_event_files_per_bot,
            "smoke_max_log_files": smoke_max_log_files,
            "smoke_log_tail_lines": smoke_log_tail_lines,
            "smoke_max_log_matches": smoke_max_log_matches,
            "incident_bundle_output": _display_path(incident_bundle_output),
        },
        "supervisor_config": {
            "path": _display_path(supervisor.get("path")),
            "exists": supervisor.get("exists"),
            "error": supervisor.get("error"),
            "expected_live_commands": len(bots),
        },
        "bots": planned_bots,
        "phases": [
            {
                "name": "pre_restart_readiness",
                "description": (
                    "Confirm repo state and configured bot set before any operator "
                    "action."
                ),
                "planned_commands": [
                    {"command": command, "execute": False}
                    for command in _repo_check_commands(repo_path)
                ],
            },
            {
                "name": "graceful_stop_all",
                "description": (
                    "Stop configured bots one by one and record per-bot shutdown "
                    "elapsed time."
                ),
                "bot_count": len(bots),
                "timeout_s_per_bot": shutdown_timeout_s,
            },
            {
                "name": "orphan_duplicate_check",
                "description": (
                    "Use the smoke report process comparison to find missing, "
                    "duplicate, or extra live processes."
                ),
                "planned_commands": [{"command": smoke_command, "execute": False}],
            },
            {
                "name": "supervisor_reload_start",
                "description": "Reload/start from the supervisor config exactly as configured.",
                "bot_count": len(bots),
                "wait_s": startup_wait_s,
            },
            {
                "name": "post_start_smoke_report",
                "description": "Collect bounded read-only startup smoke evidence.",
                "planned_commands": [{"command": smoke_command, "execute": False}],
            },
            {
                "name": "post_failure_incident_bundle",
                "description": (
                    "If smoke or shutdown evidence is not clean, collect a bounded "
                    "local incident bundle for review."
                ),
                "planned_commands": [
                    {"command": incident_bundle_command, "execute": False}
                ],
            },
        ],
        "smoke_report": {
            "command": smoke_command,
            "execute": False,
            "expected_fields": [
                "process liveness",
                "duplicate configured-command matches",
                "extra passivbot live processes",
                "repository branch/head/dirty tracked state",
                "recent hard log matches",
                "bounded text log scan metadata",
                "monitor problem-event counts",
                "bounded monitor event scan metadata",
                "startup timings",
                "shutdown lifecycle events",
                "remote-call health",
                "risk/HSL events",
            ],
        },
        "incident_bundle": {
            "command": incident_bundle_command,
            "execute": False,
            "output_path": _display_path(incident_bundle_output),
            "event_segments": "disabled_by_default_for_fast_restart_smoke_bundle",
            "expected_fields": [
                "manifest filters and runtime metadata",
                "bounded smoke report with process and log scan metadata",
                "problem-event report",
                "trace summary/order trace reports",
                "time-window event report",
                "monitor snapshots",
                "config hashes when configured explicitly",
            ],
        },
        "process_signal_safety": _process_signal_safety_contract(),
        "timeout_escalation_ladder": [
            {
                "level": 0,
                "condition": "bot exits before graceful shutdown timeout",
                "operator_action": "record elapsed shutdown time and continue",
                "execute": False,
            },
            {
                "level": 1,
                "condition": "bot still running after graceful shutdown timeout",
                "operator_action": (
                    "collect smoke report and inspect shutdown events/logs before "
                    "escalation"
                ),
                "planned_commands": [
                    {"command": smoke_command, "execute": False},
                    {"command": incident_bundle_command, "execute": False},
                ],
                "execute": False,
            },
            {
                "level": 2,
                "condition": "duplicate or extra passivbot live process remains",
                "operator_action": (
                    "halt reload, identify owner/pane/process, and require explicit "
                    "human approval"
                ),
                "execute": False,
            },
            {
                "level": 3,
                "condition": (
                    "post-start smoke reports hard errors, missing expected bot, or "
                    "stale startup"
                ),
                "operator_action": (
                    "stop the routine and collect an incident bundle/smoke evidence"
                ),
                "execute": False,
            },
        ],
        "execution_policy": {
            "execute_flag": "not_implemented",
            "future_execution_requires_review": True,
            "rejected_operations": [
                "ssh",
                "tmux signal/send-keys",
                "process kill/signal",
                "broad process-pattern kill/signal",
                "git pull/fetch/checkout",
                "passivbot live",
                "exchange/API calls",
                "credential loading",
            ],
        },
        "warnings": warnings,
        "issues": issues,
    }


def summarize_live_restart_smoke_plan(report: dict[str, Any]) -> dict[str, Any]:
    bots = report.get("bots")
    phases = report.get("phases")
    timeout_ladder = report.get("timeout_escalation_ladder")
    warnings = report.get("warnings")
    issues = report.get("issues")
    bot_rows = bots if isinstance(bots, list) else []
    phase_rows = phases if isinstance(phases, list) else []
    timeout_rows = timeout_ladder if isinstance(timeout_ladder, list) else []
    warning_rows = warnings if isinstance(warnings, list) else []
    issue_rows = issues if isinstance(issues, list) else []
    smoke_report = report.get("smoke_report")
    incident_bundle = report.get("incident_bundle")
    execution_policy = report.get("execution_policy")
    signal_safety = report.get("process_signal_safety")
    return {
        "tool": report.get("tool"),
        "schema_version": report.get("schema_version"),
        "ok": report.get("ok"),
        "metadata": report.get("metadata"),
        "inputs": report.get("inputs"),
        "supervisor_config": report.get("supervisor_config"),
        "bots": {
            "count": len(bot_rows),
            "names": [str(bot.get("name") or "") for bot in bot_rows[:20]],
            "truncated": max(0, len(bot_rows) - 20),
        },
        "phases": {
            "count": len(phase_rows),
            "names": [str(phase.get("name") or "") for phase in phase_rows],
        },
        "smoke_report": {
            "command": (
                smoke_report.get("command") if isinstance(smoke_report, dict) else None
            ),
            "execute": (
                smoke_report.get("execute") if isinstance(smoke_report, dict) else None
            ),
        },
        "incident_bundle": {
            "command": (
                incident_bundle.get("command")
                if isinstance(incident_bundle, dict)
                else None
            ),
            "execute": (
                incident_bundle.get("execute")
                if isinstance(incident_bundle, dict)
                else None
            ),
            "output_path": (
                incident_bundle.get("output_path")
                if isinstance(incident_bundle, dict)
                else None
            ),
            "event_segments": (
                incident_bundle.get("event_segments")
                if isinstance(incident_bundle, dict)
                else None
            ),
        },
        "process_signal_safety": {
            "strategy": (
                signal_safety.get("strategy") if isinstance(signal_safety, dict) else None
            ),
            "forbid_broad_process_pattern_signals": (
                signal_safety.get("forbid_broad_process_pattern_signals")
                if isinstance(signal_safety, dict)
                else None
            ),
        },
        "timeout_escalation_ladder": [
            {
                "level": row.get("level"),
                "condition": row.get("condition"),
                "execute": row.get("execute"),
                "planned_command_count": len(
                    row.get("planned_commands")
                    if isinstance(row.get("planned_commands"), list)
                    else []
                ),
            }
            for row in timeout_rows
            if isinstance(row, dict)
        ],
        "execution_policy": {
            "execute_flag": (
                execution_policy.get("execute_flag")
                if isinstance(execution_policy, dict)
                else None
            ),
            "future_execution_requires_review": (
                execution_policy.get("future_execution_requires_review")
                if isinstance(execution_policy, dict)
                else None
            ),
            "rejected_operation_count": len(
                execution_policy.get("rejected_operations")
                if isinstance(execution_policy, dict)
                and isinstance(execution_policy.get("rejected_operations"), list)
                else []
            ),
        },
        "warnings": {"count": len(warning_rows), "items": warning_rows[:20]},
        "issues": {"count": len(issue_rows), "items": issue_rows[:20]},
    }
