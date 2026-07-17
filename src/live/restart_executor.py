from __future__ import annotations

import importlib
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from live.restart_smoke_targets import (
    MAX_RESTART_TARGETS,
    MAX_TARGET_SAMPLE_INTERVAL_S,
    MAX_TARGET_SAMPLES,
    _tmux_pane_inventory,
    build_live_restart_target_report,
)
from live.smoke_report import (
    _parse_tmuxp_live_commands,
    _running_live_processes,
    _supervisor_command_contract,
)
from rust_utils import source_fingerprint, verify_loaded_runtime_extension


DEFAULT_PREFLIGHT_SAMPLES = 3
DEFAULT_PREFLIGHT_INTERVAL_S = 5.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 90.0
DEFAULT_STARTUP_TIMEOUT_S = 180.0
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_VERIFICATION_SAMPLES = 3
DEFAULT_VERIFICATION_INTERVAL_S = 5.0
MAX_PHASE_TIMEOUT_S = 600.0
MAX_LAUNCH_COMMAND_CHARS = 8192
POST_STOP_PANE_SETTLE_TIMEOUT_S = 5.0
ALLOWED_PANE_SHELL_COMMANDS = {"bash", "dash", "fish", "ksh", "sh", "zsh"}

SAFETY_CONTRACT = {
    "local_only": True,
    "direct_network_access": False,
    "direct_exchange_access": False,
    "configured_live_processes_may_contact_exchanges": True,
    "reads": [
        "tmux_pane_inventory",
        "process_table",
        "supervisor_config",
        "referenced_bot_configs",
    ],
    "direct_file_writes": False,
    "signals_processes": True,
    "starts_processes": True,
    "configured_live_processes_may_write_files": True,
    "signal_method": "tmux_send_keys_exact_pane_ctrl_c",
    "start_method": "tmux_send_keys_exact_pane_literal_command",
    "automatic_force_signal": False,
    "broad_process_pattern_signals": False,
    "requires_execute_confirmation": True,
    "requires_expected_supervisor_fingerprint": True,
    "requires_expected_repository_head": True,
    "requires_tracked_clean_repository": True,
    "requires_source_matched_rust_extension": True,
}


def _bounded_phase_value(value: float, *, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > MAX_PHASE_TIMEOUT_S:
        raise ValueError(
            f"{field} must be greater than 0 and at most {MAX_PHASE_TIMEOUT_S:g}"
        )
    return parsed


def _bounded_samples(value: int, *, field: str) -> int:
    parsed = int(value)
    if parsed < 2 or parsed > MAX_TARGET_SAMPLES:
        raise ValueError(f"{field} must be between 2 and {MAX_TARGET_SAMPLES}")
    return parsed


def _bounded_sample_interval(value: float, *, field: str) -> float:
    parsed = float(value)
    if (
        not math.isfinite(parsed)
        or parsed <= 0.0
        or parsed > MAX_TARGET_SAMPLE_INTERVAL_S
    ):
        raise ValueError(
            f"{field} must be greater than 0 and at most "
            f"{MAX_TARGET_SAMPLE_INTERVAL_S:g}"
        )
    return parsed


def _valid_fingerprint(value: str) -> str:
    fingerprint = str(value or "").strip()
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(
            "expected_supervisor_fingerprint must be 64 lowercase hex characters"
        )
    return fingerprint


def _valid_repository_head(value: str) -> str:
    head = str(value or "").strip()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError(
            "expected_repository_head must be 40 lowercase hex characters"
        )
    return head


def _issue(code: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "severity": "error", **fields}


def _git_contract_output(arguments: list[str]) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git_failed:{exc.__class__.__name__}"
    if result.returncode != 0:
        return None, "git_failed"
    return result.stdout.strip(), None


def _build_runtime_contract(expected_repository_head: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "expected_repository_head": expected_repository_head,
        "observed_repository_head": None,
        "tracked_clean": None,
        "tracked_changes": None,
        "repository_snapshot_stable": None,
        "rust_extension": {
            "loaded": False,
            "source_fingerprint": None,
            "compiled_source_stamp": None,
            "compiled_sha256": None,
            "source_matched": False,
        },
        "issues": [],
    }
    issues: list[str] = report["issues"]

    root_output, error = _git_contract_output(["rev-parse", "--show-toplevel"])
    if error is not None or not root_output:
        issues.append("repository_unavailable")
        return report
    root = Path(root_output).resolve()

    head, error = _git_contract_output(["rev-parse", "HEAD"])
    if error is not None or head is None:
        issues.append("repository_head_unavailable")
        return report
    report["observed_repository_head"] = head
    if head != expected_repository_head:
        issues.append("repository_head_mismatch")

    status, error = _git_contract_output(
        ["status", "--porcelain", "--untracked-files=no"]
    )
    if error is not None or status is None:
        issues.append("repository_status_unavailable")
        return report
    changes = [line for line in status.splitlines() if line.strip()]
    report["tracked_clean"] = not changes
    report["tracked_changes"] = len(changes)
    if changes:
        issues.append("repository_tracked_dirty")

    try:
        fingerprint = source_fingerprint(root / "passivbot-rust")
    except OSError as exc:
        report["rust_extension"]["error_class"] = exc.__class__.__name__
        issues.append("rust_source_fingerprint_unavailable")
        return report
    rust_report = report["rust_extension"]
    rust_report["source_fingerprint"] = fingerprint
    if fingerprint is None:
        issues.append("rust_source_fingerprint_unavailable")
        return report

    try:
        importlib.import_module("passivbot_rust")
        runtime = verify_loaded_runtime_extension(fingerprint=fingerprint)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        rust_report["error_class"] = exc.__class__.__name__
        issues.append("rust_extension_verification_failed")
        return report

    compiled_stamp = runtime.get("runtime_compiled_source_stamp")
    rust_report.update(
        {
            "loaded": True,
            "compiled_source_stamp": compiled_stamp,
            "compiled_sha256": runtime.get("runtime_compiled_sha256"),
            "source_matched": compiled_stamp == fingerprint,
        }
    )
    if compiled_stamp != fingerprint:
        issues.append("rust_extension_source_mismatch")

    final_head, head_error = _git_contract_output(["rev-parse", "HEAD"])
    final_status, status_error = _git_contract_output(
        ["status", "--porcelain", "--untracked-files=no"]
    )
    if head_error is not None or status_error is not None:
        issues.append("repository_recheck_unavailable")
    else:
        final_changes = [
            line for line in (final_status or "").splitlines() if line.strip()
        ]
        snapshot_stable = final_head == head and final_changes == changes
        report["repository_snapshot_stable"] = snapshot_stable
        if not snapshot_stable:
            issues.append("repository_changed_during_runtime_check")

    report["ok"] = not issues
    return report


def _preflight_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "hard_failures": int(report.get("hard_failures") or 0),
        "session_name": report.get("session_name"),
        "expected_targets": int(report.get("expected_targets") or 0),
        "resolved_targets": int(report.get("resolved_targets") or 0),
        "relaunch_ready_targets": int(report.get("relaunch_ready_targets") or 0),
        "supervisor_contract": report.get("supervisor_contract"),
        "sampling": report.get("sampling"),
    }


def _load_launch_snapshot(
    supervisor_config: str | Path,
) -> tuple[dict[str, dict[str, str]], str | None, str | None]:
    config = _parse_tmuxp_live_commands(supervisor_config)
    if config.get("error"):
        return {}, None, str(config["error"])
    rows = config.get("expected")
    expected = rows if isinstance(rows, list) else []
    if not expected:
        return {}, None, "no_expected_live_commands"
    if len(expected) > MAX_RESTART_TARGETS:
        return {}, None, "expected_target_limit_exceeded"

    commands: dict[str, dict[str, str]] = {}
    for row in expected:
        name = str(row.get("name") or "").strip()
        command = str(row.get("_launch_command") or "").strip()
        match_key = str(row.get("_match_key") or "").strip()
        if not name or name in commands:
            return {}, None, "invalid_or_duplicate_window_name"
        if (
            not command
            or not match_key
            or len(command) > MAX_LAUNCH_COMMAND_CHARS
            or any(character in command for character in ("\x00", "\r", "\n"))
        ):
            return {}, None, "invalid_launch_command"
        commands[name] = {"command": command, "match_key": match_key}

    contract = _supervisor_command_contract(expected)
    fingerprint = str(contract.get("fingerprint") or "").strip()
    if len(fingerprint) != 64:
        return {}, None, "invalid_supervisor_contract"
    return commands, fingerprint, None


def _run_tmux(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["tmux", *arguments],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"tmux_failed:{exc.__class__.__name__}"
    if result.returncode != 0:
        return "tmux_failed"
    return None


def _send_graceful_interrupt(pane_id: str) -> str | None:
    return _run_tmux(["send-keys", "-t", pane_id, "C-c"])


def _send_launch_command(pane_id: str, command: str) -> str | None:
    error = _run_tmux(["send-keys", "-t", pane_id, "-l", "--", command])
    if error is not None:
        return error
    return _run_tmux(["send-keys", "-t", pane_id, "Enter"])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_exits(
    targets: list[dict[str, Any]],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> tuple[dict[int, float], list[int]]:
    started = time.monotonic()
    pending = {
        int(target["process_pid"])
        for target in targets
        if int(target.get("process_pid") or 0) > 0
    }
    elapsed_by_pid: dict[int, float] = {}
    deadline = started + timeout_s
    while pending:
        now = time.monotonic()
        for pid in list(pending):
            if not _pid_alive(pid):
                elapsed_by_pid[pid] = max(0.0, now - started)
                pending.remove(pid)
        if not pending or now >= deadline:
            break
        time.sleep(min(poll_interval_s, max(0.0, deadline - now)))
    return elapsed_by_pid, sorted(pending)


def _target_identities(report: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(target.get("window_name") or ""),
            str(target.get("pane_id") or ""),
            int(target.get("pane_pid") or 0),
            int(target.get("process_pid") or 0),
            str(target.get("ownership_proof") or ""),
        )
        for target in report.get("targets") or []
    )


def _post_stop_process_recheck(
    launch_snapshot: dict[str, dict[str, str]],
    *,
    relaunch_window_names: set[str],
) -> tuple[bool, str | None]:
    scan = _running_live_processes(command_match="")
    if scan.get("scan_error") is not None:
        return False, "process_scan_failed"
    running = scan.get("running") if isinstance(scan.get("running"), list) else []
    expected_match_keys = {
        row["match_key"] for row in launch_snapshot.values()
    }
    if any(
        str(process.get("_match_key") or "") not in expected_match_keys
        for process in running
    ):
        return False, "unexpected_live_process"
    for window_name, row in launch_snapshot.items():
        match_count = sum(
            1
            for process in running
            if str(process.get("_match_key") or "") == row["match_key"]
        )
        expected_count = 0 if window_name in relaunch_window_names else 1
        if match_count != expected_count:
            return False, "configured_process_count_changed"
    return True, None


def _post_stop_pane_recheck(
    targets: list[dict[str, Any]],
    *,
    session_name: str,
    relaunch_pane_ids: set[str],
) -> tuple[bool, str | None]:
    panes, error = _tmux_pane_inventory()
    if error is not None:
        return False, error
    session_panes = [
        pane for pane in panes if pane.get("session_name") == session_name
    ]
    expected_ids = {str(target.get("pane_id") or "") for target in targets}
    if {str(pane.get("pane_id") or "") for pane in session_panes} != expected_ids:
        return False, "session_pane_set_changed"
    by_id = {str(pane["pane_id"]): pane for pane in session_panes}
    for target in targets:
        pane = by_id.get(str(target.get("pane_id") or ""))
        if pane is None:
            return False, "pane_missing"
        if (
            str(pane.get("window_name") or "")
            != str(target.get("window_name") or "")
            or int(pane.get("pane_index") or 0)
            != int(target.get("pane_index") or 0)
            or int(pane.get("pane_pid") or 0) != int(target.get("pane_pid") or 0)
        ):
            return False, "pane_identity_changed"
        if str(pane["pane_id"]) in relaunch_pane_ids:
            current_command = Path(str(pane.get("current_command") or "")).name
            if current_command not in ALLOWED_PANE_SHELL_COMMANDS:
                return False, "pane_not_at_shell_prompt"
    return True, None


def _wait_for_post_stop_pane_recheck(
    targets: list[dict[str, Any]],
    *,
    session_name: str,
    relaunch_pane_ids: set[str],
    poll_interval_s: float,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + POST_STOP_PANE_SETTLE_TIMEOUT_S
    while True:
        ok, error = _post_stop_pane_recheck(
            targets,
            session_name=session_name,
            relaunch_pane_ids=relaunch_pane_ids,
        )
        if ok:
            return True, None
        now = time.monotonic()
        if now >= deadline:
            return False, error
        time.sleep(min(poll_interval_s, max(0.0, deadline - now)))


def _wait_for_startup_report(
    supervisor_config: str | Path,
    *,
    session_name: str,
    config_base_dir: Path | None,
    timeout_s: float,
    poll_interval_s: float,
    expected_supervisor_fingerprint: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    latest: dict[str, Any] = {}
    while True:
        latest = build_live_restart_target_report(
            supervisor_config,
            session_name=session_name,
            config_base_dir=config_base_dir,
        )
        observed_fingerprint = str(
            (latest.get("supervisor_contract") or {}).get("fingerprint") or ""
        )
        if latest.get("ok") and observed_fingerprint == expected_supervisor_fingerprint:
            return latest
        now = time.monotonic()
        if now >= deadline:
            return latest
        time.sleep(min(poll_interval_s, max(0.0, deadline - now)))


def execute_live_restart(
    supervisor_config: str | Path,
    *,
    session_name: str,
    expected_supervisor_fingerprint: str,
    expected_repository_head: str,
    config_base_dir: Path | None = None,
    preflight_samples: int = DEFAULT_PREFLIGHT_SAMPLES,
    preflight_interval_s: float = DEFAULT_PREFLIGHT_INTERVAL_S,
    shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    verification_samples: int = DEFAULT_VERIFICATION_SAMPLES,
    verification_interval_s: float = DEFAULT_VERIFICATION_INTERVAL_S,
    execute: bool = False,
) -> dict[str, Any]:
    if not execute:
        raise ValueError("execute=true confirmation is required")
    confirmed_session = str(session_name or "").strip()
    if not confirmed_session:
        raise ValueError("session_name must be non-empty")
    expected_fingerprint = _valid_fingerprint(expected_supervisor_fingerprint)
    expected_head = _valid_repository_head(expected_repository_head)
    preflight_samples = _bounded_samples(
        preflight_samples, field="preflight_samples"
    )
    verification_samples = _bounded_samples(
        verification_samples, field="verification_samples"
    )
    preflight_interval_s = _bounded_sample_interval(
        preflight_interval_s, field="preflight_interval_s"
    )
    verification_interval_s = _bounded_sample_interval(
        verification_interval_s, field="verification_interval_s"
    )
    shutdown_timeout_s = _bounded_phase_value(
        shutdown_timeout_s, field="shutdown_timeout_s"
    )
    startup_timeout_s = _bounded_phase_value(
        startup_timeout_s, field="startup_timeout_s"
    )
    poll_interval_s = _bounded_phase_value(
        poll_interval_s, field="poll_interval_s"
    )

    runtime_contract = _build_runtime_contract(expected_head)
    preflight: dict[str, Any] = {}
    if runtime_contract["ok"]:
        preflight = build_live_restart_target_report(
            supervisor_config,
            session_name=confirmed_session,
            config_base_dir=config_base_dir,
            samples=preflight_samples,
            sample_interval_s=preflight_interval_s,
        )
    report: dict[str, Any] = {
        "tool": "live-restart-executor",
        "schema_version": 2,
        "ok": False,
        "outcome": "preflight_failed",
        "action_started": False,
        "session_name": confirmed_session,
        "safety": dict(SAFETY_CONTRACT),
        "runtime_contract": runtime_contract,
        "preflight": _preflight_summary(preflight),
        "issues": [],
        "targets": [],
    }
    issues: list[dict[str, Any]] = report["issues"]
    if not runtime_contract["ok"]:
        issues.extend(_issue(code) for code in runtime_contract["issues"])
        report["hard_failures"] = len(issues)
        return report

    sampling = preflight.get("sampling") or {}
    resolved_targets = int(preflight.get("resolved_targets") or 0)
    if (
        not preflight.get("ok")
        or not sampling.get("stable")
        or not sampling.get("supervisor_contract_stable")
        or int(preflight.get("relaunch_ready_targets") or 0) != resolved_targets
        or resolved_targets <= 0
    ):
        issues.append(_issue("preflight_failed"))
        report["hard_failures"] = len(issues)
        return report

    contract = preflight.get("supervisor_contract") or {}
    observed_fingerprint = str(contract.get("fingerprint") or "")
    if observed_fingerprint != expected_fingerprint:
        issues.append(_issue("supervisor_fingerprint_mismatch"))
        report["hard_failures"] = len(issues)
        return report

    launch_snapshot, snapshot_fingerprint, snapshot_error = _load_launch_snapshot(
        supervisor_config
    )
    if snapshot_error is not None:
        issues.append(_issue("launch_snapshot_unavailable", reason=snapshot_error))
        report["hard_failures"] = len(issues)
        return report
    if snapshot_fingerprint != expected_fingerprint:
        issues.append(_issue("supervisor_changed_after_preflight"))
        report["hard_failures"] = len(issues)
        return report

    targets = sorted(
        [dict(target) for target in preflight.get("targets") or []],
        key=lambda target: str(target.get("window_name") or ""),
    )
    if {str(target.get("window_name") or "") for target in targets} != set(
        launch_snapshot
    ):
        issues.append(_issue("launch_target_set_mismatch"))
        report["hard_failures"] = len(issues)
        return report

    action_snapshot = build_live_restart_target_report(
        supervisor_config,
        session_name=confirmed_session,
        config_base_dir=config_base_dir,
    )
    action_fingerprint = str(
        (action_snapshot.get("supervisor_contract") or {}).get("fingerprint") or ""
    )
    if (
        not action_snapshot.get("ok")
        or action_fingerprint != expected_fingerprint
        or _target_identities(action_snapshot) != _target_identities(preflight)
    ):
        issues.append(_issue("target_changed_after_preflight"))
        report["hard_failures"] = len(issues)
        return report
    report["action_snapshot"] = _preflight_summary(action_snapshot)

    action_runtime_contract = _build_runtime_contract(expected_head)
    report["action_runtime_contract"] = action_runtime_contract
    if (
        not action_runtime_contract["ok"]
        or action_runtime_contract != runtime_contract
    ):
        issues.append(_issue("runtime_contract_changed_after_preflight"))
        report["hard_failures"] = len(issues)
        return report

    target_results: list[dict[str, Any]] = []
    result_by_pid: dict[int, dict[str, Any]] = {}
    stop_wait_targets: list[dict[str, Any]] = []
    report["action_started"] = True
    report["outcome"] = "restart_in_progress"
    for target in targets:
        pid = int(target["process_pid"])
        result = {
            "window_name": target["window_name"],
            "pane_id": target["pane_id"],
            "pane_pid": target["pane_pid"],
            "old_process_pid": pid,
            "stop_requested": False,
            "exited": False,
            "relaunch_requested": False,
            "relaunch_succeeded": False,
        }
        target_results.append(result)
        result_by_pid[pid] = result
        error = _send_graceful_interrupt(str(target["pane_id"]))
        if error is not None:
            result["stop_error"] = error
            issues.append(
                _issue(
                    "graceful_interrupt_failed",
                    window_name=target["window_name"],
                    pane_id=target["pane_id"],
                )
            )
            continue
        result["stop_requested"] = True
        stop_wait_targets.append(target)

    elapsed_by_pid, still_running = _wait_for_process_exits(
        stop_wait_targets,
        timeout_s=shutdown_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    for pid, elapsed_s in elapsed_by_pid.items():
        result_by_pid[pid]["exited"] = True
        result_by_pid[pid]["shutdown_elapsed_s"] = round(elapsed_s, 3)
    if still_running:
        issues.append(
            _issue(
                "shutdown_timeout",
                timed_out_targets=len(still_running),
            )
        )

    relaunch_pane_ids = {
        str(target["pane_id"])
        for target in targets
        if result_by_pid[int(target["process_pid"])]["exited"]
    }
    relaunch_window_names = {
        str(target["window_name"])
        for target in targets
        if result_by_pid[int(target["process_pid"])]["exited"]
    }
    processes_stable, process_error = _post_stop_process_recheck(
        launch_snapshot,
        relaunch_window_names=relaunch_window_names,
    )
    report["post_stop_process_recheck"] = {
        "ok": processes_stable,
        "error": process_error,
    }
    if not processes_stable:
        issues.append(
            _issue("post_stop_process_recheck_failed", reason=process_error)
        )
        report["targets"] = target_results
        report["hard_failures"] = len(issues)
        report["outcome"] = "manual_recovery_required"
        return report

    panes_stable, pane_error = _wait_for_post_stop_pane_recheck(
        targets,
        session_name=confirmed_session,
        relaunch_pane_ids=relaunch_pane_ids,
        poll_interval_s=poll_interval_s,
    )
    report["post_stop_pane_recheck"] = {
        "ok": panes_stable,
        "error": pane_error,
    }
    if not panes_stable:
        issues.append(
            _issue("post_stop_pane_recheck_failed", reason=pane_error)
        )
        report["targets"] = target_results
        report["hard_failures"] = len(issues)
        report["outcome"] = "manual_recovery_required"
        return report

    relaunch_snapshot, relaunch_fingerprint, relaunch_snapshot_error = (
        _load_launch_snapshot(supervisor_config)
    )
    supervisor_stable = (
        relaunch_snapshot_error is None
        and relaunch_fingerprint == expected_fingerprint
        and relaunch_snapshot == launch_snapshot
    )
    report["pre_relaunch_supervisor_recheck"] = {
        "ok": supervisor_stable,
        "error": relaunch_snapshot_error,
    }
    if not supervisor_stable:
        issues.append(_issue("supervisor_changed_before_relaunch"))
        report["targets"] = target_results
        report["hard_failures"] = len(issues)
        report["outcome"] = "manual_recovery_required"
        return report

    processes_stable, process_error = _post_stop_process_recheck(
        launch_snapshot,
        relaunch_window_names=relaunch_window_names,
    )
    report["pre_relaunch_process_recheck"] = {
        "ok": processes_stable,
        "error": process_error,
    }
    if not processes_stable:
        issues.append(
            _issue("process_changed_before_relaunch", reason=process_error)
        )
        report["targets"] = target_results
        report["hard_failures"] = len(issues)
        report["outcome"] = "manual_recovery_required"
        return report

    relaunch_runtime_contract = _build_runtime_contract(expected_head)
    report["pre_relaunch_runtime_contract"] = relaunch_runtime_contract
    if (
        not relaunch_runtime_contract["ok"]
        or relaunch_runtime_contract != runtime_contract
    ):
        issues.append(_issue("runtime_contract_changed_before_relaunch"))
        report["targets"] = target_results
        report["hard_failures"] = len(issues)
        report["outcome"] = "manual_recovery_required"
        return report

    for target in targets:
        result = result_by_pid[int(target["process_pid"])]
        if not result["exited"]:
            continue
        result["relaunch_requested"] = True
        error = _send_launch_command(
            str(target["pane_id"]),
            launch_snapshot[str(target["window_name"])]["command"],
        )
        if error is not None:
            result["relaunch_error"] = error
            issues.append(
                _issue(
                    "relaunch_command_failed",
                    window_name=target["window_name"],
                    pane_id=target["pane_id"],
                )
            )
            continue
        result["relaunch_succeeded"] = True

    startup_report = _wait_for_startup_report(
        supervisor_config,
        session_name=confirmed_session,
        config_base_dir=config_base_dir,
        timeout_s=startup_timeout_s,
        poll_interval_s=poll_interval_s,
        expected_supervisor_fingerprint=expected_fingerprint,
    )
    if startup_report.get("ok"):
        verification = build_live_restart_target_report(
            supervisor_config,
            session_name=confirmed_session,
            config_base_dir=config_base_dir,
            samples=verification_samples,
            sample_interval_s=verification_interval_s,
        )
    else:
        verification = startup_report
        issues.append(_issue("startup_verification_timeout"))

    final_by_window = {
        str(target.get("window_name") or ""): target
        for target in verification.get("targets") or []
    }
    for result in target_results:
        final = final_by_window.get(str(result["window_name"]))
        if final is not None:
            result["final_process_pid"] = final.get("process_pid")

    report["targets"] = target_results
    report["verification"] = _preflight_summary(verification)
    verified_fingerprint = str(
        (verification.get("supervisor_contract") or {}).get("fingerprint") or ""
    )
    if verified_fingerprint != expected_fingerprint:
        issues.append(_issue("supervisor_changed_during_restart"))
    if not verification.get("ok"):
        if not any(issue["code"] == "startup_verification_timeout" for issue in issues):
            issues.append(_issue("stable_verification_failed"))
    report["hard_failures"] = len(issues)
    report["ok"] = not issues and bool(verification.get("ok"))
    report["outcome"] = "completed" if report["ok"] else "recovered_with_errors"
    return report
