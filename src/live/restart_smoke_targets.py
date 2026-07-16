from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any

from live.smoke_report import build_live_process_report, summarize_live_process_report


MAX_RESTART_TARGETS = 64
MAX_TMUX_PANES = 256
MAX_TMUX_OUTPUT_CHARS = 262_144
MAX_TMUX_IDENTIFIER_CHARS = 160
MAX_TARGET_SAMPLES = 5
MAX_TARGET_SAMPLE_INTERVAL_S = 30.0
TMUX_PANE_FORMAT = (
    "#{pane_id}\t#{session_name}\t#{window_name}\t#{pane_index}\t"
    "#{pane_pid}\t#{pane_current_command}"
)

SAFETY_CONTRACT = {
    "local_only": True,
    "reads": [
        "tmux_pane_inventory",
        "process_table",
        "supervisor_config",
        "referenced_bot_configs",
    ],
    "network": False,
    "exchange_access": False,
    "credential_store_access": False,
    "process_control": False,
    "signals_processes": False,
    "starts_processes": False,
    "writes_files": False,
}


def _bounded_text(value: Any, *, max_len: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}...<truncated>"


def _tmux_pane_inventory() -> tuple[list[dict[str, Any]], str | None]:
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", TMUX_PANE_FORMAT],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"tmux_list_panes_failed:{exc.__class__.__name__}"
    if result.returncode != 0:
        return [], "tmux_list_panes_failed"
    if len(result.stdout) > MAX_TMUX_OUTPUT_CHARS:
        return [], "tmux_list_panes_output_too_large"

    panes: list[dict[str, Any]] = []
    pane_ids: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            return [], "tmux_list_panes_malformed_row"
        (
            pane_id,
            session_name,
            window_name,
            pane_index_raw,
            pane_pid_raw,
            current_command,
        ) = fields
        pane_id = pane_id.strip()
        session_name = session_name.strip()
        window_name = window_name.strip()
        if (
            not pane_id.startswith("%")
            or not pane_id[1:].isdigit()
            or len(pane_id) > MAX_TMUX_IDENTIFIER_CHARS
            or not session_name
            or not window_name
            or len(session_name) > MAX_TMUX_IDENTIFIER_CHARS
            or len(window_name) > MAX_TMUX_IDENTIFIER_CHARS
        ):
            return [], "tmux_list_panes_invalid_identifier"
        try:
            pane_index = int(pane_index_raw)
            pane_pid = int(pane_pid_raw)
        except ValueError:
            return [], "tmux_list_panes_malformed_row"
        if pane_index < 0 or pane_pid <= 0:
            return [], "tmux_list_panes_malformed_row"
        if pane_id in pane_ids:
            return [], "tmux_list_panes_duplicate_id"
        pane_ids.add(pane_id)
        panes.append(
            {
                "pane_id": pane_id,
                "session_name": session_name,
                "window_name": window_name,
                "pane_index": pane_index,
                "pane_pid": pane_pid,
                "current_command": _bounded_text(current_command, max_len=80),
            }
        )
        if len(panes) > MAX_TMUX_PANES:
            return [], "tmux_list_panes_too_many_rows"
    return panes, None


def _build_live_restart_target_snapshot(
    supervisor_config: str | Path,
    *,
    session_name: str,
    config_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve exact read-only tmux targets before any restart orchestration."""
    confirmed_session = str(session_name or "").strip()
    if not confirmed_session:
        raise ValueError("session_name must be non-empty")
    if len(confirmed_session) > MAX_TMUX_IDENTIFIER_CHARS:
        raise ValueError(
            f"session_name must not exceed {MAX_TMUX_IDENTIFIER_CHARS} characters"
        )

    processes = build_live_process_report(
        supervisor_config=supervisor_config,
        config_base_dir=config_base_dir,
    )
    expected = processes.get("expected")
    expected_rows = expected if isinstance(expected, list) else []
    panes, scan_error = _tmux_pane_inventory()
    session_panes = [
        pane for pane in panes if pane.get("session_name") == confirmed_session
    ]

    issues: list[dict[str, Any]] = []
    if scan_error is not None:
        issues.append({"code": scan_error, "severity": "error"})
    if len(expected_rows) > MAX_RESTART_TARGETS:
        issues.append(
            {
                "code": "expected_target_limit_exceeded",
                "severity": "error",
                "expected_total": len(expected_rows),
                "max_targets": MAX_RESTART_TARGETS,
            }
        )

    targets: list[dict[str, Any]] = []
    expected_window_names: list[str] = []
    for row in expected_rows[:MAX_RESTART_TARGETS]:
        window_name = str(row.get("name") or "").strip()
        if not window_name:
            issues.append(
                {
                    "code": "expected_window_name_missing",
                    "severity": "error",
                }
            )
            continue
        if len(window_name) > MAX_TMUX_IDENTIFIER_CHARS:
            issues.append(
                {
                    "code": "expected_window_name_too_long",
                    "severity": "error",
                }
            )
            continue
        if window_name in expected_window_names:
            issues.append(
                {
                    "code": "duplicate_expected_window_name",
                    "severity": "error",
                    "window_name": window_name,
                }
            )
            continue
        expected_window_names.append(window_name)
        matches = [
            pane for pane in session_panes if pane.get("window_name") == window_name
        ]
        if len(matches) != 1:
            issues.append(
                {
                    "code": (
                        "tmux_target_missing"
                        if not matches
                        else "tmux_target_duplicated"
                    ),
                    "severity": "error",
                    "window_name": window_name,
                    "match_count": len(matches),
                }
            )
            continue
        pane = matches[0]
        matched_processes = row.get("matched_processes")
        matched_rows = (
            matched_processes if isinstance(matched_processes, list) else []
        )
        if len(matched_rows) != 1:
            issues.append(
                {
                    "code": "process_target_not_unique",
                    "severity": "error",
                    "window_name": window_name,
                    "match_count": len(matched_rows),
                }
            )
            continue
        process = matched_rows[0]
        process_pid = int(process.get("pid") or 0)
        process_ppid = int(process.get("ppid") or 0)
        pane_pid = int(pane["pane_pid"])
        if process_pid <= 0 or (
            process_pid != pane_pid and process_ppid != pane_pid
        ):
            issues.append(
                {
                    "code": "tmux_process_parent_mismatch",
                    "severity": "error",
                    "window_name": window_name,
                    "pane_pid": pane_pid,
                    "process_pid": process_pid or None,
                    "process_ppid": process_ppid or None,
                }
            )
            continue
        targets.append(
            {
                "window_name": window_name,
                "target": pane["pane_id"],
                "pane_id": pane["pane_id"],
                "pane_index": int(pane["pane_index"]),
                "pane_pid": pane_pid,
                "process_pid": process_pid,
                "current_command": pane.get("current_command"),
                "ownership_proof": (
                    "matched_process_pid_equals_pane_pid"
                    if process_pid == pane_pid
                    else "matched_process_ppid_equals_pane_pid"
                ),
            }
        )

    extra_panes = [
        pane
        for pane in session_panes
        if pane.get("window_name") not in set(expected_window_names)
    ]
    if extra_panes:
        issues.append(
            {
                "code": "unconfigured_session_panes",
                "severity": "error",
                "count": len(extra_panes),
            }
        )

    target_hard_failures = sum(
        1 for issue in issues if issue.get("severity") == "error"
    )
    process_hard_failures = int(processes.get("hard_failures") or 0)
    hard_failures = process_hard_failures + target_hard_failures
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": hard_failures == 0,
        "hard_failures": hard_failures,
        "safety": SAFETY_CONTRACT,
        "session_name": confirmed_session,
        "expected_targets": len(expected_rows),
        "resolved_targets": len(targets),
        "session_panes": len(session_panes),
        "processes": summarize_live_process_report(processes),
        "targets": targets,
        "extra_panes": extra_panes,
        "issues": issues,
        "tmux_scan_error": scan_error,
    }


def _target_identity(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "pane_id": target["pane_id"],
        "pane_pid": target["pane_pid"],
        "process_pid": target["process_pid"],
        "ownership_proof": target["ownership_proof"],
    }


def _summarize_target_sampling(
    snapshots: list[dict[str, Any]],
    *,
    interval_s: float,
) -> dict[str, Any]:
    identity_maps = [
        {
            str(target.get("window_name") or ""): _target_identity(target)
            for target in snapshot.get("targets") or []
            if str(target.get("window_name") or "")
        }
        for snapshot in snapshots
    ]
    window_names = sorted(
        {window_name for identities in identity_maps for window_name in identities}
    )
    stable_targets = 0
    changed_target_rows: list[dict[str, Any]] = []
    for window_name in window_names:
        observations = [identities.get(window_name) for identities in identity_maps]
        first = observations[0]
        if first is not None and all(observation == first for observation in observations):
            stable_targets += 1
            continue
        changed_target_rows.append(
            {
                "window_name": window_name,
                "observations": [
                    {
                        "sample": sample_index,
                        "identity": observation,
                    }
                    for sample_index, observation in enumerate(observations, start=1)
                ],
            }
        )

    failed_samples = sum(1 for snapshot in snapshots if not snapshot.get("ok"))
    failed_sample_issues = [
        {
            "sample": sample_index,
            "hard_failures": int(snapshot.get("hard_failures") or 0),
            "issue_codes": [
                _bounded_text(issue.get("code"), max_len=80)
                for issue in (snapshot.get("issues") or [])[:10]
                if isinstance(issue, dict) and issue.get("code")
            ],
        }
        for sample_index, snapshot in enumerate(snapshots, start=1)
        if not snapshot.get("ok")
    ]
    stable = failed_samples == 0 and not changed_target_rows
    return {
        "requested_samples": len(snapshots),
        "collected_samples": len(snapshots),
        "interval_s": interval_s,
        "stable": stable,
        "successful_samples": len(snapshots) - failed_samples,
        "failed_samples": failed_samples,
        "failed_sample_issues": failed_sample_issues,
        "stable_targets": stable_targets,
        "changed_target_count": len(changed_target_rows),
        "changed_targets_truncated": max(
            0, len(changed_target_rows) - MAX_RESTART_TARGETS
        ),
        "changed_targets": changed_target_rows[:MAX_RESTART_TARGETS],
    }


def build_live_restart_target_report(
    supervisor_config: str | Path,
    *,
    session_name: str,
    config_base_dir: Path | None = None,
    samples: int = 1,
    sample_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Resolve exact read-only tmux targets before any restart orchestration."""
    samples = int(samples)
    sample_interval_s = float(sample_interval_s)
    if samples < 1 or samples > MAX_TARGET_SAMPLES:
        raise ValueError(f"samples must be between 1 and {MAX_TARGET_SAMPLES}")
    if (
        not math.isfinite(sample_interval_s)
        or sample_interval_s < 0.0
        or sample_interval_s > MAX_TARGET_SAMPLE_INTERVAL_S
    ):
        raise ValueError(
            "sample_interval_s must be between 0 and "
            f"{MAX_TARGET_SAMPLE_INTERVAL_S:g}"
        )

    snapshots: list[dict[str, Any]] = []
    for sample_index in range(samples):
        if sample_index > 0 and sample_interval_s > 0.0:
            time.sleep(sample_interval_s)
        snapshots.append(
            _build_live_restart_target_snapshot(
                supervisor_config,
                session_name=session_name,
                config_base_dir=config_base_dir,
            )
        )

    report = snapshots[-1]
    if samples == 1:
        return report

    sampling = _summarize_target_sampling(
        snapshots,
        interval_s=sample_interval_s,
    )
    report["sampling"] = sampling
    if not sampling["stable"]:
        report["issues"] = [
            *report["issues"],
            {
                "code": "target_sampling_unstable",
                "severity": "error",
                "failed_samples": sampling["failed_samples"],
                "changed_target_count": sampling["changed_target_count"],
            },
        ]
        report["hard_failures"] = int(report["hard_failures"]) + 1
        report["ok"] = False
    return report
