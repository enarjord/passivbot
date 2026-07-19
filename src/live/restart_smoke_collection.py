from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from live.restart_smoke_evidence import (
    MAX_ISSUES,
    SCHEMA_VERSION,
    build_live_restart_smoke_evidence,
    validate_live_restart_smoke_epoch_window,
    validate_live_restart_smoke_expectations,
)
from live.restart_smoke_targets import (
    MAX_TARGET_SAMPLE_INTERVAL_S,
    MAX_TARGET_SAMPLES,
    build_live_restart_target_report,
)
from live.smoke_report import build_live_smoke_report, default_logs_root_for_monitor


DEFAULT_TARGET_SAMPLES = 3
DEFAULT_TARGET_SAMPLE_INTERVAL_S = 1.0
MAX_WINDOW_EVENT_FILES_PER_BOT = 8
MAX_WINDOW_EVENT_FILES_TOTAL = 128
MAX_WINDOW_EVENT_BYTES_TOTAL = 128 * 1024 * 1024
MAX_SELECTION_ISSUE_CODES = 10
MAX_PROJECTED_COUNT = 1_000_000_000
SEGMENT_SELECTION_ISSUE_CODES = frozenset(
    {
        "current_event_segment_duplicated",
        "current_event_segment_missing",
        "event_segment_name_unparseable",
        "event_window_segment_limit_exceeded",
        "event_window_segment_size_failed",
        "event_window_start_coverage_unavailable",
        "event_window_total_byte_limit_exceeded",
        "event_window_total_file_limit_exceeded",
    }
)

SAFETY_CONTRACT = {
    "local_only": True,
    "read_only": True,
    "local_filesystem_reads": True,
    "subprocess_execution": True,
    "bounded_local_subprocess_inventory": ["git", "ps", "tmux"],
    "network": False,
    "exchange_access": False,
    "credential_store_access": False,
    "process_control": False,
    "signals_processes": False,
    "starts_processes": False,
    "writes_files": False,
    "ssh": False,
    "git_pull": False,
    "builds": False,
}


def _required_text(value: str | Path, *, label: str) -> str:
    if value is None:
        raise ValueError(f"{label} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _validate_collection_inputs(
    *,
    supervisor_config: str | Path,
    session_name: str,
    monitor_root: str | Path,
    expected_repository_head: str,
    expected_supervisor_fingerprint: str,
    expected_targets: int,
    since_ms: int,
    until_ms: int,
) -> None:
    _required_text(supervisor_config, label="supervisor_config")
    _required_text(session_name, label="session_name")
    _required_text(monitor_root, label="monitor_root")
    validate_live_restart_smoke_expectations(
        expected_repository_head=expected_repository_head,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_targets=expected_targets,
    )
    validate_live_restart_smoke_epoch_window(since_ms=since_ms, until_ms=until_ms)


def _collection_policy(
    *,
    target_samples: int,
    target_sample_interval_s: float,
    since_ms: int,
    until_ms: int,
    logs_root_was_supplied: bool,
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    event_window = smoke_report.get("event_window")
    event_window = event_window if isinstance(event_window, dict) else {}
    selection = event_window.get("segment_selection")
    selection = selection if isinstance(selection, dict) else {}

    def projected_count(key: str) -> int | None:
        value = selection.get(key)
        if type(value) is not int or value < 0 or value > MAX_PROJECTED_COUNT:
            return None
        return value

    issue_counts = selection.get("issue_counts")
    issue_counts = issue_counts if isinstance(issue_counts, dict) else {}
    known_issue_counts = [
        (str(code), count)
        for code, count in sorted(issue_counts.items())
        if str(code) in SEGMENT_SELECTION_ISSUE_CODES
    ]
    projected_issue_counts = {
        code: int(count)
        for code, count in known_issue_counts[:MAX_SELECTION_ISSUE_CODES]
        if type(count) is int and 0 < count <= MAX_PROJECTED_COUNT
    }
    selection_evidence = {
        "complete": selection.get("complete") is True,
        "groups": projected_count("groups"),
        "files_before_selection": projected_count("files_before_selection"),
        "candidate_files_selected": projected_count("candidate_files_selected"),
        "candidate_scan_bytes": projected_count("candidate_scan_bytes"),
        "files_selected": projected_count("files_selected"),
        "files_skipped": projected_count("files_skipped"),
        "issue_counts": projected_issue_counts,
        "issue_codes_truncated": max(
            0,
            len(issue_counts) - len(projected_issue_counts),
        ),
    }
    return {
        "target_sampling": {
            "samples": target_samples,
            "sample_interval_s": target_sample_interval_s,
        },
        "smoke_collection": {
            "include_rotated": True,
            "include_processes": False,
            "event_segment_selection": "rotation_end_overlap_with_predecessor",
            "max_window_event_files_per_bot": MAX_WINDOW_EVENT_FILES_PER_BOT,
            "max_window_event_files_total": MAX_WINDOW_EVENT_FILES_TOTAL,
            "max_window_event_bytes_total": MAX_WINDOW_EVENT_BYTES_TOTAL,
            "event_segment_evidence": selection_evidence,
            "log_window_unparsed_policy": "drop",
            "logs_root_source": "caller" if logs_root_was_supplied else "monitor_default",
            "event_window": {"since_ms": since_ms, "until_ms": until_ms},
        },
    }


def build_live_restart_smoke_collection(
    supervisor_config: str | Path,
    *,
    session_name: str,
    monitor_root: str | Path,
    expected_repository_head: str,
    expected_supervisor_fingerprint: str,
    expected_targets: int,
    since_ms: int,
    until_ms: int,
    logs_root: str | Path | None = None,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    target_sample_interval_s: float = DEFAULT_TARGET_SAMPLE_INTERVAL_S,
) -> dict[str, Any]:
    """Collect and evaluate bounded local restart-smoke evidence in memory."""
    _validate_collection_inputs(
        supervisor_config=supervisor_config,
        session_name=session_name,
        monitor_root=monitor_root,
        expected_repository_head=expected_repository_head,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_targets=expected_targets,
        since_ms=since_ms,
        until_ms=until_ms,
    )
    if (
        type(target_samples) is not int
        or target_samples < 2
        or target_samples > MAX_TARGET_SAMPLES
    ):
        raise ValueError(
            f"target_samples must be between 2 and {MAX_TARGET_SAMPLES}"
        )
    if (
        not isinstance(target_sample_interval_s, (int, float))
        or isinstance(target_sample_interval_s, bool)
        or not math.isfinite(target_sample_interval_s)
        or target_sample_interval_s < 0.0
        or target_sample_interval_s > MAX_TARGET_SAMPLE_INTERVAL_S
    ):
        raise ValueError(
            "target_sample_interval_s must be between 0 and "
            f"{MAX_TARGET_SAMPLE_INTERVAL_S:g}"
        )

    if logs_root is not None:
        _required_text(logs_root, label="logs_root")
    target_report = build_live_restart_target_report(
        supervisor_config,
        session_name=session_name,
        config_base_dir=Path.cwd(),
        samples=target_samples,
        sample_interval_s=float(target_sample_interval_s),
    )
    resolved_logs_root = (
        logs_root
        if logs_root is not None
        else default_logs_root_for_monitor(monitor_root)
    )
    smoke_report = build_live_smoke_report(
        monitor_root,
        logs_root=resolved_logs_root,
        include_rotated=True,
        include_processes=False,
        since_ms=since_ms,
        until_ms=until_ms,
        select_event_segments_for_window=True,
        max_window_event_files_per_bot=MAX_WINDOW_EVENT_FILES_PER_BOT,
        max_window_event_files_total=MAX_WINDOW_EVENT_FILES_TOTAL,
        max_window_event_bytes_total=MAX_WINDOW_EVENT_BYTES_TOTAL,
        log_window_unparsed_policy="drop",
    )
    evaluation = build_live_restart_smoke_evidence(
        target_report,
        smoke_report,
        expected_repository_head=expected_repository_head,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_targets=expected_targets,
    )
    return {
        "tool": "live-restart-smoke-collect",
        "schema_version": SCHEMA_VERSION,
        "ok": evaluation["ok"],
        "hard_failures": evaluation["hard_failures"],
        "issues": evaluation["issues"][:MAX_ISSUES],
        "safety": SAFETY_CONTRACT,
        "collection": _collection_policy(
            target_samples=target_samples,
            target_sample_interval_s=float(target_sample_interval_s),
            since_ms=since_ms,
            until_ms=until_ms,
            logs_root_was_supplied=logs_root is not None,
            smoke_report=smoke_report,
        ),
        "gates": evaluation["gates"],
        "evidence": evaluation["evidence"],
    }
