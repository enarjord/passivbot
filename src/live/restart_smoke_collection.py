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
) -> dict[str, Any]:
    return {
        "target_sampling": {
            "samples": target_samples,
            "sample_interval_s": target_sample_interval_s,
        },
        "smoke_collection": {
            "include_rotated": True,
            "include_processes": False,
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
        ),
        "gates": evaluation["gates"],
        "evidence": evaluation["evidence"],
    }
