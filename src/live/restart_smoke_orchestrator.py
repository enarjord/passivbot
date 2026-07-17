from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

# isort: off
from live.restart_executor import (
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_PREFLIGHT_INTERVAL_S,
    DEFAULT_PREFLIGHT_SAMPLES,
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    DEFAULT_STARTUP_TIMEOUT_S,
    DEFAULT_VERIFICATION_INTERVAL_S,
    DEFAULT_VERIFICATION_SAMPLES,
    execute_live_restart,
)
from live.restart_smoke_collection import (
    DEFAULT_TARGET_SAMPLE_INTERVAL_S,
    DEFAULT_TARGET_SAMPLES,
    build_live_restart_smoke_collection,
)
from live.restart_smoke_evidence import (
    validate_live_restart_smoke_expectations,
)
from live.restart_smoke_targets import (
    MAX_TARGET_SAMPLE_INTERVAL_S,
    MAX_TARGET_SAMPLES,
    build_live_restart_target_report,
)
# isort: on

DEFAULT_SMOKE_WAIT_S = 600.0
MAX_SMOKE_WAIT_S = 1800.0
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RESTART_OUTCOMES = frozenset(
    {
        "completed",
        "manual_recovery_required",
        "preflight_failed",
        "recovered_with_errors",
        "restart_in_progress",
    }
)

SAFETY_CONTRACT = {
    "local_only": True,
    "ssh": False,
    "git_pull": False,
    "builds": False,
    "direct_network_access": False,
    "direct_exchange_access": False,
    "credential_store_access": False,
    "signals_live_processes": True,
    "starts_live_processes": True,
    "configured_live_processes_may_contact_exchanges": True,
    "configured_live_processes_may_write_files": True,
    "signals_exact_tmux_panes_only": True,
    "automatic_force_signal": False,
    "broad_process_pattern_signals": False,
    "post_restart_collection_read_only": True,
    "writes_report_files": False,
    "requires_execute_confirmation": True,
    "requires_expected_repository_head": True,
    "requires_expected_rust_source_fingerprint": True,
    "requires_expected_supervisor_fingerprint": True,
    "requires_expected_targets": True,
}


def _bounded_smoke_wait(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "smoke_wait_s must be greater than 0 and at most "
            f"{MAX_SMOKE_WAIT_S:g}"
        )
    wait_s = float(value)
    if (
        not math.isfinite(wait_s)
        or wait_s <= 0.0
        or wait_s > MAX_SMOKE_WAIT_S
    ):
        raise ValueError(
            "smoke_wait_s must be greater than 0 and at most "
            f"{MAX_SMOKE_WAIT_S:g}"
        )
    return wait_s


def _required_text(value: str | Path, *, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _validate_smoke_collection_options(
    *,
    target_samples: int,
    target_sample_interval_s: float,
) -> None:
    if (
        type(target_samples) is not int
        or target_samples < 2
        or target_samples > MAX_TARGET_SAMPLES
    ):
        raise ValueError(
            f"smoke_target_samples must be between 2 and {MAX_TARGET_SAMPLES}"
        )
    if (
        not isinstance(target_sample_interval_s, (int, float))
        or isinstance(target_sample_interval_s, bool)
        or not math.isfinite(target_sample_interval_s)
        or target_sample_interval_s < 0.0
        or target_sample_interval_s > MAX_TARGET_SAMPLE_INTERVAL_S
    ):
        raise ValueError(
            "smoke_target_interval_s must be between 0 and "
            f"{MAX_TARGET_SAMPLE_INTERVAL_S:g}"
        )


def _valid_rust_fingerprint(value: str) -> str:
    fingerprint = str(value or "").strip()
    if not _SHA256_PATTERN.fullmatch(fingerprint):
        raise ValueError(
            "expected_rust_source_fingerprint must be 64 lowercase hex "
            "characters"
        )
    return fingerprint


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _epoch_ms() -> int:
    return time.time_ns() // 1_000_000


def _target_preflight_ok(
    report: dict[str, Any],
    *,
    expected_targets: int,
    expected_supervisor_fingerprint: str,
) -> bool:
    issues = report.get("issues")
    extra_panes = report.get("extra_panes")
    targets = report.get("targets")
    sampling = report.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    contract = report.get("supervisor_contract")
    contract = contract if isinstance(contract, dict) else {}
    return bool(
        report.get("tool") == "live-restart-target-report"
        and _exact_int(report.get("schema_version"), 1)
        and report.get("ok") is True
        and _exact_int(report.get("hard_failures"), 0)
        and issues == []
        and extra_panes == []
        and isinstance(targets, list)
        and len(targets) == expected_targets
        and _exact_int(report.get("expected_targets"), expected_targets)
        and _exact_int(report.get("resolved_targets"), expected_targets)
        and _exact_int(
            report.get("relaunch_ready_targets"), expected_targets
        )
        and _exact_int(report.get("relaunch_unready_targets"), 0)
        and type(sampling.get("requested_samples")) is int
        and sampling.get("requested_samples") >= 2
        and _exact_int(
            sampling.get("collected_samples"),
            sampling["requested_samples"],
        )
        and _exact_int(
            sampling.get("successful_samples"),
            sampling["requested_samples"],
        )
        and _exact_int(sampling.get("failed_samples"), 0)
        and sampling.get("failed_sample_issues") == []
        and sampling.get("stable") is True
        and sampling.get("supervisor_contract_stable") is True
        and sampling.get("supervisor_contract_changed") is False
        and _exact_int(sampling.get("stable_targets"), expected_targets)
        and _exact_int(sampling.get("changed_target_count"), 0)
        and _exact_int(sampling.get("changed_targets_truncated"), 0)
        and sampling.get("changed_targets") == []
        and contract.get("fingerprint") == expected_supervisor_fingerprint
        and contract.get("source") == "parsed_supervisor_config"
        and contract.get("algorithm") == "sha256"
        and _exact_int(contract.get("target_count"), expected_targets)
        and contract.get("command_content_exposed") is False
    )


def _strict_nonnegative_int(value: Any) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _restart_summary(
    report: dict[str, Any],
    *,
    expected_targets: int,
    expected_supervisor_fingerprint: str,
) -> dict[str, Any]:
    targets = report.get("targets")
    target_rows = targets if isinstance(targets, list) else []
    verification = report.get("verification")
    verification = verification if isinstance(verification, dict) else {}
    verification_contract = verification.get("supervisor_contract")
    verification_contract = (
        verification_contract
        if isinstance(verification_contract, dict)
        else {}
    )
    hard_failures = _strict_nonnegative_int(report.get("hard_failures"))
    verified_targets = _strict_nonnegative_int(
        verification.get("resolved_targets")
    )
    target_counts = {
        "targets": len(target_rows),
        "stop_requested": sum(
            row.get("stop_requested") is True
            for row in target_rows
            if isinstance(row, dict)
        ),
        "exited": sum(
            row.get("exited") is True
            for row in target_rows
            if isinstance(row, dict)
        ),
        "relaunch_requested": sum(
            row.get("relaunch_requested") is True
            for row in target_rows
            if isinstance(row, dict)
        ),
        "relaunch_succeeded": sum(
            row.get("relaunch_succeeded") is True
            for row in target_rows
            if isinstance(row, dict)
        ),
        "verified_targets": verified_targets,
    }
    report_shape_valid = bool(
        report.get("tool") == "live-restart-executor"
        and _exact_int(report.get("schema_version"), 3)
        and report.get("outcome") in _RESTART_OUTCOMES
        and type(report.get("action_started")) is bool
        and hard_failures is not None
        and isinstance(report.get("issues"), list)
        and isinstance(targets, list)
        and all(isinstance(row, dict) for row in target_rows)
    )
    completed_contract_valid = bool(
        report_shape_valid
        and report.get("outcome") == "completed"
        and report.get("action_started") is True
        and hard_failures == 0
        and report.get("issues") == []
        and all(count == expected_targets for count in target_counts.values())
        and verification.get("ok") is True
        and _exact_int(verification.get("hard_failures"), 0)
        and _exact_int(
            verification.get("expected_targets"), expected_targets
        )
        and _exact_int(
            verification.get("relaunch_ready_targets"), expected_targets
        )
        and verification_contract.get("fingerprint")
        == expected_supervisor_fingerprint
        and _exact_int(
            verification_contract.get("target_count"), expected_targets
        )
        and verification_contract.get("command_content_exposed") is False
    )
    contract_valid = bool(
        report_shape_valid
        and (
            completed_contract_valid
            if report.get("ok") is True
            else hard_failures > 0
        )
    )
    return {
        "ok": report.get("ok") is True and completed_contract_valid,
        "contract_valid": contract_valid,
        "outcome": (
            report.get("outcome")
            if report.get("outcome") in _RESTART_OUTCOMES
            else None
        ),
        "action_started": report.get("action_started") is True,
        "hard_failures": hard_failures,
        **target_counts,
        "verification_ok": verification.get("ok") is True,
    }


def _issue(code: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "severity": "error", **fields}


def _finish_report(report: dict[str, Any]) -> dict[str, Any]:
    issues = report.get("issues")
    issue_rows = issues if isinstance(issues, list) else []
    report["hard_failures"] = len(issue_rows)
    return report


def _smoke_collection_contract_valid(report: dict[str, Any]) -> bool:
    hard_failures = _strict_nonnegative_int(report.get("hard_failures"))
    issues = report.get("issues")
    return bool(
        report.get("tool") == "live-restart-smoke-collect"
        and _exact_int(report.get("schema_version"), 1)
        and type(report.get("ok")) is bool
        and hard_failures is not None
        and isinstance(issues, list)
        and all(isinstance(issue, dict) for issue in issues)
        and isinstance(report.get("gates"), dict)
        and isinstance(report.get("evidence"), dict)
        and (
            hard_failures == 0 and issues == []
            if report.get("ok") is True
            else hard_failures > 0
        )
    )


def execute_live_restart_smoke(
    supervisor_config: str | Path,
    *,
    session_name: str,
    monitor_root: str | Path,
    expected_repository_head: str,
    expected_rust_source_fingerprint: str,
    expected_supervisor_fingerprint: str,
    expected_targets: int,
    logs_root: str | Path | None = None,
    smoke_wait_s: float = DEFAULT_SMOKE_WAIT_S,
    preflight_samples: int = DEFAULT_PREFLIGHT_SAMPLES,
    preflight_interval_s: float = DEFAULT_PREFLIGHT_INTERVAL_S,
    shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    verification_samples: int = DEFAULT_VERIFICATION_SAMPLES,
    verification_interval_s: float = DEFAULT_VERIFICATION_INTERVAL_S,
    smoke_target_samples: int = DEFAULT_TARGET_SAMPLES,
    smoke_target_interval_s: float = DEFAULT_TARGET_SAMPLE_INTERVAL_S,
    execute: bool = False,
) -> dict[str, Any]:
    """Restart exact local targets, then collect one bounded smoke window."""
    supervisor_config = _required_text(
        supervisor_config, field="supervisor_config"
    )
    session_name = _required_text(session_name, field="session_name")
    monitor_root = _required_text(monitor_root, field="monitor_root")
    if logs_root is not None:
        logs_root = _required_text(logs_root, field="logs_root")
    validate_live_restart_smoke_expectations(
        expected_repository_head=expected_repository_head,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_targets=expected_targets,
    )
    rust_fingerprint = _valid_rust_fingerprint(
        expected_rust_source_fingerprint
    )
    wait_s = _bounded_smoke_wait(smoke_wait_s)
    _validate_smoke_collection_options(
        target_samples=smoke_target_samples,
        target_sample_interval_s=smoke_target_interval_s,
    )
    if not execute:
        raise ValueError("execute must be true")

    report: dict[str, Any] = {
        "tool": "live-restart-smoke-run",
        "schema_version": 1,
        "ok": False,
        "outcome": "preflight_failed",
        "action_started": False,
        "hard_failures": 0,
        "restart": None,
        "smoke": None,
        "window": None,
        "issues": [],
        "safety": dict(SAFETY_CONTRACT),
    }
    issues: list[dict[str, Any]] = report["issues"]
    preflight = build_live_restart_target_report(
        supervisor_config,
        session_name=session_name,
        config_base_dir=Path.cwd(),
        samples=preflight_samples,
        sample_interval_s=preflight_interval_s,
    )
    if not _target_preflight_ok(
        preflight,
        expected_targets=expected_targets,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
    ):
        issues.append(_issue("target_preflight_failed"))
        return _finish_report(report)

    since_ms = _epoch_ms()
    restart = execute_live_restart(
        supervisor_config,
        session_name=session_name,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_repository_head=expected_repository_head,
        expected_rust_source_fingerprint=rust_fingerprint,
        config_base_dir=Path.cwd(),
        preflight_samples=preflight_samples,
        preflight_interval_s=preflight_interval_s,
        shutdown_timeout_s=shutdown_timeout_s,
        startup_timeout_s=startup_timeout_s,
        poll_interval_s=poll_interval_s,
        verification_samples=verification_samples,
        verification_interval_s=verification_interval_s,
        execute=True,
    )
    restart_summary = _restart_summary(
        restart,
        expected_targets=expected_targets,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
    )
    report["restart"] = restart_summary
    report["action_started"] = restart_summary["action_started"]
    if not restart_summary["ok"]:
        report["outcome"] = (
            "manual_recovery_required"
            if restart_summary["action_started"]
            else "preflight_failed"
        )
        issues.append(
            _issue(
                "restart_failed"
                if restart_summary["contract_valid"]
                else "restart_contract_invalid"
            )
        )
        return _finish_report(report)

    report["outcome"] = "restart_completed_smoke_pending"
    time.sleep(wait_s)
    until_ms = _epoch_ms()
    report["window"] = {
        "since_ms": since_ms,
        "until_ms": until_ms,
        "observation_wait_s": wait_s,
    }
    if until_ms <= since_ms:
        report["outcome"] = "restart_completed_smoke_failed"
        issues.append(_issue("smoke_window_clock_invalid"))
        return _finish_report(report)

    try:
        smoke = build_live_restart_smoke_collection(
            supervisor_config,
            session_name=session_name,
            monitor_root=monitor_root,
            logs_root=logs_root,
            expected_repository_head=expected_repository_head,
            expected_supervisor_fingerprint=expected_supervisor_fingerprint,
            expected_targets=expected_targets,
            since_ms=since_ms,
            until_ms=until_ms,
            target_samples=smoke_target_samples,
            target_sample_interval_s=smoke_target_interval_s,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        report["outcome"] = "restart_completed_smoke_failed"
        issues.append(
            _issue(
                "smoke_collection_failed",
                error_class=exc.__class__.__name__,
            )
        )
        return _finish_report(report)
    if not _smoke_collection_contract_valid(smoke):
        report["outcome"] = "restart_completed_smoke_failed"
        issues.append(_issue("smoke_contract_invalid"))
        return _finish_report(report)
    report["smoke"] = smoke
    if smoke.get("ok") is not True:
        report["outcome"] = "restart_completed_smoke_failed"
        issues.append(_issue("smoke_failed"))
        return _finish_report(report)

    report["ok"] = True
    report["outcome"] = "completed"
    return _finish_report(report)
