from __future__ import annotations

import re
from typing import Any

from live.restart_smoke_targets import MAX_RESTART_TARGETS, MAX_TARGET_SAMPLES


SCHEMA_VERSION = 1
MAX_ISSUES = 16
MAX_BOT_IDENTIFIER_CHARS = 160
MAX_PROJECTED_COUNT = 1_000_000_000
MAX_EPOCH_MS = 253_402_300_799_999
_GIT_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

SAFETY_CONTRACT = {
    "local_only": True,
    "read_only": True,
    "subprocess_execution": False,
    "signals_processes": False,
    "starts_processes": False,
    "network": False,
    "exchange_access": False,
    "writes_files": False,
}


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_non_negative_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _is_epoch_ms(value: Any) -> bool:
    return _is_int(value) and 0 <= value <= MAX_EPOCH_MS


def _count(value: Any) -> int:
    if not _is_non_negative_int(value):
        return 0
    return min(value, MAX_PROJECTED_COUNT)


def _valid_head(value: Any) -> bool:
    return type(value) is str and bool(_GIT_HEAD_PATTERN.fullmatch(value))


def _valid_fingerprint(value: Any) -> bool:
    return type(value) is str and bool(_SHA256_PATTERN.fullmatch(value))


def validate_live_restart_smoke_expectations(
    *,
    expected_repository_head: str,
    expected_supervisor_fingerprint: str,
    expected_targets: int,
) -> None:
    if not _valid_head(expected_repository_head):
        raise ValueError("expected_repository_head must be 40 lowercase hex characters")
    if not _valid_fingerprint(expected_supervisor_fingerprint):
        raise ValueError(
            "expected_supervisor_fingerprint must be 64 lowercase hex characters"
        )
    if (
        not _is_int(expected_targets)
        or expected_targets < 1
        or expected_targets > MAX_RESTART_TARGETS
    ):
        raise ValueError(
            f"expected_targets must be between 1 and {MAX_RESTART_TARGETS}"
        )


def validate_live_restart_smoke_epoch_window(
    *, since_ms: int, until_ms: int
) -> None:
    if not _is_epoch_ms(since_ms) or not _is_epoch_ms(until_ms):
        raise ValueError(
            f"since_ms and until_ms must be epoch-ms integers between 0 and {MAX_EPOCH_MS}"
        )
    if until_ms <= since_ms:
        raise ValueError("until_ms must be greater than since_ms")


def _issue(issues: list[dict[str, Any]], code: str) -> None:
    if len(issues) < MAX_ISSUES:
        issues.append({"code": code, "severity": "error", "count": 1})


def _target_gate(
    report: dict[str, Any],
    *,
    expected_targets: int,
    expected_supervisor_fingerprint: str,
) -> dict[str, Any]:
    contract = report.get("supervisor_contract")
    contract = contract if isinstance(contract, dict) else {}
    sampling = report.get("sampling")
    sampling = sampling if isinstance(sampling, dict) else {}
    report_issues = report.get("issues")
    report_issues = report_issues if isinstance(report_issues, list) else None
    extra_panes = report.get("extra_panes")
    extra_panes = extra_panes if isinstance(extra_panes, list) else None

    expected_value = report.get("expected_targets")
    resolved_value = report.get("resolved_targets")
    relaunch_ready_value = report.get("relaunch_ready_targets")
    requested_value = sampling.get("requested_samples")
    collected_value = sampling.get("collected_samples")
    successful_value = sampling.get("successful_samples")
    failed_value = sampling.get("failed_samples")
    stable_targets_value = sampling.get("stable_targets")
    changed_targets_value = sampling.get("changed_target_count")
    expected_count = _count(expected_value)
    resolved_count = _count(resolved_value)
    relaunch_ready_count = _count(relaunch_ready_value)
    requested_samples = _count(requested_value)
    successful_samples = _count(successful_value)
    failed_samples = _count(failed_value)
    stable_targets = _count(stable_targets_value)
    changed_targets = _count(changed_targets_value)

    tool_schema_ok = (
        report.get("tool") == "live-restart-target-report"
        and report.get("schema_version") == 1
        and _is_non_negative_int(report.get("schema_version"))
    )
    report_health_ok = (
        report.get("ok") is True
        and report.get("hard_failures") == 0
        and _is_non_negative_int(report.get("hard_failures"))
        and report_issues == []
    )
    target_counts_ok = (
        expected_value == expected_targets
        and resolved_value == expected_targets
        and relaunch_ready_value == expected_targets
        and _is_non_negative_int(expected_value)
        and _is_non_negative_int(resolved_value)
        and _is_non_negative_int(relaunch_ready_value)
        and extra_panes == []
    )
    supervisor_contract_ok = (
        contract.get("source") == "parsed_supervisor_config"
        and contract.get("algorithm") == "sha256"
        and contract.get("fingerprint") == expected_supervisor_fingerprint
        and contract.get("target_count") == expected_targets
        and _is_non_negative_int(contract.get("target_count"))
        and contract.get("command_content_exposed") is False
    )
    sampling_ok = (
        _is_non_negative_int(requested_value)
        and 2 <= requested_value <= MAX_TARGET_SAMPLES
        and collected_value == requested_value
        and successful_value == requested_value
        and failed_value == 0
        and _is_non_negative_int(collected_value)
        and _is_non_negative_int(successful_value)
        and _is_non_negative_int(failed_value)
        and sampling.get("stable") is True
        and sampling.get("supervisor_contract_stable") is True
        and sampling.get("supervisor_contract_changed") is False
        and stable_targets_value == expected_targets
        and changed_targets_value == 0
        and _is_non_negative_int(stable_targets_value)
        and _is_non_negative_int(changed_targets_value)
    )
    ok = (
        tool_schema_ok
        and report_health_ok
        and target_counts_ok
        and supervisor_contract_ok
        and sampling_ok
    )
    return {
        "ok": ok,
        "tool_schema_ok": tool_schema_ok,
        "report_health_ok": report_health_ok,
        "target_counts_ok": target_counts_ok,
        "supervisor_contract_ok": supervisor_contract_ok,
        "sampling_ok": sampling_ok,
        "expected_targets": expected_targets,
        "expected_count": expected_count,
        "resolved_count": resolved_count,
        "relaunch_ready_count": relaunch_ready_count,
        "extra_pane_count": len(extra_panes) if extra_panes is not None else 0,
        "requested_samples": requested_samples,
        "successful_samples": successful_samples,
        "failed_samples": failed_samples,
        "stable_targets": stable_targets,
        "changed_target_count": changed_targets,
    }


def _smoke_contract_gate(report: dict[str, Any]) -> dict[str, Any]:
    tool_ok = "tool" not in report or report.get("tool") == "live-smoke-report"
    schema_ok = "schema_version" not in report or (
        report.get("schema_version") == 1
        and _is_non_negative_int(report.get("schema_version"))
    )
    hard_failures = _count(report.get("hard_failures"))
    report_ok = (
        report.get("ok") is True
        and report.get("hard_failures") == 0
        and _is_non_negative_int(report.get("hard_failures"))
    )
    return {
        "ok": tool_ok and schema_ok and report_ok,
        "tool_schema_ok": tool_ok and schema_ok,
        "report_ok": report_ok,
        "hard_failures": hard_failures,
    }


def _repository_gate(
    report: dict[str, Any], *, expected_repository_head: str
) -> dict[str, Any]:
    repository = report.get("repository")
    repository = repository if isinstance(repository, dict) else {}
    head_matches_expected = repository.get("head_full") == expected_repository_head
    tracked_clean = (
        repository.get("is_git_repo") is True
        and repository.get("error") is None
        and repository.get("dirty") is False
        and repository.get("tracked_changes") == 0
        and _is_non_negative_int(repository.get("tracked_changes"))
    )
    return {
        "ok": head_matches_expected and tracked_clean,
        "head_matches_expected": head_matches_expected,
        "tracked_clean": tracked_clean,
        "tracked_change_count": _count(repository.get("tracked_changes")),
    }


def _window_gate(window: Any) -> tuple[bool, int | None, int | None]:
    row = window if isinstance(window, dict) else {}
    since_ms = row.get("since_ms")
    until_ms = row.get("until_ms")
    valid_since_ms = _is_epoch_ms(since_ms)
    valid_until_ms = _is_epoch_ms(until_ms)
    ok = (
        row.get("enabled") is True
        and valid_since_ms
        and valid_until_ms
        and until_ms > since_ms
    )
    return (
        ok,
        since_ms if valid_since_ms else None,
        until_ms if valid_until_ms else None,
    )


def _event_window_gate(report: dict[str, Any]) -> dict[str, Any]:
    ok, since_ms, until_ms = _window_gate(report.get("event_window"))
    return {"ok": ok, "since_ms": since_ms, "until_ms": until_ms}


def _monitor_gate(report: dict[str, Any]) -> dict[str, Any]:
    monitor = report.get("monitor")
    monitor = monitor if isinstance(monitor, dict) else {}
    files_scanned = _count(monitor.get("files_scanned"))
    error_count = _count(monitor.get("error_count"))
    return {
        "ok": (
            _is_non_negative_int(monitor.get("files_scanned"))
            and files_scanned > 0
            and monitor.get("error_count") == 0
            and _is_non_negative_int(monitor.get("error_count"))
        ),
        "files_scanned": files_scanned,
        "error_count": error_count,
        "warning_count": _count(monitor.get("warning_count")),
    }


def _logs_gate(report: dict[str, Any], event_window: dict[str, Any]) -> dict[str, Any]:
    logs = report.get("logs")
    logs = logs if isinstance(logs, dict) else {}
    window_ok, since_ms, until_ms = _window_gate(logs.get("window"))
    same_bounds = (
        window_ok
        and since_ms == event_window["since_ms"]
        and until_ms == event_window["until_ms"]
    )
    files_scanned = _count(logs.get("files_scanned"))
    hard_matches = _count(logs.get("hard_matches"))
    dropped_unparsed_hard_matches = _count(logs.get("dropped_unparsed_hard_matches"))
    return {
        "ok": (
            window_ok
            and same_bounds
            and _is_non_negative_int(logs.get("files_scanned"))
            and files_scanned > 0
            and logs.get("hard_matches") == 0
            and _is_non_negative_int(logs.get("hard_matches"))
            and logs.get("dropped_unparsed_hard_matches") == 0
            and _is_non_negative_int(logs.get("dropped_unparsed_hard_matches"))
        ),
        "window_ok": window_ok,
        "same_bounds_as_events": same_bounds,
        "files_scanned": files_scanned,
        "hard_matches": hard_matches,
        "dropped_unparsed_hard_matches": dropped_unparsed_hard_matches,
        "attention_matches": _count(logs.get("attention_matches")),
    }


def _shutdown_gate(report: dict[str, Any], *, expected_targets: int) -> dict[str, Any]:
    shutdown = report.get("shutdown_events")
    shutdown = shutdown if isinstance(shutdown, dict) else {}
    event_types = shutdown.get("event_types")
    event_types = event_types if isinstance(event_types, dict) else {}
    stopping_count = _count(event_types.get("bot.stopping"))
    stopped_count = _count(event_types.get("bot.stopped"))
    return {
        "ok": (
            _is_non_negative_int(event_types.get("bot.stopping"))
            and _is_non_negative_int(event_types.get("bot.stopped"))
            and stopping_count >= expected_targets
            and stopped_count >= expected_targets
        ),
        "stopping_count": stopping_count,
        "stopped_count": stopped_count,
    }


def _startup_gate(report: dict[str, Any], *, expected_targets: int) -> dict[str, Any]:
    rows = report.get("startup_timings")
    rows = rows if isinstance(rows, list) else []
    bot_rows = {
        row.get("bot")
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("bot"), str)
        and 0 < len(row["bot"]) <= MAX_BOT_IDENTIFIER_CHARS
        and isinstance(row.get("phases"), dict)
        and bool(row["phases"])
    }
    startup_bot_count = min(len(bot_rows), MAX_PROJECTED_COUNT)
    return {
        "ok": startup_bot_count >= expected_targets,
        "startup_bot_count": startup_bot_count,
    }


def _attention_evidence(
    report: dict[str, Any], monitor: dict[str, Any], logs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "reported": report.get("attention") is True,
        "reported_count": _count(report.get("attention_count")),
        "monitor_warning_count": monitor["warning_count"],
        "log_attention_matches": logs["attention_matches"],
    }


def build_live_restart_smoke_evidence(
    target_report: dict[str, Any],
    smoke_report: dict[str, Any],
    *,
    expected_repository_head: str,
    expected_supervisor_fingerprint: str,
    expected_targets: int,
) -> dict[str, Any]:
    """Evaluate existing bounded reports without performing any live operation."""
    validate_live_restart_smoke_expectations(
        expected_repository_head=expected_repository_head,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
        expected_targets=expected_targets,
    )
    if not isinstance(target_report, dict) or not isinstance(smoke_report, dict):
        raise ValueError("target_report and smoke_report must be JSON objects")

    target = _target_gate(
        target_report,
        expected_targets=expected_targets,
        expected_supervisor_fingerprint=expected_supervisor_fingerprint,
    )
    smoke = _smoke_contract_gate(smoke_report)
    repository = _repository_gate(
        smoke_report, expected_repository_head=expected_repository_head
    )
    event_window = _event_window_gate(smoke_report)
    monitor = _monitor_gate(smoke_report)
    logs = _logs_gate(smoke_report, event_window)
    shutdown = _shutdown_gate(smoke_report, expected_targets=expected_targets)
    startup = _startup_gate(smoke_report, expected_targets=expected_targets)

    issues: list[dict[str, Any]] = []
    if not target["ok"]:
        _issue(issues, "target_contract_invalid")
    if not smoke["tool_schema_ok"]:
        _issue(issues, "smoke_contract_invalid")
    if not smoke["report_ok"]:
        _issue(issues, "smoke_hard_failures")
    if not repository["ok"]:
        _issue(issues, "repository_mismatch")
    if not event_window["ok"]:
        _issue(issues, "event_window_invalid")
    if not monitor["ok"]:
        _issue(issues, "monitor_scan_invalid")
    if not logs["ok"]:
        _issue(issues, "log_scan_invalid")
    if not shutdown["ok"]:
        _issue(issues, "shutdown_evidence_missing")
    if not startup["ok"]:
        _issue(issues, "startup_evidence_missing")

    return {
        "tool": "live-restart-smoke-evidence",
        "schema_version": SCHEMA_VERSION,
        "ok": not issues,
        "hard_failures": len(issues),
        "issues": issues,
        "safety": SAFETY_CONTRACT,
        "gates": {
            "target_contract": target,
            "smoke_contract": smoke,
            "repository": repository,
            "event_window": event_window,
            "monitor": monitor,
            "logs": logs,
            "shutdown": shutdown,
            "startup": startup,
        },
        "evidence": {
            "attention": _attention_evidence(smoke_report, monitor, logs),
        },
    }
