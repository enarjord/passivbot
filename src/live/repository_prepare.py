from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

CANONICAL_BRANCH = "master"
CANONICAL_REMOTE = "origin"
CANONICAL_REMOTE_REF = "refs/remotes/origin/master"
CANONICAL_REMOTE_URLS = {
    "https://github.com/enarjord/passivbot",
    "https://github.com/enarjord/passivbot.git",
}
DEFAULT_BUILD_TIMEOUT_S = 1800.0
MAX_BUILD_TIMEOUT_S = 3600.0
RUST_RESULT_PREFIX = "PASSIVBOT_REPOSITORY_PREPARE_RESULT="

SAFETY_CONTRACT = {
    "local_only": True,
    "ssh": False,
    "network": True,
    "git_network_access": True,
    "git_remote": CANONICAL_REMOTE,
    "git_branch": CANONICAL_BRANCH,
    "fast_forward_only": True,
    "direct_exchange_access": False,
    "direct_credential_store_access": False,
    "git_public_remote_only": True,
    "signals_live_processes": False,
    "starts_live_processes": False,
    "subprocess_execution": True,
    "reads_supervisor_config": False,
    "writes_repository_checkout": True,
    "writes_rust_build_artifacts": True,
    "preserves_untracked_files": True,
    "automatic_force_checkout": False,
    "automatic_force_process_signal": False,
    "build_timeout_process_group_only": True,
    "signals_build_process_group_on_timeout": True,
    "requires_execute_confirmation": True,
    "requires_expected_current_head": True,
    "requires_expected_target_head": True,
    "requires_expected_rust_source_fingerprint": True,
    "requires_canonical_remote_url": True,
    "requires_tracked_clean_repository": True,
}


def _valid_head(value: str, *, field: str) -> str:
    head = str(value or "").strip()
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise ValueError(f"{field} must be 40 lowercase hex characters")
    return head


def _valid_fingerprint(value: str) -> str:
    fingerprint = str(value or "").strip()
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(
            "expected_rust_source_fingerprint must be 64 lowercase hex "
            "characters"
        )
    return fingerprint


def _bounded_build_timeout(value: float) -> float:
    timeout = float(value)
    if (
        not math.isfinite(timeout)
        or timeout <= 0.0
        or timeout > MAX_BUILD_TIMEOUT_S
    ):
        raise ValueError(
            "build_timeout_s must be greater than 0 and at most "
            f"{MAX_BUILD_TIMEOUT_S:g}"
        )
    return timeout


def _run_git(
    root: Path,
    arguments: list[str],
    *,
    timeout_s: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _git_stdout(
    root: Path, arguments: list[str]
) -> tuple[str | None, str | None]:
    try:
        result = _run_git(root, arguments)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git_failed:{exc.__class__.__name__}"
    if result.returncode != 0:
        return None, "git_failed"
    return result.stdout.strip(), None


def _repository_root() -> Path | None:
    output, error = _git_stdout(Path.cwd(), ["rev-parse", "--show-toplevel"])
    if error is not None or not output:
        return None
    return Path(output).resolve()


def _git_path_exists(root: Path, name: str) -> bool:
    output, error = _git_stdout(root, ["rev-parse", "--git-path", name])
    if error is not None or not output:
        return True
    path = Path(output)
    if not path.is_absolute():
        path = root / path
    return path.exists()


def _repository_snapshot(root: Path) -> dict[str, Any]:
    branch, branch_error = _git_stdout(
        root, ["symbolic-ref", "--quiet", "--short", "HEAD"]
    )
    head, head_error = _git_stdout(root, ["rev-parse", "HEAD"])
    status, status_error = _git_stdout(
        root, ["status", "--porcelain", "--untracked-files=no"]
    )
    operation_names = (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "BISECT_LOG",
        "rebase-apply",
        "rebase-merge",
    )
    operations = [
        name for name in operation_names if _git_path_exists(root, name)
    ]
    changes = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "branch": branch,
        "head": head,
        "tracked_clean": status_error is None and not changes,
        "tracked_changes": len(changes) if status_error is None else None,
        "operation_in_progress": bool(operations),
        "available": not any((branch_error, head_error, status_error)),
    }


def _canonical_remote_configured(root: Path) -> bool:
    remote_url, error = _git_stdout(
        root, ["remote", "get-url", CANONICAL_REMOTE]
    )
    return error is None and remote_url in CANONICAL_REMOTE_URLS


def _snapshot_issue_codes(
    snapshot: dict[str, Any], *, expected_head: str
) -> list[str]:
    issues: list[str] = []
    if not snapshot.get("available"):
        issues.append("repository_snapshot_unavailable")
        return issues
    if snapshot.get("branch") != CANONICAL_BRANCH:
        issues.append("repository_branch_mismatch")
    if snapshot.get("head") != expected_head:
        issues.append("repository_head_mismatch")
    if snapshot.get("tracked_clean") is not True:
        issues.append("repository_tracked_dirty")
    if snapshot.get("operation_in_progress") is not False:
        issues.append("repository_operation_in_progress")
    return issues


def _sanitized_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "branch": snapshot.get("branch"),
        "head": snapshot.get("head"),
        "tracked_clean": snapshot.get("tracked_clean"),
        "tracked_changes": snapshot.get("tracked_changes"),
        "operation_in_progress": snapshot.get("operation_in_progress"),
    }


RUST_PREPARE_SCRIPT = r"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from pathlib import Path

from rust_utils import (
    check_and_maybe_compile,
    extension_needs_rebuild,
    latest_source_mtime,
    preferred_compiled_path,
    source_fingerprint,
    verify_loaded_runtime_extension,
)

PREFIX = "PASSIVBOT_REPOSITORY_PREPARE_RESULT="
expected = sys.argv[1]
result = {
    "ok": False,
    "build_required": None,
    "build_attempted": False,
    "source_fingerprint": None,
    "final_source_fingerprint": None,
    "compiled_source_stamp": None,
    "compiled_sha256": None,
    "source_matched": False,
    "source_snapshot_stable": False,
    "reason": None,
}
try:
    before = source_fingerprint(Path("passivbot-rust"))
    result["source_fingerprint"] = before
    if before != expected:
        result["reason"] = "rust_source_fingerprint_mismatch"
    else:
        build_required = extension_needs_rebuild(
            preferred_compiled_path(), latest_source_mtime(), before
        )
        result["build_required"] = bool(build_required)
        result["build_attempted"] = bool(build_required)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(
                sink
            ):
                check_and_maybe_compile()
                importlib.import_module("passivbot_rust")
                runtime = verify_loaded_runtime_extension(fingerprint=expected)
        after = source_fingerprint(Path("passivbot-rust"))
        stamp = runtime.get("runtime_compiled_source_stamp")
        result.update(
            {
                "final_source_fingerprint": after,
                "compiled_source_stamp": stamp,
                "compiled_sha256": runtime.get("runtime_compiled_sha256"),
                "source_matched": stamp == expected,
                "source_snapshot_stable": before == after == expected,
            }
        )
        result["ok"] = bool(
            result["source_matched"] and result["source_snapshot_stable"]
        )
        if not result["ok"]:
            result["reason"] = "rust_extension_source_mismatch"
except Exception as exc:
    result["reason"] = "rust_prepare_failed"
    result["error_class"] = exc.__class__.__name__
print(PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if result["ok"] else 1)
"""


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5.0)
    except (OSError, subprocess.SubprocessError):
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=5.0)
        except (OSError, subprocess.SubprocessError):
            pass


def _prepare_rust_runtime(
    root: Path,
    *,
    expected_fingerprint: str,
    timeout_s: float,
) -> dict[str, Any]:
    env = os.environ.copy()
    src_path = str(root / "src")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else src_path
    )
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", RUST_PREPARE_SCRIPT, expected_fingerprint],
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        return {
            "ok": False,
            "reason": "rust_prepare_failed",
            "error_class": exc.__class__.__name__,
        }
    try:
        stdout, _ = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return {"ok": False, "reason": "rust_prepare_timeout"}

    payload_line = next(
        (
            line
            for line in reversed(stdout.splitlines())
            if line.startswith(RUST_RESULT_PREFIX)
        ),
        None,
    )
    if payload_line is None:
        return {"ok": False, "reason": "rust_prepare_result_missing"}
    try:
        payload = json.loads(payload_line.removeprefix(RUST_RESULT_PREFIX))
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "reason": "rust_prepare_result_invalid"}
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "rust_prepare_result_invalid"}
    if process.returncode != 0:
        payload["ok"] = False
    return payload


def _issue(code: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "severity": "error", **fields}


def prepare_live_repository(
    *,
    expected_current_head: str,
    expected_target_head: str,
    expected_rust_source_fingerprint: str,
    build_timeout_s: float = DEFAULT_BUILD_TIMEOUT_S,
    execute: bool = False,
) -> dict[str, Any]:
    """Fast-forward master and prove its local Rust runtime contract."""
    current_head = _valid_head(
        expected_current_head, field="expected_current_head"
    )
    target_head = _valid_head(
        expected_target_head, field="expected_target_head"
    )
    rust_fingerprint = _valid_fingerprint(expected_rust_source_fingerprint)
    timeout = _bounded_build_timeout(build_timeout_s)
    if not execute:
        raise ValueError("execute must be true")

    report: dict[str, Any] = {
        "tool": "live-repository-prepare",
        "schema_version": 1,
        "ok": False,
        "outcome": "rejected",
        "action_started": False,
        "repository_updated": False,
        "repository": {
            "expected_current_head": current_head,
            "expected_target_head": target_head,
            "fetched_target_head": None,
            "initial": None,
            "final": None,
        },
        "rust_extension": {
            "expected_source_fingerprint": rust_fingerprint,
            "ok": False,
            "build_required": None,
            "build_attempted": False,
        },
        "issues": [],
        "safety": dict(SAFETY_CONTRACT),
    }
    issues: list[dict[str, Any]] = report["issues"]
    root = _repository_root()
    if root is None:
        issues.append(_issue("repository_unavailable"))
        return report

    initial = _repository_snapshot(root)
    report["repository"]["initial"] = _sanitized_snapshot(initial)
    initial_issues = _snapshot_issue_codes(initial, expected_head=current_head)
    if not _canonical_remote_configured(root):
        initial_issues.append("repository_remote_mismatch")
    if initial_issues:
        issues.extend(_issue(code) for code in initial_issues)
        return report

    report["action_started"] = True
    report["outcome"] = "failed"
    try:
        fetch = _run_git(
            root,
            [
                "fetch",
                "--no-tags",
                CANONICAL_REMOTE,
                f"refs/heads/{CANONICAL_BRANCH}:{CANONICAL_REMOTE_REF}",
            ],
        )
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append(
            _issue("git_fetch_failed", error_class=exc.__class__.__name__)
        )
        return report
    if fetch.returncode != 0:
        issues.append(_issue("git_fetch_failed"))
        return report

    after_fetch = _repository_snapshot(root)
    after_fetch_issues = _snapshot_issue_codes(
        after_fetch, expected_head=current_head
    )
    if after_fetch_issues:
        issues.append(_issue("repository_changed_during_fetch"))
        return report

    fetched_head, fetched_error = _git_stdout(
        root, ["rev-parse", CANONICAL_REMOTE_REF]
    )
    report["repository"]["fetched_target_head"] = fetched_head
    if fetched_error is not None or fetched_head is None:
        issues.append(_issue("fetched_target_head_unavailable"))
        return report
    if fetched_head != target_head:
        issues.append(_issue("fetched_target_head_mismatch"))
        return report

    try:
        ancestry = _run_git(
            root, ["merge-base", "--is-ancestor", current_head, target_head]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        issues.append(
            _issue(
                "git_ancestry_check_failed",
                error_class=exc.__class__.__name__,
            )
        )
        return report
    if ancestry.returncode == 1:
        issues.append(_issue("target_not_fast_forward"))
        return report
    if ancestry.returncode != 0:
        issues.append(_issue("git_ancestry_check_failed"))
        return report

    if current_head != target_head:
        try:
            merge = _run_git(
                root,
                [
                    "-c",
                    "core.hooksPath=/dev/null",
                    "merge",
                    "--ff-only",
                    "--no-edit",
                    CANONICAL_REMOTE_REF,
                ],
            )
        except (OSError, subprocess.SubprocessError) as exc:
            issues.append(
                _issue(
                    "git_fast_forward_failed",
                    error_class=exc.__class__.__name__,
                )
            )
            return report
        if merge.returncode != 0:
            issues.append(_issue("git_fast_forward_failed"))
            return report
        report["repository_updated"] = True

    prepared_snapshot = _repository_snapshot(root)
    report["repository"]["final"] = _sanitized_snapshot(prepared_snapshot)
    prepared_issues = _snapshot_issue_codes(
        prepared_snapshot, expected_head=target_head
    )
    if prepared_issues:
        issues.append(_issue("repository_post_update_mismatch"))
        return report

    rust_report = _prepare_rust_runtime(
        root,
        expected_fingerprint=rust_fingerprint,
        timeout_s=timeout,
    )
    allowed_rust_fields = {
        "ok",
        "build_required",
        "build_attempted",
        "source_fingerprint",
        "final_source_fingerprint",
        "compiled_source_stamp",
        "compiled_sha256",
        "source_matched",
        "source_snapshot_stable",
        "reason",
        "error_class",
    }
    report["rust_extension"] = {
        "expected_source_fingerprint": rust_fingerprint,
        **{
            key: rust_report.get(key)
            for key in allowed_rust_fields
            if key in rust_report
        },
    }
    if rust_report.get("ok") is not True:
        reason = str(rust_report.get("reason") or "rust_prepare_failed")
        if reason not in {
            "rust_extension_source_mismatch",
            "rust_prepare_failed",
            "rust_prepare_result_invalid",
            "rust_prepare_result_missing",
            "rust_prepare_timeout",
            "rust_source_fingerprint_mismatch",
        }:
            reason = "rust_prepare_failed"
        issues.append(_issue(reason))
        return report

    final = _repository_snapshot(root)
    report["repository"]["final"] = _sanitized_snapshot(final)
    final_issues = _snapshot_issue_codes(final, expected_head=target_head)
    if final_issues:
        issues.append(_issue("repository_changed_during_rust_prepare"))
        return report

    report["ok"] = True
    report["outcome"] = "completed"
    return report
