from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import live.repository_prepare as prepare_module
from live.repository_prepare import prepare_live_repository
from passivbot_cli import main as cli_main
from tools import live_repository_prepare

RUST_FINGERPRINT = "c" * 64


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(root: Path, message: str, content: str) -> str:
    (root / "tracked.txt").write_text(content, encoding="utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository_pair(tmp_path: Path, monkeypatch) -> dict[str, Path | str]:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    origin.mkdir()
    _git(origin, "init", "--bare", "--initial-branch=master")
    seed.mkdir()
    _git(seed, "init", "--initial-branch=master")
    _git(seed, "config", "user.name", "Repository Prepare Test")
    _git(seed, "config", "user.email", "repository-prepare@example.invalid")
    current = _commit(seed, "initial", "initial\n")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "master")
    _git(tmp_path, "clone", "--branch", "master", str(origin), str(checkout))
    _git(checkout, "config", "user.name", "Repository Prepare Test")
    _git(
        checkout,
        "config",
        "user.email",
        "repository-prepare@example.invalid",
    )
    target = _commit(seed, "target", "target\n")
    _git(seed, "push", "origin", "master")
    monkeypatch.setattr(
        prepare_module, "_canonical_remote_configured", lambda _root: True
    )
    return {
        "origin": origin,
        "seed": seed,
        "checkout": checkout,
        "current": current,
        "target": target,
    }


def _rust_ready(*_args, **_kwargs) -> dict:
    return {
        "ok": True,
        "build_required": False,
        "build_attempted": False,
        "source_fingerprint": RUST_FINGERPRINT,
        "final_source_fingerprint": RUST_FINGERPRINT,
        "compiled_source_stamp": RUST_FINGERPRINT,
        "compiled_sha256": "d" * 64,
        "source_matched": True,
        "source_snapshot_stable": True,
        "reason": None,
    }


def _prepare(repository_pair: dict[str, Path | str], monkeypatch) -> dict:
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(prepare_module, "_prepare_rust_runtime", _rust_ready)
    return prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        build_timeout_s=1.0,
        execute=True,
    )


def test_repository_prepare_fast_forwards_exact_master_and_preserves_untracked(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    artifact = checkout / "local-config.json"
    artifact.write_text("{}\n", encoding="utf-8")

    report = _prepare(repository_pair, monkeypatch)

    assert report["ok"] is True
    assert report["outcome"] == "completed"
    assert report["action_started"] is True
    assert report["repository_updated"] is True
    assert (
        report["repository"]["fetched_target_head"]
        == repository_pair["target"]
    )
    assert report["repository"]["final"]["head"] == repository_pair["target"]
    assert report["rust_extension"]["ok"] is True
    assert _git(checkout, "rev-parse", "HEAD") == repository_pair["target"]
    assert artifact.read_text(encoding="utf-8") == "{}\n"
    assert report["safety"]["fast_forward_only"] is True
    assert report["safety"]["signals_live_processes"] is False
    assert report["safety"]["starts_live_processes"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert str(checkout) not in serialized
    assert str(repository_pair["origin"]) not in serialized


def test_repository_prepare_rejects_tracked_changes_before_fetch(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    (checkout / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(prepare_module, "_prepare_rust_runtime", _rust_ready)

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["ok"] is False
    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "repository_tracked_dirty"
    }
    assert _git(checkout, "rev-parse", "HEAD") == repository_pair["current"]


def test_repository_prepare_rejects_unexpected_fetched_target(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(prepare_module, "_prepare_rust_runtime", _rust_ready)

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head="e" * 40,
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["ok"] is False
    assert report["action_started"] is True
    assert report["repository_updated"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "fetched_target_head_mismatch"
    }
    assert _git(checkout, "rev-parse", "HEAD") == repository_pair["current"]


def test_repository_prepare_rejects_non_fast_forward_target(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    local_head = _commit(checkout, "local divergence", "local\n")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(prepare_module, "_prepare_rust_runtime", _rust_ready)

    report = prepare_live_repository(
        expected_current_head=local_head,
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["ok"] is False
    assert report["repository_updated"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "target_not_fast_forward"
    }
    assert _git(checkout, "rev-parse", "HEAD") == local_head


def test_repository_prepare_reports_rust_failure_after_safe_fast_forward(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(
        prepare_module,
        "_prepare_rust_runtime",
        lambda *_args, **_kwargs: {
            "ok": False,
            "build_required": False,
            "build_attempted": False,
            "source_fingerprint": "e" * 64,
            "reason": "rust_source_fingerprint_mismatch",
        },
    )

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["ok"] is False
    assert report["repository_updated"] is True
    assert report["repository"]["final"]["head"] == repository_pair["target"]
    assert report["rust_extension"]["build_attempted"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "rust_source_fingerprint_mismatch"
    }
    assert _git(checkout, "rev-parse", "HEAD") == repository_pair["target"]


def test_rust_prepare_child_rejects_unexpected_source_before_build():
    report = prepare_module._prepare_rust_runtime(
        Path.cwd(),
        expected_fingerprint="e" * 64,
        timeout_s=30.0,
    )

    assert report["ok"] is False
    assert report["reason"] == "rust_source_fingerprint_mismatch"
    assert report["build_attempted"] is False


def test_repository_prepare_requires_canonical_master(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    _git(checkout, "checkout", "-b", "feature")
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(prepare_module, "_prepare_rust_runtime", _rust_ready)

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "repository_branch_mismatch"
    }


def test_repository_prepare_rejects_in_progress_operation(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    original = prepare_module._git_path_exists
    monkeypatch.setattr(
        prepare_module,
        "_git_path_exists",
        lambda root, name: (
            True if name == "MERGE_HEAD" else original(root, name)
        ),
    )

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "repository_operation_in_progress"
    }


def test_repository_prepare_rejects_noncanonical_origin(
    repository_pair, monkeypatch
):
    checkout = repository_pair["checkout"]
    assert isinstance(checkout, Path)
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(
        prepare_module, "_canonical_remote_configured", lambda _root: False
    )

    report = prepare_live_repository(
        expected_current_head=str(repository_pair["current"]),
        expected_target_head=str(repository_pair["target"]),
        expected_rust_source_fingerprint=RUST_FINGERPRINT,
        execute=True,
    )

    assert report["action_started"] is False
    assert {issue["code"] for issue in report["issues"]} == {
        "repository_remote_mismatch"
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_current_head", "abc", "40 lowercase hex"),
        ("expected_target_head", "A" * 40, "40 lowercase hex"),
        ("expected_rust_source_fingerprint", "abc", "64 lowercase hex"),
    ],
)
def test_repository_prepare_validates_expectations_before_git(
    monkeypatch, field, value, message
):
    monkeypatch.setattr(
        prepare_module,
        "_repository_root",
        lambda: pytest.fail("git must not run for malformed expectations"),
    )
    kwargs = {
        "expected_current_head": "a" * 40,
        "expected_target_head": "b" * 40,
        "expected_rust_source_fingerprint": RUST_FINGERPRINT,
        "execute": True,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        prepare_live_repository(**kwargs)


def test_repository_prepare_cli_requires_execute(capsys):
    with pytest.raises(SystemExit) as exc:
        live_repository_prepare.main(
            [
                "--expected-current-head",
                "a" * 40,
                "--expected-target-head",
                "b" * 40,
                "--expected-rust-source-fingerprint",
                RUST_FINGERPRINT,
            ]
        )
    assert exc.value.code == 2
    assert "--execute is required" in capsys.readouterr().err


def test_repository_prepare_cli_emits_sanitized_report(monkeypatch, capsys):
    monkeypatch.setattr(
        live_repository_prepare,
        "prepare_live_repository",
        lambda **_kwargs: {
            "tool": "live-repository-prepare",
            "schema_version": 1,
            "ok": True,
        },
    )

    result = live_repository_prepare.main(
        [
            "--expected-current-head",
            "a" * 40,
            "--expected-target-head",
            "b" * 40,
            "--expected-rust-source-fingerprint",
            RUST_FINGERPRINT,
            "--execute",
            "--compact",
        ]
    )

    assert result == 0
    assert (
        json.loads(capsys.readouterr().out)["tool"]
        == "live-repository-prepare"
    )


def test_repository_prepare_dispatches_through_unified_cli(monkeypatch):
    captured: dict[str, object] = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = cli_main.sys.argv[:]
        captured["prog_env"] = cli_main.os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(
        cli_main, "_invoke_module_main", fake_invoke_module_main
    )
    monkeypatch.setattr(
        cli_main, "_missing_full_install_markers", lambda: []
    )

    result = cli_main.main(
        [
            "tool",
            "live-repository-prepare",
            "--expected-current-head",
            "a" * 40,
        ]
    )

    assert result == 0
    assert captured["module_name"] == "tools.live_repository_prepare"
    assert captured["argv"] == [
        "passivbot tool live-repository-prepare",
        "--expected-current-head",
        "a" * 40,
    ]
    assert captured["prog_env"] == "passivbot tool live-repository-prepare"
