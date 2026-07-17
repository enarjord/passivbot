from __future__ import annotations

import json

import pytest

import live.restart_smoke_collection as collection_module
from live.restart_smoke_collection import build_live_restart_smoke_collection
from tools import live_restart_smoke_collection


HEAD = "a" * 40
FINGERPRINT = "b" * 64
SINCE_MS = 1_752_789_600_123
UNTIL_MS = 1_752_789_660_456


def _target_report(*, targets: int = 2) -> dict:
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": True,
        "hard_failures": 0,
        "issues": [],
        "expected_targets": targets,
        "resolved_targets": targets,
        "relaunch_ready_targets": targets,
        "extra_panes": [],
        "supervisor_contract": {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": FINGERPRINT,
            "target_count": targets,
            "command_content_exposed": False,
        },
        "sampling": {
            "requested_samples": 3,
            "collected_samples": 3,
            "successful_samples": 3,
            "failed_samples": 0,
            "stable": True,
            "supervisor_contract_stable": True,
            "supervisor_contract_changed": False,
            "stable_targets": targets,
            "changed_target_count": 0,
        },
        "targets": [
            {
                "window_name": "private_bot",
                "pane_id": "%999",
                "process_pid": 123456,
                "current_command": "private command",
            }
        ],
    }


def _smoke_report(*, head: str = HEAD, attention: bool = False) -> dict:
    window = {"enabled": True, "since_ms": SINCE_MS, "until_ms": UNTIL_MS}
    return {
        "tool": "live-smoke-report",
        "schema_version": 1,
        "ok": True,
        "attention": attention,
        "hard_failures": 0,
        "attention_count": 1 if attention else 0,
        "repository": {
            "is_git_repo": True,
            "head_full": head,
            "dirty": False,
            "tracked_changes": 0,
            "error": None,
            "root": "/private/repository",
        },
        "event_window": window,
        "monitor": {
            "root": "/private/monitor",
            "files_scanned": 2,
            "error_count": 0,
            "warning_count": 1 if attention else 0,
        },
        "logs": {
            "root": "/private/logs",
            "files_scanned": 2,
            "hard_matches": 0,
            "attention_matches": 1 if attention else 0,
            "dropped_unparsed_hard_matches": 0,
            "window": window,
            "samples": ["secret log line"],
        },
        "shutdown_events": {
            "event_types": {"bot.stopping": 2, "bot.stopped": 2}
        },
        "startup_timings": [
            {"bot": "private/bot-one", "phases": {"startup": {}}},
            {"bot": "private/bot-two", "phases": {"startup": {}}},
        ],
    }


def _kwargs() -> dict:
    return {
        "supervisor_config": "private-supervisor.yaml",
        "session_name": "private-session",
        "monitor_root": "private-monitor",
        "logs_root": "private-logs",
        "expected_repository_head": HEAD,
        "expected_supervisor_fingerprint": FINGERPRINT,
        "expected_targets": 2,
        "since_ms": SINCE_MS,
        "until_ms": UNTIL_MS,
    }


def test_collection_calls_producers_in_order_and_projects_sanitized_verdict(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    def fake_target(*args, **kwargs):
        calls.append(("target", args, kwargs))
        return _target_report()

    def fake_smoke(*args, **kwargs):
        calls.append(("smoke", args, kwargs))
        return _smoke_report()

    monkeypatch.setattr(collection_module, "build_live_restart_target_report", fake_target)
    monkeypatch.setattr(collection_module, "build_live_smoke_report", fake_smoke)

    report = build_live_restart_smoke_collection(**_kwargs())

    assert report["ok"] is True
    assert report["safety"] == {
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
    assert [call[0] for call in calls] == ["target", "smoke"]
    assert calls[0][1] == ("private-supervisor.yaml",)
    assert calls[0][2] == {
        "session_name": "private-session",
        "config_base_dir": collection_module.Path.cwd(),
        "samples": 3,
        "sample_interval_s": 1.0,
    }
    assert calls[1][1] == ("private-monitor",)
    assert calls[1][2] == {
        "logs_root": "private-logs",
        "include_rotated": True,
        "include_processes": False,
        "since_ms": SINCE_MS,
        "until_ms": UNTIL_MS,
        "log_window_unparsed_policy": "drop",
    }
    assert report["collection"] == {
        "target_sampling": {"samples": 3, "sample_interval_s": 1.0},
        "smoke_collection": {
            "include_rotated": True,
            "include_processes": False,
            "log_window_unparsed_policy": "drop",
            "logs_root_source": "caller",
            "event_window": {"since_ms": SINCE_MS, "until_ms": UNTIL_MS},
        },
    }
    serialized = json.dumps(report, sort_keys=True)
    for secret in (
        "private-supervisor.yaml",
        "private-session",
        "private-monitor",
        "private-logs",
        "/private/repository",
        "/private/logs",
        "private command",
        "secret log line",
        FINGERPRINT,
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    "override",
    [
        {"expected_repository_head": "A" * 40},
        {"expected_supervisor_fingerprint": True},
        {"expected_targets": True},
        {"since_ms": True},
        {"until_ms": SINCE_MS},
        {"until_ms": 253_402_300_800_000},
    ],
)
def test_invalid_expectations_or_bounds_fail_before_producers(monkeypatch, override):
    calls: list[str] = []
    kwargs = _kwargs()
    kwargs.update(override)
    monkeypatch.setattr(
        collection_module,
        "build_live_restart_target_report",
        lambda *_args, **_kwargs: calls.append("target"),
    )
    monkeypatch.setattr(
        collection_module,
        "build_live_smoke_report",
        lambda *_args, **_kwargs: calls.append("smoke"),
    )

    with pytest.raises(ValueError):
        build_live_restart_smoke_collection(**kwargs)

    assert calls == []


def test_collection_uses_monitor_default_logs_root_without_exposing_path(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        collection_module,
        "default_logs_root_for_monitor",
        lambda _root: "/private/default-logs",
    )
    monkeypatch.setattr(
        collection_module,
        "build_live_restart_target_report",
        lambda *_args, **_kwargs: _target_report(),
    )

    def fake_smoke(*_args, **kwargs):
        captured.update(kwargs)
        return _smoke_report()

    monkeypatch.setattr(collection_module, "build_live_smoke_report", fake_smoke)
    kwargs = _kwargs()
    kwargs.pop("logs_root")

    report = build_live_restart_smoke_collection(**kwargs)

    assert captured["logs_root"] == "/private/default-logs"
    assert report["collection"]["smoke_collection"]["logs_root_source"] == "monitor_default"
    assert "/private/default-logs" not in json.dumps(report, sort_keys=True)


def test_red_evidence_stays_red_and_attention_stays_non_hard(monkeypatch):
    monkeypatch.setattr(
        collection_module,
        "build_live_restart_target_report",
        lambda *_args, **_kwargs: _target_report(),
    )
    monkeypatch.setattr(
        collection_module,
        "build_live_smoke_report",
        lambda *_args, **_kwargs: _smoke_report(head="c" * 40),
    )

    report = build_live_restart_smoke_collection(**_kwargs())

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["issues"] == [
        {"code": "repository_mismatch", "severity": "error", "count": 1}
    ]

    monkeypatch.setattr(
        collection_module,
        "build_live_smoke_report",
        lambda *_args, **_kwargs: _smoke_report(attention=True),
    )
    attention_report = build_live_restart_smoke_collection(**_kwargs())

    assert attention_report["ok"] is True
    assert attention_report["hard_failures"] == 0
    assert attention_report["evidence"]["attention"]["reported"] is True


def test_cli_compact_output_and_exit_codes(monkeypatch, capsys):
    green = {
        "tool": "live-restart-smoke-collect",
        "schema_version": 1,
        "ok": True,
        "hard_failures": 0,
    }
    monkeypatch.setattr(
        live_restart_smoke_collection,
        "build_live_restart_smoke_collection",
        lambda *_args, **_kwargs: green,
    )
    argv = [
        "supervisor.yaml",
        "monitor",
        "--session-name",
        "passivbot",
        "--expected-repository-head",
        HEAD,
        "--expected-supervisor-fingerprint",
        FINGERPRINT,
        "--expected-targets",
        "2",
        "--since-ms",
        str(SINCE_MS),
        "--until-ms",
        str(UNTIL_MS),
        "--compact",
    ]

    assert live_restart_smoke_collection.main(argv) == 0
    assert capsys.readouterr().out == json.dumps(green, sort_keys=True) + "\n"

    monkeypatch.setattr(
        live_restart_smoke_collection,
        "build_live_restart_smoke_collection",
        lambda *_args, **_kwargs: {**green, "ok": False, "hard_failures": 1},
    )
    assert live_restart_smoke_collection.main(argv) == 1


def test_cli_rejects_invalid_bounds_before_collection(monkeypatch):
    called = False

    def fake_collection(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("collection should not run")

    monkeypatch.setattr(
        live_restart_smoke_collection,
        "build_live_restart_smoke_collection",
        fake_collection,
    )
    with pytest.raises(SystemExit) as exc:
        live_restart_smoke_collection.main(
            [
                "supervisor.yaml",
                "monitor",
                "--session-name",
                "passivbot",
                "--expected-repository-head",
                HEAD,
                "--expected-supervisor-fingerprint",
                FINGERPRINT,
                "--expected-targets",
                "2",
                "--since-ms",
                str(UNTIL_MS),
                "--until-ms",
                str(SINCE_MS),
            ]
        )

    assert exc.value.code == 2
    assert called is False
