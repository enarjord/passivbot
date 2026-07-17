from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

import live.restart_smoke_targets as target_module
import live.smoke_report as smoke_report_module
from live.restart_smoke_targets import build_live_restart_target_report
from passivbot_cli import main as cli_main
from tools import live_restart_target_report


def _process_report(*names: str) -> dict:
    expected = [
        {
            "name": name,
            "command": f"passivbot live configs/private.json -u {name}",
            "config_path": "configs/private.json",
            "match_count": 1,
            "matched_processes": [
                {
                    "pid": index + 100,
                    "ppid": 20 + index * 10,
                }
            ],
        }
        for index, name in enumerate(names)
    ]
    return {
        "enabled": True,
        "ok": True,
        "hard_failures": 0,
        "supervisor_contract": smoke_report_module._supervisor_command_contract(
            [
                {
                    "name": row["name"],
                    "_match_key": row["command"],
                    "config_path": row["config_path"],
                }
                for row in expected
            ]
        ),
        "expected_total": len(expected),
        "matched_expected": len(expected),
        "running_live_total": len(expected),
        "expected": expected,
        "missing_expected": [],
        "duplicate_configured_command_matches": [],
        "extra_passivbot_live_processes": [],
        "unexpected_running": [],
        "config_checks": {
            "enabled": True,
            "ok": True,
            "checked": len(expected),
            "skipped": 0,
            "hard_failures": 0,
            "issues": [],
        },
    }


def _pane(session: str, window: str, index: int, pid: int) -> dict:
    return {
        "pane_id": f"%{pid}",
        "session_name": session,
        "window_name": window,
        "pane_index": index,
        "pane_pid": pid,
        "current_command": "python3",
    }


def _target_snapshot(
    *,
    pane_id: str = "%20",
    pane_pid: int = 20,
    process_pid: int = 100,
    relaunch_ready: bool = True,
    supervisor_contract_fingerprint: str = "a" * 64,
    ok: bool = True,
) -> dict:
    return {
        "tool": "live-restart-target-report",
        "schema_version": 1,
        "ok": ok,
        "hard_failures": 0 if ok else 1,
        "supervisor_contract": {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": supervisor_contract_fingerprint,
            "target_count": 1,
            "command_content_exposed": False,
        },
        "targets": [
            {
                "window_name": "binance_01",
                "pane_id": pane_id,
                "pane_pid": pane_pid,
                "process_pid": process_pid,
                "ownership_proof": "matched_process_ppid_equals_pane_pid",
                "relaunch": {
                    "ready": relaunch_ready,
                    "method": (
                        "exact_pane_input_after_verified_exit"
                        if relaunch_ready
                        else None
                    ),
                },
            }
        ],
        "issues": (
            []
            if ok
            else [{"code": "tmux_target_missing", "severity": "error"}]
        ),
    }


def test_live_restart_target_report_resolves_exact_panes_and_ignores_other_sessions(
    monkeypatch,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01", "gateio_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (
            [
                _pane("misc", "bash", 0, 10),
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "gateio_01", 0, 30),
            ],
            None,
        ),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["expected_targets"] == 2
    assert report["resolved_targets"] == 2
    assert report["relaunch_ready_targets"] == 2
    assert report["relaunch_unready_targets"] == 0
    assert report["session_panes"] == 2
    assert report["supervisor_contract"] == {
        "source": "parsed_supervisor_config",
        "algorithm": "sha256",
        "fingerprint": report["supervisor_contract"]["fingerprint"],
        "target_count": 2,
        "command_content_exposed": False,
    }
    assert len(report["supervisor_contract"]["fingerprint"]) == 64
    assert [target["target"] for target in report["targets"]] == [
        "%20",
        "%30",
    ]
    assert [target["pane_pid"] for target in report["targets"]] == [20, 30]
    assert [target["process_pid"] for target in report["targets"]] == [100, 101]
    assert {
        target["ownership_proof"] for target in report["targets"]
    } == {"matched_process_ppid_equals_pane_pid"}
    assert {target["relaunch"]["ready"] for target in report["targets"]} == {
        True
    }
    assert {target["relaunch"]["method"] for target in report["targets"]} == {
        "exact_pane_input_after_verified_exit"
    }
    assert all(
        target["relaunch"]["command_source"] == "supervisor_config"
        and target["relaunch"]["requires_process_exit"] is True
        and target["relaunch"]["requires_post_stop_pane_recheck"] is True
        for target in report["targets"]
    )
    assert report["extra_panes"] == []
    assert report["issues"] == []
    assert "sampling" not in report
    assert report["safety"]["process_control"] is False
    assert report["safety"]["signals_processes"] is False
    assert report["safety"]["starts_processes"] is False
    assert "passivbot live" not in json.dumps(report, sort_keys=True)
    assert "private.json" not in json.dumps(report, sort_keys=True)


def test_supervisor_contract_fingerprint_uses_full_private_commands():
    hidden_suffix = "S" * 600
    rows = [
        {
            "name": "binance_01",
            "_match_key": (
                "passivbot live configs/private.json -u binance_01 "
                f"--api-key FIRST-{hidden_suffix}"
            ),
            "config_path": "configs/private.json",
        },
        {
            "name": "gateio_01",
            "_match_key": "passivbot live configs/private.json -u gateio_01",
            "config_path": "configs/private.json",
        },
    ]
    first = smoke_report_module._supervisor_command_contract(rows)
    reordered = smoke_report_module._supervisor_command_contract(list(reversed(rows)))
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["_match_key"] = changed_rows[0]["_match_key"].replace(
        "FIRST-", "SECOND-"
    )
    changed = smoke_report_module._supervisor_command_contract(changed_rows)

    assert first == reordered
    assert first["fingerprint"] != changed["fingerprint"]
    assert first["command_content_exposed"] is False
    serialized = json.dumps(first, sort_keys=True)
    assert "passivbot live" not in serialized
    assert "private.json" not in serialized
    assert "FIRST" not in serialized
    assert hidden_suffix not in serialized


@pytest.mark.parametrize(
    "contract",
    [
        None,
        {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": "not-a-digest",
            "target_count": 1,
            "command_content_exposed": False,
        },
        {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": "a" * 64,
            "target_count": 2,
            "command_content_exposed": False,
        },
        {
            "source": "parsed_supervisor_config",
            "algorithm": "sha256",
            "fingerprint": "a" * 64,
            "target_count": 1,
            "command_content_exposed": True,
        },
    ],
)
def test_live_restart_target_report_fails_without_valid_full_supervisor_contract(
    monkeypatch,
    contract,
):
    processes = _process_report("binance_01")
    if contract is None:
        processes.pop("supervisor_contract")
    else:
        processes["supervisor_contract"] = contract
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([_pane("passivbot", "binance_01", 0, 20)], None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert {issue["code"] for issue in report["issues"]} == {
        "supervisor_contract_unavailable"
    }


def test_tmux_pane_inventory_only_lists_and_bounds_metadata(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout="%12\tpassivbot\tbinance_01\t0\t856294\tpython3\n",
            stderr="",
        )

    monkeypatch.setattr(target_module.subprocess, "run", fake_run)

    panes, error = target_module._tmux_pane_inventory()

    assert error is None
    assert captured["command"] == [
        "tmux",
        "list-panes",
        "-a",
        "-F",
        target_module.TMUX_PANE_FORMAT,
    ]
    assert captured["kwargs"] == {
        "capture_output": True,
        "text": True,
        "timeout": 5.0,
        "check": False,
    }
    assert panes == [
        {
            "pane_id": "%12",
            "session_name": "passivbot",
            "window_name": "binance_01",
            "pane_index": 0,
            "pane_pid": 856294,
            "current_command": "python3",
        }
    ]


def test_tmux_pane_inventory_rejects_malformed_rows(monkeypatch):
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-a-valid-row\n",
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_malformed_row",
    )


def test_tmux_pane_inventory_rejects_oversized_identifiers(monkeypatch):
    long_name = "x" * (target_module.MAX_TMUX_IDENTIFIER_CHARS + 1)
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"%1\tpassivbot\t{long_name}\t0\t20\tpython3\n",
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_invalid_identifier",
    )


def test_tmux_pane_inventory_rejects_duplicate_pane_ids(monkeypatch):
    monkeypatch.setattr(
        target_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "%1\tpassivbot\tbinance_01\t0\t20\tpython3\n"
                "%1\tpassivbot\tgateio_01\t0\t30\tpython3\n"
            ),
            stderr="",
        ),
    )

    assert target_module._tmux_pane_inventory() == (
        [],
        "tmux_list_panes_duplicate_id",
    )


@pytest.mark.parametrize(
    ("panes", "expected_code"),
    [
        ([], "tmux_target_missing"),
        (
            [
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "binance_01", 1, 21),
            ],
            "tmux_target_duplicated",
        ),
    ],
)
def test_live_restart_target_report_fails_missing_or_duplicate_target(
    monkeypatch,
    panes,
    expected_code,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (panes, None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["resolved_targets"] == 0
    assert {issue["code"] for issue in report["issues"]} == {expected_code}


def test_live_restart_target_report_fails_unconfigured_pane_in_target_session(
    monkeypatch,
):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report("binance_01"),
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: (
            [
                _pane("passivbot", "binance_01", 0, 20),
                _pane("passivbot", "unexpected", 0, 30),
            ],
            None,
        ),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["resolved_targets"] == 1
    assert report["extra_panes"][0]["window_name"] == "unexpected"
    assert report["issues"] == [
        {
            "code": "unconfigured_session_panes",
            "severity": "error",
            "count": 1,
        }
    ]


def test_live_restart_target_report_fails_process_parent_mismatch(monkeypatch):
    processes = _process_report("binance_01")
    processes["expected"][0]["matched_processes"][0]["ppid"] = 999
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([_pane("passivbot", "binance_01", 0, 20)], None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["resolved_targets"] == 0
    assert report["issues"] == [
        {
            "code": "tmux_process_parent_mismatch",
            "severity": "error",
            "window_name": "binance_01",
            "pane_pid": 20,
            "process_pid": 100,
            "process_ppid": 999,
        }
    ]


def test_live_restart_target_report_accepts_direct_pane_process(monkeypatch):
    processes = _process_report("binance_01")
    processes["expected"][0]["matched_processes"][0] = {
        "pid": 20,
        "ppid": 1,
    }
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([_pane("passivbot", "binance_01", 0, 20)], None),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is True
    assert report["targets"][0]["ownership_proof"] == (
        "matched_process_pid_equals_pane_pid"
    )
    assert report["relaunch_ready_targets"] == 0
    assert report["relaunch_unready_targets"] == 1
    assert report["targets"][0]["relaunch"] == {
        "ready": False,
        "method": None,
        "reason": "bot_process_is_direct_tmux_pane_process",
        "command_source": "supervisor_config",
        "requires_process_exit": True,
        "requires_post_stop_pane_recheck": True,
    }


def test_live_restart_target_report_propagates_process_and_tmux_failures(
    monkeypatch,
):
    processes = _process_report("binance_01")
    processes["ok"] = False
    processes["hard_failures"] = 2
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: processes,
    )
    monkeypatch.setattr(
        target_module,
        "_tmux_pane_inventory",
        lambda: ([], "tmux_list_panes_failed"),
    )

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 4
    assert report["processes"]["hard_failures"] == 2
    assert {issue["code"] for issue in report["issues"]} == {
        "tmux_list_panes_failed",
        "tmux_target_missing",
    }


def test_live_restart_target_report_requires_explicit_session(monkeypatch):
    monkeypatch.setattr(
        target_module,
        "build_live_process_report",
        lambda **_kwargs: _process_report(),
    )

    with pytest.raises(ValueError, match="session_name must be non-empty"):
        build_live_restart_target_report("bots.yaml", session_name="")

    with pytest.raises(ValueError, match="session_name must not exceed"):
        build_live_restart_target_report(
            "bots.yaml",
            session_name="x" * (target_module.MAX_TMUX_IDENTIFIER_CHARS + 1),
        )


def test_live_restart_target_report_proves_stable_sampled_identities(monkeypatch):
    snapshots = [_target_snapshot() for _ in range(3)]
    sleeps = []
    monkeypatch.setattr(
        target_module,
        "_build_live_restart_target_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(target_module.time, "sleep", sleeps.append)

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
        samples=3,
        sample_interval_s=0.25,
    )

    assert report["ok"] is True
    assert report["hard_failures"] == 0
    assert report["sampling"] == {
        "requested_samples": 3,
        "collected_samples": 3,
        "interval_s": 0.25,
        "stable": True,
        "successful_samples": 3,
        "failed_samples": 0,
        "failed_sample_issues": [],
        "supervisor_contract_stable": True,
        "supervisor_contract_changed": False,
        "supervisor_contract_observations": [],
        "stable_targets": 1,
        "changed_target_count": 0,
        "changed_targets_truncated": 0,
        "changed_targets": [],
    }
    assert sleeps == [0.25, 0.25]


def test_live_restart_target_report_fails_changed_sampled_identity(monkeypatch):
    snapshots = [
        _target_snapshot(),
        _target_snapshot(pane_id="%21", pane_pid=21),
    ]
    monkeypatch.setattr(
        target_module,
        "_build_live_restart_target_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(target_module.time, "sleep", lambda _seconds: None)

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
        samples=2,
        sample_interval_s=0.0,
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["sampling"]["stable"] is False
    assert report["sampling"]["failed_samples"] == 0
    assert report["sampling"]["changed_target_count"] == 1
    assert report["sampling"]["changed_targets_truncated"] == 0
    assert report["sampling"]["changed_targets"][0] == {
        "window_name": "binance_01",
        "observations": [
            {
                "sample": 1,
                "identity": {
                    "pane_id": "%20",
                    "pane_pid": 20,
                    "process_pid": 100,
                    "ownership_proof": "matched_process_ppid_equals_pane_pid",
                    "relaunch_ready": True,
                    "relaunch_method": "exact_pane_input_after_verified_exit",
                },
            },
            {
                "sample": 2,
                "identity": {
                    "pane_id": "%21",
                    "pane_pid": 21,
                    "process_pid": 100,
                    "ownership_proof": "matched_process_ppid_equals_pane_pid",
                    "relaunch_ready": True,
                    "relaunch_method": "exact_pane_input_after_verified_exit",
                },
            },
        ],
    }
    assert report["issues"][-1]["code"] == "target_sampling_unstable"


def test_live_restart_target_report_fails_changed_supervisor_contract(monkeypatch):
    snapshots = [
        _target_snapshot(supervisor_contract_fingerprint="a" * 64),
        _target_snapshot(supervisor_contract_fingerprint="b" * 64),
    ]
    monkeypatch.setattr(
        target_module,
        "_build_live_restart_target_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(target_module.time, "sleep", lambda _seconds: None)

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
        samples=2,
        sample_interval_s=0.0,
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["sampling"]["stable"] is False
    assert report["sampling"]["changed_target_count"] == 0
    assert report["sampling"]["supervisor_contract_stable"] is False
    assert report["sampling"]["supervisor_contract_changed"] is True
    assert report["sampling"]["supervisor_contract_observations"] == [
        {"sample": 1, "fingerprint": "a" * 64},
        {"sample": 2, "fingerprint": "b" * 64},
    ]
    assert report["issues"][-1] == {
        "code": "target_sampling_unstable",
        "severity": "error",
        "failed_samples": 0,
        "changed_target_count": 0,
        "supervisor_contract_changed": True,
    }


def test_live_restart_target_report_fails_changed_relaunch_proof(monkeypatch):
    snapshots = [
        _target_snapshot(),
        _target_snapshot(relaunch_ready=False),
    ]
    monkeypatch.setattr(
        target_module,
        "_build_live_restart_target_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(target_module.time, "sleep", lambda _seconds: None)

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
        samples=2,
        sample_interval_s=0.0,
    )

    assert report["ok"] is False
    assert report["sampling"]["stable"] is False
    assert report["sampling"]["changed_target_count"] == 1
    observations = report["sampling"]["changed_targets"][0]["observations"]
    assert observations[0]["identity"]["relaunch_ready"] is True
    assert observations[1]["identity"]["relaunch_ready"] is False
    assert report["issues"][-1]["code"] == "target_sampling_unstable"


def test_live_restart_target_report_fails_when_any_sample_is_hard_red(
    monkeypatch,
):
    snapshots = [_target_snapshot(ok=False), _target_snapshot()]
    monkeypatch.setattr(
        target_module,
        "_build_live_restart_target_snapshot",
        lambda *_args, **_kwargs: snapshots.pop(0),
    )
    monkeypatch.setattr(target_module.time, "sleep", lambda _seconds: None)

    report = build_live_restart_target_report(
        "bots.yaml",
        session_name="passivbot",
        samples=2,
        sample_interval_s=0.0,
    )

    assert report["ok"] is False
    assert report["hard_failures"] == 1
    assert report["sampling"]["failed_samples"] == 1
    assert report["sampling"]["failed_sample_issues"] == [
        {
            "sample": 1,
            "hard_failures": 1,
            "issue_codes": ["tmux_target_missing"],
        }
    ]
    assert report["issues"][-1] == {
        "code": "target_sampling_unstable",
        "severity": "error",
        "failed_samples": 1,
        "changed_target_count": 0,
        "supervisor_contract_changed": False,
    }


def test_target_sampling_bounds_changed_target_rows():
    def targets(prefix):
        return [
            {
                "window_name": f"{prefix}_{index}",
                "pane_id": f"%{index}",
                "pane_pid": index + 100,
                "process_pid": index + 200,
                "ownership_proof": "matched_process_ppid_equals_pane_pid",
            }
            for index in range(target_module.MAX_RESTART_TARGETS)
        ]

    sampling = target_module._summarize_target_sampling(
        [
            {"ok": True, "targets": targets("first")},
            {"ok": True, "targets": targets("second")},
        ],
        interval_s=0.0,
    )

    assert sampling["stable"] is False
    assert sampling["changed_target_count"] == target_module.MAX_RESTART_TARGETS * 2
    assert len(sampling["changed_targets"]) == target_module.MAX_RESTART_TARGETS
    assert sampling["changed_targets_truncated"] == target_module.MAX_RESTART_TARGETS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 0}, "samples must be between"),
        ({"samples": target_module.MAX_TARGET_SAMPLES + 1}, "samples must be between"),
        ({"sample_interval_s": -1.0}, "sample_interval_s must be between"),
        (
            {"sample_interval_s": target_module.MAX_TARGET_SAMPLE_INTERVAL_S + 1.0},
            "sample_interval_s must be between",
        ),
        ({"sample_interval_s": float("nan")}, "sample_interval_s must be between"),
    ],
)
def test_live_restart_target_report_rejects_unbounded_sampling(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_live_restart_target_report(
            "bots.yaml",
            session_name="passivbot",
            **kwargs,
        )


def test_live_restart_target_report_cli_preserves_verdict(monkeypatch, capsys):
    captured = {}

    def fake_report(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"ok": False, "hard_failures": 3, "targets": []}

    monkeypatch.setattr(
        live_restart_target_report,
        "build_live_restart_target_report",
        fake_report,
    )

    assert (
        live_restart_target_report.main(
            ["bots.yaml", "--session-name", "passivbot", "--compact"]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["hard_failures"] == 3
    assert captured["args"] == ("bots.yaml",)
    assert captured["kwargs"]["session_name"] == "passivbot"
    assert captured["kwargs"]["samples"] == 1
    assert captured["kwargs"]["sample_interval_s"] == 1.0


def test_live_restart_target_report_cli_forwards_sampling(monkeypatch, capsys):
    captured = {}

    def fake_report(*_args, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "hard_failures": 0, "targets": []}

    monkeypatch.setattr(
        live_restart_target_report,
        "build_live_restart_target_report",
        fake_report,
    )

    assert (
        live_restart_target_report.main(
            [
                "bots.yaml",
                "--session-name",
                "passivbot",
                "--samples",
                "3",
                "--interval-s",
                "2.5",
                "--compact",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert captured["samples"] == 3
    assert captured["sample_interval_s"] == 2.5


def test_live_restart_target_report_tool_dispatch_forwards_module(monkeypatch):
    captured = {}

    def fake_invoke_module_main(module_name):
        captured["module_name"] = module_name
        captured["argv"] = sys.argv[:]
        captured["prog_env"] = os.environ.get("PASSIVBOT_CLI_PROG")
        return True, 0

    monkeypatch.setattr(cli_main, "_invoke_module_main", fake_invoke_module_main)
    monkeypatch.setattr(cli_main, "_missing_full_install_markers", lambda: [])

    assert (
        cli_main.main(
            [
                "tool",
                "live-restart-target-report",
                "bots.yaml",
                "--session-name",
                "passivbot",
            ]
        )
        == 0
    )
    assert captured == {
        "module_name": "tools.live_restart_target_report",
        "argv": [
            "passivbot tool live-restart-target-report",
            "bots.yaml",
            "--session-name",
            "passivbot",
        ],
        "prog_env": "passivbot tool live-restart-target-report",
    }
