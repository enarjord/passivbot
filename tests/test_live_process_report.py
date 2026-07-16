from __future__ import annotations

import json

import pytest

import live.smoke_report as smoke_report_module
from live.smoke_report import build_live_process_report
from tools import live_process_report


def test_build_live_process_report_samples_process_table_without_smoke_inputs(
    monkeypatch,
):
    scans = iter(
        [
            (["123 1 10 D 1.0 2.0 100 passivbot live configs/a.json -u bot_a"], None),
            (["123 1 11 R 1.0 2.0 100 passivbot live configs/a.json -u bot_a"], None),
        ]
    )
    sleeps = []
    monkeypatch.setattr(smoke_report_module, "_ps_process_rows", lambda: next(scans))
    monkeypatch.setattr(smoke_report_module.time, "sleep", sleeps.append)

    report = build_live_process_report(
        process_samples=2,
        process_sample_interval_s=3.0,
    )

    assert report["ok"] is True
    assert report["running_live_total"] == 1
    assert report["running"][0]["stat"] == "R"
    assert report["sampling"]["requested_samples"] == 2
    assert report["sampling"]["uninterruptible_recovered_count"] == 1
    assert sleeps == [3.0]


def test_live_process_report_cli_declares_and_enforces_local_only_boundary(
    monkeypatch,
    capsys,
):
    captured = {}

    def fake_build_live_process_report(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "hard_failures": 0, "enabled": True}

    monkeypatch.setattr(
        live_process_report,
        "build_live_process_report",
        fake_build_live_process_report,
    )

    assert not hasattr(live_process_report, "build_live_smoke_report")
    assert not hasattr(live_process_report, "default_logs_root_for_monitor")
    assert (
        live_process_report.main(
            ["--samples", "4", "--interval-s", "5", "--compact"]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["tool"] == "live-process-report"
    assert report["safety"] == {
        "local_only": True,
        "reads": [
            "process_table",
            "optional_supervisor_config",
            "optional_referenced_bot_configs",
        ],
        "monitor_events": False,
        "text_logs": False,
        "network": False,
        "exchange_access": False,
        "credential_store_access": False,
        "process_control": False,
        "writes_files": False,
    }
    assert captured["process_samples"] == 4
    assert captured["process_sample_interval_s"] == 5.0
    assert "monitor_root" not in captured
    assert "logs_root" not in captured


def test_live_process_report_cli_returns_process_verdict(monkeypatch, capsys):
    monkeypatch.setattr(
        live_process_report,
        "build_live_process_report",
        lambda **_kwargs: {"ok": False, "hard_failures": 2, "enabled": True},
    )

    assert live_process_report.main(["--compact"]) == 1
    assert json.loads(capsys.readouterr().out)["hard_failures"] == 2


def test_live_process_report_cli_rejects_unbounded_sampling(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_process_report.main(
            ["--samples", str(smoke_report_module.MAX_PROCESS_SAMPLES + 1)]
        )

    assert exc_info.value.code == 2
    assert "process_samples must be between" in capsys.readouterr().err
