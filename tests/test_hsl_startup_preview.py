from __future__ import annotations

import json
import os
import sys

import pytest

from passivbot_cli import main as cli_main
from tools import hsl_startup_preview


def _write_config(path, config):
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")


def _sample_config() -> dict:
    return {
        "config_version": "v8.0.0",
        "live": {
            "user": "binance_01",
            "exchange": "binance",
            "hsl_signal_mode": "coin",
            "hsl_position_during_cooldown_policy": "panic",
            "api_key": "super-secret-api-key",
        },
        "bot": {
            "long": {
                "hsl": {
                    "enabled": True,
                    "red_threshold": 0.10,
                    "cooldown_minutes_after_red": 45,
                    "no_restart_drawdown_threshold": 0.20,
                    "ema_span_minutes": 120,
                    "tier_ratios": {"yellow": 0.5, "orange": 0.8},
                    "orange_tier_mode": "tp_only",
                    "panic_close_order_type": "limit",
                }
            },
            "short": {
                "hsl": {
                    "enabled": False,
                    "red_threshold": 0.12,
                    "cooldown_minutes_after_red": 30,
                }
            },
        },
    }


def _monitor_row(
    *,
    event_type: str,
    seq: int,
    ts: int,
    reason_code: str,
    symbol: str | None,
    pside: str,
    data: dict,
) -> dict:
    live_event = {
        "schema_version": 1,
        "event_id": f"evt_{seq}",
        "event_type": event_type,
        "level": "info",
        "source": "live",
        "component": "risk",
        "exchange": "binance",
        "user": "binance_01",
        "symbol": symbol,
        "pside": pside,
        "status": "succeeded",
        "reason_code": reason_code,
        "data": data,
        "ids": {"cycle_id": "cy_1"},
    }
    return {
        "exchange": "binance",
        "user": "binance_01",
        "kind": event_type,
        "tags": ["hsl", "risk"],
        "payload": {"_live_event": live_event},
        "seq": seq,
        "ts": ts,
    }


def _write_ndjson(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_hsl_startup_preview_reports_config_and_latest_local_hsl_status(tmp_path):
    config_path = tmp_path / "live.json"
    monitor_root = tmp_path / "monitor"
    _write_config(config_path, _sample_config())
    _write_ndjson(
        monitor_root / "binance" / "binance_01" / "events" / "current.ndjson",
        [
            _monitor_row(
                event_type="hsl.status",
                seq=1,
                ts=1_000,
                reason_code="yellow",
                symbol="SOL/USDT:USDT",
                pside="long",
                data={
                    "signal_mode": "coin",
                    "tier": "yellow",
                    "drawdown_raw": 0.03,
                    "drawdown_ema": 0.04,
                    "drawdown_score": 0.04,
                    "dist_to_red": 0.06,
                    "red_threshold": 0.10,
                    "slot_budget": 25.0,
                    "realized_pnl": -1.0,
                    "peak_realized_pnl": 2.0,
                    "unrealized_pnl": -0.5,
                    "secret": "must-not-render",
                },
            ),
            _monitor_row(
                event_type="hsl.status",
                seq=2,
                ts=2_000,
                reason_code="red",
                symbol="SOL/USDT:USDT",
                pside="long",
                data={
                    "signal_mode": "coin",
                    "tier": "red",
                    "drawdown_raw": 0.11,
                    "drawdown_ema": 0.09,
                    "drawdown_score": 0.09,
                    "dist_to_red": 0.01,
                    "red_threshold": 0.10,
                    "cooldown_until_ms": 122_000,
                    "cooldown_remaining_seconds": 120.0,
                },
            ),
        ],
    )

    report = hsl_startup_preview.build_hsl_startup_preview_report(
        config_path,
        monitor_root=monitor_root,
        now_ms=62_000,
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["ok"] is True
    assert report["config"]["hsl"]["signal_mode"] == "coin"
    assert report["config"]["hsl"]["sides"]["long"]["enabled"] is True
    assert report["inputs"]["monitor_events"]["hsl_events_seen"] == 2
    assert report["hsl_status"]["counts_by_status"] == {"red": 1}
    latest = report["hsl_status"]["latest_by_target"][0]
    assert latest["symbol"] == "SOL/USDT:USDT"
    assert latest["status"] == "red"
    assert latest["drawdown_to_red"]["value"] == 0.01
    assert latest["current_drawdown"]["available"] is False
    assert latest["cooldown"]["remaining_seconds_at_preview"] == pytest.approx(60.0)
    assert report["startup_panic_orders"]["available"] is False
    assert "super-secret-api-key" not in rendered
    assert "must-not-render" not in rendered


def test_hsl_startup_preview_config_only_marks_runtime_inputs_unavailable(tmp_path):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    report = hsl_startup_preview.build_hsl_startup_preview_report(
        config_path,
        monitor_root="",
        now_ms=62_000,
    )

    assert report["ok"] is True
    assert report["inputs"]["monitor_events"]["available"] is False
    assert report["hsl_status"]["available"] is False
    assert report["inputs"]["current_drawdown"]["available"] is False
    assert report["inputs"]["startup_panic_order_prediction"]["available"] is False
    assert report["summary"]["hsl_targets_with_local_status"] == 0


def test_hsl_startup_preview_invalid_json_returns_nonzero(tmp_path, capsys):
    config_path = tmp_path / "bad.json"
    config_path.write_text("{bad-json", encoding="utf-8")

    assert hsl_startup_preview.main([str(config_path), "--compact"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["ok"] is False
    assert report["issues"][0]["code"] == "config_json_decode_failed"


def test_hsl_startup_preview_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc_info:
        hsl_startup_preview.main(["--help"])

    assert exc_info.value.code == 0
    assert "hsl-startup-preview" in capsys.readouterr().out


def test_hsl_startup_preview_tool_dispatch_forwards_module_and_prog(monkeypatch):
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
            ["tool", "hsl-startup-preview", "configs/live.json", "--compact"]
        )
        == 0
    )

    assert captured["module_name"] == "tools.hsl_startup_preview"
    assert captured["argv"] == [
        "passivbot tool hsl-startup-preview",
        "configs/live.json",
        "--compact",
    ]
    assert captured["prog_env"] == "passivbot tool hsl-startup-preview"
