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
            "api_key": "super-secret-api-key",
        },
        "bot": {
            "long": {
                "hsl": {
                    "enabled": True,
                    "red_threshold": 0.10,
                    "cooldown_minutes_after_red": 45,
                    "ema_span_minutes": 120,
                    "panic_close_order_type": "limit",
                    "restart_after_red_policy": "always",
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


@pytest.mark.parametrize(
    "now,expected",
    [(3000, "red"), (4000, "stale_or_unavailable"), (62000, "stale_or_unavailable")],
)
def test_hsl_startup_preview_reports_current_scopes_and_marks_expired_observations(
    tmp_path, now, expected
):
    config_path = tmp_path / "live.json"
    monitor_root = tmp_path / "monitor"
    _write_config(config_path, _sample_config())
    rows = [
        _monitor_row(
            event_type="hsl.status",
            reason_code=None,
            symbol=None,
            pside=None,
            seq=1,
            ts=1000,
            data={
                "signal_mode": "coin",
                "tier": "green",
                "observation_status": "current",
                "captured_at_ms": 1000,
                "input_expires_at_ms": 2000,
            },
        ),
        _monitor_row(
            event_type="hsl.status",
            reason_code=None,
            symbol=None,
            pside=None,
            seq=2,
            ts=2000,
            data={
                "signal_mode": "coin",
                "tier": "red",
                "observation_status": "current",
                "captured_at_ms": 2000,
                "input_expires_at_ms": 4000,
                "scope_count": 1,
                "counts": {"red": 1, "green": 0, "secret": "must-not-render"},
                "scopes": [
                    {
                        "symbol": "SOL/USDT:USDT",
                        "pside": "long",
                        "tier": "red",
                        "action": "panic",
                        "raw": 0.12,
                        "ema": 0.11,
                        "score": 0.11,
                        "threshold": 0.1,
                        "secret": "nested-must-not-render",
                    }
                ],
                "secret": "must-not-render",
            },
        ),
    ]
    _write_ndjson(
        monitor_root / "binance" / "binance_01" / "events" / "current.ndjson", rows
    )
    report = hsl_startup_preview.build_hsl_startup_preview_report(
        config_path, monitor_root=monitor_root, now_ms=now
    )
    assert report["ok"] is True
    assert report["config"]["hsl"]["sides"]["long"]["enabled"] is True
    assert report["inputs"]["monitor_events"]["hsl_events_seen"] == 2
    assert report["hsl_status"]["counts_by_status"] == {expected: 1}
    latest = report["hsl_status"]["latest_by_target"][0]
    assert latest["last_observed_tier"] == "red"
    assert latest["latest_data"]["scopes"][0]["score"] == 0.11
    assert latest["current_drawdown"]["available"] is False
    assert latest["cooldown"]["available"] is False
    assert report["startup_panic_orders"]["available"] is False
    assert "must-not-render" not in json.dumps(report)
    assert "super-secret-api-key" not in json.dumps(report)


def test_hsl_preview_degraded_capture_never_becomes_current_by_timestamp():
    result = hsl_startup_preview._status_record_preview(
        {
            "latest_data": {
                "captured_at_ms": 1000,
                "input_expires_at_ms": 4000,
                "tier": "green",
                "observation_status": "unavailable",
            }
        },
        now_ms=2000,
    )
    assert result["status"] == "stale_or_unavailable"


def test_hsl_preview_portfolio_policy_uses_explicit_block():
    config = _sample_config()
    config["live"]["hsl_signal_mode"] = "unified"
    config["bot"]["hsl"] = {
        "enabled": True,
        "red_threshold": 0.25,
        "secret": "do-not-render",
    }
    report = hsl_startup_preview._config_report(config)
    assert report["hsl"]["portfolio"]["red_threshold"] == 0.25
    assert "do-not-render" not in json.dumps(report)


def test_hsl_startup_preview_reports_flat_hsl_config(tmp_path):
    config = _sample_config()
    config["bot"]["long"] = {
        "hsl_enabled": True,
        "hsl_red_threshold": 0.10,
        "hsl_cooldown_minutes_after_red": 45,
        "hsl_ema_span_minutes": 120,
        "hsl_panic_close_order_type": "limit",
        "hsl_restart_after_red_policy": "always",
    }
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = hsl_startup_preview.build_hsl_startup_preview_report(
        config_path,
        monitor_root="",
        now_ms=62_000,
    )

    assert report["ok"] is True
    assert report["config"]["hsl"]["sides"]["long"] == {
        "present": True,
        "enabled": True,
        "scale_budget_with_excess_allowance": None,
        "red_threshold": 0.10,
        "cooldown_minutes_after_red": 45,
        "ema_span_minutes": 120,
        "panic_close_order_type": "limit",
        "restart_after_red_policy": "always",
    }


def test_hsl_startup_preview_missing_config_path_is_user_safe():
    report = hsl_startup_preview.build_hsl_startup_preview_report(
        "/root/passivbot/configs/missing.json",
        monitor_root="",
        now_ms=62_000,
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["ok"] is False
    assert report["config_path"] == "~/passivbot/configs/missing.json"
    assert report["issues"][0]["path"] == "~/passivbot/configs/missing.json"
    assert "/root" not in rendered


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
        cli_main.main(["tool", "hsl-startup-preview", "configs/live.json", "--compact"])
        == 0
    )

    assert captured["module_name"] == "tools.hsl_startup_preview"
    assert captured["argv"] == [
        "passivbot tool hsl-startup-preview",
        "configs/live.json",
        "--compact",
    ]
    assert captured["prog_env"] == "passivbot tool hsl-startup-preview"
