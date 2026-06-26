from __future__ import annotations

import json
import os
import sys

import pytest

from passivbot_cli import main as cli_main
from tools import live_config_preflight


def _write_config(path, config):
    path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")


def _sample_config() -> dict:
    return {
        "config_version": "v8.0.0",
        "backtest": {"exchanges": ["binance"]},
        "live": {
            "user": "binance_01",
            "hsl_signal_mode": "unified",
            "hsl_position_during_cooldown_policy": "panic",
            "approved_coins": {
                "long": ["BTC", "ETH", "SOL", "XMR"],
                "short": ["BTC"],
            },
            "ignored_coins": {"long": ["DOGE"], "short": []},
            "max_forager_candle_staleness_minutes": 12,
            "max_forager_candle_refresh_seconds": 45,
            "forager_score_hysteresis_pct": 0.02,
            "force_cold_startup": False,
            "max_warmup_minutes": 30,
            "pnls_max_lookback_days": 7,
            "api_key": "super-secret-api-key",
        },
        "bot": {
            "long": {
                "risk": {"n_positions": 3},
                "forager": {"volume_drop_pct": 0.02},
                "hsl": {
                    "enabled": True,
                    "red_threshold": 0.05,
                    "cooldown_minutes_after_red": 60,
                    "no_restart_drawdown_threshold": 0.08,
                    "ema_span_minutes": 120,
                    "tier_ratios": {"yellow": 0.5, "orange": 0.75},
                    "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                    "panic_close_order_type": "limit",
                },
            },
            "short": {
                "risk": {"n_positions": 1},
                "hsl": {
                    "enabled": False,
                    "red_threshold": 0.04,
                    "cooldown_minutes_after_red": 30,
                },
            },
        },
    }


def test_live_config_preflight_reports_risk_relevant_shape_and_bounds_symbols(tmp_path):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        sample_size=2,
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["ok"] is True
    assert report["identity"] == {
        "account": "binance_01",
        "backtest_exchanges": ["binance"],
        "exchange": None,
        "exchange_source": "not_in_config",
        "user": "binance_01",
        "user_exchange_hint": "binance",
    }
    assert report["hsl"]["signal_mode"] == "unified"
    assert report["hsl"]["sides"]["long"]["enabled"] is True
    assert report["hsl"]["sides"]["long"]["red_threshold"] == 0.05
    assert report["hsl"]["sides"]["short"]["enabled"] is False
    assert report["universe"]["approved_coins"]["long"] == {
        "count": 4,
        "mode": "list",
        "present": True,
        "sample": ["BTC", "ETH"],
        "truncated": 2,
    }
    assert report["forager"]["sides"]["long"]["n_positions"] == 3
    assert report["forager"]["total_configured_n_positions"] == 4.0
    assert report["cache"]["live_settings"]["pnls_max_lookback_days"] == 7
    assert "super-secret-api-key" not in rendered
    assert "api_key" not in rendered


def test_live_config_preflight_handles_all_and_invalid_coin_shapes(tmp_path):
    config = _sample_config()
    config["live"]["approved_coins"] = "all"
    config["live"]["ignored_coins"] = {"long": {"BTC": True}, "short": []}
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is True
    assert report["universe"]["approved_coins"]["long"]["mode"] == "all"
    assert report["universe"]["approved_coins"]["short"]["mode"] == "all"
    assert report["universe"]["ignored_coins"]["long"]["mode"] == "invalid"
    assert report["summary"] == {"error_count": 0, "warning_count": 1}
    assert report["issues"][0]["code"] == "coin_list_shape_invalid"


def test_live_config_preflight_malformed_structure_returns_nonzero(tmp_path, capsys):
    config_path = tmp_path / "bad.json"
    _write_config(config_path, {"live": []})

    assert live_config_preflight.main([str(config_path), "--compact"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["ok"] is False
    assert [issue["code"] for issue in report["issues"]] == [
        "required_section_invalid",
        "required_section_invalid",
    ]
    assert "required config section" in report["issues"][0]["message"]


def test_live_config_preflight_invalid_json_returns_nonzero(tmp_path, capsys):
    config_path = tmp_path / "bad.json"
    config_path.write_text("{not-json", encoding="utf-8")

    assert live_config_preflight.main([str(config_path)]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["ok"] is False
    assert report["issues"][0]["code"] == "json_decode_failed"


def test_live_config_preflight_rejects_negative_sample_size(tmp_path):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        sample_size=-1,
    )

    assert report["universe"]["approved_coins"]["long"]["sample"] == []
    assert report["universe"]["approved_coins"]["long"]["truncated"] == 4


def test_live_config_preflight_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc_info:
        live_config_preflight.main(["--help"])

    assert exc_info.value.code == 0
    assert "live-config-preflight" in capsys.readouterr().out


def test_live_config_preflight_tool_dispatch_forwards_module_and_prog(monkeypatch):
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
            ["tool", "live-config-preflight", "configs/live.json", "--compact"]
        )
        == 0
    )

    assert captured["module_name"] == "tools.live_config_preflight"
    assert captured["argv"] == [
        "passivbot tool live-config-preflight",
        "configs/live.json",
        "--compact",
    ]
    assert captured["prog_env"] == "passivbot tool live-config-preflight"
