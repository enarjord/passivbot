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
    assert report["cache"]["readiness"]["status"] == "attention"
    assert report["cache"]["readiness"]["summary"] == {
        "attention_count": 7,
        "disabled_surface_count": 0,
        "not_proven_count": 3,
    }
    assert report["cache"]["readiness"]["root_hints"]["fill_events_root"] == {
        "available": True,
        "path": "caches/fill_events/binance/binance_01",
    }
    assert {
        item["code"]
        for item in report["cache"]["readiness"]["surfaces"]["candles"]["attention"]
    } == {
        "defer_broad_candle_warmup_missing",
        "max_active_candle_tail_gap_minutes_missing",
        "max_disk_candles_per_symbol_per_tf_missing",
        "max_memory_candles_per_symbol_missing",
        "warmup_ratio_missing",
    }
    assert "super-secret-api-key" not in rendered
    assert "api_key" not in rendered


def test_live_config_preflight_reports_cache_readiness_without_artifact_claims(tmp_path):
    config = _sample_config()
    config["live"].update(
        {
            "defer_broad_candle_warmup": True,
            "enable_archive_candle_fetch": False,
            "fills_confirmation_overlap_minutes": 60,
            "fills_recent_overlap_minutes": 10,
            "max_active_candle_tail_gap_minutes": 10,
            "max_disk_candles_per_symbol_per_tf": 2_000_000,
            "max_memory_candles_per_symbol": 200_000,
            "warmup_ratio": 0.3,
        }
    )
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)
    readiness = report["cache"]["readiness"]

    assert report["ok"] is True
    assert readiness["status"] == "settings_compatible_artifacts_not_checked"
    assert readiness["summary"] == {
        "attention_count": 0,
        "disabled_surface_count": 0,
        "not_proven_count": 3,
    }
    assert readiness["surfaces"]["candles"]["not_proven"] == [
        {
            "code": "local_candle_artifacts_not_scanned",
            "message": (
                "preflight inspects config only; run cache-integrity-doctor "
                "to prove local candle coverage"
            ),
        }
    ]
    assert readiness["surfaces"]["fills"]["not_proven"][0]["code"] == (
        "local_fill_artifacts_not_scanned"
    )
    assert readiness["surfaces"]["hsl"]["not_proven"][0]["code"] == (
        "local_hsl_artifacts_not_scanned"
    )


def test_live_config_preflight_reports_flat_shared_bot_keys(tmp_path):
    config = _sample_config()
    config["bot"]["long"] = {
        "n_positions": 3,
        "forager_volume_drop_pct": 0.02,
        "hsl_enabled": True,
        "hsl_red_threshold": 0.05,
        "hsl_cooldown_minutes_after_red": 60,
        "hsl_no_restart_drawdown_threshold": 0.08,
        "hsl_ema_span_minutes": 120,
        "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
        "hsl_orange_tier_mode": "tp_only_with_active_entry_cancellation",
        "hsl_panic_close_order_type": "limit",
    }
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is True
    assert report["hsl"]["sides"]["long"]["present"] is True
    assert report["hsl"]["sides"]["long"]["enabled"] is True
    assert report["hsl"]["sides"]["long"]["red_threshold"] == 0.05
    assert report["hsl"]["sides"]["long"]["tier_ratios"] == {
        "yellow": 0.5,
        "orange": 0.75,
    }
    assert report["forager"]["sides"]["long"]["n_positions"] == 3
    assert report["forager"]["sides"]["long"]["forager_present"] is True
    assert report["forager"]["sides"]["long"]["settings"] == {
        "volume_drop_pct": 0.02
    }


def test_live_config_preflight_compare_reports_bounded_risk_relevant_changes(tmp_path):
    baseline = _sample_config()
    target = _sample_config()
    target["live"]["user"] = "okx_01"
    target["live"]["exchange"] = "okx"
    target["live"]["hsl_signal_mode"] = "per_side"
    target["live"]["approved_coins"]["long"] = ["BTC", "ETH", "SOL", "ADA", "BNB"]
    target["live"]["ignored_coins"]["short"] = ["DOGE", "XRP"]
    target["live"]["max_forager_candle_staleness_minutes"] = 20
    target["live"]["max_warmup_minutes"] = 45
    target["live"]["api_key"] = "new-secret-api-key"
    target["bot"]["long"]["hsl"]["enabled"] = False
    target["bot"]["short"]["risk"] = {"n_positions": 2}

    baseline_path = tmp_path / "baseline.json"
    target_path = tmp_path / "target.json"
    _write_config(baseline_path, baseline)
    _write_config(target_path, target)

    report = live_config_preflight.build_live_config_preflight_report(
        target_path,
        compare_config_path=baseline_path,
        sample_size=1,
    )
    rendered = json.dumps(report, sort_keys=True)
    changes = {change["field"]: change for change in report["diff"]["changes"]}

    assert report["ok"] is True
    assert report["diff"]["ok"] is True
    assert report["diff"]["summary"]["category_counts"]["hsl"] == 2
    assert changes["live.user"]["before"] == {
        "present": True,
        "value": "binance_01",
    }
    assert changes["live.exchange"]["after"] == {"present": True, "value": "okx"}
    assert changes["live.hsl_signal_mode"]["after"] == {
        "present": True,
        "value": "per_side",
    }
    assert changes["bot.long.hsl.enabled"]["after"] == {
        "present": True,
        "value": False,
    }
    assert changes["live.max_forager_candle_staleness_minutes"]["after"] == {
        "present": True,
        "value": 20,
    }
    assert changes["live.max_warmup_minutes"]["after"] == {
        "present": True,
        "value": 45,
    }
    assert changes["cache.readiness.summary"]["category"] == "cache"
    assert changes["cache.readiness.summary"]["before"]["value"][
        "disabled_surface_count"
    ] == 0
    assert changes["cache.readiness.summary"]["after"]["value"][
        "disabled_surface_count"
    ] == 1
    assert changes["live.approved_coins.long"]["added_count"] == 2
    assert changes["live.approved_coins.long"]["removed_count"] == 1
    assert changes["live.approved_coins.long"]["added_sample"] == ["ADA"]
    assert changes["live.approved_coins.long"]["added_truncated"] == 1
    assert changes["live.ignored_coins.short"]["added_count"] == 2
    assert changes["live.ignored_coins.short"]["added_sample"] == ["DOGE"]
    assert changes["live.ignored_coins.short"]["added_truncated"] == 1
    assert "super-secret-api-key" not in rendered
    assert "new-secret-api-key" not in rendered
    assert "api_key" not in rendered


def test_live_config_preflight_compare_resolves_flat_shared_bot_keys(tmp_path):
    baseline = _sample_config()
    target = _sample_config()
    target["bot"]["short"] = {
        "n_positions": 2,
        "hsl_enabled": True,
    }
    baseline_path = tmp_path / "baseline.json"
    target_path = tmp_path / "target.json"
    _write_config(baseline_path, baseline)
    _write_config(target_path, target)

    report = live_config_preflight.build_live_config_preflight_report(
        target_path,
        compare_config_path=baseline_path,
    )
    changes = {change["field"]: change for change in report["diff"]["changes"]}

    assert report["ok"] is True
    assert changes["bot.short.hsl.enabled"]["before"] == {
        "present": True,
        "value": False,
    }
    assert changes["bot.short.hsl.enabled"]["after"] == {
        "present": True,
        "value": True,
    }
    assert changes["bot.short.risk.n_positions"]["before"] == {
        "present": True,
        "value": 1,
    }
    assert changes["bot.short.risk.n_positions"]["after"] == {
        "present": True,
        "value": 2,
    }


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


def test_live_config_preflight_missing_config_path_is_user_safe():
    report = live_config_preflight.build_live_config_preflight_report(
        "/root/passivbot/configs/missing.json"
    )

    assert report["ok"] is False
    assert report["config_path"] == "~/passivbot/configs/missing.json"
    assert report["issues"][0]["path"] == "~/passivbot/configs/missing.json"
    rendered = json.dumps(report, sort_keys=True)
    assert "/root" not in rendered


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


def test_live_config_preflight_compare_cli_emits_diff(tmp_path, capsys):
    baseline_path = tmp_path / "baseline.json"
    target_path = tmp_path / "target.json"
    baseline = _sample_config()
    target = _sample_config()
    target["live"]["force_cold_startup"] = True
    _write_config(baseline_path, baseline)
    _write_config(target_path, target)

    assert (
        live_config_preflight.main(
            [str(target_path), "--compare", str(baseline_path), "--compact"]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)

    changes = {change["field"]: change for change in report["diff"]["changes"]}

    assert report["diff"]["summary"]["change_count"] == 2
    assert changes["live.force_cold_startup"]["after"] == {
        "present": True,
        "value": True,
    }
    assert changes["cache.readiness.summary"]["after"]["value"][
        "attention_count"
    ] == 8
