from __future__ import annotations
from config.universe import set_side_universe_sources

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
        "config_version": "v9.0.0",
        "_raw": {"live": {"api_key": "super-secret-api-key"}},
        "backtest": {"exchanges": ["binance"]},
        "live": {
            "account": {"user": "binance_01"},
            "candles": {
                "max_forager_candle_staleness_minutes": 12,
                "max_forager_candle_refresh_seconds": 45,
            },
            "startup": {
                "force_cold_startup": False,
                "risk_bootstrap_max_lookback_minutes": 129600.0,
            },
        },
        "bot": {
            "portfolio": {"unstuck": {"memory_ema_span_minutes": 10080.0}, "loss_gate": {
                    "memory_ema_span_minutes": 20160.0,
                    "max_drawdown_ratio": 0.2,
                }, "hsl": {
                    "scope": "portfolio",
                    "enabled": True,
                    "drawdown_ratio_threshold": 0.05,
                    "cooldown_after_red_minutes": 60.0,
                    "memory_ema_span_minutes": 43200.0,
                    "drawdown_ratio_ema_span_minutes": 120.0,
                    "scale_budget_with_excess_allowance": False,
                    "panic_close_order_type": "limit",
                }, "warmup": {"max_minutes": 30}}, "global": {"forager": {"score_hysteresis_margin": 0.02}},
            "long": {
                "risk": {"max_positions": 3, "total_wallet_exposure_limit": 1.0},
                "unstuck": {"enabled": True},
                "forager": {"volume_ema_span_minutes": 120.0},
                "hsl": {
                    "enabled": True,
                    "drawdown_ratio_threshold": 0.05,
                    "cooldown_after_red_minutes": 60,
                    "memory_ema_span_minutes": 43200.0,
                    "drawdown_ratio_ema_span_minutes": 120.0,
                    "scale_budget_with_excess_allowance": False,
                    "panic_close_order_type": "limit",
                },
                "universe": {
                    "approved_coins": ["BTC", "ETH", "SOL", "XMR"],
                    "ignored_coins": ["DOGE"],
                },
            },
            "short": {
                "risk": {"max_positions": 1, "total_wallet_exposure_limit": 0.0},
                "unstuck": {"enabled": False},
                "hsl": {
                    "enabled": False,
                    "drawdown_ratio_threshold": 0.04,
                    "cooldown_after_red_minutes": 30,
                    "memory_ema_span_minutes": 43200.0,
                    "drawdown_ratio_ema_span_minutes": 120.0,
                    "scale_budget_with_excess_allowance": False,
                    "panic_close_order_type": "limit",
                },
                "universe": {"approved_coins": ["BTC"], "ignored_coins": []},
            },
        },
    }


@pytest.mark.parametrize("cap,expected", [(None, None), (30.0, None), (0, "zero")])
def test_preflight_recognizes_explicit_uncapped_warmup(tmp_path, cap, expected):
    config = _sample_config()
    config["bot"]["portfolio"]["warmup"]["max_minutes"] = cap
    path = tmp_path / "config.json"
    _write_config(path, config)
    report = live_config_preflight.build_live_config_preflight_report(path)
    attention = report["cache"]["readiness"]["surfaces"]["candles"]["attention"]
    codes = {row["code"] for row in attention if row["code"].startswith("max_warmup_minutes_")}
    assert codes == ({f"max_warmup_minutes_{expected}"} if expected else set())


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
    assert report["hsl"]["signal_mode"] == "portfolio"
    assert report["hsl"]["sides"]["long"]["enabled"] is True
    assert report["hsl"]["sides"]["long"]["drawdown_ratio_threshold"] == 0.05
    assert report["hsl"]["sides"]["short"]["enabled"] is False
    hsl_evidence = report["cache"]["readiness"]["surfaces"]["hsl"]["evidence"]
    assert {item["code"] for item in hsl_evidence} >= {
        "hsl_enabled",
        "hsl_history_reinterpretation_caveat",
    }
    caveat = next(
        item
        for item in hsl_evidence
        if item["code"] == "hsl_history_reinterpretation_caveat"
    )
    assert caveat["doc"] == "docs/equity_hard_stop_loss.md"
    assert "deposits" in caveat["message"]
    assert "HSL mode changes" in caveat["message"]
    assert report["universe"]["approved_coins"]["long"] == {
        "count": 4,
        "mode": "list",
        "present": True,
        "sample": ["BTC", "ETH"],
        "truncated": 2,
    }
    assert report["forager"]["sides"]["long"]["max_positions"] == 3
    assert report["forager"]["total_configured_max_positions"] == 4.0
    assert report["risk"]["balance_hysteresis_snap_fraction"] == {
        "basis": (
            "snapped balance is used for sizing/gating surfaces where hysteresis "
            "is intentional; raw balance remains the basis for exact exposure repair"
        ),
        "default_value": 0.02,
        "effective_value": 0.02,
        "present": False,
        "status": "defaulted",
        "warning_threshold": 0.05,
    }
    fill_evidence = {row["code"]: row for row in report["cache"]["readiness"]["surfaces"]["fills"]["evidence"]}
    assert fill_evidence["unstuck_memory_ema_span_minutes_configured"]["value"] == 10080.0
    assert fill_evidence["loss_gate_memory_ema_span_minutes_configured"]["value"] == 20160.0
    assert "realized_loss_gate_lookback_minutes" not in report["cache"]["live_settings"]
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


def test_live_preflight_requires_explicit_v8_migration(tmp_path):
    config = _sample_config()
    config["config_version"] = "v8.0.0"
    path = tmp_path / "old.json"
    _write_config(path, config)
    report = live_config_preflight.build_live_config_preflight_report(path)
    assert not report["ok"]
    assert any(row["code"] == "config_migration_required" and "migrate-config-v9" in row["message"]
               for row in report["issues"])


def test_live_preflight_rejects_unknown_secret_field_without_rendering_value(tmp_path):
    config = _sample_config()
    config["live"]["api_key"] = "unknown-field-secret"
    path = tmp_path / "unknown.json"
    _write_config(path, config)
    report = live_config_preflight.build_live_config_preflight_report(path)
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" and "live.api_key" in issue["message"]
               for issue in report["issues"])
    assert "unknown-field-secret" not in json.dumps(report)


def test_live_preflight_rejects_cooldown_beyond_reconstructable_bootstrap_cap(tmp_path):
    config = _sample_config()
    config["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = 30.0
    path = tmp_path / "bound.json"
    _write_config(path, config)
    report = live_config_preflight.build_live_config_preflight_report(path)
    assert not report["ok"]
    assert any(row["code"] == "hsl_bootstrap_capability_invalid" for row in report["issues"])


def test_live_config_preflight_warns_on_high_balance_hysteresis(tmp_path):
    config = _sample_config()
    config["live"]["account"]["balance_hysteresis_snap_fraction"] = 0.06
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is True
    assert report["summary"]["warning_count"] == 1
    assert report["risk"]["balance_hysteresis_snap_fraction"] == {
        "basis": (
            "snapped balance is used for sizing/gating surfaces where hysteresis "
            "is intentional; raw balance remains the basis for exact exposure repair"
        ),
        "default_value": 0.02,
        "present": True,
        "raw_value": 0.06,
        "status": "high",
        "value": 0.06,
        "warning_threshold": 0.05,
    }
    assert report["issues"] == [
        {
            "code": "balance_hysteresis_snap_pct_high",
            "message": (
                "live.account.balance_hysteresis_snap_fraction is above 5%; snapped-balance entry sizing/gating may diverge noticeably from raw-balance exposure repair near risk boundaries"
            ),
            "path": "live.account.balance_hysteresis_snap_fraction",
            "severity": "warning",
        }
    ]


@pytest.mark.parametrize("value", [-0.01, "wide", True])
def test_live_config_preflight_warns_on_invalid_balance_hysteresis(tmp_path, value):
    config = _sample_config()
    config["live"]["account"]["balance_hysteresis_snap_fraction"] = value
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is True
    assert report["summary"]["warning_count"] == 1
    assert report["risk"]["balance_hysteresis_snap_fraction"]["status"] in {
        "invalid",
        "invalid_bool",
    }
    assert report["issues"][0]["path"] == "live.account.balance_hysteresis_snap_fraction"


def test_live_config_preflight_reports_cache_readiness_without_artifact_claims(
    tmp_path,
):
    config = _sample_config()
    config["bot"]["portfolio"]["warmup"]["span_multiplier"] = 0.3
    config["live"].update(
        {
            "startup": {
                **config["live"].get("startup", {}),
                "defer_broad_candle_warmup": True,
            },
            "candles": {
                **config["live"].get("candles", {}),
                "enable_archive_candle_fetch": False,
                "max_active_candle_tail_gap_minutes": 10,
                "max_disk_candles_per_symbol_per_tf": 2_000_000,
                "max_memory_candles_per_symbol": 200_000,
            },
            "fills": {
                **config["live"].get("fills", {}),
                "fills_confirmation_overlap_minutes": 60,
                "fills_recent_overlap_minutes": 10,
            },
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


def test_live_config_preflight_flags_account_hsl_with_config_balance_override(tmp_path):
    config = _sample_config()
    config["live"]["account"]["balance_override"] = 1000
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is False
    assert report["hsl"]["effective_signal_mode"] == "portfolio"
    assert report["hsl"]["balance_override"] == {
        "active": True,
        "present": True,
        "source": "live.account.balance_override",
        "status": "valid",
        "value": 1000.0,
    }
    assert report["summary"]["error_count"] == 1
    assert report["issues"][0]["code"] == (
        "hsl_balance_override_account_level_replay_unsafe"
    )
    assert report["issues"][0]["path"] == "live.account.balance_override"
    hsl_attention = report["cache"]["readiness"]["surfaces"]["hsl"]["attention"]
    assert [item["code"] for item in hsl_attention] == [
        "hsl_balance_override_account_level_replay_unsafe"
    ]


def test_live_config_preflight_flags_account_hsl_with_launch_balance_override(tmp_path):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        balance_override="1000",
    )

    assert report["ok"] is False
    assert report["hsl"]["balance_override"]["active"] is True
    assert report["hsl"]["balance_override"]["source"] == "argument"
    assert report["issues"][0]["code"] == (
        "hsl_balance_override_account_level_replay_unsafe"
    )
    assert report["issues"][0]["path"] == "argument.balance_override"


@pytest.mark.parametrize("override_value", [0, -1, False, "0"])
def test_live_config_preflight_blocks_invalid_runtime_override_with_account_hsl(
    tmp_path,
    override_value,
):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        balance_override=override_value,
    )
    issue_codes = [issue["code"] for issue in report["issues"]]

    assert report["ok"] is False
    assert report["hsl"]["balance_override"]["active"] is True
    assert "hsl_balance_override_account_level_replay_unsafe" in issue_codes
    assert any(code.startswith("balance_override_invalid") for code in issue_codes)


def test_live_config_preflight_allows_coin_hsl_with_balance_override(tmp_path):
    config = _sample_config()
    config["bot"]["portfolio"]["hsl"]["scope"] = "coin"
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        balance_override=1000,
    )

    assert report["ok"] is True
    assert report["hsl"]["effective_signal_mode"] == "coin"
    assert report["hsl"]["balance_override"]["active"] is True
    assert not [
        issue
        for issue in report["issues"]
        if issue["code"] == "hsl_balance_override_account_level_replay_unsafe"
    ]


def test_live_config_preflight_uses_schema_default_for_missing_hsl_signal_mode(
    tmp_path,
):
    config = _sample_config()
    del config["bot"]["portfolio"]["hsl"]["scope"]
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(
        config_path,
        balance_override=1000,
    )

    assert report["ok"] is True
    assert report["hsl"]["signal_mode"] is None
    assert report["hsl"]["effective_signal_mode"] == "coin"
    assert not [
        issue
        for issue in report["issues"]
        if issue["code"] == "hsl_balance_override_account_level_replay_unsafe"
    ]


def test_sparse_canonical_preflight_cli_serializes_absent_hsl_scope(tmp_path, capsys):
    path = tmp_path / "sparse.json"
    _write_config(path, {
        "config_version": "v9.0.0", "bot": {}, "live": {}, "backtest": {},
    })
    assert live_config_preflight.main([str(path), "--compact"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["hsl"]["signal_mode"] is None
    assert report["hsl"]["effective_signal_mode"] == "coin"


def test_live_config_preflight_reports_canonical_shared_bot_keys(tmp_path):
    config = _sample_config()
    config["bot"]["long"] = {
        "risk": {"max_positions": 3},
        "forager": {"volume_ema_span_minutes": 120.0},
        "hsl": {
            "enabled": True,
            "drawdown_ratio_threshold": 0.05,
            "cooldown_after_red_minutes": 60,
            "memory_ema_span_minutes": 43200.0,
            "drawdown_ratio_ema_span_minutes": 120.0,
            "panic_close_order_type": "limit",
        },
    }
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is True
    assert report["hsl"]["sides"]["long"]["present"] is True
    assert report["hsl"]["sides"]["long"]["enabled"] is True
    assert report["hsl"]["sides"]["long"]["drawdown_ratio_threshold"] == 0.05
    assert "restart_after_red_policy" not in report["hsl"]["sides"]["long"]
    assert report["forager"]["sides"]["long"]["max_positions"] == 3
    assert report["forager"]["sides"]["long"]["forager_present"] is True
    assert report["forager"]["sides"]["long"]["settings"] == {"volume_ema_span_minutes": 120.0}


def test_live_config_preflight_compare_reports_bounded_risk_relevant_changes(tmp_path):
    baseline = _sample_config()
    target = _sample_config()
    target["live"]["account"]["user"] = "okx_01"
    target["backtest"]["exchanges"] = ["okx"]
    target["bot"]["portfolio"]["hsl"]["scope"] = "pside"
    target['bot']["long"]['universe']['approved_coins'] = ["BTC", "ETH", "SOL", "ADA", "BNB"]
    target['bot']["short"]['universe']['ignored_coins'] = ["DOGE", "XRP"]
    target["live"]["candles"]["max_forager_candle_staleness_minutes"] = 20
    target['bot']["portfolio"]['warmup']['max_minutes'] = 45
    target["_raw"]["live"]["api_key"] = "new-secret-api-key"
    target["bot"]["long"]["hsl"]["enabled"] = False
    target["bot"]["short"]["risk"]["max_positions"] = 2

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
    assert changes["live.account.user"]["before"] == {
        "present": True,
        "value": "binance_01",
    }
    assert changes["backtest.exchanges"]["after"] == {"present": True, "value": ["okx"]}
    assert changes["bot.portfolio.hsl.scope"]["after"] == {
        "present": True,
        "value": "pside",
    }
    assert changes["bot.long.hsl.enabled"]["after"] == {
        "present": True,
        "value": False,
    }
    assert changes["live.candles.max_forager_candle_staleness_minutes"]["after"] == {
        "present": True,
        "value": 20,
    }
    assert changes['bot.portfolio.warmup.max_minutes']["after"] == {
        "present": True,
        "value": 45,
    }
    assert changes["cache.readiness.summary"]["category"] == "cache"
    assert (
        changes["cache.readiness.summary"]["before"]["value"]["disabled_surface_count"]
        == 0
    )
    assert (
        changes["cache.readiness.summary"]["after"]["value"]["disabled_surface_count"]
        == 1
    )
    assert changes["bot.long.universe.approved_coins"]["added_count"] == 2
    assert changes["bot.long.universe.approved_coins"]["removed_count"] == 1
    assert changes["bot.long.universe.approved_coins"]["added_sample"] == ["ADA"]
    assert changes["bot.long.universe.approved_coins"]["added_truncated"] == 1
    assert changes["bot.short.universe.ignored_coins"]["added_count"] == 2
    assert changes["bot.short.universe.ignored_coins"]["added_sample"] == ["DOGE"]
    assert changes["bot.short.universe.ignored_coins"]["added_truncated"] == 1
    assert "super-secret-api-key" not in rendered
    assert "new-secret-api-key" not in rendered
    assert "api_key" not in rendered


def test_live_config_preflight_compare_resolves_canonical_shared_bot_keys(tmp_path):
    baseline = _sample_config()
    target = _sample_config()
    target["bot"]["short"] = {
        "risk": {"max_positions": 2},
        "hsl": {"enabled": True},
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
    assert changes["bot.short.risk.max_positions"]["before"] == {
        "present": True,
        "value": 1,
    }
    assert changes["bot.short.risk.max_positions"]["after"] == {
        "present": True,
        "value": 2,
    }


def test_live_config_preflight_handles_all_and_invalid_coin_shapes(tmp_path):
    config = _sample_config()
    set_side_universe_sources(config, 'approved_coins', "all")
    set_side_universe_sources(config, 'ignored_coins', {"long": {"BTC": True}, "short": []})
    config_path = tmp_path / "live.json"
    _write_config(config_path, config)

    report = live_config_preflight.build_live_config_preflight_report(config_path)

    assert report["ok"] is False
    assert report["universe"]["approved_coins"]["long"]["mode"] == "all"
    assert report["universe"]["approved_coins"]["short"]["mode"] == "all"
    assert report["universe"]["ignored_coins"]["long"]["mode"] == "invalid"
    assert report["summary"] == {"error_count": 1, "warning_count": 1}
    assert [issue["code"] for issue in report["issues"]] == [
        "config_schema_invalid", "coin_list_shape_invalid",
    ]


def test_live_config_preflight_malformed_structure_returns_nonzero(tmp_path, capsys):
    config_path = tmp_path / "bad.json"
    _write_config(config_path, {"config_version": "v9.0.0", "live": []})

    assert live_config_preflight.main([str(config_path), "--compact"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["ok"] is False
    assert [issue["code"] for issue in report["issues"]] == [
        "required_section_invalid",
        "required_section_invalid",
        "config_schema_invalid",
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
    target["live"]["startup"]["force_cold_startup"] = True
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
    assert changes["live.startup.force_cold_startup"]["after"] == {
        "present": True,
        "value": True,
    }
    assert changes["cache.readiness.summary"]["after"]["value"]["attention_count"] == 8


def test_live_config_preflight_cli_balance_override_returns_nonzero(tmp_path, capsys):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    assert (
        live_config_preflight.main(
            [str(config_path), "--balance-override", "1000", "--compact"]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)

    assert report["ok"] is False
    assert report["issues"][0]["code"] == (
        "hsl_balance_override_account_level_replay_unsafe"
    )
    assert report["hsl"]["balance_override"]["source"] == "argument"


def test_live_config_preflight_cli_zero_balance_override_returns_nonzero(
    tmp_path, capsys
):
    config_path = tmp_path / "live.json"
    _write_config(config_path, _sample_config())

    assert (
        live_config_preflight.main(
            [str(config_path), "--balance-override", "0", "--compact"]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    issue_codes = [issue["code"] for issue in report["issues"]]

    assert report["ok"] is False
    assert report["hsl"]["balance_override"]["active"] is True
    assert "balance_override_invalid" in issue_codes
    assert "hsl_balance_override_account_level_replay_unsafe" in issue_codes


@pytest.mark.parametrize("portfolio_enabled", [False, True])
def test_portfolio_preflight_uses_only_portfolio_policy(tmp_path, portfolio_enabled):
    config = _sample_config()
    config["live"]["account"]["balance_override"] = 1000
    config["bot"]["portfolio"]["hsl"]["enabled"] = portfolio_enabled
    for side in ("long", "short"):
        config["bot"][side]["hsl"]["enabled"] = not portfolio_enabled
    path = tmp_path / "config.json"
    _write_config(path, config)
    report = live_config_preflight.build_live_config_preflight_report(path)
    assert report["hsl"]["policies"]["portfolio"]["enabled"] is portfolio_enabled
    errors = [
        i
        for i in report["issues"]
        if i["code"] == "hsl_balance_override_account_level_replay_unsafe"
    ]
    assert bool(errors) is portfolio_enabled
