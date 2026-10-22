"""Preflight inspects resolved policy without changing authored declarations."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.live_config_preflight import build_live_config_diff_report, build_live_config_preflight_report


def _hsl(enabled):
    return dict(enabled=enabled, drawdown_ratio_threshold=0.05,
                cooldown_after_red_minutes=60.0, memory_ema_span_minutes=43200.0,
                drawdown_ratio_ema_span_minutes=120.0,
                scale_budget_with_excess_allowance=False, panic_close_order_type="limit")


def _config():
    return {"config_version": "v9.0.0", "backtest": {},
            "live": {"account": {"balance_override": 1000.0},
                     "startup": {"risk_bootstrap_max_lookback_minutes": 129600.0}},
            "bot": {"portfolio": {"hsl": {"scope": "position_side"}},
                    "global": {"hsl": _hsl(True)}}}


def _write(tmp_path, name, config):
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("empty_sides", [False, True])
@pytest.mark.parametrize("short_pin", [None, False])
def test_sparse_inherited_hsl_remains_protective_and_reports_effective_side_defaults(tmp_path, empty_sides, short_pin):
    config = _config()
    if empty_sides:
        config["bot"].update(long={}, short={})
    if short_pin is not None:
        config["bot"]["short"] = {"hsl": {"enabled": short_pin}}
    config["_raw"] = {"bot": {"global": {"hsl": {"enabled": False}}}}
    path = _write(tmp_path, "config.json", config)
    original = path.read_bytes()
    report = build_live_config_preflight_report(path)
    assert report["ok"] is False
    assert report["hsl"]["policies"]["long"]["enabled"] is True
    assert report["hsl"]["policies"]["short"]["enabled"] is (short_pin is None)
    assert report["hsl"]["sides"] == report["hsl"]["policies"]
    codes = {issue["code"] for issue in report["issues"]}
    assert "hsl_balance_override_account_level_replay_unsafe" in codes
    assert "side_config_missing" not in codes
    assert report["cache"]["readiness"]["surfaces"]["hsl"]["status"] != "disabled"
    assert report["forager"]["total_configured_max_positions"] == 14.0
    assert all(report["forager"]["sides"][side]["max_positions"] == 7 for side in ("long", "short"))
    assert path.read_bytes() == original


def test_explicit_false_hsl_pins_remain_false_under_true_global_policy(tmp_path):
    config = _config()
    config["bot"].update({side: {"hsl": {"enabled": False}} for side in ("long", "short")})
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is True
    assert all(policy["enabled"] is False for policy in report["hsl"]["policies"].values())
    assert report["cache"]["readiness"]["surfaces"]["hsl"]["status"] == "disabled"


@pytest.mark.parametrize("portfolio_enabled", [False, True])
@pytest.mark.parametrize("scope", ["portfolio", " PORTFOLIO "])
def test_sparse_portfolio_hsl_uses_only_explicit_portfolio_policy(tmp_path, portfolio_enabled, scope):
    config = _config()
    config["bot"]["portfolio"]["hsl"] = {"scope": scope, **_hsl(portfolio_enabled)}
    config["bot"]["global"]["hsl"]["enabled"] = not portfolio_enabled
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["hsl"]["policies"]["portfolio"]["enabled"] is portfolio_enabled
    assert report["ok"] is (not portfolio_enabled)
    assert any(issue["code"] == "hsl_balance_override_account_level_replay_unsafe"
               for issue in report["issues"]) is portfolio_enabled


def test_inherited_cooldown_still_checks_bootstrap_cap(tmp_path):
    config = _config()
    config["live"]["account"]["balance_override"] = None
    config["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = 30.0
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "hsl_bootstrap_capability_invalid" for issue in report["issues"])


def test_diff_resolves_shared_hsl_and_capacity_without_overriding_side_pins(tmp_path):
    baseline = _config()
    baseline["bot"]["global"]["hsl"]["enabled"] = False
    baseline["bot"]["short"] = {"hsl": {"enabled": False}, "risk": {"max_positions": 2}}
    target = deepcopy(baseline)
    target["bot"]["global"]["hsl"]["enabled"] = True
    target["bot"]["global"]["risk"] = {"max_positions": 3}
    before = _write(tmp_path, "before.json", baseline)
    after = _write(tmp_path, "after.json", target)
    report = build_live_config_diff_report(before, after)
    changes = {change["field"]: change for change in report["changes"]}
    assert changes["bot.long.hsl.enabled"]["before"]["value"] is False
    assert changes["bot.long.hsl.enabled"]["after"]["value"] is True
    assert "bot.short.hsl.enabled" not in changes
    assert changes["bot.long.risk.max_positions"]["before"]["value"] == 7
    assert changes["bot.long.risk.max_positions"]["after"]["value"] == 3
    assert "bot.short.risk.max_positions" not in changes


@pytest.mark.parametrize("scope", ["global", "long"])
def test_diff_reports_effective_hysteresis_per_side(tmp_path, scope):
    baseline = _config()
    baseline["bot"]["short"] = {"forager": {"score_hysteresis_margin": 0.1}}
    target = deepcopy(baseline)
    target["bot"].setdefault(scope, {}).setdefault("forager", {})["score_hysteresis_margin"] = 0.25
    report = build_live_config_diff_report(_write(tmp_path, "before.json", baseline),
                                          _write(tmp_path, "after.json", target))
    changes = {change["field"]: change for change in report["changes"]}
    assert changes["bot.long.forager.score_hysteresis_margin"]["before"]["value"] == 0.02
    assert changes["bot.long.forager.score_hysteresis_margin"]["after"]["value"] == 0.25
    assert "bot.short.forager.score_hysteresis_margin" not in changes


def test_equal_effective_side_pin_remains_visible_as_authored_diff(tmp_path):
    baseline = _config()
    target = deepcopy(baseline)
    target["bot"]["long"] = {"hsl": {"enabled": True}}
    report = build_live_config_diff_report(_write(tmp_path, "before.json", baseline),
                                          _write(tmp_path, "after.json", target))
    change = next(change for change in report["changes"] if change["field"] == "bot.long.hsl.enabled")
    assert change["before"] == change["after"] == {"present": True, "value": True}
    assert change["authored_before"] == {"present": False}
    assert change["authored_after"] == {"present": True, "value": True}
    assert change["effective_changed"] is False and change["authored_changed"] is True


def test_policy_inspection_never_prepares_config_or_reads_coin_source_files(tmp_path, monkeypatch):
    config = _config()
    coins = tmp_path / "coins.json"
    coins.write_text('{"not": "a coin source"}')
    config["bot"]["global"]["universe"] = {"approved_coins": "file:coins.json"}
    path = _write(tmp_path, "config.json", config)
    read_text = Path.read_text
    def checked_read(target, *args, **kwargs):
        assert target.absolute() != coins
        return read_text(target, *args, **kwargs)
    def forbidden(*_args, **_kwargs):
        pytest.fail("preflight must not prepare or format the runtime config")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "read_text", checked_read)
    monkeypatch.setattr("config.prepare_config", forbidden)
    monkeypatch.setattr("config.bot.format_bot_config", forbidden)
    report = build_live_config_preflight_report(path)
    assert report["hsl"]["policies"]["long"]["enabled"] is True
    assert report["universe"]["approved_coins"]["long"]["sample"] == ["file:coins.json"]


def test_malformed_policy_returns_explicit_issue_without_defaulted_disabled_claim(tmp_path):
    config = _config()
    config["bot"]["global"]["hsl"]["enabled"] = 0
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" for issue in report["issues"])
    assert "hsl" not in report


@pytest.mark.parametrize("enabled", [0, 1, "true", None])
def test_malformed_portfolio_policy_is_not_reported_as_safely_disabled(tmp_path, enabled):
    config = _config()
    config["bot"]["portfolio"]["hsl"] = {"scope": "portfolio", **_hsl(enabled)}
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" and "enabled" in issue["message"]
               for issue in report["issues"])
    assert "hsl" not in report


@pytest.mark.parametrize("bad_policy", [{"scope": "unknown"}, {"scope": "portfolio", "enabled": False}])
def test_unknown_scope_or_incomplete_portfolio_is_an_explicit_issue(tmp_path, bad_policy):
    config = _config()
    config["bot"]["portfolio"]["hsl"] = bad_policy
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" for issue in report["issues"])
    assert "hsl" not in report


def test_malformed_diff_policy_returns_error_instead_of_effective_defaults(tmp_path):
    baseline = _config()
    target = deepcopy(baseline)
    target["bot"]["global"]["hsl"]["enabled"] = 0
    report = build_live_config_diff_report(_write(tmp_path, "before.json", baseline),
                                          _write(tmp_path, "after.json", target))
    assert report["ok"] is False
    assert report["summary"]["change_count"] == 0
    assert [issue["code"] for issue in report["issues"]] == ["target_config_schema_invalid"]


def test_owned_policy_inspection_retains_false_pin_and_source_authority(monkeypatch):
    from config import prepare_config
    from config.bot_inheritance import bot_policy_source_view, get_bot_policy
    from tools.live_config_preflight import _effective_bot_config, _hsl_policy_reports
    raw = _config()
    raw["bot"]["short"] = {"hsl": {"enabled": False}}
    config = prepare_config(raw, live_only=True, verbose=False)
    source = bot_policy_source_view(config)
    policy = get_bot_policy(config)
    def forbidden(*_args, **_kwargs):
        pytest.fail("inspection must not prepare or format a runtime config")
    monkeypatch.setattr("config.prepare_config", forbidden)
    monkeypatch.setattr("config.bot.format_bot_config", forbidden)
    effective = _effective_bot_config(config)
    assert effective["bot"]["long"]["hsl"]["enabled"] is True
    assert effective["bot"]["short"]["hsl"]["enabled"] is False
    assert _hsl_policy_reports(config)["long"]["enabled"] is True
    assert bot_policy_source_view(config) == source
    assert get_bot_policy(config) is policy


@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("disabled", [False, True])
def test_negative_hysteresis_is_invalid_even_for_an_explicitly_disabled_side(tmp_path, scope, disabled):
    config = _config()
    config["bot"].setdefault(scope, {}).setdefault("forager", {})["score_hysteresis_margin"] = -0.1
    config["bot"][scope]["enabled"] = not disabled
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" and "non-negative" in issue["message"]
               for issue in report["issues"])
    assert "forager" not in report


def test_negative_hysteresis_diff_is_an_explicit_policy_error(tmp_path):
    baseline = _config()
    target = deepcopy(baseline)
    target["bot"]["short"] = {"enabled": False, "forager": {"score_hysteresis_margin": -0.1}}
    report = build_live_config_diff_report(_write(tmp_path, "before.json", baseline),
                                          _write(tmp_path, "after.json", target))
    assert report["ok"] is False
    assert [issue["code"] for issue in report["issues"]] == ["target_config_schema_invalid"]


@pytest.mark.parametrize("malformed", [[], None, 1])
def test_malformed_portfolio_hsl_container_returns_schema_issue(tmp_path, malformed):
    config = _config()
    config["bot"]["portfolio"]["hsl"] = malformed
    report = build_live_config_preflight_report(_write(tmp_path, "config.json", config))
    assert report["ok"] is False
    assert any(issue["code"] == "config_schema_invalid" and "hsl" in issue["message"]
               for issue in report["issues"])
    assert "hsl" not in report


def test_inherited_hsl_cli_reports_unsafe_balance_override_without_starting_runtime(tmp_path, capsys):
    from tools.live_config_preflight import main
    path = _write(tmp_path, "config.json", _config())
    assert main([str(path), "--compact"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["hsl"]["policies"]["long"]["enabled"] is True
    assert any(issue["code"] == "hsl_balance_override_account_level_replay_unsafe"
               for issue in report["issues"])
